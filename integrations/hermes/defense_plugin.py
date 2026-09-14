"""Register project guards using upstream Hermes execution middleware APIs."""

from __future__ import annotations

import copy
import json
import os
import threading
from urllib.request import Request, urlopen


class GuardStopped(BaseException):
    # Hermes intentionally skips failed plugins on Exception. A policy stop must
    # cross that boundary without allowing an unguarded downstream call.
    def __init__(self, reason, output="", failed=False):
        self.reason, self.output, self.failed = reason, output, failed
        super().__init__(reason)


class ProviderCallFailed(BaseException):
    def __init__(self, original):
        self.original = original
        super().__init__(str(original))


class DefensePlugin:
    def __init__(self, agent, request, mapping, event):
        self.agent, self.request, self.mapping, self.event = (
            agent,
            request,
            mapping,
            event,
        )
        self.reverse = {v: k for k, v in mapping.items()}
        self.lock = threading.RLock()
        self.messages = []
        self.handles = []

    def rpc(self, op, **payload):
        try:
            req = Request(
                self.request["guard_url"] + "/guard",
                data=json.dumps({"op": op, **payload}).encode(),
                headers={
                    "Authorization": "Bearer " + os.environ["BENCHMARK_GUARD_TOKEN"],
                    "Content-Type": "application/json",
                },
            )
            with urlopen(req, timeout=self.request["timeout_seconds"]) as response:
                return json.load(response)
        except Exception as exc:
            raise GuardStopped(
                f"Defense bridge failed: {type(exc).__name__}: {exc}", failed=True
            ) from exc

    def install(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

        ctx = PluginContext(
            PluginManifest(name="project-defenses"), get_plugin_manager()
        )
        tools = [
            {
                "name": self.mapping.get(t["function"]["name"], t["function"]["name"]),
                "description": t["function"].get("description", ""),
            }
            for t in self.agent.tools
        ]
        self.rpc("initialize", tools=tools, task=self.request["prompt"])
        self.handles = [
            ctx.register_middleware("llm_execution", self.model),
            ctx.register_middleware("tool_execution", self.tool),
        ]

    def _messages(self, messages):
        result = json.loads(json.dumps(messages, default=str))
        for message in result:
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                fn["name"] = self.mapping.get(fn.get("name"), fn.get("name"))
            if message.get("name"):
                message["name"] = self.mapping.get(message["name"], message["name"])
        return result

    def model(self, request, next_call, **context):
        try:
            from responses_wire import (
                guard_messages,
                response_view,
                rewrite_request,
                rewrite_response,
            )

            responses = (
                context.get("api_mode", getattr(self.agent, "api_mode", ""))
                == "codex_responses"
            )
            self.messages = (
                guard_messages(request) if responses else request.get("messages", [])
            )
            canonical = self._messages(self.messages)
            before = self.rpc("model_input", messages=canonical)
            if not before["allowed"] or before.get("terminate"):
                raise GuardStopped(
                    before.get("reason", "Input blocked"),
                    before.get("content") or "Input blocked by defense.",
                )
            messages = before.get("messages", canonical)
            for message in messages:
                for call in message.get("tool_calls") or []:
                    fn = call["function"]
                    fn["name"] = self.reverse.get(fn["name"], fn["name"])
                if message.get("name"):
                    message["name"] = self.reverse.get(message["name"], message["name"])
            outgoing = (
                rewrite_request(request, self.messages, messages)
                if responses
                else {**request, "messages": messages}
            )
            try:
                response = next_call(outgoing)
            except Exception as exc:
                raise ProviderCallFailed(exc) from exc
            choice = (
                response_view(self.agent, response)
                if responses
                else response.choices[0]
            )
            original_message = copy.deepcopy(choice.message)
            message = choice.message
            calls = [
                {
                    "id": c.id,
                    "name": self.mapping.get(c.function.name, c.function.name),
                    "arguments": json.loads(c.function.arguments),
                }
                for c in (message.tool_calls or [])
            ]
            after = self.rpc(
                "model_output", content=message.content or "", tool_calls=calls
            )
            if after.get("retry"):
                feedback = (
                    "Safety middleware requested a final-response revision. Do not call tools unless "
                    "the feedback explicitly concerns a proposed tool action.\n"
                    + after.get("feedback", "")
                )
                if not self.agent.redirect(feedback):
                    raise GuardStopped(
                        "Hermes could not schedule defense revision", failed=True
                    )
                return response
            message.content = after["content"]
            from openai.types.chat import ChatCompletionMessageToolCall

            message.tool_calls = [
                ChatCompletionMessageToolCall(
                    id=c["id"],
                    type="function",
                    function={
                        "name": self.reverse.get(c["name"], c["name"]),
                        "arguments": json.dumps(c["arguments"]),
                    },
                )
                for c in after["tool_calls"]
            ] or None
            if not message.tool_calls:
                choice.finish_reason = "stop"
            if not after["allowed"] or after.get("terminate"):
                message.tool_calls = None
                message.content = (
                    after["content"]
                    or after.get("reason")
                    or "Output blocked by defense."
                )
                choice.finish_reason = "stop"
            return (
                rewrite_response(response, original_message, message)
                if responses
                else response
            )
        except GuardStopped:
            raise
        except ProviderCallFailed as exc:
            raise exc.original
        except Exception as exc:
            raise GuardStopped(
                f"Defense model adapter failed: {exc}", failed=True
            ) from exc

    def tool(self, tool_name, args, next_call, **context):
        # Serialize checks + execution + updates for stateful project policies,
        # including Hermes' concurrently dispatched native tools.
        with self.lock:
            try:
                canonical = self.mapping.get(tool_name, tool_name)
                if canonical == "skill_view":
                    self.event("skill_view_attempt", {"arguments": args})
                before = self.rpc("before_tool", name=canonical, arguments=args)
                if before.get("terminate"):
                    raise GuardStopped(
                        before.get("reason") or "Tool blocked",
                        before.get("replacement_result") or "Tool blocked by defense.",
                    )
                if not before["allowed"]:
                    value = before.get("replacement_result") or json.dumps(
                        {
                            "error": "Tool blocked by defense",
                            "reason": before.get("reason"),
                        }
                    )
                    self.event(
                        "defense_tool_blocked",
                        {"tool": canonical, "reason": before.get("reason")},
                    )
                    after = self.rpc(
                        "after_tool",
                        name=canonical,
                        arguments=args,
                        result=value,
                        failed=True,
                    )
                    return after.get("result", value)
                effective = before.get("tool_name") or canonical
                arguments = (
                    before.get("arguments")
                    if before.get("arguments") is not None
                    else args
                )
                if effective != canonical:
                    from model_tools import handle_function_call

                    rewritten = self.reverse.get(effective, effective)
                    allowed = {t["function"]["name"] for t in self.agent.tools}
                    if rewritten not in allowed:
                        raise GuardStopped(
                            "Defense rewrote tool outside the granted inventory",
                            failed=True,
                        )
                    value = handle_function_call(
                        rewritten,
                        arguments,
                        task_id=context.get("task_id"),
                        session_id=context.get("session_id"),
                        tool_call_id=context.get("tool_call_id"),
                        enabled_tools=list(allowed),
                        skip_tool_execution_middleware=True,
                    )
                else:
                    value = next_call(arguments)
                text = (
                    value if isinstance(value, str) else json.dumps(value, default=str)
                )
                failed = False
                try:
                    parsed = json.loads(text)
                    failed = isinstance(parsed, dict) and bool(
                        parsed.get("error") or parsed.get("isError")
                    )
                except ValueError:
                    pass
                after = self.rpc(
                    "after_tool",
                    name=effective,
                    arguments=arguments,
                    result=text,
                    failed=failed,
                )
                if effective == "skill_view":
                    self.event(
                        "skill_view",
                        {
                            "arguments": arguments,
                            "failed": failed,
                            "result_allowed": after["allowed"],
                            "result_changed": after.get("result", text) != text,
                        },
                    )
                return after.get("result", text)
            except GuardStopped:
                raise
            except Exception as exc:
                raise GuardStopped(
                    f"Defense tool adapter failed: {exc}", failed=True
                ) from exc


class ReplayRecorderPlugin:
    """Record an immutable guard lifecycle while leaving Hermes untouched."""

    def __init__(self, agent, request, mapping, event, path):
        self.agent, self.request, self.mapping, self.event = (
            agent,
            request,
            mapping,
            event,
        )
        self.path = path
        self.lock = threading.Lock()
        self.sequence = 0
        self.messages = []
        self.handles = []

    def _append(self, op, **payload):
        with self.lock:
            entry = {"sequence": self.sequence, "op": op, **payload}
            self.sequence += 1
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _emit(self, kind, payload):
        try:
            self.event(kind, payload)
        except Exception:
            pass

    def _record_error(self, op, exc):
        self._emit(
            "defense_replay_record_error",
            {"op": op, "error": f"{type(exc).__name__}: {exc}"},
        )

    def _record(self, op, **payload):
        try:
            self._append(op, **payload)
        except Exception as exc:
            self._record_error(op, exc)

    def _messages(self, messages):
        result = json.loads(json.dumps(messages, default=str))
        for message in result:
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                fn["name"] = self.mapping.get(fn.get("name"), fn.get("name"))
            if message.get("name"):
                message["name"] = self.mapping.get(message["name"], message["name"])
        return result

    def install(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

        ctx = PluginContext(
            PluginManifest(name="project-defense-replay-recorder"),
            get_plugin_manager(),
        )
        tools = [
            {
                "name": self.mapping.get(t["function"]["name"], t["function"]["name"]),
                "description": t["function"].get("description", ""),
            }
            for t in self.agent.tools
        ]
        self._record("initialize", tools=tools, task=self.request["prompt"])
        self.handles = [
            ctx.register_middleware("llm_execution", self.model),
            ctx.register_middleware("tool_execution", self.tool),
        ]

    def model(self, request, next_call, **context):
        responses = (
            context.get("api_mode", getattr(self.agent, "api_mode", ""))
            == "codex_responses"
        )
        try:
            if responses:
                from responses_wire import guard_messages

                self.messages = guard_messages(request)
            else:
                self.messages = request.get("messages", [])
            self._record("model_input", messages=self._messages(self.messages))
        except Exception as exc:
            self._record_error("model_input", exc)
        response = next_call(request)
        try:
            if responses:
                from responses_wire import response_view

                choice = response_view(self.agent, response)
            else:
                choice = response.choices[0]
            message = choice.message
            calls = []
            for call in message.tool_calls or []:
                raw_arguments = call.function.arguments
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, ValueError):
                    arguments = raw_arguments
                calls.append(
                    {
                        "id": call.id,
                        "name": self.mapping.get(
                            call.function.name, call.function.name
                        ),
                        "arguments": arguments,
                    }
                )
            self._record(
                "model_output", content=message.content or "", tool_calls=calls
            )
        except Exception as exc:
            self._record_error("model_output", exc)
        return response

    def tool(self, tool_name, args, next_call, **context):
        canonical = self.mapping.get(tool_name, tool_name)
        if canonical == "skill_view":
            self._emit("skill_view_attempt", {"arguments": args})
        self._record("before_tool", name=canonical, arguments=args)
        try:
            value = next_call(args)
        except BaseException as exc:
            self._record(
                "after_tool",
                name=canonical,
                arguments=args,
                result=str(exc),
                failed=True,
            )
            raise
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        failed = False
        try:
            parsed = json.loads(text)
            failed = isinstance(parsed, dict) and bool(
                parsed.get("error") or parsed.get("isError")
            )
        except ValueError:
            pass
        self._record(
            "after_tool",
            name=canonical,
            arguments=args,
            result=text,
            failed=failed,
        )
        if canonical == "skill_view":
            self._emit(
                "skill_view",
                {
                    "arguments": args,
                    "failed": failed,
                    "result_allowed": True,
                    "result_changed": False,
                },
            )
        return value
