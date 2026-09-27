"""SafeAgent Core's session-based MCP decisions at agent lifecycle boundaries."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ..tool_results import unwrap_hermes_result


class SafeAgentClient:
    def __init__(self, url: str, timeout: float, api_key_env: str) -> None:
        self.url, self.timeout, self.api_key_env = url, timeout, api_key_env

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError as exc:
            raise RuntimeError("SafeAgent client requires fastmcp; install requirements-safeagent.txt") from exc
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else None
        async with Client(StreamableHttpTransport(self.url, headers=headers), timeout=self.timeout) as client:
            result = await client.call_tool(tool, arguments, timeout=self.timeout)
        data = getattr(result, "data", None) or getattr(result, "structuredContent", None)
        if not isinstance(data, dict):
            raise ValueError(f"SafeAgent {tool} returned no structured mapping")
        return data

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # The project lifecycle is synchronous; Hermes may invoke it from an event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._call(tool, arguments))
        output: list[Any] = []

        def run() -> None:
            try:
                output.append(asyncio.run(self._call(tool, arguments)))
            except Exception as exc:
                output.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join()
        if isinstance(output[0], Exception):
            raise output[0]
        return output[0]




# Actions each hook accepts, as upstream's client (safeagent_coder
# agent/middlewares.py, agent/tool_warpper.py ALLOWED_ACTIONS).
_ALLOWED = {
    "before_agent": {"APPROVE", "OVERRIDE", "REJECT"},
    "after_agent": {"APPROVE", "OVERRIDE", "REJECT"},
    "after_model": {"APPROVE", "REPLAN", "REJECT"},
    "before_model": {"APPROVE", "OVERRIDE", "ROLLBACK", "TERMINATE", "REJECT"},
    "tool_wrapper": {"CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"},
}
_BLOCKING = {"REJECT", "REPLAN", "ROLLBACK", "TERMINATE", "CALL_BLOCK", "CALL_JIT_APPROVAL"}

# Upstream client texts (agent/middlewares.py), quoted verbatim.
_UNAVAILABLE_PLAN = (
    "[SafeAgent Controller] Safety core unavailable. "
    "Tool plan cannot be executed for this turn."
)
_ROLLBACK_GUIDANCE = (
    "[SafeAgent Core] The previous reasoning path or tool output was "
    "determined to be unsafe or unreliable. You must re-evaluate the "
    "task carefully.\n"
    "- Think twice when planning a function calling, they could be unreliable.\n"
    "- Follow the session's safety policy strictly.\n"
    "- Provide a safer, more compliant alternative plan.\n"
)
_TERMINATE_TEXT = "[SafeAgent Core] This conversation turn has been terminated due to safety policy."
_REJECT_TOOL_GUIDANCE = (
    "[SafeAgent Core] The previous tool result was unsafe and has "
    "been removed. The tool used in the last step is now disabled for "
    "the remainder of this session.\n"
    "- You MUST NOT call that tool again.\n"
    "- You MUST answer the user directly using only the available context.\n"
)
_REDACTED_TOOL_OUTPUT = (
    "[SafeAgent Controller] The original tool output has been "
    "redacted due to safety policy. You may assume the tool "
    "completed, but MUST NOT infer or reconstruct any sensitive "
    "details from it."
)


class SafeAgentMiddleware(Middleware):
    """Mirror of upstream's LangChain client over the project lifecycle.

    Upstream hook -> project hook: before_agent -> first guard_model_input;
    after_model -> guard_model_output with tool calls (one step per model
    turn); after_agent -> guard_model_output without tool calls;
    tool_wrapper -> before_tool; before_model (one step per ToolMessage) ->
    after_tool, with ROLLBACK/TERMINATE applied at the next model input.
    """

    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.safeagent
        self.client = client or SafeAgentClient(
            self.settings.mcp_url, self.settings.timeout_seconds, self.settings.api_key_env
        )
        self.developer_cfg = self._read_config(self.settings.developer_config_path)
        self.runtime_cfg = self._read_config(self.settings.runtime_config_path)

    def _read_config(self, name: str) -> dict[str, Any]:
        path = Path(name)
        if not path.is_absolute():
            path = Path(self.cfg.config_dir) / path
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"SafeAgent config must be a mapping: {path}")
        return value

    @property
    def _enforce(self) -> bool:
        return self.settings.mode == "block"

    def _judge(self, state: dict[str, Any], hook: str, observation: dict[str, Any]) -> dict[str, Any]:
        session = state.setdefault("_safeagent_session_id", str(uuid.uuid4()))
        event: dict[str, Any] = {"hook": hook, "action": "ERROR", "override": None,
                                 "violations": [], "error": ""}
        try:
            if not state.get("_safeagent_registered"):
                registration = self.client.call("safeagent_register_session", {
                    "session_id": session, "runtime_cfg": self.runtime_cfg,
                    "dev_cfg": self.developer_cfg,
                })
                if registration.get("ok") is not True:
                    raise RuntimeError(f"session registration failed: {registration.get('error')}")
                state["_safeagent_registered"] = True
            response = self.client.call("safeagent_step", {
                "session_id": session,
                "core_request": {"hook": hook, "observation": observation},
            })
        except Exception as exc:
            event.update(error=f"{type(exc).__name__}: {exc}", failure="unavailable")
        else:
            if not isinstance(response, dict):
                response = {}
            action = str(response.get("action") or "").upper().strip()
            # The core's `error` accumulates across steps (states.py:48,
            # operator.add) and upstream's client never reads it: the decision
            # is `action` alone, and `error` is kept as metadata.
            event.update(
                core_action=response.get("action"),
                override=response.get("override"),
                violations=response.get("violations") or [],
                core_error=response.get("error"),
                allow_long_term_memory=response.get("allow_long_term_memory"),
                notices={key: response[key] for key in ("user_notice", "safety_rationale")
                         if key in response},
            )
            if action in _ALLOWED[hook]:
                event["action"] = action
            else:
                event.update(error=f"unexpected SafeAgent {hook} action: {action!r}",
                             failure="unknown_action", unknown_action=action)
        event["blocked"] = self._enforce and (
            event["action"] in _BLOCKING
            or (event["action"] == "ERROR" and self.settings.fail_closed)
        )
        state["_last_safeagent_decision"] = event
        harness = state.setdefault("harness", {}).setdefault("safeagent", {})
        harness.update({"enabled": True, "mode": self.settings.mode,
                        "status": "error" if event["action"] == "ERROR" else "active",
                        "event_count": harness.get("event_count", 0) + 1,
                        "last_action": event["action"]})
        return event

    def _fail(self, event: dict[str, Any]) -> bool:
        """An error is enforced only in block mode with fail_closed."""
        return event["action"] == "ERROR" and self._enforce and self.settings.fail_closed

    # -- model input: before_agent plus deferred before_model effects --------

    def guard_model_input(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision:
        if not state.get("_safeagent_before_agent_done"):
            state["_safeagent_before_agent_done"] = True
            blocked = self._before_agent(state, messages)
            if blocked is not None:
                return blocked
        pending = state.pop("_safeagent_pending", None)
        if pending and pending.get("kind") == "terminate":
            return ModelDecision(False, pending["reason"], messages=messages,
                                 content=pending["text"], terminate=True,
                                 decision_type="safeagent_terminate")
        if pending and pending.get("kind") == "rollback":
            anchor = _last_user_index(messages)
            if anchor is None:
                text = ("[SafeAgent Controller] ROLLBACK requested but no HumanMessage "
                        "found in history. Request blocked by zero-trust runtime.")
                return ModelDecision(False, "SafeAgent ROLLBACK without a user message",
                                     messages=messages, content=text, terminate=True,
                                     decision_type="safeagent_rollback")
            end = min(len(state.get("messages") or messages), len(messages))
            state.setdefault("_safeagent_cuts", []).append(
                {"start": anchor + 1, "end": end, "guidance": _ROLLBACK_GUIDANCE}
            )
        updated = self._view(state, messages)
        if updated != messages:
            return ModelDecision(messages=updated, decision_type="safeagent_context")
        return ModelDecision(messages=messages)

    def _before_agent(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision | None:
        index = _last_user_index(messages)
        content = messages[index]["content"] if index is not None else str(
            state.get("_runtime_user_request") or self.cfg.agent.task or ""
        )
        if not content:
            return None
        event = self._judge(state, "before_agent", {"role": "user", "content": content})
        action, text = event["action"], None
        if action == "ERROR":
            if not self._fail(event):
                return None
            text = ("[SafeAgent Controller] Safety core unavailable. Request rejected."
                    if event.get("failure") == "unavailable" else
                    f"[SafeAgent Controller] Unknown action '{event.get('unknown_action', '')}'. "
                    "Request blocked by zero-trust runtime.")
        elif not self._enforce or action == "APPROVE":
            return None
        elif action == "REJECT":
            text = event["notices"].get(
                "user_notice", "[SafeAgent Core] The request has been rejected by security policy."
            )
        elif action == "OVERRIDE":
            override = event["override"]
            # The core wraps overrides as {"override": text} (modules/override.py:
            # 254-258), which upstream's client rejects as non-string.
            if not isinstance(override, str) or not override:
                text = "[SafeAgent Controller] Invalid override. Request blocked."
            elif index is not None:
                state["_safeagent_user_override"] = {"index": index, "content": override}
                return None
        if text is None:
            return None
        return ModelDecision(False, f"SafeAgent {action} at before_agent", messages=messages,
                             content=str(text), terminate=True, decision_type="safeagent_reject")

    def _view(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-apply upstream's in-place history edits; Hermes keeps its own copy."""
        view = [dict(message) for message in messages]
        override = state.get("_safeagent_user_override")
        if override and override["index"] < len(view) and view[override["index"]].get("role") == "user":
            view[override["index"]]["content"] = override["content"]
        for cut in reversed(state.get("_safeagent_cuts") or []):
            if cut["end"] <= len(view):
                view[cut["start"]:cut["end"]] = [{"role": "assistant", "content": cut["guidance"]}]
        return view

    # -- model output: after_model / after_agent ------------------------------

    def guard_model_output(self, state: dict[str, Any], content: str, tool_call: Any) -> ModelDecision:
        if tool_call is None:
            return self._after_agent(state, content)
        calls = state.get("_model_output_calls") or [
            {"id": "", "name": tool_call[0], "arguments": tool_call[1]}
        ]
        index = int(state.get("_model_output_index") or 0)
        turn = json.dumps([state.get("iterations"), content, calls], sort_keys=True, default=str)
        cached = state.get("_safeagent_turn") or {}
        if cached.get("key") == turn:
            # One core step per model turn; its decision covers every call.
            event = copy.deepcopy(cached["event"])
            state["_last_safeagent_decision"] = event
        else:
            observation = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"name": call.get("name"), "args": call.get("arguments") or {},
                     "id": call.get("id") or "", "type": "tool_call"}
                    for call in calls
                ],
                "last_user": self._last_user(state),
            }
            event = self._judge(state, "after_model", observation)
            state["_safeagent_turn"] = {"key": turn, "event": copy.deepcopy(event)}
        event.update(tool=tool_call[0], call_index=index)
        action = event["action"]
        if action == "ERROR":
            if not self._fail(event):
                return ModelDecision(content=content, tool_call=tool_call)
            text = (_UNAVAILABLE_PLAN if event.get("failure") == "unavailable" else
                    f"[SafeAgent Controller] Unknown plan action '{event.get('unknown_action', '')}'. "
                    "Plan rejected by zero-trust runtime.")
            return ModelDecision(False, "SafeAgent after_model error", content=text, tool_call=None,
                                 terminate=True, decision_type="safeagent_error")
        if not self._enforce or action == "APPROVE":
            return ModelDecision(content=content, tool_call=tool_call)
        notices = event.get("notices") or {}
        rationale = notices.get("safety_rationale")
        if action == "REPLAN":
            lines = [
                "[SafeAgent Notice] The previous tool plan has been rejected by the safety runtime.",
                "You must now propose a NEW tool plan that strictly follows the current safety policy "
                "and still helps the user achieve their goal.",
                "Do NOT call tools that would violate capability or safety constraints in the current session_policy.",
            ]
        else:
            lines = [
                "[SafeAgent Notice] The proposed tool plan CANNOT be safely executed.",
                "For THIS TURN, you MUST NOT call any tools.",
                "Instead, you MUST answer the user's request directly in natural language, "
                "based only on the existing conversation context.",
            ]
            notice = notices.get("user_notice")
            if isinstance(notice, str) and notice.strip():
                lines.append(
                    f"When answering, you MUST explain that tools were refused because: {notice.strip()}"
                )
        if isinstance(rationale, str) and rationale.strip():
            lines.append(f"Safety rationale (for your internal reasoning only): {rationale.strip()}")
        # Upstream clears the tool calls, appends this guidance to the AI
        # message and jumps back to the model.
        return ModelDecision(content=content, tool_call=None, retry=True, feedback="\n".join(lines),
                             decision_type=f"safeagent_{action.lower()}")

    def _after_agent(self, state: dict[str, Any], content: str) -> ModelDecision:
        if not content.strip():
            return ModelDecision(content=content, tool_call=None)
        event = self._judge(state, "after_agent", {"role": "assistant", "content": content})
        action, text = event["action"], None
        if action == "ERROR":
            if self._fail(event):
                text = (_UNAVAILABLE_PLAN if event.get("failure") == "unavailable" else
                        f"[SafeAgent Controller] Unknown action '{event.get('unknown_action', '')}'. "
                        "Final response replaced by zero-trust runtime.")
        elif self._enforce and action == "REJECT":
            text = event["notices"].get(
                "user_notice", "[SafeAgent Core] The response has been rejected by security policy."
            )
        elif self._enforce and action == "OVERRIDE":
            override = event["override"]
            if isinstance(override, str) and override:
                return ModelDecision(content=override, tool_call=None, decision_type="safeagent_override")
            text = ("[SafeAgent Controller] Invalid override_context. "
                    "Final response replaced by zero-trust runtime.")
        if text is None:
            return ModelDecision(content=content, tool_call=None)
        return ModelDecision(False, f"SafeAgent {action} at after_agent", content=str(text),
                             tool_call=None, terminate=True, decision_type="safeagent_reject")

    # -- tools: tool_wrapper / before_model ----------------------------------

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        description = next((t.description for t in self.cfg.tools if t.name == name), "")
        event = self._judge(state, "tool_wrapper", {
            "plan": str(state.get("_model_output_content") or ""),
            "name": name, "args": payload, "description": description,
        })
        action, text = event["action"], None
        if action == "ERROR":
            if self._fail(event):
                text = (f"[SafeAgent Controller] Safety core unavailable. "
                        f"Tool call '{name}' has been blocked. ({event['error']})"
                        if event.get("failure") == "unavailable" else
                        f"[SafeAgent Controller] Unknown action '{event.get('unknown_action', '')}'. "
                        f"Tool call '{name}' blocked by zero-trust runtime.")
        elif not self._enforce or action == "CALL_ALLOW":
            return ToolDecision()
        elif action == "CALL_BLOCK":
            text = event["notices"].get(
                "safety_rationale", f"[SafeAgent Core] Tool '{name}' call blocked by safety policy."
            )
        elif action == "CALL_REWRITE":
            # Upstream passes the core's {"override": ...} wrapper as the new args.
            if isinstance(event["override"], dict):
                return ToolDecision(arguments=event["override"], decision_type="safeagent_rewrite")
            text = ("[SafeAgent Controller] Invalid override payload in CALL_REWRITE. "
                    f"Tool call '{name}' blocked.")
        elif action == "CALL_JIT_APPROVAL":
            # No human approver: the call is not executed (upstream's pending state).
            text = event["notices"].get(
                "user_notice", f"[SafeAgent Core] Tool '{name}' requires human confirmation."
            )
        if text is None:
            return ToolDecision()
        return ToolDecision(False, f"SafeAgent {action} blocked tool {name}",
                            replacement_result=str(text), decision_type="safeagent_block")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        if state.get("_safeagent_pending"):
            # Upstream stops reviewing once ROLLBACK/TERMINATE removed the turn.
            return ResultDecision(result=result)
        # Blocked calls' messages are reviewed too, as upstream reviews every
        # ToolMessage after the last AI message.
        event = self._judge(state, "before_model", {
            "role": "tool", "name": name,
            "tool_call_id": _call_id(state, name, payload),
            "content": unwrap_hermes_result(str(result)),
        })
        action = event["action"]
        withheld = "Tool output withheld by SafeAgent Core."
        if action == "ERROR":
            if not self._fail(event):
                return ResultDecision(result=result)
            text = (_UNAVAILABLE_PLAN if event.get("failure") == "unavailable" else
                    f"[SafeAgent Controller] Unknown before_model action '{event.get('unknown_action', '')}'. "
                    "Request blocked by zero-trust runtime.")
            state["_safeagent_pending"] = {"kind": "terminate", "text": text,
                                           "reason": "SafeAgent before_model error"}
            return ResultDecision(False, "SafeAgent before_model error", result=withheld,
                                  decision_type="safeagent_error")
        if not self._enforce or action == "APPROVE":
            return ResultDecision(result=result)
        if action == "OVERRIDE":
            override = event["override"]
            text = override if isinstance(override, str) and override else _REDACTED_TOOL_OUTPUT
            return ResultDecision(result=text, decision_type="safeagent_override")
        if action == "REJECT":
            return ResultDecision(False, "SafeAgent REJECT replaced tool output",
                                  result=_REJECT_TOOL_GUIDANCE, decision_type="safeagent_reject")
        if action == "ROLLBACK":
            state["_safeagent_pending"] = {"kind": "rollback"}
        else:  # TERMINATE
            state["_safeagent_pending"] = {"kind": "terminate", "text": _TERMINATE_TEXT,
                                           "reason": "SafeAgent TERMINATE"}
        return ResultDecision(False, f"SafeAgent {action} rejected tool output",
                              result=withheld, decision_type=f"safeagent_{action.lower()}")

    def _last_user(self, state: dict[str, Any]) -> str:
        """Upstream sends the latest HumanMessage's content (middlewares.py:483-489)."""
        messages = state.get("messages") or []
        index = _last_user_index(messages)
        if index is not None:
            return str(messages[index]["content"])
        return str(state.get("_runtime_user_request") or self.cfg.agent.task or "(no user request available)")


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and isinstance(message.get("content"), str) and message["content"].strip():
            return index
    return None


def _call_id(state: dict[str, Any], name: str, payload: dict[str, Any]) -> str:
    for call in state.get("_model_output_calls") or []:
        if call.get("name") == name and call.get("arguments") == payload and call.get("id"):
            return str(call["id"])
    return str(uuid.uuid4())
