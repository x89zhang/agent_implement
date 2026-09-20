from __future__ import annotations

import ast
import functools
import inspect
import json
import re
from pathlib import Path
from typing import Any
import time

from langgraph.graph import END, StateGraph

from .config import AppConfig
from .llm import LLMAdapter
from .skills import load_enabled_skills, render_skill_context, validate_skill_tools
from .planner import render_plan_context, mark_plan_progress, complete_plan_on_final
from .middleware import (
    ToolExecutionTerminated,
    build_middleware_manager,
    output_revision_limit,
)
from .agentdog.trajectory import (
    build_revision_messages,
    normalize_react_intermediate_steps,
)
from .nodes import (
    agent_node,
    load_tool,
    tool_node,
    _append_trace_message,
    _flush_trace_snapshot,
    _update_usage_totals,
    render_tool_output_security_prompt,
)


class _SerialNativeToolModel:
    """Supply request-level serial tool binding to LangChain's agent factory."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.model.bind_tools(
            tools, **{**kwargs, "parallel_tool_calls": False}
        )


def _validate_native_tool_turn(parsed: Any, record: Any = None) -> Any:
    """Check native turns; explanatory text never executes tools."""
    from langchain_core.agents import AgentFinish
    from langchain_core.exceptions import OutputParserException

    def fail(reason: str) -> Any:
        if record is not None:
            actions = parsed if isinstance(parsed, list) else []
            record({
                "event": "validation_failed", "reason": reason,
                "parsed_type": type(parsed).__name__,
                "content": [
                    getattr(message, "content", "")
                    for action in actions
                    for message in getattr(action, "message_log", [])
                ] or getattr(parsed, "log", ""),
                "tool_calls": [
                    {"name": getattr(action, "tool", ""),
                     "arguments": getattr(action, "tool_input", None)}
                    for action in actions
                ],
            })
        # Keep rejected text out of the shared text-action recovery path.
        raise OutputParserException(
            reason, observation=reason, llm_output="", send_to_llm=True
        )

    if isinstance(parsed, AgentFinish):
        return fail("End with a native finish call, not a plain-text answer.")
    if not isinstance(parsed, list) or len(parsed) != 1:
        return fail("Issue exactly one native tool call per turn, including finish.")
    action = parsed[0]
    messages = getattr(action, "message_log", [])
    content = getattr(messages[0], "content", "") if messages else ""
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    if not isinstance(content, str):
        content = ""
    pattern = r"\s*Thought:[ \t]*([^\n]+)\n[ \t]*Action:[ \t]*([^\n]+)\s*"
    match = re.fullmatch(pattern, content)
    if not match or not match[1].strip():
        return fail(
            "Include Thought: brief action rationale (not private reasoning), then "
            "Action: exact_tool_name on the next line, matching the native call."
        )
    # Normalize only the explanatory text; never rewrite the native call.
    declared_tool = match[2].strip().removeprefix("functions.")
    if declared_tool != action.tool:
        return fail(f"Action must match the native tool name: {action.tool}")
    if action.tool == "finish":
        args = action.tool_input
        answer = args.get("answer") if isinstance(args, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            return fail("The finish call requires a non-empty answer string.")
        return AgentFinish(return_values={"output": answer.strip()}, log=action.log)
    return parsed


def _build_previous_response_id_text_model(model: Any) -> tuple[Any, Any]:
    """Add Responses API continuation without changing the text-ReAct loop.

    create_react_agent rebuilds the complete prompt on every iteration and
    discards the AIMessage metadata that LangChain normally uses to infer
    previous_response_id. This wrapper retains that ID and submits only the
    newly appended ReAct scratchpad suffix. If the prompt is not an exact
    continuation, it safely starts a new response chain with the full prompt.
    """
    try:
        from langchain_core.runnables import RunnableLambda  # type: ignore
        from langchain_core.runnables.config import RunnableConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime import
        raise RuntimeError("Missing dependency: langchain_core") from exc

    chain: dict[str, str | None] = {
        "response_id": None,
        "prompt": None,
        "output": None,
    }

    def reset() -> None:
        chain.update(response_id=None, prompt=None, output=None)

    def invoke(input_value: Any, config: RunnableConfig) -> Any:
        if hasattr(input_value, "to_string"):
            prompt = str(input_value.to_string())
        else:
            prompt = str(input_value)

        request_input = prompt
        invoke_options: dict[str, Any] = {}
        response_id = chain["response_id"]
        prior_prompt = chain["prompt"]
        prior_output = chain["output"]
        if response_id and prior_prompt is not None and prior_output is not None:
            expected_prefix = prior_prompt + prior_output
            if prompt.startswith(expected_prefix):
                request_input = prompt[len(expected_prefix) :]
                invoke_options["previous_response_id"] = response_id
            else:
                # Never combine a previous response with a duplicated or
                # unrelated full prompt.
                reset()

        result = model.invoke(request_input, config=config, **invoke_options)
        metadata = getattr(result, "response_metadata", None)
        next_id = metadata.get("id") if isinstance(metadata, dict) else None
        output = getattr(result, "content", None)
        if (
            isinstance(next_id, str)
            and next_id.startswith("resp_")
            and isinstance(output, str)
        ):
            chain.update(response_id=next_id, prompt=prompt, output=output)
        else:
            reset()
        return result

    return RunnableLambda(invoke, name="responses_previous_response_id"), reset


def _build_react_user_input(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    user_parts = [
        str(message.get("content", "")).strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if user_parts:
        return "\n\n".join(user_parts)
    if messages and isinstance(messages[-1], dict):
        return str(messages[-1].get("content", ""))
    return ""



def _extract_react_actions(log_text: str) -> list[dict[str, Any]]:
    """Recover legacy Action blocks from an otherwise invalid model response."""
    actions: list[dict[str, Any]] = []
    if not log_text:
        return actions

    text = str(log_text)
    pos = 0
    while True:
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text[pos:])
        if not action_match:
            break
        action_name = action_match.group(1)
        action_abs_start = pos + action_match.start()
        action_abs_end = pos + action_match.end()
        input_match = re.search(r"Action Input:\s*", text[action_abs_end:])
        if not input_match:
            break
        input_start = action_abs_end + input_match.end()
        while input_start < len(text) and text[input_start].isspace():
            input_start += 1

        payload: Any = ""
        next_pos = input_start
        if input_start < len(text) and text[input_start] in "{[":
            opener = text[input_start]
            closer = "}" if opener == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            end = None
            for idx in range(input_start, len(text)):
                ch = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "\"":
                        in_string = False
                    continue
                if ch == "\"":
                    in_string = True
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end is None:
                break
            payload = _maybe_parse_json(text[input_start:end])
            next_pos = end
        else:
            line_end = text.find("\n", input_start)
            if line_end == -1:
                line_end = len(text)
            payload = text[input_start:line_end].strip()
            next_pos = line_end

        thought_text = ""
        for line in reversed(text[:action_abs_start].splitlines()):
            if line.strip().lower().startswith("thought"):
                thought_text = line.strip()
                break

        actions.append(
            {
                "tool": action_name,
                "tool_input": payload,
                "log": text[action_abs_start:next_pos].strip(),
                "thought": thought_text or None,
            }
        )
        pos = next_pos

    return actions


def _expand_react_steps(
    intermediate_steps: list[Any],
    estimate_tokens: Any,
    print_trace: bool,
    raw_tools: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", "")
        log_text = getattr(action, "log", "")

        if tool_name == "_Exception":
            if raw_tools:
                for recovered in _extract_react_actions(str(log_text)):
                    recovered_name = str(recovered.get("tool", ""))
                    recovered_input = recovered.get("tool_input", {})
                    if recovered_name not in raw_tools:
                        recovered_output = f"Tool not found: {recovered_name}"
                    else:
                        try:
                            if isinstance(recovered_input, dict):
                                recovered_output = str(
                                    raw_tools[recovered_name](**recovered_input)
                                )
                            else:
                                recovered_output = str(
                                    raw_tools[recovered_name](recovered_input)
                                )
                        except Exception as exc:
                            recovered_output = f"Tool execution failed: {exc}"
                    usage = {
                        "input_tokens": estimate_tokens(str(recovered_input)),
                        "output_tokens": estimate_tokens(str(recovered_output)),
                        "total_tokens": estimate_tokens(str(recovered_input))
                        + estimate_tokens(str(recovered_output)),
                        "source": "estimated",
                    }
                    expanded.append(
                        {
                            "tool": recovered_name,
                            "tool_input": recovered_input,
                            "log": str(recovered.get("log", "")),
                            "observation": recovered_output,
                            "usage": usage,
                            "thought": recovered.get("thought"),
                        }
                    )
                    if print_trace:
                        thought_text = str(
                            recovered.get("thought") or ""
                        ).strip()
                        if thought_text:
                            print("\n[THOUGHT]")
                            print(thought_text)
                        print("\n[TOOL INPUT]")
                        print(f"{recovered_name} {recovered_input}")
                        print("[TOOL OUTPUT]")
                        print(recovered_output)
            continue

        thought_text = ""
        for line in str(log_text).splitlines():
            if line.strip().lower().startswith("thought"):
                thought_text = line.strip()
                break
        usage = {
            "input_tokens": estimate_tokens(str(tool_input)),
            "output_tokens": estimate_tokens(str(observation)),
            "total_tokens": estimate_tokens(str(tool_input)) + estimate_tokens(str(observation)),
            "source": "estimated",
        }
        expanded.append(
            {
                "tool": tool_name,
                "tool_input": tool_input,
                "log": log_text,
                "observation": observation,
                "usage": usage,
                "thought": thought_text or None,
            }
        )
    return expanded



def _literal_ast_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node) if hasattr(ast, "unparse") else ""


def _parse_function_style_action(action_text: str, tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    text = str(action_text).strip()
    if not text:
        return None
    call_match = re.search(r"([a-zA-Z_]\w*)\s*\(", text)
    if not call_match:
        return None
    tool_name = call_match.group(1)
    if tool_name not in tool_names:
        return None

    start = call_match.start()
    depth = 0
    in_string = False
    quote = ""
    escape = False
    end = None
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None

    call_text = text[start:end]
    try:
        expr = ast.parse(call_text, mode="eval")
    except SyntaxError:
        return None
    call = expr.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.func.id != tool_name:
        return None

    payload: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg:
            payload[keyword.arg] = _literal_ast_value(keyword.value)
    if call.args:
        payload["__arg"] = _literal_ast_value(call.args[0])
    return tool_name, payload



def _extract_first_json_value(value: str) -> Any | None:
    text = str(value).strip()
    if not text or text[0] not in "{[":
        return None
    opener = text[0]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[: idx + 1])
                except Exception:
                    return None
    return None


def _normalize_react_tool_name(tool: Any, tool_names: set[str]) -> str:
    text = str(tool).strip().strip("` ,.:;\n\t")
    if text in tool_names:
        return text
    for name in sorted(tool_names, key=len, reverse=True):
        if text == name or text.startswith(name + " ") or text.startswith(name + "`") or text.startswith(name + ","):
            return name
    return text


def _normalize_react_tool_input(tool_input: Any) -> Any:
    if isinstance(tool_input, str):
        parsed = _extract_first_json_value(tool_input)
        if parsed is not None:
            return parsed
    return tool_input

def _build_react_output_parser(tool_names: set[str]) -> Any | None:
    try:
        from langchain_core.agents import AgentAction, AgentFinish  # type: ignore
        from langchain_core.exceptions import OutputParserException  # type: ignore
    except Exception:
        try:
            from langchain.schema import AgentAction, AgentFinish  # type: ignore
            from langchain.schema import OutputParserException  # type: ignore
        except Exception:
            return None

    base_parser_cls = None
    for module_name in (
        "langchain.agents.output_parsers.react_single_input",
        "langchain_classic.agents.output_parsers.react_single_input",
    ):
        try:
            module = __import__(module_name, fromlist=["ReActSingleInputOutputParser"])
            base_parser_cls = getattr(module, "ReActSingleInputOutputParser")
            break
        except Exception:
            continue
    if base_parser_cls is None:
        return None

    class _ActionOnlyReActOutputParser(base_parser_cls):  # type: ignore[misc, valid-type]
        def parse(self, text: str) -> Any:
            def fail(reason: str) -> Any:
                raise OutputParserException(
                    reason,
                    observation=reason,
                    llm_output=text,
                    send_to_llm=True,
                )

            if re.search(r"(?im)^\s*Final Answer\s*:", text):
                return fail(
                    "Final Answer is not valid in the action-only protocol. "
                    "Use Action: finish with Action Input JSON containing answer."
                )
            thoughts = re.findall(r"(?im)^\s*Thought\s*:", text)
            actions = re.findall(r"(?im)^\s*Action\s*:", text)
            action_inputs = re.findall(r"(?im)^\s*Action Input\s*:", text)
            if len(thoughts) != 1 or len(actions) != 1 or len(action_inputs) != 1:
                return fail(
                    "Expected exactly one Thought, one Action, and one Action Input; "
                    f"found {len(thoughts)}, {len(actions)}, and {len(action_inputs)}."
                )
            try:
                parsed = super().parse(text)
            except Exception as exc:
                return fail(f"Could not parse the ReAct action: {exc}")
            tool = getattr(parsed, "tool", None)
            tool_input = _normalize_react_tool_input(
                getattr(parsed, "tool_input", "")
            )
            if not isinstance(tool, str):
                return fail("The response did not contain a valid action name.")
            normalized_name = _normalize_react_tool_name(
                tool, tool_names | {"finish"}
            )
            if not isinstance(tool_input, dict):
                return fail("Action Input must be one valid JSON object.")
            if normalized_name == "finish":
                answer = tool_input.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    return fail(
                        "The finish action requires a non-empty string field named answer."
                    )
                return AgentFinish(
                    return_values={"output": answer.strip()}, log=text
                )
            if normalized_name not in tool_names:
                return fail(f"Unknown action {normalized_name!r}.")
            return AgentAction(
                tool=normalized_name,
                tool_input=tool_input,
                log=getattr(parsed, "log", text),
            )

    return _ActionOnlyReActOutputParser()



def _build_legacy_react_output_parser(tool_names: set[str]) -> Any | None:
    try:
        from langchain_core.agents import AgentAction  # type: ignore
    except Exception:
        try:
            from langchain.schema import AgentAction  # type: ignore
        except Exception:
            return None

    base_parser_cls = None
    for module_name in (
        "langchain.agents.output_parsers.react_single_input",
        "langchain_classic.agents.output_parsers.react_single_input",
    ):
        try:
            module = __import__(
                module_name, fromlist=["ReActSingleInputOutputParser"]
            )
            base_parser_cls = getattr(module, "ReActSingleInputOutputParser")
            break
        except Exception:
            continue
    if base_parser_cls is None:
        return None

    class _CompatReActOutputParser(base_parser_cls):  # type: ignore[misc, valid-type]
        def parse(self, text: str) -> Any:
            parsed = super().parse(text)
            tool = getattr(parsed, "tool", None)
            tool_input = _normalize_react_tool_input(
                getattr(parsed, "tool_input", "")
            )
            if isinstance(tool, str):
                normalized_name = _normalize_react_tool_name(tool, tool_names)
                if normalized_name in tool_names:
                    return AgentAction(
                        tool=normalized_name,
                        tool_input=tool_input,
                        log=getattr(parsed, "log", text),
                    )
                normalized = _parse_function_style_action(tool, tool_names)
                if normalized is not None:
                    tool_name, payload = normalized
                    if "__arg" in payload and len(payload) == 1:
                        tool_input = payload["__arg"]
                    else:
                        payload.pop("__arg", None)
                        tool_input = payload
                    return AgentAction(
                        tool=tool_name,
                        tool_input=tool_input,
                        log=getattr(parsed, "log", text),
                    )
            return parsed

    return _CompatReActOutputParser()


def _build_react_callbacks(enabled: bool) -> list[Any]:
    if not enabled:
        return []

    BaseCallbackHandler = None
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore
    except Exception:
        try:
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
        except Exception:
            return []

    class _RealtimeReactCallback(BaseCallbackHandler):  # type: ignore[misc]
        def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
            tool_name = str(getattr(action, "tool", "") or "")
            if tool_name == "_Exception":
                return
            log_text = str(getattr(action, "log", "") or "")
            thought_text = ""
            for line in log_text.splitlines():
                if line.strip().lower().startswith("thought"):
                    thought_text = line.strip()
                    break
            if thought_text:
                print("\n[THOUGHT]", flush=True)
                print(thought_text, flush=True)

        def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
            return None

        def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
            values = getattr(finish, "return_values", {}) or {}
            output = values.get("output", "")
            if output:
                print("\n[LLM OUTPUT]", flush=True)
                print(output, flush=True)

    return [_RealtimeReactCallback()]


def _react_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs:
        return dict(kwargs)
    if len(args) == 1 and isinstance(args[0], dict):
        return dict(args[0])
    if len(args) == 1:
        return {"__arg": args[0]}
    return {"args": list(args)}


def _render_react_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    return list(args)


def _record_react_runtime_step(state: dict[str, Any], result: str) -> None:
    events = state.get("_react_guard_events", [])
    if not events:
        return
    event = events[-1]
    state.setdefault("_react_runtime_steps", []).append(
        {
            "tool": event.get("tool", ""),
            "tool_input": event.get("tool_input", ""),
            "observation": result,
            "log": "",
            "thought": None,
            "usage": {},
            "aegis": event.get("aegis"),
            "progent": event.get("progent"),
            "pro2guard": event.get("pro2guard"),
            "agentspec": event.get("agentspec"),
            "toolsafe": event.get("toolsafe"),
            "agentdog": event.get("agentdog"),
            "agentguard": event.get("agentguard"),
            "blocked": bool(event.get("blocked")),
        }
    )


def _build_traced_react_tool(name: str, fn: Any, cfg: AppConfig, middleware: Any, get_state: Any, tool_lookup: dict[str, Any]) -> Any:
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        rendered_input = _render_react_payload(args, kwargs)
        payload = _react_payload(args, kwargs)
        state = get_state()
        if isinstance(state, dict):
            state.pop("_last_aegis_decision", None)
            state.pop("_last_progent_decision", None)
            state.pop("_last_pro2guard_decision", None)
            state.pop("_last_agentspec_decision", None)
            state.pop("_last_toolsafe_decision", None)
            state.pop("_last_agentdog_decision", None)
            state.pop("_last_agentguard_decision", None)
        decision = middleware.before_tool(state if isinstance(state, dict) else {}, name, payload)
        aegis_decision = None
        progent_decision = None
        pro2guard_decision = None
        agentspec_decision = None
        toolsafe_decision = None
        agentdog_decision = None
        agentguard_decision = None
        if isinstance(state, dict):
            aegis_decision = state.pop("_last_aegis_decision", None)
            progent_decision = state.pop("_last_progent_decision", None)
            pro2guard_decision = state.pop("_last_pro2guard_decision", None)
            agentspec_decision = state.pop("_last_agentspec_decision", None)
            toolsafe_decision = state.pop("_last_toolsafe_decision", None)
            agentdog_decision = state.pop("_last_agentdog_decision", None)
            agentguard_decision = state.get("_last_agentguard_decision")
        effective_name = decision.tool_name or name
        effective_payload = decision.arguments if decision.arguments is not None else payload

        if cfg.monitoring.print_trace:
            print("\n[TOOL INPUT]", flush=True)
            print(f"{name} {rendered_input}", flush=True)

        if not decision.allowed:
            result = decision.replacement_result or f"Tool execution blocked by middleware: {decision.reason}"
            result_decision = middleware.after_tool(state if isinstance(state, dict) else {}, effective_name, effective_payload, result, True)
            result = str(result_decision.result)
            if isinstance(state, dict):
                state.setdefault("_react_guard_events", []).append({
                    "tool": name,
                    "tool_input": rendered_input,
                    "aegis": aegis_decision,
                    "progent": {"before": progent_decision, "after": state.pop("_last_progent_decision", None)},
                    "pro2guard": pro2guard_decision,
                    "agentspec": agentspec_decision,
                    "toolsafe": toolsafe_decision,
                    "agentdog": agentdog_decision,
                    "agentguard": {"before": agentguard_decision, "after": (state.get("_last_agentguard_decision") if isinstance(state, dict) else None)},
                    "blocked": True,
                })
                _record_react_runtime_step(state, result)
            if cfg.monitoring.print_trace:
                print("[TOOL OUTPUT]", flush=True)
                print(result, flush=True)
            if decision.terminate:
                raise ToolExecutionTerminated(
                    result, effective_name, effective_payload
                )
            return result

        try:
            target = tool_lookup.get(effective_name)
            if target is None:
                raise ValueError(f"Tool not found: {effective_name}")
            if decision.tool_name is not None or decision.arguments is not None:
                if "__arg" in effective_payload and len(effective_payload) == 1:
                    result = target(effective_payload["__arg"])
                else:
                    result = target(**effective_payload)
            else:
                result = fn(*args, **kwargs)
        except Exception as exc:
            failure_result = f"Tool execution failed: {exc}"
            middleware.after_tool(
                state if isinstance(state, dict) else {},
                effective_name,
                effective_payload,
                failure_result,
                True,
            )
            if isinstance(state, dict):
                state.setdefault("_react_guard_events", []).append({
                    "tool": name,
                    "tool_input": rendered_input,
                    "aegis": aegis_decision,
                    "progent": {"before": progent_decision, "after": state.pop("_last_progent_decision", None)},
                    "pro2guard": pro2guard_decision,
                    "agentspec": agentspec_decision,
                    "toolsafe": toolsafe_decision,
                    "agentdog": agentdog_decision,
                    "agentguard": {"before": agentguard_decision, "after": (state.get("_last_agentguard_decision") if isinstance(state, dict) else None)},
                    "blocked": False,
                })
                _record_react_runtime_step(state, failure_result)
            raise

        result_decision = middleware.after_tool(state if isinstance(state, dict) else {}, effective_name, effective_payload, str(result), False)
        result = result_decision.result
        if isinstance(state, dict):
            state.setdefault("_react_guard_events", []).append({
                "tool": name,
                "tool_input": rendered_input,
                "aegis": aegis_decision,
                "progent": {"before": progent_decision, "after": state.pop("_last_progent_decision", None)},
                "pro2guard": pro2guard_decision,
                "agentspec": agentspec_decision,
                "toolsafe": toolsafe_decision,
                "agentdog": agentdog_decision,
                "agentguard": {"before": agentguard_decision, "after": state.get("_last_agentguard_decision")},
                "blocked": False,
            })
            _record_react_runtime_step(state, str(result))
        if cfg.monitoring.print_trace:
            print("[TOOL OUTPUT]", flush=True)
            print(result, flush=True)
        return result

    try:
        _wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        _wrapped.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {})
    except Exception:
        pass
    return _wrapped


def build_graph(cfg: AppConfig) -> Any:
    if cfg.graph.type == "langchain_react":
        return _build_langchain_react_graph(cfg)

    llm = LLMAdapter(cfg.llm)
    tools = {t.name: load_tool(t) for t in cfg.tools}

    builder: StateGraph = StateGraph(dict)
    builder.add_node("agent", agent_node(cfg, llm))
    builder.add_node("tool", tool_node(cfg, tools, llm.estimate_tokens))

    def _route(state: dict[str, Any]) -> str:
        if state.get("tool_call"):
            if int(state.get("iterations", 0)) >= cfg.graph.max_iters:
                return END
            return "tool"
        return END

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", _route)

    def _after_tool_route(state: dict[str, Any]) -> str:
        if state.get("_terminate_after_tool"):
            return END
        return "agent"

    builder.add_conditional_edges("tool", _after_tool_route)
    return builder.compile()


def _langchain_react_supports_stop(cfg: AppConfig) -> bool:
    provider = cfg.llm.provider.strip().lower()
    model = cfg.llm.model.strip().lower()
    return not (provider == "openai" and model.startswith("gpt-5"))


def _react_parsing_feedback(error: Exception) -> str:
    return (
        "FORMAT_ERROR: "
        + str(error)
        + "\nEvery response must contain exactly one Thought, one Action, and "
        "one Action Input JSON object. Use a registered tool action to continue. "
        "To end, use Action: finish with "
        'Action Input: {"answer": "your final response"}. '
        "Plain-text answers and Final Answer are invalid."
    )



def _build_langchain_react_graph(cfg: AppConfig) -> Any:
    llm = LLMAdapter(cfg.llm)

    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        from langchain_core.prompts import (  # type: ignore
            ChatPromptTemplate,
            MessagesPlaceholder,
            PromptTemplate,
        )
        from langchain_core.messages import SystemMessage  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime import
        raise RuntimeError("Missing dependency: langchain_core") from exc

    create_react_agent = None
    create_tool_calling_agent = None
    initialize_agent = None
    AgentType = None
    AgentExecutor = None

    try:
        from langchain_classic.agents import create_react_agent, AgentExecutor  # type: ignore
    except Exception:
        try:
            from langchain.agents import create_react_agent  # type: ignore
        except Exception:
            try:
                from langchain.agents.react.agent import create_react_agent  # type: ignore
            except Exception:
                try:
                    from langchain.agents import initialize_agent, AgentType  # type: ignore
                except Exception:
                    try:
                        from langchain_classic.agents import initialize_agent, AgentType  # type: ignore
                    except Exception as exc:  # pragma: no cover - runtime import
                        raise RuntimeError(
                            "Missing dependency: langchain or langchain-classic (react agent)"
                        ) from exc

    if AgentExecutor is None:
        try:
            from langchain.agents import AgentExecutor  # type: ignore
        except Exception:
            try:
                from langchain.agents.agent import AgentExecutor  # type: ignore
            except Exception:
                try:
                    from langchain.agents.agent_executor import AgentExecutor  # type: ignore
                except Exception:
                    try:
                        from langchain_classic.agents import AgentExecutor  # type: ignore
                    except Exception as exc:  # pragma: no cover - runtime import
                        raise RuntimeError(
                            "Missing dependency: langchain or langchain-classic (AgentExecutor)"
                        ) from exc

    for module_name in ("langchain_classic.agents", "langchain.agents"):
        try:
            module = __import__(
                module_name, fromlist=["create_tool_calling_agent"]
            )
            create_tool_calling_agent = getattr(
                module, "create_tool_calling_agent"
            )
            break
        except Exception:
            continue

    react_protocol = str(
        getattr(cfg.graph, "react_protocol", "action_only") or "action_only"
    ).strip().lower()
    if react_protocol not in {"action_only", "legacy", "native_tool_calling"}:
        raise ValueError(
            "graph.react_protocol must be action_only, legacy, or native_tool_calling, got "
            f"{react_protocol!r}"
        )

    openai_transport = str(
        getattr(cfg.graph, "openai_transport", "chat_completions")
        or "chat_completions"
    ).strip().lower()
    if openai_transport not in {"chat_completions", "responses"}:
        raise ValueError(
            "graph.openai_transport must be chat_completions or responses, got "
            f"{openai_transport!r}"
        )

    use_previous_response_id = bool(
        getattr(cfg.graph, "openai_use_previous_response_id", False)
    )

    provider = cfg.llm.provider.strip().lower()
    model = cfg.llm.model.strip().lower()
    if react_protocol == "native_tool_calling":
        if provider != "openai" or not model.startswith("gpt"):
            raise ValueError(
                "graph.react_protocol=native_tool_calling currently requires "
                "an OpenAI GPT model"
            )
        lc_model = llm.get_lc_chat_model(use_responses_api=True)
        effective_transport = "responses"
    elif (
        react_protocol == "action_only"
        and openai_transport == "responses"
        and provider == "openai"
        and model.startswith("gpt")
    ):
        lc_model = llm.get_lc_chat_model(
            use_responses_api=True,
            output_version="v0",
        )
        effective_transport = "responses"
    else:
        lc_model = llm.get_lc_chat_model()
        effective_transport = "chat_completions"

    reset_response_chain = None
    if (
        use_previous_response_id
        and react_protocol == "action_only"
        and effective_transport == "responses"
        and provider == "openai"
        and model.startswith("gpt")
    ):
        lc_model, reset_response_chain = _build_previous_response_id_text_model(
            lc_model
        )

    prompt_text = cfg.graph.react_prompt.strip()
    role_parts = [cfg.agent.system_prompt.strip()]
    skills = load_enabled_skills(cfg)
    skill_context = render_skill_context(skills)
    if skill_context:
        role_parts.append(skill_context)
    security_prompt = render_tool_output_security_prompt(cfg)
    if security_prompt:
        role_parts.append(security_prompt)
    missing_tool_warnings = validate_skill_tools(
        skills, {tool.name for tool in cfg.tools}
    )
    if missing_tool_warnings:
        role_parts.append(
            "# Harness Warnings\n"
            + "\n".join(f"- {item}" for item in missing_tool_warnings)
        )

    if react_protocol == "native_tool_calling":
        if prompt_text:
            role_parts.append(prompt_text)
        role_parts.append(
            "# Native tool protocol\n"
            "Every turn must contain exactly two text lines: Thought: a brief action "
            "rationale (not private reasoning), then Action: the exact tool name. "
            "Then issue exactly one matching native tool call with structured arguments. "
            "Text only describes the call; it never executes it. Never invent observations. "
            "To end, call finish with a non-empty answer using the same Thought/Action "
            "format. Plain-text answers cannot end the task. "
            "Use finish only after the task is actually complete."
        )
        system_content = "\n\n".join(
            part for part in role_parts if part
        ) or "You are a helpful assistant."
        PROMPT = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_content),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    elif react_protocol == "legacy":
        role_prefix = "\n\n".join(part for part in role_parts if part)
        if prompt_text:
            if role_prefix:
                prompt_text = f"{role_prefix}\n\n{prompt_text}"
            PROMPT = PromptTemplate.from_template(prompt_text)
        else:
            try:
                from langchain.agents.react.prompt import (  # type: ignore
                    PROMPT as LEGACY_PROMPT,
                )

                extra_rules = (
                    "Additional rules:\n"
                    "- Output exactly one action per response.\n"
                    "- Do not emit multiple Action blocks in one message.\n"
                    "- Do not invent Observation lines; wait for the tool result.\n"
                    "- If you are done, output Final Answer instead of another Thought-only message.\n"
                    "- Do not output </think> or other XML-style reasoning tags.\n"
                    "- The Action line must contain only the tool name, for example: Action: research_search.\n"
                    "- Put all arguments only in Action Input JSON; never write Python calls like research_search(query=...).\n\n"
                )
                if role_prefix:
                    PROMPT = PromptTemplate.from_template(
                        f"{role_prefix}\n\n{extra_rules}{LEGACY_PROMPT.template}"
                    )
                else:
                    PROMPT = PromptTemplate.from_template(
                        f"{extra_rules}{LEGACY_PROMPT.template}"
                    )
            except Exception:
                prompt_text = (
                    "You are a helpful assistant.\n\n"
                    "Answer the following questions as best you can. You have access to the following tools:\n\n"
                    "{tools}\n\n"
                    "Use the following format:\n\n"
                    "Question: the input question you must answer\n"
                    "Thought: you should always think about what to do\n"
                    "Action: the action to take, should be one of [{tool_names}]\n"
                    "Action Input: the input to the action\n"
                    "Observation: the result of the action\n"
                    "... (this Thought/Action/Action Input/Observation can repeat)\n"
                    "Thought: I now know the final answer\n"
                    "Final Answer: the final answer to the original question\n\n"
                    "Additional rules:\n"
                    "- Output exactly one action per response.\n"
                    "- Do not emit multiple Action blocks in one message.\n"
                    "- Do not invent Observation lines; wait for the tool result.\n"
                    "- If you are done, output Final Answer instead of another Thought-only message.\n"
                    "- Do not output </think> or other XML-style reasoning tags.\n"
                    "- The Action line must contain only the tool name, for example: Action: research_search.\n"
                    "- Put all arguments only in Action Input JSON; never write Python calls like research_search(query=...).\n\n"
                    "Question: {input}\n"
                    "{agent_scratchpad}"
                )
                if role_prefix:
                    prompt_text = f"{role_prefix}\n\n{prompt_text}"
                PROMPT = PromptTemplate.from_template(prompt_text)
    else:
        protocol_rules = (
            "# Action-only ReAct protocol\n"
            "The tools listed below are available and executable. Every response "
            "must contain exactly one Thought, one Action, and one Action Input. "
            "Action Input must be one valid JSON object matching the selected tool. "
            "Never output Observation; wait for the runtime result. Never return a "
            "plain-text answer or refusal. Final Answer syntax is invalid. To end "
            "the task, use Action: finish and Action Input: "
            "{{\"answer\": \"your final response\"}}. Use finish only after the task is "
            "actually complete."
        )
        role_parts.append(protocol_rules)
        role_prefix = "\n\n".join(part for part in role_parts if part)
        if prompt_text:
            prompt_text = f"{role_prefix}\n\n{prompt_text}"
        else:
            prompt_text = (
                f"{role_prefix}\n\n"
                "You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use exactly this format on every turn:\n\n"
                "Thought: explain the next step briefly\n"
                "Action: exactly one name from [{tool_names}]\n"
                "Action Input: one valid JSON object\n\n"
                "When the requested task has been completed, end with exactly:\n\n"
                "Thought: the task is complete\n"
                "Action: finish\n"
                "Action Input: {{\"answer\": \"concise final response\"}}\n\n"
                "Question: {input}\n"
                "{agent_scratchpad}"
            )
        PROMPT = PromptTemplate.from_template(prompt_text)

    tool_functions = {tool.name: load_tool(tool) for tool in cfg.tools}
    raw_tools: dict[str, Any] = {}
    tools = []
    middleware = build_middleware_manager(cfg)
    active_state: dict[str, Any] | None = None

    def _get_active_state() -> dict[str, Any]:
        return active_state if active_state is not None else {}

    def _guard_firewall_actions(turn: Any) -> Any:
        from .llamafirewall.middleware import guard_react_actions
        return guard_react_actions(turn, middleware, _get_active_state())

    def _with_firewall_actions(agent: Any) -> Any:
        if not cfg.llamafirewall.enabled:
            return agent
        from langchain_core.runnables import RunnableLambda
        return agent | RunnableLambda(_guard_firewall_actions)

    for t in cfg.tools:
        fn = tool_functions[t.name]
        lc_fn = _build_traced_react_tool(t.name, fn, cfg, middleware, _get_active_state, tool_functions)
        raw_tools[t.name] = lc_fn
        tools.append(
            StructuredTool.from_function(
                lc_fn,
                name=t.name,
                description=t.description or "",
                handle_tool_error=True,
                handle_validation_error=True,
            )
        )

    output_parser = None
    parsing_error_handler: Any = True
    if react_protocol == "action_only":
        def _finish(answer: str) -> str:
            """Finish the task and return the final answer to the user."""
            return answer

        tools.append(
            StructuredTool.from_function(
                _finish,
                name="finish",
                description=(
                    "Finish the task only after it is complete. The answer "
                    "argument is returned to the user."
                ),
            )
        )
        output_parser = _build_react_output_parser(
            {t.name for t in cfg.tools}
        )
        parsing_error_handler = _react_parsing_feedback
    elif react_protocol == "legacy":
        output_parser = _build_legacy_react_output_parser(
            {t.name for t in cfg.tools}
        )

    callbacks = _build_react_callbacks(cfg.monitoring.print_trace)
    native_diagnostics: list[dict[str, Any]] = []

    def record_native(event: dict[str, Any]) -> None:
        entry = {"timestamp": time.time(), **event}
        native_diagnostics.append(entry)
        if active_state is None:
            return
        persist = active_state.get("_trace_persist", {})
        run_dir = persist.get("run_dir") or persist.get("job_dir")
        if run_dir:
            try:
                with (Path(run_dir) / "native_protocol_debug.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                # Diagnostics must not change agent execution behavior.
                if cfg.monitoring.print_trace:
                    print(f"[NATIVE DIAGNOSTIC WRITE ERROR] {exc}", flush=True)

    if react_protocol == "native_tool_calling":
        from langchain_core.callbacks import BaseCallbackHandler

        class NativeDiagnosticCallback(BaseCallbackHandler):
            def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
                record_native({"event": "model_call_started"})

            def on_llm_end(self, response: Any, **kwargs: Any) -> None:
                for group in response.generations:
                    for generation in group:
                        message = getattr(generation, "message", None)
                        record_native({
                            "event": "model_response",
                            "content": getattr(message, "content", generation.text),
                            "tool_calls": getattr(message, "tool_calls", []),
                            "invalid_tool_calls": getattr(message, "invalid_tool_calls", []),
                        })

            def on_agent_action(self, action: Any, **kwargs: Any) -> None:
                if getattr(action, "tool", "") == "_Exception":
                    record_native({"event": "parser_error", "reason": action.tool_input,
                                   "log": action.log})
                    if cfg.monitoring.print_trace:
                        print(f"[PARSER ERROR] {action.tool_input}", flush=True)

        callbacks = [*callbacks, NativeDiagnosticCallback()]

    if react_protocol == "native_tool_calling":
        from langchain_core.runnables import RunnableLambda

        def _native_finish(answer: str) -> str:
            """Finish the completed task with its final answer."""
            return answer

        tools.append(StructuredTool.from_function(_native_finish, name="finish"))
        if create_tool_calling_agent is None:
            raise RuntimeError(
                "native_tool_calling requires LangChain "
                "create_tool_calling_agent support"
            )
        agent = create_tool_calling_agent(_SerialNativeToolModel(lc_model), tools, PROMPT)
        agent = agent | RunnableLambda(lambda parsed: _validate_native_tool_turn(parsed, record_native))
        agent = _with_firewall_actions(agent)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=cfg.graph.react_max_iterations,
            max_execution_time=cfg.graph.react_max_execution_time,
            handle_parsing_errors=True,
            callbacks=callbacks,
        )
    else:
        if output_parser is None:
            raise RuntimeError(
                f"{react_protocol} ReAct requires a compatible output parser"
            )
        if not create_react_agent:
            raise RuntimeError(
                f"{react_protocol} ReAct requires LangChain "
                "create_react_agent support"
            )
        create_kwargs: dict[str, Any] = {
            "output_parser": output_parser,
        }
        if (
            reset_response_chain is not None
            or not _langchain_react_supports_stop(cfg)
        ):
            # Responses continuation and GPT-5 endpoints do not use the
            # LangChain text-ReAct stop request parameter.
            # Native tool calling does not use this text stop parameter.
            create_kwargs["stop_sequence"] = False
        agent = create_react_agent(lc_model, tools, PROMPT, **create_kwargs)
        agent = _with_firewall_actions(agent)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=cfg.graph.react_max_iterations,
            max_execution_time=cfg.graph.react_max_execution_time,
            handle_parsing_errors=parsing_error_handler,
            callbacks=callbacks,
        )


    def _node(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_state
        start = time.time()
        active_state = state
        native_diagnostics.clear()
        if reset_response_chain is not None:
            # A response chain belongs to one top-level agent run only.
            reset_response_chain()
        state.pop("_react_runtime_steps", None)
        state.pop("_agentdog_react_steps", None)
        runtime_steps: list[dict[str, Any]] = []
        user_input = _build_react_user_input(state)
        plan_context = render_plan_context(state)
        if plan_context:
            user_input = f"{user_input}\n\n{plan_context}"
        if cfg.monitoring.print_trace:
            print("\n[LLM INPUT]", flush=True)
            if user_input:
                print(user_input, flush=True)
            else:
                print("(no user input)", flush=True)
        invoke_kwargs: dict[str, Any] = {}
        if callbacks:
            invoke_kwargs["callbacks"] = callbacks
        input_guard = middleware.guard_model_input(
            state, ([{"role": "system", "content": role_prefix}]
                    if cfg.llamafirewall.enabled else [])
            + [{"role": "user", "content": user_input}]
        )
        if input_guard.messages:
            user_input = "\n\n".join(
                str(message.get("content", ""))
                for message in input_guard.messages
                if message.get("role") != "system"
            )
        if not input_guard.allowed:
            result = {
                "output": input_guard.content
                or json.dumps(
                    {"agentguard": "blocked", "phase": "llm_before", "reason": input_guard.reason},
                    ensure_ascii=False,
                ),
                "intermediate_steps": [],
            }
            output = str(result["output"])
        else:
            attempts = 0
            try:
                result = executor.invoke({"input": user_input}, **invoke_kwargs)
            except ToolExecutionTerminated as exc:
                runtime_steps = list(
                    state.pop("_react_runtime_steps", []) or []
                )
                result = {
                    "output": exc.result,
                    "intermediate_steps": [],
                }
                output = exc.result
            except Exception:
                active_state = None
                raise
            else:
                raw_output = str(result.get("output", ""))
                normalized_steps = normalize_react_intermediate_steps(
                    result.get("intermediate_steps", []) or []
                )
                if normalized_steps:
                    state["_agentdog_react_steps"] = normalized_steps
                while True:
                    output_guard = middleware.guard_model_output(
                        state, raw_output, None
                    )
                    output = (
                        output_guard.content
                        if output_guard.content is not None
                        else raw_output
                    )
                    if (
                        not output_guard.retry
                        or attempts >= output_revision_limit(cfg)
                    ):
                        break
                    attempts += 1
                    revision = llm.chat(
                        build_revision_messages(
                            cfg,
                            state,
                            rejected_content=raw_output,
                            feedback=output_guard.feedback,
                        )
                    )
                    raw_output = revision.content
                state.pop("_agentdog_react_steps", None)
        if not runtime_steps:
            state.pop("_react_runtime_steps", None)
        state["messages"].append({"role": "assistant", "content": output})
        middleware.after_model(state, output, None)

        if react_protocol == "native_tool_calling":
            raw_steps = result.get("intermediate_steps", []) or []
            parser_errors = [
                {"reason": action.tool_input, "log": action.log, "observation": observation}
                for action, observation in raw_steps
                if getattr(action, "tool", "") == "_Exception"
            ]
            diagnostics = {
                "parser_error_count": len(parser_errors),
                "model_call_count": sum(e["event"] == "model_call_started" for e in native_diagnostics),
                "executed_tool_step_count": len(runtime_steps) if runtime_steps else sum(
                    getattr(action, "tool", "") != "_Exception" for action, _ in raw_steps
                ),
                "elapsed_seconds": time.time() - start,
                "max_iterations": cfg.graph.react_max_iterations,
                "max_execution_time": cfg.graph.react_max_execution_time,
            }
            record_native({"event": "execution_ended", **diagnostics})
            state.setdefault("trace", []).append({
                "step": "native_protocol_diagnostics", "timestamp": time.time(),
                "output": {**diagnostics, "parser_errors": parser_errors,
                           "events": list(native_diagnostics)},
            })

        steps = runtime_steps or _expand_react_steps(
            result.get("intermediate_steps", []),
            estimate_tokens=llm.estimate_tokens,
            print_trace=cfg.monitoring.print_trace and bool(callbacks),
            raw_tools=raw_tools if react_protocol == "legacy" else None,
        )
        guard_events = state.pop("_react_guard_events", [])
        for idx, step in enumerate(steps):
            if idx < len(guard_events):
                step["aegis"] = guard_events[idx].get("aegis")
                step["progent"] = guard_events[idx].get("progent")
                step["pro2guard"] = guard_events[idx].get("pro2guard")
                step["agentspec"] = guard_events[idx].get("agentspec")
                step["toolsafe"] = guard_events[idx].get("toolsafe")
                step["agentdog"] = guard_events[idx].get("agentdog")
                step["agentguard"] = guard_events[idx].get("agentguard")
                step["blocked"] = bool(guard_events[idx].get("blocked"))
        active_state = None
        for step in steps:
            mark_plan_progress(state, "react_tool", str(step.get("tool", "")))
            tool_input = step.get("tool_input", "")
            log_text = step.get("log", "")
            thought_text = step.get("thought")
            tool_usage = step.get("usage", {})
            aegis_decision = step.get("aegis")
            progent_decision = step.get("progent")
            pro2guard_decision = step.get("pro2guard")
            agentspec_decision = step.get("agentspec")
            toolsafe_decision = step.get("toolsafe")
            agentdog_decision = step.get("agentdog")
            agentguard_decision = step.get("agentguard")
            blocked = bool(step.get("blocked"))
            _append_trace_message(
                state,
                {
                    "role": "assistant",
                    "content": log_text or "",
                    "tool_calls": [
                        {
                            "type": "tool_call",
                            "name": step.get("tool", ""),
                            "arguments": _maybe_parse_json(tool_input),
                        }
                    ],
                    "function_call": None,
                    "provider_specific_fields": {
                        "refusal": None,
                        "reasoning": thought_text or None,
                    },
                    "extra": {
                        "timestamp": time.time(),
                        "response": {
                            "model": cfg.llm.model,
                            "provider": cfg.llm.provider,
                            "log": log_text,
                        },
                        "actions": [
                            {
                                "tool": step.get("tool", ""),
                                "arguments": _maybe_parse_json(tool_input),
                            }
                        ],
                        "usage": tool_usage,
                        "aegis": aegis_decision,
                        "progent": progent_decision,
                        "pro2guard": pro2guard_decision,
                        "agentspec": agentspec_decision,
                        "toolsafe": toolsafe_decision,
                        "agentdog": agentdog_decision,
                        "agentguard": agentguard_decision,
                    },
                },
            )
            _append_trace_message(
                state,
                {
                    "role": "user",
                    "content": str(step.get("observation", "")),
                    "extra": {
                        "tool": step.get("tool", ""),
                        "tool_input": tool_input,
                        "raw_output": str(step.get("observation", "")),
                        "returncode": 1 if blocked else 0,
                        "exception_info": str(step.get("observation", "")) if blocked else "",
                        "timestamp": time.time(),
                        "usage": tool_usage,
                        "aegis": aegis_decision,
                        "progent": progent_decision,
                        "pro2guard": pro2guard_decision,
                        "agentspec": agentspec_decision,
                        "toolsafe": toolsafe_decision,
                        "agentdog": agentdog_decision,
                        "agentguard": agentguard_decision,
                    },
                },
            )
            _flush_trace_snapshot(state)
            if cfg.monitoring.print_trace:
                if not callbacks:
                    if thought_text:
                        print("\n[THOUGHT]")
                        print(thought_text)
                    print("\n[TOOL INPUT]")
                    print(f"{step.get('tool', '')} {tool_input}")
                print("[TOOL OUTPUT]")
                print(step.get("observation", ""))
                print(f"[TOKENS] input={tool_usage.get('input_tokens')} output={tool_usage.get('output_tokens')} total={tool_usage.get('total_tokens')} source={tool_usage.get('source')}")

        complete_plan_on_final(state)
        trace = state.setdefault("trace", [])
        prompt_tokens = llm.estimate_tokens(user_input)
        completion_tokens = llm.estimate_tokens(output)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "source": "estimated",
        }
        _update_usage_totals(state, usage)
        trace.append(
            {
                "step": "langchain_react",
                "timestamp": start,
                "latency_ms": int((time.time() - start) * 1000),
                "input": {
                    "input": user_input,
                    "protocol": react_protocol,
                    "transport": effective_transport,
                },
                "output": {"content": output, "intermediate_steps": steps},
                "usage": usage,
            }
        )
        _append_trace_message(
            state,
            {
                "role": "assistant",
                "content": output,
                "tool_calls": None,
                "function_call": None,
                "provider_specific_fields": {
                    "refusal": None,
                    "reasoning": None,
                },
                "extra": {
                    "timestamp": time.time(),
                    "response": {
                        "model": cfg.llm.model,
                        "provider": cfg.llm.provider,
                        "intermediate_steps": len(steps),
                        "protocol": react_protocol,
                        "transport": effective_transport,
                    },
                    "actions": [],
                    "latency_ms": int((time.time() - start) * 1000),
                    "usage": usage,
                },
            },
        )
        _flush_trace_snapshot(state)
        if cfg.monitoring.print_trace and not callbacks:
            print("[LLM OUTPUT]")
            print(output)
            print(f"[TOKENS] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')} source={usage.get('source')}")
        elif cfg.monitoring.print_trace:
            print(f"[TOKENS] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')} source={usage.get('source')}")
        return state

    builder: StateGraph = StateGraph(dict)
    builder.add_node("react", _node)
    builder.set_entry_point("react")
    builder.add_edge("react", END)
    return builder.compile()


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            return value
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return value
    return value
