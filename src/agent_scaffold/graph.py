from __future__ import annotations

import ast
import functools
import inspect
import json
import re
from typing import Any
import time

from langgraph.graph import END, StateGraph

from .config import AppConfig
from .llm import LLMAdapter
from .skills import load_enabled_skills, render_skill_context, validate_skill_tools
from .planner import render_plan_context, mark_plan_progress, complete_plan_on_final
from .middleware import build_middleware_manager
from .nodes import (
    agent_node,
    load_tool,
    tool_node,
    _append_trace_message,
    _flush_trace_snapshot,
    _update_usage_totals,
    render_tool_output_security_prompt,
)


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
    raw_tools: dict[str, Any],
    estimate_tokens: Any,
    print_trace: bool,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", "")
        log_text = getattr(action, "log", "")

        if tool_name == "_Exception":
            recovered_actions = _extract_react_actions(str(log_text))
            if recovered_actions:
                for recovered in recovered_actions:
                    recovered_name = str(recovered.get("tool", ""))
                    recovered_input = recovered.get("tool_input", {})
                    if recovered_name not in raw_tools:
                        recovered_output = f"Tool not found: {recovered_name}"
                    else:
                        try:
                            if isinstance(recovered_input, dict):
                                recovered_output = str(raw_tools[recovered_name](**recovered_input))
                            else:
                                recovered_output = str(raw_tools[recovered_name](recovered_input))
                        except Exception as exc:
                            recovered_output = f"Tool execution failed: {exc}"
                    usage = {
                        "input_tokens": estimate_tokens(str(recovered_input)),
                        "output_tokens": estimate_tokens(str(recovered_output)),
                        "total_tokens": estimate_tokens(str(recovered_input)) + estimate_tokens(str(recovered_output)),
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
                        thought_text = str(recovered.get("thought") or "").strip()
                        if thought_text:
                            print("\n[THOUGHT]")
                            print(thought_text)
                        print("\n[TOOL INPUT]")
                        print(f"{recovered_name} {recovered_input}")
                        print("[TOOL OUTPUT]")
                        print(recovered_output)
                        print(
                            f"[TOKENS] input={usage.get('input_tokens')} output={usage.get('output_tokens')} "
                            f"total={usage.get('total_tokens')} source={usage.get('source')}"
                        )
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
            module = __import__(module_name, fromlist=["ReActSingleInputOutputParser"])
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
            tool_input = _normalize_react_tool_input(getattr(parsed, "tool_input", ""))
            if isinstance(tool, str):
                normalized_name = _normalize_react_tool_name(tool, tool_names)
                if normalized_name in tool_names:
                    return AgentAction(tool=normalized_name, tool_input=tool_input, log=getattr(parsed, "log", text))
                normalized = _parse_function_style_action(tool, tool_names)
                if normalized is not None:
                    tool_name, payload = normalized
                    if "__arg" in payload and len(payload) == 1:
                        tool_input = payload["__arg"]
                    else:
                        payload.pop("__arg", None)
                        tool_input = payload
                    return AgentAction(tool=tool_name, tool_input=tool_input, log=getattr(parsed, "log", text))
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


def _build_traced_react_tool(name: str, fn: Any, cfg: AppConfig, middleware: Any, get_state: Any) -> Any:
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        rendered_input = _render_react_payload(args, kwargs)
        payload = _react_payload(args, kwargs)
        state = get_state()
        if isinstance(state, dict):
            state.pop("_last_aegis_decision", None)
        decision = middleware.before_tool(state if isinstance(state, dict) else {}, name, payload)
        aegis_decision = None
        if isinstance(state, dict):
            aegis_decision = state.pop("_last_aegis_decision", None)

        if cfg.monitoring.print_trace:
            print("\n[TOOL INPUT]", flush=True)
            print(f"{name} {rendered_input}", flush=True)

        if not decision.allowed:
            result = f"Tool execution blocked by middleware: {decision.reason}"
            middleware.after_tool(state if isinstance(state, dict) else {}, name, payload, result, True)
            if isinstance(state, dict):
                state.setdefault("_react_guard_events", []).append({
                    "tool": name,
                    "tool_input": rendered_input,
                    "aegis": aegis_decision,
                    "blocked": True,
                })
            if cfg.monitoring.print_trace:
                print("[TOOL OUTPUT]", flush=True)
                print(result, flush=True)
            return result

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            middleware.after_tool(state if isinstance(state, dict) else {}, name, payload, f"Tool execution failed: {exc}", True)
            if isinstance(state, dict):
                state.setdefault("_react_guard_events", []).append({
                    "tool": name,
                    "tool_input": rendered_input,
                    "aegis": aegis_decision,
                    "blocked": False,
                })
            raise

        middleware.after_tool(state if isinstance(state, dict) else {}, name, payload, str(result), False)
        if isinstance(state, dict):
            state.setdefault("_react_guard_events", []).append({
                "tool": name,
                "tool_input": rendered_input,
                "aegis": aegis_decision,
                "blocked": False,
            })
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
    builder.add_edge("tool", "agent")
    return builder.compile()


def _build_langchain_react_graph(cfg: AppConfig) -> Any:
    llm = LLMAdapter(cfg.llm)
    lc_model = llm.get_lc_chat_model()

    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        from langchain_core.prompts import PromptTemplate  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime import
        raise RuntimeError("Missing dependency: langchain_core") from exc

    create_react_agent = None
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

    prompt_text = cfg.graph.react_prompt.strip()
    role_parts = [cfg.agent.system_prompt.strip()]
    skills = load_enabled_skills(cfg)
    skill_context = render_skill_context(skills)
    if skill_context:
        role_parts.append(skill_context)
    security_prompt = render_tool_output_security_prompt(cfg)
    if security_prompt:
        role_parts.append(security_prompt)
    missing_tool_warnings = validate_skill_tools(skills, {tool.name for tool in cfg.tools})
    if missing_tool_warnings:
        role_parts.append("# Harness Warnings\n" + "\n".join(f"- {item}" for item in missing_tool_warnings))
    role_prefix = "\n\n".join(part for part in role_parts if part)
    if prompt_text:
        if role_prefix:
            prompt_text = f"{role_prefix}\n\n{prompt_text}"
        PROMPT = PromptTemplate.from_template(prompt_text)
    else:
        try:
            from langchain.agents.react.prompt import PROMPT  # type: ignore
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
                PROMPT = PromptTemplate.from_template(f"{role_prefix}\n\n{extra_rules}{PROMPT.template}")
            else:
                PROMPT = PromptTemplate.from_template(f"{extra_rules}{PROMPT.template}")
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

    raw_tools: dict[str, Any] = {}
    tools = []
    middleware = build_middleware_manager(cfg)
    active_state: dict[str, Any] | None = None

    def _get_active_state() -> dict[str, Any]:
        return active_state if active_state is not None else {}

    for t in cfg.tools:
        fn = load_tool(t)
        lc_fn = _build_traced_react_tool(t.name, fn, cfg, middleware, _get_active_state)
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

    callbacks = _build_react_callbacks(cfg.monitoring.print_trace)
    output_parser = _build_react_output_parser({t.name for t in cfg.tools})

    if create_react_agent:
        create_kwargs: dict[str, Any] = {}
        if output_parser is not None:
            create_kwargs["output_parser"] = output_parser
        agent = create_react_agent(lc_model, tools, PROMPT, **create_kwargs)
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
        executor = initialize_agent(
            tools,
            lc_model,
            agent=AgentType.REACT_DESCRIPTION,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=cfg.graph.react_max_iterations,
            max_execution_time=cfg.graph.react_max_execution_time,
            handle_parsing_errors=True,
            callbacks=callbacks,
        )

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_state
        start = time.time()
        active_state = state
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
        try:
            result = executor.invoke({"input": user_input}, **invoke_kwargs)
        except Exception:
            active_state = None
            raise
        output = result.get("output", "")
        state["messages"].append({"role": "assistant", "content": output})

        steps = _expand_react_steps(
            result.get("intermediate_steps", []),
            raw_tools=raw_tools,
            estimate_tokens=llm.estimate_tokens,
            print_trace=cfg.monitoring.print_trace and bool(callbacks),
        )
        guard_events = state.pop("_react_guard_events", [])
        for idx, step in enumerate(steps):
            if idx < len(guard_events):
                step["aegis"] = guard_events[idx].get("aegis")
                step["blocked"] = bool(guard_events[idx].get("blocked"))
        active_state = None
        for step in steps:
            mark_plan_progress(state, "react_tool", str(step.get("tool", "")))
            tool_input = step.get("tool_input", "")
            log_text = step.get("log", "")
            thought_text = step.get("thought")
            tool_usage = step.get("usage", {})
            aegis_decision = step.get("aegis")
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
                "input": {"input": user_input},
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
