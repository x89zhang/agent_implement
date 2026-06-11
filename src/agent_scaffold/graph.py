from __future__ import annotations

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
from .nodes import (
    agent_node,
    load_tool,
    tool_node,
    _append_trace_message,
    _flush_trace_snapshot,
    _update_usage_totals,
)


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


def _build_traced_react_tool(name: str, fn: Any, enabled: bool) -> Any:
    if not enabled:
        return fn

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            rendered_input = kwargs
        elif len(args) == 1:
            rendered_input = args[0]
        else:
            rendered_input = list(args)
        print("\n[TOOL INPUT]", flush=True)
        print(f"{name} {rendered_input}", flush=True)
        result = fn(*args, **kwargs)
        print("[TOOL OUTPUT]", flush=True)
        print(result, flush=True)
        return result

    try:
        _wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
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
                "- Do not output </think> or other XML-style reasoning tags.\n\n"
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
                "- Do not output </think> or other XML-style reasoning tags.\n\n"
                "Question: {input}\n"
                "{agent_scratchpad}"
            )
            if role_prefix:
                prompt_text = f"{role_prefix}\n\n{prompt_text}"
            PROMPT = PromptTemplate.from_template(prompt_text)

    raw_tools: dict[str, Any] = {}
    tools = []
    for t in cfg.tools:
        fn = load_tool(t)
        raw_tools[t.name] = fn
        lc_fn = _build_traced_react_tool(t.name, fn, cfg.monitoring.print_trace)
        tools.append(
            StructuredTool.from_function(
                lc_fn,
                name=t.name,
                description=t.description or "",
                handle_tool_error=True,
            )
        )

    callbacks = _build_react_callbacks(cfg.monitoring.print_trace)

    if create_react_agent:
        agent = create_react_agent(lc_model, tools, PROMPT)
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
        start = time.time()
        user_input = state["messages"][-1]["content"] if state.get("messages") else ""
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
        result = executor.invoke({"input": user_input}, **invoke_kwargs)
        output = result.get("output", "")
        state["messages"].append({"role": "assistant", "content": output})

        steps = _expand_react_steps(
            result.get("intermediate_steps", []),
            raw_tools=raw_tools,
            estimate_tokens=llm.estimate_tokens,
            print_trace=cfg.monitoring.print_trace and bool(callbacks),
        )
        for step in steps:
            mark_plan_progress(state, "react_tool", str(step.get("tool", "")))
            tool_input = step.get("tool_input", "")
            log_text = step.get("log", "")
            thought_text = step.get("thought")
            tool_usage = step.get("usage", {})
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
                        "returncode": 0,
                        "exception_info": "",
                        "timestamp": time.time(),
                        "usage": tool_usage,
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
