from __future__ import annotations

import importlib
import json
import json
import re
import time
from typing import Any, Callable

from .config import AppConfig, ToolConfig
from .llm import LLMAdapter


ToolFn = Callable[..., str]


def load_tool(tool_cfg: ToolConfig) -> ToolFn:
    module_name, attr = tool_cfg.import_path.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"Tool is not callable: {tool_cfg.import_path}")
    return fn


def _tool_prompt(tools: list[ToolConfig], call_format: str) -> str:
    lines = [
        "You can use the following tools:",
    ]
    for t in tools:
        desc = f" - {t.name}: {t.description}".rstrip()
        lines.append(desc)
    lines.append("")
    lines.append(f"To call a tool, output exactly: {call_format}")
    lines.append('The <json> is the argument object, e.g.: TOOL_CALL: calculator {"expression": "1+2"}')
    return "\n".join(lines)


def build_initial_messages(cfg: AppConfig) -> list[dict[str, str]]:
    system_prompt = cfg.agent.system_prompt
    if cfg.tools:
        system_prompt = f"{system_prompt}\n\n{_tool_prompt(cfg.tools, cfg.graph.tool_call_format)}"
    return [{"role": "system", "content": system_prompt}]


_TOOL_RE = re.compile(r"^TOOL_CALL:\s*([a-zA-Z0-9_\-]+)\s*(\{.*\})\s*$", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    match = _TOOL_RE.match(text.strip())
    if not match:
        return None
    name = match.group(1)
    payload = json.loads(match.group(2))
    if not isinstance(payload, dict):
        raise ValueError("Tool payload must be a JSON object")
    return name, payload


def agent_node(cfg: AppConfig, llm: LLMAdapter) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        messages = state["messages"]
        input_messages = [dict(m) for m in messages]
        if cfg.monitoring.print_trace:
            last_user = ""
            for msg in reversed(input_messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            print("\n[LLM INPUT]")
            if last_user:
                print(last_user)
            else:
                print("(no user input)")
        response = llm.chat(messages)
        messages.append({"role": "assistant", "content": response.content})
        call = parse_tool_call(response.content)
        state["tool_call"] = call
        end = time.time()
        usage = response.usage
        if not usage:
            prompt_tokens = sum(llm.estimate_tokens(m.get("content", "")) for m in input_messages)
            completion_tokens = llm.estimate_tokens(response.content)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "source": "estimated",
            }
        trace = state.setdefault("trace", [])
        trace.append(
            {
                "step": "agent",
                "timestamp": start,
                "latency_ms": int((end - start) * 1000),
                "input": {"messages": input_messages},
                "output": {"content": response.content, "tool_call": call},
                "usage": usage,
            }
        )
        if cfg.monitoring.print_trace:
            print("[LLM OUTPUT]")
            print(response.content)
            if usage:
                print(f"[TOKENS] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')} source={usage.get('source')}")
            if call:
                print(f"[TOOL CALL] {call[0]} {call[1]}")
        return state

    return _run


def tool_node(
    cfg: AppConfig, tools: dict[str, ToolFn], estimate_tokens: Callable[[str], int]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        call = state.get("tool_call")
        if not call:
            return state
        start = time.time()
        name, payload = call
        if name not in tools:
            result = f"Tool not found: {name}"
        else:
            try:
                result = str(tools[name](**payload))
            except Exception as exc:
                result = f"Tool execution failed: {exc}"
        state["messages"].append({"role": "assistant", "content": f"TOOL_RESULT: {result}"})
        state["tool_call"] = None
        state["iterations"] = int(state.get("iterations", 0)) + 1
        end = time.time()
        input_text = json.dumps({"tool": name, "args": payload}, ensure_ascii=False)
        output_text = str(result)
        usage = {
            "input_tokens": estimate_tokens(input_text),
            "output_tokens": estimate_tokens(output_text),
            "total_tokens": estimate_tokens(input_text) + estimate_tokens(output_text),
            "source": "estimated",
        }
        trace = state.setdefault("trace", [])
        trace.append(
            {
                "step": "tool",
                "timestamp": start,
                "latency_ms": int((end - start) * 1000),
                "input": {"tool": name, "args": payload},
                "output": {"result": result},
                "usage": usage,
            }
        )
        if cfg.monitoring.print_trace:
            print("\n[TOOL INPUT]")
            print(f"{name} {payload}")
            print("[TOOL OUTPUT]")
            print(result)
            print(f"[TOKENS] input={usage.get('input_tokens')} output={usage.get('output_tokens')} total={usage.get('total_tokens')} source={usage.get('source')}")
        return state

    return _run
