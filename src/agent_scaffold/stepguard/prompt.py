"""Adapt Hermes messages to the released StepGuard prompt contract."""

from __future__ import annotations

import json
from typing import Any

from .upstream_prompt import RISK_SOURCES, STEPGUARD_PROMPT, STEPGUARD_TRAJ_PROMPT


def action_text(name: str, arguments: dict[str, Any]) -> str:
    return "[ACTION]: " + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, default=str
    )


def _render_tools(tools: list[dict[str, Any]]) -> str:
    lines = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        payload = tool.get("function", tool)
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name", ""))
        first = str(payload.get("description", "")).split("\n", 1)[0].strip()
        lines.append(f"- {name}: {first}" if first else f"- {name}")
    return "\n".join(lines)


def _render_history(messages: list[dict[str, Any]], maximum: int, task: str) -> tuple[str, str, int]:
    system = "\n\n".join(str(item.get("content") or "") for item in messages
                          if isinstance(item, dict) and item.get("role") == "system")
    parts: list[str] = []
    action_count = 0
    last_action: int | None = None
    initial_user_seen = False
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        if role == "system":
            continue
        content = str(item.get("content") or "")
        extra = item.get("extra") or {}
        if role == "tool" or (isinstance(extra, dict) and extra.get("tool")):
            prefix = (f"[Observation after Step {last_action}] [ENVIRONMENT]:"
                      if last_action is not None else "[ENVIRONMENT]:")
            parts.append(f"{prefix} {content}")
        elif role == "assistant":
            calls = item.get("tool_calls") or []
            if calls:
                for call in calls:
                    action_count += 1
                    last_action = action_count
                    if isinstance(call, dict):
                        payload = call.get("function", call)
                        name = payload.get("name", "") if isinstance(payload, dict) else ""
                        args = payload.get("arguments", {}) if isinstance(payload, dict) else {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                pass
                        action = action_text(name, args)
                    else:
                        action = f"[ACTION]: {call}"
                    thought = f"[THOUGHT]: {content}\n" if content else ""
                    parts.append(f"[Step {action_count}] [AGENT]:\n{thought}{action}")
            elif content:
                action_count += 1
                last_action = action_count
                parts.append(f"[Step {action_count}] [AGENT]:\n[ACTION]: {content}")
        elif role == "user":
            if not initial_user_seen and content == task:
                initial_user_seen = True
                continue
            initial_user_seen = True
            parts.append(f"[USER]: {content}")
    # Keep complete recent steps; never cut through an observation or action.
    while len(parts) > 1 and len("\n\n".join(parts)) > maximum:
        parts.pop(0)
    return system, "\n\n".join(parts) or "(no prior history)", action_count


def build_prompt(*, task: str, history: list[dict[str, Any]],
                 tools: list[dict[str, Any]], action: str,
                 trajectory: bool = False,
                 max_history_chars: int = 40000) -> str:
    system, history_text, count = _render_history(history, max_history_chars, task)
    current = f"[Step {count + 1}] [AGENT]:\n{action}"
    tool_text = _render_tools(tools)
    if trajectory:
        parts = []
        if system:
            parts.extend(("=== System/Profile ===", system, ""))
        parts.extend(("=== Conversation History ===", f"\n[USER]: {task}",
                      "\n" + history_text, "\n" + current))
        return STEPGUARD_TRAJ_PROMPT.format(
            risk_sources=RISK_SOURCES, user_request=task,
            trajectory="\n".join(parts), tool_list_text=tool_text,
        )
    if system:
        history_text = f"=== System/Profile ===\n{system}\n\n{history_text}"
    return STEPGUARD_PROMPT.format(
        risk_sources=RISK_SOURCES, user_request=task,
        history_text=history_text, current_action_text=current,
        tool_list_text=tool_text,
    )
