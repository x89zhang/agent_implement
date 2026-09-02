from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig


@dataclass(frozen=True)
class AgentDoGTrajectory:
    formatted: str
    tool_list_text: str
    step_count: int
    truncated: bool = False


def build_agentdog_trajectory(
    cfg: AppConfig,
    state: dict[str, Any],
    *,
    candidate_content: str = "",
    candidate_tool_call: Any = None,
    max_chars: int = 0,
) -> AgentDoGTrajectory:
    source = state.get("trace_messages") or state.get("messages") or []
    profile_parts: list[str] = []
    conversation: list[str] = []

    for message in source:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = _content(message.get("content", ""))
        if role == "system":
            if content:
                profile_parts.append(content)
            continue
        rendered = _render_message(message, role, content)
        if rendered:
            conversation.append(rendered)

    react_steps = state.get("_agentdog_react_steps")
    if not isinstance(react_steps, list):
        react_steps = state.get("_react_runtime_steps", []) or []
    for step in react_steps:
        if not isinstance(step, dict):
            continue
        conversation.extend(_render_react_step(step))

    if candidate_tool_call:
        name, arguments = _tool_call_parts(candidate_tool_call)
        parts = []
        if candidate_content and not _looks_like_tool_syntax(candidate_content):
            parts.append(f"[CONTENT]: {candidate_content.strip()}")
        parts.append(
            "[ACTION]: "
            + json.dumps(
                {"name": name, "arguments": arguments}, ensure_ascii=False, default=str
            )
        )
        conversation.append("[AGENT]:\n" + "\n".join(parts))
    elif candidate_content.strip():
        conversation.append(f"[AGENT]:\n[CONTENT]: {candidate_content.strip()}")

    configured_profile = str(getattr(cfg.agent, "system_prompt", "") or "").strip()
    if configured_profile and configured_profile not in profile_parts:
        profile_parts.insert(0, configured_profile)
    tool_list_text = _tool_list(cfg)
    if tool_list_text:
        profile_parts.append("Available tools:\n" + tool_list_text)

    prefix = ""
    if profile_parts:
        prefix = "=== Agent Profile ===\n" + "\n\n".join(profile_parts) + "\n\n"
    header = prefix + "=== Conversation History ==="
    formatted, truncated = _limit_trajectory(header, conversation, max_chars)
    return AgentDoGTrajectory(
        formatted=formatted,
        tool_list_text=tool_list_text,
        step_count=len(conversation),
        truncated=truncated,
    )


def build_revision_messages(
    cfg: AppConfig,
    state: dict[str, Any],
    *,
    rejected_content: str,
    feedback: str,
) -> list[dict[str, str]]:
    trajectory = build_agentdog_trajectory(
        cfg, state, candidate_content=rejected_content
    )
    system = str(getattr(cfg.agent, "system_prompt", "") or "").strip()
    instruction = (
        "Revise only the candidate final response using the safety diagnosis below. "
        "Do not call tools or claim that any new action was performed. Preserve useful, "
        "truthful parts of the answer while removing or correcting the diagnosed unsafe "
        "behavior. Return only the revised final response.\n\n"
        f"Safety diagnosis:\n{feedback}\n\n"
        f"Execution trajectory and rejected candidate:\n{trajectory.formatted}"
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})
    return messages


def normalize_react_intermediate_steps(
    intermediate_steps: list[Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in intermediate_steps:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        action, observation = item
        if _content(getattr(action, "tool", "")) == "_Exception":
            continue
        log = _content(getattr(action, "log", ""))
        thought = ""
        for line in log.splitlines():
            if line.strip().lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
        normalized.append(
            {
                "tool": _content(getattr(action, "tool", "")),
                "tool_input": getattr(action, "tool_input", {}),
                "observation": _content(observation),
                "log": log,
                "thought": thought,
            }
        )
    return normalized


def _render_message(message: dict[str, Any], role: str, content: str) -> str:
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    if role in {"tool", "environment"} or extra.get("tool"):
        raw = extra.get("raw_output", content)
        return f"[ENVIRONMENT]: {_content(raw)}"
    if role == "user":
        if content.startswith("TOOL_RESULT:"):
            return f"[ENVIRONMENT]: {content.removeprefix('TOOL_RESULT:').strip()}"
        return f"[USER]: {content}" if content else ""
    if role != "assistant":
        return f"[{role.upper()}]: {content}" if content else ""

    parts: list[str] = []
    reasoning = ""
    provider = message.get("provider_specific_fields")
    if isinstance(provider, dict):
        reasoning = _content(provider.get("reasoning", ""))
    if reasoning:
        parts.append(f"[THOUGHT]: {reasoning}")
    if content and not _looks_like_tool_syntax(content):
        parts.append(f"[CONTENT]: {content}")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            parts.append(
                "[ACTION]: "
                + json.dumps(
                    {
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
    if not parts and content:
        parts.append(f"[CONTENT]: {content}")
    return "[AGENT]:\n" + "\n".join(parts) if parts else ""


def _render_react_step(step: dict[str, Any]) -> list[str]:
    thought = _content(step.get("thought") or step.get("log") or "")
    name = _content(step.get("tool", ""))
    arguments = step.get("tool_input", {})
    agent_parts = []
    if thought:
        agent_parts.append(f"[THOUGHT]: {thought}")
    agent_parts.append(
        "[ACTION]: "
        + json.dumps(
            {"name": name, "arguments": arguments}, ensure_ascii=False, default=str
        )
    )
    return [
        "[AGENT]:\n" + "\n".join(agent_parts),
        f"[ENVIRONMENT]: {_content(step.get('observation', ''))}",
    ]


def _tool_call_parts(tool_call: Any) -> tuple[str, Any]:
    if isinstance(tool_call, tuple) and len(tool_call) >= 2:
        return str(tool_call[0]), tool_call[1]
    if isinstance(tool_call, dict):
        return str(tool_call.get("name", "")), tool_call.get("arguments", {})
    return "tool", tool_call


def _tool_list(cfg: AppConfig) -> str:
    rendered = []
    for tool in cfg.tools:
        rendered.append(
            json.dumps(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "capabilities": list(tool.capabilities),
                    "labels": dict(tool.labels),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    return "\n".join(rendered)


def _limit_trajectory(
    header: str, conversation: list[str], max_chars: int
) -> tuple[str, bool]:
    full = header + ("\n\n" + "\n\n".join(conversation) if conversation else "")
    if max_chars <= 0 or len(full) <= max_chars:
        return full, False
    marker = "[SYSTEM]: Earlier trajectory steps omitted because agentdog.max_trajectory_chars was exceeded."
    budget = max_chars - len(header) - len(marker) - 4
    if budget <= 0:
        raise ValueError(
            "agentdog.max_trajectory_chars is too small for the AgentDoG profile"
        )
    kept: list[str] = []
    used = 0
    for item in reversed(conversation):
        cost = len(item) + (2 if kept else 0)
        if used + cost > budget:
            if not kept:
                raise ValueError(
                    "agentdog.max_trajectory_chars is too small to retain the "
                    "latest trajectory step"
                )
            break
        kept.append(item)
        used += cost
    kept.reverse()
    return header + "\n\n" + marker + ("\n\n" + "\n\n".join(kept) if kept else ""), True


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in value
        ).strip()
    if value is None:
        return ""
    return str(value).strip()


def _looks_like_tool_syntax(content: str) -> bool:
    value = content.strip()
    return value.startswith("TOOL_CALL:") or (
        "Action:" in value and "Action Input:" in value
    )


def is_tool_syntax(content: str) -> bool:
    return _looks_like_tool_syntax(content)
