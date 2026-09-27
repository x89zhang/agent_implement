"""Build upstream StepGuard prompts from Hermes (OpenAI chat) messages.

History construction follows upstream's AgentDojo dynamic adapter
(``src/evals/dynamic/agentdojo.py`` ``_build_history``/``query``); rendering is
ported from ``src/guardrail/prompts/stepguard.py`` (``_render_action``,
``_serialize_history``, ``_serialize_trajectory``, ``StepGuardProfile``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from ..tool_results import unwrap_hermes_result
from .upstream_prompt import RISK_SOURCES, STEPGUARD_PROMPT, STEPGUARD_TRAJ_PROMPT


@dataclass
class Action:
    thought: str = ""
    raw_text: str = ""
    step: int = 0


@dataclass
class Observation:
    content: str
    step: int = 0


def _text(content: Any) -> str:
    """AgentDojo's get_text_content_as_str for OpenAI-style content."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _call_parts(call: Any) -> tuple[str, str, Any]:
    if not isinstance(call, dict):
        return "", "", {}
    payload = call.get("function", call)
    if not isinstance(payload, dict):
        return str(call.get("id") or ""), "", {}
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            pass
    return str(call.get("id") or ""), str(payload.get("name", "")), arguments


def action_text(name: str, arguments: Any) -> str:
    """Upstream AgentDojo raw action text: ``fn({json args})``."""
    return f"{name}({json.dumps(arguments, ensure_ascii=False)})"


def build_history(
    messages: list[dict[str, Any]],
    *,
    blocked_call_ids: Iterable[str] = (),
    blocked_contents: Iterable[str] = (),
) -> tuple[str, list[Action | Observation], int]:
    """Return ``(initial_state, steps, agent_turns)`` as upstream builds them.

    User messages are skipped (the initial request is ``user_request``; later
    ones include one-shot replan feedback). Calls blocked by StepGuard and the
    assistant text of a blocked turn are removed, as upstream's ``clean``
    blocked-history mode drops the whole blocked turn.
    """
    # A blocked call whose result is present did run (e.g. monitor-mode
    # replay of a trajectory); only calls that never executed are hidden.
    executed = {
        str(message.get("tool_call_id") or "") for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    blocked_ids = set(blocked_call_ids) - executed
    blocked_text = {text for text in blocked_contents if text}
    initial_state = ""
    steps: list[Action | Observation] = []
    step = 0
    turns = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            initial_state = _text(message.get("content"))
        elif role == "assistant":
            thought = _text(message.get("content"))
            calls = message.get("tool_calls") or []
            if calls:
                kept = False
                for call in calls:
                    call_id, name, arguments = _call_parts(call)
                    if call_id and call_id in blocked_ids:
                        continue
                    kept = True
                    steps.append(Action(thought=thought, raw_text=action_text(name, arguments), step=step))
                    step += 1
                    thought = ""
                turns += int(kept)
            elif thought and thought not in blocked_text:
                steps.append(Action(thought=thought, raw_text=thought, step=step))
                step += 1
        elif role == "tool":
            content = unwrap_hermes_result(_text(message.get("content")))
            steps.append(Observation(content=content, step=step))
    return initial_state, steps, turns


def _render_action(action: Action) -> str:
    lines: list[str] = []
    if action.thought:
        lines.append(f"[THOUGHT]: {action.thought}")
    if action.raw_text:
        lines.append(f"[ACTION]: {action.raw_text}")
    return "\n".join(lines) if lines else "[ACTION]: (empty action)"


def _serialize_history(initial_state: str, steps: list[Action | Observation]) -> str:
    parts: list[str] = []
    if initial_state:
        parts.extend(("=== System/Profile ===", initial_state, ""))
    if not steps:
        return "(no prior history)"

    action_count = 0
    last_action_step: int | None = None
    for step in steps:
        if isinstance(step, Action):
            action_count += 1
            step_id = int(step.step) if step.step else action_count
            last_action_step = step_id
            parts.append(f"[Step {step_id}] [AGENT]:\n{_render_action(step)}")
        elif isinstance(step, Observation):
            prefix = (
                f"[Observation after Step {last_action_step}] [ENVIRONMENT]:"
                if last_action_step is not None
                else "[ENVIRONMENT]:"
            )
            parts.append(f"{prefix} {step.content}")
    return "\n\n".join(parts) if parts else "(no prior history)"


def _serialize_trajectory(
    initial_state: str, user_request: str,
    steps: list[Action | Observation], action: Action,
) -> str:
    parts: list[str] = []
    if initial_state:
        parts.extend(("=== System/Profile ===", initial_state, ""))
    parts.extend(("=== Conversation History ===", f"\n[USER]: {user_request}"))

    action_count = 0
    last_action_step: int | None = None
    for step in [*steps, action]:
        if isinstance(step, Action):
            action_count += 1
            step_id = int(step.step) if step.step else action_count
            last_action_step = step_id
            parts.append(f"\n[Step {step_id}] [AGENT]:\n{_render_action(step)}")
        elif isinstance(step, Observation):
            prefix = (
                f"\n[Observation after Step {last_action_step}] [ENVIRONMENT]:"
                if last_action_step is not None
                else "\n[ENVIRONMENT]:"
            )
            parts.append(f"{prefix} {step.content}")
    return "\n".join(parts)


def _render_tools(tools: list[Any]) -> str:
    lines: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            payload = tool.get("function", tool)
            name = str(payload.get("name", "")) if isinstance(payload, dict) else ""
            description = str(payload.get("description", "")) if isinstance(payload, dict) else ""
        else:
            name, description = str(tool), ""
        first_line = description.split("\n", 1)[0].strip()
        lines.append(f"- {name}: {first_line}" if first_line else f"- {name}")
    return "\n".join(lines)


def build_prompt(
    *, task: str, history: list[dict[str, Any]], tools: list[dict[str, Any]],
    action: Action, trajectory: bool = False,
    blocked_call_ids: Iterable[str] = (), blocked_contents: Iterable[str] = (),
) -> str:
    initial_state, steps, turns = build_history(
        history, blocked_call_ids=blocked_call_ids, blocked_contents=blocked_contents,
    )
    tool_text = _render_tools(tools)
    if trajectory:
        return STEPGUARD_TRAJ_PROMPT.format(
            risk_sources=RISK_SOURCES, user_request=task,
            trajectory=_serialize_trajectory(initial_state, task, steps, action),
            tool_list_text=tool_text,
        )
    # Upstream: action.step is the number of agent turns with tool calls,
    # including the one under review (agentdojo.py _count_agent_steps).
    step = int(action.step) if action.step else sum(isinstance(item, Action) for item in steps) + 1
    return STEPGUARD_PROMPT.format(
        risk_sources=RISK_SOURCES, user_request=task,
        history_text=_serialize_history(initial_state, steps),
        current_action_text=f"[Step {step}] [AGENT]:\n{_render_action(action)}",
        tool_list_text=tool_text,
    )


def agent_turns(history: list[dict[str, Any]], blocked_call_ids: Iterable[str] = ()) -> int:
    return build_history(history, blocked_call_ids=blocked_call_ids)[2]
