from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ..tool_results import unwrap_hermes_result
from .client import OpenAICompatibleGuardClient
from .model import ToolSafeResult
from .parser import parse_guard_response
from .prompt import build_guard_prompt, format_tools_for_prompt

# Modes that act on the guard's outcome; monitor and warn always execute.
_ENFORCING_MODES = {"replan", "block"}

# Hermes frames non-empty tool results for the agent as
# <untrusted_tool_result source="...">\n<notice>\n\n<result>\n</untrusted_tool_result>.
# The notice is Hermes' own defense, not tool output, so it is not an Observation.
_HERMES_UNTRUSTED = re.compile(
    r"^\s*<untrusted_tool_result\b[^>]*>\n(?:[^\n]*\n\n)?(.*?)\n?</untrusted_tool_result>\s*$",
    re.S,
)


class ToolSafeMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.guard = cfg.toolsafe
        api_key = self.guard.api_key
        if not api_key and self.guard.api_key_env:
            api_key = os.environ.get(self.guard.api_key_env, "")
        self.client = client or OpenAICompatibleGuardClient(
            base_url=self.guard.base_url,
            api_key=api_key,
            model=self.guard.model,
            timeout_seconds=self.guard.timeout_seconds,
        )
        self.environment = [
            {"name": tool.name, "description": tool.description}
            for tool in cfg.tools
        ]

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        # Upstream judges the assistant's own ReAct turn (Thought/Action/Action
        # Input), so remember the turn that produced this tool call.
        if tool_call is not None:
            thought = str(state.get("_model_output_content", content) or "")
            index = state.get("_model_output_index")
            turn = state.get("_toolsafe_turn")
            if index is not None:
                # GuardController reports every call of the turn with its id.
                if index == 0 or not isinstance(turn, dict):
                    calls = [
                        {"id": call.get("id"), "name": call.get("name"),
                         "arguments": call.get("arguments"), "used": False}
                        for call in state.get("_model_output_calls") or []
                        if isinstance(call, dict)
                    ]
                    state["_toolsafe_turn"] = {"thought": thought, "calls": calls, "done": []}
            else:
                key = [len(state.get("messages", []) or []), thought]
                if not isinstance(turn, dict) or turn.get("key") != key:
                    turn = {"key": key, "thought": thought, "calls": [], "done": []}
                    state["_toolsafe_turn"] = turn
                name, arguments = tool_call
                turn["calls"].append(
                    {"id": None, "name": name, "arguments": arguments, "used": False}
                )
        return ModelDecision(content=content, tool_call=tool_call)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_toolsafe_warning", "")
        return [warning] if warning else []

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        started = time.monotonic()
        call_id = _claim_call(state, name, payload)
        history, current_action = _agent_action(state, name, payload)
        limit = self.guard.max_history_steps
        if limit:
            history = history[-limit:]
        prompt = build_guard_prompt(
            user_request=_user_request(state),
            interaction_history=history,
            current_action=current_action,
            env_info=format_tools_for_prompt(
                state.get("_toolsafe_tools") or self.environment
            ),
        )
        raw_response = ""
        risk_score: float | None = None
        analysis = ""
        error = ""
        try:
            # Upstream TS-Guard retries an unparseable verdict up to three turns
            # and then returns no risk rating (model.py:139-159).
            for _ in range(3):
                raw_response = self.client.complete(prompt)
                try:
                    risk_score, analysis = parse_guard_response(raw_response)
                    break
                except ValueError:
                    continue
        except Exception as exc:
            # An endpoint exception propagates to the agent loop's try in
            # upstream, so the tool does not run (sec_react_agent.py:68-123).
            error = str(exc)
            risk_score = None
        latency_ms = int((time.monotonic() - started) * 1000)
        if error:
            parse_status = "endpoint_error"
        elif risk_score is None:
            parse_status = "no_rating"
            if self.guard.fail_closed:
                # Non-upstream opt-in: treat a missing rating like a guard error.
                error = "TS-Guard returned no parseable verdict after 3 attempts"
        else:
            parse_status = "parsed"

        enforcing = self.guard.mode in _ENFORCING_MODES
        if error:
            allowed = not enforcing
            reason = f"ToolSafe guard error: {error}"
        elif risk_score is None:
            # Upstream executes the tool when no risk rating is returned
            # (model.py:159, sec_react_agent.py:80). Not an error, not a detection.
            allowed = True
            reason = "TS-Guard returned no risk rating after 3 attempts"
        else:
            allowed = risk_score < self.guard.threshold
            reason = (
                ""
                if allowed
                else (
                    f"ToolSafe risk score {risk_score:.1f} meets or exceeds "
                    f"threshold {self.guard.threshold:.1f}: {analysis}"
                )
            )
        result = ToolSafeResult(
            risk_score=risk_score,
            threshold=self.guard.threshold,
            allowed=allowed,
            reason=reason,
            mode=self.guard.mode,
            model=self.guard.model,
            latency_ms=latency_ms,
            raw_response=raw_response,
            error=error,
        )

        replan_count = int(state.get("_toolsafe_replan_count", 0))
        max_replans = self.guard.max_replans
        action = "allow"
        decision_reason = result.reason
        if result.error:
            # Enforcing modes withhold the call with upstream's error observation
            # (sec_react_agent.py:119-123) and let the agent continue; passive
            # modes execute and only record the error.
            action = "error"
            if enforcing:
                decision = ToolDecision(
                    False,
                    result.reason,
                    replacement_result=f"Error occurred: {result.error}\n",
                    decision_type="error",
                )
            else:
                decision = ToolDecision(True, "", decision_type="error")
        elif result.allowed:
            decision = ToolDecision(True, "", decision_type="allow")
        elif self.guard.mode == "monitor":
            action = "monitor"
            decision = ToolDecision(True, "", decision_type="monitor")
        elif self.guard.mode == "warn":
            action = "warn"
            state["_toolsafe_warning"] = _feedback(result)
            decision = ToolDecision(True, "", decision_type="warn")
        elif self.guard.mode == "replan":
            replan_count += 1
            state["_toolsafe_replan_count"] = replan_count
            if max_replans is None or replan_count <= max_replans:
                action = "replan"
                decision = ToolDecision(
                    False,
                    result.reason,
                    replacement_result=_feedback(result),
                    decision_type="replan",
                )
            else:
                action = "block"
                decision_reason = (
                    f"{result.reason} ToolSafe replan budget exhausted "
                    f"({max_replans})."
                )
                decision = ToolDecision(
                    False,
                    decision_reason,
                    replacement_result=_feedback(result),
                    decision_type="block",
                    terminate=True,
                )
        else:
            action = "block"
            decision = ToolDecision(
                False,
                result.reason,
                replacement_result=_feedback(result),
                decision_type="block",
                terminate=True,
            )

        decision_data = result.to_dict()
        decision_data.update(
            {
                "action": action,
                "reason": decision_reason,
                "parse_status": parse_status,
                "detected": result.error == "" and not result.allowed,
                "replan_count": replan_count,
                "max_replans": max_replans,
                "terminate": decision.terminate,
            }
        )
        state["_last_toolsafe_decision"] = decision_data
        state.setdefault("toolsafe_events", []).append(decision_data)
        # Only the judged action itself is kept for the parallel-call history.
        state["_toolsafe_pending_action"] = {"action": current_action, "id": call_id}
        return decision

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        pending = state.pop("_toolsafe_pending_action", None)
        observation = _observation_text(result)
        if isinstance(pending, dict):
            if pending.get("id"):
                # Later history renders this call's Observation from the tool
                # result itself, not from the agent-facing message.
                state.setdefault("_toolsafe_observations", {})[pending["id"]] = observation
            turn = state.get("_toolsafe_turn")
            if isinstance(turn, dict):
                # Later parallel calls of the same assistant turn see earlier ones
                # as ordinary ReAct steps; the next model input supersedes this.
                turn["done"].extend([pending["action"], _observation(observation)])
        warning = state.pop("_toolsafe_warning", "")
        if warning:
            return ResultDecision(
                result=f"{result}\n\n{warning}", decision_type="warn"
            )
        return ResultDecision(result=result)


def _feedback(result: ToolSafeResult) -> str:
    # Exact TS-Flow observation from sec_react_agent.py:81-84; the guard's
    # reasoning is never shown to the agent.
    guard_res = {"risk rating": result.risk_score}
    return (
        f"Security Validation Before Execution:\n{guard_res}\n\n"
        "Perhaps you should try other safer tool calls."
    )


def _user_request(state: dict[str, Any]) -> str:
    # Upstream passes the query the agent received (sec_react_agent.py:77).
    for key in ("_runtime_user_request", "_toolsafe_user_request"):
        explicit = str(state.get(key) or "").strip()
        if explicit:
            return explicit
    for message in state.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = _text(message.get("content"))
        if content and not content.startswith("TOOL_RESULT:"):
            return content.strip()
    return ""


def _claim_call(state: dict[str, Any], name: str, payload: dict[str, Any]) -> Any:
    """Match this execution to a call of the current turn; return its id."""
    turn = state.get("_toolsafe_turn")
    if not isinstance(turn, dict):
        return None
    open_calls = [call for call in turn.get("calls", []) if not call["used"]]
    for call in (
        [c for c in open_calls if c["name"] == name and _same_args(c["arguments"], payload)]
        or [c for c in open_calls if c["name"] == name]
    ):
        call["used"] = True
        return call["id"]
    return None


def _same_args(left: Any, right: Any) -> bool:
    return _arguments(left) == _arguments(right)


def _agent_action(
    state: dict[str, Any], name: str, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild upstream ``bot.messages[2:]`` (sec_react_agent.py:75-77)."""
    messages = list(state.get("messages", []) or [])
    turn = state.get("_toolsafe_turn")
    if isinstance(turn, dict):
        # A native loop may already have appended the current assistant turn,
        # whose calls have no results yet.
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()
    history, pending = _react_messages(messages, state.get("_toolsafe_observations") or {})
    if isinstance(turn, dict):
        history.extend(turn["done"])
        # Reasoning the agent emitted right before this turn's calls belongs to
        # the current action's Thought.
        thought = _join([*pending, turn.get("thought", "")])
        current = {"role": "assistant", "content": _react_turn(thought, name, payload)}
        return history, current
    if pending:
        history.append({"role": "assistant", "content": _join(pending)})
    if history and history[-1]["role"] == "assistant":
        # Text-protocol graphs: the trailing assistant turn is the action.
        return history[:-1], history[-1]
    return history, {"role": "assistant", "content": _react_turn("", name, payload)}


def _react_messages(
    messages: list[Any], observations: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Render a chat transcript as upstream's plain ReAct conversation.

    Each action is followed by the observation with the matching
    ``tool_call_id``, as upstream's one-action-one-observation loop produces.
    Assistant text without tool calls (e.g. Responses-API reasoning summaries)
    is merged into the next action's Thought; empty assistant messages are
    dropped. Returns the history and assistant text not yet attached to an
    action.
    """
    results = {
        str(message.get("tool_call_id")): message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_call_id")
    }
    emitted: set[str] = set()
    history: list[dict[str, Any]] = []
    pending: list[str] = []
    seen_query = False

    def flush() -> None:
        # Assistant text followed by no tool call is a turn of its own.
        if pending:
            history.append({"role": "assistant", "content": _join(pending)})
            pending.clear()

    def observe(message: dict[str, Any]) -> None:
        call_id = str(message.get("tool_call_id") or "")
        if call_id and call_id in observations:
            text = observations[call_id]
        else:
            text = _observation_text(_text(message.get("content")))
        history.append(_observation(text))

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = _text(message.get("content"))
        if role in {"system", "developer"}:
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id and call_id in emitted:
                continue
            flush()
            observe(message)
        elif role == "user" and content.startswith("TOOL_RESULT:"):
            flush()
            history.append(_observation(_observation_text(content.removeprefix("TOOL_RESULT:").strip())))
        elif role == "user":
            flush()
            # The first user message is the query, which upstream strips.
            if not seen_query:
                seen_query = True
                continue
            history.append({"role": "user", "content": content})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if not calls:
                if content.strip():
                    pending.append(content)
                continue
            thought = _join([*pending, content])
            pending.clear()
            for call in calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                history.append(
                    {
                        "role": "assistant",
                        "content": _react_turn(
                            thought, str(fn.get("name", "")), _arguments(fn.get("arguments", {}))
                        ),
                    }
                )
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                if call_id and call_id in results:
                    emitted.add(call_id)
                    observe(results[call_id])
    return history, pending


def _react_turn(thought: str, name: str, arguments: Any) -> str:
    """Render a native tool call in the ReAct format of agent_prompts.py."""
    thought = str(thought or "").strip()
    if name and re.search(rf"Action:\s*{re.escape(name)}\b", thought):
        return thought
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
    return f"(1) Thought: {thought}\n(2) Action: {name}\n(3) Action Input: {arguments}"


def _arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _observation_text(result: Any) -> str:
    """The bare tool result, as upstream's Observation holds it."""
    text = "" if result is None else str(result)
    match = _HERMES_UNTRUSTED.match(text)
    if match:
        text = match.group(1)
    return unwrap_hermes_result(text)


def _observation(text: str) -> dict[str, Any]:
    return {"role": "user", "content": f"Observation: {text}"}


def _join(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if str(part or "").strip())


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return "" if content is None else str(content)
