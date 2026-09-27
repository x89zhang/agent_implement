from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from .client import OpenAICompatibleGuardClient
from .model import ToolSafeResult
from .parser import parse_guard_response
from .prompt import build_guard_prompt, format_tools_for_prompt


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
        # Input), so remember the content that produced this tool call.
        if tool_call is not None:
            key = [len(state.get("messages", []) or []), str(content or "")]
            turn = state.get("_toolsafe_turn")
            if not isinstance(turn, dict) or turn.get("key") != key:
                state["_toolsafe_turn"] = {"key": key, "done": []}
        return ModelDecision(content=content, tool_call=tool_call)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_toolsafe_warning", "")
        return [warning] if warning else []

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        started = time.monotonic()
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
        try:
            # Upstream TS-Guard retries an unparseable verdict up to three turns
            # (model.py:139-159) and otherwise returns no risk rating.
            for attempt in range(3):
                raw_response = self.client.complete(prompt)
                try:
                    risk_score, analysis = parse_guard_response(raw_response)
                    break
                except ValueError:
                    if attempt == 2:
                        raise
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
                latency_ms=int((time.monotonic() - started) * 1000),
                raw_response=raw_response,
            )
        except Exception as exc:
            allowed = not self.guard.fail_closed
            result = ToolSafeResult(
                risk_score=None,
                threshold=self.guard.threshold,
                allowed=allowed,
                reason=(
                    f"ToolSafe failed {'closed' if self.guard.fail_closed else 'open'}: "
                    f"{exc}"
                ),
                mode=self.guard.mode,
                model=self.guard.model,
                latency_ms=int((time.monotonic() - started) * 1000),
                raw_response=raw_response,
                error=str(exc),
            )

        replan_count = int(state.get("_toolsafe_replan_count", 0))
        max_replans = self.guard.max_replans
        action = "allow"
        decision_reason = result.reason
        if result.allowed:
            # Fail open: upstream executes the tool when no risk rating is
            # returned (model.py:159).
            action = "error" if result.error else "allow"
            decision = ToolDecision(True, "", decision_type="allow")
        elif result.error:
            # A guard failure under fail_closed is an error, not a detection.
            # Enforcing modes withhold the call with upstream's error
            # observation (sec_react_agent.py:112-116); passive modes execute.
            action = "error"
            if self.guard.mode in {"monitor", "warn"}:
                decision = ToolDecision(True, "", decision_type="error")
            else:
                decision = ToolDecision(
                    False,
                    result.reason,
                    replacement_result=f"Error occurred: {result.error}\n",
                    decision_type="error",
                )
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
                "detected": result.error == "" and not result.allowed,
                "replan_count": replan_count,
                "max_replans": max_replans,
                "terminate": decision.terminate,
            }
        )
        state["_last_toolsafe_decision"] = decision_data
        state.setdefault("toolsafe_events", []).append(decision_data)
        # Only the judged action itself is kept for the parallel-call history.
        state["_toolsafe_pending_action"] = current_action
        return decision

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        action = state.pop("_toolsafe_pending_action", None)
        turn = state.get("_toolsafe_turn")
        if action is not None and isinstance(turn, dict):
            # Later parallel calls of the same assistant turn see earlier ones
            # as ordinary ReAct steps; the next model input supersedes this.
            turn["done"].extend([action, _observation(result)])
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
    explicit = str(state.get("_toolsafe_user_request") or "").strip()
    if explicit:
        return explicit
    for message in state.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = _text(message.get("content"))
        if content and not content.startswith("TOOL_RESULT:"):
            return content.strip()
    return ""


def _agent_action(
    state: dict[str, Any], name: str, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild upstream ``bot.messages[2:]`` (sec_react_agent.py:75-77)."""
    history = _react_messages(state.get("messages", []) or [])
    turn = state.get("_toolsafe_turn")
    if isinstance(turn, dict):
        content = turn["key"][1]
        # The native loop already appended the current assistant turn.
        if (
            history
            and history[-1]["role"] == "assistant"
            and history[-1]["content"] == content
        ):
            history.pop()
        history.extend(turn["done"])
        current = {"role": "assistant", "content": _react_turn(content, name, payload)}
    elif history and history[-1]["role"] == "assistant":
        current = history.pop()
    else:
        current = {"role": "assistant", "content": _react_turn("", name, payload)}
    return history, current


def _react_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Render a chat transcript as upstream's plain ReAct conversation."""
    history: list[dict[str, Any]] = []
    seen_query = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = _text(message.get("content"))
        if role == "system":
            continue
        if role == "tool" or content.startswith("TOOL_RESULT:"):
            history.append(_observation(content.removeprefix("TOOL_RESULT:").strip()))
        elif role == "user":
            # The first user message is the query, which upstream strips.
            if not seen_query:
                seen_query = True
                continue
            history.append({"role": "user", "content": content})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if not calls:
                history.append({"role": "assistant", "content": content})
            for call in calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                arguments = fn.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                history.append(
                    {
                        "role": "assistant",
                        "content": _react_turn(
                            content, str(fn.get("name", "")), arguments
                        ),
                    }
                )
    return history


def _react_turn(content: str, name: str, arguments: Any) -> str:
    """Render a native tool call in the ReAct format of agent_prompts.py."""
    content = str(content or "").strip()
    if name and re.search(rf"Action:\s*{re.escape(name)}\b", content):
        return content
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
    if content:
        return (
            f"(1) Thought: {content}\n(2) Action: {name}\n"
            f"(3) Action Input: {arguments}"
        )
    return f"(1) Action: {name}\n(2) Action Input: {arguments}"


def _observation(result: Any) -> dict[str, Any]:
    return {"role": "user", "content": f"Observation: {result}"}


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return "" if content is None else str(content)
