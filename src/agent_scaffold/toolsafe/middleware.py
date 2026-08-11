from __future__ import annotations

import json
import os
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ResultDecision, ToolDecision
from .client import OpenAICompatibleGuardClient
from .model import ToolSafeResult
from .parser import parse_guard_response
from .prompt import build_guard_prompt


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
            {
                "name": tool.name,
                "description": tool.description,
                "capabilities": list(tool.capabilities),
                "labels": dict(tool.labels),
            }
            for tool in cfg.tools
        ]

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_toolsafe_warning", "")
        return [warning] if warning else []

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        started = time.monotonic()
        prompt = build_guard_prompt(
            user_request=_user_request(state),
            interaction_history=_interaction_history(
                state, name, self.guard.max_history_steps
            ),
            tool_name=name,
            arguments=payload,
            environment=self.environment,
        )
        raw_response = ""
        try:
            raw_response = self.client.complete(prompt)
            risk_score, analysis = parse_guard_response(raw_response)
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
        action = "allow"
        decision_reason = result.reason
        if result.allowed:
            decision = ToolDecision(True, "", decision_type="allow")
        elif self.guard.mode == "monitor":
            action = "monitor"
            decision = ToolDecision(True, "", decision_type="monitor")
        elif self.guard.mode == "warn":
            action = "warn"
            state["_toolsafe_warning"] = _feedback(result, action="warn")
            decision = ToolDecision(True, "", decision_type="warn")
        elif self.guard.mode == "replan":
            replan_count += 1
            state["_toolsafe_replan_count"] = replan_count
            if replan_count <= self.guard.max_replans:
                action = "replan"
                decision = ToolDecision(
                    False,
                    result.reason,
                    replacement_result=_feedback(result, action="replan"),
                    decision_type="replan",
                )
            else:
                action = "block"
                decision_reason = (
                    f"{result.reason} ToolSafe replan budget exhausted "
                    f"({self.guard.max_replans})."
                )
                decision = ToolDecision(
                    False,
                    decision_reason,
                    replacement_result=_feedback(
                        result, action="block", reason=decision_reason
                    ),
                    decision_type="block",
                    terminate=True,
                )
        else:
            action = "block"
            decision = ToolDecision(
                False,
                result.reason,
                replacement_result=_feedback(result, action="block"),
                decision_type="block",
                terminate=True,
            )

        decision_data = result.to_dict()
        decision_data.update(
            {
                "action": action,
                "reason": decision_reason,
                "replan_count": replan_count,
                "max_replans": self.guard.max_replans,
                "terminate": decision.terminate,
            }
        )
        state["_last_toolsafe_decision"] = decision_data
        state.setdefault("toolsafe_events", []).append(decision_data)
        state["_toolsafe_pending_call"] = {
            "tool_name": name,
            "arguments": dict(payload),
            "decision": decision_data,
        }
        return decision

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        pending = state.pop("_toolsafe_pending_call", None) or {
            "tool_name": name,
            "arguments": dict(payload),
        }
        pending["result"] = str(result)
        pending["failed"] = bool(failed)
        history = state.setdefault("_toolsafe_tool_history", [])
        history.append(pending)
        limit = self.guard.max_history_steps
        if limit == 0:
            history.clear()
        elif len(history) > limit:
            del history[:-limit]
        warning = state.pop("_toolsafe_warning", "")
        if warning:
            return ResultDecision(
                result=f"{result}\n\n{warning}", decision_type="warn"
            )
        return ResultDecision(result=result)


def _feedback(
    result: ToolSafeResult, *, action: str, reason: str | None = None
) -> str:
    payload = {
        "risk_score": result.risk_score,
        "threshold": result.threshold,
        "reason": reason or result.reason,
        "action": action,
    }
    messages = {
        "replan": (
            "The proposed tool call was not executed. Reconsider the security "
            "feedback and choose a safer action that still follows the original "
            "user request."
        ),
        "block": (
            "The proposed tool call was not executed and this agent run was "
            "terminated by ToolSafe."
        ),
        "warn": (
            "The tool call was permitted by warn mode. Reassess the result and "
            "prefer a safer next action."
        ),
    }
    return (
        "Security Validation Before Execution:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n"
        + messages[action]
    )


def _user_request(state: dict[str, Any]) -> str:
    explicit = str(state.get("_toolsafe_user_request") or "").strip()
    if explicit:
        return explicit
    for message in state.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content and not content.startswith("TOOL_RESULT:"):
            return content
    return ""


def _interaction_history(
    state: dict[str, Any], current_tool: str, max_steps: int
) -> list[dict[str, Any]]:
    if max_steps == 0:
        return []
    history: list[dict[str, Any]] = []
    for message in state.get("messages", []) or []:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        history.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    if history:
        last = history[-1]
        if last["role"] == "assistant" and current_tool in last["content"]:
            history.pop()
    history.extend(state.get("_toolsafe_tool_history", []) or [])
    return history[-max_steps:]
