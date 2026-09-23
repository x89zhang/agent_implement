"""StepGuard checks proposed tools and optionally audits final trajectories."""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ToolDecision
from .client import StepGuardClient
from .prompt import action_text, build_prompt


_REPLAN_FEEDBACK = (
    "The previous tool call was not executed because it failed a safety check. "
    "Return to the original user request and trusted execution history. "
    "Choose a safer, narrowly scoped next action."
)


class StepGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.stepguard
        key = self.settings.api_key or os.environ.get(self.settings.api_key_env, "")
        self.client = client or StepGuardClient(self.settings, api_key=key)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        event = self._judge(state, "pre_action", action_text(name, payload))
        state["_last_stepguard_decision"] = event
        if not event["blocked"]:
            return ToolDecision()
        count = int(state.get("_stepguard_block_count", 0)) + 1
        state["_stepguard_block_count"] = count
        exhausted = count > self.settings.max_replans
        return ToolDecision(
            allowed=False,
            reason=event["reason"],
            replacement_result=(
                "StepGuard replan budget exhausted."
                if exhausted else _REPLAN_FEEDBACK
            ),
            decision_type="stepguard_block",
            terminate=exhausted,
        )

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        if tool_call is not None or not self.settings.audit_final:
            return ModelDecision(content=content, tool_call=tool_call)
        event = self._judge(state, "final_trajectory", f"[ACTION]: {content}")
        if event["blocked"]:
            return ModelDecision(
                allowed=False,
                reason=event["reason"],
                content=self.settings.replacement_message,
                tool_call=None,
                decision_type="stepguard_final_block",
                terminate=True,
            )
        return ModelDecision(content=content, tool_call=None)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_stepguard_warning", "")
        return [warning] if warning else []

    def _judge(
        self, state: dict[str, Any], checkpoint: str, action: str
    ) -> dict[str, Any]:
        started = time.monotonic()
        task = str(state.get("_stepguard_user_request") or self.cfg.agent.task)
        tools = state.get("_stepguard_tools") or [
            {"name": tool.name, "description": tool.description}
            for tool in self.cfg.tools
        ]
        history = state.get("trace_messages") or state.get("messages") or []
        prompt = build_prompt(
            task=task,
            history=history,
            tools=tools,
            action=action,
            trajectory=checkpoint == "final_trajectory",
            max_history_chars=self.settings.max_history_chars,
        )
        try:
            verdict = self.client.judge(prompt)
            if verdict.label not in {"safe", "unsafe"}:
                raise ValueError(f"Invalid StepGuard label: {verdict.label}")
            flagged = (
                verdict.label == "unsafe"
                and verdict.confidence >= self.settings.confidence_threshold
            )
            error = ""
            reason = "StepGuard classified the action as unsafe" if flagged else ""
            usage = verdict.usage
            stats = state.setdefault("trace_stats", {})
            stats["api_calls"] = int(stats.get("api_calls", 0)) + 1
            for key, value in usage.items():
                stats[key] = int(stats.get(key, 0)) + int(value)
        except Exception as exc:
            verdict = None
            flagged = self.settings.fail_closed
            error = f"StepGuard failed: {type(exc).__name__}: {exc}"
            reason = error
            usage = {}
        blocked = flagged and self.settings.mode == "block"
        event = {
            "checkpoint": checkpoint,
            "label": verdict.label if verdict else None,
            "confidence": verdict.confidence if verdict else None,
            "risk_source": verdict.risk_source if verdict else "",
            "unsafe_step": verdict.unsafe_step if verdict else None,
            "flagged": flagged,
            "blocked": blocked,
            "reason": reason,
            "error": error,
            "usage": usage,
            "mode": self.settings.mode,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        state["_last_stepguard_decision"] = event
        events = state.setdefault("stepguard_events", [])
        events.append(event)
        state.setdefault("trace", []).append({
            "step": "stepguard_judgment",
            "timestamp": time.time(),
            "output": event,
        })
        state.setdefault("harness", {})["stepguard"] = {
            "enabled": True,
            "mode": self.settings.mode,
            "status": "error" if error else "blocked" if blocked else "active",
            "event_count": len(events),
            "last_decision": event,
        }
        if flagged and self.settings.mode == "warn":
            state["_stepguard_warning"] = "StepGuard safety warning: " + reason
        return event
