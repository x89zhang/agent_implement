"""JANUS/Vanguard anticipation and adjudication at action boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ToolDecision
from .client import VanguardClient


class JanusMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: VanguardClient | None = None) -> None:
        self.settings = cfg.janus
        api_key = self.settings.api_key or os.environ.get(self.settings.api_key_env, "")
        self.client = client or VanguardClient(self.settings, api_key=api_key)
        self.task = cfg.agent.task

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        candidate = _candidate(content, tool_call)
        event = self._check(state, candidate, "model_output")
        if tool_call is not None and not event["blocked"]:
            key = _action_key(tool_call[0], tool_call[1])
            state.setdefault("_janus_checked_actions", {})[key] = event
        if event["blocked"]:
            return ModelDecision(
                allowed=False,
                reason=event["reason"],
                content="Output blocked by JANUS safety guard.",
                tool_call=None,
                terminate=True,
                decision_type="janus",
            )
        return ModelDecision(content=content, tool_call=tool_call)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        cached = state.setdefault("_janus_checked_actions", {}).pop(
            _action_key(name, payload), None
        )
        event = cached or self._check(
            state, {"role": "assistant", "tool_calls": [{"name": name, "arguments": payload}]},
            "before_tool",
        )
        state["_last_janus_decision"] = event
        return ToolDecision(
            allowed=not event["blocked"],
            reason=event["reason"] if event["blocked"] else "",
            decision_type="janus" if event["blocked"] else "",
        )

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_janus_warning", "")
        return [f"JANUS safety warning: {warning}"] if warning else []

    def _check(
        self, state: dict[str, Any], candidate: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            observed = _observed(state, candidate, self.settings.max_history_steps)
            instruction = str(state.get("_janus_user_request") or self.task)
            verdict = self.client.judge(instruction, observed)
            label = verdict.label
            if label not in {"safe", "potential_unsafe", "unsafe"}:
                raise ValueError(f"Invalid JANUS label: {label}")
            flagged = label == "unsafe" or (
                self.settings.strict and label == "potential_unsafe"
            )
            blocked = flagged and self.settings.mode == "block"
            reason = f"JANUS classified candidate as {label}" if flagged else ""
            event = {
                "phase": phase, "label": label, "flagged": flagged,
                "blocked": blocked, "reason": reason, "error": "",
                "predicted_future": verdict.predicted_future,
                "reasoning": verdict.reasoning,
                "usage": verdict.usage,
            }
            stats = state.setdefault("trace_stats", {})
            stats["api_calls"] = int(stats.get("api_calls", 0)) + 2
            for key, value in verdict.usage.items():
                stats[key] = int(stats.get(key, 0)) + int(value)
        except Exception as exc:
            reason = f"JANUS guard failed: {type(exc).__name__}: {exc}"
            event = {
                "phase": phase, "label": None, "flagged": False,
                "blocked": self.settings.fail_closed,
                "reason": reason, "error": reason,
                "predicted_future": "", "reasoning": "", "usage": {},
            }
        event.update({
            "mode": self.settings.mode,
            "latency_ms": round((time.monotonic() - started) * 1000),
        })
        state["_last_janus_decision"] = event
        state.setdefault("janus_events", []).append(event)
        state.setdefault("trace", []).append({
            "step": "janus_judgment", "timestamp": time.time(), "output": event,
        })
        state.setdefault("harness", {})["janus"] = {
            "enabled": True, "mode": self.settings.mode,
            "status": "error" if event["error"] else "blocked" if event["blocked"] else "active",
            "event_count": len(state["janus_events"]), "last_decision": event,
        }
        if event["flagged"] and self.settings.mode == "warn":
            state["_janus_warning"] = reason
        return event


def _candidate(content: str, tool_call: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_call is not None:
        item["tool_calls"] = [{"name": tool_call[0], "arguments": tool_call[1]}]
    return item


def _action_key(name: str, payload: Any) -> str:
    encoded = json.dumps([name, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _observed(
    state: dict[str, Any], candidate: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    source = state.get("trace_messages") or state.get("messages") or []
    observed = [
        {key: message[key] for key in ("role", "content", "tool_calls", "extra") if key in message}
        for message in source
        if isinstance(message, dict)
    ]
    return observed[-max(1, limit - 1):] + [candidate]
