"""JANUS/Vanguard anticipation and adjudication before each assistant step.

Upstream's Vanguard runtime (``eval_framework_offline/.../vllm_guard.py``)
judges a trajectory prefix ending in the step under review. Here the prefix is
the full conversation the agent sent to the model plus the proposed assistant
step with all of its tool calls, judged once; the verdict applies to every
call of the turn.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision
from ..tool_results import unwrap_hermes_result
from .client import VanguardClient


class JanusMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: VanguardClient | None = None) -> None:
        self.settings = cfg.janus
        api_key = self.settings.api_key or os.environ.get(self.settings.api_key_env, "")
        self.client = client or VanguardClient(self.settings, api_key=api_key)

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        index = int(state.get("_model_output_index", 0) or 0)
        if index == 0 or "_janus_turn_event" not in state:
            calls = state.get("_model_output_calls")
            if calls is None:
                # Native project graph: one candidate per model output.
                calls = [] if tool_call is None else [
                    {"id": "", "name": tool_call[0], "arguments": tool_call[1]}
                ]
            content = str(state.get("_model_output_content", content) or "")
            state["_janus_turn_event"] = self._check(state, trajectory(state, content, calls), len(calls))
        # One judgment per assistant step, recorded for each of its calls.
        event = dict(state["_janus_turn_event"], call_index=index)
        state["_last_janus_decision"] = event
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

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_janus_warning", "")
        return [f"JANUS safety warning: {warning}"] if warning else []

    def _check(self, state: dict[str, Any], history: list[dict[str, Any]], call_count: int) -> dict[str, Any]:
        started = time.monotonic()
        try:
            verdict = self.client.judge(history)
            label = verdict.label
            if verdict.parse_error:
                raise ValueError(verdict.parse_error)
            if label not in {"safe", "potential_unsafe", "unsafe"}:
                raise ValueError(f"Invalid JANUS label: {label}")
            # Upstream lenient mode blocks unsafe only; strict also blocks
            # potential_unsafe (cli.py --mode).
            flagged = label == "unsafe" or (
                self.settings.strict and label == "potential_unsafe"
            )
            reason = f"JANUS classified candidate as {label}" if flagged else ""
            event = {
                "phase": "model_output", "label": label, "flagged": flagged,
                "blocked": flagged and self.settings.mode == "block",
                "reason": reason, "error": "",
                "predicted_future": verdict.predicted_future,
                "reasoning": verdict.reasoning,
                "usage": verdict.usage,
            }
            stats = state.setdefault("trace_stats", {})
            stats["api_calls"] = int(stats.get("api_calls", 0)) + 2
            for key, value in verdict.usage.items():
                stats[key] = int(stats.get(key, 0)) + int(value)
        except Exception as exc:
            # Upstream fails open (request_error / parse_error verdicts are
            # not flagged); fail_closed is an explicit project option.
            reason = f"JANUS guard failed: {type(exc).__name__}: {exc}"
            event = {
                "phase": "model_output", "label": None, "flagged": False,
                "blocked": self.settings.fail_closed,
                "reason": reason, "error": reason,
                "predicted_future": "", "reasoning": "", "usage": {},
            }
        event.update({
            "step_index": history[-1]["index"] if history else None,
            "call_count": call_count,
            "mode": self.settings.mode,
            "latency_ms": round((time.monotonic() - started) * 1000),
        })
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


def _content(value: Any) -> str:
    # agentdojo loader: non-string content is JSON-encoded.
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except ValueError:
            return {"_raw": value}
    return dict(value) if isinstance(value, dict) else {}


def _tool_calls(calls: list[Any]) -> list[dict[str, Any]] | None:
    normalized = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        normalized.append({
            "name": str(function.get("name", "")),
            "arguments": _arguments(function.get("arguments")),
            "call_id": call.get("id"),
        })
    return normalized or None


def trajectory(state: dict[str, Any], content: str, calls: list[Any]) -> list[dict[str, Any]]:
    """Upstream ``Step`` sequence (agentdojo loader ``_parse_messages``) for the
    conversation so far plus the proposed assistant step."""
    source = state.get("trace_messages") or state.get("messages") or []
    steps: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    # Absolute message positions, as the loader's enumerate(raw_messages).
    for index, message in enumerate(source):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = _tool_calls(message.get("tool_calls") or [])
            for call in tool_calls or []:
                if call["call_id"]:
                    names[str(call["call_id"])] = call["name"]
            steps.append({"index": index, "role": "assistant",
                          "content": _content(message.get("content")), "tool_calls": tool_calls})
        elif role == "tool":
            call_id = message.get("tool_call_id")
            steps.append({
                "index": index, "role": "tool",
                "content": unwrap_hermes_result(_content(message.get("content"))),
                "tool_name": message.get("name") or names.get(str(call_id or "")),
                "tool_call_id": call_id,
            })
        elif role in ("system", "user"):
            steps.append({"index": index, "role": role, "content": _content(message.get("content"))})
    steps.append({
        "index": len(source), "role": "assistant", "content": content,
        "tool_calls": _tool_calls(calls),
    })
    return steps
