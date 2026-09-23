"""ClawSentry pre-action enforcement and post-action observation."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from ..config import AppConfig
from ..middleware import Middleware, ResultDecision, ToolDecision
from .client import ClawSentryClient


class ClawSentryMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: ClawSentryClient | None = None) -> None:
        self.settings = cfg.clawsentry
        self.client = client or ClawSentryClient(self.settings)
        self.agent_id = cfg.agent.name or "agent"
        self.task = cfg.agent.task

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_clawsentry_warning", "")
        return [f"ClawSentry safety warning: {warning}"] if warning else []

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        state["_clawsentry_tool_blocked"] = False
        event = self._check(state, "pre_action", name, {**payload, "arguments": payload})
        verdict = event["verdict"]
        if event["error"]:
            blocked = self.settings.mode == "block" and self.settings.fail_closed
            event["blocked"] = blocked
            state["_clawsentry_tool_blocked"] = blocked
            return ToolDecision(allowed=not blocked, reason=event["reason"] if blocked else "", decision_type="clawsentry" if blocked else "")
        flagged = verdict in {"block", "defer", "modify"}
        if flagged and self.settings.mode == "warn":
            state["_clawsentry_warning"] = event["reason"]
        if self.settings.mode != "block" or verdict == "allow":
            return ToolDecision()
        if verdict in {"block", "defer"}:
            event["blocked"] = True
            state["_clawsentry_tool_blocked"] = True
            return ToolDecision(False, event["reason"], decision_type="clawsentry")
        modified = event["modified_payload"]
        arguments = None
        if isinstance(modified, dict) and modified.get("tool_name", name) == name:
            nested = modified.get("tool_input", modified.get("arguments"))
            if isinstance(nested, dict):
                arguments = dict(nested)
            # AHP command rewrites may use the canonical top-level command
            # rather than rewriting our adapter-specific nested arguments.
            command = modified.get("command")
            if isinstance(command, str):
                for key in ("command", "cmd", "input"):
                    if key in payload:
                        arguments = dict(arguments if arguments is not None else payload)
                        arguments[key] = command
                        break
        if not isinstance(arguments, dict):
            event["blocked"] = True
            state["_clawsentry_tool_blocked"] = True
            event["reason"] = "ClawSentry modification could not be safely mapped to tool arguments"
            state["harness"]["clawsentry"]["status"] = "blocked"
            return ToolDecision(False, event["reason"], decision_type="clawsentry")
        event["applied_modification"] = True
        return ToolDecision(arguments=arguments, decision_type="clawsentry")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        blocked = state.pop("_clawsentry_tool_blocked", False)
        not_executed = str(result).startswith((
            "Tool execution blocked by middleware:", "Tool not found:"
        ))
        if self.settings.observe_tool_result and not blocked and not not_executed:
            self._check(state, "post_action", name, {
                "arguments": payload,
                "result": str(result)[:self.settings.max_result_chars],
                "output": str(result)[:self.settings.max_result_chars],
                "failed": failed,
            })
        # post_action is observation-only in AHP; the action has already happened.
        return ResultDecision(result=result)

    def _check(self, state: dict[str, Any], phase: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        session_id = state.setdefault("_clawsentry_session_id", f"session-{uuid4()}")
        event: dict[str, Any] = {
            "phase": phase, "tool": name, "verdict": None, "reason": "",
            "policy_id": "", "risk_level": "", "error": "", "blocked": False,
            "modified_payload": None, "applied_modification": False,
        }
        try:
            decision = self.client.decide(
                event_type=phase, session_id=session_id, agent_id=self.agent_id,
                tool_name=name, payload=payload,
                current_task=str(state.get("_clawsentry_user_request") or self.task),
            )
            event.update({
                "verdict": decision["decision"],
                "reason": str(decision.get("reason") or f"ClawSentry {decision['decision']} decision"),
                "policy_id": str(decision.get("policy_id") or ""),
                "risk_level": str(decision.get("risk_level") or ""),
                "modified_payload": decision.get("modified_payload"),
            })
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            event["reason"] = f"ClawSentry gateway failed: {event['error']}"
        event["blocked"] = (
            phase == "pre_action" and self.settings.mode == "block" and
            (event["verdict"] in {"block", "defer"} or (bool(event["error"]) and self.settings.fail_closed))
        )
        event["latency_ms"] = round((time.monotonic() - started) * 1000)
        event["mode"] = self.settings.mode
        state["_last_clawsentry_decision"] = event
        state.setdefault("clawsentry_events", []).append(event)
        state.setdefault("trace", []).append({"step": "clawsentry_decision", "timestamp": time.time(), "output": event})
        state.setdefault("harness", {})["clawsentry"] = {
            "enabled": True, "mode": self.settings.mode,
            "status": "error" if event["error"] else "blocked" if event["blocked"] else "active",
            "event_count": len(state["clawsentry_events"]), "last_decision": event,
        }
        return event
