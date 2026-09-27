"""ClawSentry pre-action enforcement and post-action observation."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from ..config import AppConfig
from ..middleware import Middleware, ResultDecision, ToolDecision
from ..tool_results import unwrap_hermes_result
from .client import ClawSentryClient, GatewayUnavailable
from .upstream import fallback_decision

# Post-action tiers upstream turns into an indirect-injection contamination
# alert (supervision_gateway.py:677-711).
_FINDING_TIERS = {"escalate", "emergency"}


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
        if verdict is None:
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
            output = unwrap_hermes_result(str(result))[:self.settings.max_result_chars]
            event = self._check(state, "post_action", name, {
                "arguments": payload,
                "result": output,
                "output": output,
                "failed": failed,
            })
            if output and not event["error"]:
                self._record_post_action_finding(state, event)
        # post_action is observation-only in AHP; the action has already happened.
        return ResultDecision(result=result)

    def _check(self, state: dict[str, Any], phase: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        session_id = state.setdefault("_clawsentry_session_id", f"session-{uuid4()}")
        event_id = f"evt-{uuid4()}"
        event: dict[str, Any] = {
            "phase": phase, "tool": name, "event_id": event_id, "verdict": None, "reason": "",
            "policy_id": "", "risk_level": "", "error": "", "blocked": False,
            "modified_payload": None, "applied_modification": False,
        }
        decision = None
        try:
            decision = self.client.decide(
                event_type=phase, session_id=session_id, agent_id=self.agent_id,
                tool_name=name, payload=payload, event_id=event_id,
                # The prompt the agent actually received.
                current_task=str(
                    state.get("_runtime_user_request")
                    or state.get("_clawsentry_user_request") or self.task
                ),
            )
            event.update(getattr(self.client, "last_response_metadata", None) or {})
        except GatewayUnavailable as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            event["reason"] = f"ClawSentry gateway failed: {event['error']}"
            # Upstream adapters fall back to make_fallback_decision when the
            # gateway is unreachable (a3s_adapter.py:420-437): pre_action
            # DEFER, or BLOCK for high-danger calls; observations allow.
            try:
                decision = fallback_decision(exc.event)
                event["gateway_transport"] = "fallback_local"
            except Exception as fallback_exc:
                event["fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            event["reason"] = f"ClawSentry gateway failed: {event['error']}"
        if decision is not None:
            event.update({
                "verdict": decision["decision"],
                "reason": str(decision.get("reason") or f"ClawSentry {decision['decision']} decision"),
                "policy_id": str(decision.get("policy_id") or ""),
                "risk_level": str(decision.get("risk_level") or ""),
                "modified_payload": decision.get("modified_payload"),
            })
        event["blocked"] = (
            phase == "pre_action" and self.settings.mode == "block" and
            (event["verdict"] in {"block", "defer"} or (event["verdict"] is None and self.settings.fail_closed))
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

    def _record_post_action_finding(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        """Attach the gateway's background post-action finding to ``event``.

        Upstream analyzes tool output asynchronously and reports findings only
        through session state and alerts (sync_decision_flow.py:1327-1370);
        the synchronous post_action verdict is always allow. Polling until the
        finding for this event exists also keeps a replayed next pre_action
        from racing the contamination update (bounded by
        ``post_action_wait_seconds``).
        """
        started = time.monotonic()
        deadline = started + self.settings.post_action_wait_seconds
        finding = None
        error = ""
        while True:
            try:
                finding = next(
                    (item for item in self.client.post_action_scores(state["_clawsentry_session_id"])
                     if item.get("event_id") == event["event_id"]),
                    None,
                )
                error = ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if finding is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        tier = str((finding or {}).get("tier") or "")
        event["post_action_finding"] = finding
        event["post_action_finding_status"] = (
            "recorded" if finding is not None else f"error: {error}" if error else "timeout"
        )
        # A separate detection signal: the verdict stays the gateway's allow.
        event["post_action_flagged"] = tier in _FINDING_TIERS
        event["post_action_wait_ms"] = round((time.monotonic() - started) * 1000)
        if event["post_action_flagged"]:
            event["reason"] = (
                f"ClawSentry post-action finding: tier={tier}, "
                f"score={finding.get('score')}, patterns={finding.get('patterns_matched')}"
            )
        state.setdefault("clawsentry_post_action_findings", []).append(
            {"event_id": event["event_id"], "tool": event["tool"], "finding": finding}
        )
