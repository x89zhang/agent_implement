"""AEGIS Gateway adapter for pre-execution checks and action traces.

Classification, policies, DSL rules, and anomaly detection run in the original
AEGIS Gateway. This module only translates the project's tool lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import AppConfig


RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass
class AegisDecision:
    allowed: bool = True
    reason: str = ""
    risk_level: str = "LOW"
    category: str = "unknown"
    signals: list[Any] = field(default_factory=list)
    mode: str = "off"
    policy: str = ""
    decision: str = "allow"
    gateway_decision: str = "allow"
    check_id: str = ""
    error: str = ""
    anomaly: dict[str, Any] | None = None
    dsl: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity(cfg: AppConfig, state: dict[str, Any]) -> tuple[str, str]:
    agent_id = cfg.aegis.agent_id or str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"agent-scaffold:{cfg.agent.name}")
    )
    uuid.UUID(agent_id)  # Required by the upstream action-trace schema.
    session_id = state.setdefault("_aegis_session_id", str(uuid.uuid4()))
    return agent_id, session_id


def _headers(cfg: AppConfig, agent_id: str, session_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "x-aegis-agent-id": agent_id,
        "x-aegis-session-id": session_id,
    }
    key = cfg.aegis.api_key or os.environ.get(cfg.aegis.api_key_env, "")
    if key:
        headers["x-api-key"] = key
    for env, header in (
        ("AEGIS_AGENT_SECRET", "x-aegis-agent-secret"),
        ("AEGIS_AGENT_TOKEN", "x-aegis-agent-token"),
    ):
        if os.environ.get(env):
            headers[header] = os.environ[env]
    return headers


def _request(
    cfg: AppConfig, state: dict[str, Any], method: str, path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent_id, session_id = _identity(cfg, state)
    data = json.dumps(payload, ensure_ascii=False, default=str).encode() if payload is not None else None
    request = urllib.request.Request(
        cfg.aegis.gateway_url.rstrip("/") + path,
        data=data,
        headers=_headers(cfg, agent_id, session_id),
        method=method,
    )
    with urllib.request.urlopen(request, timeout=cfg.aegis.timeout_seconds) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"AEGIS Gateway returned an invalid response: {result!r}")
    return result


def _resolve_pending(cfg: AppConfig, state: dict[str, Any], check_id: str) -> dict[str, Any]:
    if not check_id:
        raise ValueError("AEGIS returned pending without check_id")
    deadline = time.monotonic() + cfg.aegis.human_approval_timeout_seconds
    while time.monotonic() < deadline:
        result = _request(cfg, state, "GET", f"/api/v1/check/{check_id}/decision")
        if result.get("decision") in {"allow", "block"}:
            return result
        if result.get("decision") != "pending":
            raise ValueError(f"AEGIS returned unknown approval decision: {result!r}")
        time.sleep(min(cfg.aegis.poll_interval_seconds, max(0, deadline - time.monotonic())))
    return {"decision": "block", "reason": "AEGIS approval timed out"}


def check_tool_call(cfg: AppConfig, state: dict[str, Any], name: str, payload: Any) -> AegisDecision:
    if not cfg.aegis.enabled:
        return AegisDecision(mode="off")
    mode = cfg.aegis.mode.lower().strip()
    try:
        if not isinstance(payload, dict):
            raise TypeError("AEGIS tool arguments must be an object")
        # The upstream SDK skips gateway checks for explicitly allowed tools.
        if name.lower() in {item.lower() for item in cfg.aegis.allow_tools}:
            return AegisDecision(mode=mode, reason="AEGIS SDK allow_tools", policy="sdk_allow_tools")
        agent_id, _ = _identity(cfg, state)
        result = _request(cfg, state, "POST", "/api/v1/check", {
            "agent_id": agent_id,
            "tool_name": name,
            "arguments": payload,
            "environment": cfg.aegis.environment,
            "blocking": cfg.aegis.blocking and mode == "block",
        })
        gateway_decision = str(result.get("decision") or "")
        if gateway_decision not in {"allow", "block", "pending"}:
            raise ValueError(f"AEGIS Gateway returned unknown decision: {gateway_decision!r}")
        risk = str(result.get("risk_level") or "LOW").upper()
        if risk not in RISK_ORDER:
            raise ValueError(f"AEGIS Gateway returned unknown risk level: {risk!r}")
        check_id = str(result.get("check_id") or "")
        final = result
        if gateway_decision == "pending" and mode == "block":
            final = _resolve_pending(cfg, state, check_id)
        resolved = str(final.get("decision") or gateway_decision)
        if resolved not in {"allow", "block", "pending"}:
            raise ValueError(f"AEGIS returned unknown final decision: {resolved!r}")
        # Matches the upstream SDK's block_threshold; the Gateway verdict is
        # recorded separately and never replaced by this enforcement filter.
        threshold = cfg.aegis.risk_threshold.upper()
        if threshold not in RISK_ORDER:
            raise ValueError(f"Unknown AEGIS risk_threshold: {threshold!r}")
        above_threshold = RISK_ORDER[risk] >= RISK_ORDER[threshold]
        allowed = mode != "block" or resolved == "allow" or not above_threshold
        decision = AegisDecision(
            allowed=allowed,
            reason=str(final.get("reason") or result.get("reason") or ""),
            risk_level=risk,
            category=str(result.get("category") or "unknown"),
            signals=list(result.get("signals") or []),
            mode=mode,
            policy=str((result.get("dsl") or {}).get("rule") or ""),
            decision=resolved,
            gateway_decision=gateway_decision,
            check_id=check_id,
            anomaly=result.get("anomaly"),
            dsl=result.get("dsl"),
        )
        state.setdefault("_aegis_pending_traces", []).append({
            "name": name, "arguments": payload, "decision": decision.to_dict(),
            "started": time.time(),
        })
        return decision
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return AegisDecision(
            allowed=not cfg.aegis.fail_closed or mode != "block",
            reason=f"AEGIS Gateway check failed: {error}",
            risk_level="CRITICAL" if cfg.aegis.fail_closed else "LOW",
            mode=mode,
            decision="error",
            gateway_decision="error",
            error=error,
        )


def record_tool_result(
    cfg: AppConfig, state: dict[str, Any], name: str, payload: dict[str, Any],
    result: str, failed: bool,
) -> None:
    """Send an upstream-schema action trace after an executed tool call."""
    pending_list = state.get("_aegis_pending_traces") or []
    index = next((i for i, item in enumerate(pending_list)
                  if item["name"] == name and item["arguments"] == payload), None)
    if index is None:
        return
    pending = pending_list.pop(index)
    if not pending["decision"]["allowed"]:
        return
    agent_id, _ = _identity(cfg, state)
    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(uuid.uuid4())
    sequence = int(state.get("_aegis_trace_sequence", 0))
    previous = state.get("_aegis_previous_hash")
    trace: dict[str, Any] = {
        "trace_id": trace_id,
        "agent_id": agent_id,
        "timestamp": now,
        "sequence_number": sequence,
        "input_context": {"prompt": str(state.get("_aegis_user_request") or cfg.agent.task or "")},
        "thought_chain": {"raw_tokens": "Captured at tool boundary", "parsed_steps": []},
        "tool_call": {"tool_name": name, "function": name, "arguments": payload, "timestamp": now},
        "observation": {
            "raw_output": result,
            "error": str(result) if failed else None,
            "duration_ms": max((time.time() - pending["started"]) * 1000, 0.001),
        },
        "previous_hash": previous,
        "environment": cfg.aegis.environment,
        "safety_validation": {
            "policy_name": pending["decision"]["policy"] or "AEGIS Gateway",
            "passed": pending["decision"]["decision"] == "allow",
            "violations": [pending["decision"]["reason"]] if pending["decision"]["reason"] else [],
            "risk_level": pending["decision"]["risk_level"],
        },
    }
    hash_fields = {key: trace.get(key, "") for key in (
        "trace_id", "agent_id", "timestamp", "input_context", "thought_chain",
        "tool_call", "observation", "previous_hash",
    )}
    trace["integrity_hash"] = hashlib.sha256(
        json.dumps(hash_fields, sort_keys=True, default=str).encode()
    ).hexdigest()
    try:
        _request(cfg, state, "POST", "/api/v1/traces", trace)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        state.setdefault("aegis_trace_errors", []).append(error)
        state.setdefault("harness", {}).setdefault("aegis", {})["trace_errors"] = list(
            state["aegis_trace_errors"]
        )
        state.setdefault("trace", []).append({
            "step": "aegis_trace_error", "tool": name, "error": error,
        })
        return
    state["_aegis_trace_sequence"] = sequence + 1
    state["_aegis_previous_hash"] = trace["integrity_hash"]
