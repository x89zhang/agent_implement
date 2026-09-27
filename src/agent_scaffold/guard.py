"""AEGIS adapter: Gateway L1 plus the paper's L2/L3 cascade.

Classification, policies, DSL rules, and anomaly detection run in the original
AEGIS Gateway (L1). When ``aegis.cascade`` is on, calls the Gateway allows go
through the paper's L2 XGBoost classifier and, in its ambiguous band, the L3
LLM judge (``aegis/cascade.py``, from ``research/cascade/pipeline.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, LLMConfig
from .llm import LLMAdapter


RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
# Upstream's research client retries 429 (aegis_rules_http.py:61-67); gateway
# restarts and proxies also surface as these transient statuses.
_TRANSIENT_HTTP = {429, 502, 503, 504}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_L2_MODELS: dict[str, Any] = {}


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
    # Cascade verdict (L1 -> L2 -> L3) and the layer that produced it.
    cascade_decision: str = ""
    layer_fired: str = ""
    l2_score: float | None = None
    l2_thresholds: dict[str, float] | None = None
    l3: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity(cfg: AppConfig, state: dict[str, Any]) -> tuple[str, str]:
    # One Gateway agent per run: the Gateway's behavioral profile
    # (check.ts:162-230) would otherwise carry state across runs and
    # conditions. A fixed aegis.agent_id restores a shared profile.
    agent_id = cfg.aegis.agent_id or state.get("_aegis_agent_id")
    if not agent_id:
        run_dir = str((state.get("_trace_persist") or {}).get("run_dir") or "")
        seed = f"agent-scaffold:{cfg.agent.name}:{run_dir}" if run_dir else str(uuid.uuid4())
        agent_id = state["_aegis_agent_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
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
    attempts = max(1, int(getattr(cfg.aegis, "max_retries", 5)))
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=cfg.aegis.timeout_seconds) as response:
                result = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in _TRANSIENT_HTTP or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(0.1 * (2 ** attempt))
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


def _l2_model(cfg: AppConfig) -> Any:
    from .aegis import cascade as aegis_cascade

    path = Path(cfg.aegis.l2_model_path)
    path = path if path.is_absolute() else _REPO_ROOT / path
    key = str(path)
    if key not in _L2_MODELS:
        _L2_MODELS[key] = aegis_cascade.L2Model.load(path)
    return _L2_MODELS[key]


def _judge_llm_config(cfg: AppConfig) -> LLMConfig:
    settings, base = cfg.aegis, cfg.llm
    return LLMConfig(
        provider=settings.judge_provider or base.provider,
        model=settings.judge_model or base.model,
        # llm_judge.py calls the judge with temperature=0.0.
        temperature=0.0,
        base_url=settings.judge_base_url or base.base_url,
        api_key=(os.environ.get(settings.judge_api_key_env, "")
                 if settings.judge_api_key_env else base.api_key),
        api_key_env=settings.judge_api_key_env or base.api_key_env,
        request_timeout=getattr(base, "request_timeout", 120),
    )


def _judge(cfg: AppConfig, state: dict[str, Any], name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """L3: research/baselines/llm_judge.py on the inherited model."""
    from .aegis import cascade as aegis_cascade

    llm_cfg = _judge_llm_config(cfg)
    user_query = str(state.get("_runtime_user_request") or state.get("_aegis_user_request")
                     or cfg.agent.task or "")
    response = LLMAdapter(llm_cfg).chat([
        {"role": "system", "content": aegis_cascade.SYSTEM_PROMPT},
        {"role": "user", "content": aegis_cascade.build_prompt(
            name, payload, cfg.aegis.framework, user_query)},
    ])
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    decision, risk, rationale, parse_error = aegis_cascade.parse_decision(str(content or ""))
    return {
        "decision": decision, "risk_score": risk, "rationale": rationale,
        "parse_error": parse_error, "model": llm_cfg.model, "usage": response.usage,
    }


def _run_cascade(
    cfg: AppConfig, state: dict[str, Any], name: str, payload: dict[str, Any],
    gateway_decision: str,
) -> dict[str, Any]:
    """research/cascade/pipeline.py:CascadePipeline.predict after the Gateway."""
    from .aegis import cascade as aegis_cascade

    if gateway_decision in {"block", "pending"}:
        return aegis_cascade.aggregate(gateway_decision, None, 1.0, 0.0)
    model = _l2_model(cfg)
    score = model.score(payload)
    judge = (lambda: _judge(cfg, state, name, payload)) if cfg.aegis.use_l3 else None
    result = aegis_cascade.aggregate(
        gateway_decision, score, model.tau_high, model.tau_low,
        judge=judge, disable_l2_block=cfg.aegis.disable_l2_block,
    )
    result["l2_score"] = score
    result["l2_thresholds"] = {"tau_high": model.tau_high, "tau_low": model.tau_low}
    return result


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
        reason = str(final.get("reason") or result.get("reason") or "")
        cascade: dict[str, Any] = {}
        error = ""
        if cfg.aegis.cascade:
            try:
                cascade = _run_cascade(cfg, state, name, payload, gateway_decision)
            except Exception as exc:
                error = f"AEGIS cascade failed: {type(exc).__name__}: {exc}"
                cascade = {"decision": "error", "layer_fired": "error"}
        layer = cascade.get("layer_fired", "L1")
        if error:
            resolved, reason = "error", error
            allowed = mode != "block" or not cfg.aegis.fail_closed
        elif layer in {"L2", "L3"}:
            # L2/L3 verdicts are the research pipeline's; the SDK threshold
            # only filters Gateway risk levels.
            resolved = cascade["decision"]
            allowed = mode != "block" or resolved == "allow"
            l3 = cascade.get("l3") or {}
            reason = (f"AEGIS L3 judge: {l3.get('rationale', '')}" if layer == "L3"
                      else f"AEGIS L2 P(malicious)={cascade['l2_score']:.4f}")
        decision = AegisDecision(
            allowed=allowed,
            reason=reason,
            risk_level=risk,
            category=str(result.get("category") or "unknown"),
            signals=list(result.get("signals") or []),
            mode=mode,
            policy=str((result.get("dsl") or {}).get("rule") or ""),
            decision=resolved,
            gateway_decision=gateway_decision,
            check_id=check_id,
            error=error,
            anomaly=result.get("anomaly"),
            dsl=result.get("dsl"),
            cascade_decision=str(cascade.get("decision") or gateway_decision),
            layer_fired=layer,
            l2_score=cascade.get("l2_score"),
            l2_thresholds=cascade.get("l2_thresholds"),
            l3=cascade.get("l3"),
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
            cascade_decision="error",
            layer_fired="error",
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
