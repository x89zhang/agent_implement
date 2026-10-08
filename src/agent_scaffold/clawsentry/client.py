"""Small HTTP client for ClawSentry's AHP SyncDecision gateway."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from ..config import ClawSentryConfig
from .upstream import enrich_event

_VERDICTS = {"allow", "block", "modify", "defer"}
# Upstream A3SCodeAdapter defaults (a3s_adapter.py:180-190).
_MAX_RPC_RETRIES = 1
_RETRY_BACKOFF_MS = 50


class GatewayUnavailable(Exception):
    """The gateway could not be reached or answered with a transport error."""

    def __init__(self, message: str, event: dict[str, Any]) -> None:
        super().__init__(message)
        self.event = event


class ClawSentryClient:
    def __init__(self, settings: ClawSentryConfig) -> None:
        self.settings = settings
        self.last_response_metadata: dict[str, Any] = {}

    def _url(self) -> str:
        managed_url = (
            os.environ.get("AGENT_CLAWSENTRY_URL", "")
            if self.settings.auto_start and os.environ.get("AGENT_CONTAINERIZED") == "1"
            else ""
        )
        return (managed_url or self.settings.base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.settings.api_key_env, "") if self.settings.api_key_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def decide(
        self,
        *,
        event_type: str,
        session_id: str,
        agent_id: str,
        tool_name: str,
        payload: dict[str, Any],
        current_task: str,
        event_id: str = "",
    ) -> dict[str, Any]:
        request_id = f"req-{uuid4()}"
        # Upstream adapters attach risk hints and the content origin to every
        # event (a3s_adapter.py:303-311).
        event = enrich_event({
            "schema_version": "ahp.1.0",
            "event_id": event_id or f"evt-{uuid4()}",
            "trace_id": session_id,
            "event_type": event_type,
            "session_id": session_id,
            "agent_id": agent_id,
            "source_framework": "agent-scaffold",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "payload": payload,
        })
        deadline_ms = min(900000, max(1, int(self.settings.timeout_seconds * 1000)))
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "ahp/sync_decision",
            "params": {
                "rpc_version": "sync_decision.1.0",
                "request_id": request_id,
                "deadline_ms": deadline_ms,
                "decision_tier": self.settings.decision_tier,
                "event": event,
                "context": {
                    "caller_adapter": "agent-scaffold.clawsentry.v1",
                    "current_task": current_task,
                },
            },
        }
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        # Retry loop and fallback order of upstream A3SCodeAdapter.request_decision
        # (a3s_adapter.py:363-437): one retry for transport failures and
        # retry-eligible RPC errors while the deadline budget allows.
        started = time.monotonic()
        last_error = "gateway unreachable"
        answer = None
        for attempt in range(1 + _MAX_RPC_RETRIES):
            remaining_ms = deadline_ms - (time.monotonic() - started) * 1000
            if attempt > 0:
                if remaining_ms < _RETRY_BACKOFF_MS + 20:
                    break
                time.sleep(_RETRY_BACKOFF_MS / 1000.0)
            request = Request(self._url() + "/ahp", data=data, headers=self._headers(), method="POST")
            try:
                # +0.5 s so the gateway can still send DEADLINE_EXCEEDED
                # (upstream a3s_adapter.py:458-466).
                with urlopen(request, timeout=deadline_ms / 1000.0 + 0.5) as response:
                    answer = json.load(response)
            except HTTPError as exc:
                try:
                    answer = json.load(exc)
                except Exception:
                    last_error = f"HTTPError: {exc.code}"
                    answer = None
                    continue
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                answer = None
                continue
            if not isinstance(answer, dict) or answer.get("jsonrpc") != "2.0" or answer.get("id") != request_id:
                last_error = "invalid ClawSentry JSON-RPC envelope"
                answer = None
                continue
            if "error" not in answer:
                break
            error = answer["error"] if isinstance(answer["error"], dict) else {}
            error_data = error.get("data") if isinstance(error.get("data"), dict) else {}
            last_error = f"ClawSentry RPC error: {error_data.get('rpc_error_code') or error.get('code') or 'unknown'}"
            if error_data.get("retry_eligible") and attempt < _MAX_RPC_RETRIES:
                answer = None
                continue
            # The gateway's own decision travels in the error data
            # (sync_decision_flow.py:2003-2012; a3s_adapter.py:400-406).
            fallback = error_data.get("fallback_decision")
            if isinstance(fallback, dict) and fallback.get("decision") in _VERDICTS:
                self.last_response_metadata = {"gateway_transport": "rpc_fallback", "gateway_error": last_error}
                return fallback
            answer = None
            break
        if answer is None:
            raise GatewayUnavailable(last_error, event)
        result = answer.get("result")
        if not isinstance(result, dict) or result.get("request_id") != request_id or result.get("rpc_status") != "ok":
            raise GatewayUnavailable("invalid ClawSentry SyncDecision response", event)
        decision = result.get("decision")
        if not isinstance(decision, dict) or decision.get("decision") not in _VERDICTS:
            raise GatewayUnavailable("invalid ClawSentry verdict", event)
        self.last_response_metadata = {
            "gateway_transport": "http",
            **{key: result[key] for key in ("actual_tier", "l3_state") if result.get(key) is not None},
        }
        return decision

    def session_records(self, session_id: str) -> list[dict[str, Any]]:
        """Stored decision records of a session (GET /report/session/{id}).

        Each record carries the gateway's risk snapshot, whose
        ``l2_l3_summary`` holds the L2 outcome that SyncDecision omits.
        """
        request = Request(
            f"{self._url()}/report/session/{quote(session_id, safe='')}?limit=1000",
            headers=self._headers(), method="GET",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            answer = json.load(response)
        records = answer.get("records") if isinstance(answer, dict) else None
        return [item for item in records or [] if isinstance(item, dict)]

    def post_action_scores(self, session_id: str) -> list[dict[str, Any]]:
        """Background post-action findings (GET /report/session/{id}/post-action)."""
        request = Request(
            f"{self._url()}/report/session/{quote(session_id, safe='')}/post-action?limit=1000",
            headers=self._headers(), method="GET",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            answer = json.load(response)
        scores = answer.get("post_action_scores") if isinstance(answer, dict) else None
        return [item for item in scores or [] if isinstance(item, dict)]
