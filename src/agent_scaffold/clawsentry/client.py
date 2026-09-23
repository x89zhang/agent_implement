"""Small HTTP client for ClawSentry's AHP SyncDecision gateway."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from ..config import ClawSentryConfig


class ClawSentryClient:
    def __init__(self, settings: ClawSentryConfig) -> None:
        self.settings = settings

    def decide(
        self,
        *,
        event_type: str,
        session_id: str,
        agent_id: str,
        tool_name: str,
        payload: dict[str, Any],
        current_task: str,
    ) -> dict[str, Any]:
        request_id = f"req-{uuid4()}"
        event = {
            "schema_version": "ahp.1.0",
            "event_id": f"evt-{uuid4()}",
            "trace_id": session_id,
            "event_type": event_type,
            "session_id": session_id,
            "agent_id": agent_id,
            "source_framework": "agent-scaffold",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "payload": payload,
        }
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "ahp/sync_decision",
            "params": {
                "rpc_version": "sync_decision.1.0",
                "request_id": request_id,
                "deadline_ms": min(900000, max(1, int(self.settings.timeout_seconds * 1000))),
                "decision_tier": self.settings.decision_tier,
                "event": event,
                "context": {
                    "caller_adapter": "agent-scaffold.clawsentry.v1",
                    "current_task": current_task,
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.settings.api_key_env, "") if self.settings.api_key_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        managed_url = (
            os.environ.get("AGENT_CLAWSENTRY_URL", "")
            if self.settings.auto_start and os.environ.get("AGENT_CONTAINERIZED") == "1"
            else ""
        )
        request = Request(
            (managed_url or self.settings.base_url).rstrip("/") + "/ahp",
            data=json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            answer = json.load(response)
        if not isinstance(answer, dict) or answer.get("jsonrpc") != "2.0" or answer.get("id") != request_id:
            raise ValueError("invalid ClawSentry JSON-RPC envelope")
        if "error" in answer:
            error = answer["error"]
            raise ValueError(f"ClawSentry RPC error: {error.get('code') if isinstance(error, dict) else 'unknown'}")
        result = answer.get("result")
        if not isinstance(result, dict) or result.get("request_id") != request_id or result.get("rpc_status") != "ok":
            raise ValueError("invalid ClawSentry SyncDecision response")
        decision = result.get("decision")
        if not isinstance(decision, dict) or decision.get("decision") not in {"allow", "block", "modify", "defer"}:
            raise ValueError("invalid ClawSentry verdict")
        return decision
