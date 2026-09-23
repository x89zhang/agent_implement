"""Bridge ADR session detection into the project guard lifecycle."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ResultDecision


class ADRMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.settings = cfg.adr

    def after_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any],
        result: str, failed: bool,
    ) -> ResultDecision:
        started = time.monotonic()
        try:
            detection = self._detect(state, name, payload, result, failed)
            malicious = detection["is_malicious"]
            if not isinstance(malicious, bool):
                raise ValueError("ADR result is missing a boolean is_malicious")
            confidence = float(detection["confidence_score"])
            flagged = malicious and confidence >= self.settings.threshold
            reason = _reason(detection) if flagged else ""
            error = ""
        except Exception as exc:
            detection = {}
            confidence = None
            flagged = False
            error = f"ADR detector failed: {type(exc).__name__}: {exc}"
            reason = error

        quarantined = (flagged and self.settings.mode == "block") or (
            bool(error) and self.settings.fail_closed
        )
        event = {
            "phase": "after_tool",
            "tool": name,
            "flagged": flagged,
            "blocked": quarantined,
            "quarantined": quarantined,
            "tool_executed": True,
            "confidence": confidence,
            "reason": reason,
            "error": error,
            "mode": self.settings.mode,
            "method": detection.get("method"),
            "model_used": detection.get("model_used"),
            "detections": detection.get("detections", []),
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
        state["_last_adr_decision"] = event
        state.setdefault("adr_events", []).append(event)
        state.setdefault("harness", {})["adr"] = {
            "enabled": True,
            "mode": self.settings.mode,
            "status": "error" if error else "flagged" if flagged else "clean",
            "event_count": len(state["adr_events"]),
            "last_decision": event,
        }
        state.setdefault("trace", []).append(
            {"step": "adr_detection", "output": event, "timestamp": time.time()}
        )
        if flagged and self.settings.mode == "warn":
            state["_adr_warning"] = reason
        return ResultDecision(
            allowed=not quarantined,
            reason=reason if quarantined else "",
            result=(
                "Tool result quarantined by ADR detection. " + reason
                if quarantined else result
            ),
            decision_type="adr_error" if error else "adr_detection",
        )

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_adr_warning", "")
        return [f"ADR detection warning: {warning}"] if warning else []

    def _detect(
        self, state: dict[str, Any], name: str, payload: dict[str, Any],
        result: str, failed: bool,
    ) -> dict[str, Any]:
        settings = self.settings
        detection_root = Path(settings.detection_root).expanduser().resolve()
        if not (detection_root / "guardrail" / "adr_agent" / "adr_baseline.py").is_file():
            raise ValueError("adr.detection_root must point to Uber ADR/Detection")
        interpreter = Path(settings.python_executable).expanduser()
        if not interpreter.is_file():
            raise ValueError(f"ADR Python interpreter not found: {interpreter}")
        messages = [
            {"role": str(message.get("role", "user")), "content": str(message.get("content") or "")}
            for message in state.get("messages", [])
            if isinstance(message, dict) and message.get("role") in {"user", "assistant", "tool", "system"}
        ]
        if not messages:
            messages = [{"role": "user", "content": str(state.get("_adr_user_request") or "")}]
        messages.append({
            "role": "assistant",
            "content": "Tool call: " + json.dumps(
                {"name": name, "arguments": payload}, ensure_ascii=False, default=str
            ),
        })
        messages.append({
            "role": "tool",
            "content": json.dumps(
                {"name": name, "failed": failed, "result": result},
                ensure_ascii=False, default=str,
            ),
        })
        request = {
            "detection_root": str(detection_root),
            "benchmark_type": settings.benchmark_type,
            "config": {
                "adr_framework": {
                    "enable_triage": settings.enable_triage,
                    "triage_llm": {"model": settings.triage_model},
                    "reasoning_agent": {
                        "model": settings.reasoning_model,
                        "timeout": settings.reasoning_timeout_seconds,
                        "max_turns": settings.max_turns,
                        "enable_threat_intelligence": settings.enable_threat_intelligence,
                        "enable_source_code": settings.enable_source_code,
                        "enable_policy": settings.enable_policy,
                    },
                }
            },
            "task": {
                "task_id": "project_" + uuid.uuid4().hex,
                "messages": messages,
            },
        }
        completed = subprocess.run(
            [str(interpreter), str(Path(__file__).with_name("worker.py"))],
            input=json.dumps(request, ensure_ascii=False, default=str),
            text=True,
            capture_output=True,
            timeout=settings.timeout_seconds,
            check=False,
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ADR worker exited {completed.returncode}: {completed.stderr[-500:]}"
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "ADR worker failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("ADR worker returned an invalid result")
        return result


def _reason(result: dict[str, Any]) -> str:
    detections = result.get("detections") or []
    for item in detections:
        if isinstance(item, dict) and item.get("description"):
            return str(item["description"])
    return "ADR detected a potentially malicious action"
