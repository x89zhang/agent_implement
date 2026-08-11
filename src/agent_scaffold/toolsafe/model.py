from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSafeResult:
    risk_score: float | None
    threshold: float
    allowed: bool
    reason: str
    mode: str
    model: str
    latency_ms: int
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "threshold": self.threshold,
            "allowed": self.allowed,
            "reason": self.reason,
            "mode": self.mode,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "raw_response": self.raw_response,
            "error": self.error,
        }
