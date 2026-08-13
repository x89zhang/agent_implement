from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentDoGResult:
    judgment: str
    analysis: str
    failure_modes: list[str] = field(default_factory=list)
    risk_consequences: list[str] = field(default_factory=list)
    risk_sources: list[str] = field(default_factory=list)
    checkpoint: str = "pre_reply"
    task: str = "unified"
    mode: str = "diagnose"
    model: str = ""
    latency_ms: int = 0
    trajectory_steps: int = 0
    trajectory_chars: int = 0
    truncated: bool = False
    raw_response: str = ""
    error: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def safe(self) -> bool | None:
        if self.judgment == "safe":
            return True
        if self.judgment == "unsafe":
            return False
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgment": self.judgment,
            "safe": self.safe,
            "analysis": self.analysis,
            "failure_modes": list(self.failure_modes),
            "risk_consequences": list(self.risk_consequences),
            "risk_sources": list(self.risk_sources),
            "checkpoint": self.checkpoint,
            "task": self.task,
            "mode": self.mode,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "trajectory_steps": self.trajectory_steps,
            "trajectory_chars": self.trajectory_chars,
            "truncated": self.truncated,
            "raw_response": self.raw_response,
            "error": self.error,
            "usage": dict(self.usage),
        }
