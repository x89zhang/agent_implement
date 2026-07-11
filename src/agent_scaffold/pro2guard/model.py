from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pro2GuardResult:
    probability: float | None
    state: str
    matched_state: str
    threshold: float
    allowed: bool
    reason: str
    mode: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "state": self.state,
            "matched_state": self.matched_state,
            "threshold": self.threshold,
            "allowed": self.allowed,
            "reason": self.reason,
            "mode": self.mode,
            "source": self.source,
        }


class JsonDTMC:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.state_index = {str(k): int(v) for k, v in (raw.get("state_index") or {}).items()}
        self.state_aliases = {str(k): str(v) for k, v in (raw.get("state_aliases") or {}).items()}
        self.unsafe_states = {str(item) for item in (raw.get("unsafe_states") or [])}
        self.transitions = _load_transitions(raw.get("transition_probs") or raw.get("transitions") or {})

    def probability_to_unsafe(self, state: str, unsafe_states: list[str], horizon: int = 20) -> tuple[float, str]:
        matched_state = self._match_state(state)
        unsafe = set(unsafe_states or []) | self.unsafe_states
        unsafe_ids = {self._state_id(item) for item in unsafe}
        unsafe_ids.discard(None)
        start = self._state_id(matched_state)
        if start is None:
            return 0.0, matched_state
        if start in unsafe_ids:
            return 1.0, matched_state

        current = {start: 1.0}
        hit = 0.0
        for _ in range(max(1, horizon)):
            next_dist: dict[int, float] = {}
            for src, mass in current.items():
                row = self.transitions.get(src) or {src: 1.0}
                for dst, prob in row.items():
                    value = mass * prob
                    if dst in unsafe_ids:
                        hit += value
                    else:
                        next_dist[dst] = next_dist.get(dst, 0.0) + value
            current = next_dist
            if not current:
                break
        return min(1.0, max(0.0, hit)), matched_state

    def _match_state(self, state: str) -> str:
        if state in self.state_index:
            return state
        return self.state_aliases.get(state, state)

    def _state_id(self, state: str) -> int | None:
        if state in self.state_index:
            return self.state_index[state]
        try:
            return int(state)
        except (TypeError, ValueError):
            return None


def _load_transitions(raw: dict[str, Any]) -> dict[int, dict[int, float]]:
    transitions: dict[int, dict[int, float]] = {}
    for src, row in raw.items():
        src_id = int(src)
        transitions[src_id] = {}
        for dst, prob in dict(row).items():
            transitions[src_id][int(dst)] = _probability(prob)
    return transitions


def _probability(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        num, denom = text.split("/", 1)
        return float(num) / float(denom)
    return float(text)
