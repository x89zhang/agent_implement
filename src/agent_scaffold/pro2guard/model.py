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
        self.transitions = _load_transitions(raw.get("transition_probs") or raw.get("transitions") or {})

    def probability_to_unsafe(self, state: str, unsafe_states: list[str], horizon: int = 20) -> tuple[float, str]:
        """Unbounded reachability, matching upstream's PRISM ``F unsafe`` query.

        ``horizon`` remains accepted for old callers but does not truncate the
        property: upstream ProbGuard does not impose a step horizon.
        """
        matched_state = self._match_state(state)
        unsafe_ids = {self._state_id(item) for item in unsafe_states or []}
        unsafe_ids.discard(None)
        start = self._state_id(matched_state)
        if start is None:
            raise ValueError(f"ProbGuard state is absent from the learned DTMC: {matched_state}")
        if not unsafe_ids:
            return 0.0, matched_state
        if start in unsafe_ids:
            return 1.0, matched_state

        # States with no positive-probability path to an unsafe state have
        # reachability zero. Removing them also makes the remaining linear
        # system nonsingular, including models with safe absorbing cycles.
        predecessors: dict[int, set[int]] = {}
        for src, row in self.transitions.items():
            for dst, probability in row.items():
                if probability > 0:
                    predecessors.setdefault(dst, set()).add(src)
        reachable = set(unsafe_ids)
        queue = list(unsafe_ids)
        while queue:
            for src in predecessors.get(queue.pop(), ()):
                if src not in reachable:
                    reachable.add(src)
                    queue.append(src)
        if start not in reachable:
            return 0.0, matched_state
        unknown = sorted(reachable - unsafe_ids)
        positions = {node: index for index, node in enumerate(unknown)}
        matrix = [[0.0] * (len(unknown) + 1) for _ in unknown]
        for src in unknown:
            index = positions[src]
            matrix[index][index] = 1.0
            for dst, probability in (self.transitions.get(src) or {src: 1.0}).items():
                if dst in unsafe_ids:
                    matrix[index][-1] += probability
                elif dst in positions:
                    matrix[index][positions[dst]] -= probability
        # Pivoted Gaussian elimination computes the same eventual hitting
        # probability as PRISM for this finite DTMC, without a step cutoff.
        size = len(unknown)
        for column in range(size):
            pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
            if abs(matrix[pivot][column]) < 1e-14:
                raise ValueError("ProbGuard model has a singular transition matrix")
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            divisor = matrix[column][column]
            for item in range(column, size + 1):
                matrix[column][item] /= divisor
            for row in range(column + 1, size):
                factor = matrix[row][column]
                if factor:
                    for item in range(column, size + 1):
                        matrix[row][item] -= factor * matrix[column][item]
        solved = [0.0] * size
        for row in range(size - 1, -1, -1):
            solved[row] = matrix[row][-1] - sum(
                matrix[row][column] * solved[column]
                for column in range(row + 1, size)
            )
        return min(1.0, max(0.0, solved[positions[start]])), matched_state

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
