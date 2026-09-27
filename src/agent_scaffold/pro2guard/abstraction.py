"""Predicate abstraction of tool-agent state (upstream ``safereach/abstraction.py``).

A state is the bitstring of the abstraction's quantified predicates evaluated
on the agent state (``EmbodiedAbstraction.encode``); ``FINISH`` is the single
absorbing terminal appended to every trace. Unsafe states are the states that
satisfy the unsafe specification (``abs.filter(unsafe_spec)``,
``embodied/monitor.py:138-139``).
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Iterable

from ..tool_results import unwrap_hermes_result
from .predicate import QuantifiedPredicate, atomic_from_dict, conjunction, predicate_from_dict

FINISH = "finish"
# Upstream enumerates every bitstring; beyond this size filter() only
# inspects the supplied (learned) states.
_ENUMERATION_LIMIT = 16


def step_observation(name: str, payload: Any, result: Any, failed: bool) -> dict[str, Any]:
    """One executed tool step: the call, its arguments and its observed result."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return {
        "tool": str(name or ""),
        "args": payload if isinstance(payload, dict) else {"input": payload},
        "result": unwrap_hermes_result(text),
        "failed": bool(failed),
    }


class PredicateAbstraction:
    """``EmbodiedAbstraction`` over tool steps instead of simulator objects."""

    def __init__(self, predicates: list[QuantifiedPredicate], unsafe_spec: list[list[tuple[str, bool]]] | None = None,
                 unsafe_descriptions: list[str] | None = None,
                 conditions: list[dict[str, Any]] | None = None) -> None:
        self.predicates = list(predicates)
        # Disjunction of upstream proposition masks [(str(predicate), value)].
        self.unsafe_spec = [list(item) for item in (unsafe_spec or [])]
        self.unsafe_descriptions = list(unsafe_descriptions or [])
        # The authored (generated) unsafe specification, kept verbatim.
        self.conditions = list(conditions or [])
        self.state_idx: dict[str, int] | None = None
        self.state_interpretation: dict[str, Any] | None = None

    # -- upstream Abstraction interface ------------------------------------
    def get_state_idx(self, states: Iterable[str]) -> dict[str, int]:
        if self.state_idx is None:
            self.state_idx = {s: i for i, s in enumerate(states)}
        return self.state_idx

    def get_state_interpretation(self, states: Iterable[str]) -> dict[str, Any]:
        if self.state_interpretation is None:
            self.state_interpretation = {s: self.decode(s) for s in states}
        return self.state_interpretation

    def encode(self, observations: Any) -> str:
        if observations == FINISH:
            return FINISH
        return "".join("1" if pred.state_eval(observations) else "0" for pred in self.predicates)

    def decode(self, bitstr: str) -> Any:
        if bitstr == FINISH:
            return FINISH
        return [(pred.to_dict(), bit) for pred, bit in zip(self.predicates, bitstr)]

    def valid_trans(self, state1: str, state2: str) -> bool:
        # embodied/abstraction.py: FINISH has no successors; all else is valid.
        return state1 != FINISH

    def filter(self, propositions: list[tuple[str, bool]], states: Iterable[str] | None = None) -> set[str]:
        pred_idxes = {str(pred): i for i, pred in enumerate(self.predicates)}
        masks: dict[int, str] = {}
        for predicate, value in propositions:
            if str(predicate) not in pred_idxes:
                raise ValueError("Unknown predicate for the current abstraction")
            masks[pred_idxes[str(predicate)]] = "1" if value else "0"
        if states is None:
            if len(self.predicates) > _ENUMERATION_LIMIT:
                raise ValueError("too many predicates to enumerate; pass the learned states")
            states = ("".join(bits) for bits in itertools.product("01", repeat=len(self.predicates)))
        return {
            state for state in states
            if state != FINISH and len(state) == len(self.predicates)
            and all(state[idx] == bit for idx, bit in masks.items())
        }

    def unsafe_states(self, states: Iterable[str] | None = None) -> set[str]:
        states = list(states) if states is not None else None
        result: set[str] = set()
        for propositions in self.unsafe_spec:
            result |= self.filter(propositions, states)
        return result

    def unsafe_spec_text(self) -> str:
        """String form of the unsafe specification used in the reflection message.

        Upstream interpolates ``str(task_obj["unsafe_state"])``, the authored
        spec; the generated conditions are its counterpart here.
        """
        if self.conditions:
            return json.dumps(self.conditions, ensure_ascii=False, sort_keys=True)
        parts = [" and ".join(f"{p}" if v else f"not {p}" for p, v in props) for props in self.unsafe_spec]
        return " or ".join(f"[{part}]" for part in parts)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "predicates": [pred.to_dict() for pred in self.predicates],
            "unsafe_spec": [[[p, v] for p, v in props] for props in self.unsafe_spec],
            "unsafe_descriptions": self.unsafe_descriptions,
            "unsafe_conditions": self.conditions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredicateAbstraction":
        predicates = [predicate_from_dict(item) for item in raw.get("predicates") or []]
        if not all(isinstance(pred, QuantifiedPredicate) for pred in predicates):
            raise ValueError("abstraction predicates must be quantified")
        spec = [[(str(p), bool(v)) for p, v in props] for props in raw.get("unsafe_spec") or []]
        return cls(predicates, spec, raw.get("unsafe_descriptions") or [],
                   raw.get("unsafe_conditions") or [])


def abstraction_from_conditions(conditions: list[dict[str, Any]]) -> PredicateAbstraction:
    """Build the abstraction from unsafe conditions as ``embodied/build.py:22-52``.

    Each condition is a conjunction of atomic checks on one executed step. Every
    atom becomes ``exist atom`` and the conjunction becomes ``exist conj``; the
    unsafe specification is that the conjunction predicate holds.
    """
    predicates: list[QuantifiedPredicate] = []
    seen: set[str] = set()
    spec: list[list[tuple[str, bool]]] = []
    descriptions: list[str] = []

    def add(pred: QuantifiedPredicate) -> None:
        if str(pred) not in seen:
            seen.add(str(pred))
            predicates.append(pred)

    for condition in conditions:
        atoms = [atomic_from_dict(item) for item in condition.get("all") or []]
        if not atoms:
            raise ValueError("unsafe condition needs at least one atomic check")
        for atom in atoms:
            add(QuantifiedPredicate("exist", atom))
        conj = QuantifiedPredicate("exist", conjunction(atoms))
        add(conj)
        spec.append([(str(conj), True)])
        descriptions.append(str(condition.get("description") or ""))
    if not spec:
        raise ValueError("at least one unsafe condition is required")
    return PredicateAbstraction(predicates, spec, descriptions, conditions)
