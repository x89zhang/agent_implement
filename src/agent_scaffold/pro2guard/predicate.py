"""ProbGuard predicates over observed tool-agent state.

Mirrors upstream ``safereach/predicate.py``: an ``AtomicPredicate`` evaluates
``observation[lhs] op rhs`` (with an optional negation), a ``BinaryPredicate``
joins two predicates with ``and``/``or``, and a ``QuantifiedPredicate``
quantifies a predicate over a list of observations (``exist``/``all``).

Tool-agent adaptation: an observation is one executed tool step
``{"tool", "args", "result", "failed"}`` and the agent state after step t is
the list of all step observations so far, playing the role of the embodied
object list. ``lhs`` is a dotted path into the observation (``tool``,
``failed``, ``result``, ``args`` or ``args.<param>[.<key>...]``). Besides the
upstream comparison operators, the text operators ``contains``, ``in`` and
``matches`` are provided because tool arguments and results are strings.
Evaluation is deterministic; no LLM is involved at runtime.
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass
from typing import Any

_MISSING = object()

COMPARISON_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}
# Upstream NEGATE_OP: a negated comparison uses the complementary operator.
NEGATE_OP = {"==": "!=", "!=": "==", ">": "<=", "<": ">=", ">=": "<", "<=": ">"}
TEXT_OPS = {"contains", "in", "matches"}
OPS = set(COMPARISON_OPS) | TEXT_OPS
# Accepted spellings for negated text operators in generated specifications.
_NEGATED_ALIASES = {"not_contains": "contains", "not_in": "in", "not_matches": "matches"}
_FIELD = re.compile(r"^(tool|failed|result|args)(\.[A-Za-z0-9_\-]+)*$")


@dataclass(frozen=True)
class AtomicPredicate:
    lhs: str
    op: str
    rhs: Any
    neg: bool = False

    def __str__(self) -> str:
        n = "!" if self.neg else ""
        return f"{n}({self.lhs} {self.op} {json.dumps(self.rhs, ensure_ascii=False, sort_keys=True)})"

    def state_eval(self, observation: dict[str, Any]) -> bool:
        value = lookup(observation, self.lhs)
        # Upstream indexes observation[lhs] directly. A field the step does not
        # have (another tool's parameter, or a null argument) satisfies neither
        # the predicate nor its negation.
        if value is _MISSING or value is None:
            return False
        if self.op in COMPARISON_OPS:
            op = NEGATE_OP[self.op] if self.neg else self.op
            return _compare(value, op, self.rhs)
        result = _text_eval(value, self.op, self.rhs)
        return not result if self.neg else result

    def to_dict(self) -> dict[str, Any]:
        return {"lhs": self.lhs, "op": self.op, "rhs": self.rhs, "neg": self.neg}


@dataclass(frozen=True)
class BinaryPredicate:
    lhs: Any
    op: str
    rhs: Any

    def __str__(self) -> str:
        return f"({self.lhs}) {self.op} ({self.rhs})"

    def state_eval(self, observation: dict[str, Any]) -> bool:
        if self.op == "and":
            return self.lhs.state_eval(observation) and self.rhs.state_eval(observation)
        return self.lhs.state_eval(observation) or self.rhs.state_eval(observation)

    def to_dict(self) -> dict[str, Any]:
        return {"lhs": self.lhs.to_dict(), "op": self.op, "rhs": self.rhs.to_dict()}


@dataclass(frozen=True)
class QuantifiedPredicate:
    quantifier: str
    predicate: Any

    def __str__(self) -> str:
        return f"{self.quantifier} {self.predicate}"

    def state_eval(self, observations: list[dict[str, Any]]) -> bool:
        if self.quantifier == "exist":
            return any(self.predicate.state_eval(o) for o in observations)
        return all(self.predicate.state_eval(o) for o in observations)

    def to_dict(self) -> dict[str, Any]:
        return {"quantifier": self.quantifier, "predicate": self.predicate.to_dict()}


def atomic_from_dict(raw: Any) -> AtomicPredicate:
    if not isinstance(raw, dict):
        raise ValueError("atomic predicate must be an object")
    lhs = str(raw.get("lhs", "")).strip()
    op = str(raw.get("op", "")).strip().lower()
    neg = bool(raw.get("neg", False))
    if op in _NEGATED_ALIASES:
        op, neg = _NEGATED_ALIASES[op], not neg
    if not _FIELD.match(lhs):
        raise ValueError(f"unsupported predicate field: {lhs!r}")
    if op not in OPS:
        raise ValueError(f"unsupported predicate operator: {op!r}")
    if "rhs" not in raw:
        raise ValueError("atomic predicate needs rhs")
    rhs = raw["rhs"]
    if op == "in" and not isinstance(rhs, list):
        raise ValueError("operator 'in' needs a list rhs")
    if op == "matches":
        re.compile(str(rhs))
    if op in {">", "<", ">=", "<="} and _number(rhs) is None:
        raise ValueError(f"operator {op!r} needs a numeric rhs")
    if isinstance(rhs, (dict,)):
        raise ValueError("rhs must be a literal or a list of literals")
    return AtomicPredicate(lhs=lhs, op=op, rhs=rhs, neg=neg)


def predicate_from_dict(raw: Any) -> Any:
    if isinstance(raw, dict) and "quantifier" in raw:
        quantifier = str(raw["quantifier"]).lower()
        if quantifier not in {"exist", "all"}:
            raise ValueError(f"unsupported quantifier: {quantifier!r}")
        return QuantifiedPredicate(quantifier, predicate_from_dict(raw["predicate"]))
    if isinstance(raw, dict) and str(raw.get("op", "")).lower() in {"and", "or"}:
        return BinaryPredicate(
            predicate_from_dict(raw["lhs"]), str(raw["op"]).lower(),
            predicate_from_dict(raw["rhs"]),
        )
    return atomic_from_dict(raw)


def conjunction(predicates: list[Any]) -> Any:
    """Left-nested conjunction, built as in upstream ``embodied/build.py``."""
    result = None
    for predicate in predicates:
        result = predicate if result is None else BinaryPredicate(result, "and", predicate)
    if result is None:
        raise ValueError("conjunction needs at least one predicate")
    return result


def lookup(observation: Any, path: str) -> Any:
    value = observation
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _compare(value: Any, op: str, rhs: Any) -> bool:
    if isinstance(rhs, bool) or isinstance(value, bool):
        left = _boolean(value)
        right = _boolean(rhs)
        if left is None or right is None:
            return False
        return COMPARISON_OPS[op](left, right) if op in {"==", "!="} else False
    left_number, right_number = _number(value), _number(rhs)
    if left_number is not None and right_number is not None:
        return COMPARISON_OPS[op](left_number, right_number)
    if op in {"==", "!="}:
        return COMPARISON_OPS[op](_norm(value), _norm(rhs))
    return False


def _text_eval(value: Any, op: str, rhs: Any) -> bool:
    if op == "in":
        return any(_compare(value, "==", item) for item in rhs)
    text = _text(value)
    if op == "contains":
        items = rhs if isinstance(rhs, list) else [rhs]
        return any(_norm(item) in text.casefold() for item in items)
    return re.search(str(rhs), text, flags=re.IGNORECASE) is not None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _norm(value: Any) -> str:
    return _text(value).strip().casefold()
