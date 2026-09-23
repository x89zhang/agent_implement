"""The enforcement clamp.

A router reads only the trusted request, so an injection cannot steer it, but a weak router can still
mistakenly loosen a parameter below its floor default. The clamp honors an override only when it is at
least as strict as the floor and drops the rest, so a router error costs utility, never soundness.
"""

from __future__ import annotations

from .markers import marker_to_str
from .policy_synthesis import TaskScope
from .scopes_io import floor_to_dict, load_floor

# IPI-threat permissiveness order (read off origin.py), as marker strings.
_ORIGIN_FAMILY = {"SOURCED", "PROMPT", "RECORD"}


def _pinned(m: str) -> bool:
    return m.startswith("CONST:") or m.startswith("ONEOF:")


def _more_permissive(x: str, y: str) -> bool:
    """True if marker x admits ATTACKER-REACHABLE provenance that y blocks (x looser than y). A
    named-source difference is deliberately NOT flagged: the router only sees the trusted request, so
    any source it emits is request-derived (trusted), a utility matter, not a soundness hole."""
    if y != "FREE" and x == "FREE":
        return True
    if y == "EXPLICIT" and x != "EXPLICIT":
        return True
    if _pinned(y) and (x in _ORIGIN_FAMILY or x == "FREE"):
        return True
    return False


def _default_markers(suite: str) -> dict:
    """{(tool, arg): default-marker-str} from the suite's audited floor."""
    floor = floor_to_dict(load_floor(suite))
    return {(tool, arg): m for tool, args in floor.items() for arg, m in args.items()}


def clamp_taskscope(suite: str, scope: TaskScope) -> TaskScope:
    """Drop overrides that are looser than the floor default (revert those params to the floor)."""
    defaults = _default_markers(suite)
    kept: dict = {}
    for tool, argmap in (scope.overrides or {}).items():
        for arg, rule in argmap.items():
            dflt = defaults.get((tool, arg))
            if dflt is None:                       # not a guarded sensitive param -> compiler ignores it
                kept.setdefault(tool, {})[arg] = rule
                continue
            if _more_permissive(marker_to_str(rule), dflt):
                continue                            # looser than floor -> drop -> revert to floor default
            kept.setdefault(tool, {})[arg] = rule
    return TaskScope(bucket=scope.bucket, named_source=scope.named_source, overrides=kept)

