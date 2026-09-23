from __future__ import annotations

import json

# nullary marker tuple kind -> the marker string the router prompt uses
_KIND_TO_STR = {
    "sourced": "SOURCED", "prompt": "PROMPT",
    "record": "RECORD", "dest": "DEST", "explicit": "EXPLICIT", "free": "FREE",
    "magnitude": "MAGNITUDE", "cred": "CRED",
}


def marker_to_str(rule: tuple) -> str:
    """A policy_synthesis marker tuple -> the router's marker string (the normal form for comparison)."""
    kind = rule[0]
    if kind == "const":
        return "CONST:" + json.dumps(rule[1])
    if kind == "oneof":
        return "ONEOF:" + json.dumps(sorted(rule[1]))
    if kind in _KIND_TO_STR:
        return _KIND_TO_STR[kind]
    raise ValueError(f"markers: cannot serialize rule {rule!r}")


_STR_TO_KIND = {v: k for k, v in _KIND_TO_STR.items()}


def str_to_rule(s: str) -> tuple:
    """Inverse of marker_to_str: a router marker string -> a policy_synthesis rule tuple."""
    if s.startswith("CONST:"):
        return ("const", json.loads(s[len("CONST:"):]))
    if s.startswith("ONEOF:"):
        return ("oneof", json.loads(s[len("ONEOF:"):]))
    if s in _STR_TO_KIND:
        return (_STR_TO_KIND[s],)
    raise ValueError(f"markers: cannot parse marker string {s!r}")


def scope_from_dict(d: dict):
    """A normal-form scope dict (bucket / named_source / overrides-as-marker-strings) -> a TaskScope with
    rule-tuple overrides, ready for compile_policy. Inverse of scope_to_dict."""
    from .policy_synthesis import TaskScope
    overrides = {tool: {arg: str_to_rule(m) for arg, m in argmap.items()}
                 for tool, argmap in (d.get("overrides") or {}).items()}
    ns = d.get("named_source") or None
    return TaskScope(bucket=d["bucket"], named_source=ns, overrides=overrides)


def overrides_to_str(overrides: dict) -> dict:
    """{tool: {arg: rule-tuple}} -> {tool: {arg: marker-str}} (normal form)."""
    return {tool: {arg: marker_to_str(rule) for arg, rule in argmap.items()}
            for tool, argmap in (overrides or {}).items()}


def scope_to_dict(scope) -> dict:
    """A TaskScope -> a plain, comparable/serializable dict in normal form."""
    return {
        "bucket": scope.bucket,
        "named_source": sorted(scope.named_source) if scope.named_source else [],
        "overrides": overrides_to_str(scope.overrides),
    }


def sensitive_to_pairs(sensitive: dict) -> set[tuple[str, str]]:
    """A *_SENSITIVE table {tool: {arg: default}} -> the set of guarded (tool, arg) pairs."""
    return {(tool, arg) for tool, args in sensitive.items() for arg in args}

