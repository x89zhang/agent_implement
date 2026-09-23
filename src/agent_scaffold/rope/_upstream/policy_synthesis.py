from __future__ import annotations

from dataclasses import dataclass, field

from .origin import OriginPredicate, DestinationPredicate

ALLOW, FORBID = 0, 1
FALLBACK_ERROR = 0
PRIO = 10


# ── rule markers for a sensitive parameter ────────────────────────────────────────
def CONST(v): return ("const", v)    # must equal v exactly (a value the user wrote / a T2-resolved target)
SOURCED = ("sourced",)               # in the user's request OR from the named source
PROMPT = ("prompt",)                 # in the user's request only (no source) -- the tightest value origin
RECORD = ("record",)                 # in the user's request OR the user's own AUTHORITATIVE RECORD (order
                                     # history) -- never an injectable source. For a state-changing action
                                     # whose legit target is something the user already owns/did (e.g.
                                     # rebuy from order history); an injected extra item is in neither.
CRED = ("cred",)                     # credential value: in the request OR a NAMED non-injectable sender OR the
                                     # user's own credential store (a /system/.ssh-class read, filed as an
                                     # authoritative record by the OriginTracker). = SOURCED + the record leg.
                                     # No longer any floor's default; kept for scope compat.
def ONEOF(values): return ("oneof", list(values))  # one of an explicit set (e.g. the user's own repos)
MAGNITUDE = ("magnitude",)           # must be a number
DEST = ("dest",)                     # filesystem-write destination guard (block credential/system dirs unless named)
EXPLICIT = ("explicit",)             # explicit-only action: the WHOLE action is irreversible / credential-changing,
                                     # so origin-gating one argument cannot defend it (deleting a *legitimately-named*
                                     # repo passes any origin check yet is still catastrophic). Therefore the action
                                     # is blocked UNLESS the user's own REQUEST explicitly authorizes it -- expressed
                                     # as a per-task override that pins the exact named target (CONST). It is NEVER
                                     # granted via a named source or because a target name happens to look
                                     # trusted. The default (no override) is block. This is realistic and online-
                                     # decidable (irreversibility is a harm-taxonomy property of the TOOL; the
                                     # authorization is read from the trusted request) -- it does NOT depend on
                                     # surveying which tasks exist in the suite.
FREE = ("free",)                     # no restriction on this arg


def _always_false(value):  # the EXPLICIT default (no request authorization) -- a callable secagent runs; falsy => block
    return False


# Policy mode collapses the per-task conditioning, for the ablation:
#   "routed"                 -- the per-task marker as authored. The defense.
#   "always-fully-specified" -- every origin marker becomes request-only.
#   "always-action-open"     -- every origin marker becomes the loosest injection-safe set.
# Only origin markers move. The structural guards (DEST, EXPLICIT, FREE) are bucket-independent, so an
# irreversible action still needs the request's own authorization even under action-open.
POLICY_MODES = ("routed", "always-fully-specified", "always-action-open")
_ORIGIN_FAMILY = frozenset({"sourced", "prompt", "oneof", "record", "cred"})  # origin markers


def apply_policy_mode(rule, mode: str):
    """Collapse a rule marker to the given under-specification level (see POLICY_MODES). Identity for
    'routed'. Leaves structural markers (const kept as the fully-specified exemplar; dest/explicit/
    magnitude/free) untouched except where noted."""
    if mode == "routed":
        return rule
    kind = rule[0]
    if mode == "always-fully-specified":
        # request-only: every provenance marker -> PROMPT; CONST already pins a request value (keep it)
        return PROMPT if kind in _ORIGIN_FAMILY else rule
    if mode == "always-action-open":
        # loosest injection-safe origin: every provenance marker (incl. the CONST pin) -> SOURCED
        return SOURCED if (kind in _ORIGIN_FAMILY or kind == "const") else rule
    raise ValueError(f"unknown policy_mode: {mode!r} (use one of {POLICY_MODES})")


def _restriction(arg, rule, *, named, origin_map, prompt_ids):
    """Map a rule marker to a secagent restriction (a jsonschema dict or a callable predicate), or
    None when the parameter is unconstrained (FREE)."""
    kind = rule[0]
    if kind == "const":     return {"const": rule[1]}
    if kind == "oneof":     return {"enum": rule[1]}
    if kind == "magnitude": return {"type": "number"}
    if kind == "free":      return None
    if kind == "sourced":   return OriginPredicate(arg, named, origin_map, prompt_ids)
    if kind == "prompt":    return OriginPredicate(arg, None, origin_map, prompt_ids)
    if kind == "record":    return OriginPredicate(arg, None, origin_map, prompt_ids, use_records=True)
    if kind == "cred":      return OriginPredicate(arg, named, origin_map, prompt_ids, use_records=True)
    if kind == "dest":      return DestinationPredicate(arg, prompt_ids)
    if kind == "explicit":  return _always_false   # no per-task override authorized this action -> block
    raise ValueError(f"unknown rule kind: {kind!r}")


# ── the deployable model: a GLOBAL state-changing-tool table + a shrunk per-task scope ────────────
@dataclass
class TaskScope:
    """The shrunk per-task spec: which bucket, where the user pointed the agent (named_source), and
    any per-parameter overrides of the GLOBAL sensitive-tool defaults (e.g. a CONST pin when the user
    names a specific recipient). It no longer lists tools -- which tools are guarded is global."""
    bucket: str
    named_source: list[str] | None = None
    overrides: dict = field(default_factory=dict)   # {tool: {arg: rule}}


def compile_policy(sensitive: dict, scope: TaskScope, *, origin_map, prompt_ids,
                   policy_mode: str = "routed") -> dict:
    """Build this task's secagent policy from the GLOBAL state-changing-tool table + the per-task scope.
    For each sensitive (tool, arg) use the task's override rule if present, else the table default,
    then collapse it to the given under-specification level (policy_mode; 'routed' = as authored).
    Benign tools are NOT here -- the bootstrap puts them on the always-allowed list (default-allow).

    The global sensitive table is one universe partitioned into two DISJOINT classes (a tool is in
    exactly one): (i) ORIGIN-CHECKED SENSITIVE PARAMETERS -- the action is generally fine but one argument carries
    redirectable harm, gated by where the value came from (origin markers, which slide with the
    under-specification bucket); and (ii) EXPLICIT-ONLY ACTIONS -- the whole action is irreversible/
    credential-changing, blocked unless the request explicitly authorizes it via a per-task override
    (bucket-independent). The override path below is what lets an explicit-only action be authorized."""
    named = set(scope.named_source) if scope.named_source else None
    policy: dict = {}
    for tool, argrules in sensitive.items():
        condition = {}
        for arg, default_rule in argrules.items():
            rule = scope.overrides.get(tool, {}).get(arg, default_rule)
            rule = apply_policy_mode(rule, policy_mode)
            r = _restriction(arg, rule, named=named, origin_map=origin_map, prompt_ids=prompt_ids)
            if r is not None:
                condition[arg] = r
        policy[tool] = [(PRIO, ALLOW, condition, FALLBACK_ERROR)]
    return policy

