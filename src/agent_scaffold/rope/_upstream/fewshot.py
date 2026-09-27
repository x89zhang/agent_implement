from __future__ import annotations

import json

from .markers import scope_to_dict
from .scopes_io import load_router_scopes

# exemplar pool = the AgentDyn suites (what the ablation drew from); order fixed for reproducibility.
_POOL = ("github", "shopping", "dailylife")
# the authenticated user's own repositories (github), for trusted-facts self-reference resolution.
OWN_REPOS = ["emmajohnson/file_compression", "emmajohnson/image_transformation",
             "emmajohnson/linear_algebra_operation"]


def _pick_task_exemplars(tasks: dict, k: int = 2):
    """Up to k (prompt, scope) exemplars, preferring bucket coverage and a named_source/override.
    Verbatim from the router ablation."""
    items = list(tasks.items())
    chosen, seen_buckets = [], set()
    for want_rich in (True, False):
        for prompt, scope in items:
            if len(chosen) >= k:
                break
            rich = bool(scope.named_source) or bool(scope.overrides)
            if scope.bucket not in seen_buckets and (rich == want_rich):
                chosen.append((prompt, scope))
                seen_buckets.add(scope.bucket)
    for prompt, scope in items:
        if len(chosen) >= k:
            break
        if (prompt, scope) not in chosen:
            chosen.append((prompt, scope))
    return chosen[:k]


def fewshot(target_suite: str, k_per_suite: int = 2) -> str:
    """Leave-target-out few-shot block (TASK -> scope-JSON), from the cached opus scopes of the pool."""
    blocks = []
    for suite in _POOL:
        if suite == target_suite:
            continue
        _, scopes = load_router_scopes("opus", suite)
        for prompt, scope in _pick_task_exemplars(scopes, k_per_suite):
            blocks.append(f"TASK: {prompt}\nANSWER: " + json.dumps(scope_to_dict(scope)))
    return "\n\n".join(blocks)


def trusted_facts(suite: str) -> str:
    """The user's own non-attacker-writable account facts (only github has a gold-relevant anchor)."""
    if suite == "github":
        return "  the authenticated user's own repositories: " + ", ".join(OWN_REPOS)
    return ""
