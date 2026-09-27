"""Entry points executed by the isolated ClawSentry interpreter.

This file runs under ``/opt/clawsentry-venv/bin/python`` and imports only
upstream ClawSentry, never ``agent_scaffold``.

``gateway``: start the unmodified upstream gateway (``clawsentry gateway``)
after one model adaptation. Upstream's OpenAI provider calls Chat
Completions with ``max_tokens`` and ``temperature``
(``gateway/llm/provider.py:303-320``), which GPT-5 and o-series models reject;
for those models only, ``max_tokens`` becomes ``max_completion_tokens`` and
``temperature`` is dropped. Prompts and parsing are untouched.

``fallback``: read a canonical event (JSON) on stdin and print upstream's
local fallback decision for an unreachable gateway, computed exactly as the
upstream adapter does (``adapters/a3s_adapter.py:432-437``).
"""

from __future__ import annotations

import json
import os
import re
import sys

# Running this file by path puts its directory first on sys.path; keep the
# sibling adapter modules (client.py, ...) from shadowing upstream imports.
if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)


def _needs_completion_tokens(model: str) -> bool:
    return bool(re.match(r"^(gpt-5|o\d)", str(model or "").strip().lower()))


def _adapt_request(kwargs: dict) -> dict:
    if not _needs_completion_tokens(kwargs.get("model", "")):
        return kwargs
    adapted = dict(kwargs)
    adapted.pop("temperature", None)
    if "max_tokens" in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
    return adapted


def _patch_openai_provider() -> None:
    from clawsentry.gateway.llm import provider

    original = provider.OpenAIProvider._get_client
    if getattr(original, "_agent_scaffold_adapted", False):
        return

    def _get_client(self):
        client = original(self)
        completions = client.chat.completions
        if not getattr(completions, "_agent_scaffold_adapted", False):
            create = completions.create

            async def adapted_create(**kwargs):
                return await create(**_adapt_request(kwargs))

            completions.create = adapted_create
            completions._agent_scaffold_adapted = True
        return client

    _get_client._agent_scaffold_adapted = True
    provider.OpenAIProvider._get_client = _get_client


def _fallback() -> int:
    from clawsentry.gateway.models import CanonicalEvent

    try:
        from clawsentry.gateway.policy_engine import make_fallback_decision
    except ImportError:
        from clawsentry.gateway.policy.engine import make_fallback_decision

    event = CanonicalEvent(**json.load(sys.stdin))
    has_high_danger = bool(
        set(event.risk_hints) & {"destructive_pattern", "shell_execution"}
    )
    decision = make_fallback_decision(
        event, risk_hints_contain_high_danger=has_high_danger
    )
    print(json.dumps(decision.model_dump(mode="json")))
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "gateway":
        _patch_openai_provider()
        from clawsentry.cli.main import main as clawsentry_main

        clawsentry_main(["gateway", *argv[1:]])
        return 0
    if command == "fallback":
        return _fallback()
    print("usage: venv_entry.py gateway|fallback", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
