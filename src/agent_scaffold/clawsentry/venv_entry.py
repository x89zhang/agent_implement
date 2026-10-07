"""Entry points executed by the isolated ClawSentry interpreter.

This file runs under ``/opt/clawsentry-venv/bin/python`` and imports only
upstream ClawSentry, never ``agent_scaffold``.

``gateway``: start the unmodified upstream gateway (``clawsentry gateway``)
after two runtime adaptations. Upstream's OpenAI provider calls Chat
Completions with ``max_tokens`` and ``temperature``
(``gateway/llm/provider.py:303-320``), which GPT-5 and o-series models reject;
for those models only, ``max_tokens`` becomes ``max_completion_tokens`` and
``temperature`` is dropped. Each provider call gets its own SDK client
(see ``_per_call_client``). Prompts and parsing are untouched.

``fallback``: read a canonical event (JSON) on stdin and print upstream's
local fallback decision for an unreachable gateway, computed exactly as the
upstream adapter does (``adapters/a3s_adapter.py:432-437``).
"""

from __future__ import annotations

import contextvars
import copy
import inspect
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


_CALL_CLIENT = contextvars.ContextVar("agent_scaffold_clawsentry_client", default=None)


def _per_call_client(provider_cls, adapt=lambda client: client) -> None:
    """Give every ``complete`` call its own SDK client, closed in its own loop.

    Upstream runs each L2 analysis in a fresh event loop on a two-thread pool
    (``policy/engine.py:508, 1230-1244``) but caches one async client per
    provider (``gateway/llm/provider.py:149-158, 246-255``). The cached httpx
    pool stays bound to the first loop, so later calls fail with "Event loop
    is closed" and fall back to L1. The client is built exactly as upstream
    builds it, on a copy of the provider so the shared cache is never touched
    across threads, and closed as upstream's ``aclose`` does before its loop
    exits.
    """
    build = provider_cls._get_client
    complete = provider_cls.complete
    if getattr(complete, "_agent_scaffold_adapted", False):
        return

    def fresh(self):
        shadow = copy.copy(self)
        shadow._client = None
        return adapt(build(shadow))

    def _get_client(self):
        return _CALL_CLIENT.get() or fresh(self)

    async def _complete(self, *args, **kwargs):
        client = fresh(self)
        token = _CALL_CLIENT.set(client)
        try:
            return await complete(self, *args, **kwargs)
        finally:
            _CALL_CLIENT.reset(token)
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    _complete._agent_scaffold_adapted = True
    provider_cls._get_client = _get_client
    provider_cls.complete = _complete


def _adapt_completions(client):
    completions = client.chat.completions
    if not getattr(completions, "_agent_scaffold_adapted", False):
        create = completions.create

        async def adapted_create(**kwargs):
            return await create(**_adapt_request(kwargs))

        completions.create = adapted_create
        completions._agent_scaffold_adapted = True
    return client


def _patch_openai_provider() -> None:
    from clawsentry.gateway.llm import provider

    _per_call_client(provider.OpenAIProvider, _adapt_completions)
    _per_call_client(provider.AnthropicProvider)


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


def _check_llm_dependencies() -> None:
    """Fail at startup instead of letting every L2 call fall back to L1.

    Upstream builds its SDK clients on ``httpx.AsyncClient``
    (``gateway/llm/provider.py:34-39``). Without it each analysis raises
    ``ModuleNotFoundError`` and the gateway silently serves L1-only verdicts.
    """
    provider = os.environ.get("CS_LLM_PROVIDER", "").strip().lower()
    if not provider:
        return
    import importlib

    for module in ("httpx", "anthropic" if provider == "anthropic" else "openai"):
        importlib.import_module(module)


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "gateway":
        _check_llm_dependencies()
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
