"""Disk-backed cache for the router's LLM completions.

The router reads only the trusted user task, so its inputs repeat across a task's clean and injected
runs and the completions are safe to memoize. Caching also freezes one result per input, removing
temp-0 nondeterminism. The agent LLM is deliberately not cached: its calls depend on injected runtime
content.

The key is sha256(model || system || user), so editing a prompt or switching models invalidates
automatically. Storage is one JSON object rewritten atomically on each miss; a corrupt file fails
loudly rather than silently resetting.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable


# CANONICAL gemini-via-OpenRouter config: ALWAYS
# pin the google-ai-studio provider and request DYNAMIC thinking (max_tokens -1 = Google's
# native default). Applied only to google/gemini* models; no-op otherwise.
def _gemini_or_eb_kwargs(model_name):
    if not str(model_name).startswith("google/gemini"):
        return {}
    return {"extra_body": {
        "provider": {"order": ["google-ai-studio"], "allow_fallbacks": False},
        "reasoning": {"max_tokens": -1},
    }}



def cache_key(model: str, system: str, user: str) -> str:
    """sha256 over (model, system prompt, user content), NUL-separated so fields can't run together."""
    h = hashlib.sha256()
    for part in (model, system, user):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class LLMCache:
    """A small persistent {key -> completion} store. `enabled=False` turns it into a no-op so callers
    can keep one code path with caching switched off (e.g. a `--no-cache` flag)."""

    def __init__(self, path: str | os.PathLike, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._store: dict[str, dict] = {}
        if self.enabled and self.path.exists():
            # fail loud on a corrupt cache rather than silently starting fresh
            with self.path.open(encoding="utf-8") as f:
                self._store = json.load(f)

    def get(self, model: str, system: str, user: str) -> str | None:
        if not self.enabled:
            return None
        entry = self._store.get(cache_key(model, system, user))
        return entry["response"] if entry is not None else None

    def put(self, model: str, system: str, user: str, response: str) -> None:
        if not self.enabled:
            return
        self._store[cache_key(model, system, user)] = {
            "model": model,
            "response": response,
            "user_preview": user[:200],  # keeps the cache file human-readable / inspectable
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)  # atomic: a kill mid-write can't truncate the cache
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def __len__(self) -> int:
        return len(self._store)


def cached_chat_completer(client, model: str, cache: LLMCache | None = None,
                          *, temperature: float = 0.0) -> Callable[[str, str], str]:
    """Return an `llm_complete(system, user) -> str` that asks `client` (an OpenAI-style client:
    `client.chat.completions.create(...)`). If `cache` is given AND temperature == 0 (deterministic
    intent), completions are memoized on disk; otherwise every call hits the API. This is the exact
    shape the router calls expect, so it drops straight in.

    NOTE: do not wrap the agent LLM with this — the agent must see fresh runtime content."""
    use_cache = cache is not None and cache.enabled and temperature == 0.0

    def llm_complete(system: str, user: str) -> str:
        if use_cache:
            hit = cache.get(model, system, user)
            if hit is not None:
                return hit
        resp = client.chat.completions.create(
            model=model,
            **_gemini_or_eb_kwargs(model), temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        if use_cache:
            cache.put(model, system, user, text)
        return text

    return llm_complete
