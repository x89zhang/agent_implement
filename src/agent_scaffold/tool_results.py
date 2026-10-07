"""Helpers shared by guards that inspect tool results produced through Hermes."""

from __future__ import annotations

import json
import re

# Per-run identifiers in tool descriptions (e.g. Hermes' per-run home
# directory) are normalized so generated guard inputs and their cache keys
# are stable across runs of the same task.
_RUN_SPECIFIC = (
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
     "<id>"),
    (re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{16,}(?![0-9A-Za-z])"), "<id>"),
    (re.compile(r"\b\d{8}T\d{6}Z\b"), "<timestamp>"),
    (re.compile(r"\brun_\d+\b"), "run_<n>"),
    (re.compile(r"\btmp[A-Za-z0-9_]{6,}\b"), "tmp<id>"),
)


def normalize_run_specific(text: str) -> str:
    for pattern, replacement in _RUN_SPECIFIC:
        text = pattern.sub(replacement, text)
    return text


def unwrap_hermes_result(result: str) -> str:
    """Return the bare tool text inside Hermes' MCP envelope.

    Hermes renders an MCP result as ``{"result": text}``, optionally with
    ``structuredContent``/``_meta`` (tools/mcp_tool_handlers.py). Upstream
    guards read the bare tool text, so anything else is returned unchanged.
    """
    try:
        value = json.loads(result)
    except (TypeError, ValueError):
        return result
    if not isinstance(value, dict) or "result" not in value or set(value) - {"result", "structuredContent", "_meta"}:
        return result
    inner = value["result"]
    if isinstance(inner, str):
        return inner
    return json.dumps(inner, ensure_ascii=False)
