"""Helpers shared by guards that inspect tool results produced through Hermes."""

from __future__ import annotations

import json


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
