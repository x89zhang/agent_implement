"""Upstream ClawSentry adapter helpers used by the agent-side client.

ClawSentry 0.8.7 lives in an isolated interpreter (``/opt/clawsentry-venv``),
so the two small pure helpers the upstream a3s adapter applies to every event
are copied verbatim here; the fallback decision is computed by upstream code
in that interpreter (see ``venv_entry.py``).
"""

from __future__ import annotations

import json
import os
import re as _re
import subprocess
from pathlib import Path
from typing import Any, Optional

CLAWSENTRY_PYTHON = "/opt/clawsentry-venv/bin/python"
VENV_ENTRY = Path(__file__).with_name("venv_entry.py")

# --- Verbatim from clawsentry 0.8.7 adapters/a3s_adapter.py:43-77 ---------

_EXTERNAL_TOOLS = frozenset({
    "http_request", "web_fetch", "fetch", "web_search",
    "mcp__fetch__fetch",
})
_USER_TOOLS = frozenset({
    "read_file", "write_file", "edit_file", "grep", "glob",
    "read", "write", "edit",
})
_BASH_EXTERNAL_PATTERN = _re.compile(
    r"(?:\bcurl\b|\bwget\b|https?://)", _re.IGNORECASE
)


def infer_content_origin(tool_name: str | None, payload: dict[str, Any]) -> str:
    """Infer whether event content originates from external or user sources.

    Returns ``"external"``, ``"user"``, or ``"unknown"``.
    """
    tool = (tool_name or "").lower()
    if tool in _EXTERNAL_TOOLS:
        return "external"
    if tool in _USER_TOOLS:
        # Files in /tmp/ or similar may be external-originated
        path = str(payload.get("file_path", "") or payload.get("path", "") or "")
        if path.startswith("/tmp/") or path.startswith("/var/tmp/"):
            return "external"
        return "user"
    if tool in ("bash", "shell", "terminal", "command", "exec"):
        cmd = str(payload.get("command", ""))
        if _BASH_EXTERNAL_PATTERN.search(cmd):
            return "external"
        return "user"
    return "unknown"


# --- Verbatim from clawsentry 0.8.7 gateway/models.py:1771-1782 -----------

def extract_risk_hints(tool_name: Optional[str], command: str) -> list[str]:
    """Extract risk hints from tool_name and command string.

    Shared across A3S and OpenClaw adapters.
    """
    hints: list[str] = []
    if tool_name and tool_name.lower() in ("bash", "shell", "exec", "sudo"):
        hints.append("shell_execution")
    cmd_lower = command.lower()
    if "rm " in cmd_lower or "sudo" in cmd_lower:
        hints.append("destructive_pattern")
    return hints


def enrich_event(event: dict[str, Any]) -> dict[str, Any]:
    """Add risk hints and content origin as ``normalize_hook_event`` does."""
    payload = dict(event.get("payload") or {})
    tool_name = event.get("tool_name")
    meta = payload.get("_clawsentry_meta")
    merged = dict(meta) if isinstance(meta, dict) else {}
    merged["content_origin"] = infer_content_origin(tool_name, payload)
    payload["_clawsentry_meta"] = merged
    return {
        **event,
        "payload": payload,
        "risk_hints": extract_risk_hints(tool_name, str(payload.get("command", ""))),
    }


def fallback_decision(event: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Upstream ``make_fallback_decision`` for an unreachable gateway."""
    python = os.environ.get("AGENT_CLAWSENTRY_PYTHON") or CLAWSENTRY_PYTHON
    completed = subprocess.run(
        [python, str(VENV_ENTRY), "fallback"],
        input=json.dumps(event, default=str), capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise RuntimeError(
            "upstream fallback failed: " + (detail[-1] if detail else f"exit {completed.returncode}")
        )
    decision = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(decision, dict) or "decision" not in decision:
        raise RuntimeError("upstream fallback returned no decision")
    return decision


# --- Managed gateway L2/L3 provider -------------------------------------

_SUPPORTED_PROVIDERS = {"openai", "anthropic"}
# OpenAI-compatible project providers use upstream's OpenAI provider.
_OPENAI_COMPATIBLE = {"openai", "openrouter", "vllm", "azure_openai", "together"}


def gateway_llm_environment(cfg: Any, source: dict[str, str]) -> dict[str, str]:
    """Return the upstream ``CS_LLM_*`` settings for the managed gateway.

    Empty ``clawsentry.llm`` fields inherit the top-level ``llm`` (the guard
    model is never switched to an upstream default). Variables already set in
    ``source`` win, so an operator can still configure the gateway directly.
    """
    settings = cfg.clawsentry.llm
    if not settings.enabled:
        return {}
    top = cfg.llm
    provider = (settings.provider or top.provider or "").strip().lower()
    if provider in _OPENAI_COMPATIBLE:
        provider = "openai"
    if provider not in _SUPPORTED_PROVIDERS:
        return {}
    inherit = not settings.provider or settings.provider.strip().lower() == str(top.provider).strip().lower()
    model = settings.model or (top.model if inherit else "")
    base_url = settings.base_url or (top.base_url if inherit else "")
    key_env = settings.api_key_env or (top.api_key_env if inherit else "")
    api_key = source.get(key_env, "") if key_env else ""
    if not api_key and inherit and not settings.api_key_env:
        api_key = str(getattr(top, "api_key", "") or "")
    env = {"CS_LLM_PROVIDER": provider}
    if model:
        env["CS_LLM_MODEL"] = model
    if base_url:
        env["CS_LLM_BASE_URL"] = base_url
    if api_key:
        env["CS_LLM_API_KEY"] = api_key
    if settings.l3_enabled:
        env["CS_L3_ENABLED"] = "true"
    return {name: value for name, value in env.items() if not source.get(name)}
