from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolState:
    tool: str
    category: str
    sensitivity: str
    side_effect: str
    outcome: str

    def encode(self) -> str:
        return "|".join([self.tool, self.category, self.sensitivity, self.side_effect, self.outcome])


class ToolTraceAbstraction:
    """Default abstraction from scaffold tool events to finite Pro2Guard states."""

    def encode_tool_call(self, state: dict[str, Any], name: str, payload: Any) -> str:
        previous = state.get("_pro2guard_last_outcome", "unknown")
        return ToolState(
            tool=_bucket_tool_name(name),
            category=_tool_category(name, payload),
            sensitivity=_sensitivity(payload),
            side_effect=_side_effect(name, payload),
            outcome=str(previous),
        ).encode()

    def encode_tool_result(self, name: str, payload: Any, result: str, failed: bool) -> str:
        return ToolState(
            tool=_bucket_tool_name(name),
            category=_tool_category(name, payload),
            sensitivity=_sensitivity(payload),
            side_effect=_side_effect(name, payload),
            outcome="failed" if failed else "ok",
        ).encode()


def _bucket_tool_name(name: str) -> str:
    text = str(name or "").strip().lower()
    return re.sub(r"[^a-z0-9_:-]+", "_", text)[:80] or "unknown"


def _tool_category(name: str, payload: Any) -> str:
    text = f"{name}\n{_payload_text(payload)}".lower()
    groups = [
        ("communication", ["email", "mail", "send", "slack", "message", "notify", "sms", "post", "comment"]),
        ("network", ["http", "request", "fetch", "url", "web", "download", "upload", "browse", "search", "scrape"]),
        ("file", ["file", "read", "write", "open", "path", "dir", "folder", "mkdir", "delete", "remove"]),
        ("code", ["python", "shell", "bash", "cmd", "exec", "terminal", "subprocess", "repl"]),
        ("repository", ["github", "git", "issue", "repo", "commit", "pull", "push"]),
        ("database", ["sql", "query", "database", "db", "postgres", "sqlite", "mysql"]),
    ]
    for category, keywords in groups:
        if any(keyword in text for keyword in keywords):
            return category
    return "unknown"


def _side_effect(name: str, payload: Any) -> str:
    text = f"{name}\n{_payload_text(payload)}".lower()
    if re.search(r"\b(delete|remove|drop|truncate|rm\s+-|destroy)\b", text):
        return "destructive"
    if re.search(r"\b(write|upsert|create|update|send|post|publish|deploy|commit|push)\b", text):
        return "mutating"
    return "read_only"


def _sensitivity(payload: Any) -> str:
    text = _payload_text(payload)
    if _contains_secret(text):
        return "secret"
    if _contains_pii(text):
        return "pii"
    if len(text) > 10_000:
        return "large"
    return "normal"


def _payload_text(payload: Any) -> str:
    try:
        return str(payload)
    except Exception:
        return repr(payload)


def _contains_secret(text: str) -> bool:
    patterns = [
        r"(?i)(api[_-]?key|secret|token|password|passwd|credential)[\s:=]+[^\s,;]{8,}",
        r"sk-[A-Za-z0-9_-]{16,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"AKIA[0-9A-Z]{16}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _contains_pii(text: str) -> bool:
    return bool(
        re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        or re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)
        or re.search(r"\b(?:\d[ -]*?){13,16}\b", text)
    )
