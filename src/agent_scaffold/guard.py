from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .config import AppConfig


RISK_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass
class AegisDecision:
    allowed: bool = True
    reason: str = ""
    risk_level: str = "LOW"
    category: str = "unknown"
    signals: list[str] | None = None
    mode: str = "off"
    policy: str = "none"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["signals"] = list(self.signals or [])
        return data


def check_tool_call(cfg: AppConfig, state: dict[str, Any], name: str, payload: Any) -> AegisDecision:
    if not cfg.aegis.enabled:
        return AegisDecision(mode="off")
    try:
        decision = _check_tool_call(cfg, state, name, payload)
    except Exception as exc:
        if cfg.aegis.fail_closed:
            return AegisDecision(
                allowed=False,
                reason=f"Aegis guard failed closed: {exc}",
                risk_level="CRITICAL",
                category="unknown",
                signals=["guard_error"],
                mode=cfg.aegis.mode,
                policy="guard_error",
            )
        return AegisDecision(
            allowed=True,
            reason=f"Aegis guard failed open: {exc}",
            risk_level="LOW",
            category="unknown",
            signals=["guard_error"],
            mode=cfg.aegis.mode,
            policy="guard_error",
        )
    return decision


def _check_tool_call(cfg: AppConfig, state: dict[str, Any], name: str, payload: Any) -> AegisDecision:
    mode = cfg.aegis.mode.lower().strip() or "block"
    category = classify_tool(name, payload)
    signals = collect_signals(category, name, payload, state)

    blocked_by_config = name in set(cfg.aegis.block_tools)
    if blocked_by_config:
        signals.append(("configured_block_tool", "CRITICAL", f"Tool {name} is configured as blocked"))
    if not blocked_by_config and name in set(cfg.aegis.allow_tools):
        return AegisDecision(
            allowed=True,
            reason="Tool explicitly allowed by Aegis config",
            risk_level="LOW",
            category=category,
            signals=["configured_allow_tool"],
            mode=mode,
            policy="configured_allow_tool",
        )

    risk_level = "LOW"
    policy = "none"
    details: list[str] = []
    for signal, severity, detail in signals:
        details.append(signal)
        if RISK_ORDER[severity] > RISK_ORDER[risk_level]:
            risk_level = severity
            policy = signal

    threshold = cfg.aegis.risk_threshold.upper()
    if threshold not in RISK_ORDER:
        threshold = "HIGH"
    should_block = RISK_ORDER[risk_level] >= RISK_ORDER[threshold]
    if mode in {"monitor", "audit", "log"}:
        should_block = False

    reason = ""
    if signals:
        first_detail = next((detail for _, severity, detail in signals if severity == risk_level), signals[0][2])
        reason = f"Aegis {risk_level} risk: {first_detail}"

    return AegisDecision(
        allowed=not should_block,
        reason=reason,
        risk_level=risk_level,
        category=category,
        signals=details,
        mode=mode,
        policy=policy,
    )


def classify_tool(name: str, payload: Any) -> str:
    text = name.lower()
    groups = [
        ("database", ["sql", "query", "database", "db", "postgres", "sqlite", "mysql"]),
        ("file", ["file", "read", "write", "open", "path", "dir", "folder", "glob", "stat", "mkdir", "remove", "delete"]),
        ("network", ["http", "request", "fetch", "url", "web", "download", "upload", "browse", "search", "scrape"]),
        ("shell", ["shell", "bash", "cmd", "command", "exec", "subprocess", "terminal", "powershell"]),
        ("communication", ["email", "mail", "send", "slack", "message", "notify", "sms", "post"]),
        ("supply-chain", ["publish", "deploy", "release", "docker", "npm", "package", "registry", "commit", "push"]),
    ]
    for category, keywords in groups:
        if any(keyword in text for keyword in keywords):
            return category
    values = _flatten_strings(payload)
    if any(_looks_like_url(value) for value in values):
        return "network"
    return "unknown"


def collect_signals(category: str, name: str, payload: Any, state: dict[str, Any]) -> list[tuple[str, str, str]]:
    values = _flatten_strings(payload)
    joined = "\n".join(values)
    signals: list[tuple[str, str, str]] = []

    for value in values:
        normalized = value.replace("\\", "/")
        if "../" in normalized or normalized.startswith("~/") or "%2e%2e" in normalized.lower():
            signals.append(("path_traversal", "HIGH", f"Path traversal pattern in {value[:80]!r}"))
        for marker in ("/.env", "/.ssh/", "/.aws/", "/.kube/", "/etc/passwd", "/etc/shadow", "/proc/", "id_rsa", "id_ed25519"):
            if marker in normalized:
                signals.append(("sensitive_path", "CRITICAL", f"Sensitive path access: {marker}"))
                break

    if _contains_secret(joined):
        severity = "CRITICAL" if category in {"network", "communication"} else "HIGH"
        signals.append(("secret_in_arguments", severity, "Tool arguments contain credential-like material"))

    if category == "database":
        if re.search(r"\b(drop|truncate|alter\s+table|delete\s+from)\b", joined, re.I):
            signals.append(("destructive_sql", "HIGH", "Destructive SQL keyword in tool arguments"))
        if re.search(r"\b(union\s+select|or\s+['\"]?1['\"]?\s*=\s*['\"]?1|--|;\s*drop)\b", joined, re.I):
            signals.append(("sql_injection", "HIGH", "SQL injection pattern in tool arguments"))

    if category == "shell":
        if re.search(r"[;&|`]\s*|\$\(|\|\|", joined):
            signals.append(("shell_metacharacters", "HIGH", "Shell metacharacters in command arguments"))
        if re.search(r"\brm\s+-rf\s+(/|~|\*)", joined):
            signals.append(("destructive_shell", "CRITICAL", "Destructive shell command pattern"))
        if re.search(r"\b(curl|wget|nc|ncat)\b.*https?://", joined, re.I):
            signals.append(("shell_network_egress", "HIGH", "Shell command performs network egress"))

    if category in {"network", "communication"}:
        if _large_payload(payload):
            signals.append(("large_external_payload", "MEDIUM", "Large payload sent through external-facing tool"))
        if _contains_pii(joined):
            signals.append(("pii_external_egress", "HIGH", "PII-like data in external-facing tool arguments"))

    if category == "supply-chain" and re.search(r"\b(publish|deploy|release|push)\b", name, re.I):
        signals.append(("supply_chain_side_effect", "MEDIUM", "Supply-chain or deployment side effect"))

    return signals


def _flatten_strings(value: Any, limit: int = 200) -> list[str]:
    out: list[str] = []

    def visit(item: Any) -> None:
        if len(out) >= limit:
            return
        if item is None:
            return
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (int, float, bool)):
            out.append(str(item))
        elif isinstance(item, dict):
            for key, val in item.items():
                out.append(str(key))
                visit(val)
        elif isinstance(item, (list, tuple, set)):
            for val in item:
                visit(val)
        else:
            out.append(str(item))

    visit(value)
    return out


def _looks_like_url(value: str) -> bool:
    return bool(re.search(r"https?://|ftp://", value, re.I))


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


def _large_payload(value: Any) -> bool:
    try:
        return len(str(value)) > 10_000
    except Exception:
        return False
