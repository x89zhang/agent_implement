from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml


def email_check_inbox(
    imap_host: str | None = None,
    imap_port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    mailbox: str = "INBOX",
    unseen_only: bool = True,
    limit: int = 10,
    use_ssl: bool | None = None,
) -> str:
    """
    Virtual inbox check for local testing.
    Signature is intentionally aligned with the real email tool.
    """
    if isinstance(imap_host, str) and ("=" in imap_host or imap_host.strip().startswith("{")):
        parsed = _parse_tool_input(imap_host)
        if parsed:
            username = parsed.get("username", username)
            mailbox = str(parsed.get("mailbox", mailbox))
            if "unseen_only" in parsed:
                unseen_only = _coerce_bool(parsed.get("unseen_only"))
            if "limit" in parsed:
                limit = int(parsed.get("limit"))

    cfg = _load_email_defaults()
    strict_target = bool(cfg.get("strict_target", False))
    if strict_target:
        username = str(cfg.get("username") or "virtual.user@example.local").strip()
        mailbox = str(cfg.get("mailbox") or "INBOX").upper()
    else:
        username = str(username or cfg.get("username") or "virtual.user@example.local").strip()
        mailbox = str(mailbox or "INBOX").upper()
    if mailbox != "INBOX":
        return json.dumps({"error": f"virtual mailbox '{mailbox}' is not supported"}, ensure_ascii=False, indent=2)

    state = _load_state()
    account = _ensure_account(state, username)
    inbox = account.get("inbox", [])
    items = [m for m in inbox if isinstance(m, dict)]
    if unseen_only:
        items = [m for m in items if not bool(m.get("seen", False))]
    items = list(reversed(items))[: max(1, int(limit))]

    response: list[dict[str, Any]] = []
    for msg in items:
        response.append(
            {
                "id": str(msg.get("id", "")),
                "from": msg.get("from", ""),
                "to": msg.get("to", username),
                "subject": msg.get("subject", ""),
                "date": msg.get("date", ""),
                "snippet": str(msg.get("body", ""))[:300],
            }
        )

    return json.dumps(response, ensure_ascii=False, indent=2)


def email_send(
    to: str | None = None,
    subject: str = "",
    body: str = "",
    cc: str = "",
    bcc: str = "",
    from_email: str | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    use_ssl: bool | None = None,
) -> str:
    """
    Virtual email send for local testing.
    Signature is intentionally aligned with the real email tool.
    """
    if isinstance(to, str) and ("=" in to or to.strip().startswith("{")):
        parsed = _parse_tool_input(to)
        if parsed:
            to = parsed.get("to", to)
            subject = str(parsed.get("subject", subject))
            body = str(parsed.get("body", body))
            cc = str(parsed.get("cc", cc))
            bcc = str(parsed.get("bcc", bcc))
            from_email = parsed.get("from_email", from_email)
            username = parsed.get("username", username)

    cfg = _load_email_defaults()
    strict_target = bool(cfg.get("strict_target", False))
    if strict_target:
        username = str(cfg.get("username") or "virtual.user@example.local").strip()
        from_email = str(cfg.get("from_email") or username).strip()
    else:
        username = str(username or cfg.get("username") or "virtual.user@example.local").strip()
        from_email = str(from_email or cfg.get("from_email") or username).strip()
    if not to or not subject or not body:
        return json.dumps({"error": "to, subject, and body are required"}, ensure_ascii=False, indent=2)

    state = _load_state()
    sender = _ensure_account(state, from_email)
    recipients = _flatten_recipients(to, cc, bcc)
    msg_id = _next_message_id(state)
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    base_message = {
        "id": msg_id,
        "from": from_email,
        "to": to,
        "cc": cc,
        "bcc": bcc,
        "subject": subject,
        "body": body,
        "date": now,
    }

    sender.setdefault("sent", []).append(base_message | {"seen": True})
    for addr in recipients:
        recipient = _ensure_account(state, addr)
        recipient.setdefault("inbox", []).append(base_message | {"to": addr, "seen": False})

    _save_state(state)
    return json.dumps(
        {
            "status": "sent",
            "to": to,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "virtual_message_id": msg_id,
        },
        ensure_ascii=False,
        indent=2,
    )


def virtual_email_reset() -> str:
    """
    Utility method to reset local virtual mailbox state.
    """
    state = _initial_state()
    _save_state(state)
    return json.dumps({"status": "reset", "accounts": list((state.get("accounts") or {}).keys())}, ensure_ascii=False)


def _load_email_defaults() -> dict[str, Any]:
    run_dir = Path.cwd().resolve()
    cfg_path = run_dir.parent / f"{run_dir.name}.yaml"
    if not cfg_path.exists():
        return {}
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    email_cfg = raw.get("email") or {}
    return email_cfg if isinstance(email_cfg, dict) else {}


def _state_path() -> Path:
    cfg = _load_email_defaults()
    run_dir = Path.cwd().resolve()
    relative = str(cfg.get("virtual_mailbox_file") or "virtual_mailbox.json").strip()
    target = (run_dir / relative).resolve()
    if not str(target).startswith(str(run_dir)):
        raise ValueError("virtual_mailbox_file must be under the run directory")
    return target


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        state = _initial_state()
        _save_state(state)
        return state
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("accounts"), dict):
        data["accounts"] = {}
    if not isinstance(data.get("next_id"), int):
        data["next_id"] = 1
    return data


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _initial_state() -> dict[str, Any]:
    cfg = _load_email_defaults()
    username = str(cfg.get("username") or "virtual.user@example.local").strip()
    accounts: dict[str, Any] = {}
    _ensure_account({"accounts": accounts, "next_id": 1}, username)
    seed_messages = cfg.get("virtual_seed_messages") or []
    state = {"accounts": accounts, "next_id": 1}
    if isinstance(seed_messages, list):
        for idx, item in enumerate(seed_messages, start=1):
            if not isinstance(item, dict):
                continue
            to = str(item.get("to") or username).strip()
            recipient = _ensure_account(state, to)
            msg = {
                "id": str(state["next_id"]),
                "from": str(item.get("from") or "noreply@example.local"),
                "to": to,
                "subject": str(item.get("subject") or f"Seed message {idx}"),
                "body": str(item.get("body") or ""),
                "date": str(item.get("date") or dt.datetime.now(dt.timezone.utc).isoformat()),
                "seen": bool(item.get("seen", False)),
            }
            recipient.setdefault("inbox", []).append(msg)
            state["next_id"] += 1
    return state


def _ensure_account(state: dict[str, Any], email: str) -> dict[str, Any]:
    accounts = state.setdefault("accounts", {})
    account = accounts.get(email)
    if not isinstance(account, dict):
        account = {"inbox": [], "sent": []}
        accounts[email] = account
    if not isinstance(account.get("inbox"), list):
        account["inbox"] = []
    if not isinstance(account.get("sent"), list):
        account["sent"] = []
    return account


def _next_message_id(state: dict[str, Any]) -> str:
    next_id = int(state.get("next_id") or 1)
    state["next_id"] = next_id + 1
    return str(next_id)


def _flatten_recipients(to: str, cc: str, bcc: str) -> list[str]:
    raw = ",".join([to or "", cc or "", bcc or ""])
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _parse_tool_input(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    if raw.startswith("{") and raw.endswith("}"):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
        except Exception:
            pass
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" in line and "=" not in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result
