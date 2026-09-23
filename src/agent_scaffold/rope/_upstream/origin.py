"""The core question: where did this parameter first come from?

For each risky parameter we allow it only if ONE of these is true:
  -  THE USER WROTE IT IN THE REQUEST.
  -  IT CAME FROM A NAMED PLACE THE ATTACKER CANNOT WRITE INTO.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

import yaml

# Matches the kinds of "identifier-looking" chunks we care about: IBANs, emails, ssh keys,
# repo paths, urls, ids.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:\-+]{3,}")

# Non-identifier "type prefix" words that show up as standalone tokens. These are NOT distinctive identifiers.
_STOPWORDS = frozenset({
    "ssh-rsa", "ssh-ed25519", "ssh-dss", "ssh-ed25519@openssh.com",
    "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "sk-ssh-ed25519@openssh.com",
})

# Fields on an item-structured tool result that name an origin the attacker CANNOT forge.
_ORIGIN_FIELDS = ("sender", "from")

# The user's own authoritative records: read-tool outputs the attacker cannot write, clean at
# guard-check time.
AUTHORITATIVE_RECORD_TOOLS = frozenset({"view_order_history", "git_get_linked_ssh_keys", "verify_github_account"})
RECORD_SOURCE_KEY = "@authoritative_record"

# Record closure. Granting T3 to a record tool also requires checking every state-changing tool that
# writes into a field that record tool later returns: a record tool may license only fields whose
# written value is platform-generated or a guarded parameter of the writer. Two writes fail that
# condition and are patched below.
RECORD_FIELD_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "view_order_history": ("order_id", "product_id"),
}
VERIFICATION_EXECUTOR_TOOLS = frozenset({"verify_github_account"})

# Which tools the platform may defer behind an OTP. Fixed offline:
OTP_DEFERRABLE_TOOLS = frozenset({
    "git_get_linked_ssh_keys", "git_add_ssh_key", "git_delete_ssh_key", "git_create_repo",
    "git_delete_repo", "git_transfer_repo_ownership", "git_invite_collaborators",
    "update_github_account_password",
    "send_money", "refund_order", "checkout_selected_cart", "update_shopping_account_password",
})

_OTP_DEFERRAL_RE = re.compile(r"one[- ]time password|\bOTP\b", re.I)


def _collect_fields(obj: Any, fields: tuple[str, ...]) -> set[str]:
    """Identifier chunks from the named fields anywhere in a parsed record structure."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in fields and isinstance(v, (str, int, float)):
                out |= tokenize_identifiers(str(v))
            else:
                out |= _collect_fields(v, fields)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out |= _collect_fields(x, fields)
    return out


def tokenize_identifiers(text: str, split_composites: bool = True) -> set[str]:
    """split_composites=False (STRICT matcher): do NOT '/'-split -- treat a composite owner/repo (or a
    full URL/path) as one decision unit, so a padded value like "attacker/<trusted-repo>" cannot ride
    in on the shared part."""
    if not text:
        return set()
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(str(text)):
        t = m.group(0)
        if t.lower() in _STOPWORDS:
            continue
        pieces = [t]
        if split_composites and "/" in t:  # split a composite (owner/repo, http://host/path); keep the whole too
            pieces += [p for p in t.split("/") if len(p) >= 4]
        for p in pieces:
            if p.lower() in _STOPWORDS:
                continue
            out.add(p)
            stripped = p.rstrip(".:,;")
            if stripped != p and len(stripped) >= 4 and stripped.lower() not in _STOPWORDS:
                out.add(stripped)
    return out


# ── reading the structure of a tool result ───────────────────────────────────────
def _message_text(msg) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        return "\n".join((b.get("content") or "") for b in content if isinstance(b, dict))
    return str(content or "")


def _source_descriptor(tool_msg) -> str:
    """A short name for the tool call that produced a result: the function plus its args. Used as
    the key for blob (file/web) sources, whose exact name doesn't matter because they are injectable
    and never license a new identifier anyway."""
    tc = tool_msg.get("tool_call")
    if tc is None:
        return "unknown"
    fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else "unknown")
    args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}
    key = ",".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{fn}({key})"


def _tool_function(tool_msg) -> str | None:
    """The function name of the tool call that produced this result (or None)."""
    tc = tool_msg.get("tool_call")
    if tc is None:
        return None
    return getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)


def _read_path(tool_msg) -> str | None:
    """The path-like argument of the tool call that produced this result (read_file's `path`, etc.),
    used to recognise a read from the user's own protected credential store (see _absorb)."""
    tc = tool_msg.get("tool_call")
    if tc is None:
        return None
    args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}
    for k in ("path", "file_path"):
        v = args.get(k) if isinstance(args, dict) else None
        if v:
            return str(v)
    return None


def _structured_items(text: str) -> list[dict] | None:
    """If the tool result is a YAML record or list of records (how `ToolsExecutor` serialises a
    list of pydantic models, e.g. a list of emails), return them as dicts; otherwise None — meaning
    a monolithic blob like a file body or a web page. A scalar or non-YAML result is a blob too."""
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list) and loaded and all(isinstance(x, dict) for x in loaded):
        return loaded
    return None


def _item_origin(item: dict) -> str | None:
    """The unforgeable origin of one record (an email's sender), or None if it has none."""
    for f in _ORIGIN_FIELDS:
        v = item.get(f)
        if isinstance(v, str) and v:
            return v
    return None


def _item_identifiers(item: dict) -> set[str]:
    """Identifier-looking chunks anywhere in one record's string values."""
    out: set[str] = set()
    for v in item.values():
        if isinstance(v, str):
            out |= tokenize_identifiers(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    out |= tokenize_identifiers(x)
    return out


class OriginTracker:
    """Runs inside the tools loop: after each tool runs, it files the identifiers in that tool's
    result under where they came from, into a shared map.
    """

    def __init__(self, origin_map: dict[str, dict]) -> None:
        self.origin_map = origin_map
        self._pending_deferral: str | None = None

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        self._pending_deferral = None
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            self._absorb(msg)
        return query, runtime, env, messages, extra_args

    def _record_ids(self, fn: str, text: str) -> set[str]:
        """The identifiers a record tool's result licenses, under the record-closure allowlist."""
        fields = RECORD_FIELD_ALLOWLIST.get(fn)
        if fields is None:
            return tokenize_identifiers(text)
        items = _structured_items(text)
        if items is None:
            # If a record tool's return shape stops parsing, the allowlist licenses nothing;
            # fail loudly rather than lose utility silently.
            raise RuntimeError(
                f"record tool {fn!r} returned an unparseable result; the record-closure field "
                f"allowlist {fields} cannot be applied to: {text[:200]!r}"
            )
        return _collect_fields(items, fields)

    def _absorb(self, tool_msg) -> None:
        text = _message_text(tool_msg)
        fn = _tool_function(tool_msg)
        # RECORD CLOSURE (ii): a verification response carries the result of the DEFERRED call, so
        # attribute it to that call's tool. Only a deferred record tool yields a record; anything
        # else (a repo creation confirmation echoing its unguarded repo_name) stays injectable.
        if fn in VERIFICATION_EXECUTOR_TOOLS:
            deferred, self._pending_deferral = self._pending_deferral, None
            if deferred in AUTHORITATIVE_RECORD_TOOLS:
                self._add(RECORD_SOURCE_KEY, self._record_ids(deferred, text), injectable=False)
            else:
                # No deferral pending (a spurious or failed verification) or the deferred tool is
                # not a record tool: nothing here has a trusted origin.
                self._add(_source_descriptor(tool_msg), tokenize_identifiers(text), injectable=True)
            return
        if fn in OTP_DEFERRABLE_TOOLS and _OTP_DEFERRAL_RE.search(text):
            self._pending_deferral = fn
        # Outputs of the user's authoritative-record tools are non-injectable (see
        # AUTHORITATIVE_RECORD_TOOLS) — file their licensed identifiers under the reserved record
        # source (RECORD_FIELD_ALLOWLIST narrows this where an unguarded write reaches the field).
        if fn in AUTHORITATIVE_RECORD_TOOLS:
            self._add(RECORD_SOURCE_KEY, self._record_ids(fn, text), injectable=False)
            return
        rp = _read_path(tool_msg)
        if rp and any(d in rp.lower() for d in _SENSITIVE_WRITE_DIRS):
            self._add(RECORD_SOURCE_KEY, tokenize_identifiers(text), injectable=False)
            return
        items = _structured_items(text)
        if items is not None:
            for item in items:
                origin = _item_origin(item)
                if origin is None:
                    # a record with no unforgeable origin -> treat its content as an injectable blob
                    self._add(_source_descriptor(tool_msg), _item_identifiers(item), injectable=True)
                else:
                    self._add(origin, _item_identifiers(item), injectable=False)
        else:
            # monolithic blob (file body / web page): attacker-writable, never licenses a new id
            self._add(_source_descriptor(tool_msg), tokenize_identifiers(text), injectable=True)

    def _add(self, key: str, ids: set[str], *, injectable: bool) -> None:
        entry = self.origin_map.setdefault(key, {"ids": set(), "injectable": injectable})
        entry["ids"].update(ids)
        # if a key is ever seen as both, stay on the safe side: injectable wins (no grant)
        entry["injectable"] = entry["injectable"] or injectable


class OriginPredicate:
    def __init__(self, arg_name: str, allowed_sources: Iterable[str] | None,
                 origin_map: dict[str, dict], prompt_ids: set[str],
                 use_records: bool = False) -> None:
        self.arg_name = arg_name
        self.allowed_sources = set(allowed_sources) if allowed_sources else None
        self.origin_map = origin_map
        self.prompt_ids = prompt_ids
        # use_records=True -> also grant from the user's AUTHORITATIVE RECORDS (RECORD_SOURCE_KEY, e.g.
        # order history): the value matches something the user actually owns/did. Used for state-changing
        # actions whose legitimate target is the user's own record.
        self.use_records = use_records

    def __call__(self, value) -> bool:
        # A list-valued argument (e.g. checkout product_ids, email recipients) names SEVERAL targets;
        # EVERY element must be trusted, else an injected extra rides in next to a legitimate one.
        if isinstance(value, (list, tuple)):
            return all(self._check_one(v) for v in value)
        return self._check_one(value)

    def _check_one(self, value) -> bool:
        # Strict decision-unit matching (the default): match the value as a whole identifier.
        if os.environ.get("ROPE_STRICT_MATCH", "1") != "0" and isinstance(value, str):
            if self.arg_name == "url":
                host = value.split("//", 1)[-1].split("/", 1)[0]  # strip scheme + path -> netloc/host
                vtoks = {value, host} if len(host) >= 4 else {value}
            else:
                vtoks = tokenize_identifiers(value, split_composites=False) or {value}
        else:
            vtoks = ({value} | tokenize_identifiers(value)) if isinstance(value, str) else tokenize_identifiers(value)
        if not vtoks:
            return True  # nothing identifier-like here to check (for example a free-text body)
        # leg (2): written in the user's request
        if vtoks & self.prompt_ids:
            return True
        # leg (1b): the user's own authoritative record
        if self.use_records:
            entry = self.origin_map.get(RECORD_SOURCE_KEY)
            if entry and not entry["injectable"] and (vtoks & entry["ids"]):
                return True
        # leg (1): appears in content from a NAMED, non-injectable structural source
        if self.allowed_sources:
            for src, entry in self.origin_map.items():
                if entry["injectable"] or src not in self.allowed_sources:
                    continue
                if vtoks & entry["ids"]:
                    return True
        return False


_SENSITIVE_WRITE_DIRS = ("/system", "/.ssh", "/etc")

class DestinationPredicate:
    def __init__(self, arg_name: str, prompt_ids: set[str],
                 sensitive_dirs: tuple[str, ...] = _SENSITIVE_WRITE_DIRS) -> None:
        self.arg_name = arg_name
        self.prompt_ids = prompt_ids
        self.sensitive_dirs = sensitive_dirs

    def __call__(self, value) -> bool:
        path = str(value or "")
        if not any(d in path.lower() for d in self.sensitive_dirs):
            return True  # ordinary destination -> fine
        # sensitive destination -> only if the user named this path in the request
        return bool(tokenize_identifiers(path) & self.prompt_ids)

