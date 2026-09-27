"""Two-stage JANUS/Vanguard inference over an OpenAI-compatible endpoint.

Prompt rendering, the input budget and the label parser are ported from
upstream ``eval_framework_offline/eval_framework/guards/vllm_guard.py``
(``_render_template``, ``_format_observed``, ``_default_format_history``,
``_step_to_dict``, ``fit_messages_to_char_budget``, ``_default_parse``).
A step is a dict with upstream ``Step`` fields: ``index``, ``role``,
``content``, ``tool_calls`` ({name, arguments, call_id}), ``tool_name``,
``tool_call_id``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class JanusVerdict:
    label: str
    predicted_future: str
    reasoning: str
    usage: dict[str, int]
    parse_error: str = ""


# --- Ported from upstream vllm_guard.py ---------------------------------

def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep head + tail of a long string, replace the middle with a marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker_tmpl = "\n... [truncated {n} chars] ...\n"
    # Rough estimate of marker length after formatting; leave room for it.
    marker_slack = len(marker_tmpl.format(n=len(text)))
    keep = max(0, max_chars - marker_slack)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker_tmpl.format(n=len(text) - keep) + (text[-tail:] if tail else "")


def fit_messages_to_char_budget(
    messages: list[dict[str, str]],
    budget: int,
) -> list[dict[str, str]]:
    """If the total content char length across all messages exceeds `budget`,
    proportionally mid-truncate each message's content until the total fits.
    Deterministic, model-agnostic, no tokenizer required. Iterates up to 3
    times with a 5% cushion to absorb per-message truncation-marker overhead
    and the 200-char floor on small messages."""
    if not messages:
        return messages
    msgs = [dict(m) for m in messages]
    for _ in range(3):
        total = sum(len(m.get("content", "") or "") for m in msgs)
        if total <= budget:
            return msgs
        scale = (budget / total) * 0.95
        for m in msgs:
            c = m.get("content") or ""
            target = max(200, int(len(c) * scale))
            if target < len(c):
                m["content"] = _truncate_middle(c, target)
    return msgs


def _step_to_dict(s: dict[str, Any]) -> dict[str, Any]:
    content = s.get("content") or ""
    item: dict[str, Any] = {"index": s["index"], "role": s["role"], "content": content}
    if s.get("tool_calls"):
        item["tool_calls"] = [
            {"name": tc["name"], "arguments": tc["arguments"], "id": tc.get("call_id")}
            for tc in s["tool_calls"]
        ]
    if s.get("tool_name"):
        item["tool_name"] = s["tool_name"]
    if s.get("tool_call_id"):
        item["tool_call_id"] = s["tool_call_id"]
    return item


def _default_format_history(history: Sequence[dict[str, Any]]) -> str:
    payload = [_step_to_dict(s) for s in history]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_instruction(history: Sequence[dict[str, Any]]) -> str:
    for s in history:
        if s["role"] == "user" and (s.get("content") or "").strip():
            return s["content"]
    return ""


def _observed_tail_steps(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything after the first user instruction."""
    first_user_seen = False
    tail: list[dict[str, Any]] = []
    for s in history:
        if not first_user_seen and s["role"] == "user" and (s.get("content") or "").strip():
            first_user_seen = True
            continue
        tail.append(s)
    return tail


def _format_observed(history: Sequence[dict[str, Any]]) -> str:
    # Upstream judge()/summary call _format_observed(history) without a
    # per-step budget; only the whole prompt is fitted afterwards.
    steps = _observed_tail_steps(history)
    if not steps:
        return "(no steps yet)"
    return _default_format_history(steps)


def _render_template(
    template: str,
    history: Sequence[dict[str, Any]],
    *,
    future_summary: Optional[str] = None,
) -> str:
    instruction = _extract_instruction(history)
    observed = _format_observed(history)
    # Use str.replace so literal braces in the template (e.g. `\boxed{label}`)
    # don't trip `str.format`.
    return (
        template
        .replace("{instruction}", instruction)
        .replace("{tpast}", observed)
        .replace("{future_summary}", future_summary or "")
        .replace("{history_json}", _default_format_history(history))
    )


_VALID_LABELS = {"safe", "potential_unsafe", "unsafe"}
_LABEL_ALIASES = {
    "benign": "safe",
    "ok": "safe",
    "allow": "safe",
    "suspicious": "potential_unsafe",
    "uncertain": "potential_unsafe",
    "caution": "potential_unsafe",
    "warn": "potential_unsafe",
    "potentially_unsafe": "potential_unsafe",
    "potentialunsafe": "potential_unsafe",
    "potential-unsafe": "potential_unsafe",
    "controversial": "potential_unsafe",
    "malicious": "unsafe",
    "harmful": "unsafe",
    "block": "unsafe",
    "deny": "unsafe",
}


def _normalise_label(raw: str) -> Optional[str]:
    key = (raw or "").strip().lower().replace(" ", "_").strip("`'\"")
    if key in _VALID_LABELS:
        return key
    return _LABEL_ALIASES.get(key)


_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z_\- ]+?)\s*\}")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_label(text: str) -> Optional[str]:
    """Label of upstream ``_default_parse``: last valid ``\\boxed{}``, else
    the first label token after ``</think>``; None when there is none."""
    label: Optional[str] = None
    for m in _BOXED_RE.finditer(text or ""):
        cand = _normalise_label(m.group(1))
        if cand is not None:
            label = cand  # last boxed wins, in case the model emits multiple

    if label is None:
        tail = (text or "").split("</think>")[-1]
        for token in re.findall(r"[A-Za-z_\-]+", tail):
            cand = _normalise_label(token)
            if cand is not None:
                label = cand
                break
    return label


# --- Client ---------------------------------------------------------------

def _template(name: str) -> str:
    return (Path(__file__).parent / name).read_text(encoding="utf-8")


class VanguardClient:
    def __init__(self, settings: Any, *, api_key: str = "") -> None:
        self.settings = settings
        self.api_key = api_key

    def judge(self, history: Sequence[dict[str, Any]]) -> JanusVerdict:
        """Judge the last step of ``history`` (predicted-summary mode)."""
        budget = self.settings.max_input_chars
        summary_prompt = _render_template(_template("predict_future_summary.prompt"), history)
        try:
            summary, first_usage = self._complete(
                fit_messages_to_char_budget([{"role": "user", "content": summary_prompt}], budget),
                max_tokens=getattr(self.settings, "summary_max_tokens", 256),
                temperature=getattr(self.settings, "summary_temperature", None),
            )
            summary = summary.strip()
        except Exception as exc:
            # Upstream continues to the judge with an explicit predictor error.
            summary, first_usage = f"(summary_request_error: {exc})", {}
        judge_prompt = _render_template(
            _template("judge_user_with_predicted_summary.prompt"), history, future_summary=summary,
        )
        answer, second_usage = self._complete(
            fit_messages_to_char_budget([{"role": "user", "content": judge_prompt}], budget)
        )
        label = parse_label(answer)
        usage = {
            key: int(first_usage.get(key, 0)) + int(second_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        think = _THINK_RE.search(answer or "")
        return JanusVerdict(
            label or "",
            summary,
            think.group(1).strip() if think else answer.strip(),
            usage,
            "" if label else f"parse_error: no label in {answer[:160]!r}",
        )

    def _complete(self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
                  temperature: float | None = None) -> tuple[str, dict[str, int]]:
        settings = self.settings
        if not settings.base_url or not settings.model:
            raise ValueError("janus.base_url and janus.model are required")
        endpoint = settings.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            endpoint += "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "model": settings.model,
                "messages": messages,
                "temperature": settings.temperature if temperature is None else temperature,
                "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
                **settings.extra_body,
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"JANUS endpoint returned HTTP {exc.code}") from exc
        try:
            message = payload["choices"][0]["message"]["content"] or ""
            if isinstance(message, list):
                message = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in message
                )
            if not isinstance(message, str):
                raise TypeError("content is not text")
            usage = payload.get("usage") or {}
            return message, usage if isinstance(usage, dict) else {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("JANUS endpoint returned an invalid chat response") from exc
