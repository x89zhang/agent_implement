"""Two-stage JANUS inference over an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BOXED = re.compile(r"\\boxed\{\s*(safe|potential_unsafe|unsafe)\s*\}", re.I)
_LABEL = re.compile(r"^\s*(safe|potential_unsafe|unsafe)\s*$", re.I)


@dataclass(frozen=True)
class JanusVerdict:
    label: str
    predicted_future: str
    reasoning: str
    usage: dict[str, int]


class VanguardClient:
    def __init__(self, settings: Any, *, api_key: str = "") -> None:
        self.settings = settings
        self.api_key = api_key

    def judge(self, instruction: str, observed: list[dict[str, Any]]) -> JanusVerdict:
        history = _format_observed(instruction, observed, self.settings.max_input_chars)
        summary_prompt = _template("predict_future_summary.prompt")
        judge_prompt = _template("judge_user_with_predicted_summary.prompt")
        try:
            summary, first_usage = self._complete(
                _truncate_middle(_render(summary_prompt, instruction, history),
                                 self.settings.max_input_chars),
                max_tokens=getattr(self.settings, "summary_max_tokens", 256),
                temperature=getattr(self.settings, "summary_temperature", None),
            )
            summary = summary.strip()
        except Exception as exc:
            # Upstream continues to the judge with an explicit predictor error.
            summary, first_usage = f"(summary_request_error: {exc})", {}
        answer, second_usage = self._complete(
            _truncate_middle(_render(judge_prompt, instruction, history, summary),
                             self.settings.max_input_chars)
        )
        label = parse_label(answer)
        usage = {
            key: int(first_usage.get(key, 0)) + int(second_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return JanusVerdict(label, summary, answer.strip(), usage)

    def _complete(self, prompt: str, *, max_tokens: int | None = None,
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
                "messages": [{"role": "user", "content": prompt}],
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
            message = payload["choices"][0]["message"]["content"]
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


def parse_label(answer: str) -> str:
    matches = _BOXED.findall(answer)
    if matches:
        return matches[-1].lower()
    tail = answer.split("</think>")[-1]
    match = _LABEL.fullmatch(tail)
    if match:
        return match.group(1).lower()
    raise ValueError("JANUS judge response has no valid safety label")


def _template(name: str) -> str:
    return (Path(__file__).parent / name).read_text(encoding="utf-8")


def _render(template: str, instruction: str, history: str, summary: str = "") -> str:
    return (template.replace("{instruction}", instruction)
            .replace("{tpast}", history)
            .replace("{future_summary}", summary))


def _truncate_middle(value: str, maximum: int) -> str:
    if maximum <= 0 or len(value) <= maximum:
        return value
    marker = "\n... [truncated {n} chars] ...\n"
    keep = max(0, maximum - len(marker.format(n=len(value))))
    head = keep // 2
    return value[:head] + marker.format(n=len(value) - keep) + (value[-(keep-head):] if keep-head else "")


def _format_observed(instruction: str, observed: list[dict[str, Any]], maximum: int) -> str:
    # The upstream formatter excludes the first user instruction, caps large
    # steps individually, and drops only complete old steps.
    steps = list(enumerate(observed))
    for position, (_, step) in enumerate(steps):
        if step.get("role") == "user" and str(step.get("content") or "").strip():
            del steps[position]
            break
    if not steps:
        return "(no steps yet)"
    per_step_cap = max(512, maximum // 3)
    serialized = []
    for index, step in steps:
        content = step.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        item = {"index": index, "role": step.get("role", ""),
                "content": _truncate_middle(content, per_step_cap)}
        if step.get("tool_calls"):
            item["tool_calls"] = step["tool_calls"]
        extra = step.get("extra") or {}
        if isinstance(extra, dict) and extra.get("tool"):
            item["tool_name"] = extra["tool"]
        serialized.append(json.dumps(item, ensure_ascii=False, indent=2, default=str))
    total = sum(map(len, serialized)) + 2 + 2 * max(0, len(serialized) - 1)
    if total <= maximum:
        return "[\n" + ",\n".join(serialized) + "\n]"
    kept_rev = []
    running = 2
    for item in reversed(serialized):
        cost = len(item) + 2
        if running + cost + 80 > maximum and kept_rev:
            break
        kept_rev.append(item)
        running += cost
    kept = list(reversed(kept_rev))
    dropped = len(serialized) - len(kept)
    marker = json.dumps({"truncated_steps": dropped}, ensure_ascii=False, indent=2)
    return "[\n" + marker + (",\n" + ",\n".join(kept) if kept else "") + "\n]"
