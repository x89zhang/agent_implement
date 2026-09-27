"""StepGuard inference through an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


_RISK_SOURCE = re.compile(r"<RiskSource>\s*([^<\n]+)", re.IGNORECASE)
_UNSAFE_STEP = re.compile(r"<UnsafeStep>\s*(None|\d+)", re.IGNORECASE)


def _parse_safe_unsafe(raw_text: str) -> tuple[int | None, float]:
    """Parse 'safe' / 'unsafe' output (AgentDoG trajectory format).

    Verbatim from upstream src/guardrail/prompts/base.py:45-104, the parser of
    the released StepGuard profiles (prompts/stepguard.py:206-207, 231-232).

    Supports multiple output conventions:
      - "safe" / "unsafe" (AgentDoG native)
      - "0" / "1" (TS-Guard, general instruct models)
      - "[OUTPUT] 0/1" (ShieldAgent native)
      - "Safety: Safe/Unsafe/Controversial" (Qwen3Guard native)
    """
    # Strip <think>/<Think> block if present (handles both </think> and <\Think> closing tags)
    stripped = raw_text.strip()
    for close_tag in ["</think>", "<\\think>", "</Think>", "<\\Think>"]:
        if close_tag in stripped:
            stripped = stripped.split(close_tag, 1)[-1].strip()
            break
    if not stripped:
        return None, 0.0
    first_line = stripped.splitlines()[0].strip().lower()

    # Exact match: "safe" / "unsafe"
    if first_line == "unsafe":
        return 1, 1.0
    if first_line == "safe":
        return 0, 1.0

    # <Judgment> safe/unsafe </Judgment> or <\Judgment> tag (predict prompt format)
    judgment_matches = list(re.finditer(
        r"<Judgment>\s*(unsafe|safe)\s*(?:</Judgment>|<\\Judgment>)",
        stripped,
        flags=re.IGNORECASE,
    ))
    if judgment_matches:
        # Prompts ask for a final judgment after analysis. Some models emit a
        # provisional judgment early and then correct it; use the final tag.
        judgment_match = judgment_matches[-1]
        return (1, 1.0) if judgment_match.group(1).lower() == "unsafe" else (0, 1.0)

    # Qwen3Guard: "Safety: Safe/Unsafe/Controversial"
    qwen3_match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", stripped, flags=re.IGNORECASE)
    if qwen3_match:
        val = qwen3_match.group(1).lower()
        if val == "unsafe" or val == "controversial":
            return 1, 0.9
        return 0, 0.9

    # ShieldAgent: "[OUTPUT] 0/1"
    output_match = re.search(r"\[OUTPUT\]\s*([01])", stripped)
    if output_match:
        return int(output_match.group(1)), 0.9

    # Fallback: search "unsafe"/"safe" anywhere in text
    match = re.search(r"\b(unsafe|safe)\b", stripped, flags=re.IGNORECASE)
    if match:
        return (1, 0.8) if match.group(1).lower() == "unsafe" else (0, 0.8)

    # Fallback: bare "0" or "1" (first line)
    if first_line in ("0", "1"):
        return int(first_line), 0.85

    return None, 0.0


@dataclass(frozen=True)
class StepGuardVerdict:
    label: str
    confidence: float
    risk_source: str = ""
    unsafe_step: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    parse_status: str = "parsed"
    raw: str = ""


def parse_verdict(content: str, usage: dict[str, Any] | None = None) -> StepGuardVerdict:
    label, confidence = _parse_safe_unsafe(content)
    parse_status = "parsed"
    if label is None:
        # Upstream guardrail.py:236-241: a malformed guard response must not
        # silently permit a live tool call; it is judged unsafe.
        label, confidence, parse_status = 1, 1.0, "parse_failed_closed"
    source = _RISK_SOURCE.findall(content)
    steps = _UNSAFE_STEP.findall(content)
    step = int(steps[-1]) if steps and steps[-1].isdigit() else None
    return StepGuardVerdict(
        label="unsafe" if label == 1 else "safe",
        confidence=float(confidence),
        risk_source=source[-1].strip() if source else "",
        unsafe_step=step,
        usage={
            key: int((usage or {}).get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        parse_status=parse_status,
        raw=content,
    )


class StepGuardClient:
    def __init__(self, settings: Any, *, api_key: str = "") -> None:
        self.settings = settings
        self.api_key = api_key

    def judge(self, prompt: str) -> StepGuardVerdict:
        settings = self.settings
        if not settings.base_url or not settings.model:
            raise ValueError("stepguard.base_url and stepguard.model are required")
        endpoint = settings.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            endpoint += "/chat/completions"
        payload = {
            "model": settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"StepGuard endpoint returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"StepGuard endpoint request failed: {exc.reason}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if not isinstance(content, str):
                raise TypeError("content is not text")
            return parse_verdict(content, body.get("usage"))
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("StepGuard endpoint returned an invalid chat response") from exc
