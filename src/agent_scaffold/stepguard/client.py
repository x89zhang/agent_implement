"""StepGuard inference through an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


_JUDGMENT = re.compile(
    r"<Judgment>\s*(safe|unsafe)\s*(?:</Judgment>|<\\Judgment>)",
    re.IGNORECASE,
)
_RISK_SOURCE = re.compile(r"<RiskSource>\s*([^<\n]+)", re.IGNORECASE)
_UNSAFE_STEP = re.compile(r"<UnsafeStep>\s*(None|\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class StepGuardVerdict:
    label: str
    confidence: float
    risk_source: str = ""
    unsafe_step: int | None = None
    usage: dict[str, int] = field(default_factory=dict)


def parse_verdict(content: str, usage: dict[str, Any] | None = None) -> StepGuardVerdict:
    # The released StepGuard parser uses the last complete judgment tag.
    labels = _JUDGMENT.findall(content)
    if not labels:
        raise ValueError("StepGuard response has no complete <Judgment> tag")
    source = _RISK_SOURCE.findall(content)
    steps = _UNSAFE_STEP.findall(content)
    step = int(steps[-1]) if steps and steps[-1].isdigit() else None
    return StepGuardVerdict(
        label=labels[-1].lower(),
        confidence=1.0,
        risk_source=source[-1].strip() if source else "",
        unsafe_step=step,
        usage={
            key: int((usage or {}).get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
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
