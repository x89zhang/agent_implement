from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentDoGCompletion:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleAgentDoGClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> AgentDoGCompletion:
        if not self.base_url:
            raise ValueError("agentdog.base_url is required when AgentDoG is enabled")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            _chat_completions_url(self.base_url),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"AgentDoG endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"AgentDoG endpoint request failed: {exc.reason}"
            ) from exc

        decoded: Any = json.loads(body)
        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "AgentDoG endpoint returned an invalid chat response"
            ) from exc
        if isinstance(content, list):
            rendered = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            rendered = str(content)
        usage = decoded.get("usage") if isinstance(decoded, dict) else None
        return AgentDoGCompletion(
            content=rendered,
            usage=dict(usage) if isinstance(usage, dict) else {},
        )


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"
