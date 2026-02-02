from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LLMConfig


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, Any] | None = None


class LLMAdapter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None
        self._tokenizer = None

    def _lazy_init(self) -> None:
        if self._client is not None:
            return
        provider = self.config.provider.lower()
        if provider == "mock":
            self._client = "mock"
            return
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime import
                raise RuntimeError("Missing dependency: langchain_openai") from exc
            kwargs: dict[str, Any] = {}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            self._client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                **kwargs,
            )
            return
        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime import
                raise RuntimeError("Missing dependency: langchain_anthropic") from exc
            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            self._client = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                **kwargs,
            )
            return
        if provider == "vllm_openai":
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime import
                raise RuntimeError("Missing dependency: langchain_openai") from exc
            if not self.config.base_url:
                raise ValueError("vllm_openai requires llm.base_url")
            kwargs = {"base_url": self.config.base_url}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            else:
                # vLLM servers often do not require auth, but the OpenAI client enforces api_key.
                kwargs["api_key"] = "local-vllm"
            self._client = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                **kwargs,
            )
            return
        raise ValueError(f"Unknown provider: {self.config.provider}")

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        self._lazy_init()
        if self._client == "mock":
            last = messages[-1]["content"] if messages else ""
            return LLMResponse(content=f"MOCK: {last}")

        # LangChain model: expects list of BaseMessage
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime import
            raise RuntimeError("Missing dependency: langchain_core") from exc

        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        result = self._client.invoke(lc_messages)
        usage = _extract_usage(result)
        return LLMResponse(content=result.content, usage=usage)

    def get_lc_chat_model(self) -> Any:
        self._lazy_init()
        if self._client == "mock":
            raise RuntimeError("mock provider does not support LangChain ReAct agent")
        return self._client

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.config.model)
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)


def _load_tokenizer(model_name: str) -> Any | None:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _extract_usage(result: Any) -> dict[str, Any] | None:
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        total = usage.get("total_tokens")
        if prompt is not None or completion is not None or total is not None:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "source": "reported",
            }
    metadata = getattr(result, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(token_usage, dict):
            return {
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
                "source": "reported",
            }
    return None
