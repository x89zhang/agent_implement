from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LLMConfig


@dataclass
class LLMResponse:
    content: str


class LLMAdapter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

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
        return LLMResponse(content=result.content)

    def get_lc_chat_model(self) -> Any:
        self._lazy_init()
        if self._client == "mock":
            raise RuntimeError("mock provider does not support LangChain ReAct agent")
        return self._client
