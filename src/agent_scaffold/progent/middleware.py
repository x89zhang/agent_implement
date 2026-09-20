from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig, LLMConfig
from ..llm import LLMAdapter
from ..middleware import Middleware, ResultDecision, ToolDecision
from .runtime import ProgentRuntime, RuntimeResult
from .tools import normalize_tool_definitions, tool_definitions_from_config


class ProgentMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.settings = cfg.progent

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> Any:
        error = self._ensure_policy(state)
        if error:
            return self._model_failure(state, messages, error)
        from ..middleware import ModelDecision

        return ModelDecision(messages=messages)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        error = self._ensure_policy(state)
        if error:
            allowed = not self.settings.fail_closed
            event = self._event(
                allowed=allowed,
                enforced=not allowed,
                phase="before_tool",
                reason=error,
                tool=name,
                source="error",
            )
            state["_last_progent_decision"] = event
            return ToolDecision(allowed, "" if allowed else error)

        runtime = self._runtime(state)
        result = runtime.check(name, payload)
        denied = not result.allowed
        enforced = denied and self.settings.mode == "block"
        event = self._event(
            allowed=result.allowed,
            enforced=enforced,
            phase="before_tool",
            reason=result.reason,
            tool=name,
            source="policy",
        )
        state["_progent_policy"] = copy.deepcopy(result.policy)
        state["_last_progent_decision"] = event
        state.setdefault("progent_events", []).append(event)
        self._update_harness(state, event)
        if denied and self.settings.mode == "warn":
            state["_progent_warning"] = result.reason
        return ToolDecision(not enforced, result.reason if enforced else "")

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_progent_warning", "")
        return [f"Progent policy warning: {warning}"] if warning else []

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        if not self.settings.update_after_tool or failed:
            return ResultDecision(result=result)
        runtime = self._runtime(state)
        update = runtime.update(
            name,
            payload,
            result,
            only_allow_narrow=self.settings.only_allow_narrow,
        )
        state["_progent_policy"] = copy.deepcopy(update.policy)
        event = self._event(
            allowed=update.allowed,
            enforced=False,
            phase="after_tool",
            reason=update.reason,
            tool=name,
            source="policy_update",
        )
        state["_last_progent_decision"] = event
        state.setdefault("progent_events", []).append(event)
        self._add_usage(state, update.usage)
        self._update_harness(state, event)
        if not update.allowed and self.settings.fail_closed:
            return ResultDecision(False, update.reason, result, "progent_update_error")
        return ResultDecision(result=result)

    def _ensure_policy(self, state: dict[str, Any]) -> str:
        if state.get("_progent_initialized"):
            return str(state.get("_progent_init_error") or "")
        state["_progent_initialized"] = True
        started = time.time()
        runtime = self._runtime(state, use_config_policy=True)
        generated = RuntimeResult(True, policy=copy.deepcopy(runtime.policy))
        if self.settings.generate_policy:
            generated = runtime.generate()
        state["_progent_policy"] = copy.deepcopy(generated.policy)
        state["_progent_init_error"] = "" if generated.allowed else generated.reason
        self._add_usage(state, generated.usage)
        event = {
            "step": "progent_policy_generate",
            "timestamp": started,
            "latency_ms": int((time.time() - started) * 1000),
            "input": {
                "task": self._query(state),
                "tool_count": len(self._tool_definitions(state)),
            },
            "output": {
                "enabled": True,
                "status": "generated" if generated.allowed else "failed",
                "reason": generated.reason,
                "policy_tool_count": len(generated.policy or {}),
            },
            "usage": dict(generated.usage),
        }
        state.setdefault("trace", []).append(event)
        self._persist_policy(state, event)
        self._update_harness(
            state,
            self._event(
                allowed=generated.allowed,
                enforced=not generated.allowed and self.settings.fail_closed,
                phase="initialize",
                reason=generated.reason,
                source="generated" if self.settings.generate_policy else "configured",
            ),
        )
        return str(state["_progent_init_error"])

    def _runtime(
        self, state: dict[str, Any], *, use_config_policy: bool = False
    ) -> ProgentRuntime:
        policy = (
            _configured_policy(self.settings)
            if use_config_policy
            else copy.deepcopy(state.get("_progent_policy"))
        )
        llm = LLMAdapter(_policy_llm_config(self.cfg))

        def complete(system: str, user: str, temperature: float) -> tuple[str, Any]:
            llm.config.temperature = temperature
            response = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
            return response.content, response.usage

        return ProgentRuntime(
            tools=normalize_tool_definitions(self._tool_definitions(state)),
            query=self._query(state),
            policy=policy,
            completion=complete,
        )

    def _tool_definitions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        tools = state.get("_progent_tools")
        return copy.deepcopy(tools) if isinstance(tools, list) else tool_definitions_from_config(self.cfg)

    def _query(self, state: dict[str, Any]) -> str:
        configured = state.get("_progent_user_request")
        if configured:
            return str(configured)
        for message in state.get("messages", []):
            if message.get("role") == "user" and message.get("content"):
                return str(message["content"])
        return self.cfg.agent.task

    def _model_failure(
        self,
        state: dict[str, Any],
        messages: list[dict[str, Any]],
        error: str,
    ) -> Any:
        from ..middleware import ModelDecision

        event = self._event(
            allowed=not self.settings.fail_closed,
            enforced=self.settings.fail_closed,
            phase="model_input",
            reason=error,
            source="error",
        )
        state["_last_progent_decision"] = event
        if self.settings.fail_closed:
            return ModelDecision(
                allowed=False,
                reason=error,
                messages=messages,
                content="Input blocked because Progent policy initialization failed.",
                decision_type="progent_error",
                terminate=True,
            )
        return ModelDecision(messages=messages)

    def _event(self, **values: Any) -> dict[str, Any]:
        return {"mode": self.settings.mode, **values}

    def _update_harness(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        harness = state.setdefault("harness", {}).setdefault("progent", {})
        harness.update(
            {
                "enabled": True,
                "mode": self.settings.mode,
                "status": "failed" if event.get("source") == "error" else "active",
                "policy_tool_count": len(state.get("_progent_policy") or {}),
                "event_count": len(state.get("progent_events") or []),
                "last_decision": event,
            }
        )

    def _add_usage(self, state: dict[str, Any], usage: dict[str, Any]) -> None:
        stats = state.setdefault("trace_stats", {})
        if any(int(value or 0) for value in usage.values()):
            stats["api_calls"] = int(stats.get("api_calls", 0)) + 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            stats[key] = int(stats.get(key, 0)) + int(usage.get(key) or 0)

    def _persist_policy(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        persist = state.get("_trace_persist") or {}
        run_dir = persist.get("run_dir")
        if not run_dir:
            return
        path = Path(str(run_dir)) / "progent_policy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"generation": event, "policy": state.get("_progent_policy")},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


def _configured_policy(settings: Any) -> dict[str, Any]:
    policy = copy.deepcopy(settings.policy)
    for name in settings.always_allow_tools:
        policy.setdefault(name, []).insert(0, (1, 0, {}, 0))
    for name in settings.always_block_tools:
        policy.setdefault(name, []).insert(0, (1, 1, {}, 0))
    return policy


def _policy_llm_config(cfg: AppConfig) -> LLMConfig:
    settings = cfg.progent
    base = cfg.llm
    return LLMConfig(
        provider=settings.provider or base.provider,
        model=settings.model or base.model,
        temperature=(
            settings.temperature if settings.temperature is not None else base.temperature
        ),
        base_url=settings.base_url or base.base_url,
        api_key=settings.api_key or base.api_key,
        api_key_env=settings.api_key_env or base.api_key_env,
        request_timeout=(
            settings.request_timeout
            if settings.request_timeout is not None
            else base.request_timeout
        ),
    )
