from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig, LLMConfig
from ..llm import LLMAdapter
from ..middleware import Middleware, ResultDecision, ToolDecision
from . import always_allow
from .runtime import ProgentRuntime, RuntimeResult
from .tools import normalize_tool_definitions, tool_definitions_from_config


class ProgentMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.settings = cfg.progent
        # Every benchmark uses upstream's generic secagent mode. Upstream's ASB
        # agent (asb/pyopenagi/agents/react_agent_attack.py:275-282, :138)
        # registers tools without arguments, checks names only and never updates
        # the policy; it is kept only as the explicit ``profile: asb`` option.
        self.asb = str(getattr(self.settings, "profile", "general")) == "asb"

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> Any:
        error = self._ensure_policy(state)
        if error:
            state.pop("_progent_batch", None)
            return self._model_failure(state, messages, error)
        self._flush_update(state)
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
                error=error,
            )
            state["_last_progent_decision"] = event
            return ToolDecision(allowed, "" if allowed else error)

        runtime = self._runtime(state)
        result = runtime.check(name, {} if self.asb else payload)
        denied = not result.allowed
        # A denied call raises inside upstream's wrapper, so its update sees "".
        # Parallel calls may interleave before/after_tool, so the flag is keyed
        # by the call itself (the hook carries no tool-call id).
        state.setdefault("_progent_pending_denied", {}).setdefault(
            _call_key(name, payload), []
        ).append(denied)
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
        denied = _pop_pending_denied(state, name, payload)
        if not self.settings.update_after_tool or self.asb:
            return ResultDecision(result=result)
        # Upstream updates once per tool batch after every call ran, with failed
        # and blocked calls contributing an empty result (agentdojo
        # tool_execution.py:106-123, functions_runtime.py run_function). The
        # batch is flushed before the next model call.
        state.setdefault("_progent_batch", []).append(
            {
                "call": {"name": name, "args": copy.deepcopy(payload)},
                "result": "" if failed or denied else str(result),
            }
        )
        return ResultDecision(result=result)

    def _flush_update(self, state: dict[str, Any]) -> None:
        batch = state.pop("_progent_batch", None)
        if not batch:
            return
        runtime = self._runtime(state)
        update = runtime.update(
            [item["call"] for item in batch],
            str([item["result"] for item in batch]),
            only_allow_narrow=self.settings.only_allow_narrow,
        )
        # SECAGENT_IGNORE_UPDATE_ERROR=True (agentdojo/run.sh): a failed update
        # keeps the previous policy and never withholds tool results. An update
        # never blocks a call, so it is recorded as allowed; errors that upstream
        # would raise (decide_whether_to_update, tool.py:415-432, is not covered
        # by ignore_update_error) become an error event, not a detection.
        state["_progent_policy"] = copy.deepcopy(update.policy)
        values: dict[str, Any] = {
            "status": "error" if update.error else ("updated" if update.allowed else "discarded"),
            "source": "error" if update.error else "policy_update",
        }
        if update.error:
            values["error"] = update.error
        event = self._event(
            allowed=True,
            enforced=False,
            phase="policy_update",
            reason=update.reason,
            tool=[item["call"]["name"] for item in batch],
            **values,
        )
        state["_last_progent_decision"] = event
        state.setdefault("progent_events", []).append(event)
        self._add_usage(state, update.usage)
        self._update_harness(state, event)

    def _ensure_policy(self, state: dict[str, Any]) -> str:
        if state.get("_progent_initialized"):
            return str(state.get("_progent_init_error") or "")
        state["_progent_initialized"] = True
        started = time.time()
        runtime = self._runtime(state, use_config_policy=True, generation=True)
        if not self.asb:
            self._register_always_allow(state, runtime)
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
                "task": self._query(state, generation=True),
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

    def _register_always_allow(
        self, state: dict[str, Any], runtime: ProgentRuntime
    ) -> None:
        started = time.time()
        output = self._always_allow(state, runtime.tools)
        usage = output.pop("usage", {})
        if output["tools"] or output["allow_all_no_arg_tools"]:
            registered = runtime.allow_always(
                output["tools"],
                allow_all_no_arg_tools=output["allow_all_no_arg_tools"],
            )
            if not registered.allowed:
                output.update(status="failed", error=registered.reason)
        self._add_usage(state, usage)
        event = {
            "step": "progent_always_allow",
            "timestamp": started,
            "latency_ms": int((time.time() - started) * 1000),
            "output": output,
            "usage": dict(usage),
        }
        state.setdefault("trace", []).append(event)
        self._write_artifact(state, "progent_always_allow.json", event)

    def _always_allow(
        self, state: dict[str, Any], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        names = [tool["name"] for tool in tools]
        source = str(getattr(self.settings, "always_allow", "generate"))
        no_arg = bool(getattr(self.settings, "allow_all_no_arg_tools", False))
        if source == "upstream_agentdojo":
            # Explicit non-default option: upstream's hand-written tables.
            suite = self.cfg.agentdojo.suite if self.cfg.agentdojo.enabled else ""
            upstream = always_allow.upstream_always_allow(suite) or {
                "tools": [], "allow_all_no_arg_tools": False
            }
            return {
                "status": "upstream",
                "suite": suite,
                "tools": [name for name in upstream["tools"] if name in names],
                "missing": [name for name in upstream["tools"] if name not in names],
                "allow_all_no_arg_tools": upstream["allow_all_no_arg_tools"],
            }
        if source == "none" or not self.settings.generate_always_allow:
            return {"status": "disabled", "tools": [], "allow_all_no_arg_tools": no_arg}
        cached = state.get("_progent_always_allow")
        if isinstance(cached, list):
            return {"status": "run_cache", "tools": list(cached), "allow_all_no_arg_tools": no_arg}
        inventory = always_allow.inventory(tools)
        llm_cfg = _policy_llm_config(self.cfg)
        identity = {"provider": llm_cfg.provider, "model": llm_cfg.model, "seed": 0}
        key = always_allow.fingerprint(inventory, identity)
        cache = always_allow.batch_cache_path()
        hit = always_allow.load_cached(cache, key, inventory)
        if hit is not None:
            state["_progent_always_allow"] = hit
            return {"status": "cache", "tools": hit, "allow_all_no_arg_tools": no_arg,
                    "cache_path": str(cache), "context_mode": "benign_only"}
        llm = LLMAdapter(llm_cfg)
        usage: dict[str, int] = {}

        def complete(system: str, user: str) -> str:
            content, reported = _chat(llm, system, user, 0.0)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[field] = usage.get(field, 0) + int((reported or {}).get(field) or 0)
            return str(content)

        try:
            generated, transcript = always_allow.generate(inventory, complete)
        except Exception as exc:
            # Without a list upstream registers no always-allowed tools.
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "tools": [], "allow_all_no_arg_tools": no_arg, "usage": usage}
        always_allow.store_cached(cache, key, generated)
        state["_progent_always_allow"] = generated
        self._write_artifact(state, "progent_always_allow_raw.json", transcript)
        return {"status": "llm", "tools": generated, "allow_all_no_arg_tools": no_arg,
                "context_mode": "benign_only", "attempts": len(transcript),
                "llm": {"provider": llm_cfg.provider, "model": llm_cfg.model},
                "usage": usage}

    def _runtime(
        self,
        state: dict[str, Any],
        *,
        use_config_policy: bool = False,
        generation: bool = False,
    ) -> ProgentRuntime:
        policy = (
            _configured_policy(self.settings)
            if use_config_policy
            else copy.deepcopy(state.get("_progent_policy"))
        )
        llm_cfg = _policy_llm_config(self.cfg)
        llm = LLMAdapter(llm_cfg)

        def complete(system: str, user: str, temperature: float) -> tuple[str, Any]:
            return _chat(llm, system, user, temperature)

        tools = normalize_tool_definitions(self._tool_definitions(state))
        if self.asb:
            tools = [{**tool, "args": {}} for tool in tools]
        return ProgentRuntime(
            tools=tools,
            query=self._query(state, generation=generation),
            policy=policy,
            completion=complete,
            policy_model=llm_cfg.model,
            task_type="asb" if self.asb else "general",
        )

    def _tool_definitions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        tools = state.get("_progent_tools")
        return copy.deepcopy(tools) if isinstance(tools, list) else tool_definitions_from_config(self.cfg)

    def _query(self, state: dict[str, Any], *, generation: bool = False) -> str:
        """USER_QUERY: the prompt the agent received (upstream passes the task
        prompt to ``generate_security_policy``, basic_elements.py:24).

        Generation reads the clean task and the runtime prompts (updates and
        denial messages) read the prompt actually delivered; the two are
        identical except in ASB memory-poison phases.
        """
        key = "_generation_task" if generation else "_runtime_user_request"
        for name in (key, "_progent_user_request"):
            if state.get(name):
                return str(state[name])
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
            error=error,
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

    def _write_artifact(self, state: dict[str, Any], name: str, value: Any) -> None:
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if not run_dir:
            return
        path = Path(str(run_dir)) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

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


def _call_key(name: str, arguments: Any) -> str:
    return json.dumps([name, arguments], sort_keys=True, default=str)


def _pop_pending_denied(state: dict[str, Any], name: str, payload: Any) -> bool:
    pending = state.get("_progent_pending_denied")
    if not isinstance(pending, dict) or not pending:
        return False
    key = _call_key(name, payload)
    if key not in pending and len(pending) == 1:
        # Another guard rewrote this call's arguments between the hooks; with a
        # single outstanding call the flag is unambiguous.
        key = next(iter(pending))
    queue = pending.get(key) or []
    denied = bool(queue.pop(0)) if queue else False
    if not queue:
        pending.pop(key, None)
    return denied


def _configured_policy(settings: Any) -> dict[str, Any]:
    policy = copy.deepcopy(settings.policy)
    for name in settings.always_allow_tools:
        policy.setdefault(name, []).insert(0, (1, 0, {}, 0))
    for name in settings.always_block_tools:
        policy.setdefault(name, []).insert(0, (1, 1, {}, 0))
    return policy


def _chat(
    llm: LLMAdapter, system: str, user: str, temperature: float
) -> tuple[str, Any]:
    """Send one policy request with upstream's per-attempt temperature and seed=0.

    Upstream raises the temperature by 0.2 on each retry and pins ``seed=0``
    (secagent/tool.py:239-250). The adapter builds its client once, so the
    per-request values are applied to a copy of that client.
    """
    llm.config.temperature = temperature
    llm._lazy_init()
    client = llm._client
    if hasattr(client, "model_copy"):
        updates: dict[str, Any] = {"temperature": temperature}
        if "seed" in getattr(type(client), "model_fields", {}):
            updates["seed"] = 0
        llm._client = client.model_copy(update=updates)
    try:
        response = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
    finally:
        llm._client = client
    return response.content, response.usage


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
