from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig, LLMConfig
from ..llm import LLMAdapter
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ._upstream import markers, origin, scopes_io
from ._upstream.clamp import _more_permissive
from ._upstream.enforce import Enforcer
from ._upstream.policy_synthesis import TaskScope, compile_policy
from ._upstream.router import route_and_scope
from .floor_generator import generate_floor, inventory_from_config


class RopeMiddleware(Middleware):
    """ROPE's audited sensitive-argument floor, task scope and provenance guard."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.settings = cfg.rope

    def guard_model_input(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision:
        error = self._initialize(state)
        if error and self.settings.fail_closed:
            return ModelDecision(False, error, messages=messages, decision_type="rope_init_error", terminate=True)
        return ModelDecision(messages=messages)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_rope_warning", "")
        return [f"ROPE policy warning: {warning}"] if warning else []

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        error = self._initialize(state)
        if error:
            return self._record(state, name, not self.settings.fail_closed, error, "error")
        try:
            floor = scopes_io.floor_from_dict(state.get("_rope_floor") or {})
            if name not in floor:
                if self.settings.floor_generation == "audited_only":
                    return self._record(state, name, True, "tool is outside the audited ROPE floor", "floor")
                if name in state.get("_rope_read_only_tools", []):
                    return self._record(state, name, True, "tool classified as read-only", "floor")
                return self._record(state, name, False, "tool is not covered by a ROPE rule", "floor")
            scope_data = state.get("_rope_scope")
            if not scope_data:
                return self._record(state, name, False, "no task scope matched; sensitive tool denied", "scope")
            scope = markers.scope_from_dict(scope_data)
            origins = self._origins(state)
            policy = compile_policy(
                floor, scope, origin_map=origins,
                prompt_ids=origin.tokenize_identifiers(self._query(state)),
            )
            # The upstream enforcer only checks arguments that are present. A missing
            # sensitive parameter must not evade its floor rule.
            missing = sorted(set(floor[name]) - set(payload))
            if missing:
                raise ValueError(f"missing sensitive parameter(s): {', '.join(missing)}")
            if name in state.get("_rope_generated_tools", []):
                for arg, marker in (state.get("_rope_floor") or {}).get(name, {}).items():
                    if marker == "PROMPT" and not self._value_in_request(payload[arg], self._query(state)):
                        raise ValueError(f"value for {name}.{arg} is absent from the trusted request")
                    if marker == "SOURCED" and not self._short_value_authorized(payload[arg], self._query(state)):
                        raise ValueError(f"short value for {name}.{arg} is absent from the trusted request")
            enforcer = Enforcer()
            enforcer.set_user_query(self._query(state))
            enforcer.update_security_policy(policy)
            enforcer.check_tool_call(name, payload)
            return self._record(state, name, True, "allowed by ROPE", "policy")
        except Exception as exc:
            return self._record(state, name, False, str(exc), "policy")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        if failed or state.get("_rope_init_error"):
            return ResultDecision(result=result)
        try:
            origins = self._origins(state)
            tracker = origin.OriginTracker(origins)
            tracker._pending_deferral = state.get("_rope_pending_deferral")
            if name in self.settings.trusted_origin_tools:
                tracker._absorb({"content": result, "tool_call": {"function": name, "args": payload}})
            else:
                # Never parse an untrusted blob's self-asserted sender/from field as
                # authenticated provenance. The tool must be opted in explicitly.
                tracker._add(f"{name}(untrusted)", origin.tokenize_identifiers(str(result)), injectable=True)
            state["_rope_origins"] = {
                key: {"ids": sorted(value["ids"]), "injectable": value["injectable"]}
                for key, value in origins.items()
            }
            state["_rope_pending_deferral"] = tracker._pending_deferral
            state["_last_rope_decision"] = {"phase": "after_tool", "tool": name, "source": "origin", "trusted": name in self.settings.trusted_origin_tools}
        except Exception as exc:
            if self.settings.fail_closed:
                return ResultDecision(False, f"ROPE origin tracking failed: {exc}", result, "rope_origin_error")
        return ResultDecision(result=result)

    def _record(self, state: dict[str, Any], name: str, allowed: bool, reason: str, source: str) -> ToolDecision:
        enforced = not allowed and self.settings.mode == "block"
        event = {"phase": "before_tool", "tool": name, "allowed": allowed, "enforced": enforced, "reason": reason, "source": source, "mode": self.settings.mode}
        state["_last_rope_decision"] = event
        state.setdefault("rope_events", []).append(event)
        state.setdefault("harness", {}).setdefault("rope", {}).update({
            "enabled": True, "mode": self.settings.mode,
            "status": "failed" if source == "error" else "active",
            "bucket": (state.get("_rope_scope") or {}).get("bucket"),
            "last_decision": event, "event_count": len(state["rope_events"]),
        })
        if not allowed and self.settings.mode == "warn":
            state["_rope_warning"] = reason
        return ToolDecision(not enforced, reason if enforced else "", decision_type="rope" if enforced else "")

    def _initialize(self, state: dict[str, Any]) -> str:
        if state.get("_rope_initialized"):
            return str(state.get("_rope_init_error") or "")
        state["_rope_initialized"] = True
        started = time.time()
        try:
            floor = self._prepare_floor(state)
            query = self._query(state)
            if not query.strip():
                raise ValueError("ROPE requires a trusted user request")
            scope = self._scope(query, floor)
            if scope is not None:
                self._validate_scope(scope, floor)
                scope = self._clamp_generated_scope(scope, state, query)
                if self.settings.clamp:
                    scope = self._clamp(scope, floor)
            state["_rope_scope"] = markers.scope_to_dict(scope) if scope else None
            error = ""
        except Exception as exc:
            error = f"ROPE initialization failed: {exc}"
            state["_rope_scope"] = None
        state["_rope_init_error"] = error
        event = {"step": "rope_scope_generate", "timestamp": started,
                 "latency_ms": int((time.time() - started) * 1000),
                 "output": {"status": "failed" if error else "active", "reason": error,
                            "scope": state["_rope_scope"],
                            "floor_source": state.get("_rope_floor_source"),
                            "floor_tool_count": len(state.get("_rope_floor") or {})}}
        state.setdefault("trace", []).append(event)
        state.setdefault("harness", {}).setdefault("rope", {}).update({
            "enabled": True, "mode": self.settings.mode,
            "status": "failed" if error else "active",
            "bucket": (state["_rope_scope"] or {}).get("bucket"),
        })
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if run_dir:
            path = Path(str(run_dir)) / "rope_scope.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        return error

    def _prepare_floor(self, state: dict[str, Any]) -> dict:
        inventory = inventory_from_config(self.cfg, state)
        state["_rope_known_tools"] = [tool["name"] for tool in inventory]
        floor: dict = {}
        source = "none"
        if self.settings.floor_path:
            path = self._path(self.settings.floor_path)
            floor = scopes_io.floor_from_dict(json.loads(path.read_text(encoding="utf-8")))
            source = "configured"
        else:
            suite = self.settings.suite or (self.cfg.agentdojo.suite if self.cfg.agentdojo.enabled else "")
            if suite:
                try:
                    floor = scopes_io.load_floor(suite)
                    source = "audited"
                except FileNotFoundError:
                    if self.settings.floor_generation != "llm":
                        raise
            elif self.settings.floor_generation != "llm":
                raise ValueError("set rope.suite or rope.floor_path")
        unknown = [tool for tool in inventory if tool["name"] not in floor]
        generated: dict[str, dict[str, str]] = {}
        blocked: list[str] = []
        read_only: list[str] = []
        if unknown and self.settings.floor_generation == "llm":
            llm = self._llm()

            def complete(system: str, user: str) -> str:
                return llm.chat([{"role": "system", "content": system},
                                 {"role": "user", "content": user}]).content

            generated, blocked, read_only = generate_floor(unknown, complete)
            floor.update(scopes_io.floor_from_dict(generated))
            source = f"{source}+llm" if source in {"audited", "configured"} else "llm"
        elif not floor and self.settings.floor_generation == "llm":
            raise ValueError("ROPE cannot generate a floor without tool definitions")
        state["_rope_floor"] = scopes_io.floor_to_dict(floor)
        state["_rope_floor_source"] = source
        state["_rope_blocked_tools"] = blocked
        state["_rope_read_only_tools"] = read_only
        state["_rope_generated_tools"] = sorted(generated)
        state.setdefault("trace", []).append({
            "step": "rope_floor_generate", "output": {
                "source": source, "floor": state["_rope_floor"],
                "blocked_tools": blocked, "read_only_tools": read_only,
            },
        })
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if run_dir:
            path = Path(str(run_dir)) / "rope_floor.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state["trace"][-1], ensure_ascii=False, indent=2), encoding="utf-8")
        return floor

    def _scope(self, query: str, floor: dict) -> TaskScope | None:
        if self.settings.router == "static":
            raw = self.settings.scope
            if self.settings.scope_path:
                raw = json.loads(self._path(self.settings.scope_path).read_text(encoding="utf-8"))
            if not raw:
                raise ValueError("static router requires rope.scope or rope.scope_path")
            return markers.scope_from_dict(raw)
        if self.settings.router == "cached":
            suite = self.settings.suite or (self.cfg.agentdojo.suite if self.cfg.agentdojo.enabled else "")
            _, scopes = scopes_io.load_router_scopes(self.settings.cached_router, suite)
            return scopes.get(query)
        llm = self._llm()

        def complete(system: str, user: str) -> str:
            response = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            return response.content

        return route_and_scope(query, floor, complete, trusted_facts=self.settings.trusted_facts)

    def _llm(self) -> LLMAdapter:
        base = self.cfg.llm
        settings = self.settings
        return LLMAdapter(LLMConfig(
            provider=settings.provider or base.provider, model=settings.model or base.model,
            temperature=settings.temperature if settings.temperature is not None else base.temperature,
            base_url=settings.base_url or base.base_url, api_key=settings.api_key or base.api_key,
            api_key_env=settings.api_key_env or base.api_key_env,
            request_timeout=settings.request_timeout if settings.request_timeout is not None else base.request_timeout,
        ))

    def _validate_scope(self, scope: TaskScope, floor: dict) -> None:
        for tool, args in scope.overrides.items():
            if tool not in floor:
                raise ValueError(f"scope references unknown sensitive tool {tool!r}")
            for arg in args:
                if arg not in floor[tool]:
                    raise ValueError(f"scope references unknown sensitive argument {tool}.{arg}")

    def _clamp(self, scope: TaskScope, floor: dict) -> TaskScope:
        kept: dict = {}
        for tool, args in scope.overrides.items():
            for arg, rule in args.items():
                if not _more_permissive(markers.marker_to_str(rule), markers.marker_to_str(floor[tool][arg])):
                    kept.setdefault(tool, {})[arg] = rule
        return TaskScope(scope.bucket, scope.named_source, kept)

    def _clamp_generated_scope(self, scope: TaskScope, state: dict[str, Any], query: str) -> TaskScope:
        """A second LLM call must not widen a generated first-layer rule."""
        defaults = state.get("_rope_floor") or {}
        generated = set(state.get("_rope_generated_tools") or [])
        kept: dict = {}
        for tool, args in scope.overrides.items():
            for arg, rule in args.items():
                if tool not in generated:
                    kept.setdefault(tool, {})[arg] = rule
                    continue
                original = defaults[tool][arg]
                kind = rule[0]
                permitted = {
                    "PROMPT": {"prompt", "const", "oneof", "explicit"},
                    "SOURCED": {"sourced", "prompt", "const", "oneof", "explicit"},
                    "EXPLICIT": {"explicit", "const", "oneof"},
                }[original]
                if kind not in permitted:
                    continue
                if kind in {"const", "oneof"}:
                    values = [rule[1]] if kind == "const" else rule[1]
                    if not all(self._value_in_request(value, query) for value in values):
                        continue
                kept.setdefault(tool, {})[arg] = rule
        return TaskScope(scope.bucket, scope.named_source, kept)

    def _query(self, state: dict[str, Any]) -> str:
        if state.get("_rope_user_request"):
            return str(state["_rope_user_request"])
        return str(self.cfg.agent.task or "")

    def _origins(self, state: dict[str, Any]) -> dict:
        return {key: {"ids": set(value.get("ids") or []), "injectable": bool(value.get("injectable", True))}
                for key, value in (state.get("_rope_origins") or {}).items()}

    def _short_value_authorized(self, value: Any, query: str) -> bool:
        if isinstance(value, (list, tuple)):
            return all(self._short_value_authorized(item, query) for item in value)
        rendered = str(value)
        if origin.tokenize_identifiers(rendered, split_composites=False):
            return True
        return bool(rendered and re.search(r"(?<![A-Za-z0-9])" + re.escape(rendered) + r"(?![A-Za-z0-9])", query))

    def _value_in_request(self, value: Any, query: str) -> bool:
        if isinstance(value, (list, tuple)):
            return all(self._value_in_request(item, query) for item in value)
        rendered = str(value)
        return bool(rendered and re.search(r"(?<![A-Za-z0-9])" + re.escape(rendered) + r"(?![A-Za-z0-9])", query))

    def _path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.cfg.config_dir) / path
        return path
