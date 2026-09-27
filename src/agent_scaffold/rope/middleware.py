from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import AppConfig, LLMConfig
from ..llm import LLMAdapter
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ._upstream import markers, origin, scopes_io
from ._upstream.clamp import _more_permissive
from ._upstream.enforce import Enforcer
from ._upstream.fewshot import fewshot, trusted_facts
from ._upstream.llm_cache import LLMCache
from ._upstream.policy_synthesis import TaskScope, compile_policy
from ._upstream.router import route_and_scope
from .floor_generator import _parse_json, generate_floor, inventory_from_config


def _unwrap_hermes_result(result: str) -> str:
    """Return the bare tool text inside Hermes' MCP envelope.

    Hermes renders an MCP result as ``{"result": text}``, optionally with
    ``structuredContent``/``_meta`` (tools/mcp_tool_handlers.py). Upstream's
    tracker expects the bare tool text (origin.py ``_absorb``).
    """
    try:
        value = json.loads(result)
    except (TypeError, ValueError):
        return result
    if not isinstance(value, dict) or "result" not in value or set(value) - {"result", "structuredContent", "_meta"}:
        return result
    inner = value["result"]
    if isinstance(inner, str):
        return inner
    return json.dumps(inner, ensure_ascii=False)


class _SharedLLMCache(LLMCache):
    """Upstream cache, merged with the file on flush so parallel runs keep entries."""

    def _flush(self) -> None:
        if self.path.exists():
            try:
                self._store = {**json.loads(self.path.read_text(encoding="utf-8")), **self._store}
            except ValueError:
                pass
        super()._flush()


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
        decision = self._check_tool(state, name, payload)
        if state["rope_events"][-1]["allowed"]:
            state.pop("_rope_denied_call", None)
        else:
            state["_rope_denied_call"] = self._call_key(name, payload)
        return decision

    def _check_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        error = self._initialize(state)
        if error:
            return self._record(state, name, not self.settings.fail_closed, error, "error")
        try:
            floor = scopes_io.floor_from_dict(state.get("_rope_floor") or {})
            if name not in floor:
                if name in state.get("_rope_blocked_tools", []):
                    return self._record(state, name, False, "ROPE floor generation failed for this tool", "floor")
                # Upstream default-allows every tool outside the floor (pipeline.py).
                return self._record(state, name, True, "tool is outside the ROPE floor", "floor")
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
            required = (state.get("_rope_required_arguments") or {}).get(name)
            protected = set(floor[name]) if required is None else set(floor[name]) & set(required)
            missing = sorted(protected - set(payload))
            if missing:
                raise ValueError(f"missing sensitive parameter(s): {', '.join(missing)}")
            enforcer = Enforcer()
            enforcer.set_user_query(self._query(state))
            enforcer.update_security_policy(policy)
            enforcer.check_tool_call(name, payload)
            return self._record(state, name, True, "allowed by ROPE", "policy")
        except Exception as exc:
            return self._record(state, name, False, str(exc), "policy")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        denied = state.pop("_rope_denied_call", None)
        if failed or state.get("_rope_init_error"):
            return ResultDecision(result=result)
        if denied == self._call_key(name, payload):
            # Upstream returns an error with no content for a denied call, so
            # its real result (monitor/warn mode) must not license later values.
            state["_last_rope_decision"] = {"phase": "after_tool", "tool": name, "source": "origin",
                                            "skipped": "call denied by ROPE"}
            return ResultDecision(result=result)
        try:
            origins = self._origins(state)
            tracker = origin.OriginTracker(origins)
            tracker._pending_deferral = state.get("_rope_pending_deferral")
            # Feed every result to the upstream tracker. It distinguishes
            # structured sender/from records, authoritative records, and
            # injectable blobs using the tool result and call metadata.
            text = _unwrap_hermes_result(result)
            tracker._absorb({"content": text, "tool_call": {"function": name, "args": payload}})
            state["_rope_origins"] = {
                key: {"ids": sorted(value["ids"]), "injectable": value["injectable"]}
                for key, value in origins.items()
            }
            state["_rope_pending_deferral"] = tracker._pending_deferral
            state["_last_rope_decision"] = {"phase": "after_tool", "tool": name, "source": "origin"}
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
            "status": "failed" if source == "error" else "degraded" if state.get("_rope_floor_errors") else "active",
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
        status = "failed" if error else "degraded" if state.get("_rope_floor_errors") else "active"
        event = {"step": "rope_scope_generate", "timestamp": started,
                 "latency_ms": int((time.time() - started) * 1000),
                 "output": {"status": status, "reason": error,
                            "scope": state["_rope_scope"],
                            "floor_source": state.get("_rope_floor_source"),
                            "floor_tool_count": len(state.get("_rope_floor") or {}),
                            "floor_generation_errors": state.get("_rope_floor_errors") or {}}}
        state.setdefault("trace", []).append(event)
        state.setdefault("harness", {}).setdefault("rope", {}).update({
            "enabled": True, "mode": self.settings.mode,
            "status": status,
            "bucket": (state["_rope_scope"] or {}).get("bucket"),
        })
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if run_dir:
            path = Path(str(run_dir)) / "rope_scope.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        return error

    @staticmethod
    def _call_key(name: str, payload: dict[str, Any]) -> str:
        return json.dumps([name, payload], sort_keys=True, default=str)

    def _suite(self) -> str:
        return self.settings.suite or (self.cfg.agentdojo.suite if self.cfg.agentdojo.enabled else "")

    def _prepare_floor(self, state: dict[str, Any]) -> dict:
        inventory = inventory_from_config(self.cfg, state)
        state["_rope_known_tools"] = [tool["name"] for tool in inventory]
        state["_rope_required_arguments"] = {
            tool["name"]: tool["required_arguments"] for tool in inventory
        }
        floor: dict = {}
        source = "none"
        if self.settings.floor_path:
            path = self._path(self.settings.floor_path)
            floor = scopes_io.floor_from_dict(json.loads(path.read_text(encoding="utf-8")))
            source = "configured"
        elif self._suite():
            try:
                floor = scopes_io.load_floor(self._suite())
                source = "audited"
            except FileNotFoundError:
                if self.settings.floor_generation != "llm":
                    raise
        elif self.settings.floor_generation != "llm":
            raise ValueError("set rope.suite or rope.floor_path")
        generated: dict[str, dict[str, str]] = {}
        blocked: list[str] = []
        generation_errors: dict[str, str] = {}
        batch_error = ""
        # An audited or configured floor is used as-is, with every other tool
        # default-allowed (upstream pipeline.py). Only suites without one get
        # an LLM-generated floor, from the trusted tool definitions alone.
        if source == "none":
            names = [tool["name"] for tool in inventory]
            if not names:
                raise ValueError("ROPE cannot generate a floor without tool definitions")
            if any(not name for name in names) or len(names) != len(set(names)):
                raise ValueError("ROPE tool inventory has missing or duplicate names")
            complete = self._completer()
            raw_response = ""

            def recorded(system: str, user: str) -> str:
                nonlocal raw_response
                raw_response = complete(system, user)
                return raw_response

            try:
                generated = generate_floor(inventory, recorded)
            except (ValueError, TypeError) as exc:
                batch_error = str(exc)
                generated, blocked, generation_errors = self._isolate_floor_errors(
                    inventory, raw_response, batch_error
                )
            floor = scopes_io.floor_from_dict(generated)
            source = "llm"
        state["_rope_floor"] = scopes_io.floor_to_dict(floor)
        state["_rope_floor_source"] = source
        state["_rope_blocked_tools"] = blocked
        state["_rope_generated_tools"] = sorted(generated)
        state["_rope_floor_errors"] = generation_errors
        state.setdefault("trace", []).append({
            "step": "rope_floor_generate", "output": {
                "source": source, "floor": state["_rope_floor"],
                "blocked_tools": blocked,
                "batch_error": batch_error, "generation_errors": generation_errors,
            },
        })
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if run_dir:
            path = Path(str(run_dir)) / "rope_floor.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state["trace"][-1], ensure_ascii=False, indent=2), encoding="utf-8")
        return floor

    def _isolate_floor_errors(
        self, tools: list[dict[str, Any]], raw_response: str, batch_error: str
    ) -> tuple[dict[str, dict[str, str]], list[str], dict[str, str]]:
        """Keep valid tool rules from one LLM response; fail closed only bad tools."""
        try:
            rows = _parse_json(raw_response).get("tools")
        except (ValueError, TypeError):
            rows = None
        if not isinstance(rows, list):
            names = [tool["name"] for tool in tools]
            return {}, names, {name: batch_error for name in names}
        generated: dict[str, dict[str, str]] = {}
        blocked: list[str] = []
        errors: dict[str, str] = {}
        for tool in tools:
            name = tool["name"]
            matches = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
            if len(matches) != 1:
                blocked.append(name)
                errors[name] = f"expected one generated rule for {name!r}, found {len(matches)}"
                continue
            one_reply = json.dumps({"tools": matches}, ensure_ascii=False)
            try:
                rules = generate_floor([tool], lambda _system, _user: one_reply)
            except (ValueError, TypeError) as exc:
                blocked.append(name)
                errors[name] = str(exc)
                continue
            generated.update(rules)
        return generated, blocked, errors

    def _scope(self, query: str, floor: dict) -> TaskScope | None:
        if self.settings.router == "static":
            raw = self.settings.scope
            if self.settings.scope_path:
                raw = json.loads(self._path(self.settings.scope_path).read_text(encoding="utf-8"))
            if not raw:
                raise ValueError("static router requires rope.scope or rope.scope_path")
            return markers.scope_from_dict(raw)
        if self.settings.router == "cached":
            _, scopes = scopes_io.load_router_scopes(self.settings.cached_router, self._suite())
            return scopes.get(query)
        # Upstream's live router default (run_eval.py live_few_shot=True): the
        # leave-suite-out few-shot block and the suite's trusted facts.
        suite = self._suite()
        examples = fewshot(suite) if self.settings.live_few_shot else ""
        facts = self.settings.trusted_facts or (trusted_facts(suite) if self.settings.live_few_shot else "")
        return route_and_scope(query, floor, self._completer(), examples=examples, trusted_facts=facts)

    def _completer(self):
        """``llm_complete(system, user)`` memoized like upstream's cached_chat_completer."""
        config = self._llm_config()
        llm, model = LLMAdapter(config), config.model
        cache = None
        if self.settings.router_cache_path:
            cache = _SharedLLMCache(self._path(self.settings.router_cache_path))

        def complete(system: str, user: str) -> str:
            if cache is not None:
                hit = cache.get(model, system, user)
                if hit is not None:
                    return hit
            text = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}]).content or ""
            if cache is not None:
                cache.put(model, system, user, text)
            return text

        return complete

    def _llm_config(self) -> LLMConfig:
        base = self.cfg.llm
        settings = self.settings
        return LLMConfig(
            provider=settings.provider or base.provider, model=settings.model or base.model,
            # Upstream's router always runs at temperature 0 (llm_cache.py).
            temperature=0.0,
            base_url=settings.base_url or base.base_url, api_key=settings.api_key or base.api_key,
            api_key_env=settings.api_key_env or base.api_key_env,
            request_timeout=settings.request_timeout if settings.request_timeout is not None else base.request_timeout,
        )

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
        """A router override must not loosen a generated floor rule, even unclamped."""
        defaults = state.get("_rope_floor") or {}
        generated = set(state.get("_rope_generated_tools") or [])
        kept: dict = {}
        for tool, args in scope.overrides.items():
            for arg, rule in args.items():
                if tool in generated and _more_permissive(markers.marker_to_str(rule), defaults[tool][arg]):
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

    def _path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.cfg.config_dir) / path
        return path
