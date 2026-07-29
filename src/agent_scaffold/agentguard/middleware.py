from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig, ToolConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision

_SESSION_KEY = "_agentguard_session"
_LAST_KEY = "_last_agentguard_decision"
_EVENTS_KEY = "agentguard_events"


@dataclass
class _Session:
    guard: Any
    decision_type: Any
    events: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.guard.close()


def _resolve_path(cfg: AppConfig, value: str, *, run_dir: str = "") -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    config_path = (Path(cfg.config_dir) / path).resolve()
    workspace = Path(os.environ.get("AGENT_WORKSPACE_ROOT", Path.cwd())).resolve()
    repository = Path(__file__).resolve().parents[3]
    for candidate in (config_path, workspace / path, repository / path):
        if candidate.exists():
            return str(candidate.resolve())
    if run_dir:
        return str((Path(run_dir) / path).resolve())
    return str(config_path)


def _load_agentguard() -> tuple[Any, Any, Any, Any]:
    vendor_root = Path(__file__).resolve().parent / "_vendor"
    if not (vendor_root / "agentguard" / "__init__.py").is_file():
        raise ImportError(f"bundled AgentGuard runtime is missing: {vendor_root}")
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    from agentguard import AgentGuard
    from agentguard.schemas import events
    from agentguard.schemas.decisions import DecisionType
    from agentguard.tools.metadata import ToolMetadata

    return AgentGuard, DecisionType, events, ToolMetadata


def _decision_dict(decision: Any, *, phase: str, mode: str) -> dict[str, Any]:
    record = decision.to_dict()
    record["phase"] = phase
    record["mode"] = mode
    return record


def _replacement(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _blocked_value(decision: Any, *, phase: str, tool: str = "") -> str:
    payload = {
        "agentguard": (
            "pending"
            if decision.requires_user or decision.requires_remote
            else "degraded"
            if decision.decision_type.value == "degrade"
            else "blocked"
        ),
        "phase": phase,
        "decision": decision.decision_type.value,
        "reason": decision.reason,
    }
    if tool:
        payload["tool"] = tool
    return json.dumps(payload, ensure_ascii=False)


class AgentGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.settings = cfg.agentguard
        self._init_error = ""
        try:
            (
                self._agentguard_cls,
                self._decision_type,
                self._events,
                self._tool_metadata_cls,
            ) = _load_agentguard()
        except Exception as exc:  # noqa: BLE001 -- optional integration boundary
            self._agentguard_cls = None
            self._decision_type = None
            self._events = None
            self._tool_metadata_cls = None
            self._init_error = f"AgentGuard import failed: {exc}"

    def _session(self, state: dict[str, Any]) -> _Session | None:
        current = state.get(_SESSION_KEY)
        if isinstance(current, _Session) and not current.closed:
            return current
        if self._init_error:
            self._failure(state, "session", self._init_error)
            return None

        persist = state.get("_trace_persist") if isinstance(state, dict) else {}
        persist = persist if isinstance(persist, dict) else {}
        run_dir = str(persist.get("run_dir", ""))
        session_id = Path(run_dir).name if run_dir else f"{self.cfg.agent.name}-session"
        try:
            sandbox_profile = self.settings.sandbox_profile
            if isinstance(sandbox_profile, dict):
                from agentguard.sandbox import PermissionProfile

                sandbox_profile = PermissionProfile(**sandbox_profile)
            guard = self._agentguard_cls(
                session_id=session_id,
                user_id=self.settings.user_id or None,
                agent_id=self.cfg.agent.name,
                policy=_resolve_path(self.cfg, self.settings.policy, run_dir=run_dir),
                server_url=self.settings.server_url or None,
                api_key=self.settings.api_key or None,
                environment=self.settings.environment or None,
                sandbox=self.settings.sandbox,
                sandbox_profile=sandbox_profile,
                max_steps=self.settings.max_steps,
                max_tool_calls=self.settings.max_tool_calls,
                window_size=self.settings.window_size,
                audit_path=_resolve_path(
                    self.cfg, self.settings.audit_path, run_dir=run_dir
                ),
                remote_timeout_s=self.settings.remote_timeout_seconds,
                remote_retries=self.settings.remote_retries,
                plugin_config=_resolve_path(
                    self.cfg, self.settings.plugin_config, run_dir=run_dir
                ),
            )
            principal = {
                "agent_id": self.cfg.agent.name,
                "user_id": self.settings.user_id or None,
                "role": self.settings.role,
                "trust_level": self.settings.trust_level,
            }
            guard.context.metadata.update(
                {
                    "principal": {
                        key: value
                        for key, value in principal.items()
                        if value is not None
                    },
                    "role": self.settings.role,
                    "trust_level": self.settings.trust_level,
                    "goal": self.cfg.agent.task,
                    "guard_mode": self.settings.mode,
                    "guard_fail_open": not self.settings.fail_closed,
                }
            )
            for tool in self.cfg.tools:
                self._register_tool(guard, tool)
            if getattr(guard, "_remote", None) and guard._remote.enabled:
                guard._sync_remote_session()
            current = _Session(
                guard=guard, decision_type=self._decision_type, events=self._events
            )
            state[_SESSION_KEY] = current
            return current
        except Exception as exc:  # noqa: BLE001 -- fail-open/closed boundary
            self._failure(state, "session", f"AgentGuard initialization failed: {exc}")
            return None

    def _register_tool(self, guard: Any, tool: ToolConfig) -> None:
        def placeholder(**_: Any) -> None:
            return None

        placeholder.__name__ = tool.name
        labels = dict(tool.labels)
        metadata = self._tool_metadata_cls(
            name=tool.name,
            description=tool.description,
            capabilities=list(tool.capabilities),
            metadata={
                "boundary": labels.pop("boundary", "internal"),
                "sensitivity": labels.pop("sensitivity", "low"),
                "integrity": labels.pop("integrity", "trusted"),
                "tags": labels.pop("tags", tool.capabilities),
                **labels,
            },
        )
        guard.register_tool(placeholder, metadata=metadata)

    def _failure(
        self, state: dict[str, Any], phase: str, reason: str
    ) -> dict[str, Any]:
        record = {
            "decision_type": "deny" if self.settings.fail_closed else "allow",
            "reason": reason,
            "phase": phase,
            "mode": self.settings.mode,
            "route": "integration_error",
        }
        state[_LAST_KEY] = record
        state.setdefault(_EVENTS_KEY, []).append(record)
        return record

    def _evaluate(
        self, state: dict[str, Any], event: Any, phase: str, *, after: bool = False
    ) -> Any:
        session = self._session(state)
        if session is None:
            return None
        try:
            result = session.guard.runtime.guard(
                event, phase="after" if after else "before"
            )
            record = _decision_dict(
                result.decision, phase=phase, mode=self.settings.mode
            )
            record["route"] = result.route
            state[_LAST_KEY] = record
            state.setdefault(_EVENTS_KEY, []).append(record)
            if self.settings.mode == "warn" and not result.decision.is_allow:
                state["_agentguard_warning"] = (
                    f"AgentGuard warning ({result.decision.decision_type.value}): "
                    f"{result.decision.reason}"
                )
            return result.decision
        except Exception as exc:  # noqa: BLE001 -- fail-open/closed boundary
            self._failure(state, phase, f"AgentGuard evaluation failed: {exc}")
            return None

    def _enforced(self, decision: Any) -> bool:
        return (
            decision is not None
            and decision.decision_type.value != "allow"
            and self.settings.mode == "block"
        )

    def _local_tool_restriction(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        capabilities: list[str],
    ) -> ToolDecision | None:
        calls = int(state.get("_agentguard_tool_calls", 0)) + 1
        state["_agentguard_tool_calls"] = calls
        reason = ""
        route = ""
        if calls > self.settings.max_tool_calls:
            reason = (
                f"AgentGuard tool-call budget exceeded: {calls} > "
                f"{self.settings.max_tool_calls}"
            )
            route = "local_budget"
        elif isinstance(self.settings.sandbox_profile, dict):
            from agentguard.sandbox import PermissionProfile, check_permissions

            profile = PermissionProfile(**self.settings.sandbox_profile)
            permission = check_permissions(profile, capabilities, payload)
            if not permission.allowed:
                reason = permission.reason
                route = "local_sandbox"
        if not reason:
            return None

        record = {
            "decision_type": "deny",
            "reason": reason,
            "phase": "tool_before",
            "mode": self.settings.mode,
            "route": route,
            "metadata": {"tool": name, "tool_call_count": calls},
        }
        state[_LAST_KEY] = record
        state.setdefault(_EVENTS_KEY, []).append(record)
        if self.settings.mode != "block":
            if self.settings.mode == "warn":
                state["_agentguard_warning"] = f"AgentGuard warning (deny): {reason}"
            return ToolDecision()
        replacement = json.dumps(
            {
                "agentguard": "blocked",
                "phase": "tool_before",
                "decision": "deny",
                "reason": reason,
                "tool": name,
            },
            ensure_ascii=False,
        )
        return ToolDecision(
            False,
            reason,
            replacement_result=replacement,
            decision_type="deny",
        )

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_agentguard_warning", "")
        return [warning] if warning else []

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        session = self._session(state)
        if session is None:
            if self.settings.fail_closed and self.settings.mode == "block":
                return ModelDecision(
                    False, state.get(_LAST_KEY, {}).get("reason", "AgentGuard failed")
                )
            return ModelDecision()
        decision = self._evaluate(
            state,
            session.events.llm_input(session.guard.context, messages),
            "llm_before",
        )
        if not self._enforced(decision):
            return ModelDecision()
        dtype = decision.decision_type.value
        metadata = dict(decision.metadata or {})
        replacement = _replacement(
            metadata, "messages", "rewritten_messages", "sanitized_messages"
        )
        if dtype in {"sanitize", "rewrite", "repair"} and isinstance(replacement, list):
            return ModelDecision(
                messages=replacement, decision_type=dtype, reason=decision.reason
            )
        if dtype in {
            "deny",
            "degrade",
            "human_check",
            "require_approval",
            "require_remote_review",
        }:
            return ModelDecision(
                False,
                decision.reason,
                content=_blocked_value(decision, phase="llm_before"),
                decision_type=dtype,
            )
        return ModelDecision()

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        session = self._session(state)
        if session is None:
            if self.settings.fail_closed and self.settings.mode == "block":
                return ModelDecision(
                    False, state.get(_LAST_KEY, {}).get("reason", "AgentGuard failed")
                )
            return ModelDecision(content=content, tool_call=tool_call)
        decision = self._evaluate(
            state,
            session.events.llm_output(session.guard.context, content),
            "llm_after",
            after=True,
        )
        if not self._enforced(decision):
            return ModelDecision(content=content, tool_call=tool_call)
        dtype = decision.decision_type.value
        metadata = dict(decision.metadata or {})
        replacement = _replacement(
            metadata,
            "output",
            "rewritten_output",
            "sanitized_output",
            "repaired_output",
            "aligned_thought",
            "replacement",
        )
        if (
            dtype in {"sanitize", "rewrite", "repair", "align_thought"}
            and replacement is not None
        ):
            rendered = str(replacement)
            return ModelDecision(
                content=rendered,
                tool_call=None,
                decision_type=dtype,
                reason=decision.reason,
            )
        if dtype == "drop_thought":
            return ModelDecision(
                content="", tool_call=None, decision_type=dtype, reason=decision.reason
            )
        if dtype == "loop_back_to_llm":
            return ModelDecision(
                content=content,
                tool_call=None,
                retry=True,
                feedback=decision.reason,
                decision_type=dtype,
                reason=decision.reason,
            )
        if dtype in {
            "deny",
            "degrade",
            "human_check",
            "require_approval",
            "require_remote_review",
        }:
            return ModelDecision(
                False,
                decision.reason,
                content=_blocked_value(decision, phase="llm_after"),
                tool_call=None,
                decision_type=dtype,
            )
        return ModelDecision(content=content, tool_call=tool_call)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        session = self._session(state)
        if session is None:
            return ToolDecision(
                allowed=not (
                    self.settings.fail_closed and self.settings.mode == "block"
                ),
                reason=state.get(_LAST_KEY, {}).get("reason", "AgentGuard failed"),
            )
        tool_cfg = next((tool for tool in self.cfg.tools if tool.name == name), None)
        capabilities = list(tool_cfg.capabilities) if tool_cfg else []
        labels = dict(tool_cfg.labels) if tool_cfg else {}
        decision = self._evaluate(
            state,
            session.events.tool_invoke(
                session.guard.context,
                name,
                payload,
                capabilities=capabilities,
                labels=labels,
            ),
            "tool_before",
        )
        if not self._enforced(decision):
            local_restriction = self._local_tool_restriction(
                state, name, payload, capabilities
            )
            if local_restriction is not None:
                return local_restriction
        if not self._enforced(decision):
            return ToolDecision()
        dtype = decision.decision_type.value
        metadata = dict(decision.metadata or {})
        arguments = _replacement(
            metadata,
            "arguments",
            "rewritten_arguments",
            "sanitized_arguments",
            "repaired_arguments",
        )
        if dtype in {"sanitize", "rewrite", "repair"} and isinstance(arguments, dict):
            return ToolDecision(
                arguments=arguments, decision_type=dtype, reason=decision.reason
            )
        if dtype == "degrade":
            degraded_tool = _replacement(metadata, "tool", "target_tool", "degraded_to")
            degraded_args = arguments if isinstance(arguments, dict) else payload
            if degraded_tool:
                return ToolDecision(
                    tool_name=str(degraded_tool),
                    arguments=degraded_args,
                    decision_type=dtype,
                    reason=decision.reason,
                )
        if dtype in {
            "deny",
            "degrade",
            "human_check",
            "require_approval",
            "require_remote_review",
        }:
            return ToolDecision(
                False,
                decision.reason,
                replacement_result=_blocked_value(
                    decision, phase="tool_before", tool=name
                ),
                decision_type=dtype,
            )
        return ToolDecision()

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        session = self._session(state)
        if session is None:
            if self.settings.fail_closed and self.settings.mode == "block":
                return ResultDecision(
                    False, state.get(_LAST_KEY, {}).get("reason", "AgentGuard failed")
                )
            return ResultDecision(result=result)
        decision = self._evaluate(
            state,
            session.events.tool_result(
                session.guard.context,
                name,
                result,
                error=result if failed else None,
            ),
            "tool_after",
            after=True,
        )
        if not self._enforced(decision):
            return ResultDecision(result=result)
        dtype = decision.decision_type.value
        metadata = dict(decision.metadata or {})
        replacement = _replacement(
            metadata,
            "result",
            "output",
            "sanitized_result",
            "rewritten_result",
            "repaired_result",
            "replacement",
        )
        if dtype in {"sanitize", "rewrite", "repair"} and replacement is not None:
            return ResultDecision(
                result=str(replacement), decision_type=dtype, reason=decision.reason
            )
        if dtype in {
            "deny",
            "sanitize",
            "degrade",
            "human_check",
            "require_approval",
            "require_remote_review",
        }:
            return ResultDecision(
                False,
                decision.reason,
                result=_blocked_value(decision, phase="tool_after", tool=name),
                decision_type=dtype,
            )
        return ResultDecision(result=result)


def close_agentguard_session(state: dict[str, Any]) -> dict[str, Any] | None:
    session = state.pop(_SESSION_KEY, None)
    if not isinstance(session, _Session):
        return None
    try:
        audit = session.guard.flush_audit()
        session.close()
        return {"closed": True, "audit_records": len(audit)}
    except Exception as exc:  # noqa: BLE001 -- cleanup must never mask run result
        return {"closed": False, "error": str(exc)}
