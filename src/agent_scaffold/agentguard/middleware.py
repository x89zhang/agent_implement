from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..config import AppConfig, ToolConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from ..tool_results import unwrap_hermes_result

_SESSION_KEY = "_agentguard_session"
_LAST_KEY = "_last_agentguard_decision"
_EVENTS_KEY = "agentguard_events"
_OUTPUT_DECISION_KEY = "_agentguard_output_decision"


# Nobody answers review tickets in a benchmark run; pending decisions return
# at once and are treated as denials instead of waiting 600 s for a person.
_APPROVAL_WAIT_SECONDS = 0.001


@dataclass
class _Session:
    guard: Any
    decision_type: Any
    events: Any
    server: Any = None
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.guard.close()
        finally:
            if self.server is not None:
                self.server.stop()


class _FailedDecision:
    """Fail-closed stand-in for a decision the integration could not obtain."""

    requires_user = False
    requires_remote = False
    is_allow = False
    metadata: dict[str, Any] = {}

    def __init__(self, reason: str) -> None:
        self.decision_type = SimpleNamespace(value="deny")
        self.reason = reason


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


def _default_plugin_config() -> Path:
    # Upstream config/plugins.json: server rule_based_plugin at tool_before and
    # no client plugins.
    from .launcher import DEFAULT_PLUGIN_CONFIG

    return DEFAULT_PLUGIN_CONFIG


def _start_server(
    cfg: AppConfig, policy: str | None, plugin_config: str, run_dir: str
) -> Any:
    from .launcher import start_server
    from .scenario import _compiler_llm_config

    llm = _compiler_llm_config(cfg)
    return start_server(
        policy_path=policy or "",
        plugin_config=plugin_config,
        # The server's LLM_CHECK reviewer uses the policy generator's model.
        llm={
            "provider": llm.provider,
            "model": llm.model,
            "temperature": llm.temperature,
            "base_url": llm.base_url,
            "api_key": llm.api_key or os.environ.get(llm.api_key_env or "", ""),
            "api_key_env": llm.api_key_env,
            "request_timeout": llm.request_timeout,
        },
        work_dir=run_dir,
        timeout=cfg.agentguard.server_startup_timeout_seconds,
    )


def _llm_output_payload(
    content: str, calls: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Normalize one assistant turn the way upstream's LangChain adapter does.

    Hermes turns are OpenAI chat messages (content plus tool_calls), the shape
    ``LangChainAgentAdapter.normalize_llm_output`` handles; it keeps the content
    as output and extracts a thought from reasoning tags or ReAct prefixes.
    """
    message: dict[str, Any] = {"type": "ai", "content": content or ""}
    if calls:
        message["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("arguments") or {},
                "id": call.get("id"),
            }
            for call in calls
            if isinstance(call, dict)
        ]
    try:
        from agentguard.adapters.agent.langchain import LangChainAgentAdapter

        normalized = LangChainAgentAdapter().normalize_llm_output(
            label="hermes", output=message
        )
    except Exception:  # noqa: BLE001 -- fall back to the raw message
        return message, {}
    return normalized.payload, dict(normalized.metadata)


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

        if self.settings.compile_error:
            self._failure(state, "session", self.settings.compile_error)
            return None

        persist = state.get("_trace_persist") if isinstance(state, dict) else {}
        persist = persist if isinstance(persist, dict) else {}
        run_dir = str(persist.get("run_dir", ""))
        session_id = Path(run_dir).name if run_dir else f"{self.cfg.agent.name}-session"
        server = None
        try:
            policy = _resolve_path(self.cfg, self.settings.policy, run_dir=run_dir)
            plugin_config = _resolve_path(
                self.cfg, self.settings.plugin_config, run_dir=run_dir
            ) or str(_default_plugin_config())
            server_url = self.settings.server_url
            if not server_url and self.settings.auto_start_server:
                server = _start_server(self.cfg, policy, plugin_config, run_dir)
                server_url = server.url
            sandbox_profile = self.settings.sandbox_profile
            if isinstance(sandbox_profile, dict):
                from agentguard.sandbox import PermissionProfile

                sandbox_profile = PermissionProfile(**sandbox_profile)
            guard = self._agentguard_cls(
                session_id=session_id,
                user_id=self.settings.user_id or None,
                agent_id=self.cfg.agent.name,
                policy=policy,
                server_url=server_url or None,
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
                plugin_config=plugin_config,
            )
            remote = getattr(guard, "_remote", None)
            if remote is not None and hasattr(remote, "approval_wait_timeout_s"):
                remote.approval_wait_timeout_s = _APPROVAL_WAIT_SECONDS
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
                guard=guard,
                decision_type=self._decision_type,
                events=self._events,
                server=server,
            )
            state[_SESSION_KEY] = current
            return current
        except Exception as exc:  # noqa: BLE001 -- fail-open/closed boundary
            if server is not None:
                server.stop()
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
    ) -> _FailedDecision | None:
        """Record an integration failure; it is an error, never a verdict."""
        record = {
            "error": reason,
            "reason": reason,
            "phase": phase,
            "mode": self.settings.mode,
            "route": "integration_error",
            "fail_closed": self.settings.fail_closed,
        }
        state[_LAST_KEY] = record
        state.setdefault(_EVENTS_KEY, []).append(record)
        return _FailedDecision(reason) if self.settings.fail_closed else None

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
            if result.route == "remote_unavailable":
                # The per-run server did not answer: an integration failure.
                return self._failure(
                    state, phase, f"AgentGuard server unavailable: {result.decision.reason}"
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
            return self._failure(state, phase, f"AgentGuard evaluation failed: {exc}")

    def _enforced(self, decision: Any) -> bool:
        return (
            decision is not None
            and decision.decision_type.value != "allow"
            and self.settings.mode == "block"
        )

    def _tool_budget_exceeded(self, state: dict[str, Any], name: str) -> ToolDecision | None:
        """Upstream's tool-call budget (``HarnessRuntime._invoke_tool_inner``).

        Upstream returns a ``safe_error`` before any guard event, so this is not
        a policy decision: it is recorded as a budget event, not a denial.
        """
        calls = int(state.get("_agentguard_tool_calls", 0))
        if calls < self.settings.max_tool_calls:
            state["_agentguard_tool_calls"] = calls + 1
            return None
        reason = "tool call budget exceeded"
        record = {
            "event": "tool_call_budget_exceeded",
            "reason": reason,
            "phase": "tool_before",
            "mode": self.settings.mode,
            "route": "local_budget",
            "metadata": {
                "tool": name,
                "tool_call_count": calls,
                "max_tool_calls": self.settings.max_tool_calls,
            },
        }
        state[_LAST_KEY] = record
        state.setdefault(_EVENTS_KEY, []).append(record)
        if self.settings.mode != "block":
            return ToolDecision()
        replacement = json.dumps(
            {"agentguard": "blocked", "tool": name, "reason": reason, "decision": "deny"},
            ensure_ascii=False,
        )
        return ToolDecision(False, reason, replacement_result=replacement)

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
        # GuardController checks parallel calls one by one; upstream emits one
        # llm_output event per model response, so only the first call evaluates
        # it and later calls reuse that decision.
        if int(state.get("_model_output_index", 0) or 0) == 0:
            payload, metadata = _llm_output_payload(
                content, state.get("_model_output_calls") or []
            )
            decision = self._evaluate(
                state,
                session.events.llm_output(session.guard.context, payload, **metadata),
                "llm_after",
                after=True,
            )
            state[_OUTPUT_DECISION_KEY] = decision
        else:
            decision = state.get(_OUTPUT_DECISION_KEY)
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
        budget = self._tool_budget_exceeded(state, name)
        if budget is not None:
            return budget
        tool_cfg = next((tool for tool in self.cfg.tools if tool.name == name), None)
        capabilities = list(tool_cfg.capabilities) if tool_cfg else []
        decision = self._evaluate(
            state,
            session.events.tool_invoke(
                session.guard.context, name, payload, capabilities=capabilities
            ),
            "tool_before",
        )
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
        # Upstream reports a failed tool as tool_result(None, error=...)
        # (HarnessRuntime._execute).
        event = (
            session.events.tool_result(
                session.guard.context, name, None, error=unwrap_hermes_result(result)
            )
            if failed
            else session.events.tool_result(
                session.guard.context, name, unwrap_hermes_result(result)
            )
        )
        decision = self._evaluate(state, event, "tool_after", after=True)
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
