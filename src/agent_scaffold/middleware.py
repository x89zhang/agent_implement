from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .guard import check_tool_call
from .planner import complete_plan_on_final, mark_plan_progress, render_plan_context


@dataclass
class ToolDecision:
    allowed: bool = True
    reason: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    replacement_result: str | None = None
    decision_type: str = ""
    terminate: bool = False


class ToolExecutionTerminated(RuntimeError):
    def __init__(
        self, result: str, tool_name: str, payload: dict[str, Any]
    ) -> None:
        super().__init__(result)
        self.result = result
        self.tool_name = tool_name
        self.payload = payload


_UNCHANGED = object()


@dataclass
class ModelDecision:
    allowed: bool = True
    reason: str = ""
    messages: list[dict[str, Any]] | None = None
    content: str | None = None
    tool_call: Any = _UNCHANGED
    retry: bool = False
    feedback: str = ""
    decision_type: str = ""
    terminate: bool = False


@dataclass
class ResultDecision:
    allowed: bool = True
    reason: str = ""
    result: Any = _UNCHANGED
    decision_type: str = ""


class Middleware:
    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        return ModelDecision()

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        return ModelDecision(content=content, tool_call=tool_call)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        return []

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        return None

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        return ToolDecision()

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        return ResultDecision(result=result)


class AegisGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        decision = check_tool_call(self.cfg, state, name, payload)
        state["_last_aegis_decision"] = decision.to_dict()
        if not decision.allowed:
            return ToolDecision(False, decision.reason or "blocked by Aegis guard")
        return ToolDecision(True, "")


class HarnessMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def before_model(self, state: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        plan_context = render_plan_context(state)
        if plan_context:
            chunks.append(plan_context)
        recent_errors = state.get("tool_errors") or []
        if recent_errors:
            rendered = "\n".join(f"- {item}" for item in recent_errors[-3:])
            chunks.append(
                f"# Recent Tool Issues\nAvoid repeating these failed calls unless you have changed the arguments.\n{rendered}"
            )
        return chunks

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        if tool_call:
            name = str(tool_call[0]) if isinstance(tool_call, tuple) else "tool"
            mark_plan_progress(state, "selected_tool", name)
            return
        if content.strip():
            complete_plan_on_final(state)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        if name == "write_text_file":
            path = str(payload.get("path") or "")
            if ".." in path.replace("\\", "/").split("/"):
                return ToolDecision(
                    False, "write_text_file path must stay under the run directory"
                )
            if not str(payload.get("content") or "").strip():
                return ToolDecision(False, "write_text_file requires non-empty content")
        return ToolDecision(True, "")

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        detail = f"{name} returned {'failure' if failed else 'success'}"
        mark_plan_progress(state, "tool_result", detail)
        if failed:
            errors = state.setdefault("tool_errors", [])
            errors.append(f"{name}({payload}) -> {result[:300]}")
        return ResultDecision(result=result)


class MiddlewareManager:
    def __init__(self, middlewares: list[Middleware]) -> None:
        self.middlewares = middlewares

    def before_model(self, state: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        for middleware in self.middlewares:
            chunks.extend(middleware.before_model(state))
        return [chunk for chunk in chunks if chunk.strip()]

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        current = [dict(message) for message in messages]
        blocked: list[str] = []
        content: str | None = None
        decision_type = ""
        terminate = False
        for middleware in self.middlewares:
            decision = middleware.guard_model_input(state, current)
            if decision.messages is not None:
                current = [dict(message) for message in decision.messages]
            if not decision.allowed:
                blocked.append(decision.reason)
                if decision.content is not None:
                    content = decision.content
            decision_type = decision.decision_type or decision_type
            terminate = terminate or decision.terminate
        return ModelDecision(
            allowed=not blocked,
            reason="; ".join(reason for reason in blocked if reason),
            messages=current,
            content=content,
            decision_type=decision_type,
            terminate=terminate,
        )

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        current_content = content
        current_call = tool_call
        blocked: list[str] = []
        retry = False
        terminate = False
        feedback: list[str] = []
        decision_type = ""
        for middleware in self.middlewares:
            decision = middleware.guard_model_output(
                state, current_content, current_call
            )
            if decision.content is not None:
                current_content = decision.content
            if decision.tool_call is not _UNCHANGED:
                current_call = decision.tool_call
            if not decision.allowed:
                blocked.append(decision.reason)
            if decision.retry:
                retry = True
                if decision.feedback:
                    feedback.append(decision.feedback)
            terminate = terminate or decision.terminate
            decision_type = decision.decision_type or decision_type
        return ModelDecision(
            allowed=not blocked,
            reason="; ".join(reason for reason in blocked if reason),
            content=current_content,
            tool_call=current_call,
            retry=retry and not terminate,
            feedback="; ".join(feedback),
            decision_type=decision_type,
            terminate=terminate,
        )

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        for middleware in self.middlewares:
            middleware.after_model(state, content, tool_call)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        allowed = True
        reasons: list[str] = []
        tool_name = name
        arguments = dict(payload)
        replacement_result: str | None = None
        decision_type = ""
        terminate = False
        for middleware in self.middlewares:
            decision = middleware.before_tool(state, tool_name, dict(arguments))
            allowed = allowed and decision.allowed
            if decision.reason and not decision.allowed:
                reasons.append(decision.reason)
            if decision.tool_name is not None:
                tool_name = decision.tool_name
            if decision.arguments is not None:
                arguments = dict(decision.arguments)
            if decision.replacement_result is not None:
                replacement_result = decision.replacement_result
            decision_type = decision.decision_type or decision_type
            terminate = terminate or decision.terminate
        return ToolDecision(
            allowed=allowed,
            reason="; ".join(reasons),
            tool_name=tool_name,
            arguments=arguments,
            replacement_result=replacement_result,
            decision_type=decision_type,
            terminate=terminate,
        )

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        current: Any = result
        allowed = True
        reasons: list[str] = []
        decision_type = ""
        for middleware in self.middlewares:
            decision = middleware.after_tool(state, name, payload, result, failed)
            allowed = allowed and decision.allowed
            if decision.reason and not decision.allowed:
                reasons.append(decision.reason)
            if decision.result is not _UNCHANGED:
                current = decision.result
            decision_type = decision.decision_type or decision_type
        return ResultDecision(
            allowed=allowed,
            reason="; ".join(reasons),
            result=current,
            decision_type=decision_type,
        )


def build_middleware_manager(cfg: AppConfig) -> MiddlewareManager:
    middlewares: list[Middleware] = []
    if cfg.aegis.enabled:
        middlewares.append(AegisGuardMiddleware(cfg))
    if cfg.progent.enabled:
        from .progent import ProgentMiddleware

        middlewares.append(ProgentMiddleware(cfg))
    if cfg.clawsentry.enabled:
        from .clawsentry import ClawSentryMiddleware

        middlewares.append(ClawSentryMiddleware(cfg))
    if cfg.janus.enabled:
        from .janus import JanusMiddleware

        middlewares.append(JanusMiddleware(cfg))
    if cfg.stepguard.enabled:
        from .stepguard import StepGuardMiddleware

        middlewares.append(StepGuardMiddleware(cfg))
    if cfg.adr.enabled:
        from .adr import ADRMiddleware

        middlewares.append(ADRMiddleware(cfg))
    if cfg.middleware.enabled:
        middlewares.append(HarnessMiddleware(cfg))
    if cfg.pro2guard.enabled:
        from .pro2guard import Pro2GuardMiddleware

        middlewares.append(Pro2GuardMiddleware(cfg))
    if cfg.agentspec.enabled:
        from .agentspec import AgentSpecMiddleware

        middlewares.append(AgentSpecMiddleware(cfg))
    if cfg.toolsafe.enabled:
        from .toolsafe import ToolSafeMiddleware

        middlewares.append(ToolSafeMiddleware(cfg))
    if cfg.agentdog.enabled:
        from .agentdog import AgentDoGMiddleware

        # AgentDoG diagnoses the effective accumulated trajectory. It runs after
        # tool-scoped guards and immediately before AgentGuard's output policy.
        middlewares.append(AgentDoGMiddleware(cfg))
    if cfg.agentguard.enabled:
        from .agentguard import AgentGuardMiddleware

        middlewares.append(AgentGuardMiddleware(cfg))
    if cfg.llamafirewall.enabled:
        from .llamafirewall import LlamaFirewallMiddleware

        # Run last so it scans the effective output and its result isolation wins.
        middlewares.append(LlamaFirewallMiddleware(cfg))
    if cfg.safeagent.enabled:
        from .safeagent import SafeAgentMiddleware

        # Enforce the Core's decision on the effective values from earlier guards.
        middlewares.append(SafeAgentMiddleware(cfg))
    return MiddlewareManager(middlewares)


def output_revision_limit(cfg: AppConfig) -> int:
    limits = [0]
    if cfg.agentguard.enabled:
        limits.append(max(0, cfg.agentguard.max_steps - 1))
    if cfg.agentdog.enabled and cfg.agentdog.mode == "revise":
        limits.append(cfg.agentdog.max_revisions)
    if cfg.safeagent.enabled:
        limits.append(cfg.safeagent.max_replans)
    return max(limits)
