from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .guard import check_tool_call
from .planner import render_plan_context, mark_plan_progress, complete_plan_on_final


@dataclass
class ToolDecision:
    allowed: bool = True
    reason: str = ""


class Middleware:
    def before_model(self, state: dict[str, Any]) -> list[str]:
        return []

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        return None

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        return ToolDecision()

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> None:
        return None


class AegisGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
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
            chunks.append(f"# Recent Tool Issues\nAvoid repeating these failed calls unless you have changed the arguments.\n{rendered}")
        return chunks

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        if tool_call:
            name = str(tool_call[0]) if isinstance(tool_call, tuple) else "tool"
            mark_plan_progress(state, "selected_tool", name)
            return
        if content.strip():
            complete_plan_on_final(state)

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        if name == "write_text_file":
            path = str(payload.get("path") or "")
            if ".." in path.replace("\\", "/").split("/"):
                return ToolDecision(False, "write_text_file path must stay under the run directory")
            if not str(payload.get("content") or "").strip():
                return ToolDecision(False, "write_text_file requires non-empty content")
        return ToolDecision(True, "")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> None:
        detail = f"{name} returned {'failure' if failed else 'success'}"
        mark_plan_progress(state, "tool_result", detail)
        if failed:
            errors = state.setdefault("tool_errors", [])
            errors.append(f"{name}({payload}) -> {result[:300]}")


class MiddlewareManager:
    def __init__(self, middlewares: list[Middleware]) -> None:
        self.middlewares = middlewares

    def before_model(self, state: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        for middleware in self.middlewares:
            chunks.extend(middleware.before_model(state))
        return [chunk for chunk in chunks if chunk.strip()]

    def after_model(self, state: dict[str, Any], content: str, tool_call: Any) -> None:
        for middleware in self.middlewares:
            middleware.after_model(state, content, tool_call)

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        for middleware in self.middlewares:
            decision = middleware.before_tool(state, name, payload)
            if not decision.allowed:
                return decision
        return ToolDecision()

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> None:
        for middleware in self.middlewares:
            middleware.after_tool(state, name, payload, result, failed)


def build_middleware_manager(cfg: AppConfig) -> MiddlewareManager:
    middlewares: list[Middleware] = []
    if cfg.aegis.enabled:
        middlewares.append(AegisGuardMiddleware(cfg))
    if cfg.middleware.enabled:
        middlewares.append(HarnessMiddleware(cfg))
    return MiddlewareManager(middlewares)
