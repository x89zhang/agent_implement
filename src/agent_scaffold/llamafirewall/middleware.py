from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolExecutionTerminated


def guard_react_actions(turn: Any, manager: Any, state: dict[str, Any]) -> Any:
    """Scan parsed actions, including their reasoning, before executor dispatch.

    Final answers are scanned by the existing final-output hook. Do not run
    other guards here: their before_tool hooks already enforce their policies.
    """
    actions = turn if isinstance(turn, list) else [turn]
    for action in actions:
        if not hasattr(action, "tool"):
            continue
        arguments = action.tool_input
        if not isinstance(arguments, dict):
            arguments = {"__arg": arguments}
        for guard in manager.middlewares:
            if not isinstance(guard, LlamaFirewallMiddleware):
                continue
            verdict = guard.guard_model_output(
                state, str(action.log or ""), (action.tool, arguments)
            )
            if not verdict.allowed:
                raise ToolExecutionTerminated(
                    verdict.content or verdict.reason, action.tool, arguments
                )
    return turn


@dataclass
class _Runtime:
    module: Any

    @classmethod
    def load(cls) -> "_Runtime":
        return cls(importlib.import_module("llamafirewall"))

    def message(self, role: str, content: str, tool_call: Any = None) -> Any:
        if role == "system":
            return self.module.SystemMessage(content=content)
        if role == "user":
            return self.module.UserMessage(content=content)
        if role == "assistant":
            tool_calls = None
            if tool_call:
                name, arguments = tool_call
                tool_calls = [{"name": name, "arguments": arguments}]
                rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                content = (
                    f"{content}\n\nSELECTED ACTION:\nACTION: {name}\n"
                    f"ACTION INPUT: {rendered}"
                ).strip()
            return self.module.AssistantMessage(
                content=content, tool_calls=tool_calls
            )
        if role == "tool":
            return self.module.ToolMessage(content=content)
        raise ValueError(f"Unsupported LlamaFirewall role: {role}")

    def build(self, cfg: Any) -> Any:
        if cfg.factory:
            module_name, function_name = cfg.factory.rsplit(":", 1)
            factory = getattr(importlib.import_module(module_name), function_name)
            return factory(**cfg.factory_kwargs)
        if cfg.scanners:
            scanners: dict[Any, list[Any]] = {}
            for role_name, scanner_names in cfg.scanners.items():
                role = self.module.Role(role_name)
                resolved: list[Any] = []
                for scanner_name in scanner_names:
                    try:
                        resolved.append(self.module.ScannerType(scanner_name))
                    except ValueError:
                        # LlamaFirewall supports registered custom scanners by name.
                        resolved.append(scanner_name)
                scanners[role] = resolved
            return self.module.LlamaFirewall(scanners=scanners)
        if cfg.use_case:
            return self.module.LlamaFirewall.from_usecase(
                self.module.UseCase(cfg.use_case)
            )
        return self.module.LlamaFirewall()


class LlamaFirewallMiddleware(Middleware):
    """Role-aware adapter around Meta's optional llamafirewall package."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        runtime: _Runtime | None = None,
        firewall: Any = None,
    ) -> None:
        self.cfg = cfg
        self.options = cfg.llamafirewall
        self.runtime = runtime
        self.firewall = firewall
        self._init_error = ""
        try:
            self.runtime = self.runtime or _Runtime.load()
            self.firewall = self.firewall or self.runtime.build(self.options)
        except Exception as exc:
            self._init_error = str(exc)

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        if state.get("_llamafirewall_initial_input_scanned"):
            return ModelDecision(messages=messages)

        state["_llamafirewall_initial_input_scanned"] = True
        results: list[dict[str, Any]] = []
        # A run starts with system/user context. Later assistant and tool events are
        # ingested by their dedicated lifecycle hooks instead of being rescanned.
        for raw in messages:
            role = str(raw.get("role", "")).lower()
            if role == "assistant":
                break
            if role not in {"system", "user"}:
                continue
            content = str(raw.get("content") or "")
            if content:
                result = self._scan(state, f"{role}_input", role, content)
                results.append(result)
                if not self._permitted(result):
                    break

        decisive = self._decisive(results)
        if not decisive or self._permitted(decisive):
            return ModelDecision(messages=messages)
        return ModelDecision(
            allowed=False,
            reason=decisive["reason"],
            messages=messages,
            content=self._withheld(decisive),
            decision_type=decisive["decision"],
            terminate=True,
        )

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        result = self._scan(
            state, "assistant_output", "assistant", content, tool_call=tool_call
        )
        if self._permitted(result):
            return ModelDecision(content=content, tool_call=tool_call)

        return ModelDecision(
            allowed=False,
            reason=result["reason"],
            content=self._withheld(result),
            tool_call=None,
            retry=False,
            decision_type=result["decision"],
            terminate=True,
        )

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        scan = self._scan(state, "tool_output", "tool", result)
        if self._permitted(scan):
            return ResultDecision(result=result)
        # Stop on the native verdict without feeding fabricated observations back.
        raise ToolExecutionTerminated(self._withheld(scan), name, payload)

    def _scan(
        self,
        state: dict[str, Any],
        phase: str,
        role: str,
        content: str,
        *,
        tool_call: Any = None,
    ) -> dict[str, Any]:
        try:
            if self._init_error:
                raise RuntimeError(self._init_error)
            assert self.runtime is not None and self.firewall is not None
            message = self.runtime.message(role, content, tool_call)
            trace = state.get("_llamafirewall_trace") or []
            native, updated_trace = self.firewall.scan_replay_build_trace(
                message, trace
            )
            # Monitor executes rejected messages, so subsequent scans need them.
            # Enforce retains the native allow-only trace.
            state["_llamafirewall_trace"] = (
                trace + [message] if self.options.mode == "monitor" else updated_trace
            )
            event = {
                "phase": phase,
                "role": role,
                "decision": self._enum_value(native.decision),
                "reason": str(native.reason),
                "score": float(native.score),
                "status": self._enum_value(native.status),
                "mode": self.options.mode,
            }
        except Exception as exc:
            event = {
                "phase": phase,
                "role": role,
                "decision": "block" if self.options.fail_closed else "allow",
                "reason": (
                    "LlamaFirewall failed "
                    f"{'closed' if self.options.fail_closed else 'open'}: {exc}"
                ),
                "score": None,
                "status": "error",
                "mode": self.options.mode,
            }
        state["_last_llamafirewall_decision"] = event
        state.setdefault("llamafirewall_events", []).append(event)
        return event

    def _permitted(self, result: dict[str, Any]) -> bool:
        return self.options.mode == "monitor" or result["decision"] == "allow"

    @staticmethod
    def _decisive(results: list[dict[str, Any]]) -> dict[str, Any] | None:
        rank = {"allow": 0, "human_in_the_loop_required": 1, "block": 2}
        if not results:
            return None
        return max(results, key=lambda item: rank.get(item["decision"], 2))

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value)).lower()

    @staticmethod
    def _withheld(result: dict[str, Any]) -> str:
        return json.dumps(
            {
                "llamafirewall": result["decision"],
                "phase": result["phase"],
                "reason": result["reason"],
            },
            ensure_ascii=False,
        )
