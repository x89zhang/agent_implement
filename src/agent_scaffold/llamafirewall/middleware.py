from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolExecutionTerminated
from ..tool_results import unwrap_hermes_result

# Native scanners built on CustomCheckScanner call Together AI by default
# (custom_check_scanner.py: api.together.xyz, TOGETHER_API_KEY).
_TOGETHER_SCANNERS = {"agent_alignment", "pii_detection"}
_TOGETHER_KEY_ENV = "TOGETHER_API_KEY"
# CustomCheckScanner swallows every LLM failure and AlignmentCheck substitutes
# a "compromised" verdict with status SUCCESS (alignmentcheck_scanner.py
# _get_default_error_response). An unreachable or failing judge would then
# look like a detection; such scans are recorded as monitor errors instead.
_ALIGNMENT_ERROR_REASON = "Observation: Error occurred during evaluation\n"


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
        # Running ``python src/agent_scaffold/main.py`` prepends the package
        # directory to sys.path. Its local ``llamafirewall`` adapter would then
        # shadow Meta's installed top-level package and fail relative imports.
        adapter_parent = Path(__file__).resolve().parents[1]
        original_path = sys.path[:]
        try:
            sys.path[:] = [entry for entry in original_path
                           if Path(entry or ".").resolve() != adapter_parent]
            module = importlib.import_module("llamafirewall")
        finally:
            sys.path[:] = original_path
        if not hasattr(module, "LlamaFirewall") or not hasattr(module, "ScannerType"):
            raise ImportError("Installed LlamaFirewall package is unavailable")
        return cls(module)

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
        self._config_error = ""
        self._config_error_roles: set[str] = set()
        try:
            self.runtime = self.runtime or _Runtime.load()
            self.firewall = self.firewall or self.runtime.build(self.options)
        except Exception as exc:
            self._init_error = str(exc)
        else:
            self._check_credentials()

    def _check_credentials(self) -> None:
        """Detect a missing Together key before the first scan.

        Without it every AlignmentCheck scan raises inside the library. The
        run then has no verdict for those roles; record that as a
        configuration error rather than as clean scans.
        """
        if self.options.factory or os.environ.get(_TOGETHER_KEY_ENV):
            return  # factories configure their own scanner endpoints
        scanners = getattr(self.firewall, "scanners", None) or {}
        needed: set[str] = set()
        for role, items in scanners.items():
            names = {self._enum_value(item) for item in items} & _TOGETHER_SCANNERS
            if names:
                needed |= names
                self._config_error_roles.add(self._enum_value(role))
        if needed:
            self._config_error = (
                f"LlamaFirewall configuration error: scanner(s) {', '.join(sorted(needed))} "
                f"call Together AI (the library default endpoint and model) and "
                f"{_TOGETHER_KEY_ENV} is not set. Export {_TOGETHER_KEY_ENV} before "
                "the run; Hermes container configs forward it through container.env."
            )

    def guard_model_input(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelDecision:
        if state.get("_llamafirewall_initial_input_scanned"):
            return ModelDecision(messages=messages)

        state["_llamafirewall_initial_input_scanned"] = True
        results: list[dict[str, Any]] = []
        # Upstream's tool-using agent integration (examples/langchain_agent.py)
        # feeds the user prompt, each agent message and each tool output; the
        # agent's system prompt is not part of the trace. Later assistant and
        # tool events are ingested by their dedicated lifecycle hooks.
        for raw in messages:
            role = str(raw.get("role", "")).lower()
            if role == "assistant":
                break
            if role != "user":
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
        # langchain_agent.py scans str(ToolMessage.content): the tool's own
        # output, without Hermes' {"result": ...} MCP envelope.
        scan = self._scan(state, "tool_output", "tool", str(unwrap_hermes_result(result)))
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
        if self._config_error and role in self._config_error_roles:
            return self._record_config_error(state, phase, role)
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
            if event["reason"].startswith(_ALIGNMENT_ERROR_REASON):
                event["status"] = "error"
                event["error"] = "AlignmentCheck judge call failed; see the scanner log"
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

    def _record_config_error(
        self, state: dict[str, Any], phase: str, role: str
    ) -> dict[str, Any]:
        event = {
            "phase": phase,
            "role": role,
            "decision": "block" if self.options.fail_closed else "allow",
            "reason": self._config_error,
            "score": None,
            "status": "error",
            "error": self._config_error,
            "mode": self.options.mode,
        }
        state["_last_llamafirewall_decision"] = event
        if not state.get("_llamafirewall_config_error_recorded"):
            # Once per run: a run-level guard failure, not per-message noise.
            state["_llamafirewall_config_error_recorded"] = True
            state.setdefault("harness", {}).setdefault("guard_errors", {})[
                "llamafirewall"
            ] = self._config_error
            state.setdefault("llamafirewall_events", []).append(event)
            state.setdefault("trace", []).append({
                "step": "llamafirewall_configuration_error",
                "output": {"status": "error", "error": self._config_error},
            })
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
