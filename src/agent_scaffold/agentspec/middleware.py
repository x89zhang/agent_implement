from __future__ import annotations

import importlib
import json
import re
import sys
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import AppConfig
from ..middleware import Middleware, ToolDecision


@dataclass
class Evaluation:
    rule_id: str
    event: str
    enforcement: str


_MISSING = object()


class UpstreamAgentSpecRuntime:
    """Thin adapter over the modules installed by haoyuwang99/AgentSpec."""

    def __init__(self, predicate_modules: list[str]) -> None:
        try:
            self.rule_module = importlib.import_module("rule")
            self.agent_module = importlib.import_module("agent")
            previous_state = sys.modules.get("state", _MISSING)
            self.state_module = _install_upstream_state_compatibility()
            try:
                self.interpreter_module = importlib.import_module("interpreter")
            finally:
                _restore_state_module(previous_state)
            table_module = importlib.import_module("rules.manual.table")
        except ImportError as exc:
            raise RuntimeError(
                "The ICSE AgentSpec package is not installed. Install it from "
                "git+https://github.com/haoyuwang99/AgentSpec.git; the unrelated "
                "PyPI project named 'agentspec' is not compatible."
            ) from exc
        self.predicate_table = table_module.predicate_table
        for spec in predicate_modules:
            self._register_predicates(spec)

    def _register_predicates(self, import_path: str) -> None:
        module_name, separator, attribute = import_path.partition(":")
        module = importlib.import_module(module_name)
        source = getattr(module, attribute) if separator else module
        if callable(source):
            result = source(self.predicate_table)
            if result is not None:
                self.predicate_table.update(dict(result))
            return
        if not isinstance(source, Mapping):
            source = getattr(
                source, "PREDICATES", getattr(source, "predicate_table", None)
            )
        if not isinstance(source, Mapping):
            raise TypeError(
                f"AgentSpec predicate provider '{import_path}' must be a mapping or "
                "a callable accepting the upstream predicate table"
            )
        self.predicate_table.update(source)

    def validate_rule(self, text: str) -> Any:
        """Parse a rule with upstream AgentSpec without executing predicates."""
        input_stream = self.interpreter_module.InputStream(text)
        lexer = self.interpreter_module.AgentSpecLexer(input_stream)
        token_stream = self.interpreter_module.CommonTokenStream(lexer)
        parser = self.interpreter_module.AgentSpecParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(self.interpreter_module.CustomErrorListener())
        parser.program()
        if parser.getNumberOfSyntaxErrors():
            raise ValueError("AgentSpec parser rejected the generated rule")
        return self.rule_module.Rule.from_text(text)

    def supported_predicates(self) -> list[str]:
        """Return installed predicates that are also accepted by this grammar."""
        supported = ["true"]
        for name in sorted(self.predicate_table):
            probe = (
                "rule @predicate_probe\n"
                "trigger\npredicate_probe_tool\n"
                f"check\n{name}\n"
                "enforce\nnone\nend\n"
            )
            try:
                self.validate_rule(probe)
            except Exception:  # noqa: BLE001, S112 - grammar capability probe
                continue
            supported.append(str(name))
        return supported

    def load_rule(self, text: str) -> Any:
        return self.validate_rule(text)

    def evaluate(
        self,
        rule: Any,
        *,
        user_input: dict[str, Any],
        tool_name: str,
        tool_input: Any,
        intermediate_steps: list[Any],
    ) -> Evaluation:
        action = self.agent_module.Action(name=tool_name, input=tool_input, action=None)
        state = self.state_module.RuleState(
            action=action,
            agent=None,
            intermediate_steps=intermediate_steps,
            user_input=user_input,
        )
        interpreter = self.interpreter_module.RuleInterpreter(rule, state)

        # Use AgentSpec's generated parser and listener to evaluate triggers and
        # predicates. Enforcement is mapped by this scaffold after the walk so
        # self-reflection can re-enter its own agent loop instead of AgentSpec's
        # LangChain-specific executor.
        input_stream = self.interpreter_module.InputStream(rule.raw)
        lexer = self.interpreter_module.AgentSpecLexer(input_stream)
        token_stream = self.interpreter_module.CommonTokenStream(lexer)
        parser = self.interpreter_module.AgentSpecParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(self.interpreter_module.CustomErrorListener())
        tree = parser.program()
        walker = self.interpreter_module.ParseTreeWalker()
        walker.walk(interpreter, tree)
        return Evaluation(
            rule_id=str(rule.id),
            event=str(rule.event),
            enforcement=str(interpreter.enforce),
        )


def _install_upstream_state_compatibility() -> Any:
    """Supply the data-only RuleState expected by AgentSpec on LangChain 1.x.

    Upstream imports legacy LangChain agent base classes that were removed in
    1.x. The rule interpreter only needs RuleState as a mutable context object,
    so this shim keeps the upstream parser/interpreter usable without forcing
    the host application onto LangChain 0.3.
    """

    existing = sys.modules.get("state")
    if existing is not None and getattr(existing, "__agentspec_compat__", False):
        return existing

    class CompatibleRuleState(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        toolkit: str = ""
        action: Any = None
        agent: Any = None
        intermediate_steps: Any = Field(default_factory=list)
        user_input: Any = None
        run_mannager: Any = None
        merits: list[str] = Field(default_factory=list)
        critiques: list[str] = Field(default_factory=list)
        reflection_depth: int = 0

    module = types.ModuleType("state")
    module.RuleState = CompatibleRuleState
    module.__agentspec_compat__ = True
    sys.modules["state"] = module
    return module


def _restore_state_module(previous: Any) -> None:
    if previous is _MISSING:
        sys.modules.pop("state", None)
    else:
        sys.modules["state"] = previous


class AgentSpecMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, runtime: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.agentspec
        self._runtime = runtime
        self._rules: list[Any] = []
        self._init_error = ""
        try:
            if self._runtime is None:
                self._runtime = UpstreamAgentSpecRuntime(
                    self.settings.predicate_modules
                )
            self._rules = [
                self._runtime.load_rule(text) for text in self._load_rule_texts()
            ]
            if not self._rules:
                raise ValueError("AgentSpec is enabled but no rules were configured")
        except Exception as exc:  # noqa: BLE001 - configured failure boundary
            self._init_error = str(exc)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        if self._init_error:
            return self._failure_decision(state, name, payload, self._init_error)

        effective_name = name
        effective_payload = dict(payload)
        replacement: tuple[str, dict[str, Any]] | None = None
        try:
            for rule in self._rules:
                tool_input = _tool_input(effective_payload)
                if not _triggered(rule, effective_name, tool_input):
                    continue
                evaluated = self._runtime.evaluate(
                    rule,
                    user_input={"input": _user_request(state)},
                    tool_name=effective_name,
                    # Upstream AgentSpec predicates use string operations such
                    # as re.search(), .find(), and .lower() on this value.
                    # Native tool runtimes normally supply structured objects,
                    # so serialize them at the adapter boundary while keeping
                    # the original payload for execution and audit records.
                    tool_input=_predicate_input(tool_input),
                    intermediate_steps=_intermediate_steps(state),
                )
                enforcement = evaluated.enforcement
                if self.settings.mode == "monitor":
                    self._record(
                        state, evaluated, "monitor", True,
                        _reason(evaluated, f"requested {enforcement}; monitor allowed the original call"),
                    )
                    continue
                if enforcement == "none":
                    self._record(state, evaluated, "allow", True, "")
                    continue
                if enforcement == "stop":
                    reason = _reason(evaluated, "stopped the agent run")
                    return self._deny(state, evaluated, "stop", reason, terminate=True)
                if enforcement == "skip":
                    reason = _reason(evaluated, "skipped the proposed tool call")
                    return self._deny(state, evaluated, "skip", reason)
                if enforcement in {"llm_self_reflect", "llm_self_examine"}:
                    return self._self_reflect(state, evaluated)
                if enforcement == "user_inspection":
                    approved = self._approve(
                        state, evaluated, effective_name, effective_payload
                    )
                    action = "user_approved" if approved else "user_rejected"
                    reason = _reason(
                        evaluated,
                        "was approved by the user"
                        if approved
                        else "was rejected by the user",
                    )
                    self._record(state, evaluated, action, approved, reason)
                    if approved:
                        continue
                    return ToolDecision(
                        False,
                        reason,
                        replacement_result=_feedback(evaluated, action),
                        decision_type=action,
                    )
                if enforcement.startswith("invoke_action("):
                    effective_name, effective_payload = _parse_invoke_action(
                        enforcement
                    )
                    replacement = (effective_name, effective_payload)
                    self._record(
                        state,
                        evaluated,
                        "invoke_action",
                        True,
                        _reason(
                            evaluated, f"replaced the action with {effective_name}"
                        ),
                    )
                    continue
                raise ValueError(f"unsupported AgentSpec enforcement: {enforcement}")
        except Exception as exc:  # noqa: BLE001 - configured failure boundary
            return self._failure_decision(state, name, payload, str(exc))

        if replacement is not None:
            return ToolDecision(
                True,
                "",
                tool_name=replacement[0],
                arguments=replacement[1],
                decision_type="invoke_action",
            )
        return ToolDecision(True, "", decision_type=(
            "monitor" if self.settings.mode == "monitor" else "allow"
        ))

    def _self_reflect(
        self, state: dict[str, Any], evaluated: Evaluation
    ) -> ToolDecision:
        counts = state.setdefault("_agentspec_reflection_counts", {})
        count = int(counts.get(evaluated.rule_id, 0)) + 1
        counts[evaluated.rule_id] = count
        if count > self.settings.max_reflections:
            reason = _reason(
                evaluated,
                f"exhausted its reflection budget ({self.settings.max_reflections})",
            )
            self._record(state, evaluated, "skip", False, reason)
            return ToolDecision(
                False,
                reason,
                replacement_result=_feedback(evaluated, "skip", reason),
                decision_type="skip",
            )
        reason = _reason(evaluated, "requested a safer plan")
        self._record(state, evaluated, "llm_self_reflect", False, reason)
        return ToolDecision(
            False,
            reason,
            replacement_result=_feedback(evaluated, "llm_self_reflect", reason),
            decision_type="llm_self_reflect",
        )

    def _approve(
        self,
        state: dict[str, Any],
        evaluated: Evaluation,
        name: str,
        payload: dict[str, Any],
    ) -> bool:
        handler_path = self.settings.approval_handler.strip()
        if handler_path in {"", "prompt"}:
            answer = input(
                f"AgentSpec rule @{evaluated.rule_id} requests approval for "
                f"{name}({payload}). Continue? [y/N] "
            )
            return answer.strip().lower() in {"y", "yes"}
        handler = _load_attribute(handler_path)
        return bool(
            handler(
                rule_id=evaluated.rule_id,
                event=evaluated.event,
                tool_name=name,
                arguments=dict(payload),
                state=state,
            )
        )

    def _deny(
        self,
        state: dict[str, Any],
        evaluated: Evaluation,
        action: str,
        reason: str,
        *,
        terminate: bool = False,
    ) -> ToolDecision:
        self._record(state, evaluated, action, False, reason)
        return ToolDecision(
            False,
            reason,
            replacement_result=_feedback(evaluated, action, reason),
            decision_type=action,
            terminate=terminate,
        )

    def _failure_decision(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        error: str,
    ) -> ToolDecision:
        allowed = self.settings.mode == "monitor" or not self.settings.fail_closed
        reason = (
            f"AgentSpec failed {'open' if allowed else 'closed'}: "
            f"{error}"
        )
        data = {
            "rule_id": "",
            "event": name,
            "enforcement": "error",
            "mode": self.settings.mode,
            "action": "error",
            "allowed": allowed,
            "reason": reason,
            "tool_name": name,
            "arguments": dict(payload),
        }
        state["_last_agentspec_decision"] = data
        state.setdefault("agentspec_events", []).append(data)
        return ToolDecision(
            allowed,
            "" if allowed else reason,
            replacement_result=None if allowed else reason,
            decision_type="error",
            terminate=not allowed,
        )

    def _record(
        self,
        state: dict[str, Any],
        evaluated: Evaluation,
        action: str,
        allowed: bool,
        reason: str,
    ) -> None:
        data = {
            "rule_id": evaluated.rule_id,
            "event": evaluated.event,
            "enforcement": evaluated.enforcement,
            "mode": self.settings.mode,
            "action": action,
            "allowed": allowed,
            "reason": reason,
        }
        state["_last_agentspec_decision"] = data
        state.setdefault("agentspec_events", []).append(data)

    def _load_rule_texts(self) -> list[str]:
        texts = [item.strip() for item in self.settings.rules if item.strip()]
        for configured in self.settings.rule_files:
            path = Path(configured)
            if not path.is_absolute():
                path = Path(self.cfg.config_dir) / path
            content = path.read_text(encoding="utf-8")
            texts.extend(_split_rule_file(content, path))
        return texts


def _triggered(rule: Any, name: str, tool_input: Any) -> bool:
    event = str(rule.event)
    rendered = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return (
        event == "any"
        or name == event
        or rendered.strip().startswith(event.replace("_", " "))
    )


def _tool_input(payload: dict[str, Any]) -> Any:
    if set(payload) == {"__arg"}:
        return payload["__arg"]
    return dict(payload)


def _predicate_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _user_request(state: dict[str, Any]) -> str:
    explicit = str(
        state.get("_agentspec_user_request")
        or state.get("_toolsafe_user_request")
        or ""
    ).strip()
    if explicit:
        return explicit
    for message in state.get("messages", []) or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content and not content.startswith("TOOL_RESULT:"):
                return content
    return ""


def _intermediate_steps(state: dict[str, Any]) -> list[Any]:
    steps: list[Any] = []
    for item in state.get("trace", []) or []:
        if not isinstance(item, dict) or item.get("step") != "tool":
            continue
        steps.append((item.get("input", {}), item.get("output", {})))
    return steps


def _split_rule_file(content: str, path: Path) -> list[str]:
    matches = re.findall(r"(?ms)^\s*rule\s+@\w+.*?^\s*end\s*$", content)
    if not matches and content.strip():
        raise ValueError(f"no AgentSpec rules found in {path}")
    return [match.strip() for match in matches]


def _parse_invoke_action(text: str) -> tuple[str, dict[str, Any]]:
    match = re.fullmatch(r"invoke_action\((\w+),(\{.*\})\)", text)
    if not match:
        raise ValueError(f"invalid AgentSpec invoke_action: {text}")
    arguments = json.loads(match.group(2))
    if not isinstance(arguments, dict):
        raise TypeError("AgentSpec invoke_action arguments must be an object")
    return match.group(1), arguments


def _load_attribute(import_path: str) -> Callable[..., Any]:
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "agentspec.approval_handler must be 'prompt' or use "
            "'module:attribute' import syntax"
        )
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"AgentSpec approval handler '{import_path}' is not callable")
    return value


def _reason(evaluated: Evaluation, outcome: str) -> str:
    return f"AgentSpec rule @{evaluated.rule_id} {outcome}"


def _feedback(evaluated: Evaluation, action: str, reason: str | None = None) -> str:
    details = {
        "rule_id": evaluated.rule_id,
        "event": evaluated.event,
        "enforcement": evaluated.enforcement,
        "action": action,
        "reason": reason or _reason(evaluated, action),
    }
    if action == "llm_self_reflect":
        guidance = (
            "The proposed tool call was not executed. Examine the violated rule "
            "and revise the plan while still pursuing the original request safely."
        )
    elif action == "stop":
        guidance = (
            "The proposed tool call was not executed and this agent run was stopped."
        )
    else:
        guidance = "The proposed tool call was not executed."
    return (
        "AgentSpec runtime enforcement:\n"
        + json.dumps(details, ensure_ascii=False)
        + "\n"
        + guidance
    )
