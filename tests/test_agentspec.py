from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_scaffold.agentspec.middleware import AgentSpecMiddleware, Evaluation
from agent_scaffold.config import load_config
from agent_scaffold.container_runtime import _ensure_image
from agent_scaffold.middleware import Middleware, MiddlewareManager, ToolDecision


class _Rule:
    def __init__(self, text: str) -> None:
        self.raw = text
        self.id = re.search(r"rule\s+@(\w+)", text).group(1)
        self.event = re.search(r"trigger\s+(\w+)", text).group(1)


class _Runtime:
    def __init__(self, enforcements: dict[str, str]) -> None:
        self.enforcements = enforcements
        self.calls: list[dict[str, Any]] = []

    def load_rule(self, text: str) -> _Rule:
        return _Rule(text)

    def evaluate(self, rule: _Rule, **kwargs: Any) -> Evaluation:
        self.calls.append({"rule": rule.id, **kwargs})
        return Evaluation(rule.id, rule.event, self.enforcements[rule.id])


def _rule(rule_id: str, event: str = "send_email") -> str:
    return f"""rule @{rule_id}
trigger
    {event}
check
    true
enforce
    stop
end"""


def _config(
    tmp_path: Path,
    rules: list[str],
    **overrides: Any,
) -> SimpleNamespace:
    values = {
        "enabled": True,
        "rules": rules,
        "rule_files": [],
        "predicate_modules": [],
        "approval_handler": "prompt",
        "max_reflections": 3,
        "fail_closed": True,
    }
    values.update(overrides)
    return SimpleNamespace(
        agentspec=SimpleNamespace(**values), config_dir=str(tmp_path)
    )


def _state() -> dict[str, Any]:
    return {
        "_toolsafe_user_request": "Send the weekly report",
        "messages": [{"role": "user", "content": "Send the weekly report"}],
        "trace": [
            {
                "step": "tool",
                "input": {"tool": "read_file", "args": {"path": "report"}},
                "output": {"result": "ok"},
            }
        ],
    }


def test_stop_terminates_the_run_and_records_rule(tmp_path: Path) -> None:
    runtime = _Runtime({"no_send": "stop"})
    middleware = AgentSpecMiddleware(
        _config(tmp_path, [_rule("no_send")]), runtime=runtime
    )
    state = _state()

    decision = middleware.before_tool(
        state, "send_email", {"to": "outside@example.com"}
    )

    assert not decision.allowed
    assert decision.terminate
    assert decision.decision_type == "stop"
    assert state["_last_agentspec_decision"]["rule_id"] == "no_send"
    assert runtime.calls[0]["user_input"] == {"input": "Send the weekly report"}
    assert runtime.calls[0]["intermediate_steps"]


def test_unmatched_trigger_does_not_run_predicates(tmp_path: Path) -> None:
    runtime = _Runtime({"no_send": "stop"})
    middleware = AgentSpecMiddleware(
        _config(tmp_path, [_rule("no_send")]), runtime=runtime
    )

    decision = middleware.before_tool(_state(), "read_file", {"path": "report"})

    assert decision.allowed
    assert runtime.calls == []


def test_self_reflection_replans_then_degrades_to_skip(tmp_path: Path) -> None:
    runtime = _Runtime({"safer_send": "llm_self_reflect"})
    middleware = AgentSpecMiddleware(
        _config(tmp_path, [_rule("safer_send")], max_reflections=1),
        runtime=runtime,
    )
    state = _state()

    first = middleware.before_tool(state, "send_email", {})
    second = middleware.before_tool(state, "send_email", {})

    assert not first.allowed and not first.terminate
    assert first.decision_type == "llm_self_reflect"
    assert "revise the plan" in str(first.replacement_result)
    assert second.decision_type == "skip"
    assert "budget" in second.reason


def test_user_inspection_uses_configured_callback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    approval_module = types.ModuleType("test_agentspec_approval")
    approval_module.approve = lambda **kwargs: kwargs["tool_name"] == "send_email"
    monkeypatch.setitem(sys.modules, approval_module.__name__, approval_module)
    runtime = _Runtime({"inspect": "user_inspection"})
    middleware = AgentSpecMiddleware(
        _config(
            tmp_path,
            [_rule("inspect")],
            approval_handler="test_agentspec_approval:approve",
        ),
        runtime=runtime,
    )

    decision = middleware.before_tool(_state(), "send_email", {"to": "a@b.test"})

    assert decision.allowed
    assert decision.decision_type == "allow"


def test_invoke_action_replaces_call_for_later_middleware(tmp_path: Path) -> None:
    runtime = _Runtime({"replace": 'invoke_action(safe_send,{"recipient":"admin"})'})
    agentspec = AgentSpecMiddleware(
        _config(tmp_path, [_rule("replace")]), runtime=runtime
    )

    class _ObserveEffectiveCall(Middleware):
        def before_tool(
            self, state: dict[str, Any], name: str, payload: dict[str, Any]
        ) -> ToolDecision:
            state["observed"] = (name, payload)
            return ToolDecision()

    state = _state()
    decision = MiddlewareManager([agentspec, _ObserveEffectiveCall()]).before_tool(
        state, "send_email", {"to": "attacker"}
    )

    assert decision.allowed
    assert decision.tool_name == "safe_send"
    assert decision.arguments == {"recipient": "admin"}
    assert state["observed"] == ("safe_send", {"recipient": "admin"})


def test_fail_closed_on_runtime_initialization_error(tmp_path: Path) -> None:
    class _BrokenRuntime:
        def load_rule(self, text: str) -> Any:
            raise RuntimeError("parser unavailable")

    middleware = AgentSpecMiddleware(
        _config(tmp_path, [_rule("broken")]), runtime=_BrokenRuntime()
    )

    decision = middleware.before_tool(_state(), "send_email", {})

    assert not decision.allowed
    assert decision.terminate
    assert "failed closed" in decision.reason


def test_load_config_supports_relative_rule_files(tmp_path: Path) -> None:
    (tmp_path / "policy.ar").write_text(_rule("from_file"), encoding="utf-8")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        """llm:
  provider: openai
  model: test
agent:
  name: test
  system_prompt: test
tools: []
graph:
  type: single_agent
monitoring: {}
agentspec:
  enabled: true
  rule_files: [policy.ar]
  max_reflections: 2
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agentspec.enabled
    assert config.agentspec.rule_files == ["policy.ar"]
    assert config.agentspec.max_reflections == 2


def test_container_rebuilds_agentdojo_image_with_agentspec(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = SimpleNamespace(
        agentspec=SimpleNamespace(enabled=True),
        container=SimpleNamespace(
            image="agent-scaffold-agentdojo:latest",
            auto_build=True,
            dockerfile="Dockerfile",
            build_args={"INSTALL_AGENTDOJO": "true"},
        ),
    )
    calls: list[list[str]] = []
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="missing"),
            SimpleNamespace(returncode=0, stdout="built", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(cmd: list[str], cwd: Path) -> Any:
        calls.append(cmd)
        return next(responses)

    monkeypatch.setattr("agent_scaffold.container_runtime._run_checked", fake_run)

    _ensure_image(cfg, tmp_path)

    build = next(cmd for cmd in calls if cmd[:2] == ["docker", "build"])
    assert "--build-arg" in build
    assert "INSTALL_AGENTDOJO=true" in build
    assert "INSTALL_AGENTSPEC=true" in build
    probes = [cmd for cmd in calls if cmd[:3] == ["docker", "run", "--rm"]]
    assert len(probes) == 2


def test_container_reuses_image_when_agentspec_is_importable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = SimpleNamespace(
        agentspec=SimpleNamespace(enabled=True),
        container=SimpleNamespace(
            image="agent-scaffold-agentdojo:latest",
            auto_build=True,
            dockerfile="Dockerfile",
            build_args={"INSTALL_AGENTDOJO": "true"},
        ),
    )
    calls: list[list[str]] = []
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    def fake_run(cmd: list[str], cwd: Path) -> Any:
        calls.append(cmd)
        return next(responses)

    monkeypatch.setattr("agent_scaffold.container_runtime._run_checked", fake_run)

    _ensure_image(cfg, tmp_path)

    assert not any(cmd[:2] == ["docker", "build"] for cmd in calls)
