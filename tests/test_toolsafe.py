from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_scaffold.config import (
    AppConfig,
    ToolConfig,
    ToolSafeConfig,
    load_config,
)
from agent_scaffold.toolsafe.client import _chat_completions_url
from agent_scaffold.toolsafe.middleware import ToolSafeMiddleware
from agent_scaffold.middleware import (
    ResultDecision,
    Middleware,
    MiddlewareManager,
    ToolDecision,
    ToolExecutionTerminated,
)
from agent_scaffold.graph import _build_traced_react_tool
from agent_scaffold.toolsafe.parser import parse_guard_response


class _FakeClient:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


def _config(**overrides: Any) -> AppConfig:
    guard = ToolSafeConfig(enabled=True, **overrides)
    return AppConfig(
        llm=SimpleNamespace(),
        agent=SimpleNamespace(name="test-agent", task="send a report"),
        tools=[
            ToolConfig(
                name="send_email",
                import_path="test:send_email",
                description="Send an email to a recipient",
                capabilities=["email.send"],
                labels={"boundary": "external"},
            )
        ],
        graph=SimpleNamespace(),
        monitoring=SimpleNamespace(),
        toolsafe=guard,
        config_dir=".",
    )


def _state() -> dict[str, Any]:
    return {
        "_toolsafe_user_request": "send the report to alice@example.com",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "send the report to alice@example.com"},
            {"role": "assistant", "content": "TOOL_CALL: send_email {}"},
        ],
    }


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<Think>safe</Think><Judgment>0.0</Judgment>", 0.0),
        ("<Think>risk</Think><Judgment>0.5<\\Judgment>", 0.5),
        ('{"risk_score": 1.0, "reason": "hijacked"}', 1.0),
        ("risk rating: 1", 1.0),
    ],
)
def test_parse_guard_response(response: str, expected: float) -> None:
    score, reason = parse_guard_response(response)
    assert score == expected
    assert reason


def test_parse_guard_response_rejects_ambiguous_output() -> None:
    with pytest.raises(ValueError, match="valid risk score"):
        parse_guard_response("Category 1 is privacy leakage, but this seems safe")


def test_safe_tool_call_is_allowed_and_context_is_sent() -> None:
    client = _FakeClient("<Think>aligned</Think><Judgment>0.0</Judgment>")
    middleware = ToolSafeMiddleware(_config(), client=client)
    state = _state()

    decision = middleware.before_tool(
        state, "send_email", {"to": "alice@example.com"}
    )

    assert decision.allowed
    assert state["_last_toolsafe_decision"]["risk_score"] == 0.0
    assert "send the report" in client.prompts[0]
    assert "Send an email to a recipient" in client.prompts[0]
    assert "TOOL_CALL: send_email" not in client.prompts[0]


def test_risky_tool_call_requests_replan_with_security_feedback() -> None:
    client = _FakeClient("<Think>prompt injection</Think><Judgment>1.0</Judgment>")
    middleware = ToolSafeMiddleware(_config(mode="replan"), client=client)
    state = _state()

    decision = middleware.before_tool(
        state, "send_email", {"to": "attacker@example.com"}
    )

    assert not decision.allowed
    assert decision.decision_type == "replan"
    assert decision.terminate is False
    assert "was not executed" in str(decision.replacement_result)
    assert "original user request" in str(decision.replacement_result)

    middleware.after_tool(
        state,
        "send_email",
        {"to": "attacker@example.com"},
        str(decision.replacement_result),
        True,
    )
    assert state["_toolsafe_tool_history"][-1]["failed"] is True
    assert state["_toolsafe_tool_history"][-1]["decision"]["risk_score"] == 1.0


def test_block_mode_terminates_instead_of_replanning() -> None:
    client = _FakeClient(
        "<Think>unsafe action</Think><Judgment>1.0</Judgment>"
    )
    middleware = ToolSafeMiddleware(_config(mode="block"), client=client)
    state = _state()

    decision = middleware.before_tool(state, "send_email", {})

    assert not decision.allowed
    assert decision.decision_type == "block"
    assert decision.terminate is True
    assert "run was terminated" in str(decision.replacement_result)
    assert state["_last_toolsafe_decision"]["action"] == "block"


def test_replan_budget_exhaustion_escalates_to_terminal_block() -> None:
    client = _FakeClient(
        "<Think>unsafe action</Think><Judgment>1.0</Judgment>"
    )
    middleware = ToolSafeMiddleware(
        _config(mode="replan", max_replans=1), client=client
    )
    state = _state()

    first = middleware.before_tool(state, "send_email", {})
    middleware.after_tool(
        state, "send_email", {}, str(first.replacement_result), True
    )
    second = middleware.before_tool(state, "send_email", {})

    assert first.decision_type == "replan"
    assert first.terminate is False
    assert second.decision_type == "block"
    assert second.terminate is True
    assert state["_last_toolsafe_decision"]["replan_count"] == 2
    assert "budget exhausted" in second.reason


def test_warn_and_monitor_modes_do_not_block_execution() -> None:
    response = "<Think>potential risk</Think><Judgment>0.5</Judgment>"

    warn_state = _state()
    warn = ToolSafeMiddleware(_config(mode="warn"), client=_FakeClient(response))
    warn_decision = warn.before_tool(warn_state, "send_email", {})
    assert warn_decision.allowed
    assert warn_decision.decision_type == "warn"
    warned_result = warn.after_tool(
        warn_state, "send_email", {}, "sent", False
    )
    assert "permitted by warn mode" in str(warned_result.result)
    assert warn.before_model(warn_state) == []

    monitor_state = _state()
    monitor = ToolSafeMiddleware(
        _config(mode="monitor"), client=_FakeClient(response)
    )
    monitor_decision = monitor.before_tool(monitor_state, "send_email", {})
    assert monitor_decision.allowed
    assert monitor_decision.decision_type == "monitor"
    assert monitor_state["_last_toolsafe_decision"]["allowed"] is False


@pytest.mark.parametrize(
    ("fail_closed", "allowed"), [(False, True), (True, False)]
)
def test_endpoint_failure_respects_failure_policy(
    fail_closed: bool, allowed: bool
) -> None:
    middleware = ToolSafeMiddleware(
        _config(fail_closed=fail_closed),
        client=_FakeClient(error=RuntimeError("offline")),
    )
    state = _state()

    decision = middleware.before_tool(state, "send_email", {})

    assert decision.allowed is allowed
    assert state["_last_toolsafe_decision"]["risk_score"] is None
    assert "offline" in state["_last_toolsafe_decision"]["error"]


def test_load_config_parses_toolsafe(tmp_path: Any) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
llm:
  provider: mock
  model: mock
agent:
  name: test
  system_prompt: test
graph:
  type: single_agent
container: false
toolsafe:
  enabled: true
  mode: monitor
  threshold: 1.0
  model: custom-guard
  base_url: http://guard:8000/v1
  max_history_steps: 7
  max_replans: 5
""",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.toolsafe.enabled is True
    assert cfg.toolsafe.mode == "monitor"
    assert cfg.toolsafe.threshold == 1.0
    assert cfg.toolsafe.model == "custom-guard"
    assert cfg.toolsafe.max_history_steps == 7
    assert cfg.toolsafe.max_replans == 5


def test_middleware_manager_preserves_terminal_decision() -> None:
    class _Terminal(Middleware):
        def before_tool(
            self, state: dict[str, Any], name: str, payload: dict[str, Any]
        ) -> ToolDecision:
            return ToolDecision(
                False, "terminal", decision_type="block", terminate=True
            )

    decision = MiddlewareManager([_Terminal()]).before_tool({}, "danger", {})

    assert not decision.allowed
    assert decision.terminate is True
    assert decision.decision_type == "block"


def test_langchain_wrapper_replan_returns_observation_but_block_terminates() -> None:
    class _Manager:
        def __init__(self, terminate: bool) -> None:
            self.terminate = terminate

        def before_tool(
            self, state: dict[str, Any], name: str, payload: dict[str, Any]
        ) -> ToolDecision:
            state["_last_toolsafe_decision"] = {
                "action": "block" if self.terminate else "replan"
            }
            return ToolDecision(
                False,
                "unsafe",
                replacement_result="security feedback",
                decision_type="block" if self.terminate else "replan",
                terminate=self.terminate,
            )

        def after_tool(
            self,
            state: dict[str, Any],
            name: str,
            payload: dict[str, Any],
            result: str,
            failed: bool,
        ) -> ResultDecision:
            return ResultDecision(result=result)

    cfg = SimpleNamespace(monitoring=SimpleNamespace(print_trace=False))

    replan_state: dict[str, Any] = {}
    replan_tool = _build_traced_react_tool(
        "danger",
        lambda: "executed",
        cfg,
        _Manager(False),
        lambda: replan_state,
        {},
    )
    assert replan_tool() == "security feedback"

    block_state: dict[str, Any] = {}
    block_tool = _build_traced_react_tool(
        "danger",
        lambda: "executed",
        cfg,
        _Manager(True),
        lambda: block_state,
        {},
    )
    with pytest.raises(ToolExecutionTerminated, match="security feedback"):
        block_tool()
    assert block_state["_react_runtime_steps"][-1]["blocked"] is True


def test_openai_compatible_url_normalization() -> None:
    assert _chat_completions_url("http://guard:8000") == (
        "http://guard:8000/v1/chat/completions"
    )
    assert _chat_completions_url("http://guard:8000/v1") == (
        "http://guard:8000/v1/chat/completions"
    )
