from __future__ import annotations

import json
from pathlib import Path

from agent_scaffold.config import (
    AgentConfig,
    AppConfig,
    GraphConfig,
    LLMConfig,
    MonitoringConfig,
    Pro2GuardConfig,
    Pro2GuardGeneratorConfig,
    ToolConfig,
    load_config,
)
from agent_scaffold.llm import LLMResponse
from agent_scaffold.pro2guard.abstraction import ToolTraceAbstraction
from agent_scaffold.pro2guard.generator import compile_pro2guard_policy
from agent_scaffold.pro2guard.middleware import Pro2GuardMiddleware


class _LLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return LLMResponse(
            content=output,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


def _write_model(path: Path) -> None:
    states = [
        "send_email|communication|normal|mutating|ok",
        "read_file|file|normal|read_only|ok",
        "ATTACK_SUCCESS",
    ]
    path.write_text(
        json.dumps(
            {
                "state_index": {state: index for index, state in enumerate(states)},
                "transition_probs": {
                    "0": {"2": 1.0},
                    "1": {"1": 1.0},
                    "2": {"2": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    fail_closed: bool = False,
    max_attempts: int = 1,
) -> AppConfig:
    model_path = tmp_path / "dtmc.json"
    _write_model(model_path)
    return AppConfig(
        llm=LLMConfig(provider="mock", model="mock"),
        agent=AgentConfig(
            name="test-agent",
            system_prompt="Be safe",
            task="Send a report",
        ),
        tools=[
            ToolConfig(
                name="send_email",
                import_path="tests.fake:send_email",
                description="Send an email to a recipient.",
            ),
            ToolConfig(
                name="read_file",
                import_path="tests.fake:read_file",
                description="Read a local file.",
            ),
        ],
        graph=GraphConfig(type="single_agent"),
        monitoring=MonitoringConfig(),
        pro2guard=Pro2GuardConfig(
            enabled=True,
            model_path=str(model_path),
            unsafe_states=[],
            generator=Pro2GuardGeneratorConfig(
                enabled=True,
                max_attempts=max_attempts,
                state_batch_size=8,
                fail_closed=fail_closed,
            ),
        ),
        config_dir=str(tmp_path),
    )


def test_staged_generator_accepts_verbose_tagged_outputs(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    llm = _LLM(
        [
            (
                "Analysis mentions <profile>SKIP</profile>, but the final answer is "
                "<profile>mutating</profile>."
            ),
            "Reasoning omitted. <profile>file|read_only</profile>",
            (
                "Long reasoning and even {not: json} are harmless. "
                "Final classification: <unsafe>2</unsafe>"
            ),
        ]
    )

    result = compile_pro2guard_policy(
        cfg,
        "Send a report",
        tmp_path / "run",
        llm=llm,
    )

    assert result.status == "compiled"
    assert result.source == "llm"
    assert result.attempts == 3
    assert result.profile_count == 2
    assert result.unsafe_state_count == 1
    assert cfg.pro2guard.unsafe_states == ["ATTACK_SUCCESS"]
    assert Path(cfg.pro2guard.abstraction_policy_path).exists()
    policy = json.loads(Path(result.policy_path).read_text(encoding="utf-8"))
    abstraction = ToolTraceAbstraction(policy)
    middleware = Pro2GuardMiddleware(cfg)
    assert middleware.abstraction.tool_profiles["send_email"]["category"] == (
        "communication"
    )
    assert (
        abstraction.encode_tool_result("send_email", {}, "ok", False)
        == "send_email|communication|normal|mutating|ok"
    )


def test_invalid_state_indices_fall_back_when_no_unsafe_state_exists(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    llm = _LLM(
        [
            "<profile>SKIP</profile>",
            "<profile>SKIP</profile>",
            "Reasoning followed by <unsafe>99</unsafe>",
        ]
    )

    result = compile_pro2guard_policy(
        cfg,
        "Send a report",
        tmp_path / "run",
        llm=llm,
    )

    assert result.status == "fallback"
    assert result.source == "manual"
    assert "out of range" in result.warnings[-1]
    assert cfg.pro2guard.unsafe_states == []
    assert cfg.pro2guard.abstraction_policy_path == ""


def test_invalid_tool_profile_is_skipped_without_losing_valid_unsafe_state(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    llm = _LLM(
        [
            "<profile>file|read_only</profile>",
            "<profile>file|read_only</profile>",
            "<unsafe>2</unsafe>",
        ]
    )

    result = compile_pro2guard_policy(
        cfg,
        "Send a report",
        tmp_path / "run",
        llm=llm,
    )

    assert result.status == "compiled"
    assert result.profile_count == 1
    assert result.unsafe_state_count == 1
    assert "unsupported profile" in result.warnings[0]


def test_fail_closed_requires_at_least_one_unsafe_state(tmp_path: Path) -> None:
    cfg = _config(tmp_path, fail_closed=True)
    llm = _LLM(
        [
            "<profile>SKIP</profile>",
            "<profile>SKIP</profile>",
            "<unsafe>NONE</unsafe>",
        ]
    )

    try:
        compile_pro2guard_policy(
            cfg,
            "Send a report",
            tmp_path / "run",
            llm=llm,
        )
    except RuntimeError as exc:
        assert "no valid unsafe states" in str(exc)
    else:
        raise AssertionError("fail_closed should reject an empty unsafe-state set")


def test_batch_cache_reuses_policy_without_llm_calls(
    tmp_path: Path, monkeypatch
) -> None:
    batch_dir = tmp_path / "batch"
    monkeypatch.setenv("AGENT_BATCH_DIR", str(batch_dir))
    first_cfg = _config(tmp_path)
    first_llm = _LLM(
        [
            "<profile>mutating</profile>",
            "<profile>file|read_only</profile>",
            "<unsafe>2</unsafe>",
        ]
    )

    first = compile_pro2guard_policy(
        first_cfg, "Send a report", batch_dir / "run_001", llm=first_llm
    )

    second_cfg = _config(tmp_path)
    second_llm = _LLM(["this must not be called"])
    second = compile_pro2guard_policy(
        second_cfg, "Send a report", batch_dir / "run_002", llm=second_llm
    )

    assert first.source == "llm"
    assert first.attempts == 3
    assert second.source == "cache"
    assert second.attempts == 0
    assert second_llm.calls == 0
    assert second_cfg.pro2guard.unsafe_states == ["ATTACK_SUCCESS"]
    assert (batch_dir / "pro2guard_policy.batch-cache.json").exists()


def test_batch_cache_fingerprint_change_regenerates(
    tmp_path: Path, monkeypatch
) -> None:
    batch_dir = tmp_path / "batch"
    monkeypatch.setenv("AGENT_BATCH_DIR", str(batch_dir))
    first_cfg = _config(tmp_path)
    compile_pro2guard_policy(
        first_cfg,
        "Send a report",
        batch_dir / "run_001",
        llm=_LLM(
            [
                "<profile>mutating</profile>",
                "<profile>file|read_only</profile>",
                "<unsafe>2</unsafe>",
            ]
        ),
    )
    changed_cfg = _config(tmp_path)
    changed_llm = _LLM(
        [
            "<profile>mutating</profile>",
            "<profile>file|read_only</profile>",
            "<unsafe>2</unsafe>",
        ]
    )

    result = compile_pro2guard_policy(
        changed_cfg, "A different task", batch_dir / "run_002", llm=changed_llm
    )

    assert result.source == "llm"
    assert result.attempts == 3
    assert changed_llm.calls == 3


def test_config_can_disable_pro2guard_generator(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        """
llm:
  provider: mock
  model: mock
agent:
  name: test
  system_prompt: safe
tools: []
graph:
  type: single_agent
pro2guard:
  enabled: true
  generator: false
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.pro2guard.enabled
    assert not cfg.pro2guard.generator.enabled
