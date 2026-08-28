from __future__ import annotations

from types import SimpleNamespace

from agent_scaffold.agentdojo_adapter import (
    AgentDojoSession,
    _default_attack_injections,
    aggregate_asr,
    config_file_snapshot,
    redact_config_snapshot,
)
from agent_scaffold.config import load_config


def _item(*, injection_task: str | None, attack_success: bool) -> dict:
    return {
        "ok": True,
        "harness": {
            "agentdojo": {
                "injection_task": injection_task,
                "attack_success": attack_success,
            }
        },
    }


def test_agentdojo_asr_counts_only_injection_trials() -> None:
    result = aggregate_asr(
        [
            _item(injection_task="injection_task_0", attack_success=True),
            _item(injection_task="injection_task_1", attack_success=False),
            _item(injection_task=None, attack_success=False),
            {"ok": False, "error": "run failed"},
        ]
    )

    assert result == {
        "evaluated_runs": 3,
        "attack_trials": 2,
        "attack_successes": 1,
        "asr": 0.5,
    }


def test_agentdojo_asr_is_none_for_benign_runs() -> None:
    result = aggregate_asr(
        [_item(injection_task=None, attack_success=False)]
    )

    assert result == {
        "evaluated_runs": 1,
        "attack_trials": 0,
        "attack_successes": 0,
        "asr": None,
    }


def test_agentdojo_asr_ignores_non_agentdojo_items() -> None:
    assert aggregate_asr([{"harness": {"agentharm": {}}}]) is None


def test_config_snapshot_redacts_credentials() -> None:
    config = {
        "llm": {"model": "test", "api_key": "top-secret"},
        "agentdojo": {"case": "user_task_0_injection_0"},
        "nested": [{"access_token": "token-value"}],
    }

    assert redact_config_snapshot(config) == {
        "llm": {"model": "test", "api_key": "***REDACTED***"},
        "agentdojo": {"case": "user_task_0_injection_0"},
        "nested": [{"access_token": "***REDACTED***"}],
    }


def test_config_file_snapshot_only_includes_agent_and_environment_yaml(
    tmp_path,
) -> None:
    (tmp_path / "agent.yaml").write_text(
        "llm:\n  model: test\n  api_key: hidden\n",
        encoding="utf-8",
    )
    (tmp_path / "environment.yaml").write_text(
        "agentdojo:\n  case: user_task_0_injection_0\n",
        encoding="utf-8",
    )
    (tmp_path / "tools.yaml").write_text(
        "tools:\n  - name: should_not_appear\n",
        encoding="utf-8",
    )

    assert config_file_snapshot(tmp_path / "agent.yaml") == {
        "agent.yaml": {
            "llm": {
                "model": "test",
                "api_key": "***REDACTED***",
            }
        },
        "environment.yaml": {
            "agentdojo": {"case": "user_task_0_injection_0"}
        },
    }


def test_llm_api_key_is_loaded_from_environment(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_API_KEY", "environment-secret")
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: openai\n"
        "  model: test\n"
        "  api_key_env: TEST_LLM_API_KEY\n"
        "agent:\n"
        "  system_prompt: test\n"
        "container:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.api_key == "environment-secret"
    assert config.llm.api_key_env == "TEST_LLM_API_KEY"


def test_llm_rejects_inline_api_key(tmp_path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: openai\n"
        "  model: test\n"
        "  api_key: inline-secret\n"
        "agent:\n"
        "  system_prompt: test\n"
        "container:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "must not be stored in YAML" in str(exc)
    else:
        raise AssertionError("inline llm.api_key should be rejected")


class _UserTask:
    def utility_from_traces(self, *args: object) -> bool:
        return True


class _InjectionTask:
    def __init__(self, attack_success: bool) -> None:
        self.attack_success = attack_success

    def security_from_traces(self, *args: object) -> bool:
        return self.attack_success


def _session(injection_task: _InjectionTask | None) -> AgentDojoSession:
    session = AgentDojoSession.__new__(AgentDojoSession)
    session.cfg = SimpleNamespace(
        suite="workspace",
        benchmark_version="v1",
        user_task="user_task_0",
        injection_task="injection_task_0" if injection_task else "",
    )
    session.user_task = _UserTask()
    session.injection_task = injection_task
    session.pre_environment = object()
    session.environment = object()
    session.calls = []
    session.errors = []
    return session


def test_single_attack_run_records_asr() -> None:
    assert _session(_InjectionTask(True)).evaluate("done")["asr"] == 1.0
    assert _session(_InjectionTask(False)).evaluate("done")["asr"] == 0.0


def test_single_benign_run_records_null_asr() -> None:
    assert _session(None).evaluate("done")["asr"] is None


def test_disabled_injection_produces_no_payload_even_with_custom_text() -> None:
    suite = SimpleNamespace(
        get_injection_vector_defaults=lambda: {"vector": "default"}
    )
    injection_task = SimpleNamespace(GOAL="attack goal")

    assert _default_attack_injections(
        suite,
        injection_task,
        {"vector": "custom attack"},
        injection_enabled=False,
    ) == {}


def test_declared_but_disabled_injection_is_excluded_from_asr() -> None:
    result = aggregate_asr(
        [
            {
                "harness": {
                    "agentdojo": {
                        "injection_task": "injection_task_0",
                        "injection_enabled": False,
                        "attack_success": False,
                    }
                }
            }
        ]
    )

    assert result == {
        "evaluated_runs": 1,
        "attack_trials": 0,
        "attack_successes": 0,
        "asr": None,
    }


def test_config_loads_disabled_agentdojo_injection_switch(tmp_path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: openai\n"
        "  model: test\n"
        "agent:\n"
        "  system_prompt: test\n"
        "container:\n"
        "  enabled: false\n"
        "agentdojo:\n"
        "  enabled: false\n"
        "  case: user_task_0_injection_0\n"
        "  injection_enabled: false\n",
        encoding="utf-8",
    )

    config = load_config(config_path).agentdojo
    assert config.injection_task == "injection_task_0"
    assert config.injection_enabled is False
