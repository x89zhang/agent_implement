"""Configuration for external runtimes, independent of their dependencies."""

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HermesConfig:
    repo_path: str = "/home/xiaoliang_zhang/hermes-agent"
    python_executable: str = ""
    expected_commit: str = ""
    allow_dirty_checkout: bool = False
    api_mode: str = "auto"
    max_iterations: int = 30
    timeout_seconds: float = 300


@dataclass
class MemoryExperimentConfig:
    mode: str = "off"
    poisoning_input_file: str = ""
    clean_initial_memory_dir: str = ""
    poisoned_memory_dir: str = ""
    run_clean_control: bool = True


@dataclass
class ExecutionConfig:
    backend: str = "builtin"
    hermes: HermesConfig = field(default_factory=HermesConfig)
    memory: MemoryExperimentConfig = field(default_factory=MemoryExperimentConfig)


def parse_execution(raw: dict, config_dir: Path) -> ExecutionConfig:
    values = raw.get("execution") or {}
    if not isinstance(values, dict) or set(values) - {"backend", "hermes"}:
        raise ValueError("execution accepts only backend and hermes")
    backend = values.get("backend", "builtin")
    if backend not in {"builtin", "hermes"}:
        raise ValueError("execution.backend must be builtin or hermes")
    try:
        hermes = HermesConfig(**(values.get("hermes") or {}))
        memory = MemoryExperimentConfig(**(raw.get("memory_experiment") or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid external runtime configuration: {exc}") from exc
    if hermes.api_mode not in {"auto", "chat_completions", "codex_responses"}:
        raise ValueError(
            "Hermes api_mode must be auto, chat_completions, or codex_responses"
        )
    if type(hermes.max_iterations) is not int or hermes.max_iterations < 1:
        raise ValueError("Hermes max_iterations must be a positive integer")
    if (
        not isinstance(hermes.timeout_seconds, (int, float))
        or isinstance(hermes.timeout_seconds, bool)
        or not math.isfinite(hermes.timeout_seconds)
        or hermes.timeout_seconds <= 0
    ):
        raise ValueError("Hermes timeout_seconds must be positive and finite")
    if (
        type(hermes.allow_dirty_checkout) is not bool
        or type(memory.run_clean_control) is not bool
    ):
        raise ValueError("allow_dirty_checkout and run_clean_control must be booleans")
    if memory.mode not in {"off", "native_two_stage", "direct_seed"}:
        raise ValueError("Unknown memory_experiment.mode")
    if memory.mode != "off" and backend != "hermes":
        raise ValueError("memory_experiment requires execution.backend: hermes")
    if memory.mode == "native_two_stage" and not memory.poisoning_input_file:
        raise ValueError("native_two_stage requires poisoning_input_file")
    if memory.mode == "direct_seed" and not memory.poisoned_memory_dir:
        raise ValueError("direct_seed requires poisoned_memory_dir")
    for obj, names in (
        (hermes, ("repo_path", "python_executable")),
        (
            memory,
            ("poisoning_input_file", "clean_initial_memory_dir", "poisoned_memory_dir"),
        ),
    ):
        for name in names:
            value = getattr(obj, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a path string")
            if value:
                path = Path(value).expanduser()
                absolute = path if path.is_absolute() else config_dir / path
                # Resolving a venv python symlink selects the base interpreter and loses its packages.
                setattr(
                    obj,
                    name,
                    str(
                        absolute.absolute()
                        if name == "python_executable"
                        else absolute.resolve()
                    ),
                )
    return ExecutionConfig(backend, hermes, memory)
