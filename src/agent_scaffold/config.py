from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout: int | None = 120


@dataclass
class ToolConfig:
    name: str
    import_path: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    task: str = ""


@dataclass
class GraphConfig:
    type: str
    max_iters: int = 4
    tool_call_format: str = "TOOL_CALL: <name> <json>"
    stop_keyword: str = "FINAL"
    react_prompt: str = ""
    react_max_iterations: int | None = 15
    react_max_execution_time: int | None = 120


@dataclass
class MonitoringConfig:
    enabled: bool = False
    output_path: str = "trace.json"
    print_trace: bool = False


@dataclass
class SkillsConfig:
    enabled: list[Any] = field(default_factory=list)
    base_dir: str = "skills"


@dataclass
class PlannerConfig:
    enabled: bool = False
    type: str = "static"
    max_steps: int = 8
    steps: list[str] = field(default_factory=list)


@dataclass
class MiddlewareConfig:
    enabled: bool = True
    modules: list[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    trusted_tool_output_prompt: bool = True


@dataclass
class AegisConfig:
    enabled: bool = False
    mode: str = "block"
    risk_threshold: str = "HIGH"
    fail_closed: bool = True
    allow_tools: list[str] = field(default_factory=list)
    block_tools: list[str] = field(default_factory=list)


@dataclass
class Pro2GuardGeneratorConfig:
    enabled: bool = True
    context_mode: str = "benign_only"
    max_attempts: int = 2
    max_profiles: int = 50
    max_unsafe_states: int = 20
    max_model_states: int = 200
    state_batch_size: int = 8
    fail_closed: bool = False
    provider: str = ""
    model: str = ""
    temperature: float | None = None
    base_url: str = ""
    api_key: str = ""
    request_timeout: int | None = None

@dataclass
class Pro2GuardConfig:
    enabled: bool = False
    mode: str = "block"
    threshold: float = 0.1
    model_path: str = ""
    dtmc_path: str = ""
    prism_bin: str = "prism"
    abstraction: str = ""
    abstraction_policy_path: str = ""
    unsafe_states: list[str] = field(default_factory=list)
    horizon: int = 20
    timeout_seconds: int = 10
    fail_closed: bool = False
    generator: Pro2GuardGeneratorConfig = field(
        default_factory=Pro2GuardGeneratorConfig
    )


@dataclass
class AgentSpecGeneratorConfig:
    enabled: bool = True
    context_mode: str = "benign_only"
    max_attempts: int = 2
    max_rules: int = 12
    provider: str = ""
    model: str = ""
    temperature: float | None = None
    base_url: str = ""
    api_key: str = ""
    request_timeout: int | None = None


@dataclass
class AgentSpecConfig:
    enabled: bool = False
    rules: list[str] = field(default_factory=list)
    rule_files: list[str] = field(default_factory=list)
    predicate_modules: list[str] = field(default_factory=list)
    approval_handler: str = "prompt"
    max_reflections: int = 3
    fail_closed: bool = True
    generator: AgentSpecGeneratorConfig = field(
        default_factory=AgentSpecGeneratorConfig
    )


@dataclass
class LlamaFirewallConfig:
    enabled: bool = False
    mode: str = "enforce"
    use_case: str = ""
    scanners: dict[str, list[str]] = field(default_factory=dict)
    max_revisions: int = 1
    fail_closed: bool = False


@dataclass
class ToolSafeConfig:
    enabled: bool = False
    mode: str = "replan"
    threshold: float = 0.5
    provider: str = "openai_compatible"
    model: str = "TS-Guard"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "TOOLSAFE_API_KEY"
    timeout_seconds: float = 30.0
    max_history_steps: int = 20
    max_replans: int = 3
    fail_closed: bool = False


@dataclass
class AgentDoGConfig:
    enabled: bool = False
    mode: str = "diagnose"
    task: str = "unified"
    checkpoints: list[str] = field(default_factory=lambda: ["pre_reply"])
    provider: str = "openai_compatible"
    model: str = "AgentDoG1.5-Unified-Qwen3.5-4B"
    base_url: str = ""
    base_url_env: str = "AGENTDOG_BASE_URL"
    api_key: str = ""
    api_key_env: str = "AGENTDOG_API_KEY"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 1024
    max_trajectory_chars: int = 0
    max_revisions: int = 2
    fail_closed: bool = False
    include_raw_response: bool = True
    replacement_message: str = (
        "The response was withheld because AgentDoG diagnosed unsafe behavior "
        "in the accumulated agent trajectory."
    )


@dataclass
class AgentGuardScenarioCompilerConfig:
    enabled: bool = True
    context_mode: str = "full"
    max_attempts: int = 2
    provider: str = ""
    model: str = ""
    temperature: float | None = None
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    request_timeout: int | None = None


@dataclass
class AgentGuardConfig:
    enabled: bool = False
    mode: str = "block"
    policy: str = ""
    server_url: str = ""
    api_key: str = ""
    plugin_config: str = ""
    environment: str = ""
    user_id: str = ""
    role: str = "default"
    trust_level: int = 1
    sandbox: str = "local"
    sandbox_profile: dict[str, Any] | None = None
    audit_path: str = ""
    max_steps: int = 12
    max_tool_calls: int = 24
    window_size: int = 8
    remote_timeout_seconds: float = 5.0
    remote_retries: int = 2
    fail_closed: bool = True
    scenario_compiler: AgentGuardScenarioCompilerConfig = field(
        default_factory=AgentGuardScenarioCompilerConfig
    )


@dataclass
class AgentSightConfig:
    enabled: bool = False
    binary: str = "agentsight"
    capture: str = "full"
    db_path: str = "agentsight.db"
    snapshot_path: str = "agentsight_snapshot.json"
    log_path: str = "agentsight.log"
    required: bool = False
    privilege: str = "auto"
    web_server: bool = False
    server_port: int = 7395
    startup_timeout_seconds: float = 10.0
    warmup_seconds: float = 1.0
    shutdown_timeout_seconds: float = 10.0


@dataclass
class AgentDojoConfig:
    enabled: bool = False
    suite: str = "workspace"
    benchmark_version: str = "v1.2.2"
    case: str = ""
    user_task: str = "user_task_0"
    injection_task: str = ""
    injection_enabled: bool = True
    trusted_tool_output_prompt: bool = True
    custom_injection_text: str = ""
    attack_template: str = ""
    injection_vectors: list[str] = field(default_factory=list)
    injections: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentSecurityBenchConfig:
    enabled: bool = False
    data_dir: str = ""
    agent_name: str = "financial_analyst_agent"
    task_index: int = 0
    attacker_tool: str = ""
    injection_method: str = "clean"
    attack_type: str = "naive"
    defense_type: str = ""
    trigger: str = "with perspicacious discernment"


@dataclass
class AgentHarmConfig:
    enabled: bool = False
    dataset: str = "harmful"
    split: str = "test_public"
    behavior_id: str = ""
    detailed_behaviors: bool | None = None
    hint_included: bool | None = None
    n_irrelevant_tools: int = 0
    seed: int = 0
    refusal_judge: str = "openai/gpt-4o-2024-08-06"
    semantic_judge: str = "openai/gpt-4o-2024-08-06"
    force_redownload: bool = False


@dataclass
class ContainerConfig:
    enabled: bool = True
    image: str = "agent-scaffold:latest"
    auto_build: bool = True
    dockerfile: str = "Dockerfile"
    workdir: str = "/workspace"
    network: str = "host"
    remove: bool = True
    build_args: dict[str, str] = field(default_factory=dict)
    env: list[str] = field(
        default_factory=lambda: [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AGENTGUARD_API_KEY",
            "AGENTGUARD_SERVER_URL",
            "TOOLSAFE_API_KEY",
            "AGENTDOG_API_KEY",
            "AGENTDOG_BASE_URL",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "TOGETHER_API_KEY",
            "TOKENIZERS_PARALLELISM",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        ]
    )


@dataclass
class AppConfig:
    llm: LLMConfig
    agent: AgentConfig
    tools: list[ToolConfig]
    graph: GraphConfig
    monitoring: MonitoringConfig
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    aegis: AegisConfig = field(default_factory=AegisConfig)
    pro2guard: Pro2GuardConfig = field(default_factory=Pro2GuardConfig)
    agentspec: AgentSpecConfig = field(default_factory=AgentSpecConfig)
    llamafirewall: LlamaFirewallConfig = field(default_factory=LlamaFirewallConfig)
    toolsafe: ToolSafeConfig = field(default_factory=ToolSafeConfig)
    agentdog: AgentDoGConfig = field(default_factory=AgentDoGConfig)
    agentguard: AgentGuardConfig = field(default_factory=AgentGuardConfig)
    agentsight: AgentSightConfig = field(default_factory=AgentSightConfig)
    agentdojo: AgentDojoConfig = field(default_factory=AgentDojoConfig)
    agent_security_bench: AgentSecurityBenchConfig = field(
        default_factory=AgentSecurityBenchConfig
    )
    agentharm: AgentHarmConfig = field(default_factory=AgentHarmConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    trip: dict[str, Any] = field(default_factory=dict)
    research: dict[str, Any] = field(default_factory=dict)
    config_dir: str = "."


def _container_enabled_from_raw(raw: dict[str, Any]) -> bool:
    if os.environ.get("AGENT_CONTAINERIZED") == "1":
        return False
    container_raw = raw.get("container", {})
    if isinstance(container_raw, bool):
        return container_raw
    if isinstance(container_raw, dict):
        return bool(container_raw.get("enabled", True))
    return True


def _optional_int(value: Any, default: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "unlimited",
    }:
        return None
    return int(value if value is not None else default)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


_AGENTDOJO_CASE_RE = re.compile(r"^(user_task_\d+)(?:_(injection(?:_task)?_\d+))?$")


def _parse_agentdojo_case(case_id: str) -> tuple[str, str]:
    match = _AGENTDOJO_CASE_RE.fullmatch(case_id.strip())
    if not match:
        raise ValueError(
            "AgentDojo case must look like 'user_task_3' for benign runs or "
            "'user_task_3_injection_2' / 'user_task_3_injection_task_2' for attack runs."
        )
    user_task = match.group(1)
    injection_task = match.group(2) or ""
    if injection_task.startswith("injection_") and not injection_task.startswith(
        "injection_task_"
    ):
        injection_task = injection_task.replace("injection_", "injection_task_", 1)
    return user_task, injection_task


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return raw


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _resolve_harness_dir(config_path: Path, raw: dict[str, Any]) -> Path | None:
    base = config_path.parent.resolve()
    harness_raw = raw.get("harness")
    if isinstance(harness_raw, str):
        return _resolve_path(base, harness_raw)
    if isinstance(harness_raw, dict):
        path_value = harness_raw.get("path") or harness_raw.get("dir")
        if path_value:
            return _resolve_path(base, str(path_value))
    if config_path.name == "agent.yaml":
        return base
    return None


def _coerce_yaml_list(path: Path, key: str) -> list[Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return []
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        value = loaded.get(key, [])
        return value if isinstance(value, list) else []
    raise ValueError(f"YAML file must contain a list or mapping: {path}")


def _apply_harness_files(
    raw: dict[str, Any], harness_dir: Path | None
) -> dict[str, Any]:
    if harness_dir is None or not harness_dir.exists():
        return raw
    merged = dict(raw)

    system_prompt_path = harness_dir / "systemprompt.md"
    if system_prompt_path.exists():
        agent_raw = dict(merged.get("agent") or {})
        agent_raw["system_prompt"] = system_prompt_path.read_text(
            encoding="utf-8"
        ).strip()
        merged["agent"] = agent_raw

    task_path = harness_dir / "task.md"
    if task_path.exists():
        agent_raw = dict(merged.get("agent") or {})
        agent_raw["task"] = task_path.read_text(encoding="utf-8").strip()
        merged["agent"] = agent_raw

    environment_path = harness_dir / "environment.yaml"
    if environment_path.exists():
        environment_raw = _load_yaml_mapping(environment_path)
        for key, value in environment_raw.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**dict(merged[key]), **value}
            else:
                merged[key] = value

    tools_path = harness_dir / "tools.yaml"
    if tools_path.exists():
        merged["tools"] = _coerce_yaml_list(tools_path, "tools")

    skills_path = harness_dir / "skills.yaml"
    if skills_path.exists():
        merged["skills"] = _load_yaml_mapping(skills_path)

    planner_path = harness_dir / "planner.yaml"
    if planner_path.exists():
        merged["planner"] = _load_yaml_mapping(planner_path)

    middleware_path = harness_dir / "middleware.yaml"
    if middleware_path.exists():
        merged["middleware"] = _load_yaml_mapping(middleware_path)

    memory_path = harness_dir / "memory.md"
    if memory_path.exists():
        memory_text = memory_path.read_text(encoding="utf-8").strip()
        if memory_text:
            skills_raw = merged.get("skills") or {}
            if isinstance(skills_raw, dict):
                enabled = skills_raw.get("enabled", []) or []
                if not isinstance(enabled, list):
                    enabled = [enabled]
                skills_raw = dict(skills_raw)
                skills_raw["enabled"] = enabled + [
                    {
                        "name": "agent_memory",
                        "description": "Agent-specific memory loaded from harness memory.md.",
                        "instructions": memory_text,
                        "priority": 90,
                    }
                ]
                merged["skills"] = skills_raw
            else:
                merged["skills"] = {
                    "enabled": [
                        {
                            "name": "agent_memory",
                            "description": "Agent-specific memory loaded from harness memory.md.",
                            "instructions": memory_text,
                            "priority": 90,
                        }
                    ]
                }

    return merged


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return _apply_harness_files(raw, _resolve_harness_dir(config_path, raw))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = load_config_mapping(config_path)

    llm_raw = _require(raw, "llm")
    agent_raw = _require(raw, "agent")
    graph_raw = raw.get("graph", {})
    agentdojo_raw = raw.get("agentdojo", {}) or {}
    agent_security_bench_raw = raw.get("agent_security_bench", {}) or {}
    agentharm_raw = raw.get("agentharm", {}) or {}

    llm_provider = str(_require(llm_raw, "provider"))
    default_api_key_env = (
        "ANTHROPIC_API_KEY"
        if llm_provider.lower() == "anthropic"
        else "OPENAI_API_KEY"
    )
    inline_api_key = str(llm_raw.get("api_key", "") or "")
    if inline_api_key:
        raise ValueError(
            "llm.api_key must not be stored in YAML; set llm.api_key_env and "
            "export that environment variable instead"
        )
    api_key_env = str(
        llm_raw.get("api_key_env", default_api_key_env) or ""
    )
    llm = LLMConfig(
        provider=llm_provider,
        model=str(_require(llm_raw, "model")),
        temperature=float(llm_raw.get("temperature", 0.2)),
        base_url=str(llm_raw.get("base_url", "")),
        api_key=os.environ.get(api_key_env, "") if api_key_env else "",
        api_key_env=api_key_env,
        request_timeout=_optional_int(llm_raw.get("request_timeout", 120), 120),
    )

    agent = AgentConfig(
        name=str(agent_raw.get("name", "agent")),
        system_prompt=str(_require(agent_raw, "system_prompt")),
        task=str(agent_raw.get("task", "")),
    )

    tools_raw = raw.get("tools", []) or []
    tools: list[ToolConfig] = []
    for item in tools_raw:
        tools.append(
            ToolConfig(
                name=str(_require(item, "name")),
                import_path=str(_require(item, "import")),
                description=str(item.get("description", "")),
                capabilities=[
                    str(value) for value in (item.get("capabilities", []) or [])
                ],
                labels={
                    str(key): value
                    for key, value in (item.get("labels", {}) or {}).items()
                },
            )
        )

    if isinstance(agentdojo_raw, dict):
        case_id = str(agentdojo_raw.get("case", "") or "")
        raw_user_task = str(agentdojo_raw.get("user_task", "user_task_0"))
        raw_injection_task = str(agentdojo_raw.get("injection_task", "") or "")
        if case_id:
            case_user_task, case_injection_task = _parse_agentdojo_case(case_id)
            if "user_task" in agentdojo_raw and raw_user_task != case_user_task:
                raise ValueError(
                    f"AgentDojo case '{case_id}' selects user_task '{case_user_task}', "
                    f"but user_task is also set to '{raw_user_task}'. Use only case, or make them match."
                )
            if (
                "injection_task" in agentdojo_raw
                and raw_injection_task != case_injection_task
            ):
                raise ValueError(
                    f"AgentDojo case '{case_id}' selects injection_task '{case_injection_task or '<none>'}', "
                    f"but injection_task is also set to '{raw_injection_task or '<none>'}'. Use only case, or make them match."
                )
            raw_user_task = case_user_task
            raw_injection_task = case_injection_task
        elif raw_injection_task:
            raise ValueError(
                "AgentDojo attack runs must use a combined case id such as "
                "'user_task_3_injection_2' instead of setting injection_task separately."
            )

        agentdojo = AgentDojoConfig(
            enabled=bool(agentdojo_raw.get("enabled", False)),
            suite=str(agentdojo_raw.get("suite", "workspace")),
            benchmark_version=str(agentdojo_raw.get("benchmark_version", "v1.2.2")),
            case=case_id,
            user_task=raw_user_task,
            injection_task=raw_injection_task,
            injection_enabled=bool(agentdojo_raw.get("injection_enabled", True)),
            trusted_tool_output_prompt=bool(
                agentdojo_raw.get("trusted_tool_output_prompt", True)
            ),
            custom_injection_text=str(
                agentdojo_raw.get("custom_injection_text", "") or ""
            ),
            attack_template=str(agentdojo_raw.get("attack_template", "") or ""),
            injection_vectors=[
                str(item) for item in (agentdojo_raw.get("injection_vectors", []) or [])
            ],
            injections={
                str(key): str(value)
                for key, value in (
                    agentdojo_raw.get("custom_injections")
                    if agentdojo_raw.get("custom_injections") is not None
                    else agentdojo_raw.get("injections", {}) or {}
                ).items()
            },
        )
    else:
        agentdojo = AgentDojoConfig()

    if isinstance(agent_security_bench_raw, dict):
        agent_security_bench = AgentSecurityBenchConfig(
            enabled=bool(agent_security_bench_raw.get("enabled", False)),
            data_dir=str(agent_security_bench_raw.get("data_dir", "") or ""),
            agent_name=str(
                agent_security_bench_raw.get("agent_name", "financial_analyst_agent")
            ),
            task_index=int(agent_security_bench_raw.get("task_index", 0)),
            attacker_tool=str(agent_security_bench_raw.get("attacker_tool", "") or ""),
            injection_method=str(
                agent_security_bench_raw.get("injection_method", "clean")
            ),
            attack_type=str(agent_security_bench_raw.get("attack_type", "naive")),
            defense_type=str(agent_security_bench_raw.get("defense_type", "") or ""),
            trigger=str(
                agent_security_bench_raw.get(
                    "trigger", "with perspicacious discernment"
                )
            ),
        )
    else:
        agent_security_bench = AgentSecurityBenchConfig()

    if isinstance(agentharm_raw, dict):
        detailed_behaviors = agentharm_raw.get("detailed_behaviors")
        hint_included = agentharm_raw.get("hint_included")
        agentharm = AgentHarmConfig(
            enabled=bool(agentharm_raw.get("enabled", False)),
            dataset=str(agentharm_raw.get("dataset", "harmful")),
            split=str(agentharm_raw.get("split", "test_public")),
            behavior_id=str(agentharm_raw.get("behavior_id", "") or ""),
            detailed_behaviors=bool(detailed_behaviors)
            if detailed_behaviors is not None
            else None,
            hint_included=bool(hint_included) if hint_included is not None else None,
            n_irrelevant_tools=int(agentharm_raw.get("n_irrelevant_tools", 0)),
            seed=int(agentharm_raw.get("seed", 0)),
            refusal_judge=str(
                agentharm_raw.get("refusal_judge", "openai/gpt-4o-2024-08-06")
            ),
            semantic_judge=str(
                agentharm_raw.get("semantic_judge", "openai/gpt-4o-2024-08-06")
            ),
            force_redownload=bool(agentharm_raw.get("force_redownload", False)),
        )
    else:
        agentharm = AgentHarmConfig()

    enabled_benchmarks = [
        name
        for name, enabled in (
            ("agentdojo", agentdojo.enabled),
            ("agent_security_bench", agent_security_bench.enabled),
            ("agentharm", agentharm.enabled),
        )
        if enabled
    ]
    if len(enabled_benchmarks) > 1:
        raise ValueError(
            "Enable only one benchmark harness at a time: "
            + ", ".join(enabled_benchmarks)
        )

    if enabled_benchmarks and not _container_enabled_from_raw(raw):
        benchmark_name = enabled_benchmarks[0]
        adapter_module = {
            "agentdojo": "agentdojo_adapter",
            "agent_security_bench": "agent_security_bench_adapter",
            "agentharm": "agentharm_adapter",
        }[benchmark_name]
        module_name = (
            f"{__package__}.{adapter_module}"
            if __package__
            else f"agent_scaffold.{adapter_module}"
        )
        adapter = importlib.import_module(module_name)
        benchmark_cfg = {
            "agentdojo": agentdojo,
            "agent_security_bench": agent_security_bench,
            "agentharm": agentharm,
        }[benchmark_name]
        tools = [
            ToolConfig(name=name, import_path=import_path, description=description)
            for name, import_path, description in adapter.build_tool_configs(
                benchmark_cfg
            )
        ]

    graph = GraphConfig(
        type=str(graph_raw.get("type", "single_agent")),
        max_iters=int(graph_raw.get("max_iters", 4)),
        tool_call_format=str(
            graph_raw.get("tool_call_format", "TOOL_CALL: <name> <json>")
        ),
        stop_keyword=str(graph_raw.get("stop_keyword", "FINAL")),
        react_prompt=str(graph_raw.get("react_prompt", "")),
        react_max_iterations=_optional_int(
            graph_raw.get("react_max_iterations", 15), 15
        ),
        react_max_execution_time=_optional_int(
            graph_raw.get("react_max_execution_time", 120), 120
        ),
    )

    monitoring_raw = raw.get("monitoring", {}) or {}
    monitoring = MonitoringConfig(
        enabled=bool(monitoring_raw.get("enabled", False)),
        output_path=str(monitoring_raw.get("output_path", "trace.json")),
        print_trace=bool(monitoring_raw.get("print_trace", False)),
    )

    skills_raw = raw.get("skills", {}) or {}
    if isinstance(skills_raw, str):
        skills = SkillsConfig(enabled=[skills_raw])
    elif isinstance(skills_raw, list):
        skills = SkillsConfig(enabled=skills_raw)
    elif isinstance(skills_raw, dict):
        enabled_raw = skills_raw.get("enabled", []) or []
        if isinstance(enabled_raw, str):
            enabled = [enabled_raw]
        elif isinstance(enabled_raw, list):
            enabled = enabled_raw
        else:
            enabled = []
        skills = SkillsConfig(
            enabled=enabled,
            base_dir=str(skills_raw.get("base_dir", "skills")),
        )
    else:
        skills = SkillsConfig()

    planner_raw = raw.get("planner", {}) or {}
    if isinstance(planner_raw, dict):
        raw_steps = planner_raw.get("steps", []) or []
        planner = PlannerConfig(
            enabled=bool(planner_raw.get("enabled", False)),
            type=str(planner_raw.get("type", "static")),
            max_steps=int(planner_raw.get("max_steps", 8)),
            steps=[str(step) for step in raw_steps],
        )
    else:
        planner = PlannerConfig()

    middleware_raw = raw.get("middleware", {}) or {}
    if isinstance(middleware_raw, list):
        middleware = MiddlewareConfig(
            enabled=True, modules=[str(item) for item in middleware_raw]
        )
    elif isinstance(middleware_raw, dict):
        middleware = MiddlewareConfig(
            enabled=bool(middleware_raw.get("enabled", True)),
            modules=[str(item) for item in (middleware_raw.get("modules", []) or [])],
        )
    else:
        middleware = MiddlewareConfig()

    security_raw = raw.get("security", {}) or {}
    legacy_tool_prompt = None
    if (
        isinstance(agentdojo_raw, dict)
        and "trusted_tool_output_prompt" in agentdojo_raw
    ):
        legacy_tool_prompt = bool(agentdojo_raw.get("trusted_tool_output_prompt", True))
    if isinstance(security_raw, bool):
        security = SecurityConfig(trusted_tool_output_prompt=security_raw)
    elif isinstance(security_raw, dict):
        security = SecurityConfig(
            trusted_tool_output_prompt=bool(
                security_raw.get(
                    "trusted_tool_output_prompt",
                    legacy_tool_prompt if legacy_tool_prompt is not None else True,
                )
            ),
        )
    else:
        security = SecurityConfig(
            trusted_tool_output_prompt=legacy_tool_prompt
            if legacy_tool_prompt is not None
            else True,
        )

    aegis_raw = raw.get("aegis", {}) or {}
    if isinstance(aegis_raw, bool):
        aegis = AegisConfig(enabled=aegis_raw)
    elif isinstance(aegis_raw, dict):
        aegis = AegisConfig(
            enabled=bool(aegis_raw.get("enabled", False)),
            mode=str(aegis_raw.get("mode", "block")),
            risk_threshold=str(aegis_raw.get("risk_threshold", "HIGH")).upper(),
            fail_closed=bool(aegis_raw.get("fail_closed", True)),
            allow_tools=[
                str(item) for item in (aegis_raw.get("allow_tools", []) or [])
            ],
            block_tools=[
                str(item) for item in (aegis_raw.get("block_tools", []) or [])
            ],
        )
    else:
        aegis = AegisConfig()

    pro2guard_raw = raw.get("pro2guard", {}) or {}
    if isinstance(pro2guard_raw, bool):
        pro2guard = Pro2GuardConfig(enabled=pro2guard_raw)
    elif isinstance(pro2guard_raw, dict):
        generator_raw = pro2guard_raw.get("generator", True)
        if isinstance(generator_raw, bool):
            pro2guard_generator = Pro2GuardGeneratorConfig(enabled=generator_raw)
        elif isinstance(generator_raw, dict):
            generator_llm_raw = generator_raw.get("llm", {}) or {}
            if not isinstance(generator_llm_raw, dict):
                raise TypeError("pro2guard.generator.llm must be a mapping")
            pro2guard_generator = Pro2GuardGeneratorConfig(
                enabled=bool(generator_raw.get("enabled", True)),
                context_mode=str(
                    generator_raw.get("context_mode", "benign_only")
                ).lower(),
                max_attempts=int(generator_raw.get("max_attempts", 2)),
                max_profiles=int(generator_raw.get("max_profiles", 50)),
                max_unsafe_states=int(
                    generator_raw.get("max_unsafe_states", 20)
                ),
                max_model_states=int(generator_raw.get("max_model_states", 200)),
                state_batch_size=int(generator_raw.get("state_batch_size", 8)),
                fail_closed=bool(generator_raw.get("fail_closed", False)),
                provider=str(
                    generator_llm_raw.get(
                        "provider", generator_raw.get("provider", "")
                    )
                ).lower(),
                model=str(
                    generator_llm_raw.get("model", generator_raw.get("model", ""))
                ),
                temperature=(
                    float(generator_llm_raw["temperature"])
                    if "temperature" in generator_llm_raw
                    else (
                        float(generator_raw["temperature"])
                        if "temperature" in generator_raw
                        else None
                    )
                ),
                base_url=str(
                    generator_llm_raw.get(
                        "base_url", generator_raw.get("base_url", "")
                    )
                ),
                api_key=str(
                    generator_llm_raw.get(
                        "api_key", generator_raw.get("api_key", "")
                    )
                ),
                request_timeout=_optional_int(
                    generator_llm_raw.get(
                        "request_timeout", generator_raw.get("request_timeout")
                    ),
                    120,
                ),
            )
        else:
            raise TypeError("pro2guard.generator must be a boolean or mapping")
        pro2guard = Pro2GuardConfig(
            enabled=bool(pro2guard_raw.get("enabled", False)),
            mode=str(pro2guard_raw.get("mode", "block")).lower(),
            threshold=float(pro2guard_raw.get("threshold", 0.1)),
            model_path=str(pro2guard_raw.get("model_path", "")),
            dtmc_path=str(pro2guard_raw.get("dtmc_path", "")),
            prism_bin=str(pro2guard_raw.get("prism_bin", "prism")),
            abstraction=str(pro2guard_raw.get("abstraction", "")),
            abstraction_policy_path=str(
                pro2guard_raw.get("abstraction_policy_path", "")
            ),
            unsafe_states=[
                str(item) for item in (pro2guard_raw.get("unsafe_states", []) or [])
            ],
            horizon=int(pro2guard_raw.get("horizon", 20)),
            timeout_seconds=int(pro2guard_raw.get("timeout_seconds", 10)),
            fail_closed=bool(pro2guard_raw.get("fail_closed", False)),
            generator=pro2guard_generator,
        )
        if pro2guard.generator.context_mode not in {"full", "benign_only"}:
            raise ValueError(
                "pro2guard.generator.context_mode must be one of: full, benign_only"
            )
        if pro2guard.generator.max_attempts < 1:
            raise ValueError("pro2guard.generator.max_attempts must be at least 1")
        if pro2guard.generator.max_profiles < 0:
            raise ValueError("pro2guard.generator.max_profiles must be non-negative")
        if pro2guard.generator.max_unsafe_states < 0:
            raise ValueError(
                "pro2guard.generator.max_unsafe_states must be non-negative"
            )
        if pro2guard.generator.max_model_states < 1:
            raise ValueError(
                "pro2guard.generator.max_model_states must be at least 1"
            )
        if pro2guard.generator.state_batch_size < 1:
            raise ValueError(
                "pro2guard.generator.state_batch_size must be at least 1"
            )
        if pro2guard.mode not in {"block", "warn", "monitor"}:
            raise ValueError("pro2guard.mode must be one of: block, warn, monitor")
    else:
        pro2guard = Pro2GuardConfig()

    agentspec_raw = raw.get("agentspec", {}) or {}
    if isinstance(agentspec_raw, bool):
        agentspec = AgentSpecConfig(enabled=agentspec_raw)
    elif isinstance(agentspec_raw, dict):
        inline_rules = agentspec_raw.get("rules", []) or []
        rule_files = agentspec_raw.get("rule_files", []) or []
        predicate_modules = agentspec_raw.get("predicate_modules", []) or []
        if isinstance(inline_rules, str):
            inline_rules = [inline_rules]
        if isinstance(rule_files, str):
            rule_files = [rule_files]
        if isinstance(predicate_modules, str):
            predicate_modules = [predicate_modules]
        generator_raw = agentspec_raw.get("generator", True)
        if isinstance(generator_raw, bool):
            agentspec_generator = AgentSpecGeneratorConfig(enabled=generator_raw)
        elif isinstance(generator_raw, dict):
            generator_llm_raw = generator_raw.get("llm", {}) or {}
            if not isinstance(generator_llm_raw, dict):
                raise TypeError("agentspec.generator.llm must be a mapping")
            agentspec_generator = AgentSpecGeneratorConfig(
                enabled=bool(generator_raw.get("enabled", True)),
                context_mode=str(
                    generator_raw.get("context_mode", "benign_only")
                ).lower(),
                max_attempts=int(generator_raw.get("max_attempts", 2)),
                max_rules=int(generator_raw.get("max_rules", 12)),
                provider=str(
                    generator_llm_raw.get("provider", generator_raw.get("provider", ""))
                ).lower(),
                model=str(
                    generator_llm_raw.get("model", generator_raw.get("model", ""))
                ),
                temperature=(
                    float(generator_llm_raw["temperature"])
                    if "temperature" in generator_llm_raw
                    else (
                        float(generator_raw["temperature"])
                        if "temperature" in generator_raw
                        else None
                    )
                ),
                base_url=str(
                    generator_llm_raw.get("base_url", generator_raw.get("base_url", ""))
                ),
                api_key=str(
                    generator_llm_raw.get("api_key", generator_raw.get("api_key", ""))
                ),
                request_timeout=_optional_int(
                    generator_llm_raw.get(
                        "request_timeout", generator_raw.get("request_timeout")
                    ),
                    120,
                ),
            )
        else:
            raise TypeError("agentspec.generator must be a boolean or mapping")
        agentspec = AgentSpecConfig(
            enabled=bool(agentspec_raw.get("enabled", False)),
            rules=[str(item) for item in inline_rules],
            rule_files=[str(item) for item in rule_files],
            predicate_modules=[str(item) for item in predicate_modules],
            approval_handler=str(agentspec_raw.get("approval_handler", "prompt")),
            max_reflections=int(agentspec_raw.get("max_reflections", 3)),
            fail_closed=bool(agentspec_raw.get("fail_closed", True)),
            generator=agentspec_generator,
        )
        if agentspec.max_reflections < 0:
            raise ValueError("agentspec.max_reflections must be non-negative")
        if agentspec.generator.context_mode not in {"full", "benign_only"}:
            raise ValueError(
                "agentspec.generator.context_mode must be one of: full, benign_only"
            )
        if agentspec.generator.max_attempts < 1:
            raise ValueError("agentspec.generator.max_attempts must be at least 1")
        if agentspec.generator.max_rules < 1:
            raise ValueError("agentspec.generator.max_rules must be at least 1")
    else:
        agentspec = AgentSpecConfig()

    llamafirewall_raw = raw.get("llamafirewall", {}) or {}
    if isinstance(llamafirewall_raw, bool):
        llamafirewall = LlamaFirewallConfig(enabled=llamafirewall_raw)
    elif isinstance(llamafirewall_raw, dict):
        scanners_raw = llamafirewall_raw.get("scanners", {}) or {}
        if not isinstance(scanners_raw, dict):
            raise ValueError(
                "llamafirewall.scanners must be a role-to-scanners mapping"
            )
        llamafirewall = LlamaFirewallConfig(
            enabled=bool(llamafirewall_raw.get("enabled", False)),
            mode=str(llamafirewall_raw.get("mode", "enforce")).lower(),
            use_case=str(llamafirewall_raw.get("use_case", "")).lower(),
            scanners={
                str(role).lower(): (
                    [str(scanner).lower() for scanner in items]
                    if isinstance(items, list)
                    else [str(items).lower()]
                )
                for role, items in scanners_raw.items()
            },
            max_revisions=int(llamafirewall_raw.get("max_revisions", 1)),
            fail_closed=bool(llamafirewall_raw.get("fail_closed", False)),
        )
        if llamafirewall.mode not in {"enforce", "monitor"}:
            raise ValueError("llamafirewall.mode must be one of: enforce, monitor")
        if llamafirewall.use_case not in {"", "chatbot", "coding_assistant"}:
            raise ValueError(
                "llamafirewall.use_case must be one of: chatbot, coding_assistant"
            )
        if llamafirewall.max_revisions < 0:
            raise ValueError("llamafirewall.max_revisions must be non-negative")
    else:
        llamafirewall = LlamaFirewallConfig()

    toolsafe_raw = raw.get("toolsafe", {}) or {}
    if isinstance(toolsafe_raw, bool):
        toolsafe = ToolSafeConfig(enabled=toolsafe_raw)
    elif isinstance(toolsafe_raw, dict):
        toolsafe = ToolSafeConfig(
            enabled=bool(toolsafe_raw.get("enabled", False)),
            mode=str(toolsafe_raw.get("mode", "replan")).lower(),
            threshold=float(toolsafe_raw.get("threshold", 0.5)),
            provider=str(toolsafe_raw.get("provider", "openai_compatible")).lower(),
            model=str(toolsafe_raw.get("model", "TS-Guard")),
            base_url=str(toolsafe_raw.get("base_url", "")),
            api_key=str(toolsafe_raw.get("api_key", "")),
            api_key_env=str(toolsafe_raw.get("api_key_env", "TOOLSAFE_API_KEY")),
            timeout_seconds=float(toolsafe_raw.get("timeout_seconds", 30.0)),
            max_history_steps=int(toolsafe_raw.get("max_history_steps", 20)),
            max_replans=int(toolsafe_raw.get("max_replans", 3)),
            fail_closed=bool(toolsafe_raw.get("fail_closed", False)),
        )
        if toolsafe.mode not in {"replan", "block", "warn", "monitor"}:
            raise ValueError(
                "toolsafe.mode must be one of: replan, block, warn, monitor"
            )
        if not 0.0 <= toolsafe.threshold <= 1.0:
            raise ValueError("toolsafe.threshold must be between 0 and 1")
        if toolsafe.provider != "openai_compatible":
            raise ValueError(
                "toolsafe.provider currently supports only: openai_compatible"
            )
        if toolsafe.timeout_seconds <= 0:
            raise ValueError("toolsafe.timeout_seconds must be greater than zero")
        if toolsafe.max_history_steps < 0:
            raise ValueError("toolsafe.max_history_steps must be non-negative")
        if toolsafe.max_replans < 0:
            raise ValueError("toolsafe.max_replans must be non-negative")
    else:
        toolsafe = ToolSafeConfig()

    agentdog_raw = raw.get("agentdog", {}) or {}
    if isinstance(agentdog_raw, bool):
        agentdog = AgentDoGConfig(enabled=agentdog_raw)
    elif isinstance(agentdog_raw, dict):
        agentdog_task = str(agentdog_raw.get("task", "unified")).lower()
        default_agentdog_model = (
            "AgentDoG1.5-Qwen3.5-4B"
            if agentdog_task == "coarse"
            else "AgentDoG1.5-Unified-Qwen3.5-4B"
        )
        checkpoints_raw = agentdog_raw.get("checkpoints", ["pre_reply"])
        if isinstance(checkpoints_raw, str):
            checkpoints = [checkpoints_raw]
        elif isinstance(checkpoints_raw, list):
            checkpoints = [str(item) for item in checkpoints_raw]
        else:
            raise TypeError("agentdog.checkpoints must be a string or list")
        agentdog = AgentDoGConfig(
            enabled=bool(agentdog_raw.get("enabled", False)),
            mode=str(agentdog_raw.get("mode", "diagnose")).lower(),
            task=agentdog_task,
            checkpoints=[item.lower() for item in checkpoints],
            provider=str(agentdog_raw.get("provider", "openai_compatible")).lower(),
            model=str(agentdog_raw.get("model", default_agentdog_model)),
            base_url=str(agentdog_raw.get("base_url", "")),
            base_url_env=str(agentdog_raw.get("base_url_env", "AGENTDOG_BASE_URL")),
            api_key=str(agentdog_raw.get("api_key", "")),
            api_key_env=str(agentdog_raw.get("api_key_env", "AGENTDOG_API_KEY")),
            timeout_seconds=float(agentdog_raw.get("timeout_seconds", 60.0)),
            temperature=float(agentdog_raw.get("temperature", 0.0)),
            max_tokens=int(agentdog_raw.get("max_tokens", 1024)),
            max_trajectory_chars=int(agentdog_raw.get("max_trajectory_chars", 0)),
            max_revisions=int(agentdog_raw.get("max_revisions", 2)),
            fail_closed=bool(agentdog_raw.get("fail_closed", False)),
            include_raw_response=bool(agentdog_raw.get("include_raw_response", True)),
            replacement_message=str(
                agentdog_raw.get(
                    "replacement_message",
                    AgentDoGConfig().replacement_message,
                )
            ),
        )
        if agentdog.mode not in {"diagnose", "revise", "gate"}:
            raise ValueError("agentdog.mode must be one of: diagnose, revise, gate")
        if agentdog.task not in {"unified", "coarse"}:
            raise ValueError("agentdog.task must be one of: unified, coarse")
        if agentdog.provider != "openai_compatible":
            raise ValueError(
                "agentdog.provider currently supports only: openai_compatible"
            )
        if not agentdog.checkpoints:
            raise ValueError("agentdog.checkpoints must not be empty")
        invalid_checkpoints = set(agentdog.checkpoints) - {
            "pre_action",
            "pre_reply",
        }
        if invalid_checkpoints:
            raise ValueError(
                "agentdog.checkpoints supports only: pre_action, pre_reply"
            )
        if len(set(agentdog.checkpoints)) != len(agentdog.checkpoints):
            raise ValueError("agentdog.checkpoints must not contain duplicates")
        if agentdog.timeout_seconds <= 0:
            raise ValueError("agentdog.timeout_seconds must be greater than zero")
        if not 0.0 <= agentdog.temperature <= 2.0:
            raise ValueError("agentdog.temperature must be between 0 and 2")
        if agentdog.max_tokens < 1:
            raise ValueError("agentdog.max_tokens must be positive")
        if agentdog.max_trajectory_chars < 0:
            raise ValueError("agentdog.max_trajectory_chars must be non-negative")
        if agentdog.max_revisions < 0:
            raise ValueError("agentdog.max_revisions must be non-negative")
        if not agentdog.replacement_message.strip():
            raise ValueError("agentdog.replacement_message must not be empty")
    else:
        agentdog = AgentDoGConfig()

    agentguard_raw = raw.get("agentguard", {}) or {}
    if isinstance(agentguard_raw, bool):
        agentguard = AgentGuardConfig(enabled=agentguard_raw)
    elif isinstance(agentguard_raw, dict):
        sandbox_profile_raw = agentguard_raw.get("sandbox_profile")
        if sandbox_profile_raw is not None and not isinstance(
            sandbox_profile_raw, dict
        ):
            raise ValueError("agentguard.sandbox_profile must be a mapping or null")
        scenario_raw = agentguard_raw.get("scenario_compiler", True)
        if isinstance(scenario_raw, bool):
            scenario_compiler = AgentGuardScenarioCompilerConfig(enabled=scenario_raw)
        elif isinstance(scenario_raw, dict):
            scenario_llm_raw = scenario_raw.get("llm", {}) or {}
            if not isinstance(scenario_llm_raw, dict):
                raise TypeError("agentguard.scenario_compiler.llm must be a mapping")
            request_timeout_raw = scenario_llm_raw.get("request_timeout")
            inline_scenario_api_key = str(scenario_llm_raw.get("api_key", "") or "")
            if inline_scenario_api_key:
                raise ValueError(
                    "agentguard.scenario_compiler.llm.api_key must not be stored in "
                    "YAML; set api_key_env and export that environment variable instead"
                )
            scenario_api_key_env = str(
                scenario_llm_raw.get("api_key_env", "") or ""
            )
            scenario_compiler = AgentGuardScenarioCompilerConfig(
                enabled=bool(scenario_raw.get("enabled", True)),
                context_mode=str(scenario_raw.get("context_mode", "full")).lower(),
                max_attempts=int(scenario_raw.get("max_attempts", 2)),
                provider=str(scenario_llm_raw.get("provider", "")),
                model=str(scenario_llm_raw.get("model", "")),
                temperature=(
                    float(scenario_llm_raw["temperature"])
                    if scenario_llm_raw.get("temperature") is not None
                    else None
                ),
                base_url=str(scenario_llm_raw.get("base_url", "")),
                api_key=(
                    os.environ.get(scenario_api_key_env, "")
                    if scenario_api_key_env
                    else ""
                ),
                api_key_env=scenario_api_key_env,
                request_timeout=(
                    int(request_timeout_raw)
                    if request_timeout_raw is not None
                    else None
                ),
            )
        else:
            raise TypeError("agentguard.scenario_compiler must be a boolean or mapping")
        if scenario_compiler.max_attempts < 1:
            raise ValueError(
                "agentguard.scenario_compiler.max_attempts must be positive"
            )
        if scenario_compiler.context_mode not in {"full", "benign_only"}:
            raise ValueError(
                "agentguard.scenario_compiler.context_mode must be one of: "
                "full, benign_only"
            )
        agentguard = AgentGuardConfig(
            enabled=bool(agentguard_raw.get("enabled", False)),
            mode=str(agentguard_raw.get("mode", "block")).lower(),
            policy=str(agentguard_raw.get("policy", "")),
            server_url=str(
                agentguard_raw.get(
                    "server_url",
                    agentguard_raw.get(
                        "remote_url", os.environ.get("AGENTGUARD_SERVER_URL", "")
                    ),
                )
            ),
            api_key=str(
                agentguard_raw.get("api_key", os.environ.get("AGENTGUARD_API_KEY", ""))
            ),
            plugin_config=str(agentguard_raw.get("plugin_config", "")),
            environment=str(agentguard_raw.get("environment", "")),
            user_id=str(agentguard_raw.get("user_id", "")),
            role=str(agentguard_raw.get("role", "default")),
            trust_level=int(agentguard_raw.get("trust_level", 1)),
            sandbox=str(agentguard_raw.get("sandbox", "local")),
            sandbox_profile=dict(sandbox_profile_raw)
            if isinstance(sandbox_profile_raw, dict)
            else None,
            audit_path=str(agentguard_raw.get("audit_path", "")),
            max_steps=int(agentguard_raw.get("max_steps", 12)),
            max_tool_calls=int(agentguard_raw.get("max_tool_calls", 24)),
            window_size=int(agentguard_raw.get("window_size", 8)),
            remote_timeout_seconds=float(
                agentguard_raw.get("remote_timeout_seconds", 5.0)
            ),
            remote_retries=int(agentguard_raw.get("remote_retries", 2)),
            fail_closed=bool(agentguard_raw.get("fail_closed", True)),
            scenario_compiler=scenario_compiler,
        )
        if agentguard.mode not in {"block", "warn", "monitor"}:
            raise ValueError("agentguard.mode must be one of: block, warn, monitor")
        if (
            agentguard.max_steps < 1
            or agentguard.max_tool_calls < 1
            or agentguard.window_size < 1
        ):
            raise ValueError(
                "agentguard max_steps, max_tool_calls, and window_size must be positive"
            )
        if agentguard.remote_timeout_seconds <= 0 or agentguard.remote_retries < 0:
            raise ValueError("agentguard remote timeout and retries values are invalid")
    else:
        agentguard = AgentGuardConfig()

    agentsight_raw = raw.get("agentsight", {}) or {}
    if isinstance(agentsight_raw, bool):
        agentsight = AgentSightConfig(enabled=agentsight_raw)
    elif isinstance(agentsight_raw, dict):
        agentsight = AgentSightConfig(
            enabled=bool(agentsight_raw.get("enabled", False)),
            binary=str(agentsight_raw.get("binary", "agentsight")),
            capture=str(agentsight_raw.get("capture", "full")).lower(),
            db_path=str(agentsight_raw.get("db_path", "agentsight.db")),
            snapshot_path=str(
                agentsight_raw.get("snapshot_path", "agentsight_snapshot.json")
            ),
            log_path=str(agentsight_raw.get("log_path", "agentsight.log")),
            required=bool(agentsight_raw.get("required", False)),
            privilege=str(agentsight_raw.get("privilege", "auto")).lower(),
            web_server=bool(agentsight_raw.get("web_server", False)),
            server_port=int(agentsight_raw.get("server_port", 7395)),
            startup_timeout_seconds=float(
                agentsight_raw.get("startup_timeout_seconds", 10.0)
            ),
            warmup_seconds=float(agentsight_raw.get("warmup_seconds", 1.0)),
            shutdown_timeout_seconds=float(
                agentsight_raw.get("shutdown_timeout_seconds", 10.0)
            ),
        )
        if agentsight.capture not in {"system", "full"}:
            raise ValueError("agentsight.capture must be one of: system, full")
        if agentsight.privilege not in {"auto", "sudo", "none"}:
            raise ValueError("agentsight.privilege must be one of: auto, sudo, none")
        if agentsight.server_port < 1 or agentsight.server_port > 65535:
            raise ValueError("agentsight.server_port must be between 1 and 65535")
        if agentsight.startup_timeout_seconds <= 0:
            raise ValueError(
                "agentsight.startup_timeout_seconds must be greater than zero"
            )
        if agentsight.warmup_seconds < 0:
            raise ValueError("agentsight.warmup_seconds must be non-negative")
        if agentsight.warmup_seconds > agentsight.startup_timeout_seconds:
            raise ValueError(
                "agentsight.warmup_seconds must not exceed startup_timeout_seconds"
            )
        if agentsight.shutdown_timeout_seconds <= 0:
            raise ValueError(
                "agentsight.shutdown_timeout_seconds must be greater than zero"
            )
        for field_name in ("db_path", "snapshot_path", "log_path"):
            value = str(getattr(agentsight, field_name)).strip()
            if not value:
                raise ValueError(f"agentsight.{field_name} must not be empty")
            if Path(value).is_absolute():
                raise ValueError(
                    f"agentsight.{field_name} must be relative to the job directory"
                )
            if ".." in Path(value).parts:
                raise ValueError(
                    f"agentsight.{field_name} must stay inside the job directory"
                )
    else:
        agentsight = AgentSightConfig()

    container_raw = raw.get("container", {}) or {}
    if isinstance(container_raw, bool):
        container = ContainerConfig(enabled=container_raw)
    elif isinstance(container_raw, dict):
        env_raw = container_raw.get("env", None)
        if env_raw is None:
            env = ContainerConfig().env
        elif isinstance(env_raw, str):
            env = [env_raw]
        elif isinstance(env_raw, list):
            env = [str(item) for item in env_raw]
        else:
            env = []
        container = ContainerConfig(
            enabled=bool(container_raw.get("enabled", True)),
            image=str(container_raw.get("image", "agent-scaffold:latest")),
            auto_build=bool(container_raw.get("auto_build", True)),
            dockerfile=str(container_raw.get("dockerfile", "Dockerfile")),
            workdir=str(container_raw.get("workdir", "/workspace")),
            network=str(container_raw.get("network", "host")),
            remove=bool(container_raw.get("remove", True)),
            build_args={
                str(key): str(value)
                for key, value in (container_raw.get("build_args", {}) or {}).items()
            },
            env=env,
        )
    else:
        container = ContainerConfig()

    trip_raw = raw.get("trip") or {}
    trip = trip_raw if isinstance(trip_raw, dict) else {}
    research_raw = raw.get("research") or {}
    research = research_raw if isinstance(research_raw, dict) else {}

    return AppConfig(
        llm=llm,
        agent=agent,
        tools=tools,
        graph=graph,
        monitoring=monitoring,
        skills=skills,
        planner=planner,
        middleware=middleware,
        security=security,
        aegis=aegis,
        pro2guard=pro2guard,
        agentspec=agentspec,
        llamafirewall=llamafirewall,
        toolsafe=toolsafe,
        agentdog=agentdog,
        agentguard=agentguard,
        agentsight=agentsight,
        agentdojo=agentdojo,
        agent_security_bench=agent_security_bench,
        agentharm=agentharm,
        container=container,
        trip=trip,
        research=research,
        config_dir=str(config_path.parent),
    )
