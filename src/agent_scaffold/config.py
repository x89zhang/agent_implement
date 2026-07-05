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
    request_timeout: int | None = 120


@dataclass
class ToolConfig:
    name: str
    import_path: str
    description: str


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
class AgentDojoConfig:
    enabled: bool = False
    suite: str = "workspace"
    benchmark_version: str = "v1.2.2"
    case: str = ""
    user_task: str = "user_task_0"
    injection_task: str = ""
    trusted_tool_output_prompt: bool = True
    injections: dict[str, str] = field(default_factory=dict)


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
    agentdojo: AgentDojoConfig = field(default_factory=AgentDojoConfig)
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
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "unlimited"}:
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
    if injection_task.startswith("injection_") and not injection_task.startswith("injection_task_"):
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


def _apply_harness_files(raw: dict[str, Any], harness_dir: Path | None) -> dict[str, Any]:
    if harness_dir is None or not harness_dir.exists():
        return raw
    merged = dict(raw)

    system_prompt_path = harness_dir / "systemprompt.md"
    if system_prompt_path.exists():
        agent_raw = dict(merged.get("agent") or {})
        agent_raw["system_prompt"] = system_prompt_path.read_text(encoding="utf-8").strip()
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

    llm = LLMConfig(
        provider=str(_require(llm_raw, "provider")),
        model=str(_require(llm_raw, "model")),
        temperature=float(llm_raw.get("temperature", 0.2)),
        base_url=str(llm_raw.get("base_url", "")),
        api_key=str(llm_raw.get("api_key", "")),
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
            if "injection_task" in agentdojo_raw and raw_injection_task != case_injection_task:
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
            trusted_tool_output_prompt=bool(agentdojo_raw.get("trusted_tool_output_prompt", True)),
            injections={
                str(key): str(value)
                for key, value in (
                    agentdojo_raw.get("custom_injections")
                    if agentdojo_raw.get("custom_injections") is not None
                    else agentdojo_raw.get("injections", {})
                or {}).items()
            },
        )
    else:
        agentdojo = AgentDojoConfig()

    if agentdojo.enabled and not _container_enabled_from_raw(raw):
        module_name = f"{__package__}.agentdojo_adapter" if __package__ else "agent_scaffold.agentdojo_adapter"
        adapter = importlib.import_module(module_name)
        tools = [
            ToolConfig(name=name, import_path=import_path, description=description)
            for name, import_path, description in adapter.build_tool_configs(agentdojo)
        ]

    graph = GraphConfig(
        type=str(graph_raw.get("type", "single_agent")),
        max_iters=int(graph_raw.get("max_iters", 4)),
        tool_call_format=str(graph_raw.get("tool_call_format", "TOOL_CALL: <name> <json>")),
        stop_keyword=str(graph_raw.get("stop_keyword", "FINAL")),
        react_prompt=str(graph_raw.get("react_prompt", "")),
        react_max_iterations=_optional_int(graph_raw.get("react_max_iterations", 15), 15),
        react_max_execution_time=_optional_int(graph_raw.get("react_max_execution_time", 120), 120),
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
        middleware = MiddlewareConfig(enabled=True, modules=[str(item) for item in middleware_raw])
    elif isinstance(middleware_raw, dict):
        middleware = MiddlewareConfig(
            enabled=bool(middleware_raw.get("enabled", True)),
            modules=[str(item) for item in (middleware_raw.get("modules", []) or [])],
        )
    else:
        middleware = MiddlewareConfig()

    security_raw = raw.get("security", {}) or {}
    legacy_tool_prompt = None
    if isinstance(agentdojo_raw, dict) and "trusted_tool_output_prompt" in agentdojo_raw:
        legacy_tool_prompt = bool(agentdojo_raw.get("trusted_tool_output_prompt", True))
    if isinstance(security_raw, bool):
        security = SecurityConfig(trusted_tool_output_prompt=security_raw)
    elif isinstance(security_raw, dict):
        security = SecurityConfig(
            trusted_tool_output_prompt=bool(security_raw.get("trusted_tool_output_prompt", legacy_tool_prompt if legacy_tool_prompt is not None else True)),
        )
    else:
        security = SecurityConfig(
            trusted_tool_output_prompt=legacy_tool_prompt if legacy_tool_prompt is not None else True,
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
            allow_tools=[str(item) for item in (aegis_raw.get("allow_tools", []) or [])],
            block_tools=[str(item) for item in (aegis_raw.get("block_tools", []) or [])],
        )
    else:
        aegis = AegisConfig()

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
            build_args={str(key): str(value) for key, value in (container_raw.get("build_args", {}) or {}).items()},
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
        agentdojo=agentdojo,
        container=container,
        trip=trip,
        research=research,
        config_dir=str(config_path.parent),
    )
