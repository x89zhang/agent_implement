from __future__ import annotations

from dataclasses import dataclass
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
    react_max_iterations: int = 15
    react_max_execution_time: int = 120


@dataclass
class MonitoringConfig:
    enabled: bool = False
    output_path: str = "trace.json"
    print_trace: bool = False


@dataclass
class AppConfig:
    llm: LLMConfig
    agent: AgentConfig
    tools: list[ToolConfig]
    graph: GraphConfig
    monitoring: MonitoringConfig


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    llm_raw = _require(raw, "llm")
    agent_raw = _require(raw, "agent")
    graph_raw = raw.get("graph", {})

    llm = LLMConfig(
        provider=str(_require(llm_raw, "provider")),
        model=str(_require(llm_raw, "model")),
        temperature=float(llm_raw.get("temperature", 0.2)),
        base_url=str(llm_raw.get("base_url", "")),
        api_key=str(llm_raw.get("api_key", "")),
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

    graph = GraphConfig(
        type=str(graph_raw.get("type", "single_agent")),
        max_iters=int(graph_raw.get("max_iters", 4)),
        tool_call_format=str(graph_raw.get("tool_call_format", "TOOL_CALL: <name> <json>")),
        stop_keyword=str(graph_raw.get("stop_keyword", "FINAL")),
        react_prompt=str(graph_raw.get("react_prompt", "")),
        react_max_iterations=int(graph_raw.get("react_max_iterations", 15)),
        react_max_execution_time=int(graph_raw.get("react_max_execution_time", 120)),
    )

    monitoring_raw = raw.get("monitoring", {}) or {}
    monitoring = MonitoringConfig(
        enabled=bool(monitoring_raw.get("enabled", False)),
        output_path=str(monitoring_raw.get("output_path", "trace.json")),
        print_trace=bool(monitoring_raw.get("print_trace", False)),
    )

    return AppConfig(llm=llm, agent=agent, tools=tools, graph=graph, monitoring=monitoring)
