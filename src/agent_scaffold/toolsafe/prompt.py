from __future__ import annotations

from typing import Any

from .upstream_prompt import TOOLSAFE_PROMPT


def build_guard_prompt(
    *,
    user_request: str,
    interaction_history: list[dict[str, Any]],
    current_action: dict[str, Any],
    env_info: str,
) -> str:
    # Mirrors ToolSafe src/model/model.py:128-131: the agent_action dict is
    # rendered by str.format, i.e. with Python repr, not JSON.
    return TOOLSAFE_PROMPT.format(
        env_info=env_info,
        user_request=user_request,
        agent_action={
            "interaction_history": interaction_history,
            "current_action": current_action,
        },
    )


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Port of SecReAct_Agent.format_tools_for_prompt (sec_react_agent.py:22-31).

    Parameters follow the AgentDojo pipeline's parse_envs
    (agent_pipeline.py:279-308): ``{name: {"description", "type"}}``.
    """
    result = []
    for tool in tools:
        name = str(tool.get("name", ""))
        result.append(f"{name}: {tool.get('description', '')}")
        params = _tool_params(tool.get("inputSchema") or tool.get("parameters"))
        if params:
            param_str = "\n    ".join(f"{k}: {v}" for k, v in params.items())
            result.append(f"  parameters:\n    {param_str}")
        result.append("")
    return "\n".join(result)


def _tool_params(schema: Any) -> dict[str, Any]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    params: dict[str, Any] = {}
    for p_name, p_info in (properties or {}).items():
        p_info = p_info if isinstance(p_info, dict) else {}
        param_type = p_info.get("type")
        if not param_type and "anyOf" in p_info:
            param_type = " | ".join(
                t.get("type", "unknown") for t in p_info["anyOf"] if isinstance(t, dict)
            )
        params[p_name] = {
            "description": p_info.get("description", ""),
            "type": param_type,
        }
    return params
