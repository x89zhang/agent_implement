from __future__ import annotations

import json
from typing import Any

from .upstream_prompt import TOOLSAFE_PROMPT


def build_guard_prompt(
    *,
    user_request: str,
    interaction_history: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
    environment: list[dict[str, Any]],
) -> str:
    # Keep the upstream input fields: user_request, agent_action, env_info.
    agent_action = {
        "interaction_history": interaction_history,
        "current_action": {"tool_name": tool_name, "arguments": arguments},
    }
    return TOOLSAFE_PROMPT.format(
        user_request=user_request,
        agent_action=json.dumps(agent_action, ensure_ascii=False, default=str),
        env_info=json.dumps(environment, ensure_ascii=False, default=str),
    )
