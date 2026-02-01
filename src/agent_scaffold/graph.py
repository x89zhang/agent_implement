from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .config import AppConfig
from .llm import LLMAdapter
from .nodes import agent_node, load_tool, tool_node


def build_graph(cfg: AppConfig) -> Any:
    llm = LLMAdapter(cfg.llm)
    tools = {t.name: load_tool(t) for t in cfg.tools}

    builder: StateGraph = StateGraph(dict)
    builder.add_node("agent", agent_node(cfg, llm))
    builder.add_node("tool", tool_node(cfg, tools))

    def _route(state: dict[str, Any]) -> str:
        if state.get("tool_call"):
            if int(state.get("iterations", 0)) >= cfg.graph.max_iters:
                return END
            return "tool"
        return END

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", _route)
    builder.add_edge("tool", "agent")
    return builder.compile()
