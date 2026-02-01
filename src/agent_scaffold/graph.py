from __future__ import annotations

from typing import Any
import time

from langgraph.graph import END, StateGraph

from .config import AppConfig
from .llm import LLMAdapter
from .nodes import agent_node, load_tool, tool_node


def build_graph(cfg: AppConfig) -> Any:
    if cfg.graph.type == "langchain_react":
        return _build_langchain_react_graph(cfg)

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


def _build_langchain_react_graph(cfg: AppConfig) -> Any:
    llm = LLMAdapter(cfg.llm)
    lc_model = llm.get_lc_chat_model()

    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        from langchain_core.prompts import PromptTemplate  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime import
        raise RuntimeError("Missing dependency: langchain_core") from exc

    create_react_agent = None
    initialize_agent = None
    AgentType = None
    AgentExecutor = None

    try:
        from langchain_classic.agents import create_react_agent, AgentExecutor  # type: ignore
    except Exception:
        try:
            from langchain.agents import create_react_agent  # type: ignore
        except Exception:
            try:
                from langchain.agents.react.agent import create_react_agent  # type: ignore
            except Exception:
                try:
                    from langchain.agents import initialize_agent, AgentType  # type: ignore
                except Exception:
                    try:
                        from langchain_classic.agents import initialize_agent, AgentType  # type: ignore
                    except Exception as exc:  # pragma: no cover - runtime import
                        raise RuntimeError(
                            "Missing dependency: langchain or langchain-classic (react agent)"
                        ) from exc

    if AgentExecutor is None:
        try:
            from langchain.agents import AgentExecutor  # type: ignore
        except Exception:
            try:
                from langchain.agents.agent import AgentExecutor  # type: ignore
            except Exception:
                try:
                    from langchain.agents.agent_executor import AgentExecutor  # type: ignore
                except Exception:
                    try:
                        from langchain_classic.agents import AgentExecutor  # type: ignore
                    except Exception as exc:  # pragma: no cover - runtime import
                        raise RuntimeError(
                            "Missing dependency: langchain or langchain-classic (AgentExecutor)"
                        ) from exc

    prompt_text = cfg.graph.react_prompt.strip()
    role_prefix = cfg.agent.system_prompt.strip()
    if prompt_text:
        if role_prefix:
            prompt_text = f"{role_prefix}\n\n{prompt_text}"
        PROMPT = PromptTemplate.from_template(prompt_text)
    else:
        try:
            from langchain.agents.react.prompt import PROMPT  # type: ignore
            if role_prefix:
                PROMPT = PromptTemplate.from_template(f"{role_prefix}\n\n{PROMPT.template}")
        except Exception:
            prompt_text = (
                "You are a helpful assistant.\n\n"
                "Answer the following questions as best you can. You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use the following format:\n\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                "Action: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can repeat)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original question\n\n"
                "Question: {input}\n"
                "{agent_scratchpad}"
            )
            if role_prefix:
                prompt_text = f"{role_prefix}\n\n{prompt_text}"
            PROMPT = PromptTemplate.from_template(prompt_text)

    tools = []
    for t in cfg.tools:
        fn = load_tool(t)
        tools.append(
            StructuredTool.from_function(
                fn,
                name=t.name,
                description=t.description or "",
            )
        )

    if create_react_agent:
        agent = create_react_agent(lc_model, tools, PROMPT)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
        )
    else:
        executor = initialize_agent(
            tools,
            lc_model,
            agent=AgentType.REACT_DESCRIPTION,
            verbose=False,
            return_intermediate_steps=True,
        )

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        user_input = state["messages"][-1]["content"] if state.get("messages") else ""
        result = executor.invoke({"input": user_input})
        output = result.get("output", "")
        state["messages"].append({"role": "assistant", "content": output})

        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append(
                {
                    "tool": getattr(action, "tool", ""),
                    "tool_input": getattr(action, "tool_input", ""),
                    "log": getattr(action, "log", ""),
                    "observation": observation,
                }
            )

        trace = state.setdefault("trace", [])
        trace.append(
            {
                "step": "langchain_react",
                "timestamp": start,
                "latency_ms": int((time.time() - start) * 1000),
                "input": {"input": user_input},
                "output": {"content": output, "intermediate_steps": steps},
            }
        )
        return state

    builder: StateGraph = StateGraph(dict)
    builder.add_node("react", _node)
    builder.set_entry_point("react")
    builder.add_edge("react", END)
    return builder.compile()
