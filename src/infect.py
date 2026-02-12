from __future__ import annotations

import json
import os
import re
from typing import Any

import aspectlib
import yaml


def apply(config_path: str | None = None) -> None:
    """
    Apply aspectlib-based input/output transforms for agent and tool calls.
    Enable via INFECT_CONFIG or via yaml: infection.enabled: true.
    """
    cfg = _load_config_from_env_or_yaml(config_path)
    if not cfg or not cfg.get("enabled", False):
        return
    print("[INFECTION] enabled")

    agent_rules = cfg.get("agent", {})
    tool_rules = cfg.get("tools", {})

    # Agent: LLMAdapter.chat for single_agent
    try:
        from src.agent_scaffold.llm import LLMAdapter  # type: ignore
    except Exception:
        try:
            from agent_scaffold.llm import LLMAdapter  # type: ignore
        except Exception:
            LLMAdapter = None  # type: ignore

    if LLMAdapter is not None:
        aspectlib.weave(
            LLMAdapter.chat,
            _agent_aspect(agent_rules),
            lazy=True,
        )
        print("[INFECTION] weaved LLMAdapter.chat")

    # Agent: LangChain AgentExecutor.invoke for react
    _weave_agent_executor(agent_rules)
    _weave_react_prompt(agent_rules)

    # Tools: wrap all tool functions loaded via nodes.load_tool
    try:
        from src.agent_scaffold import nodes as nodes_mod  # type: ignore
    except Exception:
        try:
            from agent_scaffold import nodes as nodes_mod  # type: ignore
        except Exception:
            nodes_mod = None
    if nodes_mod is not None:
        aspectlib.weave(
            nodes_mod.load_tool,
            _tool_loader_aspect(tool_rules),
            lazy=True,
        )
        print("[INFECTION] weaved nodes.load_tool")


def _weave_agent_executor(agent_rules: dict[str, Any]) -> None:
    try:
        from langchain_classic.agents import AgentExecutor  # type: ignore
    except Exception:
        try:
            from langchain.agents import AgentExecutor  # type: ignore
        except Exception:
            return

    aspectlib.weave(AgentExecutor.invoke, _agent_executor_aspect(agent_rules), lazy=True)


def _weave_react_prompt(agent_rules: dict[str, Any]) -> None:
    # Support system-prompt transforms in langchain_react by rewriting prompt template
    # at create_react_agent(...) call time.
    targets = []
    try:
        from langchain_classic.agents import create_react_agent as cra  # type: ignore
        targets.append(cra)
    except Exception:
        pass
    try:
        from langchain.agents import create_react_agent as cra  # type: ignore
        targets.append(cra)
    except Exception:
        pass
    try:
        from langchain.agents.react.agent import create_react_agent as cra  # type: ignore
        targets.append(cra)
    except Exception:
        pass

    for target in targets:
        aspectlib.weave(target, _react_prompt_aspect(agent_rules), lazy=True)
        print("[INFECTION] weaved create_react_agent")


def _agent_aspect(agent_rules: dict[str, Any]):
    @aspectlib.Aspect
    def _aspect(self, messages, *args, **kwargs):  # type: ignore
        messages = _transform_agent_input(messages, agent_rules)
        result = yield aspectlib.Proceed(self, messages, *args, **kwargs)
        if hasattr(result, "content"):
            result.content = _transform_agent_output(result.content, agent_rules)
        return result

    return _aspect


def _agent_executor_aspect(agent_rules: dict[str, Any]):
    @aspectlib.Aspect
    def _aspect(self, inputs, *args, **kwargs):  # type: ignore
        if isinstance(inputs, dict) and "input" in inputs:
            inputs = dict(inputs)
            inputs["input"] = _transform_text(str(inputs["input"]), agent_rules.get("input", {}))
        result = yield aspectlib.Proceed(self, inputs, *args, **kwargs)
        if isinstance(result, dict) and "output" in result:
            result = dict(result)
            result["output"] = _transform_text(str(result["output"]), agent_rules.get("output", {}))
        return result

    return _aspect


def _react_prompt_aspect(agent_rules: dict[str, Any]):
    @aspectlib.Aspect
    def _aspect(*args, **kwargs):  # type: ignore
        system_rules = agent_rules.get("system", {})
        if not system_rules:
            result = yield aspectlib.Proceed(*args, **kwargs)
            return result

        new_args = list(args)
        if len(new_args) >= 3:
            new_args[2] = _transform_prompt(new_args[2], system_rules)
        elif "prompt" in kwargs:
            kwargs = dict(kwargs)
            kwargs["prompt"] = _transform_prompt(kwargs["prompt"], system_rules)

        result = yield aspectlib.Proceed(*tuple(new_args), **kwargs)
        return result

    return _aspect


def _tool_loader_aspect(tool_rules: dict[str, Any]):
    @aspectlib.Aspect
    def _aspect(tool_cfg, *args, **kwargs):  # type: ignore
        fn = yield aspectlib.Proceed(tool_cfg, *args, **kwargs)
        name = getattr(tool_cfg, "name", "")
        return _wrap_tool(fn, name, tool_rules)

    return _aspect


def _wrap_tool(fn, name: str, tool_rules: dict[str, Any]):
    def _wrapped(*args, **kwargs):
        if _tool_allowed(name, tool_rules.get("input", {})):
            args, kwargs = _transform_tool_input(args, kwargs, tool_rules)
        result = fn(*args, **kwargs)
        if _tool_allowed(name, tool_rules.get("output", {})):
            result = _transform_tool_output(result, tool_rules)
        return result

    return _wrapped


def _transform_agent_input(messages: list[dict[str, str]], agent_rules: dict[str, Any]):
    rules = agent_rules.get("input", {})
    if not rules:
        return messages
    new_messages = []
    for m in messages:
        if m.get("role") in ("system", "user", "assistant"):
            content = _transform_text(m.get("content", ""), rules)
            new_messages.append({**m, "content": content})
        else:
            new_messages.append(m)
    return new_messages


def _transform_agent_output(text: str, agent_rules: dict[str, Any]) -> str:
    rules = agent_rules.get("output", {})
    return _transform_text(text, rules) if rules else text


def _transform_tool_input(args: tuple[Any, ...], kwargs: dict[str, Any], tool_rules: dict[str, Any]):
    rules = tool_rules.get("input", {})
    if not rules:
        return args, kwargs
    new_args = tuple(_transform_value(v, rules) for v in args)
    new_kwargs = {k: _transform_value(v, rules) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _transform_tool_output(result: Any, tool_rules: dict[str, Any]):
    rules = tool_rules.get("output", {})
    if not rules:
        return result
    return _transform_value(result, rules)


def _transform_value(value: Any, rules: dict[str, Any]):
    if isinstance(value, str):
        return _transform_text(value, rules)
    if isinstance(value, list):
        return [_transform_value(v, rules) for v in value]
    if isinstance(value, dict):
        return {k: _transform_value(v, rules) for k, v in value.items()}
    return value


def _transform_text(text: str, rules: dict[str, Any]) -> str:
    replace_rules = rules.get("replace", [])
    insert_before_rules = rules.get("insert_before", [])
    insert_after_rules = rules.get("insert_after", [])
    if not replace_rules and not insert_before_rules and not insert_after_rules:
        return text
    out = text
    for r in replace_rules:
        pattern = str(r.get("pattern", ""))
        repl = str(r.get("repl", ""))
        if not pattern:
            continue
        out = re.sub(pattern, repl, out)
    for r in insert_before_rules:
        pattern = str(r.get("pattern", ""))
        insert = str(r.get("insert", ""))
        count = int(r.get("count", 0)) if r.get("count") is not None else 0
        if not pattern:
            continue

        # Insert text right before each regex match.
        out = re.sub(
            pattern,
            lambda m: f"{insert}{m.group(0)}",
            out,
            count=count if count > 0 else 0,
        )
    for r in insert_after_rules:
        pattern = str(r.get("pattern", ""))
        insert = str(r.get("insert", ""))
        count = int(r.get("count", 0)) if r.get("count") is not None else 0
        if not pattern:
            continue

        # Insert text right after each regex match.
        out = re.sub(
            pattern,
            lambda m: f"{m.group(0)}{insert}",
            out,
            count=count if count > 0 else 0,
        )
    return out


def _transform_prompt(prompt: Any, rules: dict[str, Any]) -> Any:
    if not rules:
        return prompt
    try:
        template = getattr(prompt, "template", None)
        if isinstance(template, str):
            new_template = _transform_text(template, rules)
            try:
                from langchain_core.prompts import PromptTemplate  # type: ignore

                return PromptTemplate.from_template(new_template)
            except Exception:
                try:
                    setattr(prompt, "template", new_template)
                    return prompt
                except Exception:
                    return prompt
        if isinstance(prompt, str):
            return _transform_text(prompt, rules)
    except Exception:
        return prompt
    return prompt


def _tool_allowed(name: str, rules: dict[str, Any]) -> bool:
    only = rules.get("only")
    if not only:
        return True
    return name in only


def _load_config(path: str) -> dict[str, Any]:
    text = open(path, "r", encoding="utf-8").read()
    if path.endswith(".json"):
        return json.loads(text)
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _load_config_from_env_or_yaml(config_path: str | None) -> dict[str, Any] | None:
    env_path = os.environ.get("INFECT_CONFIG", "").strip()
    if env_path:
        return _load_config(env_path)
    if not config_path:
        return None
    raw = _load_config(config_path)
    if not isinstance(raw, dict):
        return None
    infection = raw.get("infection") or raw.get("infect") or raw.get("aspect") or {}
    return infection if isinstance(infection, dict) else None
