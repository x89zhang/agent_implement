from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "agent_scaffold"

try:
    from dataclasses import asdict as _asdict
    from .config import load_config
    from .graph import build_graph
    from .nodes import build_initial_messages, _flush_trace_snapshot
    from .planner import initialize_plan
    from .skills import load_enabled_skills, validate_skill_tools
    from .tools import augment_task_with_research_context, augment_task_with_trip_context, recover_written_file
except ImportError:  # Fallback when executed as a script
    from dataclasses import asdict as _asdict
    from agent_scaffold.config import load_config
    from agent_scaffold.graph import build_graph
    from agent_scaffold.nodes import build_initial_messages, _flush_trace_snapshot
    from agent_scaffold.planner import initialize_plan
    from agent_scaffold.skills import load_enabled_skills, validate_skill_tools
    from agent_scaffold.tools import augment_task_with_research_context, augment_task_with_trip_context, recover_written_file

def _normalize_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("messages must be a list")
    messages: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"message at index {idx} must be an object")
        role = str(item.get("role", "")).strip()
        content = item.get("content", "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message at index {idx} has unsupported role: {role!r}")
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})
    return messages


def _load_context_messages(path: str) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "messages" in payload:
            return _normalize_messages(payload["messages"])
        raise ValueError("context file object must contain a 'messages' field")
    return _normalize_messages(payload)


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_messages: list[dict[str, str]] = []
    other_messages: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "system":
            system_messages.append(message)
        else:
            other_messages.append(message)
    return system_messages, other_messages


def run_once(
    cfg_path: str,
    user_input: str | None,
    context_messages: list[dict[str, str]] | None = None,
    resume_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    graph = build_graph(cfg)

    cfg_file = Path(cfg_path).resolve()
    run_dir = cfg_file.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    task = augment_task_with_trip_context(cfg.agent.task.strip(), cfg.trip)
    task = augment_task_with_research_context(task, cfg.research)
    initial_messages = [{"role": "user", "content": task}] if task else []
    input_messages = (
        [{"role": "user", "content": user_input}] if user_input else []
    )
    base_messages = build_initial_messages(cfg)
    enabled_skills = load_enabled_skills(cfg)
    skill_tool_warnings = validate_skill_tools(enabled_skills, {tool.name for tool in cfg.tools})
    plan = initialize_plan(cfg, task or (user_input or ""))
    seeded_messages = list(context_messages or [])
    resumed_messages = list(resume_messages or [])
    if resumed_messages:
        resumed_system, resumed_non_system = _split_system_messages(resumed_messages)
        state_messages = (resumed_system[:1] or base_messages) + resumed_non_system + input_messages
    else:
        state_messages = base_messages + initial_messages + seeded_messages + input_messages
    output_path: Path | None = None
    if cfg.monitoring.enabled:
        configured_output = cfg.monitoring.output_path.strip()
        if configured_output:
            output_path = Path(configured_output)
            if not output_path.is_absolute():
                output_path = run_dir / output_path
        else:
            output_path = run_dir / f"trace_{Path(cfg_path).stem}.json"
    state = {
        "messages": state_messages,
        "tool_call": None,
        "iterations": 0,
        "trace": [],
        "trace_messages": [dict(message) for message in state_messages],
        "trace_stats": {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "plan": plan,
        "tool_errors": [],
        "harness": {
            "skills": [skill.to_trace() for skill in enabled_skills],
            "skill_warnings": skill_tool_warnings,
            "planner": {
                "enabled": cfg.planner.enabled,
                "type": cfg.planner.type,
                "max_steps": cfg.planner.max_steps,
            },
            "middleware": {
                "enabled": cfg.middleware.enabled,
                "modules": cfg.middleware.modules or ["HarnessMiddleware"] if cfg.middleware.enabled else [],
            },
        },
        "_trace_persist": {
            "output_path": str(output_path) if output_path else "",
            "run_dir": str(run_dir),
            "timestamp": run_start,
            "started_at": run_start,
            "config": _asdict(cfg),
            "input": user_input,
        },
    }
    _flush_trace_snapshot(state)
    prev_cwd = Path.cwd()
    prev_cfg_env = os.environ.get("AGENT_CONFIG_PATH")
    try:
        # Ensure all generated files (including write_text_file outputs) go into run_dir.
        os.environ["AGENT_CONFIG_PATH"] = str(cfg_file)
        os.chdir(run_dir)
        result = graph.invoke(state)
    finally:
        if prev_cfg_env is None:
            os.environ.pop("AGENT_CONFIG_PATH", None)
        else:
            os.environ["AGENT_CONFIG_PATH"] = prev_cfg_env
        os.chdir(prev_cwd)
    run_end = time.time()
    recovered_output = recover_written_file(result, run_dir, task)
    if recovered_output is not None:
        result.setdefault("messages", []).append(
            {
                "role": "assistant",
                "content": f"Recovered from tool parsing failure and saved output to {recovered_output.name}.",
            }
        )
        result.setdefault("trace", []).append(
            {
                "step": "postprocess",
                "timestamp": run_end,
                "latency_ms": 0,
                "input": {"task": task},
                "output": {"recovered_file": str(recovered_output)},
                "usage": None,
            }
        )
        result.setdefault("trace_messages", []).append(
            {
                "role": "assistant",
                "content": f"Recovered from tool parsing failure and saved output to {recovered_output.name}.",
            }
        )
    _flush_trace_snapshot(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent scaffolding")
    parser.add_argument("--config", required=True, help="config.yaml path")
    parser.add_argument("--input", help="single input to run once")
    parser.add_argument("--context-file", help="JSON file containing context messages or an object with a messages field")
    parser.add_argument("--resume-from", help="JSON trace/context file to resume from; uses its messages field as prior conversation")
    args = parser.parse_args()

    try:
        src_path = str(Path(__file__).resolve().parent.parent)
        if src_path not in sys.path:
            sys.path.append(src_path)
        import infect  # type: ignore

        infect.apply(args.config)
    except Exception as exc:
        print(f"[INFECTION] disabled due to error: {exc}")

    context_messages = _load_context_messages(args.context_file) if args.context_file else None
    resume_messages = _load_context_messages(args.resume_from) if args.resume_from else None

    if args.input is not None:
        result = run_once(args.config, args.input, context_messages=context_messages, resume_messages=resume_messages)
        messages = result.get("messages", [])
        if messages:
            print(messages[-1]["content"])
        return

    cfg = load_config(args.config)
    if cfg.agent.task.strip():
        result = run_once(args.config, None, context_messages=context_messages, resume_messages=resume_messages)
        messages = result.get("messages", [])
        if messages:
            print(messages[-1]["content"])
        return

    # Simple interactive mode
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        result = run_once(args.config, user_input, context_messages=context_messages, resume_messages=resume_messages)
        messages = result.get("messages", [])
        if messages:
            print(f"agent> {messages[-1]['content']}")


if __name__ == "__main__":
    main()
