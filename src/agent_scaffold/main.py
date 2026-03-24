from __future__ import annotations

import argparse
from dataclasses import asdict
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
    from .config import load_config
    from .graph import build_graph
    from .nodes import build_initial_messages
    from .tools import augment_task_with_research_context, augment_task_with_trip_context, recover_written_file
except ImportError:  # Fallback when executed as a script
    from agent_scaffold.config import load_config
    from agent_scaffold.graph import build_graph
    from agent_scaffold.nodes import build_initial_messages
    from agent_scaffold.tools import augment_task_with_research_context, augment_task_with_trip_context, recover_written_file


def run_once(cfg_path: str, user_input: str | None) -> dict[str, Any]:
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
    state = {
        "messages": build_initial_messages(cfg) + initial_messages + input_messages,
        "tool_call": None,
        "iterations": 0,
        "trace": [],
        "trace_messages": build_initial_messages(cfg) + initial_messages + input_messages,
        "trace_stats": {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
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
    if cfg.monitoring.enabled:
        configured_output = cfg.monitoring.output_path.strip()
        if configured_output:
            output_path = Path(configured_output)
            if not output_path.is_absolute():
                output_path = run_dir / output_path
        else:
            output_path = run_dir / f"trace_{Path(cfg_path).stem}.json"
        stats = result.get("trace_stats", {})
        payload = {
            "info": {
                "model_stats": {
                    "api_calls": int(stats.get("api_calls", 0)),
                    "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                    "completion_tokens": int(stats.get("completion_tokens", 0)),
                    "total_tokens": int(stats.get("total_tokens", 0)),
                },
                "config": asdict(cfg),
                "final": result.get("messages", [])[-1]["content"] if result.get("messages") else "",
                "run_dir": str(run_dir),
                "timestamp": run_start,
                "latency_ms": int((run_end - run_start) * 1000),
            },
            "messages": result.get("trace_messages", []),
            "trajectory_format": "agent_scaffold.v2",
            "input": user_input,
            "final": result.get("messages", [])[-1]["content"] if result.get("messages") else "",
            "run_dir": str(run_dir),
            "timestamp": run_start,
            "latency_ms": int((run_end - run_start) * 1000),
            "trace": result.get("trace", []),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent scaffolding")
    parser.add_argument("--config", required=True, help="config.yaml path")
    parser.add_argument("--input", help="single input to run once")
    args = parser.parse_args()

    try:
        src_path = str(Path(__file__).resolve().parent.parent)
        if src_path not in sys.path:
            sys.path.append(src_path)
        import infect  # type: ignore

        infect.apply(args.config)
    except Exception as exc:
        print(f"[INFECTION] disabled due to error: {exc}")

    if args.input is not None:
        result = run_once(args.config, args.input)
        messages = result.get("messages", [])
        if messages:
            print(messages[-1]["content"])
        return

    cfg = load_config(args.config)
    if cfg.agent.task.strip():
        result = run_once(args.config, None)
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
        result = run_once(args.config, user_input)
        messages = result.get("messages", [])
        if messages:
            print(f"agent> {messages[-1]['content']}")


if __name__ == "__main__":
    main()
