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
    from .config import load_config
    from .graph import build_graph
    from .nodes import build_initial_messages
except ImportError:  # Fallback when executed as a script
    from agent_scaffold.config import load_config
    from agent_scaffold.graph import build_graph
    from agent_scaffold.nodes import build_initial_messages


def run_once(cfg_path: str, user_input: str | None) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    graph = build_graph(cfg)

    run_dir = (Path.cwd() / Path(cfg_path).stem).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    task = cfg.agent.task.strip()
    initial_messages = [{"role": "user", "content": task}] if task else []
    input_messages = (
        [{"role": "user", "content": user_input}] if user_input else []
    )
    state = {
        "messages": build_initial_messages(cfg) + initial_messages + input_messages,
        "tool_call": None,
        "iterations": 0,
        "trace": [],
    }
    prev_cwd = Path.cwd()
    try:
        # Ensure all generated files (including write_text_file outputs) go into run_dir.
        os.chdir(run_dir)
        result = graph.invoke(state)
    finally:
        os.chdir(prev_cwd)
    run_end = time.time()
    if cfg.monitoring.enabled:
        output_path = run_dir / f"trace_{Path(cfg_path).stem}.json"
        payload = {
            "input": user_input,
            "final": result.get("messages", [])[-1]["content"] if result.get("messages") else "",
            "run_dir": str(run_dir),
            "timestamp": run_start,
            "latency_ms": int((run_end - run_start) * 1000),
            "trace": result.get("trace", []),
        }
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
