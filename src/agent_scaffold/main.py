from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .config import load_config
from .graph import build_graph
from .nodes import build_initial_messages


def run_once(cfg_path: str, user_input: str) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    graph = build_graph(cfg)

    run_start = time.time()
    task = cfg.agent.task.strip()
    initial_messages = [{"role": "user", "content": task}] if task else []
    state = {
        "messages": build_initial_messages(cfg)
        + initial_messages
        + [{"role": "user", "content": user_input}],
        "tool_call": None,
        "iterations": 0,
        "trace": [],
    }
    result = graph.invoke(state)
    run_end = time.time()
    if cfg.monitoring.enabled:
        payload = {
            "input": user_input,
            "final": result.get("messages", [])[-1]["content"] if result.get("messages") else "",
            "timestamp": run_start,
            "latency_ms": int((run_end - run_start) * 1000),
            "trace": result.get("trace", []),
        }
        with open(cfg.monitoring.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent scaffolding")
    parser.add_argument("--config", required=True, help="config.yaml path")
    parser.add_argument("--input", help="single input to run once")
    args = parser.parse_args()

    if args.input:
        result = run_once(args.config, args.input)
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
