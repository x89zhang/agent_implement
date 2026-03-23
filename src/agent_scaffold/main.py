from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import json
import os
import re
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


def _resolve_trip_date_range(trip: dict[str, Any]) -> tuple[str | None, str | None]:
    start = str(trip.get("start", "")).strip().lower()
    days = trip.get("days")
    try:
        days_int = int(days) if days is not None else 7
    except Exception:
        days_int = 7

    today = dt.date.today()
    if start == "next monday":
        days_ahead = (7 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        start_date = today + dt.timedelta(days=days_ahead)
    else:
        try:
            start_date = dt.date.fromisoformat(start)
        except Exception:
            start_date = None

    if not start_date:
        return None, None

    end_date = start_date + dt.timedelta(days=max(1, days_int) - 1)
    return start_date.isoformat(), end_date.isoformat()


def _augment_task_with_trip_dates(task: str, trip: dict[str, Any]) -> str:
    if not task:
        return task
    start_date, end_date = _resolve_trip_date_range(trip)
    if not start_date or not end_date:
        return task
    note = (
        f"\n- Resolved trip dates for this run: {start_date} to {end_date}."
        "\n- When checking weather or mentioning dates, use these exact dates and do not guess the year."
    )
    if "Resolved trip dates for this run:" in task:
        return task
    return f"{task.rstrip()}{note}"


def _expected_output_file(run_dir: Path, task: str) -> Path:
    match = re.search(r'The file name MUST be "([^"]+)"', task)
    if match:
        return run_dir / match.group(1)
    return run_dir / f"{run_dir.name}.txt"


def _extract_write_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    marker = re.search(r"Action:\s*write_text_file\s*[\r\n]+Action Input:\s*", text, re.DOTALL)
    if not marker:
        return None
    start = marker.end()
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None
    try:
        payload = json.loads(text[start:end])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _recover_written_file(result: dict[str, Any], run_dir: Path, task: str) -> Path | None:
    target = _expected_output_file(run_dir, task)
    if target.exists():
        return target

    trace = result.get("trace", [])
    payload: dict[str, Any] | None = None
    for entry in reversed(trace if isinstance(trace, list) else []):
        output = entry.get("output", {}) if isinstance(entry, dict) else {}
        steps = output.get("intermediate_steps", []) if isinstance(output, dict) else []
        for step in reversed(steps if isinstance(steps, list) else []):
            if not isinstance(step, dict):
                continue
            tool_name = str(step.get("tool", ""))
            tool_input = step.get("tool_input")
            if tool_name == "write_text_file":
                if isinstance(tool_input, dict):
                    payload = tool_input
                elif isinstance(tool_input, str):
                    try:
                        parsed = json.loads(tool_input)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        payload = parsed
                if payload:
                    break
            log_text = str(step.get("log", ""))
            payload = _extract_write_payload(log_text)
            if payload:
                break
        if payload:
            break

    if not payload:
        return None

    path = str(payload.get("path") or target.name)
    content = str(payload.get("content") or "")
    mode = str(payload.get("mode") or "w")
    if not content:
        return None

    destination = (run_dir / path).resolve()
    if not str(destination).startswith(str(run_dir.resolve())):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, mode, encoding="utf-8") as f:
        f.write(content)
    return destination


def run_once(cfg_path: str, user_input: str | None) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    graph = build_graph(cfg)

    cfg_file = Path(cfg_path).resolve()
    run_dir = cfg_file.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    task = _augment_task_with_trip_dates(cfg.agent.task.strip(), cfg.trip)
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
    recovered_output = _recover_written_file(result, run_dir, task)
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
