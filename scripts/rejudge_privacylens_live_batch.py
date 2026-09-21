#!/usr/bin/env python3
"""Re-run only the PrivacyLens judge for an existing batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = ROOT / "scripts" / "privacylens_live_official_bridge.py"
PRIVACYLENS_REVISION = "994ac15db6fff8a5131bbf5a26e84e352e676796"
EVALUATOR_REVISION = "9c2ee07b080dc54ed4924af11d9751e81753c94d"


def load_bridge() -> Any:
    spec = importlib.util.spec_from_file_location("privacylens_live_official_bridge", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PrivacyLens bridge: {BRIDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if (
            event.get("step") == "tool"
            and event.get("tool") == "gmail_send_message"
            and event.get("status") == "completed"
        ):
            arguments = event.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"Invalid send arguments at {path}:{line_number}")
            messages.append(arguments)
    return messages


def serialize_outbound(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "GmailSendEmail\nAction Input: "
        + json.dumps(
            {
                "to": message.get("to", []),
                "subject": message.get("subject", ""),
                "body": message.get("body", ""),
            },
            ensure_ascii=False,
        )
        for message in messages
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the PrivacyLens judge without rerunning the agent or tools."
    )
    parser.add_argument("batch", type=Path)
    parser.add_argument("--case", default="item64")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--revision", default=PRIVACYLENS_REVISION)
    parser.add_argument("--evaluator-source-dir", type=Path, required=True)
    parser.add_argument("--evaluator-revision", default=EVALUATOR_REVISION)
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--judge-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = args.batch.resolve()
    if not os.environ.get(args.judge_api_key_env):
        raise SystemExit(f"{args.judge_api_key_env} is not set")
    bridge = load_bridge()
    source_dir = args.source_dir.resolve()
    revision = bridge.source_revision(source_dir)
    if revision != args.revision:
        raise SystemExit(
            f"PrivacyLens-Live revision mismatch: expected {args.revision}, found {revision}"
        )
    data_path = bridge.data_file(source_dir, "")
    row, case_index = bridge.load_case(data_path, args.case)
    request = {
        "evaluator_source_dir": str(args.evaluator_source_dir.resolve()),
        "evaluator_revision": args.evaluator_revision,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_api_key_env": args.judge_api_key_env,
        "judge_timeout_seconds": args.timeout,
    }
    results: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in batch.glob("run_*") if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"No run_* directories found under {batch}")
    for run_dir in run_dirs:
        calls_path = run_dir / "target" / "tool_calls.jsonl"
        messages = completed_messages(calls_path)
        evaluation = bridge.judge(
            {**request, "outbound": serialize_outbound(messages)}, row
        )
        old_path = run_dir / "target" / "evaluation.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        result = {
            "run": run_dir.name,
            "sent_message_count": len(messages),
            "old_has_leakage": bool(old.get("has_leakage")),
            **evaluation,
        }
        results.append(result)
        leaked = [j["item"] for j in evaluation["judgments"] if j["leaked"]]
        print(f"{run_dir.name}: leakage={evaluation['has_leakage']} leaked_items={len(leaked)}")
    leaked_runs = sum(bool(result["has_leakage"]) for result in results)
    output = {
        "benchmark": "privacylens_live",
        "revision": revision,
        "case": row.get("name", f"Item {case_index}"),
        "judge_model": args.judge_model,
        "compatibility_layer": "luna_generic_v2",
        "run_count": len(results),
        "attack_success_count": leaked_runs,
        "asr": leaked_runs / len(results),
        "results": results,
    }
    output_path = args.output or batch / "rejudge_luna_generic_v2.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ASR: {leaked_runs}/{len(results)} = {output['asr']:.1%}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
