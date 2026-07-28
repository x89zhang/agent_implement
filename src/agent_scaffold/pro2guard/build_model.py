from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .abstraction import ToolTraceAbstraction


ATTACK_SUCCESS_STATE = "ATTACK_SUCCESS"
SAFE_TERMINAL_STATE = "SAFE_TERMINAL"


def build_model_from_traces(paths: list[str], unsafe_states: list[str] | None = None) -> dict[str, Any]:
    abstraction = ToolTraceAbstraction()
    sequences: list[list[str]] = []
    for pattern in paths:
        for trace_path in glob.glob(pattern):
            sequence = _states_from_trace(Path(trace_path), abstraction)
            if sequence:
                sequences.append(sequence)

    states = sorted({state for sequence in sequences for state in sequence})
    state_index = {state: idx for idx, state in enumerate(states)}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            counts[state_index[left]][state_index[right]] += 1

    transition_probs: dict[str, dict[str, str]] = {}
    for state, src in state_index.items():
        row = counts.get(src, {})
        total = sum(row.values())
        if total == 0:
            transition_probs[str(src)] = {str(src): "1.0"}
        else:
            transition_probs[str(src)] = {
                str(dst): f"{count}/{total}" for dst, count in sorted(row.items())
            }

    return {
        "states": states,
        "state_index": state_index,
        "transition_counts": {str(src): {str(dst): count for dst, count in row.items()} for src, row in counts.items()},
        "transition_probs": transition_probs,
        "unsafe_states": list(unsafe_states or []),
        "terminal_states": {
            "attack_success": ATTACK_SUCCESS_STATE,
            "safe": SAFE_TERMINAL_STATE,
        },
        "format": "agent_scaffold.pro2guard.json_dtmc.v1",
    }


def _states_from_trace(path: Path, abstraction: ToolTraceAbstraction) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequence: list[str] = []
    for entry in raw.get("trace", []) if isinstance(raw, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("step") == "tool":
            tool_input = entry.get("input") or {}
            output = entry.get("output") or {}
            name = str(tool_input.get("tool", ""))
            payload = tool_input.get("args", {})
            result = str(output.get("result", ""))
            failed = result.startswith("Tool execution failed:") or result.startswith("Tool not found:")
            sequence.append(abstraction.encode_tool_result(name, payload, result, failed))
        elif entry.get("step") == "langchain_react":
            for step in (entry.get("output") or {}).get("intermediate_steps", []) or []:
                if not isinstance(step, dict):
                    continue
                pg_state = (step.get("pro2guard") or {}).get("state")
                if pg_state:
                    sequence.append(str(pg_state))
                    continue
                name = str(step.get("tool", ""))
                payload = step.get("tool_input", {})
                result = str(step.get("observation", ""))
                failed = bool(step.get("blocked")) or result.startswith("Tool execution failed:") or result.startswith("Tool not found:")
                sequence.append(abstraction.encode_tool_result(name, payload, result, failed))
    if sequence:
        _append_agentdojo_terminal(sequence, raw)
        return sequence

    for message in raw.get("messages", []) if isinstance(raw, dict) else []:
        extra = message.get("extra") if isinstance(message, dict) else None
        if not isinstance(extra, dict) or "tool" not in extra:
            continue
        name = str(extra.get("tool", ""))
        payload = extra.get("args", extra.get("tool_input", {}))
        result = str(extra.get("raw_output", ""))
        failed = bool(extra.get("returncode"))
        sequence.append(abstraction.encode_tool_result(name, payload, result, failed))
    _append_agentdojo_terminal(sequence, raw)
    return sequence


def _append_agentdojo_terminal(sequence: list[str], raw: dict[str, Any]) -> None:
    attack_success = _agentdojo_attack_success(raw)
    if attack_success is True:
        sequence.append(ATTACK_SUCCESS_STATE)
    elif attack_success is False:
        sequence.append(SAFE_TERMINAL_STATE)


def _agentdojo_attack_success(raw: dict[str, Any]) -> bool | None:
    for entry in raw.get("trace", []) if isinstance(raw, dict) else []:
        if not isinstance(entry, dict) or entry.get("step") != "agentdojo_eval":
            continue
        output = entry.get("output")
        if isinstance(output, dict) and "attack_success" in output:
            return bool(output.get("attack_success"))
    final = str((raw.get("info") or {}).get("final", "") or raw.get("final", ""))
    if "attack_success=True" in final:
        return True
    if "attack_success=False" in final:
        return False
    harness_eval = (raw.get("harness") or {}).get("agentdojo") if isinstance(raw, dict) else None
    if isinstance(harness_eval, dict) and "attack_success" in harness_eval:
        return bool(harness_eval.get("attack_success"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pro2Guard JSON DTMC from scaffold trace files.")
    parser.add_argument("traces", nargs="+", help="Trace JSON files or glob patterns.")
    parser.add_argument("--output", required=True, help="Output JSON DTMC path.")
    parser.add_argument("--unsafe-state", action="append", default=[], help="Unsafe state id/name. Repeatable. Use ATTACK_SUCCESS for AgentDojo attack-success terminal states.")
    args = parser.parse_args()

    model = build_model_from_traces(args.traces, args.unsafe_state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(model['states'])} states to {output}")


if __name__ == "__main__":
    main()
