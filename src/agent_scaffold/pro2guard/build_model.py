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


def build_model_from_traces(
    paths: list[str], *, alpha: float = 1.0, abstraction_policy_path: str = "",
) -> dict[str, Any]:
    """Apply upstream validity-aware Laplace counts to Hermes trace states."""
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    abstraction = (ToolTraceAbstraction.from_policy_file(abstraction_policy_path)
                   if abstraction_policy_path else ToolTraceAbstraction())
    sequences: list[list[str]] = []
    trace_paths = _resolve_trace_paths(paths)
    if not trace_paths:
        raise ValueError("No trace_agentdojo.json files matched the supplied inputs")
    for trace_path in trace_paths:
        sequence = _states_from_trace(trace_path, abstraction)
        if sequence:
            sequences.append(sequence)

    states = sorted({state for sequence in sequences for state in sequence})
    state_index = {state: idx for idx, state in enumerate(states)}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            counts[state_index[left]][state_index[right]] += 1

    transition_probs = smooth_transition_counts(states, state_index, counts, alpha=alpha)

    return {
        "states": states,
        "alpha": alpha,
        "state_index": state_index,
        "transition_counts": {str(src): {str(dst): count for dst, count in row.items()} for src, row in counts.items()},
        "transition_probs": transition_probs,
        "terminal_states": {
            "attack_success": ATTACK_SUCCESS_STATE,
            "safe": SAFE_TERMINAL_STATE,
        },
        "format": "agent_scaffold.pro2guard.json_dtmc.v1",
    }



def smooth_transition_counts(
    states: list[str], state_index: dict[str, int],
    counts: dict[int, dict[int, int]], *, alpha: float = 1.0,
) -> dict[str, dict[str, str]]:
    """Upstream-style Laplace probabilities with benchmark terminal semantics."""
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    transition_probs: dict[str, dict[str, str]] = {}
    for state, src in state_index.items():
        if state in {ATTACK_SUCCESS_STATE, SAFE_TERMINAL_STATE}:
            # Domain-specific valid_trans: benchmark outcome states end a run.
            transition_probs[str(src)] = {str(src): "1.0"}
            continue
        row = counts.get(src, {})
        # In upstream build_model, valid_trans contributes alpha to each
        # allowed successor. Hermes states have no additional transition
        # restrictions, so every observed state is an allowed successor.
        denominator = sum(row.values()) + alpha * len(states)
        if denominator == 0:
            transition_probs[str(src)] = {str(src): "1.0"}
        else:
            transition_probs[str(src)] = {
                str(dst): f"{row.get(dst, 0) + alpha}/{denominator}"
                for dst in range(len(states)) if row.get(dst, 0) + alpha > 0
            }
    return transition_probs


def _resolve_trace_paths(inputs: list[str]) -> list[Path]:
    resolved: dict[str, Path] = {}
    for value in inputs:
        direct = Path(value)
        matches = (
            [direct]
            if direct.exists()
            else [Path(item) for item in glob.glob(value, recursive=True)]
        )
        for match in matches:
            candidates = (
                match.rglob("trace_agentdojo.json") if match.is_dir() else [match]
            )
            for candidate in candidates:
                if candidate.is_file():
                    resolved[str(candidate.resolve())] = candidate
    return [resolved[key] for key in sorted(resolved)]


def _states_from_trace(path: Path, abstraction: ToolTraceAbstraction) -> list[str]:
    # In replay mode trace_agentdojo.json concatenates one copy of the same
    # Hermes trajectory for every monitor. Use the original tool journal so
    # each tool call contributes exactly once to the learned DTMC.
    lifecycle = path.parent / "target" / "guard_lifecycle.jsonl"
    journal = path.parent / "target" / "tool_calls.jsonl"
    evaluation = path.parent / "target" / "evaluation.json"
    if path.name == "trace_agentdojo.json" and lifecycle.exists():
        if not evaluation.exists():
            raise ValueError(f"ProbGuard training run has no evaluation label: {path}")
        sequence: list[str] = []
        previous_outcome = "unknown"
        pending: tuple[str, Any] | None = None
        for line in lifecycle.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("op") == "before_tool":
                if pending is not None:
                    raise ValueError(f"Unmatched ProbGuard tool call in {lifecycle}")
                name = str(event.get("name", ""))
                payload = event.get("arguments", {})
                sequence.append(abstraction.encode_tool_call(
                    {"_pro2guard_last_outcome": previous_outcome}, name, payload,
                ))
                pending = (name, payload)
            elif event.get("op") == "after_tool":
                if pending != (str(event.get("name", "")), event.get("arguments", {})):
                    raise ValueError(f"Unmatched ProbGuard tool result in {lifecycle}")
                previous_outcome = "failed" if event.get("failed") else "ok"
                pending = None
        if pending is not None:
            raise ValueError(f"Unfinished ProbGuard tool call in {lifecycle}")
        if sequence:
            label = json.loads(evaluation.read_text(encoding="utf-8"))
            _append_agentdojo_terminal(sequence, {"harness": {"agentdojo": label}})
        return sequence
    if path.name == "trace_agentdojo.json" and journal.exists():
        if not evaluation.exists():
            raise ValueError(f"ProbGuard training run has no evaluation label: {path}")
        completed: dict[int, dict[str, Any]] = {}
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("status") in {"completed", "failed"}:
                completed[int(event["sequence"])] = event
        if sorted(completed) != list(range(len(completed))):
            raise ValueError(f"ProbGuard training tool journal is incomplete: {journal}")
        sequence: list[str] = []
        previous_outcome = "unknown"
        for event in (completed[index] for index in sorted(completed)):
            sequence.append(abstraction.encode_tool_call(
                {"_pro2guard_last_outcome": previous_outcome},
                str(event.get("tool", "")), event.get("arguments", {}),
            ))
            previous_outcome = "failed" if event["status"] == "failed" else "ok"
        if sequence:
            label = json.loads(evaluation.read_text(encoding="utf-8"))
            _append_agentdojo_terminal(sequence, {"harness": {"agentdojo": label}})
        return sequence

    raw = json.loads(path.read_text(encoding="utf-8"))
    sequence: list[str] = []
    previous_outcome = "unknown"

    def append_tool(name: str, payload: Any, result: str, failed: bool,
                    recorded_state: str = "") -> None:
        nonlocal previous_outcome
        if recorded_state:
            sequence.append(recorded_state)
        else:
            # The runtime queries before_tool, before the current result exists.
            sequence.append(abstraction.encode_tool_call(
                {"_pro2guard_last_outcome": previous_outcome}, name, payload,
            ))
        previous_outcome = "failed" if failed else "ok"

    for entry in raw.get("trace", []) if isinstance(raw, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("step") == "tool":
            tool_input = entry.get("input") or {}
            output = entry.get("output") or ""
            # Hermes lifecycle events store the tool name at the top level and
            # arguments directly in input. Older scaffold traces nest both.
            name = str(entry.get("tool") or tool_input.get("tool", ""))
            payload = (tool_input.get("args", {}) if "tool" in tool_input
                       else tool_input)
            result = str(output.get("result", "") if isinstance(output, dict)
                         else output)
            failed = (entry.get("status") == "failed" or bool(entry.get("is_error"))
                      or result.startswith("Tool execution failed:")
                      or result.startswith("Tool not found:"))
            append_tool(name, payload, result, failed)
        elif entry.get("step") == "langchain_react":
            for step in (entry.get("output") or {}).get("intermediate_steps", []) or []:
                if not isinstance(step, dict):
                    continue
                name = str(step.get("tool", ""))
                payload = step.get("tool_input", {})
                result = str(step.get("observation", ""))
                failed = (bool(step.get("blocked"))
                          or result.startswith("Tool execution failed:")
                          or result.startswith("Tool not found:"))
                recorded_state = str((step.get("pro2guard") or {}).get("state") or "")
                append_tool(name, payload, result, failed, recorded_state)
    if sequence:
        _append_agentdojo_terminal(sequence, raw)
        return sequence

    for message in raw.get("messages", []) if isinstance(raw, dict) else []:
        extra = message.get("extra") if isinstance(message, dict) else None
        if not isinstance(extra, dict) or "tool" not in extra:
            continue
        append_tool(str(extra.get("tool", "")),
                    extra.get("args", extra.get("tool_input", {})),
                    str(extra.get("raw_output", "")), bool(extra.get("returncode")))
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
    parser.add_argument(
        "traces",
        nargs="+",
        help="Trace JSON files, glob patterns, or directories searched recursively.",
    )
    parser.add_argument("--output", required=True, help="Output JSON DTMC path.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Upstream Laplace smoothing strength.")
    parser.add_argument(
        "--abstraction-policy", default="",
        help="Fixed JSON tool profiles also used by the runtime abstraction.",
    )
    args = parser.parse_args()

    model = build_model_from_traces(
        args.traces, alpha=args.alpha,
        abstraction_policy_path=args.abstraction_policy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(model['states'])} states to {output}")


if __name__ == "__main__":
    main()
