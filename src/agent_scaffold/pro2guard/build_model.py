"""Learn per-task ProbGuard DTMCs from recorded Hermes lifecycles.

Mirrors upstream ``safereach/build_model.py`` (state encoding, validity-aware
Laplace smoothing, PRISM export) and ``embodied/build.py`` (one abstraction and
one DTMC per task, each trace ending in the absorbing ``FINISH`` state).

Input traces are ``guard_lifecycle.jsonl`` files written by Hermes runs under
``jobs/``. Only the executed steps (``after_tool`` events: tool, arguments,
result) are used. No benchmark outcome or attack label is read; unsafe states
come from the generated specification. Traces must come from runs that are
disjoint from the runs later evaluated with the model; ``trace_sources`` is
stored in ``model.json`` and the runtime refuses a model trained on the run it
is evaluating.

Usage::

    PYTHONPATH=src python -m agent_scaffold.pro2guard.build_model \\
        --config agents/hermes/agentdojo-all-monitors.yaml \\
        --output models/pro2guard/trained jobs/PROBGUARD_TRAINING_SPLIT
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .abstraction import FINISH, PredicateAbstraction, step_observation
from .generator import (
    ABSTRACTION_FILE,
    INDEX_FILE,
    MODEL_FILE,
    PRISM_FILE,
    generate_abstraction,
    task_key,
)


def build_model(logs: list[list[Any]], abs: PredicateAbstraction, alpha: float = 1.0) -> dict[str, Any]:
    """Upstream ``build_model``: encode observations, count, Laplace-smooth.

    ``logs`` holds, per trace, the agent state after each step (a list of step
    observations) followed by ``FINISH``.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    state_transitions: list[list[str]] = []
    state_space: list[str] = []
    seen: set[str] = set()
    for log in logs:
        state_tran = []
        for obs in log:
            state = abs.encode(obs)
            state_tran.append(state)
            if state not in seen:
                seen.add(state)
                state_space.append(state)
        state_transitions.append(state_tran)
    state_space.sort()
    K = len(state_space)
    abs.state_idx = None
    abs.state_interpretation = None
    state_idx = abs.get_state_idx(state_space)
    state_interpret = abs.get_state_interpretation(state_space)
    counts = [[0] * K for _ in range(K)]
    for state_tran in state_transitions:
        for prev, state in zip(state_tran, state_tran[1:]):
            counts[state_idx[prev]][state_idx[state]] += 1

    transition_probs: dict[int, dict[int, str]] = {}
    for s_from in state_space:
        i = state_idx[s_from]
        # Validity-aware Laplace smoothing: alpha is added to the denominator
        # only for valid successors (FINISH has none, so it stays absorbing).
        denom = sum(
            counts[i][state_idx[s_to]] + (alpha if abs.valid_trans(s_from, s_to) else 0)
            for s_to in state_space
        )
        transition_probs[i] = {
            state_idx[s_to]: _fraction(counts[i][state_idx[s_to]] + alpha, denom)
            for s_to in state_space
            if denom != 0 and abs.valid_trans(s_from, s_to)
        }
        if not transition_probs[i]:
            transition_probs[i][i] = "1.0"
    return {
        "states": state_space,
        "state_index": state_idx,
        "state_interpret": state_interpret,
        "transition_counts": {
            i: {j: counts[i][j] for j in range(K) if counts[i][j] > 0}
            for i in range(K) if any(counts[i])
        },
        "transition_probs": transition_probs,
    }


def _fraction(numerator: float, denominator: float) -> str:
    if float(numerator).is_integer() and float(denominator).is_integer():
        return f"{int(numerator)}/{int(denominator)}"
    return f"{numerator}/{denominator}"


def export_dtmc_to_prism(model: dict[str, Any], file_path: str | Path, initial_state: int = 0) -> None:
    """Upstream ``export_dtmc_to_prism`` (extra unobserved state K included)."""
    K = len(model["states"])
    lines = ["dtmc", "", "module dtmc_model", "", f"    s : [0..{K}] init {initial_state};", ""]
    for i, row in model["transition_probs"].items():
        transitions = [f"{prob} : (s'={j})" for j, prob in row.items()]
        if transitions:
            lines.append(f"    [] s={i} -> {' + '.join(transitions)};")
    lines += [f"    [] s={K} -> 1.0: (s'={K});", "", "endmodule", ""]
    Path(file_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hermes lifecycles.

def read_lifecycle(path: str | Path) -> dict[str, Any]:
    """Return the task, tool inventory and executed steps of one lifecycle."""
    path = Path(path)
    task, tools, steps = "", [], []
    initialized = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        op = event.get("op")
        if op == "initialize":
            initialized = True
            # Generators and model keys use the clean task, as at runtime.
            task = str(event.get("generation_task") or event.get("task") or "")
            tools = list(event.get("tools") or [])
        elif op == "after_tool":
            steps.append(step_observation(
                str(event.get("name", "")), event.get("arguments", {}),
                event.get("result", ""), bool(event.get("failed", False)),
            ))
    if not initialized:
        raise ValueError(f"lifecycle has no initialize event: {path}")
    return {"task": task, "tools": tools, "steps": steps, "path": str(path.resolve())}


def trace_log(steps: list[dict[str, Any]]) -> list[Any]:
    """Agent state after each step, then ``FINISH`` (embodied/build.py:17-18)."""
    return [steps[: index + 1] for index in range(len(steps))] + [FINISH]


def find_lifecycles(inputs: list[str], exclude: list[str] | None = None) -> list[Path]:
    found: dict[str, Path] = {}
    for value in inputs:
        direct = Path(value)
        matches = [direct] if direct.exists() else [Path(item) for item in glob.glob(value, recursive=True)]
        for match in matches:
            candidates = match.rglob("guard_lifecycle.jsonl") if match.is_dir() else [match]
            for candidate in candidates:
                if candidate.is_file():
                    found[str(candidate.resolve())] = candidate.resolve()
    excluded: set[str] = set()
    for pattern in exclude or []:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            targets = path.rglob("guard_lifecycle.jsonl") if path.is_dir() else [path]
            excluded |= {str(target.resolve()) for target in targets}
    return [found[key] for key in sorted(found) if key not in excluded]


def build_models(
    cfg: Any,
    inputs: list[str],
    output: str | Path,
    *,
    alpha: float = 1.0,
    granularity: str = "task",
    exclude: list[str] | None = None,
    regenerate: bool = False,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Group lifecycles by task key and write one model directory per group."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in find_lifecycles(inputs, exclude):
        record = read_lifecycle(path)
        groups[task_key(record["task"], record["tools"], granularity)].append(record)
    if not groups:
        raise ValueError("No guard_lifecycle.jsonl files matched the supplied inputs")
    index_path = output / INDEX_FILE
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index.setdefault("models", {})
    summary: dict[str, Any] = {}
    for key, records in sorted(groups.items()):
        model_dir = output / key
        model_dir.mkdir(exist_ok=True)
        abstraction_path = model_dir / ABSTRACTION_FILE
        first = records[0]
        task = first["task"] if granularity == "task" else ""
        # Generation sees every tool any run of this task could call.
        tools = sorted(
            {str(tool.get("name", "")): tool for record in records for tool in record["tools"]}.values(),
            key=lambda tool: str(tool.get("name", "")),
        )
        if abstraction_path.exists() and not regenerate:
            # Keep the predicates fixed when more traces are added later.
            stored = json.loads(abstraction_path.read_text(encoding="utf-8"))
            abstraction = PredicateAbstraction.from_dict(stored)
            generation = stored.get("generation") or {}
        else:
            abstraction, generation = generate_abstraction(cfg, task, tools, llm=llm)
            abstraction_path.write_text(json.dumps({
                "version": 2,
                "task_key": key,
                "granularity": granularity,
                "task": task,
                "tools": [str(tool.get("name", "")) for tool in tools],
                **abstraction.to_dict(),
                "generation": generation,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        logs = [trace_log(record["steps"]) for record in records]
        model = build_model(logs, abstraction, alpha)
        unsafe = sorted(model["state_index"][s] for s in abstraction.unsafe_states(model["states"]))
        model.update({
            "format": "agent_scaffold.pro2guard.probguard.v2",
            "task_key": key,
            "granularity": granularity,
            "alpha": alpha,
            "unsafe_state_indices": unsafe,
            "trace_count": len(records),
            "trace_sources": sorted(record["path"] for record in records),
        })
        (model_dir / MODEL_FILE).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        export_dtmc_to_prism(model, model_dir / PRISM_FILE)
        index["models"][key] = {
            "dir": key,
            "task_preview": task[:200],
            "trace_count": len(records),
            "state_count": len(model["states"]),
            "unsafe_state_count": len(unsafe),
        }
        summary[key] = index["models"][key]
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn per-task ProbGuard DTMCs from Hermes guard lifecycles.")
    parser.add_argument("traces", nargs="+", help="guard_lifecycle.jsonl files, globs, or directories searched recursively.")
    parser.add_argument("--config", required=True, help="Agent YAML whose llm / pro2guard.generator settings generate the spec.")
    parser.add_argument("--output", required=True, help="Model directory (pro2guard.model_dir).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing strength (upstream 1.0).")
    parser.add_argument("--granularity", choices=["task", "tools"], default=None,
                        help="One model per task (upstream, default) or per tool inventory.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Glob of runs to leave out, e.g. the evaluation split.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate existing abstractions.")
    args = parser.parse_args()

    from ..config import load_config

    cfg = load_config(args.config)
    summary = build_models(
        cfg, args.traces, args.output, alpha=args.alpha,
        granularity=args.granularity or cfg.pro2guard.granularity,
        exclude=args.exclude, regenerate=args.regenerate,
    )
    for key, entry in summary.items():
        print(f"{key}: {entry['trace_count']} traces, {entry['state_count']} states, "
              f"{entry['unsafe_state_count']} unsafe states")


if __name__ == "__main__":
    main()
