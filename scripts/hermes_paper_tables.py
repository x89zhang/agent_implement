#!/usr/bin/env python3
"""Pool Hermes AgentDojo monitor labels over task pairs into the paper tables.

Table 1, injection vs warning: injected runs from batches that also have clean
runs, against those clean runs. Table 2, attack success vs warning: injected
runs from every batch, successful against failed attacks.

Each AgentDojo pair or ASB group contributes its first N runs per class (by
batch name, then run index), so groups with 20-run batches do not outweigh the
others; --all-runs uses every run. In ASB (--asb) each run's poisoned target
phase is the injected run and its unpoisoned control phase the clean one;
MCP-SafetyBench (--mcp-safety-bench) batches have the same two phases.
Runs where a monitor errored are left out of that monitor's counts and
reported. Labels come from each pair's or batch's analysis/run_labels.csv,
which this script regenerates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from hermes_monitor_rules import METHODS

REPO = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def pair_root(suite: str, case: str) -> Path:
    name = f"agentdojo_{case}" if suite == "slack" else f"agentdojo_{suite}_{case}"
    return REPO / "jobs" / name / "Hermes/gpt"


def read_pairs(paths: list[Path]) -> list[tuple[str, str]]:
    pairs = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and not fields[0].startswith("#"):
                pairs.append((fields[0], fields[1]))
    return list(dict.fromkeys(pairs))


def labels(root: Path) -> list[dict]:
    subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_hermes_agentdojo_monitors.py"), "--root", str(root)],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL,
    )
    with (root / "analysis/run_labels.csv").open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["experiment"] == "all_monitors"]


def first(runs: list[tuple[str, int]], limit: int | None) -> set[tuple[str, int]]:
    return set(sorted(runs)[:limit] if limit else runs)


# Paper names and order for --compact.
DISPLAY = {
    "aegis": "Aegis", "progent": "Progent", "drift": "DRIFT", "agrail": "AGrail",
    "rope": "RoPE", "melon": "MELON", "airguard": "AIRGuard", "clawsentry": "ClawSentry",
    "janus": "Janus", "toolsafe": "ToolSafe", "stepguard": "StepGuard", "piguard": "PIGuard",
}


def compact_table(title: str, positive: str, negative: str, counts: dict, expected: dict) -> str:
    lines = [
        f"### {title}",
        "",
        f"Positive: {positive}. Negative: {negative}.",
        "",
        "| Method | TP | FN | FP | TN | TPR | FPR | F1 | F1₅₀ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for method, name in DISPLAY.items():
        if method not in counts:
            continue
        c = counts[method]
        neg = c["fp"] + c["tn"]
        f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else math.nan
        fpr = c["fp"] / neg if neg else math.nan
        pos = c["tp"] + c["fn"]
        tpr = c["tp"] / pos if pos else math.nan
        # F1 at equal numbers of positives and negatives (calibrated F1), so it
        # does not depend on the attack success rate: 2·TPR / (1 + TPR + FPR).
        f1_50 = 2 * tpr / (1 + tpr + fpr) if pos and neg else math.nan
        lines.append(
            f"| {name} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} "
            f"| {tpr:.1%} | {fpr:.1%} | {f1:.1%} | {f1_50:.1%} |"
        )
        missing = expected["pos"] + expected["neg"] - c["tp"] - c["fn"] - neg - c["error"]
        if c["error"] or missing:
            parts = [f"{c['error']} monitor error{'s' * (c['error'] != 1)}"] if c["error"] else []
            parts += [f"{missing} run{'s' * (missing != 1)} not yet replayed"] if missing else []
            notes.append(f"{name}: {', '.join(parts)}")
    if notes:
        lines += ["", "Left out of the counts: " + "; ".join(notes) + "."]
    no_tools = {c["no_tools"] for c in counts.values() if c["no_tools"]}
    if no_tools:
        n = max(no_tools)
        lines += ["", f"{n} run{'s' * (n != 1)} without tool calls count as no warning for the "
                  "monitors that had no tool call to check."]
    return "\n".join(lines)


def table(title: str, positive: str, negative: str, counts: dict, expected: dict) -> str:
    lines = [
        f"### {title}",
        "",
        f"Positive: {positive}. Negative: {negative}. 95% Wilson intervals.",
        "",
        "| Monitor | TP | FN | FP | TN | Errors | TPR | FPR | J = TPR − FPR | Balanced acc. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        if method not in counts:
            continue
        c = counts[method]
        pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
        missing = expected["pos"] + expected["neg"] - pos - neg - c["error"]
        tpr = c["tp"] / pos if pos else math.nan
        fpr = c["fp"] / neg if neg else math.nan
        tlo, thi = wilson(c["tp"], pos)
        flo, fhi = wilson(c["fp"], neg)
        errors = f"{c['error']}" + (f" (+{missing} not replayed)" if missing else "")
        lines.append(
            f"| {method} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} | {errors} "
            f"| {tpr:.1%} [{tlo:.0%}, {thi:.0%}] | {fpr:.1%} [{flo:.0%}, {fhi:.0%}] "
            f"| {tpr - fpr:+.2f} | {(tpr + 1 - fpr) / 2:.1%} |"
        )
    return "\n".join(lines)


def asb_labels(batch: Path) -> list[dict]:
    subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_hermes_asb_monitors.py"), str(batch)],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL,
    )
    with (batch / "analysis/run_labels.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def agentdojo_runs(rows: list[dict]) -> list[dict]:
    return [
        {
            "batch": row["batch"], "run": int(row["run"]), "method": row["method"],
            "injected": row["injection"] == "1", "attack_success": row["attack_success"] == "1",
            # Only batches with clean runs pair their injected runs with clean ones.
            "paired": row["batch"] in {r["batch"] for r in rows if r["injection"] == "0"},
            "error": row["monitor_error"] == "1", "alarm": row["alarm"] == "1",
        }
        for row in rows
    ]


def asb_runs(batch: Path, rows: list[dict]) -> list[dict]:
    # Each ASB run has a poisoned target phase and an unpoisoned control phase.
    runs = [
        {
            "batch": batch.name, "run": int(row["run"]), "method": row["method"],
            "injected": row["phase"] == "target", "attack_success": row["attack_success"] == "True",
            "paired": True, "error": row["monitor_error"] == "True", "alarm": row["alarm"] == "True",
            "no_tools": False,
        }
        for row in rows
    ]
    # A phase where the agent called no tools gives tool-call monitors nothing to
    # check, so the replay records no label for them; deployed, they would not
    # have warned, so they count as no warning (and are reported).
    methods = {r["method"] for r in runs}
    for (run, injected), phase_rows in _by_phase(runs).items():
        evaluation = batch / f"run_{run:03d}" / ("target" if injected else "control") / "evaluation.json"
        if not evaluation.is_file() or json.loads(evaluation.read_text(encoding="utf-8")).get("tool_call_count") != 0:
            continue
        for method in sorted(methods - {r["method"] for r in phase_rows}):
            runs.append({**phase_rows[0], "method": method, "error": False, "alarm": False, "no_tools": True})
    return runs


def _by_phase(runs: list[dict]) -> dict[tuple[int, bool], list[dict]]:
    phases: dict[tuple[int, bool], list[dict]] = defaultdict(list)
    for r in runs:
        phases[(r["run"], r["injected"])].append(r)
    return phases


class Tables:
    def __init__(self) -> None:
        self.injection = defaultdict(lambda: defaultdict(int))
        self.attack = defaultdict(lambda: defaultdict(int))
        self.expected_injection = {"pos": 0, "neg": 0}
        self.expected_attack = {"pos": 0, "neg": 0}
        self.groups: list[str] = []

    def add(
        self, name: str, runs: list[dict], injection_limit: int | None, attack_limit: int | None,
        attack_table: bool = True,
    ) -> None:
        injected = {(r["batch"], r["run"]): r for r in runs if r["injected"]}
        clean = {(r["batch"], r["run"]) for r in runs if not r["injected"]}
        table1_on = first([key for key, r in injected.items() if r["paired"]], injection_limit)
        table1_off = first(list(clean), injection_limit)
        # Groups whose attacks (nearly) always succeed have no failed attacks to
        # compare against, so they can be kept out of the attack-success table.
        table2 = first(list(injected), attack_limit) if attack_table else set()
        success = {key for key in table2 if injected[key]["attack_success"]}
        self.expected_injection["pos"] += len(table1_on)
        self.expected_injection["neg"] += len(table1_off)
        self.expected_attack["pos"] += len(success)
        self.expected_attack["neg"] += len(table2) - len(success)
        attack_cell = f"{len(success)} / {len(table2)}" if attack_table else "- (injection table only)"
        self.groups.append(f"| {name} | {len(table1_on)} / {len(table1_off)} | {attack_cell} |")
        for r in runs:
            key = (r["batch"], r["run"])
            for selected, counts, positive in (
                (table1_on if r["injected"] else table1_off, self.injection, r["injected"]),
                (table2 if r["injected"] else set(), self.attack, key in success),
            ):
                if key not in selected:
                    continue
                c = counts[r["method"]]
                c["no_tools"] += r.get("no_tools", False)
                if r["error"]:
                    c["error"] += 1
                elif r["alarm"]:
                    c["tp" if positive else "fp"] += 1
                else:
                    c["fn" if positive else "tn"] += 1

    def print(self, benchmark: str, clean: str, compact: bool = False) -> None:
        render = compact_table if compact else table
        print(f"## {benchmark}")
        print()
        print(f"| Group | Table 1 injected / {clean} | Table 2 successes / injected |")
        print("|---|---|---|")
        print("\n".join(self.groups))
        print()
        # Attack success first, as in the paper.
        print(render("Attack success vs warning", "successful attack", "failed attack", self.attack, self.expected_attack))
        print()
        print(render("Injection vs warning", "injected run", f"{clean} run", self.injection, self.expected_injection))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", type=Path, nargs="*", help="files of AgentDojo 'suite case' lines")
    parser.add_argument("--asb", type=Path, action="append", default=[], help="file of ASB batch directories")
    parser.add_argument(
        "--asb-injection-only", type=Path, action="append", default=[],
        help="file of ASB batch directories whose groups enter only the injection table",
    )
    parser.add_argument(
        "--mcp-safety-bench", type=Path, action="append", default=[],
        help="file of MCP-SafetyBench batch directories (target/control phases, like ASB)",
    )
    parser.add_argument("--injection-runs", type=int, default=5, help="injected and clean runs per group (table 1)")
    parser.add_argument("--attack-runs", type=int, default=10, help="injected runs per group (table 2)")
    parser.add_argument("--all-runs", action="store_true", help="use every run of every group")
    parser.add_argument("--compact", action="store_true", help="TP/FN/FP/TN/F1/FPR tables with paper names")
    args = parser.parse_args()
    if not args.pairs and not args.asb and not args.asb_injection_only and not args.mcp_safety_bench:
        parser.error("give AgentDojo pairs files, --asb and/or --mcp-safety-bench")

    injection_limit = None if args.all_runs else args.injection_runs
    attack_limit = None if args.all_runs else args.attack_runs
    if args.pairs:
        agentdojo = Tables()
        for suite, case in read_pairs(args.pairs):
            runs = agentdojo_runs(labels(pair_root(suite, case)))
            agentdojo.add(f"{suite} {case}", runs, injection_limit, attack_limit)
        agentdojo.print("AgentDojo", "clean", args.compact)
    if args.asb or args.asb_injection_only:
        phase_tables(
            "ASB", [(p, False) for p in args.asb] + [(p, True) for p in args.asb_injection_only],
            injection_limit, attack_limit, args.compact,
        )
    if args.mcp_safety_bench:
        phase_tables(
            "MCP-SafetyBench", [(p, False) for p in args.mcp_safety_bench],
            injection_limit, attack_limit, args.compact,
        )
    return 0


def phase_tables(
    benchmark: str, files: list[tuple[Path, bool]],
    injection_limit: int | None, attack_limit: int | None, compact: bool,
) -> None:
    """Tables for benchmarks whose runs have a poisoned target and a clean control phase."""
    tables = Tables()
    # Batches of the same group (ASB: agent, task, attacker tool, attack type;
    # MCP-SafetyBench: task) are pooled into one group, like an AgentDojo pair's batches.
    groups: dict[str, list[dict]] = defaultdict(list)
    injection_only: set[str] = set()
    for path, only in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            batch = (REPO / line.strip()).resolve()
            groups[batch.parent.name] += asb_runs(batch, asb_labels(batch))
            if only:
                injection_only.add(batch.parent.name)
    for name, runs in groups.items():
        tables.add(name, runs, injection_limit, attack_limit, attack_table=name not in injection_only)
    tables.print(benchmark, "control", compact)

if __name__ == "__main__":
    raise SystemExit(main())
