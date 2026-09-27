"""Build run-level alarm labels and confusion matrices for Hermes experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from hermes_monitor_rules import METHODS, label


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--batch", help="Only analyze this batch")
args = parser.parse_args()
ROOT = args.root.resolve()
OUT_DIR = ROOT / "analysis"
rows: list[dict] = []
skipped: list[dict] = []
for summary_path in sorted(ROOT.glob("*/*/*/summary.json")):
    if args.batch and summary_path.parent.name != args.batch:
        continue
    method_dir, condition = summary_path.relative_to(ROOT).parts[:2]
    summary = json.loads(summary_path.read_text())
    for item in sorted(summary["items"], key=lambda value: value["index"]):
        run_dir = summary_path.parent / f"run_{item['index']:03d}"
        result_path = run_dir / "result.json"
        defense_path = run_dir / "target" / "defenses.json"
        try:
            result = json.loads(result_path.read_text())
            defense = json.loads(defense_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            skipped.append(
                {
                    "experiment": method_dir,
                    "batch": summary_path.parent.name,
                    "condition": condition,
                    "run": item["index"],
                    "method": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        evaluation = result.get("harness", {}).get("agentdojo", {})
        attack_success = evaluation.get("attack_success")
        utility = evaluation.get("utility")
        if defense.get("mode") == "replay":
            methods = [name for name in defense.get("methods", {}) if name in METHODS]
        else:
            failed = (defense.get("harness") or {}).get("guard_errors") or {}
            methods = (
                [method_dir]
                if method_dir in METHODS
                else [
                    name
                    for name in METHODS
                    if name in defense.get("enabled", []) or name in failed
                ]
            )
        for method in methods:
            method_defense, replay_error = defense, ""
            if defense.get("mode") == "replay":
                status = defense.get("methods", {}).get(method, {})
                if status.get("status") != "completed":
                    replay_error = str(
                        status.get("error") or status.get("status") or "missing replay status"
                    )
                method_path = (
                    run_dir / "target" / "defense_replay" / method / "defenses.json"
                )
                try:
                    method_defense = json.loads(method_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError) as exc:
                    # A missing replay is an errored run for this method.
                    method_defense = None
                    replay_error = replay_error or f"{type(exc).__name__}: {exc}"
            run_label = label(method, method_defense, replay_error)
            rows.append(
                {
                    "experiment": method_dir,
                    "batch": summary_path.parent.name,
                    "method": method,
                    "condition": condition,
                    "run": item["index"],
                    "injection": int(condition == "skill_injection"),
                    "status": run_label["status"],
                    "alarm": int(run_label["alarm"]),
                    "attack_success": "" if attack_success is None else int(attack_success),
                    "utility": "" if utility is None else int(utility),
                    "monitor_error": int(run_label["monitor_error"]),
                    "error": run_label["error"],
                    "decision_records": run_label["decision_records"],
                    "result_path": str((run_dir / "result.json").relative_to(ROOT)),
                }
            )

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "skipped_runs.json").write_text(json.dumps(skipped, indent=2) + "\n")
with (OUT_DIR / "run_labels.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=[
        "experiment", "batch", "method", "condition", "run", "injection",
        "status", "alarm", "attack_success", "utility", "monitor_error",
        "error", "decision_records", "result_path",
    ])
    writer.writeheader()
    writer.writerows(rows)


def matrix(samples: list[dict], field: str) -> list[list[int]]:
    counts = Counter((sample[field], sample["alarm"]) for sample in samples)
    return [[counts[(1, 1)], counts[(1, 0)]], [counts[(0, 1)], counts[(0, 0)]]]


summary_rows = []
groups = sorted(
    {(row["experiment"], row["batch"], row["method"]) for row in rows}
)
for experiment, batch, method in groups:
    on = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["batch"] == batch
        and row["method"] == method
        and row["injection"] == 1
    ]
    off = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["batch"] == batch
        and row["method"] == method
        and row["injection"] == 0
    ]
    # Errored runs (monitor error and no valid alarm) are not negatives.
    valid_on = [row for row in on if not row["monitor_error"]]
    valid_off = [row for row in off if not row["monitor_error"]]
    evaluated = [row for row in valid_on if row["attack_success"] != ""]
    summary_rows.append(
        {
            "experiment": experiment,
            "batch": batch,
            "method": method,
            "injection_matrix": matrix(valid_on + valid_off, "injection"),
            "attack_matrix": matrix(evaluated, "attack_success"),
            "unknown": len(valid_on) - len(evaluated),
            "errors_on": sum(row["monitor_error"] for row in on),
            "errors_off": sum(row["monitor_error"] for row in off),
            "samples_on": len(valid_on),
            "samples_off": len(valid_off),
            "utility_on": sum(row["utility"] == 1 for row in on),
            "utility_off": sum(row["utility"] == 1 for row in off),
        }
    )

rendered = json.dumps(summary_rows, indent=2)
(OUT_DIR / "confusion_matrices.json").write_text(rendered + "\n")
print(rendered)
