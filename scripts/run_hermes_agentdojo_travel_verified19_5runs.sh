#!/usr/bin/env bash
set -euo pipefail

# Re-run the 19 travel cases whose full attack goals were verified in the
# earlier sweep. Run each case five times with agents/hermes/agentdojo.yaml.
# The generated run configs select travel even when the source YAML selects
# another suite. Pass an output directory to resume an interrupted sweep.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [output-directory]"
  echo "Set PYTHON_BIN to a Python environment with agentdojo and PyYAML installed."
  exit 0
fi
if (( $# > 1 )); then
  echo "Usage: $0 [output-directory]" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x "${repo_root}/../envs/agent/bin/python" ]]; then
    python_bin="${repo_root}/../envs/agent/bin/python"
  else
    python_bin="python3"
  fi
fi

source_config="${repo_root}/agents/hermes/agentdojo.yaml"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${1:-}" ]]; then
  batch_dir="$1"
  mkdir -p "$batch_dir"
else
  mkdir -p "${repo_root}/jobs/travel_verified19_sweep"
  batch_dir="$(mktemp -d "${repo_root}/jobs/travel_verified19_sweep/${stamp}_hermes-agentdojo.XXXXXX")"
fi
batch_dir="$(cd "$batch_dir" && pwd)"
case_list="${batch_dir}/cases.txt"
temp_config="$(mktemp "${repo_root}/agents/hermes/.travel-verified19.XXXXXX.yaml")"
trap 'rm -f "$temp_config"' EXIT

"$python_bin" - "$source_config" "$case_list" <<'PY'
import sys
from pathlib import Path

import yaml
from agentdojo.task_suite.load_suites import get_suite

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
dojo = config.get("agentdojo") or {}
if not dojo.get("enabled"):
    raise SystemExit("Expected AgentDojo to be enabled in agents/hermes/agentdojo.yaml")
if not dojo.get("injection_enabled") or not (
    dojo.get("standard_injection_enabled", True) or dojo.get("skill_injection_enabled", False)
):
    raise SystemExit("Enable an injection carrier in agents/hermes/agentdojo.yaml before the sweep")
if config.get("execution", {}).get("backend") != "hermes":
    raise SystemExit("Expected the Hermes execution backend")

suite = get_suite(str(dojo.get("benchmark_version", "v1.2.2")), "travel")
cases = [
    "user_task_0_injection_2",
    "user_task_2_injection_2",
    "user_task_3_injection_2",
    "user_task_5_injection_2",
    "user_task_6_injection_2",
    "user_task_9_injection_2",
    "user_task_10_injection_2",
    "user_task_11_injection_2",
    "user_task_12_injection_2",
    "user_task_13_injection_2",
    "user_task_14_injection_2",
    "user_task_15_injection_2",
    "user_task_16_injection_2",
    "user_task_17_injection_2",
    "user_task_18_injection_2",
    "user_task_19_injection_2",
    "user_task_16_injection_4",
    "user_task_17_injection_4",
    "user_task_19_injection_4",
]
assert len(cases) == len(set(cases)) == 19
for case in cases:
    user_task, injection_task = case.split("_injection_")
    if user_task not in suite.user_tasks or f"injection_task_{injection_task}" not in suite.injection_tasks:
        raise SystemExit(f"Case is unavailable in this AgentDojo version: {case}")
case_text = "\n".join(cases) + "\n"
case_path = Path(sys.argv[2])
if case_path.exists() and case_path.read_text(encoding="utf-8") != case_text:
    raise SystemExit(f"Existing case list differs: {case_path}")
case_path.write_text(case_text, encoding="utf-8")
print(f"Travel cases: {len(cases)}; planned runs: {len(cases) * 5}")
PY

if [[ -e "${batch_dir}/source_config.yaml" ]] && \
    ! cmp -s "$source_config" "${batch_dir}/source_config.yaml"; then
  echo "Source config changed since this sweep started; use a new output directory" >&2
  exit 1
fi
cp "$source_config" "${batch_dir}/source_config.yaml"
echo "Output: $batch_dir"

while IFS= read -r case_id; do
  case_dir="${batch_dir}/${case_id}"
  if [[ -e "${case_dir}/summary.json" ]]; then
    if "$python_bin" - "${case_dir}/summary.json" <<'PY'
import json
import sys

items = json.load(open(sys.argv[1], encoding="utf-8")).get("items", [])
raise SystemExit(0 if len(items) == 5 and all(item.get("ok") for item in items) else 1)
PY
    then
      echo "Skipping completed $case_id"
      continue
    fi
  fi
  if [[ -e "$case_dir" ]]; then
    old_dir="${case_dir}.incomplete.${stamp}"
    if [[ -e "$old_dir" ]]; then
      echo "Cannot preserve incomplete batch: $old_dir already exists" >&2
      exit 1
    fi
    mv "$case_dir" "$old_dir"
    echo "Preserved incomplete batch at $old_dir"
  fi

  "$python_bin" - "${batch_dir}/source_config.yaml" "$temp_config" "$case_id" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
config["agentdojo"]["suite"] = "travel"
config["agentdojo"]["case"] = sys.argv[3]
config["agentdojo"].pop("user_task", None)
config["agentdojo"].pop("injection_task", None)
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
  mkdir -p "$case_dir"
  echo "Running $case_id (5 trials)"
  if ! PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
      --config "$temp_config" --runs 5 --runs-dir "$case_dir" \
      >"${case_dir}/console.log" 2>&1; then
    echo "Runner failed for $case_id; see ${case_dir}/console.log" >&2
  fi
done < "$case_list"

"$python_bin" - "$batch_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = (root / "cases.txt").read_text(encoding="utf-8").splitlines()
rows = []
incomplete = []
for case_id in cases:
    path = root / case_id / "summary.json"
    if not path.exists():
        incomplete.append(case_id)
        continue
    items = json.loads(path.read_text(encoding="utf-8")).get("items", [])
    if len(items) != 5 or any(not item.get("ok") for item in items):
        incomplete.append(case_id)
    for item in items:
        evaluation = (item.get("harness") or {}).get("agentdojo") or {}
        rows.append({
            "case": case_id,
            "run": item.get("index", ""),
            "ok": item.get("ok", False),
            "attack_success": evaluation.get("attack_success", ""),
            "utility": evaluation.get("utility", ""),
            "run_dir": item.get("run_dir", ""),
        })

with (root / "results.csv").open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=[
        "case", "run", "ok", "attack_success", "utility", "run_dir"
    ])
    writer.writeheader()
    writer.writerows(rows)

successes = [row for row in rows if row["attack_success"] is True]
case_asr = {}
for case_id in cases:
    evaluated = [
        row for row in rows
        if row["case"] == case_id and isinstance(row["attack_success"], bool)
    ]
    count = sum(row["attack_success"] is True for row in evaluated)
    case_asr[case_id] = {
        "evaluated_runs": len(evaluated),
        "attack_successes": count,
        "asr": count / len(evaluated) if evaluated else None,
    }
report = {
    "pairs": len(cases),
    "case_asr": case_asr,
    "planned_runs": len(cases) * 5,
    "evaluated_runs": sum(isinstance(row["attack_success"], bool) for row in rows),
    "successful_runs": len(successes),
    "successful_cases": sorted({row["case"] for row in successes}),
    "successful_trials": successes,
    "incomplete_cases": incomplete,
}
(root / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"Successful cases: {len(report['successful_cases'])}; successful runs: {len(successes)}")
for case_id in report["successful_cases"]:
    print(f"  {case_id}")
print(f"Results: {root / 'results.csv'}")
print(f"Report: {root / 'report.json'}")
if incomplete:
    raise SystemExit(f"Incomplete cases: {len(incomplete)}; see report.json")
PY
