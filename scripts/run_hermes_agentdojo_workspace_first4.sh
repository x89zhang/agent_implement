#!/usr/bin/env bash
set -euo pipefail

# Run workspace user_task_0 through user_task_3 against every injection task
# twice, using the other settings from agents/hermes/agentdojo.yaml.
# Pass an output directory to resume an interrupted sweep.
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
batch_dir="${1:-${repo_root}/jobs/workspace_first4_sweep/${stamp}_hermes-agentdojo}"
mkdir -p "$batch_dir"
batch_dir="$(cd "$batch_dir" && pwd)"
case_list="${batch_dir}/cases.txt"
temp_config="$(mktemp "${repo_root}/agents/hermes/.workspace-first4-sweep.XXXXXX.yaml")"
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

suite = get_suite(str(dojo.get("benchmark_version", "v1.2.2")), "workspace")
number = lambda task_id: int(task_id.rsplit("_", 1)[1])
cases = [
    f"{user_task}_injection_{number(injection_task)}"
    for user_task in sorted(suite.user_tasks, key=number)[:4]
    for injection_task in sorted(suite.injection_tasks, key=number)
]
Path(sys.argv[2]).write_text("\n".join(cases) + "\n", encoding="utf-8")
print(f"Workspace first-four pairs: {len(cases)}; planned runs: {len(cases) * 2}")
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
raise SystemExit(0 if len(items) == 2 and all(item.get("ok") for item in items) else 1)
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
config["agentdojo"]["suite"] = "workspace"
config["agentdojo"]["case"] = sys.argv[3]
config["agentdojo"].pop("user_task", None)
config["agentdojo"].pop("injection_task", None)
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
  mkdir -p "$case_dir"
  echo "Running $case_id (2 trials)"
  if ! PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
      --config "$temp_config" --runs 2 --runs-dir "$case_dir" \
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
    if len(items) != 2 or any(not item.get("ok") for item in items):
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
report = {
    "pairs": len(cases),
    "planned_runs": len(cases) * 2,
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
