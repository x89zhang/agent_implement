#!/usr/bin/env bash
set -euo pipefail

# Re-run the three academic memory cases whose ASR exceeded clean control:
# case_00023 CitationTrackingSoftware, case_00031 DatabaseManagementSystem,
# case_00038 CredentialVerificationTool. Five target/control pairs per case.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_hermes_asb_memory_improved3_5runs.sh [output-directory]

Runs 3 cases x 5 trials, each with an attacked target and clean control.
All defense plugins are disabled. Uses the original academic r1 source config
(gpt-5.6-luna, official_asb memory). No model override is applied.

Environment:
  PYTHON_BIN   Python with PyYAML installed (default: python)
  DRY_RUN      Enumerate cases without model calls (default: 0)
  RESUME       Skip complete target/control pairs (default: 1)

Outputs: results.tsv, summary.json, asr_comparison.tsv, asr_comparison.json.
The comparison reports original case IDs alongside newly enumerated case IDs.
Pass the same output directory to resume. Missing/failed pairs are rerun.
USAGE
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
resume="${RESUME:-1}"
for flag in "${dry_run}" "${resume}"; do
  if [[ "${flag}" != 0 && "${flag}" != 1 ]]; then
    echo "DRY_RUN and RESUME must be 0 or 1" >&2
    exit 2
  fi
done
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_dir="${1:-${repo_root}/jobs/${stamp}_hermes-asb-memory-improved3-5runs}"
config="${repo_root}/jobs/asb-memory-academic-quantum-40-r1/source-asb.yaml"

# Validate the baseline before starting any trials. The shared runner applies
# the same replacement to each generated case config.
PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - "${config}" "${repo_root}/scripts" <<'PY'
import sys
import tempfile
from pathlib import Path
import yaml
from agent_scaffold.config import load_config
sys.path.insert(0, sys.argv[2])
from hermes_monitor_rules import METHODS
raw = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in (*METHODS, "agentsight"):
    raw[name] = {"enabled": False}
raw.setdefault("memory_experiment", {}).update(
    mode="official_asb", run_clean_control=True
)
with tempfile.TemporaryDirectory(prefix="asb-baseline-config-") as directory:
    path = Path(directory) / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_config(str(path))
print("Baseline config validated: all defense plugins disabled")
PY

# The shared runner only checks target evaluation when resuming. Preserve and
# rerun any pair missing a valid boolean evaluation in either phase.
if [[ "${resume}" == 1 && "${dry_run}" == 0 && -d "${campaign_dir}/cases" ]]; then
  "${python_bin}" - "${campaign_dir}" "${stamp}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for case_id in ("case_00001", "case_00002", "case_00003"):
    for index in range(1, 6):
        run = root / "cases" / case_id / f"run_{index:03d}"
        if not run.is_dir():
            continue
        complete = True
        for phase in ("target", "control"):
            try:
                data = json.loads((run / phase / "evaluation.json").read_text())
                complete &= isinstance(data.get("attack_success"), bool)
            except (OSError, ValueError):
                complete = False
        if not complete:
            archive = run.with_name(f"{run.name}.incomplete_{sys.argv[2]}")
            suffix = 1
            while archive.exists():
                archive = run.with_name(f"{run.name}.incomplete_{sys.argv[2]}_{suffix}")
                suffix += 1
            run.rename(archive)
            print(f"Preserved incomplete pair: {archive}")
PY
fi

status=0
ASB_CONFIG="${config}" PYTHON_BIN="${python_bin}" \
  RUNS_PER_CASE=5 KEEP_DEFENSES=0 RUN_CLEAN_CONTROL=1 \
  ASB_ATTACK_TYPES=naive ASB_AGENTS=academic_search_agent ASB_TASK_INDEXES=0 \
  ASB_ATTACKER_TOOLS=CitationTrackingSoftware,DatabaseManagementSystem,CredentialVerificationTool \
  ASB_AGGRESSIVE=false ASB_CASE_LIMIT=0 DRY_RUN="${dry_run}" RESUME="${resume}" \
  bash "${repo_root}/scripts/run_hermes_asb_memory_all.sh" "${campaign_dir}" || status=$?
if (( status != 0 )); then
  exit "${status}"
fi

"${python_bin}" - "${campaign_dir}" "${dry_run}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
original_ids = {
    "CitationTrackingSoftware": "case_00023",
    "DatabaseManagementSystem": "case_00031",
    "CredentialVerificationTool": "case_00038",
}
with (root / "cases.tsv").open(newline="") as handle:
    cases = list(csv.DictReader(handle, delimiter="\t"))
if len(cases) != 3 or {c["attacker_tool"] for c in cases} != set(original_ids):
    raise SystemExit("Expected exactly the three selected academic cases")
with (root / "results.tsv").open(newline="") as handle:
    results = list(csv.DictReader(handle, delimiter="\t"))
rows = []
for case in cases:
    trials = [r for r in results if r["case_id"] == case["case_id"]]
    target = [r for r in trials if r["attack_success"] in ("True", "False")]
    control = [r for r in trials if r["control_attack_success"] in ("True", "False")]
    paired = [r for r in target if r["control_attack_success"] in ("True", "False")]
    target_successes = sum(r["attack_success"] == "True" for r in target)
    control_successes = sum(r["control_attack_success"] == "True" for r in control)
    target_asr = target_successes / len(target) if target else None
    control_asr = control_successes / len(control) if control else None
    # Compare on the same trials when either phase has missing evaluations.
    delta = sum((r["attack_success"] == "True") -
                (r["control_attack_success"] == "True") for r in paired) / len(paired) if paired else None
    rows.append({
        "original_case_id": original_ids[case["attacker_tool"]],
        "case_id": case["case_id"],
        "attacker_tool": case["attacker_tool"],
        "planned_pairs": 5,
        "target_evaluated_runs": len(target),
        "target_successful_runs": target_successes,
        "target_asr": target_asr,
        "control_evaluated_runs": len(control),
        "control_successful_runs": control_successes,
        "control_asr": control_asr,
        "paired_evaluated_runs": len(paired),
        "paired_asr_delta": delta,
        "paired_asr_delta_pp": delta * 100 if delta is not None else None,
        "complete": len(paired) == 5,
    })
with (root / "asr_comparison.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
(root / "asr_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"Comparison: {root / 'asr_comparison.tsv'}")
for row in rows:
    print(f"{row['original_case_id']} {row['attacker_tool']}: "
          f"target={row['target_successful_runs']}/{row['target_evaluated_runs']}, "
          f"control={row['control_successful_runs']}/{row['control_evaluated_runs']}, "
          f"delta_pp={row['paired_asr_delta_pp']}, complete={row['complete']}")
if sys.argv[2] != "1" and any(not row["complete"] for row in rows):
    raise SystemExit("Incomplete target/control pairs; resume with the same output directory")
PY
