#!/usr/bin/env bash
set -uo pipefail

# Run every AgentHarm case selected by agents/hermes/agentharm.yaml without
# modifying that file. Each case gets its own run directory and evaluation.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${AGENTHARM_CONFIG:-${repo_root}/agents/hermes/agentharm.yaml}"
python_bin="${PYTHON_BIN:-python}"
runs_per_case="${1:-1}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_dir="${2:-${repo_root}/jobs/${stamp}_hermes-agentharm-all}"
resume="${RESUME:-1}"

usage() {
  cat <<EOF
Usage: $0 [runs-per-case] [output-directory]

Environment variables:
  AGENTHARM_CONFIG   Config to reuse (default: agents/hermes/agentharm.yaml)
  PYTHON_BIN         Python executable (default: python)
  RESUME             Skip runs with an existing evaluation.json (default: 1)
  AGENTHARM_CASE_IDS Optional comma-separated IDs, useful for a smoke test

Examples:
  $0
  $0 3 jobs/agentharm-all-3runs
  AGENTHARM_CASE_IDS=10-1,64-3 $0 1 jobs/agentharm-smoke
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if ! [[ "${runs_per_case}" =~ ^[1-9][0-9]*$ ]]; then
  echo "runs-per-case must be a positive integer" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${config}" ]]; then
  echo "Config does not exist: ${config}" >&2
  exit 2
fi

mkdir -p "${campaign_dir}/cases"
campaign_dir="$(cd "${campaign_dir}" && pwd)"
config="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
case_config="$(mktemp "$(dirname "${config}")/.agentharm-all.XXXXXX.yaml")"

write_summary() {
  PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${campaign_dir}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
ids_path = root / "case_ids.txt"
if not ids_path.exists():
    raise SystemExit(0)

case_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
runs_per_case = int((root / "runs_per_case.txt").read_text().strip())
rows = []
for behavior_id in case_ids:
    case_dir = root / "cases" / behavior_id
    for run_number in range(1, runs_per_case + 1):
        run_dir = case_dir / f"run_{run_number:03d}"
        evaluation_path = run_dir / "target" / "evaluation.json"
        error_path = run_dir / "error.json"
        row = {
            "behavior_id": behavior_id,
            "run": run_dir.name,
            "status": "not_started" if not run_dir.exists() else "missing_evaluation",
            "attack_success": "",
            "score": "",
            "full_score": "",
            "refusal": "",
            "security": "",
            "tool_call_count": "",
            "judge_error": "",
            "error": "",
        }
        if evaluation_path.exists():
            try:
                evaluation = json.loads(evaluation_path.read_text())
                refusal_explanation = str(evaluation.get("refusal_explanation", ""))
                row.update({
                    "status": "evaluated",
                    "attack_success": evaluation.get("attack_success", ""),
                    "score": evaluation.get("score", ""),
                    "full_score": evaluation.get("full_score", ""),
                    "refusal": evaluation.get("refusal", ""),
                    "security": evaluation.get("security", ""),
                    "tool_call_count": evaluation.get("tool_call_count", ""),
                    "judge_error": (
                        refusal_explanation
                        if "Error in RefusalJudgeLLM" in refusal_explanation
                        else ""
                    ),
                })
            except Exception as exc:
                row["status"] = "invalid_evaluation"
                row["error"] = str(exc)
        elif error_path.exists():
            row["status"] = "failed"
            try:
                error = json.loads(error_path.read_text())
                row["error"] = str(error.get("error", error))
            except Exception:
                row["error"] = error_path.read_text(errors="replace").strip()
        else:
            exit_path = run_dir / "exit_code.txt"
            if exit_path.exists():
                row["error"] = f"process exit code {exit_path.read_text().strip()}"
        rows.append(row)

columns = [
    "behavior_id", "run", "status", "attack_success", "score", "full_score",
    "refusal", "security", "tool_call_count", "judge_error", "error",
]
with (root / "summary.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

evaluated = [row for row in rows if row["status"] == "evaluated"]
successful = sorted({
    row["behavior_id"]
    for row in evaluated
    if row["attack_success"] is True
})
judge_errors = sum(bool(row["judge_error"]) for row in evaluated)
summary = {
    "case_count": len(case_ids),
    "runs_per_case": runs_per_case,
    "expected_runs": len(case_ids) * runs_per_case,
    "recorded_runs": len(rows),
    "evaluated_runs": len(evaluated),
    "attack_successful_runs": sum(row["attack_success"] is True for row in evaluated),
    "attack_successful_cases": len(successful),
    "judge_error_runs": judge_errors,
    "incomplete_or_failed_runs": len(rows) - len(evaluated),
    "successful_case_ids": successful,
    "note": (
        "Rows with judge_error are not valid refusal-judge results. "
        "Fix the judge API key before treating their refusal/full_score fields as official."
    ),
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
(root / "successful_cases.txt").write_text("".join(f"{item}\n" for item in successful))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

on_exit() {
  status=$?
  rm -f "${case_config}"
  write_summary || true
  echo "Campaign directory: ${campaign_dir}"
  echo "Summary: ${campaign_dir}/summary.json"
  echo "Successful cases: ${campaign_dir}/successful_cases.txt"
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mapfile -t case_ids < <(
  "${python_bin}" - "${config}" "${AGENTHARM_CASE_IDS:-}" <<'PY'
import pathlib
import sys
import yaml

config_path, requested = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(config_path).read_text())
cfg = raw.get("agentharm", {})
dataset = str(cfg.get("dataset", "harmful"))
split = str(cfg.get("split", "test_public"))

# IDs from the inspect-evals==0.17.0 revision pinned by this repository.
base_ids = {
    "val": [2, 3, 6, 7, 8, 9, 10, 11],
    "test_public": [
        1, 2, 7, 9, 10, 11, 13, 14, 17, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45,
        46, 49, 50, 52, 53, 54, 58, 61, 62, 63, 64, 66,
    ],
}
if split not in base_ids:
    raise SystemExit(
        f"Cannot enumerate split {split!r}; the pinned public dataset only lists val and test_public"
    )
if dataset not in {"harmful", "benign", "chat"}:
    raise SystemExit(f"Unsupported AgentHarm dataset: {dataset!r}")

if dataset == "chat":
    suffixes = [2]
else:
    suffixes = [1, 2, 3, 4]
    detailed = cfg.get("detailed_behaviors")
    hinted = cfg.get("hint_included")
    if detailed is True:
        suffixes = [item for item in suffixes if item in {1, 2}]
    elif detailed is False:
        suffixes = [item for item in suffixes if item in {3, 4}]
    if hinted is True:
        suffixes = [item for item in suffixes if item in {1, 3}]
    elif hinted is False:
        suffixes = [item for item in suffixes if item in {2, 4}]

available = [f"{base}-{suffix}" for base in base_ids[split] for suffix in suffixes]
if requested.strip():
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    missing = [item for item in selected if item not in available]
    if missing:
        raise SystemExit(f"Unknown case IDs for this dataset/split: {', '.join(missing)}")
    available = selected
for item in available:
    print(item)
PY
)

if (( ${#case_ids[@]} == 0 )); then
  echo "No AgentHarm cases matched the current configuration" >&2
  exit 1
fi
printf '%s\n' "${case_ids[@]}" > "${campaign_dir}/case_ids.txt"
printf '%s\n' "${runs_per_case}" > "${campaign_dir}/runs_per_case.txt"
cp "${config}" "${campaign_dir}/source-agentharm.yaml"

if [[ "${OPENAI_API_KEY:-}" == "dummy" || -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: OPENAI_API_KEY is unset or dummy." >&2
  echo "The local vLLM call may work, but the configured OpenAI refusal/semantic judges may return 401." >&2
fi

echo "Config: ${config}"
echo "Cases: ${#case_ids[@]}"
echo "Runs per case: ${runs_per_case}"
echo "Output: ${campaign_dir}"

for behavior_id in "${case_ids[@]}"; do
  PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${config}" "${case_config}" "${behavior_id}" <<'PY'
import pathlib
import sys
import yaml

source, destination, behavior_id = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(source).read_text())
raw.setdefault("agentharm", {})["behavior_id"] = behavior_id
pathlib.Path(destination).write_text(
    yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

  for ((run_number=1; run_number<=runs_per_case; run_number++)); do
    run_name="$(printf 'run_%03d' "${run_number}")"
    run_dir="${campaign_dir}/cases/${behavior_id}/${run_name}"
    evaluation="${run_dir}/target/evaluation.json"
    if [[ "${resume}" == "1" && -f "${evaluation}" ]]; then
      echo "=== ${behavior_id} ${run_name}: already evaluated, skipping ==="
      continue
    fi

    mkdir -p "${run_dir}"
    echo "=== ${behavior_id} ${run_name} ==="
    AGENT_JOB_DIR="${run_dir}" \
    AGENT_BATCH_DIR="${campaign_dir}" \
    PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" "${repo_root}/src/agent_scaffold/main.py" \
        --config "${case_config}" 2>&1 | tee "${run_dir}/console.log"
    run_status=${PIPESTATUS[0]}
    printf '%s\n' "${run_status}" > "${run_dir}/exit_code.txt"
    if (( run_status != 0 )); then
      echo "FAILED: ${behavior_id} ${run_name} (exit ${run_status}); continuing" >&2
    fi
  done
done

write_summary
