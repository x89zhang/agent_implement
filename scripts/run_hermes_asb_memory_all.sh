#!/usr/bin/env bash
set -uo pipefail

# Run every official ASB memory-poisoning combination with Hermes through the
# pinned ASB Chroma database and official-memory bridge.
# A case is: agent_name x task_index x attacker_tool x attack_type.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${ASB_CONFIG:-${repo_root}/agents/hermes/asb.yaml}"
python_bin="${PYTHON_BIN:-python}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_dir="${1:-${repo_root}/jobs/${stamp}_hermes-asb-memory-all}"
resume="${RESUME:-1}"
dry_run="${DRY_RUN:-0}"
keep_defenses="${KEEP_DEFENSES:-0}"
run_clean_control="${RUN_CLEAN_CONTROL:-0}"
attack_types="${ASB_ATTACK_TYPES:-naive,fake_completion,escape_characters,context_ignoring,combined_attack}"
agent_filter="${ASB_AGENTS:-}"
task_filter="${ASB_TASK_INDEXES:-}"
tool_filter="${ASB_ATTACKER_TOOLS:-}"
aggressive_filter="${ASB_AGGRESSIVE:-all}"
case_limit="${ASB_CASE_LIMIT:-0}"
runs_per_case="${RUNS_PER_CASE:-1}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_hermes_asb_memory_all.sh [output-directory]

Runs each selected ASB memory-poisoning case one or more times and writes a live campaign
under jobs/. The complete official cross product is:
  agent_name x task_index x attacker_tool x attack_type

Environment variables:
  ASB_CONFIG          Source config (default: agents/hermes/asb.yaml)
  PYTHON_BIN          Python executable (default: python)
  RESUME              Skip cases with target/evaluation.json (default: 1)
  DRY_RUN             Only enumerate cases; do not run them (default: 0)
  KEEP_DEFENSES        Keep defenses from ASB_CONFIG (default: 0)
  RUN_CLEAN_CONTROL    Also run clean-control phase (default: 0)
  ASB_ATTACK_TYPES     Comma-separated templates (default: all five)
  ASB_AGENTS           Optional comma-separated agent names
  ASB_TASK_INDEXES     Optional comma-separated task indexes
  ASB_ATTACKER_TOOLS   Optional comma-separated attacker tool names
  ASB_AGGRESSIVE       all, true, or false (default: all)
  ASB_CASE_LIMIT       Stop enumeration after N cases; 0 means unlimited
  RUNS_PER_CASE        Repetitions for every case (default: 1)

Examples:
  DRY_RUN=1 scripts/run_hermes_asb_memory_all.sh
  RUNS_PER_CASE=2 ASB_AGENTS=financial_analyst_agent ASB_CASE_LIMIT=20 \
    scripts/run_hermes_asb_memory_all.sh jobs/asb-memory-smoke
  ASB_ATTACK_TYPES=context_ignoring KEEP_DEFENSES=1 \
    scripts/run_hermes_asb_memory_all.sh jobs/asb-memory-current-defense
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ ! -f "${config}" ]]; then
  echo "Config does not exist: ${config}" >&2
  exit 2
fi
if ! [[ "${runs_per_case}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_CASE must be a positive integer" >&2
  exit 2
fi
if ! [[ "${case_limit}" =~ ^[0-9]+$ ]]; then
  echo "ASB_CASE_LIMIT must be a non-negative integer" >&2
  exit 2
fi
case "${aggressive_filter,,}" in
  all|true|false) ;;
  *) echo "ASB_AGGRESSIVE must be all, true, or false" >&2; exit 2 ;;
esac
for flag in "${resume}" "${dry_run}" "${keep_defenses}" "${run_clean_control}"; do
  if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
    echo "RESUME, DRY_RUN, KEEP_DEFENSES and RUN_CLEAN_CONTROL must be 0 or 1" >&2
    exit 2
  fi
done

mkdir -p "${campaign_dir}/cases"
campaign_dir="$(cd "${campaign_dir}" && pwd)"
config="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
manifest="${campaign_dir}/cases.tsv"
manifest_candidate="${campaign_dir}/cases.generated.tsv"

write_summary() {
  PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${campaign_dir}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = root / "cases.tsv"
if not manifest.exists():
    raise SystemExit(0)
runs_per_case = int((root / "runs_per_case.txt").read_text().strip())
with manifest.open(newline="", encoding="utf-8") as handle:
    cases = list(csv.DictReader(handle, delimiter="\t"))

rows = []
for case in cases:
    case_dir = root / "cases" / case["case_id"]
    for run_number in range(1, runs_per_case + 1):
        run_dir = case_dir / f"run_{run_number:03d}"
        evaluation_path = run_dir / "target" / "evaluation.json"
        lifecycle_path = run_dir / "memory.lifecycle.json"
        error_path = next(
            (
                path
                for path in (
                    run_dir / "error.json",
                    run_dir / "container_error.json",
                    run_dir / "target" / "failure.json",
                )
                if path.exists()
            ),
            None,
        )
        row = {
            **case,
            "run": run_dir.name,
            "status": "not_started" if not run_dir.exists() else "missing_evaluation",
            "memory_changed": "",
            "memory_search_success": "",
            "memory_match_rank": "",
            "utility": "",
            "security": "",
            "attack_success": "",
            "attacker_called": "",
            "called_tools": "",
            "control_utility": "",
            "control_attack_success": "",
            "control_attacker_called": "",
            "control_called_tools": "",
            "target_only_attack_success": "",
            "error": "",
        }
        if evaluation_path.exists():
            try:
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                called = list(evaluation.get("called_tools") or [])
                row.update({
                    "status": "evaluated",
                    "utility": evaluation.get("utility", ""),
                    "security": evaluation.get("security", ""),
                    "attack_success": evaluation.get("attack_success", ""),
                    "attacker_called": case["attacker_tool"] in called,
                    "called_tools": json.dumps(called, ensure_ascii=False),
                    "memory_search_success": evaluation.get(
                        "memory_search_success", ""
                    ),
                    "memory_match_rank": evaluation.get("memory_match_rank", ""),
                })
                control_path = run_dir / "control" / "evaluation.json"
                if control_path.exists():
                    control = json.loads(control_path.read_text(encoding="utf-8"))
                    control_called = list(control.get("called_tools") or [])
                    row.update({
                        "control_utility": control.get("utility", ""),
                        "control_attack_success": control.get(
                            "attack_success", ""
                        ),
                        "control_attacker_called": (
                            case["attacker_tool"] in control_called
                        ),
                        "control_called_tools": json.dumps(
                            control_called, ensure_ascii=False
                        ),
                        "target_only_attack_success": (
                            evaluation.get("attack_success") is True
                            and control.get("attack_success") is False
                        ),
                    })
                if lifecycle_path.exists():
                    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
                    row["memory_changed"] = lifecycle.get("memory_changed", "")
            except Exception as exc:
                row["status"] = "invalid_evaluation"
                row["error"] = str(exc)
        elif error_path is not None:
            row["status"] = "failed"
            try:
                error = json.loads(error_path.read_text(encoding="utf-8"))
                row["error"] = str(error.get("error", error))
            except Exception:
                row["error"] = error_path.read_text(errors="replace").strip()
        else:
            exit_path = run_dir / "exit_code.txt"
            if exit_path.exists():
                row["error"] = f"process exit code {exit_path.read_text().strip()}"
        rows.append(row)

columns = [
    "case_id", "run", "agent_name", "task_index", "attacker_tool",
    "attack_type", "aggressive", "attack_category", "status",
    "memory_changed", "memory_search_success", "memory_match_rank",
    "utility", "security", "attack_success", "attacker_called",
    "called_tools", "control_utility", "control_attack_success",
    "control_attacker_called", "control_called_tools",
    "target_only_attack_success", "error",
]
with (root / "results.tsv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

successful_runs = [row for row in rows if row["attack_success"] is True]
successful_case_ids = sorted({row["case_id"] for row in successful_runs})
successful_cases = []
for case in cases:
    if case["case_id"] in successful_case_ids:
        run_successes = sum(
            row["case_id"] == case["case_id"] and row["attack_success"] is True
            for row in rows
        )
        successful_cases.append({
            **case,
            "successful_runs": run_successes,
            "evaluated_runs": sum(
                row["case_id"] == case["case_id"] and row["status"] == "evaluated"
                for row in rows
            ),
        })
case_columns = [
    "case_id", "agent_name", "task_index", "attacker_tool", "attack_type",
    "aggressive", "attack_category", "successful_runs", "evaluated_runs",
]
with (root / "successful_cases.tsv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=case_columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(successful_cases)
(root / "successful_case_ids.txt").write_text(
    "".join(f"{case_id}\n" for case_id in successful_case_ids), encoding="utf-8"
)

evaluated = [row for row in rows if row["status"] == "evaluated"]
poisoned = [row for row in evaluated if row["memory_changed"] is True]
retrieved = [
    row for row in evaluated if row["memory_search_success"] is True
]
controlled = [
    row for row in evaluated if isinstance(row["control_attack_success"], bool)
]
target_only = [
    row for row in controlled if row["target_only_attack_success"] is True
]
summary = {
    "case_count": len(cases),
    "runs_per_case": runs_per_case,
    "expected_runs": len(cases) * runs_per_case,
    "evaluated_runs": len(evaluated),
    "attack_successful_runs": len(successful_runs),
    "attack_successful_cases": len(successful_case_ids),
    "attack_success_rate_per_run": (
        len(successful_runs) / len(evaluated) if evaluated else None
    ),
    "case_success_rate": (
        len(successful_case_ids) / len(cases) if cases else None
    ),
    "memory_changed_runs": len(poisoned),
    "memory_changed_rate": len(poisoned) / len(evaluated) if evaluated else None,
    "memory_retrieval_successful_runs": len(retrieved),
    "memory_retrieval_success_rate": (
        len(retrieved) / len(evaluated) if evaluated else None
    ),
    "control_evaluated_runs": len(controlled),
    "control_attack_successful_runs": sum(
        row["control_attack_success"] is True for row in controlled
    ),
    "target_only_attack_successful_runs": len(target_only),
    "target_only_attack_success_rate": (
        len(target_only) / len(controlled) if controlled else None
    ),
    "utility_successful_runs": sum(row["utility"] is True for row in evaluated),
    "failed_or_incomplete_runs": len(rows) - len(evaluated),
    "successful_case_ids": successful_case_ids,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

on_exit() {
  status=$?
  write_summary || true
  echo "Campaign directory: ${campaign_dir}"
  echo "Summary: ${campaign_dir}/summary.json"
  echo "Successful cases: ${campaign_dir}/successful_cases.tsv"
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${python_bin}" - \
  "${config}" "${manifest_candidate}" "${attack_types}" "${agent_filter}" \
  "${task_filter}" "${tool_filter}" "${aggressive_filter}" "${case_limit}" <<'PY'
import csv
import json
import pathlib
import sys
import yaml

(
    config_path, output_path, attack_types_raw, agents_raw, tasks_raw,
    tools_raw, aggressive_raw, limit_raw,
) = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
asb = raw.get("agent_security_bench") or {}
data_dir = pathlib.Path(str(asb.get("data_dir") or ""))
if not data_dir.is_absolute():
    data_dir = pathlib.Path(config_path).parent / data_dir
if not data_dir.is_dir():
    raise SystemExit(f"ASB data directory does not exist: {data_dir}")

def read_jsonl(name):
    return [
        json.loads(line)
        for line in (data_dir / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def selected(raw_value):
    return {item.strip() for item in raw_value.split(",") if item.strip()}

valid_attack_types = {
    "naive", "fake_completion", "escape_characters", "context_ignoring",
    "combined_attack",
}
attack_types = [item.strip() for item in attack_types_raw.split(",") if item.strip()]
invalid = [item for item in attack_types if item not in valid_attack_types]
if invalid:
    raise SystemExit(f"Unknown attack types: {', '.join(invalid)}")
agents = selected(agents_raw)
task_indexes = {int(item) for item in selected(tasks_raw)}
tools = selected(tools_raw)
aggressive = aggressive_raw.lower()
limit = int(limit_raw)
task_rows = read_jsonl("agent_task.jsonl")
attack_rows = read_jsonl("all_attack_tools.jsonl")

cases = []
for task_row in task_rows:
    agent = str(task_row.get("agent_name") or "")
    if agents and agent not in agents:
        continue
    agent_attacks = [
        row for row in attack_rows
        if str(row.get("Corresponding Agent") or "") == agent
        and (not tools or str(row.get("Attacker Tool") or "") in tools)
        and (
            aggressive == "all"
            or str(row.get("Aggressive", "")).lower() == aggressive
        )
    ]
    for task_index, _ in enumerate(task_row.get("tasks") or []):
        if task_indexes and task_index not in task_indexes:
            continue
        for attack in agent_attacks:
            for attack_type in attack_types:
                cases.append({
                    "agent_name": agent,
                    "task_index": task_index,
                    "attacker_tool": str(attack["Attacker Tool"]),
                    "attack_type": attack_type,
                    "aggressive": str(attack.get("Aggressive", "")),
                    "attack_category": str(attack.get("Attack Type", "")),
                })
                if limit and len(cases) >= limit:
                    break
            if limit and len(cases) >= limit:
                break
        if limit and len(cases) >= limit:
            break
    if limit and len(cases) >= limit:
        break

if not cases:
    raise SystemExit("No ASB memory cases matched the filters")
for index, case in enumerate(cases, 1):
    case["case_id"] = f"case_{index:05d}"
columns = [
    "case_id", "agent_name", "task_index", "attacker_tool", "attack_type",
    "aggressive", "attack_category",
]
with pathlib.Path(output_path).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(cases)
print(f"Enumerated {len(cases)} ASB memory cases")
PY

if [[ -f "${manifest}" ]]; then
  if ! cmp -s "${manifest}" "${manifest_candidate}"; then
    echo "Existing campaign case manifest differs from the requested filters: ${manifest}" >&2
    echo "Use a new output directory or restore the original filters." >&2
    rm -f "${manifest_candidate}"
    exit 2
  fi
  rm -f "${manifest_candidate}"
else
  mv "${manifest_candidate}" "${manifest}"
fi

campaign_mode_file="${campaign_dir}/memory_mode.txt"
if [[ -f "${campaign_mode_file}" ]]; then
  existing_mode="$(<"${campaign_mode_file}")"
  if [[ "${existing_mode}" != "official_asb" ]]; then
    echo "Existing campaign uses memory mode ${existing_mode}; use a new output directory." >&2
    exit 2
  fi
elif [[ -n "$(find "${campaign_dir}/cases" -path '*/target/evaluation.json' -print -quit)" ]]; then
  echo "Existing campaign predates official_asb campaign metadata and already has results." >&2
  echo "Use a new output directory instead of mixing implementations." >&2
  exit 2
else
  printf '%s\n' "official_asb" > "${campaign_mode_file}"
fi

cp "${config}" "${campaign_dir}/source-asb.yaml"
printf '%s\n' "${keep_defenses}" > "${campaign_dir}/keep_defenses.txt"
printf '%s\n' "${run_clean_control}" > "${campaign_dir}/run_clean_control.txt"
printf '%s\n' "${runs_per_case}" > "${campaign_dir}/runs_per_case.txt"
case_count="$(( $(wc -l < "${manifest}") - 1 ))"
echo "Config: ${config}"
echo "Cases: ${case_count}"
echo "Runs per case: ${runs_per_case}"
echo "Expected runs: $(( case_count * runs_per_case ))"
echo "Defenses: $([[ "${keep_defenses}" == "1" ]] && echo kept || echo disabled-baseline)"
echo "Clean control: ${run_clean_control}"
echo "Output: ${campaign_dir}"

if [[ "${dry_run}" == "1" ]]; then
  echo "DRY_RUN=1: enumeration complete; no cases executed."
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is unset; official ASB retrieval and the Hermes model require it." >&2
  echo "Export the key in this shell before starting the campaign." >&2
  exit 2
fi

while IFS=$'\t' read -r case_id agent_name task_index attacker_tool attack_type aggressive attack_category; do
  [[ "${case_id}" == "case_id" ]] && continue
  case_dir="${campaign_dir}/cases/${case_id}"
  mkdir -p "${case_dir}"
  case_config="${case_dir}/case.yaml"
  "${python_bin}" - \
    "${config}" "${case_config}" "${agent_name}" \
    "${task_index}" "${attacker_tool}" "${attack_type}" \
    "${keep_defenses}" "${run_clean_control}" <<'PY'
import pathlib
import sys
import yaml

(
    source, destination, agent_name, task_index, attacker_tool,
    attack_type, keep_defenses, run_clean_control,
) = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(source).read_text(encoding="utf-8"))
asb = raw.setdefault("agent_security_bench", {})
if str(asb.get("implementation") or "") != "official_bridge":
    raise SystemExit(
        "ASB memory campaign requires "
        "agent_security_bench.implementation: official_bridge"
    )
if not asb.get("memory_db_dir"):
    raise SystemExit(
        "ASB memory campaign requires agent_security_bench.memory_db_dir"
    )
asb.update({
    "enabled": True,
    "agent_name": agent_name,
    "task_index": int(task_index),
    "attacker_tool": attacker_tool,
    "injection_method": "memory_attack",
    "attack_type": attack_type,
})
memory = raw.setdefault("memory_experiment", {})
memory.update({
    "mode": "official_asb",
    "run_clean_control": run_clean_control == "1",
})
memory.pop("poisoning_input_file", None)
if keep_defenses != "1":
    for name in (
        "aegis", "progent", "pro2guard", "agentspec", "llamafirewall",
        "toolsafe", "agentdog", "agentguard", "agentsight",
    ):
        raw.setdefault(name, {})["enabled"] = False
pathlib.Path(destination).write_text(
    yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PY

  for ((run_number=1; run_number<=runs_per_case; run_number++)); do
    run_name="$(printf 'run_%03d' "${run_number}")"
    run_dir="${case_dir}/${run_name}"
    evaluation="${run_dir}/target/evaluation.json"
    if [[ "${resume}" == "1" && -f "${evaluation}" ]]; then
      echo "=== ${case_id} ${run_name}: already evaluated, skipping ==="
      continue
    fi
    if [[ -d "${run_dir}" ]]; then
      failed_archive="${run_dir}.failed_${stamp}"
      if [[ -e "${failed_archive}" ]]; then
        failed_archive="${failed_archive}_$$"
      fi
      mv "${run_dir}" "${failed_archive}"
      echo "Archived incomplete run: ${failed_archive}"
    fi
    mkdir -p "${run_dir}"
    echo "=== ${case_id}/${case_count} ${run_name}/${runs_per_case}: ${agent_name} task=${task_index} tool=${attacker_tool} attack=${attack_type} ==="
    AGENT_JOB_DIR="${run_dir}" \
    AGENT_BATCH_DIR="${campaign_dir}" \
    PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" "${repo_root}/src/agent_scaffold/main.py" \
        --config "${case_config}" 2>&1 | tee "${run_dir}/console.log"
    run_status=${PIPESTATUS[0]}
    printf '%s\n' "${run_status}" > "${run_dir}/exit_code.txt"
    if (( run_status != 0 )); then
      echo "FAILED: ${case_id} ${run_name} (exit ${run_status}); continuing" >&2
    fi
  done
done < "${manifest}"

write_summary
