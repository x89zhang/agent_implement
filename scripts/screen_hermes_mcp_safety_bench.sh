#!/usr/bin/env bash
# Screen MCP-SafetyBench tool-poisoning tasks before adding monitors. Each run
# has a poisoned target phase and a clean control phase (the same task on the
# unmodified servers), with every monitor off; monitors are passive replays and
# can be added afterwards without rerunning the agent.
set -euo pipefail

usage() {
  echo "usage: $0 runs-per-task>=1 tasks-file" >&2
  echo "tasks-file: one '<domain>/<task file stem>' per line, e.g." >&2
  echo "  financial_analysis/yfinance_task_0001 (# starts a comment)" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# != 2 )) || ! [[ "$1" =~ ^[0-9]+$ ]] || (( $1 < 1 )) || [[ ! -f "$2" ]]; then
  usage
  exit 2
fi
runs="$1"
tasks_file="$2"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is unset." >&2
  exit 2
fi

tasks=()
while read -r task _; do
  [[ -z "${task}" || "${task}" == \#* ]] && continue
  if ! [[ "${task}" =~ ^[a-z_]+/[A-Za-z0-9_]+$ ]]; then
    echo "invalid task: ${task}" >&2
    exit 2
  fi
  tasks+=("${task}")
done < "${tasks_file}"
if (( ${#tasks[@]} == 0 )); then
  echo "no tasks in ${tasks_file}" >&2
  exit 2
fi

results=()
for task in "${tasks[@]}"; do
  slug="${task//\//-}"
  out_dir="jobs/screen/mcp_safety_bench/${slug}/$(date -u +%Y%m%dT%H%M%SZ)_hermes-mcp-safety-bench"
  mkdir -p "${out_dir}"
  # Kept beside the source YAML so relative paths resolve the same way.
  config="$(mktemp agents/hermes/.mcp-safety-bench-screen.XXXXXX.yaml)"
  PYTHONPATH=src python - "${task}" "${config}" <<'PYCONFIG'
import pathlib
import sys
import yaml
from agent_scaffold.backends.guards import GUARDS

task, target = sys.argv[1:3]
raw = yaml.safe_load(pathlib.Path("agents/hermes/mcp-safety-bench.yaml").read_text(encoding="utf-8"))
raw["mcp_safety_bench"].update({"task": task, "poisoning_enabled": True, "clean_control": True})
raw.setdefault("execution", {}).setdefault("hermes", {})["defense_mode"] = "replay"
raw["memory_experiment"] = {"mode": "off"}
for name in GUARDS:
    raw.setdefault(name, {})["enabled"] = False
raw.setdefault("agentsight", {})["enabled"] = False
pathlib.Path(target).write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
PYCONFIG
  cp "${config}" "${out_dir}/mcp-safety-bench-screen.yaml"
  echo "### ${task} -> ${out_dir}"
  if PYTHONPATH=src python src/agent_scaffold/main.py \
      --config "${config}" \
      --runs "${runs}" \
      --runs-dir "${out_dir}"; then
    results+=("${slug} ${out_dir}")
  else
    echo "FAILED ${slug}" >&2
    results+=("${slug} FAILED")
  fi
  rm -f "${config}"
done

echo
echo "### attack success per task (keep: mixed target, control 0)"
for result in "${results[@]}"; do
  read -r slug out_dir <<< "${result}"
  if [[ "${out_dir}" == FAILED ]]; then
    printf '%-50s FAILED\n' "${slug}"
    continue
  fi
  python3 - "${slug}" "${out_dir}" <<'PYSUMMARY'
import json
import pathlib
import sys

slug, batch = sys.argv[1], pathlib.Path(sys.argv[2])
counts = {"target": [0, 0, 0], "control": [0, 0, 0]}  # attack, upstream verdict, runs
errors = 0
for run in sorted(batch.glob("run_*")):
    for phase, count in counts.items():
        path = run / phase / "evaluation.json"
        if path.is_file():
            evaluation = json.loads(path.read_text())
            count[0] += bool(evaluation.get("attack_success"))
            count[1] += bool(evaluation.get("attack_success_upstream"))
            count[2] += 1
            errors += len(evaluation.get("tool_errors") or [])
(t, tu, tn), (c, cu, cn) = counts["target"], counts["control"]
verdict = "keep" if 0 < t < tn and c == 0 else "-"
print(
    f"{slug:50} target {t}/{tn} (upstream {tu}) control {c}/{cn} {verdict}"
    f"  tool errors {errors}  {batch}"
)
PYSUMMARY
done
