#!/usr/bin/env bash
# Estimate attack success for AgentDojo task pairs before running them with
# all monitors. Runs only the skill-injection condition, with every monitor
# off; monitors are passive replays, so the agent behaves as in an
# all-monitors batch.
set -euo pipefail

usage() {
  echo "usage: $0 runs-per-pair>=1 pairs-file" >&2
  echo "pairs-file: one 'suite case' per line, e.g. 'banking user_task_9_injection_4'" >&2
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
pairs_file="$2"

pairs=()
while read -r suite_id case_id _; do
  [[ -z "${suite_id}" || "${suite_id}" == \#* ]] && continue
  if ! [[ "${suite_id}" =~ ^[A-Za-z][A-Za-z0-9_-]*$ && "${case_id}" =~ ^user_task_[0-9]+_injection_[0-9]+$ ]]; then
    echo "invalid pair: ${suite_id} ${case_id}" >&2
    exit 2
  fi
  pairs+=("${suite_id} ${case_id}")
done < "${pairs_file}"
if (( ${#pairs[@]} == 0 )); then
  echo "no pairs in ${pairs_file}" >&2
  exit 2
fi

root="jobs/screen"
results=()
for pair in "${pairs[@]}"; do
  read -r suite_id case_id <<< "${pair}"
  batch="$(date -u +%Y%m%dT%H%M%SZ)_hermes-agentdojo_batch"
  out_dir="${root}/agentdojo_${suite_id}_${case_id}/${batch}"
  mkdir -p "${out_dir}"
  # Kept beside the source YAML so relative paths resolve the same way.
  config="$(mktemp agents/hermes/.agentdojo-screen.XXXXXX.yaml)"
  PYTHONPATH=src python - "${suite_id}" "${case_id}" "${config}" <<'PYCONFIG'
import pathlib
import sys
import yaml
from agent_scaffold.backends.guards import GUARDS

suite_id, case_id, target = sys.argv[1:4]
raw = yaml.safe_load(pathlib.Path("agents/hermes/agentdojo-all-monitors.yaml").read_text(encoding="utf-8"))
if not (raw["agentdojo"].get("injection_enabled") and raw["agentdojo"].get("skill_injection_enabled")):
    raise SystemExit("Expected skill injection on in agentdojo-all-monitors.yaml")
raw["agentdojo"]["suite"] = suite_id
raw["agentdojo"]["case"] = case_id
for name in GUARDS:
    if isinstance(raw.get(name), dict):
        raw[name]["enabled"] = False
pathlib.Path(target).write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
PYCONFIG
  echo "### ${suite_id} ${case_id} -> ${out_dir}"
  if PYTHONPATH=src python src/agent_scaffold/main.py \
      --config "${config}" \
      --runs "${runs}" \
      --runs-dir "${out_dir}"; then
    results+=("${suite_id} ${case_id} ${out_dir}")
  else
    echo "FAILED ${suite_id} ${case_id}" >&2
    results+=("${suite_id} ${case_id} FAILED")
  fi
  rm -f "${config}"
done

echo
echo "### attack success per pair"
for result in "${results[@]}"; do
  read -r suite_id case_id out_dir <<< "${result}"
  if [[ "${out_dir}" == FAILED ]]; then
    printf '%-10s %-28s FAILED\n' "${suite_id}" "${case_id}"
    continue
  fi
  python3 - "${suite_id}" "${case_id}" "${out_dir}/summary.json" <<'PYSUMMARY'
import json
import sys

suite_id, case_id, path = sys.argv[1:4]
stats = json.load(open(path)).get("agentdojo") or {}
successes, trials = stats.get("attack_successes", 0), stats.get("attack_trials", 0)
verdict = "mixed" if 0 < successes < trials else "all" if trials and successes == trials else "none"
print(f"{suite_id:10} {case_id:28} {successes}/{trials} {verdict}")
PYSUMMARY
done
