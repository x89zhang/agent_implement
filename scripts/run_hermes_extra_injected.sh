#!/usr/bin/env bash
# Add injected runs with all monitors to existing AgentDojo pairs, as a new
# batch under all_monitors/skill_injection; no clean runs. Same setup as
# scripts/run_hermes_all_monitors.sh: ToolSafe and JANUS must be serving.
# StepGuard and PIGuard are added afterwards (scripts/supplement_hermes_pairs.sh).
set -euo pipefail

usage() {
  echo "usage: $0 runs-per-pair>=1 pairs-file" >&2
  echo "pairs-file: one 'suite case' per line, e.g. 'banking user_task_0_injection_3'" >&2
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

PYTHONPATH=src python - <<'PYCHECK'
import urllib.request
from agent_scaffold.config import load_config

on = load_config('agents/hermes/agentdojo-all-monitors.yaml')
if not (on.agentdojo.injection_enabled and on.agentdojo.skill_injection_enabled):
    raise SystemExit('Expected skill injection on in agentdojo-all-monitors.yaml')
for name, url in {'toolsafe': on.toolsafe.base_url, 'janus': on.janus.base_url}.items():
    try:
        urllib.request.urlopen(url.rstrip('/') + '/models', timeout=5).close()
    except Exception as exc:
        raise SystemExit(f'{name} endpoint is unavailable at {url}: {exc}')
PYCHECK

results=()
for pair in "${pairs[@]}"; do
  read -r suite_id case_id <<< "${pair}"
  if [[ "${suite_id}" == "slack" ]]; then
    root="jobs/agentdojo_${case_id}/Hermes/gpt"
  else
    root="jobs/agentdojo_${suite_id}_${case_id}/Hermes/gpt"
  fi
  batch="$(date -u +%Y%m%dT%H%M%SZ)_hermes-agentdojo_batch"
  on_dir="${root}/all_monitors/skill_injection/${batch}"
  mkdir -p "${on_dir}"
  echo "### ${suite_id} ${case_id}: ${runs} injected runs -> ${on_dir}"
  # Kept beside the source YAML so relative paths resolve the same way.
  on_config="$(mktemp agents/hermes/.agentdojo-all-monitors.XXXXXX.yaml)"
  # A fresh AGrail memory under the batch directory, as in a regular batch.
  python3 scripts/hermes_campaign.py configure --keep-guards \
    agents/hermes/agentdojo-all-monitors.yaml "${on_config}" "${on_dir}"
  python3 - "${suite_id}" "${case_id}" "${on_config}" <<'PYCASE'
import pathlib
import sys
import yaml

suite_id, case_id, filename = sys.argv[1:4]
path = pathlib.Path(filename)
raw = yaml.safe_load(path.read_text(encoding="utf-8"))
raw["agentdojo"]["suite"] = suite_id
raw["agentdojo"]["case"] = case_id
path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
PYCASE
  if PYTHONPATH=src python src/agent_scaffold/main.py \
      --config "${on_config}" \
      --runs "${runs}" \
      --runs-dir "${on_dir}"; then
    python3 scripts/hermes_campaign.py finalize "${on_dir}"
    results+=("${suite_id} ${case_id} ${on_dir}")
  else
    echo "FAILED ${suite_id} ${case_id}" >&2
    results+=("${suite_id} ${case_id} FAILED")
  fi
  rm -f "${on_config}"
done

echo
echo "### attack success per pair (new batch)"
for result in "${results[@]}"; do
  read -r suite_id case_id on_dir <<< "${result}"
  if [[ "${on_dir}" == FAILED ]]; then
    printf '%-10s %-28s FAILED\n' "${suite_id}" "${case_id}"
    continue
  fi
  python3 - "${suite_id}" "${case_id}" "${on_dir}/summary.json" <<'PYSUMMARY'
import json
import sys

suite_id, case_id, path = sys.argv[1:4]
stats = json.load(open(path)).get("agentdojo") or {}
print(f"{suite_id:10} {case_id:28} {stats.get('attack_successes', 0)}/{stats.get('attack_trials', 0)}")
PYSUMMARY
done
