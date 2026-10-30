#!/usr/bin/env bash
# Add StepGuard and PIGuard to every all-monitors batch (injected and clean) of
# the given AgentDojo pairs. Runs that already have valid labels are skipped,
# so rerunning only retries errors. StepGuard must be served as configured in
# agents/hermes/supplement-stepguard-piguard.yaml.
# With methods after the pairs file, it instead retries those methods' replay
# errors with their batch config (e.g. janus toolsafe, with both serving).
set -euo pipefail

# The all-monitors image (see docs/hermes_mixed_pairs.md); the newest image is
# the monitor-off screen image, which lacks the monitors' dependencies.
image="agent-scaffold-agentdojo:latest-hermes-a2265dbace4a"
method_config="agents/hermes/supplement-stepguard-piguard.yaml"

usage() {
  echo "usage: $0 pairs-file [method ...]" >&2
  echo "pairs-file: one 'suite case' per line, e.g. 'banking user_task_0_injection_3'" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 1 )) || [[ ! -f "$1" ]]; then
  usage
  exit 2
fi
pairs_file="$1"
shift
if (( $# > 0 )); then
  methods=("$@")
  options=()
else
  methods=(stepguard piguard)
  options=(--add-methods --method-config "${method_config}")
fi

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

# Each served model must be listed by its endpoint; a 200 from another server
# on the port is not enough.
PYTHONPATH=src python - "${method_config}" "${methods[@]}" <<'PYCHECK'
import json
import sys
import urllib.request
import yaml
from agent_scaffold.config import load_config

method_config, *methods = sys.argv[1:]
batch = load_config('agents/hermes/agentdojo-all-monitors.yaml')
served = {
    'stepguard': yaml.safe_load(open(method_config, encoding="utf-8"))["stepguard"],
    'janus': {'base_url': batch.janus.base_url, 'model': batch.janus.model},
    'toolsafe': {'base_url': batch.toolsafe.base_url, 'model': batch.toolsafe.model},
}
for name in methods:
    if name not in served:
        continue
    url = served[name]['base_url'].rstrip('/') + '/models'
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = [item.get('id') for item in json.load(response).get('data', [])]
    except Exception as exc:
        raise SystemExit(f'{name} endpoint is unavailable at {url}: {exc}')
    if served[name]['model'] not in models:
        raise SystemExit(f"{name}: {url} serves {models}, not {served[name]['model']}")
PYCHECK
docker image inspect "${image}" > /dev/null

results=()
for pair in "${pairs[@]}"; do
  read -r suite_id case_id <<< "${pair}"
  if [[ "${suite_id}" == "slack" ]]; then
    root="jobs/agentdojo_${case_id}/Hermes/gpt"
  else
    root="jobs/agentdojo_${suite_id}_${case_id}/Hermes/gpt"
  fi
  for summary in "${root}"/all_monitors/{skill_injection,no_injection}/*/summary.json; do
    [[ -f "${summary}" ]] || continue
    batch_dir="$(dirname "${summary}")"
    echo "### ${batch_dir}"
    if python3 scripts/supplement_hermes_monitors.py "${batch_dir}" \
        --methods "${methods[@]}" "${options[@]}" --image "${image}"; then
      results+=("ok ${batch_dir}")
    else
      results+=("some-replays-invalid ${batch_dir}")
    fi
  done
done

echo
echo "### batches"
printf '%s\n' "${results[@]}"
