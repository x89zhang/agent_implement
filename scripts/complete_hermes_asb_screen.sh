#!/usr/bin/env bash
# Turn monitor-off ASB screen batches (scripts/screen_hermes_asb.sh) into
# all-monitors batches without rerunning the agent: copy each batch into
# jobs/asb_all_monitors/<group>/ and replay monitors onto its target and
# control phases, in the image an ASB all-monitors batch uses.
# Without methods it replays the monitors served during a batch (ToolSafe and
# JANUS must be serving); name methods to add others, e.g. stepguard piguard
# (StepGuard serving). Rerunning skips the copy and only retries errors.
set -euo pipefail

method_config="agents/hermes/supplement-stepguard-piguard.yaml"

usage() {
  echo "usage: $0 batches-file [method ...]" >&2
  echo "batches-file: one screen batch directory per line" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 1 )); then
  usage
  exit 2
fi
if [[ ! -f "$1" ]]; then
  echo "batches file not found: $1" >&2
  exit 2
fi
batches_file="$1"
shift

batches=()
while read -r batch _; do
  [[ -z "${batch}" || "${batch}" == \#* ]] && continue
  if [[ ! -f "${batch}/summary.json" ]]; then
    echo "screen batch has no summary.json: ${batch}" >&2
    exit 2
  fi
  batches+=("${batch%/}")
done < "${batches_file}"
if (( ${#batches[@]} == 0 )); then
  echo "no batches in ${batches_file}" >&2
  exit 2
fi

if (( $# > 0 )); then
  methods=("$@")
else
  read -r -a methods <<< "$(PYTHONPATH=src python -c "
from agent_scaffold.backends.guards import GUARDS
from agent_scaffold.config import load_config
on = load_config('agents/hermes/agentdojo-all-monitors.yaml')
print(' '.join(name for name in GUARDS if getattr(on, name).enabled))
")"
fi
options=(--add-methods)
if python3 - "${method_config}" "${methods[@]}" <<'PYSECTIONS'
import sys
import yaml

raw = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
sys.exit(0 if all(isinstance(raw.get(name), dict) for name in sys.argv[2:]) else 1)
PYSECTIONS
then
  options+=(--method-config "${method_config}")
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

# REPLAY_ALL=1 recomputes the methods' valid labels too (e.g. after a monitor fix);
# by default only missing methods and replay errors are replayed.
if [[ "${REPLAY_ALL:-0}" == 1 ]]; then
  options+=(--all-runs)
fi
# The image of an ASB all-monitors batch, from the config that script writes.
reference="$(mktemp -d)"
trap 'rm -rf "${reference}"' EXIT
DRY_RUN=1 SKIP_SERVICE_CHECKS=1 scripts/run_hermes_asb_all_monitors.sh 2 "${reference}/batch" > /dev/null
image="$(python3 scripts/hermes_image.py "${reference}/batch/asb-all-monitors.yaml")"
echo "### monitors: ${methods[*]}"
echo "### image: ${image}"

results=()
for screen in "${batches[@]}"; do
  group="$(basename "$(dirname "${screen}")")"
  dest="jobs/asb_all_monitors/${group}/$(basename "${screen}")"
  if [[ ! -f "${dest}/summary.json" ]]; then
    mkdir -p "$(dirname "${dest}")"
    cp -a "${screen}" "${dest}"
  fi
  echo "### ${dest}"
  if python3 scripts/supplement_hermes_monitors.py "${dest}" \
      --methods "${methods[@]}" "${options[@]}" --image "${image}"; then
    results+=("ok ${dest}")
  else
    results+=("some-replays-invalid ${dest}")
  fi
done

echo
echo "### batches"
printf '%s\n' "${results[@]}"
