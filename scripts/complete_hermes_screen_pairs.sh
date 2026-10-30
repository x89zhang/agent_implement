#!/usr/bin/env bash
# Turn monitor-off screen batches (scripts/screen_hermes_agentdojo.sh) into
# all-monitors pairs without rerunning the injected condition. Monitors are
# passive replays of the recorded lifecycle, so for each pair this:
#   1. copies the screen batch into all_monitors/skill_injection/<batch>;
#   2. runs the no-injection condition with all monitors under the same batch
#      name, as many runs as the screen batch has;
#   3. replays the batch-served monitors onto the copied injected runs, in the
#      image an all-monitors batch uses (the screen image lacks the monitors'
#      dependencies).
# StepGuard and PIGuard are added afterwards by supplement replays, as for the
# other all-monitors batches. Rerunning skips steps 1-2 if they completed and
# redoes step 3.
set -euo pipefail

usage() {
  echo "usage: $0 pairs-file" >&2
  echo "pairs-file: one 'suite case screen-batch-dir' per line" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# != 1 )) || [[ ! -f "$1" ]]; then
  usage
  exit 2
fi
pairs_file="$1"

pairs=()
while read -r suite_id case_id screen_dir _; do
  [[ -z "${suite_id}" || "${suite_id}" == \#* ]] && continue
  if ! [[ "${suite_id}" =~ ^[A-Za-z][A-Za-z0-9_-]*$ && "${case_id}" =~ ^user_task_[0-9]+_injection_[0-9]+$ ]]; then
    echo "invalid pair: ${suite_id} ${case_id}" >&2
    exit 2
  fi
  if [[ ! -f "${screen_dir}/summary.json" ]]; then
    echo "screen batch has no summary.json: ${screen_dir}" >&2
    exit 2
  fi
  pairs+=("${suite_id} ${case_id} ${screen_dir%/}")
done < "${pairs_file}"
if (( ${#pairs[@]} == 0 )); then
  echo "no pairs in ${pairs_file}" >&2
  exit 2
fi

# Guards served during a batch (their endpoints must answer) and the image an
# all-monitors batch runs in.
selection="$(PYTHONPATH=src python - <<'PYGUARDS'
import pathlib
import urllib.request
from unittest import mock
from agent_scaffold.backends import container
from agent_scaffold.backends.guards import GUARDS
from agent_scaffold.config import load_config

on = load_config('agents/hermes/agentdojo-all-monitors.yaml')
for name, url in {'toolsafe': on.toolsafe.base_url, 'janus': on.janus.base_url}.items():
    try:
        urllib.request.urlopen(url.rstrip('/') + '/models', timeout=5).close()
    except Exception as exc:
        raise SystemExit(f'{name} endpoint is unavailable at {url}: {exc}')


class Tagged(Exception):
    pass


# ensure_hermes_image computes the tag, then inspects it; stop there.
def inspect(command, workspace):
    raise Tagged(command[-1])


with mock.patch.object(container, '_run_checked', inspect):
    try:
        container.ensure_hermes_image(on, pathlib.Path.cwd())
        raise SystemExit('could not determine the all-monitors image')
    except Tagged as tagged:
        image = str(tagged)
container.subprocess.run(['docker', 'image', 'inspect', image], check=True, capture_output=True)
print(' '.join(name for name in GUARDS if getattr(on, name).enabled) + '\t' + image)
PYGUARDS
)"
IFS=$'\t' read -r guards image <<< "${selection}"
echo "### monitors: ${guards}"
echo "### image: ${image}"

results=()
for pair in "${pairs[@]}"; do
  read -r suite_id case_id screen_dir <<< "${pair}"
  if [[ "${suite_id}" == "slack" ]]; then
    root="jobs/agentdojo_${case_id}/Hermes/gpt"
  else
    root="jobs/agentdojo_${suite_id}_${case_id}/Hermes/gpt"
  fi
  batch="$(basename "${screen_dir}")"
  on_dir="${root}/all_monitors/skill_injection/${batch}"
  off_dir="${root}/all_monitors/no_injection/${batch}"
  runs="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["runs"])' "${screen_dir}/summary.json")"
  echo "### ${suite_id} ${case_id}: ${runs} runs per condition -> ${root}/all_monitors (${batch})"

  if [[ ! -f "${on_dir}/summary.json" ]]; then
    mkdir -p "$(dirname "${on_dir}")"
    cp -a "${screen_dir}" "${on_dir}"
  fi

  if [[ ! -f "${off_dir}/summary.json" ]]; then
    if [[ -e "${off_dir}" ]]; then
      echo "incomplete no-injection batch; remove it and rerun: ${off_dir}" >&2
      exit 1
    fi
    mkdir -p "${off_dir}"
    # Kept beside the source YAML so relative paths resolve the same way.
    off_config="$(mktemp agents/hermes/.agentdojo-all-monitors-no-injection.XXXXXX.yaml)"
    python3 scripts/hermes_campaign.py configure --keep-guards \
      agents/hermes/agentdojo-all-monitors-no-injection.yaml "${off_config}" "${off_dir}"
    python3 - "${suite_id}" "${case_id}" "${off_config}" <<'PYCASE'
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
    if ! PYTHONPATH=src python src/agent_scaffold/main.py \
        --config "${off_config}" \
        --runs "${runs}" \
        --runs-dir "${off_dir}"; then
      rm -f "${off_config}"
      echo "FAILED no-injection runs: ${suite_id} ${case_id}" >&2
      results+=("${suite_id} ${case_id} no-injection-failed")
      continue
    fi
    rm -f "${off_config}"
    python3 scripts/hermes_campaign.py finalize "${off_dir}"
  fi

  # The clean runs already have these monitors from their batch.
  # shellcheck disable=SC2086
  if python3 scripts/supplement_hermes_monitors.py "${on_dir}" \
      --methods ${guards} --add-methods --all-runs --image "${image}"; then
    results+=("${suite_id} ${case_id} ok")
  else
    results+=("${suite_id} ${case_id} some-replays-invalid")
  fi
done

echo
echo "### pairs"
printf '%s\n' "${results[@]}"
