#!/usr/bin/env bash
set -euo pipefail

runs="${1:-20}"
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  echo "usage: $0 [runs-per-condition>=2]" >&2
  exit 2
fi

root="jobs/agentdojo_user_task_15_injection_5/Hermes/gpt"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch="${stamp}_hermes-agentdojo_batch"

PYTHONPATH=src python src/agent_scaffold/main.py \
  --config agents/hermes/agentdojo-all-monitors.yaml \
  --runs "${runs}" \
  --runs-dir "${root}/all_monitors/skill_injection/${batch}"

PYTHONPATH=src python src/agent_scaffold/main.py \
  --config agents/hermes/agentdojo-all-monitors-no-injection.yaml \
  --runs "${runs}" \
  --runs-dir "${root}/all_monitors/no_injection/${batch}"

python3 "${root}/analysis_alarm.py"
