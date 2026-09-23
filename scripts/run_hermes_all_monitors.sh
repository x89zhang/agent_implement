#!/usr/bin/env bash
set -euo pipefail

runs="${1:-20}"
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  echo "usage: $0 [runs-per-condition>=2]" >&2
  exit 2
fi

case_id="$(PYTHONPATH=src python - <<'PYCONFIG'
from agent_scaffold.config import load_config
on = load_config('agents/hermes/agentdojo-all-monitors.yaml')
off = load_config('agents/hermes/agentdojo-all-monitors-no-injection.yaml')
assert on.agentdojo.suite == off.agentdojo.suite
assert on.agentdojo.case == off.agentdojo.case
assert on.agentdojo.injection_enabled and not off.agentdojo.injection_enabled
assert on.progent.enabled and off.progent.enabled
print(on.agentdojo.case)
PYCONFIG
)"
root="jobs/agentdojo_${case_id}/Hermes/gpt"
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

python3 scripts/analyze_hermes_agentdojo_monitors.py --root "${root}" --batch "${batch}"
