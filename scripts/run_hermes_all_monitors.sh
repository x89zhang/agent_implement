#!/usr/bin/env bash
set -euo pipefail

runs="${1:-20}"
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  echo "usage: $0 [runs-per-condition>=2]" >&2
  exit 2
fi

case_id="$(PYTHONPATH=src python - <<'PYCONFIG'
from agent_scaffold.backends.guards import GUARDS
from agent_scaffold.config import load_config

on = load_config('agents/hermes/agentdojo-all-monitors.yaml')
off = load_config('agents/hermes/agentdojo-all-monitors-no-injection.yaml')
if (on.agentdojo.suite, on.agentdojo.case) != (off.agentdojo.suite, off.agentdojo.case):
    raise SystemExit('Injection and control configs must select the same AgentDojo task')
if not on.agentdojo.injection_enabled or off.agentdojo.injection_enabled:
    raise SystemExit('Expected injection on in the first config and off in the control')
on_guards = {name for name in GUARDS if getattr(on, name).enabled}
off_guards = {name for name in GUARDS if getattr(off, name).enabled}
if on_guards != off_guards:
    raise SystemExit(f'Monitor sets differ: on={sorted(on_guards)}, off={sorted(off_guards)}')
if not {'toolsafe', 'agentdog'} <= on_guards:
    raise SystemExit('ToolSafe and AgentDoG must remain enabled')
for name in ('janus', 'stepguard', 'safeagent'):
    if name in on_guards:
        raise SystemExit(f'{name} requires an additional local model or service endpoint')
for name in on_guards - {'agentdog'}:
    if getattr(on, name).mode != 'monitor' or getattr(off, name).mode != 'monitor':
        raise SystemExit(f'{name} must be in passive monitor mode')
if on.agentdog.mode != 'diagnose' or off.agentdog.mode != 'diagnose':
    raise SystemExit('AgentDoG must be in passive diagnose mode')
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
