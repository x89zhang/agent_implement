#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [runs-per-condition>=2] [case-id] [--suite SUITE]" >&2
  echo "example: $0 20 user_task_0_injection_1 --suite travel" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
runs=20
case_override=""
suite_override=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  runs="$1"
  shift
fi
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  case_override="$1"
  shift
fi
while (( $# > 0 )); do
  case "$1" in
    --suite)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then
        echo "missing value for --suite" >&2
        usage
        exit 2
      fi
      suite_override="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  usage
  exit 2
fi
if [[ -n "${case_override}" ]] && ! [[ "${case_override}" =~ ^user_task_[0-9]+(_injection_[0-9]+)?$ ]]; then
  echo "invalid AgentDojo case id: ${case_override}" >&2
  usage
  exit 2
fi
if [[ -n "${suite_override}" ]] && ! [[ "${suite_override}" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]]; then
  echo "invalid AgentDojo suite: ${suite_override}" >&2
  usage
  exit 2
fi
if [[ -n "${suite_override}" && -z "${case_override}" ]]; then
  echo "specify a case id when changing the suite" >&2
  usage
  exit 2
fi

selection="$(PYTHONPATH=src python - "${case_override}" "${suite_override}" <<'PYCONFIG'
import sys
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
# Local-model guards whose servers must be up during the batch; the others
# (StepGuard, PIGuard) are added afterwards by supplement replays.
if not {'toolsafe', 'janus'} <= on_guards:
    raise SystemExit('ToolSafe and JANUS must be enabled for the batch')
for name in on_guards - {'agentdog'}:
    if getattr(on, name).mode != 'monitor' or getattr(off, name).mode != 'monitor':
        raise SystemExit(f'{name} must be in passive monitor mode')
if 'agentdog' in on_guards and (on.agentdog.mode != 'diagnose' or off.agentdog.mode != 'diagnose'):
    raise SystemExit('AgentDoG must be in passive diagnose mode')
# Endpoints that must answer before any run starts.
import urllib.request
endpoints = {
    'toolsafe': on.toolsafe.base_url.rstrip('/') + '/models',
    'janus': on.janus.base_url.rstrip('/') + '/models',
}
for name, url in endpoints.items():
    try:
        urllib.request.urlopen(url, timeout=5).close()
    except Exception as exc:
        raise SystemExit(f'{name} endpoint is unavailable at {url}: {exc}')
import os
if 'adr' in on_guards and not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('CLAUDE_CODE_OAUTH_TOKEN')):
    raise SystemExit('ADR needs ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN for its Claude reasoning step')
print(f"{sys.argv[2] or on.agentdojo.suite}\t{sys.argv[1] or on.agentdojo.case}")
PYCONFIG
)"
IFS=$'\t' read -r suite_id case_id <<< "${selection}"
if [[ "${suite_id}" == "slack" ]]; then
  root="jobs/agentdojo_${case_id}/Hermes/gpt"
else
  root="jobs/agentdojo_${suite_id}_${case_id}/Hermes/gpt"
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch="${stamp}_hermes-agentdojo_batch"

on_dir="${root}/all_monitors/skill_injection/${batch}"
off_dir="${root}/all_monitors/no_injection/${batch}"
mkdir -p "${on_dir}" "${off_dir}"
# Kept beside the source YAMLs so relative paths resolve the same way.
on_config="$(mktemp agents/hermes/.agentdojo-all-monitors.XXXXXX.yaml)"
off_config="$(mktemp agents/hermes/.agentdojo-all-monitors-no-injection.XXXXXX.yaml)"
cleanup() { rm -f "${on_config}" "${off_config}"; }
trap cleanup EXIT
# Each condition starts from a fresh AGrail memory under its batch directory;
# its path and hashes are recorded in campaign_manifest.json.
python3 scripts/hermes_campaign.py configure --keep-guards \
  agents/hermes/agentdojo-all-monitors.yaml "${on_config}" "${on_dir}"
python3 scripts/hermes_campaign.py configure --keep-guards \
  agents/hermes/agentdojo-all-monitors-no-injection.yaml "${off_config}" "${off_dir}"

if [[ -n "${case_override}" ]]; then
  python3 - "${suite_id}" "${case_id}" "${on_config}" "${off_config}" <<'PYCASE'
import pathlib
import sys
import yaml

suite_id, case_id = sys.argv[1:3]
for filename in sys.argv[3:]:
    path = pathlib.Path(filename)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["agentdojo"]["suite"] = suite_id
    raw["agentdojo"]["case"] = case_id
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
PYCASE
fi

PYTHONPATH=src python src/agent_scaffold/main.py \
  --config "${on_config}" \
  --runs "${runs}" \
  --runs-dir "${on_dir}"
python3 scripts/hermes_campaign.py finalize "${on_dir}"

PYTHONPATH=src python src/agent_scaffold/main.py \
  --config "${off_config}" \
  --runs "${runs}" \
  --runs-dir "${off_dir}"
python3 scripts/hermes_campaign.py finalize "${off_dir}"

python3 scripts/analyze_hermes_agentdojo_monitors.py --root "${root}" --batch "${batch}"
