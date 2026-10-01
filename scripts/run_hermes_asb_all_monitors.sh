#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [runs>=2] [output-directory] [--agent-name NAME] [--task-index N] [--attacker-tool TOOL] [--attack-type TYPE]" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
runs=20
batch_dir_override=""
agent_name_override=""
task_index_override=""
attacker_tool_override=""
attack_type_override=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  runs="$1"
  shift
fi
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  batch_dir_override="$1"
  shift
fi
while (( $# > 0 )); do
  option="$1"
  shift
  if (( $# == 0 )) || [[ "$1" == --* ]]; then
    echo "missing value for ${option}" >&2
    usage
    exit 2
  fi
  case "${option}" in
    --agent-name) agent_name_override="$1" ;;
    --task-index) task_index_override="$1" ;;
    --attacker-tool) attacker_tool_override="$1" ;;
    --attack-type) attack_type_override="$1" ;;
    *) echo "unknown option: ${option}" >&2; usage; exit 2 ;;
  esac
  shift
done
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  usage
  exit 2
fi
if [[ -n "${task_index_override}" ]] && ! [[ "${task_index_override}" =~ ^[0-9]+$ ]]; then
  echo "--task-index must be a non-negative integer" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_config="${ASB_CONFIG:-${repo_root}/agents/hermes/asb.yaml}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dry_run="${DRY_RUN:-0}"
skip_service_checks="${SKIP_SERVICE_CHECKS:-0}"
toolsafe_url="${TOOLSAFE_BASE_URL:-http://localhost:8001/v1}"
agentdog_url="${AGENTDOG_BASE_URL:-http://localhost:8002/v1}"

if [[ ! -f "${source_config}" ]]; then
  echo "ASB config does not exist: ${source_config}" >&2
  exit 2
fi
for flag in "${dry_run}" "${skip_service_checks}"; do
  if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
    echo "DRY_RUN and SKIP_SERVICE_CHECKS must be 0 or 1" >&2
    exit 2
  fi
done

case_slug="$(python3 - "${source_config}" "${agent_name_override}" "${task_index_override}" "${attacker_tool_override}" "${attack_type_override}" <<'PYCASE'
import pathlib, re, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
asb = raw.get("agent_security_bench") or {}
for key, value in zip(("agent_name", "task_index", "attacker_tool", "attack_type"), sys.argv[2:]):
    if value:
        asb[key] = int(value) if key == "task_index" else value
parts = [
    str(asb.get("agent_name") or "agent"),
    f"task-{asb.get('task_index', 0)}",
    str(asb.get("attacker_tool") or "tool"),
    str(asb.get("attack_type") or "attack"),
]
print("-".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", part).strip("-").lower() for part in parts))
PYCASE
)"
batch_dir="${batch_dir_override:-${repo_root}/jobs/asb_all_monitors/${case_slug}/${stamp}_hermes-asb-all-monitors}"
mkdir -p "${batch_dir}"
batch_dir="$(cd "${batch_dir}" && pwd)"
generated_config="$(mktemp "$(dirname "${source_config}")/.asb-all-monitors.XXXXXX.yaml")"
cleanup() { rm -f "${generated_config}"; }
trap cleanup EXIT

python3 - "${source_config}" "${generated_config}" "${toolsafe_url}" "${agentdog_url}" "${batch_dir}" "${repo_root}/scripts" "${agent_name_override}" "${task_index_override}" "${attacker_tool_override}" "${attack_type_override}" <<'PYCONFIG'
import pathlib, sys, yaml
source, destination, toolsafe_url, agentdog_url, batch_dir, scripts_dir = sys.argv[1:7]
sys.path.insert(0, scripts_dir)
from hermes_campaign import configure
raw = yaml.safe_load(pathlib.Path(source).read_text(encoding="utf-8"))
asb = raw.get("agent_security_bench") or {}
for key, value in zip(("agent_name", "task_index", "attacker_tool", "attack_type"), sys.argv[7:]):
    if value:
        asb[key] = int(value) if key == "task_index" else value
raw["agent_security_bench"] = asb
if not asb.get("enabled"):
    raise SystemExit("agent_security_bench.enabled must be true")
if asb.get("implementation") != "official_bridge":
    raise SystemExit("ASB all-monitors requires implementation: official_bridge")
if asb.get("injection_method") != "memory_attack":
    raise SystemExit("ASB all-monitors requires injection_method: memory_attack")
raw.setdefault("execution", {}).setdefault("hermes", {})["defense_mode"] = "replay"
raw.setdefault("memory_experiment", {}).update({
    "mode": "official_asb",
    "run_clean_control": True,
})
# Same guard set and passive modes as agents/hermes/agentdojo-all-monitors.yaml,
# with a fresh campaign-scoped AGrail memory; recorded in campaign_manifest.json.
configure(raw, pathlib.Path(batch_dir), source=source)
settings = {
    "toolsafe": {"base_url": toolsafe_url},
    "agentguard": {"policy": "", "plugin_config": ""},
    "agentdog": {"base_url": agentdog_url},
    "agentsight": {"enabled": False},
}
for name, values in settings.items():
    raw.setdefault(name, {}).update(values)
raw.setdefault("monitoring", {})["output_path"] = "trace_asb_all_monitors.json"
pathlib.Path(destination).write_text(
    yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PYCONFIG
cp "${generated_config}" "${batch_dir}/asb-all-monitors.yaml"

cat <<EOF
ASB all-monitors configuration
  Source: ${source_config}
  Case: ${case_slug}
  Runs: ${runs}
  Target + clean control per run: enabled
  Replay monitors: $(python3 -c 'import json,sys; print(", ".join(json.load(open(sys.argv[1]))["campaign"]["enabled_guards"]))' "${batch_dir}/campaign_manifest.json")
  ToolSafe: ${toolsafe_url}
  AgentDoG: ${agentdog_url}
  Output: ${batch_dir}
EOF

if [[ "${dry_run}" == "1" ]]; then
  echo "DRY_RUN=1: configuration generated; no runs executed."
  exit 0
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is unset." >&2
  exit 2
fi
export AGENTGUARD_SCENARIO_API_KEY="${AGENTGUARD_SCENARIO_API_KEY:-${OPENAI_API_KEY}}"

if [[ "${skip_service_checks}" != "1" ]]; then
  python3 - "${toolsafe_url}" "${agentdog_url}" "${batch_dir}/campaign_manifest.json" <<'PYCHECK'
import json, sys, urllib.request
enabled = set(json.load(open(sys.argv[3]))["campaign"]["enabled_guards"])
for guard, name, base in (("toolsafe", "ToolSafe", sys.argv[1]), ("agentdog", "AgentDoG", sys.argv[2])):
    if guard not in enabled:
        continue
    url = base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise SystemExit(f"{name} endpoint is unavailable at {url}: {exc}")
print("Local monitor endpoints are ready.")
PYCHECK
fi

PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${repo_root}/src/agent_scaffold/main.py" \
    --config "${generated_config}" \
    --runs "${runs}" \
    --runs-dir "${batch_dir}"
python3 "${repo_root}/scripts/hermes_campaign.py" finalize "${batch_dir}"

python3 "${repo_root}/scripts/analyze_hermes_asb_monitors.py" "${batch_dir}"
echo "Batch directory: ${batch_dir}"
echo "Plugin metrics: ${batch_dir}/analysis/plugin_metrics.json"
echo "Run labels: ${batch_dir}/analysis/run_labels.csv"
