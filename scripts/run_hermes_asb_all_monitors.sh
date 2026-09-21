#!/usr/bin/env bash
set -euo pipefail

runs="${1:-20}"
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  echo "usage: $0 [runs>=2] [output-directory]" >&2
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

case_slug="$(python3 - "${source_config}" <<'PYCASE'
import pathlib, re, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
asb = raw.get("agent_security_bench") or {}
parts = [
    str(asb.get("agent_name") or "agent"),
    f"task-{asb.get('task_index', 0)}",
    str(asb.get("attacker_tool") or "tool"),
    str(asb.get("attack_type") or "attack"),
]
print("-".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", part).strip("-").lower() for part in parts))
PYCASE
)"
batch_dir="${2:-${repo_root}/jobs/asb_all_monitors/${case_slug}/${stamp}_hermes-asb-all-monitors}"
mkdir -p "${batch_dir}"
batch_dir="$(cd "${batch_dir}" && pwd)"
generated_config="$(mktemp "$(dirname "${source_config}")/.asb-all-monitors.XXXXXX.yaml")"
cleanup() { rm -f "${generated_config}"; }
trap cleanup EXIT

python3 - "${source_config}" "${generated_config}" "${toolsafe_url}" "${agentdog_url}" <<'PYCONFIG'
import pathlib, sys, yaml
source, destination, toolsafe_url, agentdog_url = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(source).read_text(encoding="utf-8"))
asb = raw.get("agent_security_bench") or {}
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
settings = {
    "aegis": {"enabled": True, "mode": "monitor"},
    "progent": {"enabled": True, "mode": "monitor"},
    "pro2guard": {"enabled": True, "mode": "monitor"},
    "agentspec": {"enabled": True, "mode": "monitor"},
    "llamafirewall": {"enabled": True, "mode": "monitor"},
    "toolsafe": {
        "enabled": True,
        "mode": "monitor",
        "base_url": toolsafe_url,
    },
    "agentguard": {
        "enabled": True,
        "mode": "monitor",
        "policy": "",
        "plugin_config": "",
    },
    "agentdog": {
        "enabled": True,
        "mode": "diagnose",
        "base_url": agentdog_url,
    },
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
  Replay monitors: aegis, progent, pro2guard, agentspec, llamafirewall, toolsafe, agentguard, agentdog
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
  python3 - "${toolsafe_url}" "${agentdog_url}" <<'PYCHECK'
import sys, urllib.request
for name, base in (("ToolSafe", sys.argv[1]), ("AgentDoG", sys.argv[2])):
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

python3 "${repo_root}/scripts/analyze_hermes_asb_monitors.py" "${batch_dir}"
echo "Batch directory: ${batch_dir}"
echo "Plugin metrics: ${batch_dir}/analysis/plugin_metrics.json"
echo "Run labels: ${batch_dir}/analysis/run_labels.csv"
