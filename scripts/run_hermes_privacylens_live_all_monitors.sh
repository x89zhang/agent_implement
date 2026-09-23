#!/usr/bin/env bash
# Capture each Hermes run once, then replay each monitor against that same trajectory.
set -euo pipefail

runs="${1:-20}"
if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 2 )); then
  echo "usage: $0 [runs>=2] [output-directory]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_config="${PRIVACYLENS_CONFIG:-${repo_root}/agents/hermes/privacylens-live.yaml}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dry_run="${DRY_RUN:-0}"
skip_service_checks="${SKIP_SERVICE_CHECKS:-0}"
toolsafe_url="${TOOLSAFE_BASE_URL:-http://localhost:8001/v1}"
agentdog_url="${AGENTDOG_BASE_URL:-http://localhost:8002/v1}"

if [[ ! -f "${source_config}" ]]; then
  echo "PrivacyLens config does not exist: ${source_config}" >&2
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
case = str((raw.get("privacylens_live") or {}).get("case") or "item1")
print(re.sub(r"[^A-Za-z0-9_.-]+", "-", case).strip("-").lower())
PYCASE
)"
batch_dir="${2:-${repo_root}/jobs/privacylens_live_all_monitors/${case_slug}/${stamp}_hermes-privacylens-live-all-monitors}"
if [[ -e "${batch_dir}" ]]; then
  echo "Output directory already exists: ${batch_dir}" >&2
  exit 2
fi
mkdir -p "${batch_dir}"
batch_dir="$(cd "${batch_dir}" && pwd)"

# Keep this beside the source YAML so relative paths resolve the same way.
generated_config="$(mktemp "$(dirname "${source_config}")/.privacylens-all-monitors.XXXXXX.yaml")"
cleanup() { rm -f "${generated_config}"; }
trap cleanup EXIT

python3 - "${source_config}" "${generated_config}" "${toolsafe_url}" "${agentdog_url}" <<'PYCONFIG'
import pathlib, sys, yaml

source, destination, toolsafe_url, agentdog_url = sys.argv[1:]
raw = yaml.safe_load(pathlib.Path(source).read_text(encoding="utf-8"))
privacy = raw.get("privacylens_live") or {}
if not privacy.get("enabled") or privacy.get("implementation") != "official_bridge":
    raise SystemExit("PrivacyLens all-monitors requires enabled official_bridge")
if privacy.get("evaluation_mode") != "privacylens_official_protocol":
    raise SystemExit("PrivacyLens all-monitors requires privacylens_official_protocol evaluation")
if (raw.get("execution") or {}).get("backend") != "hermes":
    raise SystemExit("PrivacyLens all-monitors requires the Hermes backend")
if (raw.get("memory_experiment") or {}).get("mode", "off") != "off":
    raise SystemExit("PrivacyLens all-monitors requires memory_experiment.mode: off")

# The backend captures an unguarded lifecycle, then creates a fresh controller,
# deep-copied configuration, and output directory for each enabled monitor.
raw.setdefault("execution", {}).setdefault("hermes", {})["defense_mode"] = "replay"
settings = {
    "aegis": {"enabled": True, "mode": "monitor"},
    "progent": {"enabled": True, "mode": "monitor"},
    "pro2guard": {"enabled": True, "mode": "monitor"},
    "agentspec": {"enabled": True, "mode": "monitor"},
    "llamafirewall": {"enabled": True, "mode": "monitor"},
    "toolsafe": {"enabled": True, "mode": "monitor", "base_url": toolsafe_url},
    "agentguard": {"enabled": True, "mode": "monitor", "policy": "", "plugin_config": ""},
    "agentdog": {"enabled": True, "mode": "diagnose", "base_url": agentdog_url},
    "agentsight": {"enabled": False},
}
for name, values in settings.items():
    raw.setdefault(name, {}).update(values)
raw.setdefault("monitoring", {})["output_path"] = "trace_privacylens_live_all_monitors.json"
container_env = raw.setdefault("container", {}).setdefault("env", [])
for name in ("AGENTGUARD_SCENARIO_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    if name not in container_env:
        container_env.append(name)
pathlib.Path(destination).write_text(
    yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PYCONFIG
cp "${generated_config}" "${batch_dir}/privacylens-live-all-monitors.yaml"

cat <<EOF
PrivacyLens-Live all-monitors configuration
  Source: ${source_config}
  Case: ${case_slug}
  Runs: ${runs}
  Capture: unguarded Hermes trajectory; one isolated replay per plugin
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

python3 "${repo_root}/scripts/analyze_hermes_privacylens_live_monitors.py" "${batch_dir}"
echo "Batch directory: ${batch_dir}"
echo "Plugin metrics: ${batch_dir}/analysis/plugin_metrics.json"
