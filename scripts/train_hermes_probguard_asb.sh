#!/usr/bin/env bash
set -euo pipefail

# Train one ProbGuard task model for the ASB case selected in agents/hermes/asb.yaml.
# Each official_asb run produces an attacked target and a clean control.

usage() {
  echo "usage: $0 [runs>=2] [--agent-name NAME] [--task-index N] [--attacker-tool TOOL] [--attack-type TYPE]" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
runs=20
agent_name_override=""
task_index_override=""
attacker_tool_override=""
attack_type_override=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  runs="$1"
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
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( 10#$runs < 2 )); then
  usage
  exit 2
fi
if [[ -n "$task_index_override" ]] && ! [[ "$task_index_override" =~ ^[0-9]+$ ]]; then
  echo "--task-index must be a non-negative integer" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
base_config="agents/hermes/asb.yaml"
model_root="models/pro2guard"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for Hermes, ASB memory retrieval, and ProbGuard specification generation" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch_dir="$(mktemp -d "jobs/probguard_asb_train_${stamp}.XXXXXX")"
train_config="$(mktemp agents/hermes/.probguard-asb.XXXXXX.yaml)"
trap 'rm -f "$train_config"' EXIT

PYTHONPATH=src "$python_bin" - "$base_config" "$train_config" "$agent_name_override" "$task_index_override" "$attacker_tool_override" "$attack_type_override" <<'PY'
import sys
from pathlib import Path

import yaml
from agent_scaffold.backends.guards import GUARDS

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
asb = config.setdefault("agent_security_bench", {})
for key, value in zip(("agent_name", "task_index", "attacker_tool", "attack_type"), sys.argv[3:]):
    if value:
        asb[key] = int(value) if key == "task_index" else value
enabled = [name for name in GUARDS if (config.get(name) or {}).get("enabled")]
if enabled:
    raise SystemExit(f"Training requires all guards disabled in ASB config: {enabled}")
if config.get("execution", {}).get("backend") != "hermes":
    raise SystemExit("Training requires the Hermes backend")
if not config.get("agent_security_bench", {}).get("enabled"):
    raise SystemExit("Training requires agent_security_bench.enabled: true")
if config["agent_security_bench"].get("implementation") != "official_bridge":
    raise SystemExit("Training requires agent_security_bench.implementation: official_bridge")
if config["agent_security_bench"].get("injection_method") != "memory_attack":
    raise SystemExit("Training requires agent_security_bench.injection_method: memory_attack")
if config.get("memory_experiment", {}).get("mode") != "official_asb":
    raise SystemExit("Training requires memory_experiment.mode: official_asb")
config["execution"].setdefault("hermes", {})["defense_mode"] = "replay"
config["memory_experiment"]["run_clean_control"] = True
config["agent_security_bench"]["official_memory_enabled"] = True
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Training data: $batch_dir"
PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
  --config "$train_config" --runs "$runs" --runs-dir "$batch_dir"

# Use only target and control phases. Recursing over the batch directory could
# pick up replay or other lifecycles and would silently change training counts.
traces=()
for run_dir in "$batch_dir"/run_*; do
  [[ -d "$run_dir" ]] || continue
  for phase in target control; do
    path="$run_dir/$phase/guard_lifecycle.jsonl"
    [[ -s "$path" ]] || { echo "Missing $phase lifecycle: $path" >&2; exit 1; }
    traces+=("$path")
  done
done

PYTHONPATH=src "$python_bin" - "$batch_dir" "$runs" "${traces[@]}" <<'PY'
import json
import sys
from pathlib import Path

from agent_scaffold.pro2guard.build_model import read_lifecycle
from agent_scaffold.pro2guard.generator import task_key

batch = Path(sys.argv[1])
runs = int(sys.argv[2])
traces = [Path(value) for value in sys.argv[3:]]
summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
if len(summary.get("items", [])) != runs or any(not item.get("ok") for item in summary["items"]):
    raise SystemExit(f"Incomplete ASB batch: {batch}/summary.json")
if len(traces) != 2 * runs:
    raise SystemExit(f"Expected {2 * runs} target/control lifecycles, found {len(traces)}")
keys = set()
for path in traces:
    record = read_lifecycle(path)
    if not record["steps"]:
        raise SystemExit(f"Training lifecycle has no completed tool steps: {path}")
    keys.add(task_key(record["task"], record["tools"], "task"))
if len(keys) != 1:
    raise SystemExit(f"Target and control lifecycles have different task keys: {sorted(keys)}")
print(f"Training task key: {next(iter(keys))}")
PY

staging="$batch_dir/model-staging"
echo "Training ProbGuard model from ${#traces[@]} ASB lifecycles"
PYTHONPATH=src "$python_bin" -m agent_scaffold.pro2guard.build_model \
  --config "$train_config" --output "$staging" "${traces[@]}"

mkdir -p "$model_root"
PYTHONPATH=src flock -x "$model_root/.train.lock" "$python_bin" - "$train_config" "$staging" "$model_root" "$runs" <<'PY'
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
staging = Path(sys.argv[2])
root = Path(sys.argv[3])
expected = 2 * int(sys.argv[4])
models = json.loads((staging / "index.json").read_text(encoding="utf-8"))["models"]
if len(models) != 1:
    raise SystemExit(f"Expected one ASB task model, found {len(models)}")
key, entry = next(iter(models.items()))
if entry["trace_count"] != expected:
    raise SystemExit(f"Model has {entry['trace_count']} traces; expected {expected}")
if entry["unsafe_state_count"] < 1:
    raise SystemExit(f"Model {key} has no learned unsafe state; review its abstraction and training traces")
case = config["agent_security_bench"]
agent = re.sub(r"[^a-z0-9_-]+", "-", str(case["agent_name"]).lower()).strip("-_" )[:48]
name = f"asb-{agent}-task-{case['task_index']}-{key}"
root.mkdir(parents=True, exist_ok=True)
index_path = root / "index.json"
index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
index.setdefault("models", {})
if key in index["models"] or (root / name).exists():
    raise SystemExit(f"Model already exists for task key {key}; refusing to overwrite it")
source = staging / entry["dir"]
if not all((source / file).is_file() for file in ("model.json", "abstraction.json", "dtmc.prism")):
    raise SystemExit(f"Incomplete trained model: {source}")
shutil.move(str(source), str(root / name))
entry["dir"] = name
index["models"][key] = entry
temporary = root / f".index.{os.getpid()}.json"
try:
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, index_path)
finally:
    temporary.unlink(missing_ok=True)
print(f"Model ready: {root / name} ({expected} traces, {entry['unsafe_state_count']} unsafe states)")
PY
