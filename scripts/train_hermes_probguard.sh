#!/usr/bin/env bash
set -euo pipefail

# Capture disjoint clean and injected Hermes trajectories, then train the
# per-task ProbGuard model consumed by the Hermes all-monitors configs.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "usage: $0 [runs-per-condition] (default: 20)"
  exit 0
fi

runs="${1:-20}"
if (( $# > 1 )) || ! [[ "$runs" =~ ^[0-9]+$ ]] || (( 10#$runs < 2 )); then
  echo "usage: $0 [runs-per-condition>=2]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for Hermes runs and ProbGuard specification generation" >&2
  exit 2
fi
base_config="agents/hermes/agentdojo.yaml"
model_dir="models/pro2guard"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch_dir="$(mktemp -d "jobs/probguard_train_${stamp}.XXXXXX")"
clean_dir="$batch_dir/clean"
attack_dir="$batch_dir/attack"
clean_config="$(mktemp agents/hermes/.probguard-clean.XXXXXX.yaml)"
attack_config="$(mktemp agents/hermes/.probguard-attack.XXXXXX.yaml)"

cleanup() {
  rm -f "$clean_config" "$attack_config"
}
trap cleanup EXIT

PYTHONPATH=src "$python_bin" - "$base_config" "$clean_config" "$attack_config" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

from agent_scaffold.backends.guards import GUARDS

base = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
enabled = [name for name in GUARDS if (base.get(name) or {}).get("enabled")]
if enabled:
    raise SystemExit(f"Training capture config must have all guards disabled: {enabled}")
if base.get("execution", {}).get("backend") != "hermes":
    raise SystemExit("Training capture requires the Hermes backend")
if not base.get("agentdojo", {}).get("enabled"):
    raise SystemExit("Training capture requires AgentDojo")

for path, injected in ((sys.argv[2], False), (sys.argv[3], True)):
    config = copy.deepcopy(base)
    config["execution"]["hermes"]["defense_mode"] = "replay"
    config["agentdojo"]["injection_enabled"] = injected
    config["agentdojo"]["standard_injection_enabled"] = False
    config["agentdojo"]["skill_injection_enabled"] = injected
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Training data: $batch_dir"
echo "Capturing $runs clean runs"
PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
  --config "$clean_config" --runs "$runs" --runs-dir "$clean_dir"

echo "Capturing $runs injected runs"
PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
  --config "$attack_config" --runs "$runs" --runs-dir "$attack_dir"

PYTHONPATH=src "$python_bin" - "$clean_dir" "$attack_dir" "$runs" <<'PY'
import json
import sys
from pathlib import Path

from agent_scaffold.pro2guard.build_model import find_lifecycles, read_lifecycle
from agent_scaffold.pro2guard.generator import task_key

keys = set()
for directory in map(Path, sys.argv[1:3]):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("items", [])) != int(sys.argv[3]):
        raise SystemExit(f"Incomplete batch: {directory}")
    if any(not item.get("ok") for item in summary["items"]):
        raise SystemExit(f"Failed run in {directory}; review summary.json before training")
    lifecycles = find_lifecycles([str(directory)])
    if len(lifecycles) != int(sys.argv[3]):
        raise SystemExit(f"Expected {sys.argv[3]} lifecycles in {directory}, found {len(lifecycles)}")
    for path in lifecycles:
        record = read_lifecycle(path)
        if not record["steps"]:
            raise SystemExit(f"Training lifecycle has no completed tool steps: {path}")
        keys.add(task_key(record["task"], record["tools"], "task"))
if len(keys) != 1:
    raise SystemExit(f"Training batches contain different tasks: {sorted(keys)}")
print(f"Training task key: {next(iter(keys))}")
PY

echo "Training ProbGuard model"
PYTHONPATH=src "$python_bin" -m agent_scaffold.pro2guard.build_model \
  --config "$attack_config" --output "$model_dir" \
  "$clean_dir" "$attack_dir"

PYTHONPATH=src "$python_bin" - "$clean_dir" "$model_dir" "$runs" <<'PY'
import json
import sys
from pathlib import Path

from agent_scaffold.pro2guard.build_model import find_lifecycles, read_lifecycle
from agent_scaffold.pro2guard.generator import task_key

record = read_lifecycle(find_lifecycles([sys.argv[1]])[0])
key = task_key(record["task"], record["tools"], "task")
root = Path(sys.argv[2])
entry = json.loads((root / "index.json").read_text(encoding="utf-8"))["models"][key]
expected = 2 * int(sys.argv[3])
if entry["trace_count"] != expected:
    raise SystemExit(f"Model has {entry['trace_count']} traces, expected {expected}")
model_path = root / entry["dir"] / "model.json"
if not model_path.is_file():
    raise SystemExit(f"Model file missing for task {key}: {model_path}")
if entry["unsafe_state_count"] < 1:
    raise SystemExit(f"Model {key} has no learned unsafe state; review its abstraction and training traces")
print(f"Model ready: {model_path.parent} ({expected} traces, {entry['unsafe_state_count']} unsafe states)")
PY
