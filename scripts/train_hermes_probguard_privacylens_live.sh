#!/usr/bin/env bash
set -euo pipefail

# Train one ProbGuard task model for the PrivacyLens-Live case selected in
# agents/hermes/privacylens-live.yaml. Keep these runs separate from evaluation.

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "usage: $0 [runs] (default: 20)"
  exit 0
fi

runs="${1:-20}"
if (( $# > 1 )) || ! [[ "$runs" =~ ^[0-9]+$ ]] || (( 10#$runs < 2 )); then
  echo "usage: $0 [runs>=2]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
base_config="agents/hermes/privacylens-live.yaml"
model_root="models/pro2guard"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for Hermes, the PrivacyLens judge, and ProbGuard specification generation" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
batch_dir="$(mktemp -d "jobs/probguard_privacylens_live_train_${stamp}.XXXXXX")"
train_config="$(mktemp agents/hermes/.probguard-privacylens-live.XXXXXX.yaml)"
trap 'rm -f "$train_config"' EXIT

PYTHONPATH=src "$python_bin" - "$base_config" "$train_config" <<'PY'
import sys
from pathlib import Path

import yaml
from agent_scaffold.backends.guards import GUARDS

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
enabled = [name for name in GUARDS if (config.get(name) or {}).get("enabled")]
if enabled:
    raise SystemExit(f"Training requires all guards disabled in PrivacyLens config: {enabled}")
if config.get("execution", {}).get("backend") != "hermes":
    raise SystemExit("Training requires the Hermes backend")
benchmark = config.get("privacylens_live") or {}
if not benchmark.get("enabled") or benchmark.get("implementation") != "official_bridge":
    raise SystemExit("Training requires an enabled PrivacyLens-Live official_bridge")
if config.get("memory_experiment", {}).get("mode") != "off":
    raise SystemExit("Training requires memory_experiment.mode: off")
config["execution"].setdefault("hermes", {})["defense_mode"] = "replay"
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Training data: $batch_dir"
PYTHONPATH=src "$python_bin" src/agent_scaffold/main.py \
  --config "$train_config" --runs "$runs" --runs-dir "$batch_dir"

# Select only the live target phase; recursive collection might include replay
# lifecycles if a future runner starts writing them under the same batch.
traces=()
for run_dir in "$batch_dir"/run_*; do
  [[ -d "$run_dir" ]] || continue
  path="$run_dir/target/guard_lifecycle.jsonl"
  [[ -s "$path" ]] || { echo "Missing target lifecycle: $path" >&2; exit 1; }
  traces+=("$path")
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
    raise SystemExit(f"Incomplete PrivacyLens batch: {batch}/summary.json")
if len(traces) != runs:
    raise SystemExit(f"Expected {runs} target lifecycles, found {len(traces)}")
keys = set()
for path in traces:
    record = read_lifecycle(path)
    if not record["steps"]:
        raise SystemExit(f"Training lifecycle has no completed tool steps: {path}")
    keys.add(task_key(record["task"], record["tools"], "task"))
if len(keys) != 1:
    raise SystemExit(f"Training lifecycles have different task keys: {sorted(keys)}")
print(f"Training task key: {next(iter(keys))}")
PY

staging="$batch_dir/model-staging"
echo "Training ProbGuard model from ${#traces[@]} PrivacyLens lifecycles"
PYTHONPATH=src "$python_bin" -m agent_scaffold.pro2guard.build_model \
  --config "$train_config" --output "$staging" "${traces[@]}"

PYTHONPATH=src "$python_bin" - "$base_config" "$staging" "$model_root" "$runs" <<'PY'
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
expected = int(sys.argv[4])
models = json.loads((staging / "index.json").read_text(encoding="utf-8"))["models"]
if len(models) != 1:
    raise SystemExit(f"Expected one PrivacyLens task model, found {len(models)}")
key, entry = next(iter(models.items()))
if entry["trace_count"] != expected:
    raise SystemExit(f"Model has {entry['trace_count']} traces; expected {expected}")
if entry["unsafe_state_count"] < 1:
    raise SystemExit(f"Model {key} has no learned unsafe state; review its abstraction and training traces")
case = str(config["privacylens_live"]["case"])
case_slug = re.sub(r"[^a-z0-9_-]+", "-", case.lower()).strip("-_" )[:48]
name = f"privacylens-live-{case_slug}-{key}"
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
