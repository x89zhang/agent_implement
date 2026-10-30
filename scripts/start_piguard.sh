#!/usr/bin/env bash
# Start PIGuard with mini_swe: scripts/start_piguard.sh [server options]
# Example: GPU_ID=1 PORT=8001 scripts/start_piguard.sh
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-"$HOME/envs/mini_swe/bin/python"}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}

if [[ ! -x "$python_bin" ]]; then
  echo "mini_swe Python not found: $python_bin (override with PYTHON_BIN)" >&2
  exit 2
fi

exec "$python_bin" "$script_dir/serve_piguard.py" \
  --model "${MODEL_PATH:-/mnt/data/xiaoliang/models/PIGuard}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8001}" \
  "$@"
