#!/usr/bin/env bash
# Serve a local AlignmentCheck judge for LlamaFirewall supplement replays.
#
#   scripts/serve_llamafirewall_judge.sh [model-dir] [port]
#
# Uses both L4 GPUs (tensor parallel 2), so stop the TS-Guard (8001) and
# AgentDoG (8002) servers first. The served name must match
# agents/hermes/llamafirewall-local.yaml (factory_kwargs.model).
set -euo pipefail

model=${1:-/mnt/data/xiaoliang/models/Qwen2.5-32B-Instruct-AWQ}
port=${2:-8003}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -d "$model" ]]; then
  echo "model directory not found: $model" >&2
  exit 2
fi

# AlignmentCheck renders the whole trace, including tool outputs, into one
# prompt; 16k tokens covers the recorded AgentDojo trajectories.
exec "$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name alignment-judge \
  --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
