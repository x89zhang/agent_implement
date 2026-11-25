#!/usr/bin/env bash
# Add every monitor to monitor-off Hermes batches end to end, starting and
# stopping the vLLM servers the local-model monitors need:
#   round janus-toolsafe: Vanguard (JANUS, port 8000) + TS-Guard (ToolSafe, port 8001)
#   round stepguard:      StepGuard (port 8000); PIGuard runs on CPU alongside
# The two rounds cannot share GPUs (the three models do not fit on two 22 GB
# GPUs, and JANUS and StepGuard both use port 8000), so servers are swapped.
#
#   asb       batches-file  scripts/complete_hermes_asb_screen.sh: the 8 CPU monitors,
#                           JANUS and ToolSafe, then StepGuard and PIGuard
#   mcp_safety_bench batches-file  the same, for MCP-SafetyBench screen batches
#   agentdojo pairs-file    scripts/supplement_hermes_pairs.sh: JANUS and ToolSafe
#                           (retrying errors), then StepGuard and PIGuard
#
# Ports 8000 and 8001 must be free at the start: stop servers you started
# yourself (Ctrl-C) first. The servers this script starts are stopped when it
# exits, also on Ctrl-C; their logs go to jobs/vllm/. Each server gets a GPU with
# enough free memory (VLLM_GPUS limits the candidates, e.g. VLLM_GPUS="0 1").
set -euo pipefail

usage() {
  echo "usage: $0 asb|mcp_safety_bench batches-file | agentdojo pairs-file" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# != 2 )) || [[ "$1" != asb && "$1" != mcp_safety_bench && "$1" != agentdojo ]] || [[ ! -f "$2" ]]; then
  usage
  exit 2
fi
benchmark="$1"
list_file="$2"

vllm_python="/home/xiaoliang_zhang/envs/vllm/bin/python"
models="/mnt/data/xiaoliang/models"
log_dir="jobs/vllm"
# MiB a server needs free on its GPU (0.90 of a 22 GiB GPU, plus margin).
need_mib=20500
ready_timeout=900

# name | served model | port | model dir | extra vLLM arguments
declare -A served=([vanguard]="yuaay/vanguard" [tsguard]="TS-Guard" [stepguard]="StepGuard")
declare -A port=([vanguard]=8000 [tsguard]=8001 [stepguard]=8000)
declare -A model_dir=([vanguard]="${models}/vanguard" [tsguard]="${models}/TS-Guard" [stepguard]="${models}/StepGuard")
declare -A extra=(
  [vanguard]=""
  [tsguard]="--tensor-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 4096 --enable-prefix-caching"
  [stepguard]=""
)

declare -A pid   # running servers this script started -> process id (group leader)
declare -A gpu   # -> the GPU it was given
declare -A log   # -> its log file

serving() {  # serving PORT -> the model ids served there, or nothing
  curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true
}

stop_server() {  # stop a server this script started, with its engine processes
  local name="$1"
  [[ -n "${pid[$name]:-}" ]] || return 0
  local p="${pid[$name]}"
  unset "pid[$name]" "gpu[$name]"
  kill -0 "${p}" 2>/dev/null || return 0
  echo "### stopping ${name} (pid ${p})"
  kill -TERM -- "-${p}" 2>/dev/null || true
  for _ in $(seq 60); do
    kill -0 "${p}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${p}" 2>/dev/null || true
}

cleanup() {
  local name
  for name in "${!pid[@]}"; do
    stop_server "${name}"
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

free_gpu() {  # a GPU with need_mib free that this script has not handed out
  local index free
  while IFS=', ' read -r index free; do
    [[ -n "${VLLM_GPUS:-}" && " ${VLLM_GPUS} " != *" ${index} "* ]] && continue
    [[ " ${gpu[*]:-} " == *" ${index} "* ]] && continue
    if (( free >= need_mib )); then
      echo "${index}"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

start_server() {
  local name="$1" current device
  current="$(serving "${port[$name]}")"
  if [[ -n "${current}" ]]; then
    echo "port ${port[$name]} already serves ${current}; stop that server first (Ctrl-C in its terminal)" >&2
    exit 1
  fi
  if ! device="$(free_gpu)"; then
    echo "no GPU has ${need_mib} MiB free for ${name}:" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
    exit 1
  fi
  gpu[$name]="${device}"
  mkdir -p "${log_dir}"
  log[$name]="${log_dir}/${name}-$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "### starting ${name} on GPU ${device}, port ${port[$name]} (log: ${log[$name]})"
  # setsid: the server and its engine processes form one group, stopped together.
  # The FlashInfer sampler needs a JIT build that fails with the system nvcc.
  # shellcheck disable=SC2086
  PATH="$(dirname "${vllm_python}"):${PATH}" VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="${device}" \
    setsid "${vllm_python}" -m vllm.entrypoints.openai.api_server \
      --model "${model_dir[$name]}" --served-model-name "${served[$name]}" \
      --host 127.0.0.1 --port "${port[$name]}" --max-model-len 16384 --gpu-memory-utilization 0.90 \
      ${extra[$name]} > "${log[$name]}" 2>&1 &
  pid[$name]=$!
}

wait_ready() {
  local name="$1" waited=0
  until [[ " $(serving "${port[$name]}") " == *" ${served[$name]} "* ]]; do
    if ! kill -0 "${pid[$name]}" 2>/dev/null; then
      echo "${name} exited during startup; last lines of its log:" >&2
      tail -n 30 "${log[$name]}" >&2
      exit 1
    fi
    if (( waited >= ready_timeout )); then
      echo "${name} not ready after ${ready_timeout}s" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "### ${name} ready"
}

run_round() {  # run_round label command...: a failed round is reported, not fatal
  local label="$1"
  shift
  echo "### round ${label}: $*"
  if "$@"; then
    results+=("ok ${label}")
  else
    results+=("FAILED ${label} (exit $?)")
  fi
}

if [[ "${benchmark}" == asb || "${benchmark}" == mcp_safety_bench ]]; then
  round1=(scripts/complete_hermes_asb_screen.sh "${list_file}")
  round2=(scripts/complete_hermes_asb_screen.sh "${list_file}" stepguard piguard)
else
  round1=(scripts/supplement_hermes_pairs.sh "${list_file}" janus toolsafe)
  round2=(scripts/supplement_hermes_pairs.sh "${list_file}" stepguard piguard)
fi

results=()
start_server vanguard
start_server tsguard
wait_ready vanguard
wait_ready tsguard
run_round janus-toolsafe "${round1[@]}"
stop_server vanguard
stop_server tsguard

start_server stepguard
wait_ready stepguard
run_round stepguard "${round2[@]}"
stop_server stepguard

echo
echo "### rounds"
printf '%s\n' "${results[@]}"
