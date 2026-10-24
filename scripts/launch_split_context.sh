#!/usr/bin/env bash
# Run on the GPU server with its vLLM environment activated.
# Usage: VLLM_BIN=/path/to/vllm bash launch_split_context.sh MODEL_PATH MODEL_NAME
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROCESS_HELPER=$SCRIPT_DIR/server_process.py

MODEL_PATH=${1:?Pass the local checkpoint directory}
MODEL_NAME=${2:?Pass the served model name}
VLLM_BIN=${VLLM_BIN:-vllm}
NATIVE_PORT=${NATIVE_PORT:-9022}
YARN_PORT=${YARN_PORT:-9025}
MODEL_TAG=${MODEL_NAME//[^a-zA-Z0-9._-]/_}
NATIVE_PID_FILE=${NATIVE_PID_FILE:-vllm_${MODEL_TAG}_native.pid}
YARN_PID_FILE=${YARN_PID_FILE:-vllm_${MODEL_TAG}_yarn.pid}
NATIVE_LOG=${NATIVE_LOG:-vllm_${MODEL_TAG}_native.log}
YARN_LOG=${YARN_LOG:-vllm_${MODEL_TAG}_yarn.log}
YARN_CONFIG=${YARN_CONFIG:-vllm_${MODEL_TAG}_yarn_hf_overrides.json}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${LAUNCH_LOG:-vllm_${MODEL_TAG}_launcher.log}"
launcher_stage "read checkpoint config and prepare YaRN overrides"

# Read the actual installed Transformers config, preserving the checkpoint's
# RoPE theta. Do this before stopping an existing service.
python - "$MODEL_PATH" > "$YARN_CONFIG" <<'PY'
import json
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
config = config.get_text_config() if hasattr(config, "get_text_config") else config
if config.model_type != "qwen2" or config.max_position_embeddings != 32768:
    raise SystemExit("This launch profile requires a qwen2 checkpoint with native context 32768")
rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
if rope.get("rope_type", rope.get("type", "default")) != "default":
    raise SystemExit("The checkpoint already enables RoPE scaling; restore its native config first")
theta = rope.get("rope_theta", getattr(config, "rope_theta", None))
if theta is None:
    raise SystemExit("Cannot determine the checkpoint's rope_theta")
print(json.dumps({"rope_parameters": {"rope_type": "yarn", "factor": 2.0,
    "original_max_position_embeddings": 32768, "rope_theta": theta}}))
PY
launcher_log "YaRN overrides: $(cat "$YARN_CONFIG")"

wait_ready() {
  local port=$1 file=$2 log=$3 pid
  read -r pid < "$file"
  for _ in {1..180}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then return 0; fi
    if ! python "$PROCESS_HELPER" alive --pid "$pid"; then launcher_process_failed "$pid" "$log"; return 1; fi
    sleep 2
  done
  echo "Service on port $port is not ready; inspect $log" >&2
  return 1
}

launcher_stage "stop old native and YaRN services"
python "$PROCESS_HELPER" stop --pid-file "$NATIVE_PID_FILE" --port "$NATIVE_PORT" --model "$MODEL_NAME"
python "$PROCESS_HELPER" stop --pid-file "$YARN_PID_FILE" --port "$YARN_PORT" --model "$MODEL_NAME"
launcher_stage "check native and YaRN ports are free"
python "$PROCESS_HELPER" free --port "$NATIVE_PORT"
python "$PROCESS_HELPER" free --port "$YARN_PORT"
sleep 3

# Fractions refer to the entire GPU: 0.40 + 0.45 = 0.85 combined.
launcher_stage "start native Meno on port $NATIVE_PORT"
native_command=(env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" --host 0.0.0.0 --port "$NATIVE_PORT" \
  --dtype half --seed 42 --max-model-len 32768 --gpu-memory-utilization 0.40 \
  --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --enable-chunked-prefill \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
launcher_command nohup "${native_command[@]}"
launcher_log "Model output: $NATIVE_LOG"
launcher_log "NEW NATIVE LAUNCH" >> "$NATIVE_LOG"
nohup "${native_command[@]}" >> "$NATIVE_LOG" 2>&1 < /dev/null &
echo $! > "$NATIVE_PID_FILE"
if ! python "$PROCESS_HELPER" record --pid "$(cat "$NATIVE_PID_FILE")" --pid-file "$NATIVE_PID_FILE" --port "$NATIVE_PORT" --model "$MODEL_NAME"; then launcher_process_failed "$(cat "$NATIVE_PID_FILE")" "$NATIVE_LOG"; exit 1; fi
launcher_stage "wait for native endpoint on port $NATIVE_PORT"
wait_ready "$NATIVE_PORT" "$NATIVE_PID_FILE" "$NATIVE_LOG"

# Launch sequentially so profiling does not race between the two executors.
launcher_stage "start YaRN Meno on port $YARN_PORT"
yarn_command=(env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" --host 0.0.0.0 --port "$YARN_PORT" \
  --dtype half --seed 42 --max-model-len 65536 --gpu-memory-utilization 0.45 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enable-chunked-prefill \
  --hf-overrides "$(cat "$YARN_CONFIG")" \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
launcher_command nohup "${yarn_command[@]}"
launcher_log "Model output: $YARN_LOG"
launcher_log "NEW YARN LAUNCH" >> "$YARN_LOG"
nohup "${yarn_command[@]}" >> "$YARN_LOG" 2>&1 < /dev/null &
echo $! > "$YARN_PID_FILE"
if ! python "$PROCESS_HELPER" record --pid "$(cat "$YARN_PID_FILE")" --pid-file "$YARN_PID_FILE" --port "$YARN_PORT" --model "$MODEL_NAME"; then launcher_process_failed "$(cat "$YARN_PID_FILE")" "$YARN_LOG"; exit 1; fi
launcher_stage "wait for YaRN endpoint on port $YARN_PORT"
wait_ready "$YARN_PORT" "$YARN_PID_FILE" "$YARN_LOG"
launcher_stage "ready"
echo "Ready: native=$NATIVE_PORT, YaRN=$YARN_PORT. Use the client token-budget router."
