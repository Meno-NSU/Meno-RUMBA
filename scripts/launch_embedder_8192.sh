#!/usr/bin/env bash
# Dense BGE-M3 embeddings on the same GPU as the benchmark services.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROCESS_HELPER=$SCRIPT_DIR/server_process.py

LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${EMBEDDER_MODEL_PATH:-$LAUNCH_DIR/../models/bge-m3}
PID_FILE=${EMBEDDER_PID_FILE:-vllm_process_embedder.txt}
LOG_FILE=${EMBEDDER_LOG:-vllm_embedder.log}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${EMBEDDER_LAUNCH_LOG:-$LAUNCH_DIR/vllm_embedder_launcher.log}"
launcher_stage "activate vLLM environment"
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}

launcher_stage "stop old embedder and check port 9023"
python "$PROCESS_HELPER" stop --pid-file "$PID_FILE" --port 9023 --model bge-m3
python "$PROCESS_HELPER" free --port 9023
sleep 3

launcher_stage "start embedder on port 9023"
embedder_command=(env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name bge-m3 --runner pooling \
  --pooler-config '{"task":"embed","pooling_type":"CLS","use_activation":true}' \
  --host 0.0.0.0 --port 9023 --dtype half \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.04)
launcher_command nohup "${embedder_command[@]}"
launcher_log "Model output: $LOG_FILE"
launcher_log "NEW EMBEDDER LAUNCH" >> "$LOG_FILE"
nohup "${embedder_command[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
if ! python "$PROCESS_HELPER" record --pid "$pid" --pid-file "$PID_FILE" --port 9023 --model bge-m3; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
launcher_stage "wait for embedder endpoint on port 9023"
for _ in {1..180}; do
  if curl -fsS --max-time 2 http://127.0.0.1:9023/v1/models >/dev/null 2>&1; then
    launcher_stage "ready"
    echo "Embedder ready on port 9023; model=bge-m3; CUDA_VISIBLE_DEVICES=1; context=8192"
    exit 0
  fi
  if ! python "$PROCESS_HELPER" alive --pid "$pid"; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
  sleep 2
done
echo "Embedder is not ready; inspect $LOG_FILE" >&2
exit 1
