#!/usr/bin/env bash
# Restart from the recorded PID, or refuse to replace an untracked listener.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROCESS_HELPER=$SCRIPT_DIR/server_process.py

LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${RERANKER_MODEL_PATH:-$LAUNCH_DIR/../models/bge-reranker-v2-m3}
PID_FILE=${RERANKER_PID_FILE:-vllm_process_reranker.txt}
LOG_FILE=${RERANKER_LOG:-vllm_reranker.log}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${RERANKER_LAUNCH_LOG:-$LAUNCH_DIR/vllm_reranker_launcher.log}"
launcher_stage "activate vLLM environment"
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}

launcher_stage "stop old reranker and check port 9024"
python "$PROCESS_HELPER" stop --pid-file "$PID_FILE" --port 9024 --model bge-reranker-v2-m3
python "$PROCESS_HELPER" free --port 9024
sleep 3

launcher_stage "start reranker on port 9024"
reranker_command=(env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name bge-reranker-v2-m3 --runner pooling \
  --host 0.0.0.0 --port 9024 --dtype half \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.06)
launcher_command nohup "${reranker_command[@]}"
launcher_log "Model output: $LOG_FILE"
launcher_log "NEW RERANKER LAUNCH" >> "$LOG_FILE"
nohup "${reranker_command[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
if ! python "$PROCESS_HELPER" record --pid "$pid" --pid-file "$PID_FILE" --port 9024 --model bge-reranker-v2-m3; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
launcher_stage "wait for reranker endpoint on port 9024"
for _ in {1..180}; do
  if curl -fsS --max-time 2 http://127.0.0.1:9024/v1/models >/dev/null 2>&1; then
    launcher_stage "ready"
    echo "Reranker ready on port 9024; CUDA_VISIBLE_DEVICES=1; context=8192"
    exit 0
  fi
  if ! python "$PROCESS_HELPER" alive --pid "$pid"; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
  sleep 2
done
echo "Reranker is not ready; inspect $LOG_FILE" >&2
exit 1
