#!/usr/bin/env bash
# Restart from the recorded PID, or refuse to replace an untracked listener.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${RERANKER_MODEL_PATH:-$LAUNCH_DIR/../models/bge-reranker-v2-m3}
PID_FILE=${RERANKER_PID_FILE:-vllm_process_reranker.txt}
LOG_FILE=${RERANKER_LOG:-vllm_reranker.log}
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}

if [[ -s "$PID_FILE" ]]; then
  read -r pid < "$PID_FILE"
  [[ "$pid" =~ ^[0-9]+$ ]] || { echo "Invalid PID in $PID_FILE" >&2; exit 1; }
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Process $pid did not stop; inspect it before restarting" >&2
    exit 1
  fi
  sleep 3
fi
if curl -fsS --max-time 2 http://127.0.0.1:9024/v1/models >/dev/null 2>&1; then
  echo "Port 9024 is occupied. Set RERANKER_PID_FILE to its current PID file, or stop the old reranker first." >&2
  exit 1
fi

nohup env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name bge-reranker-v2-m3 --runner pooling \
  --host 0.0.0.0 --port 9024 --dtype half \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.06 \
  > "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
for _ in {1..180}; do
  if curl -fsS --max-time 2 http://127.0.0.1:9024/v1/models >/dev/null 2>&1; then
    echo "Reranker ready on port 9024; CUDA_VISIBLE_DEVICES=1; context=8192"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then tail -n 40 "$LOG_FILE"; exit 1; fi
  sleep 2
done
echo "Reranker is not ready; inspect $LOG_FILE" >&2
exit 1
