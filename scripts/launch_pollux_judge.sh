#!/usr/bin/env bash
# Replace the benchmark inference services with the independent generative judge.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROCESS_HELPER=$SCRIPT_DIR/server_process.py

LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${JUDGE_MODEL_PATH:-$LAUNCH_DIR/../models/pollux-judge-32b}
PID_FILE=${JUDGE_PID_FILE:-vllm_process_pollux.txt}
LOG_FILE=${JUDGE_LOG:-vllm_pollux.log}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${JUDGE_LAUNCH_LOG:-$LAUNCH_DIR/vllm_pollux_launcher.log}"
launcher_stage "activate vLLM environment"
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}

# Validate the local checkpoint before stopping any running services. The -r
# regression variant cannot produce the official RUMBA textual verdict.
launcher_stage "validate generative Pollux checkpoint"
python - "$MODEL_PATH" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
if config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("model_type") != "qwen2":
    raise SystemExit("Expected generative pollux-judge-32b / Qwen2ForCausalLM, not the -r regression model")
rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
if rope.get("rope_type", rope.get("type", "default")) != "default":
    raise SystemExit("Pollux judging requires the native checkpoint without RoPE scaling")
if config.get("max_position_embeddings", 0) < 8192:
    raise SystemExit("The checkpoint must support at least 8192 native tokens")
print(f"Pollux checkpoint: native_context={config['max_position_embeddings']}; served_context=8192; no YaRN", flush=True)
PY

launcher_stage "stop all recorded benchmark generators (native and long context)"
python "$PROCESS_HELPER" stop-benchmark --launch-dir "$PWD" --port 9022 --long-port 9025
launcher_stage "stop embedder, reranker and previous Pollux"
python "$PROCESS_HELPER" stop --pid-file "${EMBEDDER_PID_FILE:-vllm_process_embedder.txt}" --port 9023 --model bge-m3
python "$PROCESS_HELPER" stop --pid-file "${RERANKER_PID_FILE:-vllm_process_reranker.txt}" --port 9024 --model bge-reranker-v2-m3
python "$PROCESS_HELPER" stop --pid-file "$PID_FILE" --port 9026 --model pollux-judge-32b
launcher_stage "check inference and judge ports are free"
for port in 9022 9023 9024 9025 9026; do
  python "$PROCESS_HELPER" free --port "$port"
done
sleep 3
launcher_snapshot

launcher_stage "start Pollux judge on port 9026"
judge_command=(env CUDA_VISIBLE_DEVICES=1 "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name pollux-judge-32b --host 0.0.0.0 --port 9026 \
  --dtype bfloat16 --seed 42 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 4 \
  --enable-prefix-caching --enable-chunked-prefill \
  --generation-config vllm \
  --override-generation-config '{"temperature":0,"top_p":1,"top_k":-1,"min_p":0,"repetition_penalty":1}')
launcher_command nohup "${judge_command[@]}"
launcher_log "Model output: $LOG_FILE"
launcher_log "NEW POLLUX JUDGE LAUNCH" >> "$LOG_FILE"
nohup "${judge_command[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
if ! python "$PROCESS_HELPER" record --pid "$pid" --pid-file "$PID_FILE" --port 9026 --model pollux-judge-32b; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
launcher_stage "wait for Pollux endpoint on port 9026"
for _ in {1..300}; do
  if curl -fsS --max-time 2 http://127.0.0.1:9026/v1/models >/dev/null 2>&1; then
    launcher_stage "ready"
    echo "Pollux judge ready on port 9026; model=pollux-judge-32b; CUDA_VISIBLE_DEVICES=1; context=8192; dtype=bfloat16"
    exit 0
  fi
  if ! python "$PROCESS_HELPER" alive --pid "$pid"; then launcher_process_failed "$pid" "$LOG_FILE"; exit 1; fi
  sleep 2
done
echo "Pollux is not ready; inspect $LOG_FILE and $LAUNCH_LOG" >&2
exit 1
