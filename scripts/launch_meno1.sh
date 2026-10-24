#!/usr/bin/env bash
# Switch the recorded generator to Meno-Lite-0.1; run from the vLLM launch folder.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${LAUNCH_LOG:-$LAUNCH_DIR/vllm_meno1_launcher.log}"
launcher_stage "activate vLLM environment"
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${MENO1_MODEL_PATH:-$LAUNCH_DIR/../models/meno-lite-0.1}
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}
export NATIVE_PID_FILE=${NATIVE_PID_FILE:-vllm_process_meno1.txt}
export YARN_PID_FILE=${YARN_PID_FILE:-vllm_process_meno1_yarn.txt}
export NATIVE_LOG=${NATIVE_LOG:-vllm_meno1.log}
export YARN_LOG=${YARN_LOG:-vllm_meno1_long.log}
exec bash "$SCRIPT_DIR/launch_split_context.sh" "$MODEL_PATH" meno-lite-0.1
