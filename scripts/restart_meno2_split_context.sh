#!/usr/bin/env bash
# Run from the server's vLLM launch directory; all dependencies are in this repo.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCH_DIR=${VLLM_LAUNCH_DIR:-$PWD}
source "$SCRIPT_DIR/server_logging.sh"
launcher_logging_init "${LAUNCH_LOG:-$LAUNCH_DIR/vllm_meno2_launcher.log}"
launcher_stage "activate vLLM environment"
VLLM_ENV_DIR=${VLLM_ENV_DIR:-$LAUNCH_DIR/vllm_0_29_0}
MODEL_PATH=${MENO2_MODEL_PATH:-$LAUNCH_DIR/../models/meno-lite-0.2}
cd "$LAUNCH_DIR"
source "$VLLM_ENV_DIR/bin/activate"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BIN=${VLLM_BIN:-$VLLM_ENV_DIR/bin/vllm}
export NATIVE_PID_FILE=${NATIVE_PID_FILE:-vllm_process_meno2.txt}
export YARN_PID_FILE=${YARN_PID_FILE:-vllm_process_meno2_yarn.txt}
export NATIVE_LOG=${NATIVE_LOG:-vllm_meno2.log}
export YARN_LOG=${YARN_LOG:-vllm_meno2_yarn.log}
exec bash "$SCRIPT_DIR/launch_split_context.sh" "$MODEL_PATH" meno-lite-0.2
