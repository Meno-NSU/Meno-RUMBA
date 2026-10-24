#!/usr/bin/env bash
# This partition allocates whole nodes and has no GPU GRES configured.
#SBATCH --job-name=rumba-meno-pair
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-meno-pair-%j.log
set -euo pipefail
umask 077
cd "${MENO_RUMBA_PROJECT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
: "${SLURM_JOB_ID:?Use sbatch, not a login-node launch}"
export CLUSTER_LOCAL_SCRATCH=${CLUSTER_LOCAL_SCRATCH:-${SLURM_TMPDIR:-/tmp}/meno-rumba-${UID}-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}}
cache_dir=${CLUSTER_CACHE_DIR:-$CLUSTER_LOCAL_SCRATCH/cache}
mkdir -p "$cache_dir"
export PIP_CACHE_DIR="$cache_dir"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export HF_HOME="$cache_dir/huggingface"
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export ANONYMIZED_TELEMETRY=False
export OUTLINES_CACHE_DIR="$cache_dir/outlines"
export TORCH_HOME="$cache_dir/torch_cache"
export XDG_CACHE_HOME="$cache_dir/xdg_cache"
export TRITON_CACHE_DIR="$cache_dir/triton_cache"
export FLASHINFER_WORKSPACE_BASE="$cache_dir/flashinfer_workspace"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
# A running offline job must not depend on a reverse-forward to the Mac.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# The node may become available before the initial offline assets finish
# downloading on the headnode. Never start vLLM against partial installations.
for ((attempt=0; attempt<240; attempt++)); do
  preparation=$(cat runs/cluster-setup/bootstrap.status 2>/dev/null || true)
  printf '[%s] preparation=%s\n' "$(date -u +%FT%TZ)" "$preparation"
  case "$preparation" in
    ready) break ;;
    failed:*) echo "Preparation failed; inspect runs/cluster-setup/bootstrap.log" >&2; exit 1 ;;
  esac
  sleep 15
done
[[ $preparation == ready ]] || { echo "Preparation did not finish within one hour" >&2; exit 1; }
exec .venv-ragu/bin/python -u scripts/run_meno_pair.py \
  --models-dir "${MODELS_DIR:-$PWD/../models}" \
  --vllm-env "${CLUSTER_VLLM_ENV:-$PWD/.venv-vllm}" \
  --local-scratch "$CLUSTER_LOCAL_SCRATCH" \
  --checkpoint-seconds "${CHECKPOINT_SECONDS:-3600}" \
  --statistics-seconds "${STATISTICS_SECONDS:-3600}"
