#!/usr/bin/env bash
# Override partition/account/time and --gres at submission for the local cluster.
#SBATCH --job-name=meno-rumba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Keep the scheduler's CUDA_VISIBLE_DEVICES; do not call the Everest launchers.
exec bash scripts/cluster_a100.sh "$@"
