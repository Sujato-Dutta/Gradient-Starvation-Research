#!/usr/bin/env bash
#SBATCH --job-name=gs-cifar-confirm
#SBATCH --partition=gpu_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2
#SBATCH --output=results/slurm/cifar-%A_%a.out
#SBATCH --error=results/slurm/cifar-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${GS_PROJECT_ROOT:-/dgxa_home/se23uari141/Gradient-Starvation/Gradient-Starvation-Confirmation}"
DATA_ROOT="${GS_CIFAR_DATA_ROOT:-/dgxa_home/se23uari141/Gradient-Starvation/Gradient-Starvation-Research/data/cifar10}"
PYTHON_BIN="${GS_PYTHON_BIN:-/dgxa_home/se23uari141/.venvs/gs-cifar-py310/bin/python}"
CONFIG_PATH="$PROJECT_ROOT/configs/cifar_matched_preregistration.frozen.json"
ENVIRONMENT_PATH="$PROJECT_ROOT/research_scope/cifar_dgx_environment_v2.json"
SEEDS=(33001 33002 33003 33004 33005 33006 33007 33008)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid or missing SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_PATH="$PROJECT_ROOT/results/cifar_matched_frozen_v1_seed${SEED}"
if [[ -e "$OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite existing confirmation output: $OUTPUT_PATH" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Starting seed=$SEED job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID host=$(hostname) utc=$(date -u +%FT%TZ)"
nvidia-smi -L

"$PYTHON_BIN" scripts/run_cifar_matched.py \
  --config "$CONFIG_PATH" \
  --environment-manifest "$ENVIRONMENT_PATH" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_PATH" \
  --seed "$SEED" \
  --device cuda

sha256sum "$OUTPUT_PATH/metadata.json" "$OUTPUT_PATH/summary.json"
echo "Completed seed=$SEED utc=$(date -u +%FT%TZ)"
