#!/usr/bin/env bash
#SBATCH --job-name=deepus-cut-u
#SBATCH --output=logs/renderer_training/cut_unpaired_%j.out
#SBATCH --error=logs/renderer_training/cut_unpaired_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${DEEPUSSIM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${REPO_ROOT}"

mkdir -p logs/renderer_training

RUNNER="${DEEPUSSIM_RUNNER:-apptainer/run.sh}"
DATA="${DATA:-data/renderer_lc2_pairs}"
OUT="${OUT:-runs/renderer_cut_unpaired_display_ep300_b2}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-2}"
LR="${LR:-2e-4}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
LAMBDA_NCE="${LAMBDA_NCE:-1.0}"
NUM_PATCHES="${NUM_PATCHES:-256}"
NGF="${NGF:-64}"
NDF="${NDF:-64}"
N_BLOCKS="${N_BLOCKS:-9}"
DEVICE="${DEVICE:-cuda}"

cmd=(
    python renderer_training/train_cut_unpaired.py
    --data "${DATA}"
    --out "${OUT}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --lr "${LR}"
    --workers "${WORKERS}"
    --sample-every "${SAMPLE_EVERY}"
    --lambda-nce "${LAMBDA_NCE}"
    --num-patches "${NUM_PATCHES}"
    --ngf "${NGF}"
    --ndf "${NDF}"
    --n-blocks "${N_BLOCKS}"
    --device "${DEVICE}"
)

if [[ -n "${SOURCE:-}" ]]; then
    cmd+=(--source "${SOURCE}")
fi
if [[ -n "${TARGET:-}" ]]; then
    cmd+=(--target "${TARGET}")
fi
if [[ -n "${RESUME:-}" ]]; then
    cmd+=(--resume "${RESUME}")
fi

echo "repo=${REPO_ROOT}"
echo "runner=${RUNNER}"
printf 'command: %q ' "${cmd[@]}"
echo

if [[ -n "${RUNNER}" ]]; then
    read -r -a runner_args <<< "${RUNNER}"
    "${runner_args[@]}" "${cmd[@]}"
else
    "${cmd[@]}"
fi
