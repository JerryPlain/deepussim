#!/usr/bin/env bash
#SBATCH --job-name=cut-phys2real
#SBATCH --account=b306dd
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=segmentation/logs/%x_%j.out
#
# sim2real renderer: train CUT with SOURCE = physics US (build_phys_pairs.py) -> real US.
# Same losses/params as the CBCT->US paired CUT (ep300, batch2, lambda_pair 0.05) so the only
# change vs the baseline renderer is the input (physics US instead of raw CBCT).
#
#   sbatch renderer_training/cut_phys2real.slurm
set -euo pipefail

VENV="${SAM2_VENV:-$WORK/venvs/sam2seg}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p segmentation/logs
module load python/3.12-base
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python renderer_training/train_cut_paired.py \
    --pairs data/renderer_lc2_pairs_phys/pairs.npz \
    --out runs/renderer_cut_phys2real \
    --epochs 300 --batch 2 --lambda-pair 0.05 --sample-every 20
