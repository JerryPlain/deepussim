#!/usr/bin/env bash
#SBATCH --job-name=sam2-liver-A
#SBATCH --account=b306dd
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=segmentation/logs/%x_%j.out
#
# Group A baseline: frozen SAM2 encoder + seg head on LIMITED real US.
# Prereq (once, on the login node): bash segmentation/setup_env.sh
#
#   sbatch segmentation/train.slurm
set -euo pipefail

VENV="${SAM2_VENV:-$WORK/venvs/sam2seg}"
CKPT="${SAM2_CKPT:-$WORK/sam2/sam2.1_hiera_small.pt}"
CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p segmentation/logs
module load python/3.12-base
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Alex compute nodes are offline: use any HF/torch caches under $WORK, don't hit the network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Group A (default): 40 real only. Group B: set GEN_SPLIT=gen. SEED / OUT override per run.
OUT="${OUT:-runs/seg_sam2_A_limited}"
SEED="${SEED:-0}"
GEN_ARGS=""
[ -n "${GEN_SPLIT:-}" ] && GEN_ARGS="--gen-split ${GEN_SPLIT}"

# NPOS/NNEG default to the limited-real budget (40); set both large (e.g. 1000) for FULL real.
NPOS="${NPOS:-40}"
NNEG="${NNEG:-40}"

srun python segmentation/train_sam2_head.py \
    --sam2-ckpt "$CKPT" \
    --sam2-cfg  "$CFG" \
    --n-pos "$NPOS" --n-neg "$NNEG" --seed "$SEED" \
    --epochs 60 --batch 4 --lr 1e-3 \
    $GEN_ARGS --out "$OUT"
