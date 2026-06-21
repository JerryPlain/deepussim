#!/usr/bin/env bash
#SBATCH --job-name=deepussim-renderer
#SBATCH --partition=a100          # Alex GPU partition; use a40 if a100 is busy
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Single-node GPU job. B1: train the learned US renderer (CUT) on the two fan-layout pools.
#
# PREREQUISITE — build the pools ONCE on the conda env (CPU; needs trimesh/SimpleITK, not the
# training container):
#     python scripts/prep_renderer_data.py \
#         --sequences data/sequences/phantom.npz data/sequences/phantom1.npz \
#         --volume data/cbct_20260612/intensity.nrrd --mesh data/cbct_20260612/phantom_surface.stl \
#         --config configs/renderer.yaml --out data/renderer
#
# Training itself is pure torch (already baked in deepussim.sif) — no container rebuild.
#
# Submit from the repo root on Alex:        sbatch scripts/slurm/renderer_train.slurm
# Override:               EPOCHS=400 BATCH=8 OUT=$WORK/deepussim/renderer_cut sbatch ...
set -euo pipefail

module load apptainer 2>/dev/null || true

REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

DATA="${DATA:-data/renderer}"
OUT="${OUT:-runs/renderer_cut}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-4}"
# Fine-tune: set RESUME=runs/renderer_cut/generator.pt to continue from a checkpoint (new data).
# Point DATA at pools built from OLD + NEW sequences so the old regime isn't forgotten.
RESUME="${RESUME:-}"
RESUME_ARG=""; [ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"

if [ ! -f "$DATA/source_cbct.npz" ] || [ ! -f "$DATA/target_us.npz" ]; then
    echo "missing pools in $DATA — run scripts/prep_renderer_data.py first (see header)"; exit 1
fi

apptainer/run.sh python scripts/train_renderer.py \
    --data "$DATA" --out "$OUT" --epochs "$EPOCHS" --batch "$BATCH" --device cuda $RESUME_ARG
