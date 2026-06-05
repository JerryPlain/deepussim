#!/usr/bin/env bash
#SBATCH --job-name=deepussim-scaleup
#SBATCH --partition=a100          # Alex GPU partition; use a40 if a100 is busy
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Single-node GPU job — the container is NOT for multi-node MPI (NHR@FAU caveat).
#
# Submit from the repo root on Alex:   sbatch apptainer/job.slurm
set -euo pipefail

module load apptainer 2>/dev/null || true   # available by default on most NHR systems

REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

# Edit these to your CBCT paths on $WORK.
VOLUME="$WORK/deepussim/cbct/intensity.nrrd"
LABELS="$WORK/deepussim/cbct/labels.nrrd"
MESH="$WORK/deepussim/cbct/phantom_surface.stl"
OUT="$WORK/deepussim/ds_sim"

# `replay` is the only reliable trajectory until the mesh is watertight
# (see CHANGELOG "Known limitation"). --headless: no viewer on a compute node.
apptainer/run.sh python scripts/run_scaleup.py --sim \
    --volume "$VOLUME" --labels "$LABELS" --mesh "$MESH" \
    --trajectory replay --config configs/renderer.yaml \
    --out "$OUT" --headless --n 64
