#!/usr/bin/env bash
#SBATCH --job-name=deepussim-scaleup
#SBATCH --partition=a100          # Alex GPU partition; use a40 if a100 is busy
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Single-node GPU job — the container is NOT for multi-node MPI (NHR@FAU caveat).
#
# Full --sim scale-up: the FR3 follows the trajectory onto the seated phantom and
# force-servos to a realistic contact; each kept pose gets (US image, pose, anatomy mask,
# contact force). NOTE: --sim adds force/contact/reachability ONLY — the US *appearance*
# is the same placeholder renderer as the no-sim path. Making it look real is a separate
# job (train the learned renderer); this run does not change the image.
#
# Submit from the repo root on Alex:        sbatch scripts/slurm/scaleup_sim.slurm
# Override defaults:   TRAJ=replay N=64 OUT=$WORK/deepussim/ds_sim sbatch scripts/slurm/scaleup_sim.slurm
set -euo pipefail

module load apptainer 2>/dev/null || true   # available by default on most NHR systems

REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"

# Inputs live in the repo's data/ (bind-mounted via --pwd in run.sh). Override OUT to $WORK
# to keep large datasets off the repo workspace.
VOLUME="${VOLUME:-data/cbct_20260612/intensity.nrrd}"
LABELS="${LABELS:-data/cbct_20260612/labels.nrrd}"
MESH="${MESH:-data/cbct_20260612/phantom_surface.stl}"
OUT="${OUT:-data/ds_sim}"
TRAJ="${TRAJ:-contact}"   # 'contact': densify the real scanned patch, orientation from the real
                          # EE poses (reliable since the orientation fix). 'replay': re-drive the
                          # exact real poses. Both are in-regime.
N="${N:-120}"
FORCE_N="${FORCE_N:-5.0}"
# Learned renderer: set RENDERER_CKPT=runs/renderer_cut/generator.pt to render realistic US
# (CUT generator) instead of the placeholder physics model. Empty = physics placeholder.
RENDERER_CKPT="${RENDERER_CKPT:-}"
RENDER_ARG=""; [ -n "$RENDERER_CKPT" ] && RENDER_ARG="--renderer-ckpt $RENDERER_CKPT"
# Pre-extracted sequences (poses+contact) — used instead of raw bags so the container needs no
# 'rosbags' lib. Extract once on the conda env: python scripts/extract_rosbags.py ...
SEQS="${SEQS:-data/sequences/phantom.npz data/sequences/phantom1.npz}"

# --headless: no viewer on a compute node. The write gate keeps a pose only if the probe both
# contacted AND reached its target; dropped poses are logged with a reason.
apptainer/run.sh python scripts/run_scaleup.py --sim \
    --volume "$VOLUME" --labels "$LABELS" --mesh "$MESH" \
    --trajectory "$TRAJ" --config configs/renderer.yaml \
    --sequences $SEQS $RENDER_ARG \
    --out "$OUT" --headless --n "$N" --force-n "$FORCE_N"
