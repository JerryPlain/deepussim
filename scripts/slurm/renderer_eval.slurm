#!/usr/bin/env bash
#SBATCH --job-name=deepussim-rendeval
#SBATCH --partition=a100          # Alex GPU partition; use a40 if a100 is busy
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Evaluate the learned renderer (B1) on the GPU: generate the fake-US pool from the trained
# generator and compute STRUCTURE-CONSISTENCY (the anti-hallucination metric, torch-only). The
# fakes are cached to <RUN>/eval_cache.npz so FID/KID can be computed on the LOGIN node (which
# has torchvision + internet for the InceptionV3 weights, neither of which the container has):
#     python scripts/eval_renderer.py --run runs/renderer_cut --from-cache --device cpu
#
# Submit from the repo root:        sbatch scripts/slurm/renderer_eval.slurm
# Override:                         RUN=runs/foo N=600 sbatch scripts/slurm/renderer_eval.slurm
set -euo pipefail

module load apptainer 2>/dev/null || true
REPO="$SLURM_SUBMIT_DIR"; cd "$REPO"

RUN="${RUN:-runs/renderer_cut}"
DATA="${DATA:-data/renderer}"
N="${N:-600}"

apptainer/run.sh python scripts/eval_renderer.py \
    --run "$RUN" --data "$DATA" --n "$N" --device cuda
echo "structure-consistency done; for FID/KID run on the login node:"
echo "  python scripts/eval_renderer.py --run $RUN --from-cache --device cpu"
