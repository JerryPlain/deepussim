#!/bin/bash
#SBATCH --job-name=render_us_from_poses
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/renderer_training/render_us_from_poses_%j.out
#SBATCH --error=logs/renderer_training/render_us_from_poses_%j.err

set -euo pipefail

mkdir -p logs/renderer_training

TRAJECTORY=${TRAJECTORY:-data/trajectories/novel_offset_world_valid.npz}
CKPT=${CKPT:-runs/renderer_cut_paired_display_ep300_b2_lp005/generator.pt}
BATCH=${BATCH:-4}
PREVIEW=${PREVIEW:-12}

if [[ -z "${OUT:-}" ]]; then
  traj_stem=$(basename "${TRAJECTORY}" .npz)
  ckpt_run=$(basename "$(dirname "${CKPT}")")
  OUT="data/renderer_training/rendered_us/${traj_stem}/${ckpt_run}"
fi

echo "host=$(hostname)"
echo "date=$(date)"
echo "trajectory=${TRAJECTORY}"
echo "checkpoint=${CKPT}"
echo "out=${OUT}"
echo "batch=${BATCH}"
echo "preview=${PREVIEW}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

apptainer/run.sh python renderer_training/render_us_from_poses.py \
  --trajectory "${TRAJECTORY}" \
  --checkpoint "${CKPT}" \
  --out "${OUT}" \
  --batch-size "${BATCH}" \
  --preview "${PREVIEW}" \
  --device cuda
