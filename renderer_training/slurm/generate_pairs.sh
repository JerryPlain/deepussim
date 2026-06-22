#!/usr/bin/env bash
#SBATCH --job-name=deepus-pairs
#SBATCH --output=logs/renderer_training/generate_pairs_%j.out
#SBATCH --error=logs/renderer_training/generate_pairs_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${DEEPUSSIM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${REPO_ROOT}"

mkdir -p logs/renderer_training

RUNNER="${DEEPUSSIM_RUNNER:-apptainer/run.sh}"
OUT="${OUT:-data/renderer_lc2_pairs}"
LC2_DIR="${LC2_DIR:-data/lc2}"
ALLOW_MISSING_LC2="${ALLOW_MISSING_LC2:-0}"

cmd=(
    python renderer_training/pair_generation.py
    --lc2-dir "${LC2_DIR}"
    --out "${OUT}"
)

if [[ -n "${SEQUENCES:-}" ]]; then
    # Space-separated sequence paths, e.g. SEQUENCES="data/sequences/scan1.npz data/sequences/scan2.npz".
    read -r -a sequence_args <<< "${SEQUENCES}"
    cmd+=(--sequences "${sequence_args[@]}")
fi

if [[ "${ALLOW_MISSING_LC2}" == "1" ]]; then
    cmd+=(--allow-missing-lc2)
fi

if [[ -n "${REPORT:-}" ]]; then
    cmd+=(--report "${REPORT}")
fi
if [[ -n "${VOLUME_PATH:-}" ]]; then
    cmd+=(--volume-path "${VOLUME_PATH}")
fi
if [[ -n "${WORLD_FROM_PHANTOM:-}" ]]; then
    cmd+=(--world-from-phantom "${WORLD_FROM_PHANTOM}")
fi
if [[ -n "${REF_SEQUENCE:-}" ]]; then
    cmd+=(--ref-sequence "${REF_SEQUENCE}")
fi
if [[ -n "${US_SPACING:-}" ]]; then
    cmd+=(--us-spacing "${US_SPACING}")
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
