#!/usr/bin/env bash
# Run a command inside deepussim.sif with the GPU and the live repo.
#
#   apptainer/run.sh python scripts/run_scaleup.py --sim \
#       --volume data/cbct/intensity.nrrd --labels data/cbct/labels.nrrd \
#       --mesh data/cbct/phantom_surface.stl --trajectory replay \
#       --config configs/renderer.yaml --out data/ds_sim --headless
#
# - `--nv` passes through the host NVIDIA driver (Genesis CUDA kernels).
# - The host repo's src/ is bind-mounted over the baked copy, so code edits take
#   effect with no rebuild (the image installed deepussim editable). Drop the
#   src bind below for a fully frozen, reproducible run off the baked code.
# - $WORK / $HOME mount by default on NHR@FAU, so data paths under them just work.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

SIF="${DEEPUSSIM_SIF:-${WORK:-$REPO}/deepussim.sif}"
[ -f "$SIF" ] || { echo "missing image: $SIF (build it: apptainer/build.sh)"; exit 1; }

exec apptainer run --nv \
    --bind "$REPO/src:/opt/deepussim/src" \
    --pwd "$REPO" \
    "$SIF" "$@"
