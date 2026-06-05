#!/usr/bin/env bash
# Build deepussim.sif from deepussim.def on an NHR@FAU frontend (Alex/Helma/Fritz/Woody/Meggie).
# The build context is the repo root, so %files paths (src, configs, ...) resolve.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

# Keep the image and all pull/build caches on $WORK, never $HOME (small quota).
: "${WORK:?set \$WORK (NHR@FAU work filesystem) before building}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$WORK/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$WORK/.apptainer/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SIF="${1:-$WORK/deepussim.sif}"

# Build from the repo root so the .def's %files (relative) paths resolve.
cd "$REPO"

BUILD="apptainer build"
# NHR@FAU wrapper for permission issues on the frontend.
if ! apptainer build --help >/dev/null 2>&1; then
    BUILD="/apps/singularity/apptainer-wrapper.sh build"
fi

echo ">> building $SIF from apptainer/deepussim.def (context: $REPO)"
$BUILD "$SIF" apptainer/deepussim.def
echo ">> done: $SIF"
