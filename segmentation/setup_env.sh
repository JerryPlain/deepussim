#!/usr/bin/env bash
# One-time env prep for SAM2 liver segmentation. Run on the LOGIN node (needs internet;
# Alex compute nodes are offline). Creates a venv + installs torch/sam2 + downloads the
# SAM2.1 checkpoint. The SLURM job then activates this venv and runs offline.
#
#   bash segmentation/setup_env.sh
set -euo pipefail

VENV="${SAM2_VENV:-$WORK/venvs/sam2seg}"
CKPT_DIR="${SAM2_CKPT_DIR:-$WORK/sam2}"
CKPT="sam2.1_hiera_small.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${CKPT}"

module load python/3.12-base

if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip

# torch (CUDA build). Pin the CUDA wheel index matching the Alex driver; cu124 works on A40.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# SAM2 (image encoder only is needed; the optional CUDA postproc ext is not required)
python -m pip install "git+https://github.com/facebookresearch/sam2.git"
python -m pip install numpy

mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/$CKPT" ]; then
    echo "downloading $CKPT ..."
    curl -L -o "$CKPT_DIR/$CKPT" "$CKPT_URL"
fi

echo
echo "done."
echo "  venv : $VENV"
echo "  ckpt : $CKPT_DIR/$CKPT"
echo "set these in segmentation/train.slurm if you changed the defaults."
