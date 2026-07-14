#!/usr/bin/env python
"""Physics-rendered counterpart of data/liver_seg/gen — same poses & masks, physics images.

Controlled B_cut vs B_phys: reuse the existing generated-set masks (data/liver_seg/gen), but
replace each CUT-generated image with the PHYSICS render of the SAME CBCT sector (same pose).
Only the pixels differ between the two arms, so a downstream Dice/NSD gap is attributable to the
renderer (learned CUT vs physics), not to poses, masks, or confidence cropping.

    module load python/3.12-base
    python renderer_training/build_gen_phys.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deepussim.us.renderer import bmode, RendererParams                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", type=Path,
                    default=REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--gen", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "gen",
                    help="existing CUT-gen dataset whose masks/frames we reuse")
    ap.add_argument("--depth-mm", type=float, default=93.0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "gen_phys")
    args = ap.parse_args()

    cbct = np.load(args.render, allow_pickle=True)["cbct_sector"]
    masks = np.load(args.gen / "masks.npy")
    meta = list(csv.DictReader((args.gen / "meta.csv").open()))
    assert len(meta) == len(masks), (len(meta), len(masks))
    depth = np.linspace(0.0, args.depth_mm, cbct.shape[1])

    imgs = []
    for r in meta:
        j = int(r["frame"])                                          # frame index into rendered_us
        phys = bmode(cbct[j].astype(float), depth, RendererParams())  # physics render, same pose
        imgs.append(np.clip(phys * 255.0, 0, 255).astype(np.uint8))
    imgs = np.stack(imgs)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "images.npy", imgs)
    np.save(args.out / "masks.npy", masks)                            # identical masks
    shutil.copy(args.gen / "meta.csv", args.out / "meta.csv")
    npos = sum(int(r["is_positive"]) for r in meta)
    print(f"physics-gen set: {len(imgs)} frames ({npos} pos / {len(imgs)-npos} neg), "
          f"images {imgs.shape} {imgs.dtype}  masks identical to {args.gen.name}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
