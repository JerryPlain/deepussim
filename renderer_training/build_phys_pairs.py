#!/usr/bin/env python
"""Physics-source training pairs for physics-refined CUT: replace CBCT source with its physics render.

The paired CUT currently learns CBCT_sector -> real_US. For physics-refined we instead learn
physics_US -> real_US: same real-US targets, but the SOURCE is the physics render of each CBCT
sector (deepussim.us.renderer.bmode), which already carries depth attenuation / shadowing. The
network then only has to add realistic texture, not invent the physics.

Writes a new pairs.npz with `cbct` replaced by the physics render (field name kept so
train_cut_paired.py needs no change; point it at --pairs <this file>).

    module load python/3.12-base
    python renderer_training/build_phys_pairs.py
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--pairs", type=Path, default=REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz")
    ap.add_argument("--depth-mm", type=float, default=93.0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "renderer_lc2_pairs_phys" / "pairs.npz")
    args = ap.parse_args()

    d = dict(np.load(args.pairs, allow_pickle=True))
    cbct = d["cbct"]
    depth = np.linspace(0.0, args.depth_mm, cbct.shape[1])
    phys = np.stack([bmode(cbct[i].astype(float), depth, RendererParams()) for i in range(len(cbct))])
    d["cbct"] = phys.astype(np.float32)                              # source now = physics US

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)
    print(f"wrote {args.out}  ({len(phys)} pairs; source=physics US, target=real US)")
    print(f"  physics source range [{phys.min():.3f}, {phys.max():.3f}]  us target unchanged")


if __name__ == "__main__":
    main()
