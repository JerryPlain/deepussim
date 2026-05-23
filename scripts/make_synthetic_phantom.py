#!/usr/bin/env python
"""Generate a synthetic CBCT-like intensity volume + label volume for testing.

Lets you exercise the whole no-sim pipeline (reslice -> render -> mask) without any
real scan. Creates a soft-tissue background with a few embedded structures (a couple
of "organs" and a bright "vessel wall"), plus a matching integer label volume.

    python scripts/make_synthetic_phantom.py --out data/phantom --size 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deepussim.data.volume import Volume, save_nifti


def make_phantom(size: int = 128, spacing_mm: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = size
    intensity = np.full((n, n, n), 0.2, dtype=np.float32)  # soft-tissue baseline
    labels = np.zeros((n, n, n), dtype=np.int16)

    ii, jj, kk = np.mgrid[0:n, 0:n, 0:n].astype(float)
    c = (n - 1) / 2.0

    # Organ 1: large sphere (label 1).
    s1 = (ii - c) ** 2 + (jj - c * 0.8) ** 2 + (kk - c) ** 2 < (n * 0.28) ** 2
    intensity[s1] = 0.5
    labels[s1] = 1

    # Organ 2: smaller offset ellipsoid (label 2).
    s2 = ((ii - c * 1.3) / (n * 0.18)) ** 2 + ((jj - c * 1.2) / (n * 0.12)) ** 2 + (
        (kk - c) / (n * 0.18)
    ) ** 2 < 1.0
    intensity[s2] = 0.7
    labels[s2] = 2

    # Vessel: bright thin cylinder along k (label 3).
    vessel = (ii - c * 0.7) ** 2 + (jj - c) ** 2 < (n * 0.04) ** 2
    intensity[vessel] = 0.95
    labels[vessel] = 3

    intensity += rng.normal(0.0, 0.02, size=intensity.shape).astype(np.float32)
    intensity = np.clip(intensity, 0.0, 1.0)

    affine = np.eye(4)
    affine[:3, :3] *= spacing_mm
    return Volume(intensity, affine), Volume(labels, affine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--spacing", type=float, default=1.0, help="voxel size (mm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    intensity, labels = make_phantom(args.size, args.spacing, args.seed)
    save_nifti(out / "intensity.nii.gz", intensity)
    save_nifti(out / "labels.nii.gz", labels)
    print(f"wrote {out/'intensity.nii.gz'} and {out/'labels.nii.gz'} "
          f"(shape {intensity.shape}, spacing {args.spacing} mm)")


if __name__ == "__main__":
    main()
