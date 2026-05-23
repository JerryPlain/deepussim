#!/usr/bin/env python
"""Step 5 entrypoint: generate a US dataset by reslicing + rendering a CBCT volume.

No-sim by default (geometric poses, no force). Pass ``--sim`` to drive a Genesis scene
for reachable poses + contact force (requires Genesis + an implemented sim.scene).

    python scripts/run_scaleup.py \
        --volume data/phantom/intensity.nii.gz \
        --labels data/phantom/labels.nii.gz \
        --config configs/renderer.yaml \
        --out data/synth_ds --n 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from deepussim.data.volume import load_nifti
from deepussim.us.reslice import ProbeGeometry
from deepussim.us.renderer import RendererParams
from deepussim.pipeline.sampling import linear_sweep
from deepussim.pipeline.scaleup import generate_dataset


def load_config(path: str | None):
    if not path:
        return RendererParams(), ProbeGeometry()
    cfg = yaml.safe_load(Path(path).read_text())
    params = RendererParams(**cfg.get("renderer", {}))
    geom = ProbeGeometry(**cfg.get("geometry", {}))
    return params, geom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True, help="CBCT intensity volume (NIfTI)")
    ap.add_argument("--labels", help="CBCT label volume (NIfTI) for anatomy masks")
    ap.add_argument("--config", help="renderer/geometry YAML (configs/renderer.yaml)")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=64, help="number of poses to sample")
    ap.add_argument("--sim", action="store_true", help="drive a Genesis scene")
    args = ap.parse_args()

    volume = load_nifti(args.volume)
    labels = load_nifti(args.labels) if args.labels else None
    params, geom = load_config(args.config)

    # Sweep across the volume centre, probe aimed along -z into the volume.
    c = volume.center_world()
    half = volume.spacing * np.array(volume.shape) * 0.3
    start = c + np.array([-half[0], 0.0, half[2]])
    end = c + np.array([half[0], 0.0, half[2]])
    poses = linear_sweep(start, end, args.n, axial_dir=(0.0, 0.0, -1.0))

    scene = None
    if args.sim:
        from deepussim.sim.scene import UltrasoundScene, SceneConfig

        scene = UltrasoundScene(SceneConfig()).build()
        scene.reset()

    written = generate_dataset(
        args.out, volume, poses, geom, params, label_volume=labels, scene=scene
    )
    print(f"wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
