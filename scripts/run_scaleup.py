#!/usr/bin/env python
"""Step 5 entrypoint: generate a US dataset by reslicing + rendering a CBCT volume.

No-sim by default (geometric poses, no force). Pass ``--sim`` to drive a Genesis scene
for reachable poses + contact force (requires Genesis + an implemented sim.scene).

    # surface-constrained sweep over the real phantom (geometry branch):
    python scripts/run_scaleup.py \
        --volume data/cbct/intensity.nrrd --labels data/cbct/labels.nrrd \
        --mesh data/cbct/phantom_surface.stl --trajectory surface \
        --config configs/renderer.yaml --out data/ds --n 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from deepussim.data.volume import load_volume
from deepussim.us.reslice import ProbeGeometry
from deepussim.us.renderer import RendererParams
from deepussim.pipeline.sampling import (
    linear_sweep, surface_sweep, surface_raster, top_sweep_endpoints,
)
from deepussim.pipeline.scaleup import generate_dataset


def load_config(path: str | None):
    if not path:
        return RendererParams(), ProbeGeometry()
    cfg = yaml.safe_load(Path(path).read_text())
    params = RendererParams(**cfg.get("renderer", {}))
    geom = ProbeGeometry(**cfg.get("geometry", {}))
    return params, geom


def nosim_poses(args, volume):
    """Probe poses in the CBCT mm frame for the no-sim reslice path."""
    if args.trajectory in ("surface", "raster"):
        if not args.mesh:
            raise SystemExit(f"--trajectory {args.trajectory} requires --mesh (phantom STL, CBCT mm)")
        import trimesh
        mesh = trimesh.load(args.mesh)
        if args.trajectory == "raster":
            return surface_raster(mesh, axis=args.sweep_axis, span_frac=args.span_frac,
                                  cross_frac=args.cross_frac, n_lines=args.lines,
                                  n_per_line=args.per_line, standoff_mm=args.standoff_mm)
        start, end = top_sweep_endpoints(mesh, axis=args.sweep_axis, span_frac=args.span_frac)
        return surface_sweep(mesh, start, end, args.n, standoff_mm=args.standoff_mm)
    # center-sweep (default): a straight line across the volume centre, aimed -z.
    c = volume.center_world()
    half = volume.spacing * np.array(volume.shape) * 0.3
    start = c + np.array([-half[0], 0.0, half[2]])
    end = c + np.array([half[0], 0.0, half[2]])
    return linear_sweep(start, end, args.n, axial_dir=(0.0, 0.0, -1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True, help="CBCT intensity volume (.nrrd/.nii.gz)")
    ap.add_argument("--labels", help="CBCT label volume (.nrrd/.nii.gz) for anatomy masks")
    ap.add_argument("--config", help="renderer/geometry YAML (configs/renderer.yaml)")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=64, help="number of poses to sample")
    ap.add_argument("--trajectory", choices=["center-sweep", "surface", "raster"],
                    default="center-sweep",
                    help="no-sim pose source: straight sweep across the volume, a single "
                         "surface-constrained glide, or a multi-line raster over the phantom "
                         "mesh (--mesh)")
    ap.add_argument("--mesh", help="phantom surface STL (CBCT mm) for --trajectory surface/raster")
    ap.add_argument("--sweep-axis", type=int, default=0, help="surface sweep axis (0=x, 1=y)")
    ap.add_argument("--span-frac", type=float, default=0.6, help="along-sweep span fraction")
    ap.add_argument("--cross-frac", type=float, default=0.4,
                    help="raster: cross-sweep span fraction (coverage across lines)")
    ap.add_argument("--lines", type=int, default=5, help="raster: number of parallel sweep lines")
    ap.add_argument("--per-line", type=int, default=24, help="raster: poses per sweep line")
    ap.add_argument("--standoff-mm", type=float, default=2.0, help="probe standoff off the surface")
    ap.add_argument("--sim", action="store_true", help="drive a Genesis scene")
    ap.add_argument("--headless", action="store_true",
                    help="run the sim without the viewer window (default: viewer on)")
    args = ap.parse_args()

    volume = load_volume(args.volume)
    labels = load_volume(args.labels) if args.labels else None
    params, geom = load_config(args.config)

    if not args.sim:
        # No-sim: poses live directly in the CBCT frame (mm); reslice + render + free mask.
        poses = nosim_poses(args, volume)
        written = generate_dataset(args.out, volume, poses, geom, params,
                                   label_volume=labels)
        print(f"wrote {written} samples to {args.out}")
        return

    # Sim path: poses are nominal probe targets in the sim world (metres); physics gives
    # the achieved pose + contact force, then the placement bridge maps it into CBCT mm.
    from deepussim.sim.scene import UltrasoundScene, SceneConfig
    from deepussim.calib.placement import align_points_placement

    # probe_offset defaults to the placeholder probe tip on the FR3 flange (see SceneConfig).
    cfg = SceneConfig(show_viewer=not args.headless)
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    px, py, pz = cfg.phantom_pos
    top = pz + cfg.phantom_size[2] / 2.0               # box top surface in sim (m)
    # Probe targets just below the surface so the arm presses into contact.
    start = np.array([px - 0.04, py, top - 0.02])
    end = np.array([px + 0.04, py, top - 0.02])
    sim_poses = linear_sweep(start, end, args.n, axial_dir=(0.0, 0.0, -1.0))

    # Placement: map the sim box-top centre onto the CBCT volume's top-centre voxel.
    vox_top = np.array([(volume.shape[0] - 1) / 2.0, (volume.shape[1] - 1) / 2.0,
                        volume.shape[2] - 1.0])
    cbct_top = volume.voxel_to_world(vox_top)[0]
    placement = align_points_placement([px, py, top], cbct_top)

    written = generate_dataset(args.out, volume, sim_poses, geom, params,
                               label_volume=labels, scene=scene, sim_to_cbct=placement,
                               settle_steps=300)
    print(f"wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
