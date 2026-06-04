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
    ap.add_argument("--sim", action="store_true",
                    help="drive a Genesis scene: the FR3 replays the real probe trajectory onto "
                         "the phantom (placed by the measured calibration) and force-servos to "
                         "--force-n; requires --mesh (CBCT mm STL) and --bags")
    ap.add_argument("--force-n", type=float, default=5.0,
                    help="sim: target contact force (N) for the depth-vs-force servo loop")
    ap.add_argument("--contact-timeconst", type=float, default=3.0,
                    help="sim: rigid-contact softness (s); the phantom is soft tissue, so a "
                         "compliant contact (~3) gives realistic few-N forces instead of rigid 10^3 N")
    ap.add_argument("--bags", nargs="+",
                    default=["data/rosbags/phantom.bag", "data/rosbags/phantom1.bag"],
                    help="sim: rosbag(s) whose contact-frame EE poses are replayed as probe targets")
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
    # the achieved pose + contact force, then the placement bridge maps it back to CBCT mm.
    import trimesh
    from deepussim.sim.scene import UltrasoundScene, SceneConfig
    from deepussim.calib import T_WORLD_FROM_CBCT, seat_phantom_placement
    from deepussim.calib.transforms import T_EE_FROM_PROBE
    from deepussim.calib.placement import meters_to_mm
    from deepussim.data.rosbag import extract_sequence
    from deepussim.geometry import mat_to_quat, invert, compose

    if not args.mesh:
        raise SystemExit("--sim requires --mesh (phantom surface STL in CBCT mm)")

    # Trajectory: replay the real probe sweep. Rosbag contact frames are the EE poses the arm
    # actually reached on the phantom; the probe target for each is EE pose o hand-eye. Use all
    # contacts to fit the placement (below), and subsample to --n for the driven sweep.
    frames = [f for bag in args.bags for f in extract_sequence(bag).frames if f.contact]
    if not frames:
        raise SystemExit(f"no contact frames found in {args.bags}")
    all_origins = np.array([compose(f.pose, T_EE_FROM_PROBE)[:3, 3] for f in frames])
    pick = np.linspace(0, len(frames) - 1, args.n, dtype=int)
    sim_poses = [compose(frames[i].pose, T_EE_FROM_PROBE) for i in pick]

    # Phantom placement (matches scripts/view_sim.py, the visually-verified pose): the measured
    # T_WORLD_FROM_CBCT stands the phantom on end (its CBCT-scan frame is rolled 90 deg from our
    # DICOM-LPS volume); the real rig has it LYING, so seat_phantom_placement tips it onto its
    # side and seats the surface onto the real contact cloud. The reslice bridge is derived from
    # the SAME transform, so mesh, CBCT volume and probe poses stay consistent.
    mesh = trimesh.load(args.mesh)                                   # CBCT mm
    T_world_from_cbctm = seat_phantom_placement(mesh, all_origins, T_WORLD_FROM_CBCT)

    cfg = SceneConfig(
        show_viewer=not args.headless,
        phantom_mesh=args.mesh,
        phantom_scale=0.001,                              # mm mesh -> metres
        phantom_pos=tuple(T_world_from_cbctm[:3, 3]),
        phantom_quat=tuple(mat_to_quat(T_world_from_cbctm[:3, :3])),
        contact_timeconst=args.contact_timeconst,         # soft tissue -> realistic few-N contact
    )
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    # Bridge achieved sim poses back to CBCT mm (derived from the SAME placement transform).
    sim_to_cbct = meters_to_mm(invert(T_world_from_cbctm))

    written = generate_dataset(args.out, volume, sim_poses, geom, params,
                               label_volume=labels, scene=scene, sim_to_cbct=sim_to_cbct,
                               force_target_n=args.force_n, settle_steps=300)
    print(f"wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
