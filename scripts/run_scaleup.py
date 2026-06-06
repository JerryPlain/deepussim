#!/usr/bin/env python
"""Step 5 entrypoint: generate a US dataset by reslicing + rendering a CBCT volume.

No-sim by default (geometric poses, no force). Pass ``--sim`` to drive a Genesis scene
for reachable poses + contact force (requires Genesis + an implemented sim.scene). The
trajectory (``--trajectory``) is one source for both paths: a generated surface/raster sweep,
or the real rosbag ``replay`` (sim only).

    # no-sim: surface-constrained sweep over the real phantom (geometry branch):
    python scripts/run_scaleup.py \
        --volume data/cbct/intensity.nrrd --labels data/cbct/labels.nrrd \
        --mesh data/cbct/phantom_surface.stl --trajectory surface \
        --config configs/renderer.yaml --out data/ds --n 64

    # sim: drive the SAME generated raster over the phantom and read contact force (needs a GPU):
    python scripts/run_scaleup.py --sim \
        --volume data/cbct/intensity.nrrd --labels data/cbct/labels.nrrd \
        --mesh data/cbct/phantom_surface.stl --trajectory raster \
        --config configs/renderer.yaml --out data/ds_sim --headless \
        --save-trajectory data/trajectories/raster.npz
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
    linear_sweep, surface_sweep, surface_raster, contact_raster_ee, top_sweep_endpoints,
)
from deepussim.pipeline.scaleup import generate_dataset


def load_config(path: str | None):
    if not path:
        return RendererParams(), ProbeGeometry()
    cfg = yaml.safe_load(Path(path).read_text())
    params = RendererParams(**cfg.get("renderer", {}))
    geom = ProbeGeometry(**cfg.get("geometry", {}))
    return params, geom


def cbct_trajectory(args, volume, mesh=None):
    """The generated probe trajectory, in the CBCT mm frame (``T_cbct_from_probe`` poses).

    Surface-constrained (``surface``/``raster``, needs the phantom mesh) or a straight
    ``center-sweep`` across the volume. Used both for the no-sim reslice path *and*, mapped
    into the sim world, to drive the arm — so "where the probe is" and "which slice we image"
    come from one source. ``mesh`` may be a preloaded trimesh to avoid re-reading the STL.
    """
    if args.trajectory in ("surface", "raster"):
        if mesh is None:
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


def save_trajectory(path, poses_cbct_mm) -> None:
    """Save the generated trajectory (``T_cbct_from_probe`` poses, mm) to a ``.npz``.

    The trajectory is deterministic from (mesh + params), so this is optional — for
    inspection or to reuse the exact same poses across runs; the achieved poses are also
    written into the dataset itself.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, poses_cbct_mm=np.asarray(poses_cbct_mm, dtype=float))
    print(f"saved trajectory ({len(poses_cbct_mm)} poses, CBCT mm) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", required=True, help="CBCT intensity volume (.nrrd/.nii.gz)")
    ap.add_argument("--labels", help="CBCT label volume (.nrrd/.nii.gz) for anatomy masks")
    ap.add_argument("--config", help="renderer/geometry YAML (configs/renderer.yaml)")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=64, help="number of poses to sample")
    ap.add_argument("--trajectory",
                    choices=["replay", "contact", "center-sweep", "surface", "raster"],
                    default=None,
                    help="pose source. 'replay' (sim only, default with --sim) drives the arm "
                         "along the real rosbag poses; 'contact' (sim) generates a reachable "
                         "raster over the face the arm actually scanned (the real contact patch); "
                         "'surface'/'raster' generate over the mesh's geometric +z top (--mesh); "
                         "'center-sweep' (default without --sim) is a straight sweep across the "
                         "volume. Generated trajectories work for both the no-sim reslice path "
                         "and (mapped into the sim world) the --sim force channel.")
    ap.add_argument("--mesh", help="phantom surface STL (CBCT mm) for --trajectory surface/raster")
    ap.add_argument("--save-trajectory", help="also save the generated trajectory "
                    "(T_cbct_from_probe poses, mm) to this .npz, e.g. data/trajectories/raster.npz")
    ap.add_argument("--sweep-axis", type=int, default=0, help="surface sweep axis (0=x, 1=y)")
    ap.add_argument("--span-frac", type=float, default=0.6, help="along-sweep span fraction")
    ap.add_argument("--cross-frac", type=float, default=0.4,
                    help="raster: cross-sweep span fraction (coverage across lines)")
    ap.add_argument("--lines", type=int, default=5, help="raster: number of parallel sweep lines")
    ap.add_argument("--per-line", type=int, default=24, help="raster: poses per sweep line")
    ap.add_argument("--standoff-mm", type=float, default=2.0, help="probe standoff off the surface")
    ap.add_argument("--sim", action="store_true",
                    help="drive a Genesis scene: the FR3 follows the trajectory (--trajectory: "
                         "the real rosbag replay, or a generated surface/raster sweep) onto the "
                         "phantom (placed lying + seated on the contacts) and force-servos to "
                         "--force-n; requires --mesh (CBCT mm STL) and --bags (for the placement)")
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

    # Resolve the default trajectory by mode, then validate: replay/contact are sim-only
    # (they need the real rosbag contacts + the seated placement that the sim path builds).
    if args.trajectory is None:
        args.trajectory = "replay" if args.sim else "center-sweep"
    if args.trajectory in ("replay", "contact") and not args.sim:
        raise SystemExit(f"--trajectory {args.trajectory} only applies with --sim "
                         "(it needs the real contacts + placement)")

    volume = load_volume(args.volume)
    labels = load_volume(args.labels) if args.labels else None
    params, geom = load_config(args.config)

    if not args.sim:
        # No-sim: poses live directly in the CBCT frame (mm); reslice + render + free mask.
        poses = cbct_trajectory(args, volume)
        if args.save_trajectory:
            save_trajectory(args.save_trajectory, poses)
        written = generate_dataset(args.out, volume, poses, geom, params,
                                   label_volume=labels)
        print(f"wrote {written} samples to {args.out}")
        return

    # Sim path: the arm follows nominal probe targets in the sim world (metres); physics gives
    # the achieved pose + contact force, then the placement bridge maps it back to CBCT mm.
    import trimesh
    from deepussim.sim.scene import UltrasoundScene, SceneConfig
    from deepussim.calib import T_WORLD_FROM_CBCT, seat_phantom_placement
    from deepussim.calib.transforms import T_EE_FROM_PROBE
    from deepussim.calib.placement import meters_to_mm, mm_to_meters, sim_pose_to_cbct
    from deepussim.data.rosbag import extract_sequence
    from deepussim.geometry import mat_to_quat, invert, compose

    if not args.mesh:
        raise SystemExit("--sim requires --mesh (phantom surface STL in CBCT mm)")

    mesh = trimesh.load(args.mesh)                                   # CBCT mm

    # Phantom placement (matches scripts/view_sim.py): the measured T_WORLD_FROM_CBCT stands the
    # phantom on end (its CBCT-scan frame is rolled 90 deg from our DICOM-LPS volume); the real
    # rig has it LYING, so seat_phantom_placement tips it onto its side and seats the surface onto
    # the real contact cloud (always taken from the bags). The reslice bridge is derived from the
    # SAME transform, so mesh, CBCT volume and probe poses stay consistent.
    frames = [f for bag in args.bags for f in extract_sequence(bag).frames if f.contact]
    if not frames:
        raise SystemExit(f"no contact frames found in {args.bags}")
    all_origins = np.array([compose(f.pose, T_EE_FROM_PROBE)[:3, 3] for f in frames])
    T_world_from_cbctm = seat_phantom_placement(mesh, all_origins, T_WORLD_FROM_CBCT)
    sim_to_cbct = meters_to_mm(invert(T_world_from_cbctm))           # achieved world (m) -> CBCT (mm)

    # Trajectory the arm follows (nominal targets in the sim world, m):
    if args.trajectory == "replay":
        # the EE poses the arm actually reached on the phantom (reachable + on-surface),
        # subsampled to --n.
        pick = np.linspace(0, len(frames) - 1, args.n, dtype=int)
        sim_poses = [compose(frames[i].pose, T_EE_FROM_PROBE) for i in pick]
        nominal_cbct = [sim_pose_to_cbct(p, sim_to_cbct) for p in sim_poses]
    elif args.trajectory == "contact":
        # generate a reachable raster over the face the arm actually scanned: map the real
        # contact *poses* into the CBCT frame, fit a patch there, and densify (the surface/raster
        # +z-top samplers land on a different, often-unreachable face). Orientation is taken from
        # the real EE poses, not mesh normals — the real probe presses straight down and does not
        # follow the local surface normal (see sampling.contact_raster_ee).
        rc_poses = np.array([sim_pose_to_cbct(compose(f.pose, T_EE_FROM_PROBE), sim_to_cbct)
                             for f in frames])
        nominal_cbct = contact_raster_ee(mesh, rc_poses, n_lines=args.lines,
                                         n_per_line=args.per_line, standoff_mm=args.standoff_mm)
        sim_poses = [compose(T_world_from_cbctm, mm_to_meters(T)) for T in nominal_cbct]
    else:
        # a generated CBCT-frame trajectory, mapped into the sim world to drive the arm
        # (inverse of sim_to_cbct); out-of-reach / non-contacting poses are dropped downstream.
        nominal_cbct = cbct_trajectory(args, volume, mesh=mesh)
        sim_poses = [compose(T_world_from_cbctm, mm_to_meters(T)) for T in nominal_cbct]
    if args.save_trajectory:
        save_trajectory(args.save_trajectory, nominal_cbct)

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

    written = generate_dataset(args.out, volume, sim_poses, geom, params,
                               label_volume=labels, scene=scene, sim_to_cbct=sim_to_cbct,
                               force_target_n=args.force_n, settle_steps=300)
    print(f"wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
