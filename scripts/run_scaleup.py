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
    side_anchor_curves, surface_curves_from_points, pose_surface_deviation,
)
from deepussim.pipeline.scaleup import generate_dataset


def load_config(path: str | None):
    if not path:
        return RendererParams(), ProbeGeometry()
    cfg = yaml.safe_load(Path(path).read_text())
    params = RendererParams(**cfg.get("renderer", {}))
    geom = ProbeGeometry(**cfg.get("geometry", {}))
    return params, geom


def load_contact_poses_cbct(args, mesh):
    """Real in-contact probe poses mapped into the CBCT mm frame (``T_cbct_from_probe``).

    Loads the contact-frame EE poses (pre-extracted ``--sequences``, else raw ``--bags``), seats
    the phantom onto them (the same placement the sim path builds), and returns the probe poses in
    CBCT mm. Needed by ``--trajectory surface-curves`` to borrow the real *down-press* orientation
    on the no-sim path (the sim path already has these from its own loading).
    """
    from deepussim.calib import T_WORLD_FROM_CBCT, seat_phantom_placement
    from deepussim.calib.transforms import T_EE_FROM_PROBE
    from deepussim.calib.placement import meters_to_mm, sim_pose_to_cbct
    from deepussim.geometry import invert, compose

    if args.sequences:
        ee = []
        for s in args.sequences:
            d = np.load(s, allow_pickle=True)
            P, contact = d["poses"], d["contact"].astype(bool)
            ee += [P[i] for i in range(len(P)) if contact[i]]
        ee_poses = np.asarray(ee, dtype=float)
    else:
        from deepussim.data.rosbag import extract_sequence   # needs the optional 'rosbags' lib
        ee_poses = np.asarray([f.pose for bag in args.bags
                               for f in extract_sequence(bag).frames if f.contact], dtype=float)
    if len(ee_poses) == 0:
        raise SystemExit("--trajectory surface-curves needs real contact poses; none found in "
                         f"{args.sequences or args.bags} (pass --sequences or --bags)")
    face = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses])
    s2c = meters_to_mm(invert(seat_phantom_placement(mesh, face, T_WORLD_FROM_CBCT)))
    return np.array([sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), s2c) for p in ee_poses])


def surface_curves(args, mesh, contact_poses):
    """Build the ``surface-curves`` trajectory (CBCT mm) and optionally drop the Tier-B frontier.

    Auto-seeds anchor curves that sweep from the scanned patch onto the lateral sides
    (:func:`side_anchor_curves`), drapes/orients them with the real down-press
    (:func:`surface_curves_from_points`), and — if ``--max-dev-deg`` is set — keeps only poses whose
    surface-turn from the patch (:func:`pose_surface_deviation`) is within it (Tier A,
    renderer-trusted), dropping the lateral Tier-B poses that need real-frame validation first.
    """
    curves = side_anchor_curves(mesh, contact_poses, n_curves=args.lines, n_anchors=args.curve_anchors,
                                reach=args.curve_reach, along_frac=args.span_frac)
    poses = surface_curves_from_points(mesh, curves, contact_poses=contact_poses,
                                       points_per_curve=args.per_line, standoff_mm=args.standoff_mm)
    _, dev = pose_surface_deviation(mesh, poses, contact_poses)
    n_b = int(((dev > args.tier_b_deg) & (dev <= 90)).sum()); n_x = int((dev > 90).sum())
    print(f"surface-curves: {len(poses)} poses; Tier @ {args.tier_b_deg:g} deg -> "
          f"A {len(poses) - n_b - n_x}, B(lateral) {n_b}, broken(>90) {n_x}")
    if args.max_dev_deg is not None:
        kept = [p for p, d in zip(poses, dev) if d <= args.max_dev_deg]
        print(f"  --max-dev-deg {args.max_dev_deg:g}: kept {len(kept)}/{len(poses)} (dropped "
              f"{len(poses) - len(kept)} lateral)")
        poses = kept
    return poses


def cbct_trajectory(args, volume, mesh=None):
    """The generated probe trajectory, in the CBCT mm frame (``T_cbct_from_probe`` poses).

    Surface-constrained (``surface``/``raster``, needs the phantom mesh) or a straight
    ``center-sweep`` across the volume. Used both for the no-sim reslice path *and*, mapped
    into the sim world, to drive the arm — so "where the probe is" and "which slice we image"
    come from one source. ``mesh`` may be a preloaded trimesh to avoid re-reading the STL.
    """
    if args.trajectory in ("surface", "raster", "surface-curves"):
        if mesh is None:
            if not args.mesh:
                raise SystemExit(f"--trajectory {args.trajectory} requires --mesh (phantom STL, CBCT mm)")
            import trimesh
            mesh = trimesh.load(args.mesh)
        if args.trajectory == "surface-curves":
            return surface_curves(args, mesh, load_contact_poses_cbct(args, mesh))
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
    ap.add_argument("--renderer-ckpt", help="learned-renderer generator.pt; if given, the CUT "
                    "generator produces the US appearance instead of the physics model")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=64, help="number of poses to sample")
    ap.add_argument("--trajectory",
                    choices=["replay", "contact", "surface-curves", "center-sweep", "surface", "raster"],
                    default=None,
                    help="pose source. 'replay' (sim only, default with --sim) drives the arm "
                         "along the real rosbag poses; 'contact' (sim) generates a reachable "
                         "raster over the face the arm actually scanned (the real contact patch); "
                         "'surface-curves' LEAVES the patch — drapes curves onto the lateral sides "
                         "the real probe can't reach, still down-press-oriented (needs --bags/"
                         "--sequences for orientation; Tier A/B by --tier-b-deg); 'surface'/'raster' "
                         "generate over the mesh's geometric +z top (--mesh); 'center-sweep' "
                         "(default without --sim) is a straight sweep across the volume. Generated "
                         "trajectories work for both the no-sim reslice path and (mapped into the "
                         "sim world) the --sim force channel.")
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
    ap.add_argument("--curve-reach", type=float, default=2.0,
                    help="surface-curves: how far the outer curves fan past the scanned patch "
                         "(x the patch half-width; >1 leaves the patch onto the lateral sides)")
    ap.add_argument("--curve-anchors", type=int, default=6,
                    help="surface-curves: anchor points per curve (the spline is fit through these)")
    ap.add_argument("--tier-b-deg", type=float, default=35.0,
                    help="surface-curves: surface-turn from the scanned patch (deg) above which a "
                         "pose is the lateral 'Tier B' frontier (renderer-untrusted); count/split")
    ap.add_argument("--max-dev-deg", type=float, default=None,
                    help="surface-curves: if set, DROP poses whose surface-turn exceeds this (keep "
                         "only Tier A, renderer-trusted); omit to keep all poses (Tier B included)")
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
                    help="sim: rosbag(s) whose contact-frame EE poses are replayed as probe targets "
                         "(needs the 'rosbags' lib; not installed in the GPU container — prefer "
                         "--sequences there)")
    ap.add_argument("--sequences", nargs="+",
                    help="sim: pre-extracted sequence .npz (keys: poses, contact) — the EE poses "
                         "and contact mask, already parsed from the bags. Use INSTEAD of --bags to "
                         "avoid the rosbags dependency (e.g. on the Apptainer image).")
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

    renderer = None
    if args.renderer_ckpt:
        from deepussim.renderer.neural import NeuralRenderer
        renderer = NeuralRenderer(args.renderer_ckpt)
        print(f"using learned renderer {args.renderer_ckpt} (epoch {renderer.epoch}) on {renderer.device}")

    if not args.sim:
        # No-sim: poses live directly in the CBCT frame (mm); reslice + render + free mask.
        poses = cbct_trajectory(args, volume)
        if args.save_trajectory:
            save_trajectory(args.save_trajectory, poses)
        written = generate_dataset(args.out, volume, poses, geom, params, renderer=renderer,
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
    from deepussim.geometry import mat_to_quat, invert, compose

    if not args.mesh:
        raise SystemExit("--sim requires --mesh (phantom surface STL in CBCT mm)")

    mesh = trimesh.load(args.mesh)                                   # CBCT mm

    # Real contact-frame EE poses (world, m). Prefer pre-extracted sequence .npz (no rosbags dep,
    # works in the GPU container); fall back to parsing the raw bags. Both yield the same poses.
    if args.sequences:
        ee_poses = []
        for s in args.sequences:
            d = np.load(s, allow_pickle=True)
            P, contact = d["poses"], d["contact"].astype(bool)
            ee_poses += [P[i] for i in range(len(P)) if contact[i]]
        ee_poses = np.asarray(ee_poses, dtype=float)
        src = args.sequences
    else:
        from deepussim.data.rosbag import extract_sequence  # needs the optional 'rosbags' lib
        ee_poses = np.asarray([f.pose for bag in args.bags
                               for f in extract_sequence(bag).frames if f.contact], dtype=float)
        src = args.bags
    if len(ee_poses) == 0:
        raise SystemExit(f"no contact frames found in {src}")

    # Phantom placement (matches scripts/view_sim.py): the measured T_WORLD_FROM_CBCT stands the
    # phantom on end (its CBCT-scan frame is rolled 90 deg from our DICOM-LPS volume); the real
    # rig has it LYING, so seat_phantom_placement tips it onto its side and seats the surface onto
    # the real contact cloud. The reslice bridge is derived from the SAME transform, so mesh, CBCT
    # volume and probe poses stay consistent.
    all_origins = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses])
    T_world_from_cbctm = seat_phantom_placement(mesh, all_origins, T_WORLD_FROM_CBCT)
    sim_to_cbct = meters_to_mm(invert(T_world_from_cbctm))           # achieved world (m) -> CBCT (mm)

    # Trajectory the arm follows (nominal targets in the sim world, m):
    if args.trajectory == "replay":
        # the EE poses the arm actually reached on the phantom (reachable + on-surface),
        # subsampled to --n.
        pick = np.linspace(0, len(ee_poses) - 1, args.n, dtype=int)
        sim_poses = [compose(ee_poses[i], T_EE_FROM_PROBE) for i in pick]
        nominal_cbct = [sim_pose_to_cbct(p, sim_to_cbct) for p in sim_poses]
    elif args.trajectory == "contact":
        # generate a reachable raster over the face the arm actually scanned: map the real
        # contact *poses* into the CBCT frame, fit a patch there, and densify (the surface/raster
        # +z-top samplers land on a different, often-unreachable face). Orientation is taken from
        # the real EE poses, not mesh normals — the real probe presses straight down and does not
        # follow the local surface normal (see sampling.contact_raster_ee).
        rc_poses = np.array([sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), sim_to_cbct)
                             for p in ee_poses])
        nominal_cbct = contact_raster_ee(mesh, rc_poses, n_lines=args.lines,
                                         n_per_line=args.per_line, standoff_mm=args.standoff_mm)
        sim_poses = [compose(T_world_from_cbctm, mm_to_meters(T)) for T in nominal_cbct]
    elif args.trajectory == "surface-curves":
        # leave the scanned patch: draped curves onto the lateral sides, still down-press-oriented
        # (renderer in-regime on orientation, extrapolating only on position; Tier B = the sides).
        rc_poses = np.array([sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), sim_to_cbct)
                             for p in ee_poses])
        nominal_cbct = surface_curves(args, mesh, rc_poses)
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

    written = generate_dataset(args.out, volume, sim_poses, geom, params, renderer=renderer,
                               label_volume=labels, scene=scene, sim_to_cbct=sim_to_cbct,
                               force_target_n=args.force_n, settle_steps=300)
    print(f"wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
