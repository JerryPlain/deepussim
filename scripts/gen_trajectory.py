#!/usr/bin/env python
"""Generate a probe trajectory over the phantom and save it (Stage 2 pose sampling).

Generation only — no plotting (use ``plot_script.plots_reslice.trajectories
--trajectory-file`` to draw it). Wraps ``deepussim.pipeline.sampling`` (the same generator
``run_scaleup`` uses) and writes the poses (CBCT mm) to a ``.npz`` with key ``poses_cbct_mm``.

Modes:
  * ``raster`` / ``surface``  — regular sweep over the mesh's geometric +z top (mesh only);
  * ``contact``               — raster the face the arm actually scanned (needs --sequences);
  * ``surface-curves``        — drape curves out onto the lateral sides (needs --sequences),
                                extending past the scanned patch (the Stage-2 frontier).

    python scripts/gen_trajectory.py --mesh data/cbct_20260612/phantom_surface.stl --trajectory raster \
        --out data/trajectories/raster.npz
    python scripts/gen_trajectory.py --mesh data/cbct_20260612/phantom_surface.stl --trajectory surface-curves \
        --sequences data/sequences/scan*.npz --out data/trajectories/curves.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def contact_poses_cbct(mesh, sequences) -> np.ndarray:
    """Real in-contact probe poses (4x4) mapped into the CBCT mm frame (via the seated placement)."""
    from deepussim.calib import T_EE_FROM_PROBE, T_WORLD_FROM_CBCT, seat_phantom_placement
    from deepussim.calib.placement import meters_to_mm, sim_pose_to_cbct
    from deepussim.geometry import compose, invert

    ee = []
    for s in sequences:
        d = np.load(s, allow_pickle=True)
        poses, contact = d["poses"], d["contact"].astype(bool)
        ee += [poses[i] for i in range(len(poses)) if contact[i]]
    ee_poses = np.asarray(ee, dtype=float)
    if len(ee_poses) == 0:
        raise SystemExit(f"no contact frames in {sequences}")
    face = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses])
    s2c = meters_to_mm(invert(seat_phantom_placement(mesh, face, T_WORLD_FROM_CBCT)))
    return np.array([sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), s2c) for p in ee_poses])


def build_trajectory(args, mesh) -> np.ndarray:
    """Generate the requested trajectory as (N, 4, 4) poses in the CBCT mm frame."""
    from deepussim.pipeline.sampling import (contact_raster_ee, side_anchor_curves, surface_raster,
                                             surface_curves_from_points, surface_sweep,
                                             top_sweep_endpoints)

    if args.trajectory == "surface-curves":
        rc = contact_poses_cbct(mesh, args.sequences)
        curves = side_anchor_curves(mesh, rc, n_curves=args.lines, n_anchors=args.curve_anchors,
                                    reach=args.curve_reach, along_frac=args.span_frac)
        return surface_curves_from_points(mesh, curves, contact_poses=rc,
                                          points_per_curve=args.per_line, standoff_mm=args.standoff_mm)
    if args.trajectory == "contact":
        rc = contact_poses_cbct(mesh, args.sequences)
        return contact_raster_ee(mesh, rc, n_lines=args.lines, n_per_line=args.per_line,
                                 standoff_mm=args.standoff_mm)
    if args.trajectory == "raster":
        return surface_raster(mesh, axis=args.sweep_axis, span_frac=args.span_frac,
                              cross_frac=args.cross_frac, n_lines=args.lines,
                              n_per_line=args.per_line, standoff_mm=args.standoff_mm)
    start, end = top_sweep_endpoints(mesh, axis=args.sweep_axis, span_frac=args.span_frac)
    return surface_sweep(mesh, start, end, args.n, standoff_mm=args.standoff_mm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="phantom surface STL (CBCT mm)")
    ap.add_argument("--trajectory", choices=["contact", "surface-curves", "surface", "raster"],
                    default="raster")
    ap.add_argument("--sequences", nargs="+", help="contact/surface-curves: extracted sequence .npz")
    ap.add_argument("--out", required=True, help="output .npz (key: poses_cbct_mm)")
    # Generation parameters (same meaning as deepussim.pipeline.sampling / the old view_trajectory).
    ap.add_argument("--lines", type=int, default=5, help="raster: parallel sweep lines / curves")
    ap.add_argument("--per-line", type=int, default=24, help="poses per line / curve")
    ap.add_argument("--standoff-mm", type=float, default=2.0)
    ap.add_argument("--sweep-axis", type=int, default=0, help="0=x, 1=y")
    ap.add_argument("--span-frac", type=float, default=0.6)
    ap.add_argument("--cross-frac", type=float, default=0.4)
    ap.add_argument("--curve-reach", type=float, default=2.0, help="surface-curves: fan-out past patch")
    ap.add_argument("--curve-anchors", type=int, default=6, help="surface-curves: anchors per curve")
    ap.add_argument("--n", type=int, default=64, help="surface (single glide): number of poses")
    args = ap.parse_args()

    import trimesh
    mesh = trimesh.load(args.mesh)
    poses = np.asarray(build_trajectory(args, mesh), dtype=float)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, poses_cbct_mm=poses)
    P = poses[:, :3, 3]
    print(f"[gen_trajectory] {args.trajectory}: {len(poses)} poses, extent (mm) {np.ptp(P, 0).round(1)}")
    print(f"[gen_trajectory] saved -> {out}  (plot with: "
          f"python -m plot_script.plots_reslice.trajectories --trajectory-file {out})")


if __name__ == "__main__":
    main()
