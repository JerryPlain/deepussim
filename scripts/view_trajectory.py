#!/usr/bin/env python
"""Render a probe trajectory over the phantom — a publication-quality figure.

Either *generate* a surface-constrained trajectory from the phantom mesh (the same
`pipeline.sampling` that `run_scaleup` uses) or *load* one a run already saved with
`run_scaleup --save-trajectory` (`--trajectory-file`). Writes a vector **PDF** (for LaTeX) and a
PNG, and prints how well the poses sit on the surface (standoff + axial · inward-normal, where
1.0 is a perpendicular press).

The figure uses serif / Computer-Modern (mathtext) fonts so it matches a LaTeX document; pass
`--usetex` to render with a real LaTeX install if you have one.

    # generate a raster, save the figure (PDF+PNG) and the trajectory:
    python scripts/view_trajectory.py --mesh data/cbct/phantom_surface.stl --trajectory raster \
        --out data/trajectories/raster.pdf --save-trajectory data/trajectories/raster.npz

    # just render a trajectory a run already produced:
    python scripts/view_trajectory.py --trajectory-file data/trajectories/raster.npz \
        --mesh data/cbct/phantom_surface.stl --out data/trajectories/raster.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ACCENT = "#b91c1c"     # axial arrows
PATH_C = "0.35"        # scan-path line
CMAP = "viridis"       # scan order


def _paper_style(usetex: bool) -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,        # embed TrueType so the PDF is portable
        "ps.fonttype": 42,
    })
    if usetex:
        from shutil import which
        if which("latex"):
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        else:
            print("--usetex requested but no system latex found; using mathtext CM")


def _decimate(mesh, target_faces=3500):
    """Reduce the mesh to ~target_faces for a light translucent surface (best-effort)."""
    n = len(mesh.faces)
    if n <= target_faces:
        return mesh
    for call in (lambda: mesh.simplify_quadric_decimation(1.0 - target_faces / n),  # fraction-to-remove API
                 lambda: mesh.simplify_quadric_decimation(face_count=target_faces),  # face-count API
                 lambda: mesh.simplify_quadric_decimation(target_faces)):
        try:
            d = call()
            if d is not None and len(d.faces) > 0:
                return d
        except Exception:
            continue
    return None


def _contact_poses_cbct(mesh, bags):
    """The real in-contact probe poses (4x4), mapped into the CBCT mm frame (via the placement)."""
    from deepussim.data.rosbag import extract_sequence
    from deepussim.calib import T_WORLD_FROM_CBCT, T_EE_FROM_PROBE, seat_phantom_placement
    from deepussim.calib.placement import meters_to_mm, sim_pose_to_cbct
    from deepussim.geometry import invert, compose

    frames = [f for bag in bags for f in extract_sequence(bag).frames if f.contact]
    if not frames:
        raise SystemExit(f"no contact frames in {bags}")
    face = np.array([compose(f.pose, T_EE_FROM_PROBE)[:3, 3] for f in frames])
    s2c = meters_to_mm(invert(seat_phantom_placement(mesh, face, T_WORLD_FROM_CBCT)))
    return np.array([sim_pose_to_cbct(compose(f.pose, T_EE_FROM_PROBE), s2c) for f in frames])


def build_trajectory(args, mesh):
    from deepussim.pipeline.sampling import (surface_raster, surface_sweep, contact_raster_ee,
                                             top_sweep_endpoints)

    if args.trajectory == "contact":
        rc_poses = _contact_poses_cbct(mesh, args.bags)
        return contact_raster_ee(mesh, rc_poses, n_lines=args.lines, n_per_line=args.per_line,
                                 standoff_mm=args.standoff_mm)
    if args.trajectory == "raster":
        return surface_raster(mesh, axis=args.sweep_axis, span_frac=args.span_frac,
                              cross_frac=args.cross_frac, n_lines=args.lines,
                              n_per_line=args.per_line, standoff_mm=args.standoff_mm)
    start, end = top_sweep_endpoints(mesh, axis=args.sweep_axis, span_frac=args.span_frac)
    return surface_sweep(mesh, start, end, args.n, standoff_mm=args.standoff_mm)


def surface_stats(mesh, P, A):
    import trimesh

    q = trimesh.proximity.ProximityQuery(mesh)
    _, _, fid = q.on_surface(P)
    inward_normal = -mesh.face_normals[fid]
    standoff = -q.signed_distance(P)                       # signed_distance is +inside the mesh
    cos = np.einsum("ij,ij->i", A, inward_normal)
    return standoff, cos


def _equal_3d(ax, P):
    """Equal aspect for a 3D axis around the trajectory's bounding cube."""
    c = P.mean(0)
    r = (P.max(0) - P.min(0)).max() * 0.6 + 10.0
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render(P, A, deci, out: Path, title: str | None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    order = np.arange(len(P))
    fig = plt.figure(figsize=(7.2, 3.0))
    # explicit rectangles so the 3D and the equal-aspect 2D panel stay aligned
    ax = fig.add_axes([0.00, 0.02, 0.52, 0.90], projection="3d")
    if deci is not None:
        tris = Poly3DCollection(np.asarray(deci.vertices)[np.asarray(deci.faces)],
                                alpha=0.16, facecolor="#9fb3c8", edgecolor="none")
        ax.add_collection3d(tris)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", c=PATH_C, lw=0.7, alpha=0.7, zorder=2)
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], A[:, 0], A[:, 1], A[:, 2], length=12.0,
              color=ACCENT, linewidth=0.6, arrow_length_ratio=0.35, zorder=3)
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=order, cmap=CMAP, s=12,
                    depthshade=False, zorder=4)
    _equal_3d(ax, P)
    ax.set_xlabel("$x$ (mm)", labelpad=-2); ax.set_ylabel("$y$ (mm)", labelpad=-2)
    ax.set_zlabel("$z$ (mm)", labelpad=-4)
    ax.tick_params(pad=-1)
    ax.view_init(elev=22, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_edgecolor("0.85")
        pane._axinfo["grid"].update(color="0.92", linewidth=0.4)
    ax.set_title("(a) perspective")

    # (b) top view 2D — the raster pattern reads crisply in print (axial is into the page here,
    # so the arrows live in panel (a); panel (b) shows the scan path + order cleanly)
    ax2 = fig.add_axes([0.60, 0.16, 0.30, 0.74])
    if deci is not None:
        V = np.asarray(deci.vertices)
        ax2.scatter(V[:, 0], V[:, 1], s=1.5, c="0.85", alpha=0.5, edgecolors="none", zorder=0)
    ax2.plot(P[:, 0], P[:, 1], "-", c=PATH_C, lw=0.7, alpha=0.7, zorder=2)
    ax2.scatter(P[:, 0], P[:, 1], c=order, cmap=CMAP, s=13, zorder=4)
    ax2.set_aspect("equal", adjustable="datalim")     # fill the rect, pad data limits (no float)
    ax2.set_xlabel("$x$ (mm)"); ax2.set_ylabel("$y$ (mm)")
    ax2.set_title("(b) top view")
    for s in ax2.spines.values():
        s.set_color("0.6")

    cax = fig.add_axes([0.925, 0.20, 0.016, 0.62])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("scan order", fontsize=9); cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)

    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = out.with_suffix(".pdf"); png = out.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, bbox_inches="tight")
    print(f"saved figure -> {pdf}  and  {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="phantom surface STL (CBCT mm): generate from it and/or draw it")
    ap.add_argument("--trajectory-file", help="load a saved trajectory (.npz with poses_cbct_mm)")
    ap.add_argument("--trajectory", choices=["contact", "surface", "raster"], default="contact",
                    help="generate: 'contact' (default) rasters the face the arm actually scanned "
                         "(needs --bags); 'surface'/'raster' the mesh's geometric +z top")
    ap.add_argument("--bags", nargs="+",
                    default=["data/rosbags/phantom.bag", "data/rosbags/phantom1.bag"],
                    help="contact: rosbag(s) whose contact frames define the reachable scan patch")
    ap.add_argument("--sweep-axis", type=int, default=0, help="sweep axis (0=x, 1=y)")
    ap.add_argument("--span-frac", type=float, default=0.6)
    ap.add_argument("--cross-frac", type=float, default=0.4)
    ap.add_argument("--lines", type=int, default=5, help="raster: number of parallel sweep lines")
    ap.add_argument("--per-line", type=int, default=24, help="raster: poses per sweep line")
    ap.add_argument("--standoff-mm", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=64, help="surface (single glide): number of poses")
    ap.add_argument("--save-trajectory", help="also save the generated poses to this .npz")
    ap.add_argument("--out", default="data/trajectories/trajectory.pdf",
                    help="output figure path (a .pdf and a .png are written)")
    ap.add_argument("--title", help="optional figure suptitle")
    ap.add_argument("--usetex", action="store_true", help="render with a real LaTeX install")
    args = ap.parse_args()

    _paper_style(args.usetex)
    import matplotlib
    matplotlib.use("Agg")

    mesh = None
    if args.mesh:
        import trimesh
        mesh = trimesh.load(args.mesh)

    if args.trajectory_file:
        poses = np.asarray(np.load(args.trajectory_file)["poses_cbct_mm"], dtype=float)
        src = f"loaded {Path(args.trajectory_file).name}"
    else:
        if mesh is None:
            raise SystemExit("generating a trajectory needs --mesh (or pass --trajectory-file)")
        poses = np.asarray(build_trajectory(args, mesh), dtype=float)
        src = f"generated {args.trajectory}"
        if args.save_trajectory:
            sp = Path(args.save_trajectory); sp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp, poses_cbct_mm=poses)
            print(f"saved trajectory ({len(poses)} poses, CBCT mm) -> {sp}")

    P, A = poses[:, :3, 3], poses[:, :3, 2]
    print(f"{src}: {len(poses)} poses, extent (mm) {np.ptp(P, 0).round(1)}")
    if mesh is not None:
        standoff, cos = surface_stats(mesh, P, A)
        print(f"  standoff (mm): mean {standoff.mean():.2f}   "
              f"axial.inward-normal: mean {cos.mean():.3f}  min {cos.min():.3f}  (1 = perpendicular)")

    deci = _decimate(mesh) if mesh is not None else None
    if mesh is not None and deci is None:
        print("  (mesh decimation unavailable; drawing trajectory only)")
    render(P, A, deci, Path(args.out), args.title)


if __name__ == "__main__":
    main()
