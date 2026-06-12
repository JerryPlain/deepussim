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


def _contact_poses_cbct(mesh, bags=None, sequences=None):
    """The real in-contact probe poses (4x4), mapped into the CBCT mm frame (via the placement).

    Prefers pre-extracted ``sequences`` (.npz with ``poses``/``contact``, no rosbags dep — works in
    the GPU container); falls back to parsing raw ``bags``. Both yield the same contact poses.
    """
    from deepussim.calib import T_WORLD_FROM_CBCT, T_EE_FROM_PROBE, seat_phantom_placement
    from deepussim.calib.placement import meters_to_mm, sim_pose_to_cbct
    from deepussim.geometry import invert, compose

    if sequences:
        ee = []
        for s in sequences:
            d = np.load(s, allow_pickle=True)
            P, contact = d["poses"], d["contact"].astype(bool)
            ee += [P[i] for i in range(len(P)) if contact[i]]
        ee_poses = np.asarray(ee, dtype=float)
    else:
        from deepussim.data.rosbag import extract_sequence
        ee_poses = np.asarray([f.pose for bag in bags
                               for f in extract_sequence(bag).frames if f.contact], dtype=float)
    if len(ee_poses) == 0:
        raise SystemExit(f"no contact frames in {sequences or bags}")
    face = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses])
    s2c = meters_to_mm(invert(seat_phantom_placement(mesh, face, T_WORLD_FROM_CBCT)))
    return np.array([sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), s2c) for p in ee_poses])


def build_trajectory(args, mesh):
    from deepussim.pipeline.sampling import (surface_raster, surface_sweep, contact_raster_ee,
                                             surface_curves_from_points, side_anchor_curves,
                                             top_sweep_endpoints)

    if args.trajectory == "surface-curves":
        rc_poses = _contact_poses_cbct(mesh, bags=args.bags, sequences=args.sequences)
        curves = side_anchor_curves(mesh, rc_poses, n_curves=args.lines, n_anchors=args.curve_anchors,
                                    reach=args.curve_reach, along_frac=args.span_frac)
        return surface_curves_from_points(mesh, curves, contact_poses=rc_poses,
                                          points_per_curve=args.per_line, standoff_mm=args.standoff_mm)
    if args.trajectory == "contact":
        rc_poses = _contact_poses_cbct(mesh, bags=args.bags, sequences=args.sequences)
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


def _curve_lines(ax, P, seg_len, dims):
    """Draw each curve (a contiguous block of ``seg_len`` poses) as its own light line.

    Avoids the single zig-zag path that jumps between fanned curves; ``dims`` selects the columns
    (3D: (0,1,2); top view: (0,1)). Returns nothing — draws onto ``ax``.
    """
    if not seg_len:
        ax.plot(*[P[:, d] for d in dims], "-", c=PATH_C, lw=0.7, alpha=0.6, zorder=2)
        return
    for s in range(0, len(P), seg_len):
        seg = P[s:s + seg_len]
        ax.plot(*[seg[:, d] for d in dims], "-", c=PATH_C, lw=0.8, alpha=0.55, zorder=2)


def render(P, A, deci, out: Path, title: str | None, cval=None, clabel: str = "scan order",
           cmap: str | None = None, contact=None, seg_len: int | None = None,
           tier_b: float | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    order = np.arange(len(P)) if cval is None else np.asarray(cval)
    cmap = cmap or CMAP
    GREEN = "#1a9850"
    fig = plt.figure(figsize=(8.4, 4.2))
    # fixed y-bands keep every text element in its own lane (no overlap): title / captions / panels.
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="center", fontsize=11.5)
    fig.text(0.255, 0.895, "(a) 3D perspective", ha="center", fontsize=9.5, style="italic")
    fig.text(0.70, 0.895, "(b) top view ($x$–$y$)", ha="center", fontsize=9.5, style="italic")

    # ---- (a) 3D ----------------------------------------------------------------------------
    ax = fig.add_axes([0.01, 0.10, 0.49, 0.74], projection="3d")
    if deci is not None:
        tris = Poly3DCollection(np.asarray(deci.vertices)[np.asarray(deci.faces)],
                                alpha=0.11, facecolor="#9fb3c8", edgecolor="none")
        ax.add_collection3d(tris)
    if contact is not None:                              # the real scanned footprint we leave
        ax.scatter(contact[:, 0], contact[:, 1], contact[:, 2], s=5, c=GREEN, alpha=0.30,
                   edgecolors="none", depthshade=False, zorder=2)
    _curve_lines(ax, P, seg_len, (0, 1, 2))
    step = max(1, len(P) // 10)                          # sparse down-press arrows (de-clutter)
    ax.quiver(P[::step, 0], P[::step, 1], P[::step, 2], A[::step, 0], A[::step, 1], A[::step, 2],
              length=10.0, color="0.45", linewidth=0.5, arrow_length_ratio=0.4, alpha=0.6, zorder=3)
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=order, cmap=cmap, s=15,
                    depthshade=False, zorder=4)
    _equal_3d(ax, P)
    ax.set_xlabel("$x$ (mm)", labelpad=1); ax.set_ylabel("$y$ (mm)", labelpad=1)
    ax.set_zlabel("$z$ (mm)", labelpad=1)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=24, azim=-62)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_edgecolor("0.85")
        pane._axinfo["grid"].update(color="0.93", linewidth=0.4)

    # ---- (b) top view ----------------------------------------------------------------------
    ax2 = fig.add_axes([0.60, 0.17, 0.27, 0.62])
    if deci is not None:
        V = np.asarray(deci.vertices)
        ax2.scatter(V[:, 0], V[:, 1], s=1.4, c="0.86", alpha=0.5, edgecolors="none", zorder=0)
    if contact is not None:
        ax2.scatter(contact[:, 0], contact[:, 1], s=6, c=GREEN, alpha=0.28,
                    edgecolors="none", zorder=1)
    _curve_lines(ax2, P, seg_len, (0, 1))
    ax2.scatter(P[:, 0], P[:, 1], c=order, cmap=cmap, s=15, zorder=4)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("$x$ (mm)", fontsize=9); ax2.set_ylabel("$y$ (mm)", fontsize=9)
    ax2.tick_params(labelsize=7)
    for s in ax2.spines.values():
        s.set_color("0.6")

    # ---- colourbar (with the A|B split marked) ---------------------------------------------
    cax = fig.add_axes([0.905, 0.20, 0.018, 0.55])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(clabel, fontsize=8.5); cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5)
    if tier_b is not None:
        cb.ax.axhline(tier_b, color="0.15", lw=1.0)
        cb.ax.annotate("Tier A", xy=(1.0, tier_b), xytext=(3.2, -8), textcoords="offset points",
                       fontsize=7, color="#2166ac", annotation_clip=False)
        cb.ax.annotate("Tier B", xy=(1.0, tier_b), xytext=(3.2, 4), textcoords="offset points",
                       fontsize=7, color="#b2182b", annotation_clip=False)

    # ---- one-line legend strip at the bottom (its own lane) --------------------------------
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=6,
                      alpha=0.6, label="real scanned patch (what we have)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#2166ac", markersize=6,
                      label="generated: Tier A (trusted)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#b2182b", markersize=6,
                      label="generated: Tier B (lateral — validate)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.46, -0.01))

    pdf = out.with_suffix(".pdf"); png = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, bbox_inches="tight")
    print(f"saved figure -> {pdf}  and  {png}")


def render_tiers(P, dev, deci, contact, out: Path, title: str | None, tier_b: float):
    """Two side-by-side 3D panels: Tier A poses (trusted) | Tier B poses (lateral frontier).

    Same viewpoint and limits in both, the real scanned patch (green) in each for reference — so
    'which generated poses are trusted vs the frontier' reads at a glance. ``dev`` is the per-pose
    surface-turn (deg); the split is at ``tier_b``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    GREEN, BLUE, RED = "#1a9850", "#2166ac", "#b2182b"
    dev = np.asarray(dev)
    panels = [("(a) Tier A — trusted", dev <= tier_b, BLUE),
              ("(b) Tier B — lateral frontier", dev > tier_b, RED)]
    fig = plt.figure(figsize=(8.6, 4.3))
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="center", fontsize=11.5)

    for (cap, mask, col), rect, cx in zip(panels, ([0.01, 0.10, 0.47, 0.74],
                                                   [0.52, 0.10, 0.47, 0.74]), (0.25, 0.76)):
        ax = fig.add_axes(rect, projection="3d")
        if deci is not None:
            tris = Poly3DCollection(np.asarray(deci.vertices)[np.asarray(deci.faces)],
                                    alpha=0.10, facecolor="#9fb3c8", edgecolor="none")
            ax.add_collection3d(tris)
        if contact is not None:
            ax.scatter(contact[:, 0], contact[:, 1], contact[:, 2], s=5, c=GREEN, alpha=0.28,
                       edgecolors="none", depthshade=False, zorder=2)
        Q = P[mask]
        if len(Q):
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=16, c=col, depthshade=False, zorder=4)
        _equal_3d(ax, P)                                  # identical limits in both panels
        ax.set_xlabel("$x$ (mm)", labelpad=1); ax.set_ylabel("$y$ (mm)", labelpad=1)
        ax.set_zlabel("$z$ (mm)", labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=24, azim=-62)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white"); pane.pane.set_edgecolor("0.85")
            pane._axinfo["grid"].update(color="0.93", linewidth=0.4)
        fig.text(cx, 0.89, f"{cap}  ({int(mask.sum())} poses)", ha="center", fontsize=9.5,
                 style="italic", color=col)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=6,
                      alpha=0.6, label="real scanned patch (what we have)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=6,
                      label="Tier A (trusted)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=6,
                      label="Tier B (lateral — validate)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    pdf = out.with_suffix(".pdf"); png = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, bbox_inches="tight")
    print(f"saved split figure -> {pdf}  and  {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="phantom surface STL (CBCT mm): generate from it and/or draw it")
    ap.add_argument("--trajectory-file", help="load a saved trajectory (.npz with poses_cbct_mm)")
    ap.add_argument("--trajectory", choices=["contact", "surface-curves", "surface", "raster"],
                    default="contact",
                    help="generate: 'contact' (default) rasters the face the arm actually scanned "
                         "(needs --bags); 'surface-curves' drapes curves out onto the lateral sides "
                         "(needs --bags, down-press orientation, Tier A/B); 'surface'/'raster' the "
                         "mesh's geometric +z top")
    ap.add_argument("--curve-reach", type=float, default=2.0,
                    help="surface-curves: how far the outer curves fan past the scanned patch "
                         "(x the patch half-width; >1 leaves the patch onto the sides)")
    ap.add_argument("--curve-anchors", type=int, default=6,
                    help="surface-curves: anchor points per curve (the spline is fit through these)")
    ap.add_argument("--tier-b-deg", type=float, default=35.0,
                    help="surface-turn from the scanned patch (deg) above which a pose is the lateral "
                         "'Tier B' frontier (renderer-untrusted); used for the colour split + count")
    ap.add_argument("--bags", nargs="+",
                    default=["data/rosbags/phantom.bag", "data/rosbags/phantom1.bag"],
                    help="contact/surface-curves: rosbag(s) whose contact frames define the scan patch")
    ap.add_argument("--sequences", nargs="+",
                    help="contact/surface-curves: pre-extracted sequence .npz (keys: poses, contact) "
                         "INSTEAD of --bags (no rosbags dep; works in the GPU container)")
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
    ap.add_argument("--color", choices=["order", "deviation"], default="order",
                    help="scatter colour: 'order' (scan order) or 'deviation' (axial-vs-surface-"
                         "normal angle, Tier A/B; the default for --trajectory surface-curves)")
    ap.add_argument("--title", help="optional figure suptitle")
    ap.add_argument("--split", action="store_true",
                    help="surface-curves: render Tier A and Tier B in two separate panels instead "
                         "of one colour-coded figure")
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
    cval = clabel = cmap = contact = None
    seg_len = args.per_line if args.trajectory == "surface-curves" else None
    if mesh is not None and args.trajectory == "surface-curves" and not args.trajectory_file:
        # winding-independent Tier metric: normal oriented by the real approach, not mesh normals
        from deepussim.pipeline.sampling import pose_surface_deviation
        rc = _contact_poses_cbct(mesh, bags=args.bags, sequences=args.sequences)
        contact = rc[:, :3, 3]                                # the real footprint, to overlay
        standoff, dev = pose_surface_deviation(mesh, poses, rc)
        n_b = int(((dev > args.tier_b_deg) & (dev <= 90)).sum()); n_x = int((dev > 90).sum())
        print(f"  standoff (mm): mean {standoff.mean():.2f}   Tier @ {args.tier_b_deg:g} deg: "
              f"A {len(dev) - n_b - n_x}  B(lateral) {n_b}  broken(>90) {n_x}")
        cval, clabel, cmap = dev, "surface-turn from patch (deg)", "RdYlBu_r"
    elif mesh is not None:
        standoff, cos = surface_stats(mesh, P, A)
        print(f"  standoff (mm): mean {standoff.mean():.2f}   "
              f"axial.inward-normal: mean {cos.mean():.3f}  min {cos.min():.3f}  (1 = perpendicular)")
        if args.color == "deviation":
            cval = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            clabel, cmap = "axial-normal dev (deg)", "RdYlBu_r"

    deci = _decimate(mesh) if mesh is not None else None
    if mesh is not None and deci is None:
        print("  (mesh decimation unavailable; drawing trajectory only)")
    if args.split and contact is not None and cval is not None:
        render_tiers(P, cval, deci, contact, Path(args.out), args.title, args.tier_b_deg)
    else:
        tier_b = args.tier_b_deg if (contact is not None and cval is not None) else None
        render(P, A, deci, Path(args.out), args.title, cval=cval, clabel=clabel, cmap=cmap,
               contact=contact, seg_len=seg_len, tier_b=tier_b)


if __name__ == "__main__":
    main()
