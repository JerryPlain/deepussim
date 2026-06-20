"""Plot recorded probe trajectories over the phantom (coverage overview).

Shows where the real scans actually went on the phantom — useful for deciding which *new*
poses/trajectories to generate (Stage 2 should extend beyond the scanned patch).

Two frames:
  * ``world`` (default) — the robot/world frame: the phantom lies BELLY-UP (the lie-down roll
    in the placement is applied) and the probe presses down from above, matching the Genesis
    viewer. Units: metres.
  * ``cbct`` — the phantom-centred CBCT scan frame (phantom in its upright scan orientation),
    the frame ``reslice`` slices in. Units: millimetres.

    python -m plot_script.plots_reslice.trajectories                 # world frame, belly-up
    python -m plot_script.plots_reslice.trajectories --grid          # one panel per sequence
    python -m plot_script.plots_reslice.trajectories --frame cbct    # the slicing frame
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so `reslice` resolves when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reslice import pose as posemod
from reslice._geometry import compose
from reslice.io import load_transform_4x4

DEFAULT_SEQ_DIR = _REPO_ROOT / "data" / "sequences"
DEFAULT_MESH = _REPO_ROOT / "data" / "cbct" / "phantom_surface.stl"
DEFAULT_PLACEMENT = _REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_OUT = _REPO_ROOT / "figures" / "reslice" / "all_trajectories.png"


def contact_positions(sequence: Path, T_world_from_phantom: np.ndarray, frame: str) -> np.ndarray:
    """In-contact probe-face positions of one sequence, shape (N, 3).

    ``frame="world"`` returns world metres (the recorded probe pose directly); ``frame="cbct"``
    maps them into the phantom-centred CBCT mm frame.
    """
    d = np.load(sequence, allow_pickle=True)
    contact = np.asarray(d["contact"], dtype=bool)
    out = []
    for k in np.where(contact)[0]:
        T_world_from_probe = posemod.pose_from_sequence(Path(sequence), int(k), "ee", 0.0)
        if frame == "world":
            out.append(T_world_from_probe[:3, 3])
        else:
            out.append(posemod.probe_pose_in_phantom_centered_mm(T_world_from_probe, T_world_from_phantom)[:3, 3])
    return np.asarray(out, dtype=float)


def phantom_backdrop(mesh_path: Path, T_world_from_phantom: np.ndarray, frame: str, n: int = 4000) -> np.ndarray:
    """A sparse phantom-surface point cloud for context, in the requested frame.

    The mesh is stored in phantom-centred CBCT mm. For ``world`` we place it via the (lie-down)
    placement into world metres so it appears belly-up.
    """
    import trimesh

    verts = np.asarray(trimesh.load(mesh_path, process=False).vertices, dtype=float)  # CBCT mm
    if len(verts) > n:
        verts = verts[np.random.RandomState(0).choice(len(verts), n, replace=False)]
    if frame == "world":
        verts_m = verts / 1000.0                                    # mm -> m (phantom-centred)
        verts = (T_world_from_phantom @ np.c_[verts_m, np.ones(len(verts_m))].T).T[:, :3]
    return verts


def _units(frame: str) -> str:
    return "m" if frame == "world" else "mm"


def plot_overlay(sequences, mesh_path, placement, frame: str, out: Path) -> Path:
    """All sequences overlaid (3D + top view), one colour each, on the phantom backdrop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_world_from_phantom = load_transform_4x4(Path(placement))
    backdrop = phantom_backdrop(Path(mesh_path), T_world_from_phantom, frame)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(sequences), 1)))
    u = _units(frame)

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    axtop = fig.add_subplot(122)
    ax3d.scatter(*backdrop.T, s=1, c="0.85", alpha=0.3)
    axtop.scatter(backdrop[:, 0], backdrop[:, 1], s=1, c="0.85", alpha=0.3)

    for i, seq in enumerate(sequences):
        pos = contact_positions(Path(seq), T_world_from_phantom, frame)
        if len(pos) == 0:
            continue
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", c=colors[i], lw=0.8, alpha=0.8)
        axtop.plot(pos[:, 0], pos[:, 1], "-", c=colors[i], lw=0.9, alpha=0.85, label=Path(seq).stem)

    belly = "belly-up, world frame" if frame == "world" else "CBCT scan frame"
    ax3d.set_title(f"recorded probe trajectories (3D, {belly})")
    ax3d.set_xlabel(f"x ({u})"); ax3d.set_ylabel(f"y ({u})"); ax3d.set_zlabel(f"z ({u})")
    ax3d.view_init(elev=22, azim=-60)
    axtop.set_title(f"top view (x-y, {u})")
    axtop.set_aspect("equal")
    axtop.set_xlabel(f"x ({u})"); axtop.set_ylabel(f"y ({u})")
    axtop.legend(ncol=2, fontsize=7, loc="upper right")
    fig.suptitle("Probe trajectory coverage on the phantom — where the real scans went", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_grid(sequences, mesh_path, placement, frame: str, out: Path) -> Path:
    """One small top-view panel per sequence — clearer per-sequence scan pattern."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_world_from_phantom = load_transform_4x4(Path(placement))
    backdrop = phantom_backdrop(Path(mesh_path), T_world_from_phantom, frame)
    u = _units(frame)
    ncol = 5
    nrow = int(np.ceil(len(sequences) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, seq in enumerate(sequences):
        ax = axes[i // ncol, i % ncol]
        ax.axis("on")
        ax.scatter(backdrop[:, 0], backdrop[:, 1], s=0.6, c="0.85", alpha=0.4)
        pos = contact_positions(Path(seq), T_world_from_phantom, frame)
        if len(pos):
            ax.plot(pos[:, 0], pos[:, 1], "-", c="#b91c1c", lw=0.8)
        ax.set_title(f"{Path(seq).stem} ({len(pos)})", fontsize=8)
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
    fig.suptitle(f"Per-sequence probe trajectory (top view x-y, {u}; count in title)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _poses_to_frame(poses_cbct_mm: np.ndarray, T_world_from_phantom: np.ndarray, frame: str):
    """Positions + axial (beam) directions of a CBCT-mm pose stack, in the requested frame."""
    P = np.asarray(poses_cbct_mm)[:, :3, 3]
    A = np.asarray(poses_cbct_mm)[:, :3, 2]
    if frame == "world":
        Pm = P / 1000.0                                              # mm -> m (phantom-centred)
        P = (T_world_from_phantom @ np.c_[Pm, np.ones(len(Pm))].T).T[:, :3]
        A = (T_world_from_phantom[:3, :3] @ A.T).T
    return P, A


def plot_trajectory_file(traj_npz, mesh_path, placement, frame: str, out: Path) -> Path:
    """Draw one generated/loaded trajectory (key ``poses_cbct_mm``): 3D + top view, down-press arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_world_from_phantom = load_transform_4x4(Path(placement))
    poses = np.asarray(np.load(traj_npz)["poses_cbct_mm"], dtype=float)
    P, A = _poses_to_frame(poses, T_world_from_phantom, frame)
    backdrop = phantom_backdrop(Path(mesh_path), T_world_from_phantom, frame)
    u = _units(frame)
    arrow = 0.012 if frame == "world" else 12.0
    order = np.arange(len(P))

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    axtop = fig.add_subplot(122)
    ax3d.scatter(*backdrop.T, s=1, c="0.85", alpha=0.3)
    axtop.scatter(backdrop[:, 0], backdrop[:, 1], s=1, c="0.85", alpha=0.3)
    ax3d.plot(P[:, 0], P[:, 1], P[:, 2], "-", c="0.4", lw=0.7, alpha=0.6)
    step = max(1, len(P) // 12)                                       # sparse beam-direction arrows
    ax3d.quiver(P[::step, 0], P[::step, 1], P[::step, 2], A[::step, 0], A[::step, 1], A[::step, 2],
                length=arrow, color="#b91c1c", linewidth=0.6, arrow_length_ratio=0.4, alpha=0.7)
    sc = ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c=order, cmap="viridis", s=14, depthshade=False)
    axtop.plot(P[:, 0], P[:, 1], "-", c="0.4", lw=0.7, alpha=0.6)
    axtop.scatter(P[:, 0], P[:, 1], c=order, cmap="viridis", s=14)

    belly = "belly-up, world frame" if frame == "world" else "CBCT scan frame"
    ax3d.set_title(f"generated trajectory (3D, {belly})")
    ax3d.set_xlabel(f"x ({u})"); ax3d.set_ylabel(f"y ({u})"); ax3d.set_zlabel(f"z ({u})")
    ax3d.view_init(elev=22, azim=-60)
    axtop.set_title(f"top view (x-y, {u})"); axtop.set_aspect("equal")
    axtop.set_xlabel(f"x ({u})"); axtop.set_ylabel(f"y ({u})")
    fig.colorbar(sc, ax=axtop, label="scan order", fraction=0.046, pad=0.04)
    fig.suptitle(f"{Path(traj_npz).stem}: {len(P)} generated poses + beam direction (red)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", nargs="+", type=Path,
                    help="sequence .npz files (default: data/sequences/scan*.npz, natural order)")
    ap.add_argument("--trajectory-file", type=Path,
                    help="draw a generated/saved trajectory (.npz, key poses_cbct_mm) instead of sequences")
    ap.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    ap.add_argument("--placement", type=Path, default=DEFAULT_PLACEMENT)
    ap.add_argument("--frame", choices=("world", "cbct"), default="world",
                    help="world = belly-up robot frame (default); cbct = the upright slicing frame")
    ap.add_argument("--grid", action="store_true", help="one panel per sequence instead of overlay")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.trajectory_file:
        out = args.out or DEFAULT_OUT.with_name(f"{Path(args.trajectory_file).stem}.png")
        print(f"wrote {plot_trajectory_file(args.trajectory_file, args.mesh, args.placement, args.frame, out)}")
        return

    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted(DEFAULT_SEQ_DIR.glob("scan*.npz"),
                           key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))

    out = args.out or (DEFAULT_OUT.with_name("trajectories_grid.png") if args.grid else DEFAULT_OUT)
    fn = plot_grid if args.grid else plot_overlay
    print(f"wrote {fn(sequences, args.mesh, args.placement, args.frame, out)}")


if __name__ == "__main__":
    main()
