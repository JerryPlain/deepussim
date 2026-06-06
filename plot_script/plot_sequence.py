"""Visualise an extracted rosbag sequence (US frames + EE trajectory + contact/sync).

Reads the ``.npz`` produced by ``scripts/extract_rosbags.py``
(keys: images, poses, contact, stamps, mean_intensity, sync_dt_s) and emits, per
sequence, a set of standalone LaTeX-styled figures (PNG + PDF) plus a sweep GIF,
organised under ``figures/<sequence>/``:

    frames.{png,pdf}            representative US B-mode frames (contact vs lift-off)
    contact_timeline.{png,pdf}  contact mask + mean brightness over time
    sync.{png,pdf}              US <-> pose time-sync quality
    orientation.{png,pdf}       probe-axial spread (the orientation-diversity metric)
    trajectory.{png,pdf}        EE path, top + side, coloured by scan order
    sweep.gif                   the scan as an animation

    python plot_script/plot_sequence.py                       # both default sequences
    python plot_script/plot_sequence.py data/sequences/foo.npz [bar.npz ...]

Useful as an on-site quality check right after collecting a new bag — especially the
orientation figure, which tells you whether the sweep added the probe-tilt diversity
the current data lacks.
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).parent))              # style/ lives beside this script
from style.style import apply_style, figure, save, C, TYPE, FIG  # noqa: E402

# True LaTeX if a binary is installed, else the Computer-Modern mathtext look.
_USETEX = shutil.which("latex") is not None
apply_style(usetex=_USETEX)

ROOT = Path(__file__).parents[1]
FIGS = ROOT / "figures"
DEFAULT_SEQS = [ROOT / "data/sequences/phantom.npz", ROOT / "data/sequences/phantom1.npz"]


def _tex(s_tex: str, s_plain: str) -> str:
    return s_tex if _USETEX else s_plain


def _axial_spread_deg(poses: np.ndarray) -> np.ndarray:
    """Angle (deg) of each frame's probe axial (+z) from the mean axial — orientation diversity."""
    axial = poses[:, :3, 2]
    axial = axial / (np.linalg.norm(axial, axis=1, keepdims=True) + 1e-12)
    mean = axial.mean(0); mean /= np.linalg.norm(mean) + 1e-12
    return np.degrees(np.arccos(np.clip(axial @ mean, -1.0, 1.0)))


def fig_frames(d, t, contact, out):
    img = d["images"]; n = len(img)
    idxs = np.linspace(0, n - 1, 4, dtype=int)
    fig, axes = figure(ncols=4, width="double", height=2.2)
    for ax, fi in zip(axes, idxs):
        on = bool(contact[fi])
        ax.imshow(img[fi], cmap="gray", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(C["dark"] if on else C["bad"]); s.set_linewidth(1.1)
        ax.set_title(_tex(rf"\#{fi}\,|\,$t{{=}}${t[fi]:.1f}\,s", f"#{fi} | t={t[fi]:.1f}s"),
                     fontsize=TYPE["small"])
        ax.set_xlabel("contact" if on else "lift-off", fontsize=TYPE["small"],
                      color=C["dark"] if on else C["bad"], labelpad=2)
    save(fig, out); plt.close(fig)


def fig_contact_timeline(d, t, contact, out):
    meanI = d["mean_intensity"]; n = len(t)
    fig, ax = figure(width="double", height=2.0)
    ax.fill_between(t, 0, 1, where=contact, transform=ax.get_xaxis_transform(),
                    color=C["baseline_b"], alpha=0.15, step="mid", label="in contact")
    ax.plot(t, meanI, lw=1.0, color=C["ours"])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean intensity")
    ax.set_title(_tex(rf"contact \& brightness ({contact.sum()}/{n} in contact)",
                      f"contact & brightness ({contact.sum()}/{n} in contact)"))
    ax.grid(True, axis="both"); ax.legend(loc="upper right")
    save(fig, out); plt.close(fig)


def fig_sync(d, out):
    sync = d["sync_dt_s"] * 1000.0
    fig, ax = figure(width="single", height=2.0)
    ax.hist(sync, bins=30, color=C["baseline_e"])
    ax.set_xlabel(_tex(r"sync $\Delta t$ (ms)", "sync dt (ms)")); ax.set_ylabel("frames")
    ax.set_title(_tex(rf"US$\leftrightarrow$pose sync (median {np.median(sync):.1f}\,ms)",
                      f"US-pose sync (median {np.median(sync):.1f} ms)"))
    ax.grid(True, axis="y")
    save(fig, out); plt.close(fig)


def fig_orientation(poses, out):
    ang = _axial_spread_deg(poses)
    fig, ax = figure(width="single", height=2.0)
    ax.hist(ang, bins=30, color=C["baseline_c"])
    ax.set_xlabel("probe-axial angle from mean (deg)"); ax.set_ylabel("frames")
    ax.set_title(_tex(rf"orientation spread (max {ang.max():.1f}$^\circ$)",
                      f"orientation spread (max {ang.max():.1f} deg)"))
    ax.grid(True, axis="y")
    save(fig, out); plt.close(fig)


def fig_trajectory(poses, contact, out):
    pos = poses[:, :3, 3] * 1000.0                          # world m -> mm
    order = np.arange(len(pos))
    fig, axes = figure(ncols=2, width="double", height=2.8)
    for ax, (a, b, ylab, ttl) in zip(axes, [(0, 1, "Y", _tex("top (X--Y)", "top (X-Y)")),
                                            (0, 2, "Z", _tex("side (X--Z)", "side (X-Z)"))]):
        ax.scatter(pos[~contact, a], pos[~contact, b], s=5, color=C["neutral"], label="lift-off")
        sc = ax.scatter(pos[contact, a], pos[contact, b], c=order[contact], cmap="viridis", s=9)
        ax.set_xlabel("X (mm)"); ax.set_ylabel(f"{ylab} (mm)")
        ax.set_title(f"EE trajectory, {ttl}"); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, axis="both")
    cb = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("scan order", fontsize=TYPE["small"])
    axes[0].legend(loc="best")
    save(fig, out); plt.close(fig)


def fig_sweep_gif(d, out_gif):
    img = d["images"]; n = len(img)
    step = max(1, n // 120)
    imageio.mimsave(out_gif, img[::step, ::2, ::2], fps=15)
    print(f"  saved → {out_gif} ({len(img[::step])} frames)")


def plot_sequence(npz_path: Path) -> None:
    name = npz_path.stem
    d = np.load(npz_path, allow_pickle=True)
    contact = d["contact"].astype(bool)
    t = d["stamps"] - d["stamps"][0]
    poses = d["poses"]
    out_dir = FIGS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_frames(d, t, contact, str(out_dir / "frames"))
    fig_contact_timeline(d, t, contact, str(out_dir / "contact_timeline"))
    fig_sync(d, str(out_dir / "sync"))
    fig_orientation(poses, str(out_dir / "orientation"))
    fig_trajectory(poses, contact, str(out_dir / "trajectory"))
    fig_sweep_gif(d, out_dir / "sweep.gif")


def main(argv):
    seqs = [Path(a) for a in argv] if argv else DEFAULT_SEQS
    for s in seqs:
        if not s.exists():
            print(f"skip (missing): {s}"); continue
        print(f"{s.name}:")
        plot_sequence(s)


if __name__ == "__main__":
    main(sys.argv[1:])
