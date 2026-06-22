"""Visualise a tilt/fan novel trajectory in one glance.

The whole idea in one picture:
  LEFT  — a side-view cartoon: the probe stays at ONE in-volume contact point, but the beam
          fans out (grey = real straight-down beam, red = novel tilted beams), plus 3 big numbers.
  RIGHT — the proof: for several spots, the nearest REAL slice next to the NOVEL tilted slice,
          large and side by side, so the new anatomy the tilt reveals is obvious.

    apptainer/run.sh python -m plot_script.plots_reslice.tilt_novelty \
        --trajectory data/trajectories/novel_tilt_valid.npz \
        --out figures/reslice/novel_tilt_valid.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plot_script.style.style import C, TYPE, apply_style  # noqa: E402

DEFAULT_VOLUME = _REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_REPORT = _REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_REF = _REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_LC2 = _REPO_ROOT / "data" / "lc2"


def _load_cbct_frame(volume_path: Path, report_path: Path):
    import json
    from reslice.io import load_volume_data

    volume = load_volume_data(volume_path)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        key = "recon" if volume_path.suffix.lower() == ".mhd" else "dicom"
        affine = np.array(report["phantom_centered_frame"][f"{key}_affine_centered_from_ijk_mm"], dtype=float)
    else:
        import SimpleITK as sitk
        from reslice.frame import affine_from_sitk
        affine = affine_from_sitk(sitk.ReadImage(str(volume_path)))
    return volume, affine


def _display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x, [1.0, 99.0])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _draw_concept(ax, tilts_deg: np.ndarray) -> None:
    """Side-view cartoon: one contact point, grey real beam straight down, red fanned novel beams."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ax.set_xlim(0, 10); ax.set_ylim(0.2, 9.4); ax.set_aspect("equal"); ax.axis("off")
    # CBCT volume box + surface
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 8.8, 6.2, boxstyle="round,pad=0.05,rounding_size=0.4",
                                fc="#EEF1F4", ec=C["neutral"], lw=1.0, zorder=0))
    ax.text(9.2, 1.0, "CBCT volume", ha="right", va="bottom", fontsize=TYPE["small"], color=C["neutral"])
    # contact point + probe
    cx, cy = 5.0, 6.8
    ax.add_patch(Rectangle((cx - 0.55, cy), 1.1, 1.1, fc=C["dark"], ec="none", zorder=4))
    ax.plot([cx - 1.6, cx + 1.6], [cy, cy], color=C["dark"], lw=1.4, zorder=3)  # surface
    ax.scatter([cx], [cy], s=22, color="white", edgecolor=C["dark"], zorder=5)
    ax.text(cx, cy + 1.35, "one contact point\n(stays in volume)", ha="center", va="bottom",
            fontsize=TYPE["small"], color=C["dark"])
    # real beam: straight down
    L = 5.2
    ax.annotate("", xy=(cx, cy - L), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=2.4, shrinkA=0, shrinkB=0), zorder=2)
    ax.text(cx - 0.25, cy - L - 0.15, "real beam", ha="right", va="top",
            fontsize=TYPE["small"], color=C["neutral"])
    # novel beams: fan at the actual tilt magnitudes
    mags = sorted({abs(float(t)) for t in tilts_deg if abs(t) > 1e-6})
    for a in mags:
        for s in (-1, 1):
            rad = np.deg2rad(s * a)
            tip = (cx + L * np.sin(rad), cy - L * np.cos(rad))
            ax.annotate("", xy=tip, xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color=C["baseline_a"], lw=1.3, alpha=0.85,
                                        shrinkA=0, shrinkB=0), zorder=1)
    ax.text(cx + L * np.sin(np.deg2rad(max(mags))) + 0.2, cy - L * np.cos(np.deg2rad(max(mags))),
            "tilted beams\n→ new planes", ha="left", va="center",
            fontsize=TYPE["small"], color=C["baseline_a"])


def _draw_stats(ax, inside_pct: float, content_pct: float, beam_deg: float) -> None:
    """Compact key-metrics strip: restrained sizing, thin top rule (paper, not slide)."""
    ax.axis("off")
    ax.axhline(0.92, xmin=0.02, xmax=0.98, color=C["neutral"], lw=0.6)
    stats = [
        (f"{inside_pct:.0f}%", "fan inside\nvolume", C["baseline_b"]),
        (f"{content_pct:.0f}%", "new content\n(1 − corr.)", C["ours"]),
        (f"{beam_deg:.0f}°", "beam tilt vs\nnearest real", C["baseline_a"]),
    ]
    for i, (big, small, col) in enumerate(stats):
        x = (i + 0.5) / 3.0
        ax.text(x, 0.60, big, ha="center", va="center", fontsize=15, fontweight="bold",
                color=col, transform=ax.transAxes)
        ax.text(x, 0.16, small, ha="center", va="center", fontsize=TYPE["small"],
                color=C["dark"], transform=ax.transAxes)


def main() -> None:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from scipy.spatial import cKDTree

    from plot_script.plots_reslice.compare import cbct_sector_zoom
    from renderer_training.analyze_pose_coverage import load_lc2_poses

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=Path, default=_REPO_ROOT / "data" / "trajectories" / "novel_tilt_valid.npz")
    ap.add_argument("--lc2-dir", type=Path, default=DEFAULT_LC2)
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "figures" / "4_novel_trajectory_tilt" / "novel_tilt_valid.png")
    args = ap.parse_args()

    apply_style()
    d = np.load(args.trajectory, allow_pickle=True)
    poses = np.asarray(d["poses_cbct_mm"], dtype=float)
    novelty = np.asarray(d["beam_novelty_deg"], dtype=float)
    content = np.asarray(d.get("content_novelty", np.full(len(poses), np.nan)), dtype=float)
    inside = np.asarray(d["inside_ratio"], dtype=float)
    # a few representative beam-tilt magnitudes for the fan cartoon (keep it uncluttered)
    tilt_mags = np.percentile(novelty, [20, 50, 80, 95])

    real, _ = load_lc2_poses(args.lc2_dir, ["scan*_global.npz"])
    tree = cKDTree(real[:, :3, 3])

    volume, affine = _load_cbct_frame(args.volume_path, args.report)
    ref = np.load(args.ref_sequence, allow_pickle=True)
    shape = tuple(int(v) for v in ref["images"].shape[1:3])

    # pick the most clearly-novel, spatially varied examples (high content novelty)
    rank = content.copy()
    rank[~np.isfinite(rank)] = novelty[~np.isfinite(rank)] / 90.0
    cand = np.argsort(rank)[::-1][: max(args.n_examples * 4, 8)]
    chosen, seen = [], set()
    for i in cand:
        key = tuple(np.round(poses[i, :3, 3] / 15.0).astype(int))  # ~15 mm bins for variety
        if key in seen:
            continue
        seen.add(key); chosen.append(int(i))
        if len(chosen) == args.n_examples:
            break
    while len(chosen) < args.n_examples and len(cand):
        chosen.append(int(cand[len(chosen)]))

    fig = plt.figure(figsize=(11.0, 4.6))
    outer = GridSpec(1, 2, width_ratios=[0.9, 1.3], wspace=0.10, figure=fig,
                     top=0.88, bottom=0.05, left=0.015, right=0.985)
    left = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[3.2, 1.0], hspace=0.04)
    ax_concept = fig.add_subplot(left[0]); _draw_concept(ax_concept, tilt_mags)
    ax_stats = fig.add_subplot(left[1])
    _draw_stats(ax_stats, 100 * float(inside.mean()), 100 * float(np.nanmean(content)), float(novelty.mean()))

    right = GridSpecFromSubplotSpec(args.n_examples, 2, subplot_spec=outer[1], wspace=0.03, hspace=0.10)
    for r, idx in enumerate(chosen):
        _, j = tree.query(poses[idx, :3, 3])
        real_slice = _display(cbct_sector_zoom(volume, affine, real[j], shape))
        novel_slice = _display(cbct_sector_zoom(volume, affine, poses[idx], shape))
        axr = fig.add_subplot(right[r, 0]); axn = fig.add_subplot(right[r, 1])
        axr.imshow(real_slice, cmap="gray", aspect="auto"); axn.imshow(novel_slice, cmap="gray", aspect="auto")
        for a in (axr, axn):
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True); s.set_linewidth(0.9)
        for s in axr.spines.values():
            s.set_color(C["neutral"])
        for s in axn.spines.values():
            s.set_color(C["baseline_a"])
        if r == 0:
            axr.set_title("real probe (scanned)", fontsize=TYPE["small"], color=C["neutral"], pad=3)
            axn.set_title("novel tilted probe (same spot)", fontsize=TYPE["small"],
                          color=C["baseline_a"], pad=3)
        axn.text(0.96, 0.06, f"+{novelty[idx]:.0f}°", transform=axn.transAxes, ha="right", va="bottom",
                 fontsize=TYPE["small"], color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc=C["baseline_a"], ec="none", alpha=0.9))

    # panel labels (paper style; the message lives in the LaTeX caption, not an in-figure title)
    fig.text(0.015, 0.965, "(a)", fontsize=TYPE["body"], fontweight="bold", va="top")
    fig.text(0.455, 0.965, "(b)", fontsize=TYPE["body"], fontweight="bold", va="top")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
