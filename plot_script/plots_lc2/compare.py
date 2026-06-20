"""Visualize whether LC2 refinement helps: US vs CBCT@init vs CBCT@LC2-refined.

For each shown frame, three panels in the polar fan space LC2 actually optimises:

    real US (unwrapped) | CBCT fan @ init pose | CBCT fan @ LC2-refined pose

with the LC2 (and fan tissue-coverage %inside) numbers in the titles. Reads the refined
poses from an ``lc2.run`` output ``.npz`` (e.g. ``data/lc2/scanN_global.npz``) and reuses
``lc2``/``reslice`` for everything else.

    python -m plot_script.plots_lc2.compare --sequence data/sequences/scan5.npz \
        --lc2-npz data/lc2/scan5_global.npz --frames 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so `lc2` / `reslice` resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

import lc2  # noqa: F401  (path setup)
from lc2.forward import fit_us_fan, prepare

DEFAULT_REPORT = _REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = _REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_PLACEMENT = _REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_REF_SEQUENCE = _REPO_ROOT / "data" / "sequences" / "scan1.npz"   # fan fitted from here
DEFAULT_OUT_DIR = _REPO_ROOT / "figures" / "lc2_compare"
US_SPACING = 0.166112957


def _norm(a: np.ndarray) -> np.ndarray:
    """Robust 1..99 percentile stretch to [0, 1] for display."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def compare_grid(
    sequence: Path,
    lc2_npz: Path,
    n_show: int = 4,
    shape: str = "fan",
    report: Path = DEFAULT_REPORT,
    volume_path: Path = DEFAULT_VOLUME,
    placement: Path = DEFAULT_PLACEMENT,
    ref_sequence: Path = DEFAULT_REF_SEQUENCE,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    """Save an ``(n_show x [US | init | LC2-refined])`` comparison for one sequence.

    ``shape="fan"`` renders the familiar Cartesian fan (raw US frame + CBCT sector, the display
    space); ``shape="polar"`` renders the unwrapped scan-line grid LC2 actually optimises in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if shape == "fan":
        from plot_script.plots_reslice.compare import cbct_sector_zoom

    rep = json.loads(Path(report).read_text(encoding="utf-8"))
    data = np.load(lc2_npz)

    # Accept both naming schemes: the `lc2.run` CLI prefixes global outputs with "global_",
    # while a direct `register_global` dump uses the raw keys.
    def pick(*names):
        for n in names:
            if n in data.files:
                return data[n]
        raise KeyError(f"none of {names} in {lc2_npz} (has {list(data.files)})")

    frame_index = pick("frame_index", "indices")
    refined = pick("global_refined_poses", "refined_poses")
    lc2_before = pick("global_lc2_before", "lc2_before")
    lc2_after = pick("global_lc2_after", "lc2_after")
    inside_before = pick("global_inside_before", "inside_before")
    inside_after = pick("global_inside_after", "inside_after")

    # Same shared fan and the same frame selection that lc2.run used, so ctx.frames lines up
    # one-to-one with the saved refined poses.
    ref = np.load(ref_sequence, allow_pickle=True)
    fan, geom = fit_us_fan(ref["images"], ref["contact"], US_SPACING)
    ctx = prepare(sequence, rep, Path(volume_path), Path(placement), US_SPACING,
                  n_frames=len(frame_index), fan=fan, geom=geom)

    from lc2.register import register_frame

    raw_images = np.load(sequence, allow_pickle=True)["images"] if shape == "fan" else None

    def render(pose, target_shape):
        """CBCT panel for a pose: Cartesian fan sector (display) or polar grid (LC2 space)."""
        if shape == "fan":
            return cbct_sector_zoom(ctx.volume, ctx.affine_centered, pose, target_shape)
        return ctx.cbct_fan(pose)

    show = np.linspace(0, len(ctx.frames) - 1, min(n_show, len(ctx.frames)), dtype=int)
    fig, ax = plt.subplots(len(show), 4, figsize=(12, 3 * len(show)), squeeze=False)
    aspect = "auto" if shape == "polar" else "equal"
    for r, k in enumerate(show):
        f = ctx.frames[k]
        us = raw_images[f.index] if shape == "fan" else f.us_polar
        tshape = us.shape[:2]
        # Per-frame LC2 (its own 6-DoF nudge for this frame) vs the shared multi-frame correction.
        pf = register_frame(ctx, f, max_trans_mm=15.0, max_rot_deg=15.0)
        cols = [
            (us, f"real US (frame {f.index})"),
            (render(f.init_pose_mm, tshape),
             f"CBCT: NO LC2 (calibration)\nLC2 {lc2_before[k]:.2f}  |  inside {inside_before[k]*100:.0f}%"),
            (render(pf["refined_pose"], tshape),
             f"CBCT: per-frame LC2\nLC2 {pf['lc2_after']:.2f}  |  inside {pf['inside_after']*100:.0f}%"),
            (render(refined[k], tshape),
             f"CBCT: multi-frame LC2\nLC2 {lc2_after[k]:.2f}  |  inside {inside_after[k]*100:.0f}%"),
        ]
        for c, (img, title) in enumerate(cols):
            ax[r, c].imshow(_norm(img), cmap="gray", aspect=aspect)
            ax[r, c].set_title(title, fontsize=9)
            ax[r, c].axis("off")

    fig.suptitle(
        f"{Path(sequence).stem} — does LC2 pose-refinement improve the CBCT slice's match to the real US?\n"
        f"LC2 = US<->CBCT similarity (0-1, higher = better match)      "
        f"inside% = fan inside tissue (higher = better; low = grazing outside / fake gain)\n"
        f"compare columns 2->3->4:  no LC2  vs  per-frame LC2  vs  multi-frame LC2 (shared correction)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lc2_{Path(sequence).stem}_{shape}_init_vs_refined.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--lc2-npz", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--shape", choices=("fan", "polar"), default="fan",
                    help="fan = Cartesian display (default); polar = the LC2 scoring space")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    path = compare_grid(args.sequence, args.lc2_npz, n_show=args.frames, shape=args.shape,
                        out_dir=args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
