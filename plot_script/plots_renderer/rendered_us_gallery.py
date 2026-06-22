"""Gallery of pseudo-US rendered from a trajectory (e.g. the novel tilt poses).

The render step only saves a small sample preview; this draws ALL rendered frames so the full
output is inspectable:

  all_generated_us.png    contact sheet of every generated pseudo-US frame (sorted by sweep)
  cbct_to_us_per_sweep.png one representative CBCT→US pair per fanned sweep, larger

    apptainer/run.sh python -m plot_script.plots_renderer.rendered_us_gallery \
        --rendered data/rendered_us/novel_tilt_paired/rendered_us.npz \
        --out-dir figures/7_rendered_us_from_novel_tilt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_script.style.style import C, TYPE, apply_style  # noqa: E402


def _contact_sheet(gen, traj_id, ncols, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(gen)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.85, nrows * 0.66))
    fig.set_layout_engine("none")
    axes = np.atleast_2d(axes)
    for k in range(nrows * ncols):
        ax = axes.flat[k]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if k < n:
            ax.imshow(gen[k], cmap="gray", aspect="auto", vmin=0, vmax=1)
            # thin divider colour per sweep so groups are visible at a glance
            col = C["ours"] if traj_id[k] % 2 == 0 else C["baseline_a"]
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(0.7)
            ax.text(0.04, 0.94, f"{k}", transform=ax.transAxes, fontsize=4.5, color="white",
                    va="top", ha="left")
        else:
            ax.set_visible(False)
    fig.suptitle(f"All {n} generated pseudo-US frames (border colour = sweep id parity)",
                 fontsize=TYPE["title"] + 2, y=0.995)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


def _per_sweep(cbct, gen, traj_id, inside, near, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = sorted(np.unique(traj_id))
    reps = [int(np.where(traj_id == s)[0][len(np.where(traj_id == s)[0]) // 2]) for s in sweeps]
    n = len(reps)
    cols = 4                                   # 4 CBCT|US pairs per row → 8 image columns
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.5, rows * 1.5))
    fig.set_layout_engine("none")
    axes = np.atleast_2d(axes)
    for i, idx in enumerate(reps):
        r, c = divmod(i, cols)
        axc, axu = axes[r, 2 * c], axes[r, 2 * c + 1]
        axc.imshow(cbct[idx], cmap="gray", aspect="auto")
        axu.imshow(gen[idx], cmap="gray", aspect="auto", vmin=0, vmax=1)
        for ax in (axc, axu):
            ax.set_xticks([]); ax.set_yticks([])
        for s in axc.spines.values():
            s.set_color(C["neutral"]); s.set_linewidth(0.8)
        for s in axu.spines.values():
            s.set_color(C["baseline_a"]); s.set_linewidth(0.8)
        axc.set_title(f"sweep {int(traj_id[idx])}  CBCT", fontsize=TYPE["tiny"], color=C["neutral"])
        axu.set_title("→ US", fontsize=TYPE["tiny"], color=C["baseline_a"])
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, 2 * c].set_visible(False); axes[r, 2 * c + 1].set_visible(False)
    fig.suptitle("CBCT slice → paired-CUT pseudo-US, one per fanned sweep", fontsize=TYPE["title"] + 2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.06, hspace=0.25)
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rendered", type=Path,
                    default=_REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "figures" / "7_rendered_us_from_novel_tilt")
    ap.add_argument("--ncols", type=int, default=18, help="columns in the contact sheet")
    args = ap.parse_args()

    apply_style()
    d = np.load(args.rendered, allow_pickle=True)
    gen = np.asarray(d["generated_us"], dtype=np.float32)
    cbct = np.asarray(d["cbct_sector"], dtype=np.float32)
    cbct = (cbct - cbct.min()) / max(float(cbct.max() - cbct.min()), 1e-6)
    traj_id = np.asarray(d["trajectory_id"]) if "trajectory_id" in d.files else np.zeros(len(gen), int)
    inside = np.asarray(d["inside_ratio"]) if "inside_ratio" in d.files else np.ones(len(gen))
    near = np.asarray(d["nearest_existing_mm"]) if "nearest_existing_mm" in d.files else np.zeros(len(gen))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _contact_sheet(gen, traj_id, args.ncols, args.out_dir / "all_generated_us.png")
    _per_sweep(cbct, gen, traj_id, inside, near, args.out_dir / "cbct_to_us_per_sweep.png")


if __name__ == "__main__":
    main()
