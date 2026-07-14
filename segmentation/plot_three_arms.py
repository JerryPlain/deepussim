#!/usr/bin/env python
"""Three-arm figure: real vs real+CUT-gen vs real+physics-gen (multi-seed NSD + Dice)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from style.style import figure, save, C, TYPE  # noqa: E402

RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "figures" / "15_three_arm"
ARMS = [("A (40 real)", "A", C["baseline_a"]),
        ("+ CUT gen", "B", C["neutral"]),
        ("+ physics gen", "Bphys", C["ours"])]


def _grp(pfx):
    nsd, dice = [], []
    for s in (0, 1, 2):
        d = json.loads((RUNS / f"seg_sam2_{pfx}_s{s}" / "predict_test" / "summary.json").read_text())
        nsd.append(d["nsd"]); dice.append(d["dice"])
    return np.array(nsd), np.array(dice)


def main() -> None:
    data = {name: _grp(pfx) for name, pfx, _ in ARMS}
    labels = [a[0] for a in ARMS]; colors = [a[2] for a in ARMS]
    import matplotlib; matplotlib.use("Agg")
    fig, (axN, axD) = figure(ncols=2, width="double", height=2.5)
    x = np.arange(len(ARMS))
    ekw = dict(capsize=2.0, capthick=0.6, elinewidth=0.6, ecolor=C["dark"])
    for ax, mi, title in [(axN, 0, "NSD @ 3 mm (primary)"), (axD, 1, "Dice (reference)")]:
        means = [data[a[0]][mi].mean() for a in ARMS]
        stds = [data[a[0]][mi].std() for a in ARMS]
        ax.bar(x, means, 0.6, yerr=stds, error_kw=ekw, color=colors, edgecolor="white", linewidth=0.4)
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=TYPE["tiny"], color=C["dark"])
        ax.axhline(means[0], ls="--", lw=0.6, color=C["baseline_a"], alpha=0.6)  # real-only reference
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=TYPE["small"])
        ax.set_ylim(0, 0.5); ax.set_ylabel("score (liver-positive)"); ax.set_title(title)
        ax.grid(True, axis="y"); ax.grid(False, axis="x")
    fig.suptitle("Both gen arms hurt vs real; physics ≈ CUT downstream (realism gain didn't transfer)")
    save(fig, str(OUT / "three_arm"))
    for name, pfx, _ in ARMS:
        n, d = data[name]
        print(f"{name:16s} NSD {n.mean():.3f}±{n.std():.3f}  Dice {d.mean():.3f}")


if __name__ == "__main__":
    main()
