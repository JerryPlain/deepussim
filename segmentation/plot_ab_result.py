#!/usr/bin/env python
"""Paper figure: real-only vs real+generated liver segmentation (multi-seed).

Reads runs/seg_sam2_{A,B}_s{0,1,2}/predict_test/summary.json (written by predict.py) and
plots NSD@3mm (primary, fair to coarse GT) and Dice (reference), mean +/- std over 3 seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from style.style import figure, save, C, TYPE, ERRBAR  # noqa: E402

RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "figures" / "11_ab_real_vs_gen"


def _grp(g: str, seeds=(0, 1, 2)):
    nsd, dice = [], []
    for s in seeds:
        d = json.loads((RUNS / f"seg_sam2_{g}_s{s}" / "predict_test" / "summary.json").read_text())
        nsd.append(d["nsd"]); dice.append(d["dice"])
    return np.array(nsd), np.array(dice)


def main() -> None:
    An, Ad = _grp("A"); Bn, Bd = _grp("B")
    A_mean = [An.mean(), Ad.mean()]; A_std = [An.std(), Ad.std()]
    B_mean = [Bn.mean(), Bd.mean()]; B_std = [Bn.std(), Bd.std()]

    import matplotlib; matplotlib.use("Agg")
    fig, ax = figure(width="single", height=2.4)
    x = np.arange(2); w = 0.36
    ekw = dict(capsize=2.0, capthick=0.6, elinewidth=0.6, ecolor=C["dark"])
    ax.bar(x - w / 2, A_mean, w, yerr=A_std, error_kw=ekw,
           color=C["neutral"], edgecolor="white", linewidth=0.4, label="real only (40)")
    ax.bar(x + w / 2, B_mean, w, yerr=B_std, error_kw=ekw,
           color=C["ours"], edgecolor="white", linewidth=0.4, label="real + generated")

    for xi, (a, b) in enumerate(zip(A_mean, B_mean)):
        ax.text(xi - w / 2, a + A_std[xi] + 0.012, f"{a:.3f}", ha="center", va="bottom",
                fontsize=TYPE["tiny"], color=C["dark"])
        ax.text(xi + w / 2, b + B_std[xi] + 0.012, f"{b:.3f}", ha="center", va="bottom",
                fontsize=TYPE["tiny"], color=C["ours"])

    ax.set_xticks(x); ax.set_xticklabels(["NSD @ 3 mm\n(primary)", "Dice\n(reference)"])
    ax.set_ylabel("Score on liver-positive test frames")
    ax.set_ylim(0, 0.54)
    ax.grid(True, axis="y"); ax.grid(False, axis="x")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    ax.set_title("Adding generated US hurts (3 seeds, all worse)", pad=6)
    save(fig, str(OUT / "ab_real_vs_gen"))
    print(f"A: NSD={A_mean[0]:.3f}±{A_std[0]:.3f} Dice={A_mean[1]:.3f}")
    print(f"B: NSD={B_mean[0]:.3f}±{B_std[0]:.3f} Dice={B_mean[1]:.3f}")


if __name__ == "__main__":
    main()
