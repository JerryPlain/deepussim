#!/usr/bin/env python
"""Sim2real story: fixing renderer physics monotonically improves downstream augmentation.

Panel (a): four-arm downstream NSD — real-only vs +CUT / +physics / +sim2real gen.
Panel (b): renderer deep-field-confidence AUC vs real (0.5=indistinguishable) for the three
renderers — the physics-realism gap that panel (a) tracks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from style.style import figure, save, C, TYPE  # noqa: E402

RUNS = REPO_ROOT / "runs"
OUT = REPO_ROOT / "figures" / "17_sim2real_story"

ARMS = [("real only", "A", C["baseline_a"]),
        ("+ CUT", "B", C["neutral"]),
        ("+ physics", "baseline_c", None),
        ("+ sim2real", "Bs2r", C["ours"])]
PFX = {"real only": "A", "+ CUT": "B", "+ physics": "Bphys", "+ sim2real": "Bs2r"}
COL = {"real only": C["baseline_a"], "+ CUT": C["neutral"], "+ physics": C["baseline_c"], "+ sim2real": C["ours"]}
# renderer deep-conf AUC vs real (from confidence_realism / physics_sim_check)
DEEPCONF_AUC = {"CUT": 0.97, "physics": 0.66, "sim2real": 0.56}


def _nsd(pfx):
    return np.array([json.loads((RUNS / f"seg_sam2_{pfx}_s{s}" / "predict_test" / "summary.json").read_text())["nsd"]
                     for s in (0, 1, 2)])


def main() -> None:
    names = list(PFX)
    means = [_nsd(PFX[n]).mean() for n in names]
    stds = [_nsd(PFX[n]).std() for n in names]
    import matplotlib; matplotlib.use("Agg")
    fig, (axA, axB) = figure(ncols=2, width="double", height=2.6)

    x = np.arange(len(names)); ekw = dict(capsize=2.0, capthick=0.6, elinewidth=0.6, ecolor=C["dark"])
    axA.bar(x, means, 0.62, yerr=stds, error_kw=ekw, color=[COL[n] for n in names], edgecolor="white", linewidth=0.4)
    axA.axhline(means[0], ls="--", lw=0.6, color=C["baseline_a"], alpha=0.6)
    for xi, (m, s) in enumerate(zip(means, stds)):
        axA.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=TYPE["tiny"], color=C["dark"])
    axA.set_xticks(x); axA.set_xticklabels(names, fontsize=TYPE["small"])
    axA.set_ylim(0, 0.46); axA.set_ylabel("test NSD @ 3 mm (liver-positive)")
    axA.set_title("(a) downstream segmentation"); axA.grid(True, axis="y"); axA.grid(False, axis="x")

    rr = list(DEEPCONF_AUC); yv = [DEEPCONF_AUC[k] for k in rr]
    xr = np.arange(len(rr))
    axB.bar(xr, yv, 0.62, color=[C["neutral"], C["baseline_c"], C["ours"]], edgecolor="white", linewidth=0.4)
    axB.axhline(0.5, ls="--", lw=0.6, color=C["dark"], alpha=0.7)
    axB.text(len(rr) - 0.5, 0.52, "real-like (0.5)", ha="right", va="bottom", fontsize=TYPE["tiny"], color=C["dark"])
    for xi, v in enumerate(yv):
        axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=TYPE["tiny"], color=C["dark"])
    axB.set_xticks(xr); axB.set_xticklabels(rr, fontsize=TYPE["small"])
    axB.set_ylim(0, 1.05); axB.set_ylabel("deep-field confidence AUC vs real")
    axB.set_title("(b) renderer physics realism"); axB.grid(True, axis="y"); axB.grid(False, axis="x")

    fig.suptitle("Fixing renderer physics (b) monotonically improves augmentation (a)")
    save(fig, str(OUT / "sim2real_story"))
    print("four-arm NSD:", {n: round(m, 3) for n, m in zip(names, means)})


if __name__ == "__main__":
    main()
