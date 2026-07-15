#!/usr/bin/env python
"""Sim2real story: renderer physics realism predicts downstream augmentation value.

Panel (a): four-arm downstream NSD — real-only vs +CUT / +physics / +sim2real gen.
Panel (b): correlation across renderer variants (incl. the beam-blur v2 that regressed) between
the physics-realism metric (deep-field confidence AUC vs real, ->0.5 = real-like) and downstream
NSD. A strong negative correlation = the (cheap, physics) metric predicts downstream utility,
which distributional realism metrics do not.
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

# four-arm bars (best sim2real = v1)
BARS = [("real only", "A", C["baseline_a"]), ("+ CUT", "B", C["neutral"]),
        ("+ physics", "Bphys", C["baseline_c"]), ("+ sim2real", "Bs2r", C["ours"])]
# renderer variants for the correlation: deep-conf AUC vs real (measured) + downstream run prefix
RENDERERS = [("CUT", 0.97, "B", C["neutral"]),
             ("physics", 0.66, "Bphys", C["baseline_c"]),
             ("sim2real", 0.56, "Bs2r", C["ours"]),
             ("sim2real+blur", 0.79, "Bs2rv2", C["baseline_e"])]


def _nsd(pfx):
    return np.array([json.loads((RUNS / f"seg_sam2_{pfx}_s{s}" / "predict_test" / "summary.json").read_text())["nsd"]
                     for s in (0, 1, 2)])


def main() -> None:
    import matplotlib; matplotlib.use("Agg")
    fig, (axA, axB) = figure(ncols=2, width="double", height=2.7)

    # (a) four-arm bars
    means = [_nsd(p).mean() for _, p, _ in BARS]; stds = [_nsd(p).std() for _, p, _ in BARS]
    x = np.arange(len(BARS)); ekw = dict(capsize=2.0, capthick=0.6, elinewidth=0.6, ecolor=C["dark"])
    axA.bar(x, means, 0.62, yerr=stds, error_kw=ekw, color=[c for *_, c in BARS], edgecolor="white", linewidth=0.4)
    axA.axhline(means[0], ls="--", lw=0.6, color=C["baseline_a"], alpha=0.6)
    for xi, (m, s) in enumerate(zip(means, stds)):
        axA.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=TYPE["tiny"], color=C["dark"])
    axA.set_xticks(x); axA.set_xticklabels([n for n, *_ in BARS], fontsize=TYPE["small"])
    axA.set_ylim(0, 0.46); axA.set_ylabel("test NSD @ 3 mm (liver-positive)")
    axA.set_title("(a) downstream segmentation"); axA.grid(True, axis="y"); axA.grid(False, axis="x")

    # (b) correlation: deep-conf AUC vs downstream NSD
    auc = np.array([a for _, a, _, _ in RENDERERS]); nsd = np.array([_nsd(p).mean() for _, _, p, _ in RENDERERS])
    r = float(np.corrcoef(auc, nsd)[0, 1])
    axB.axhline(_nsd("A").mean(), ls="--", lw=0.6, color=C["baseline_a"], alpha=0.7)
    axB.text(0.98, _nsd("A").mean() + 0.003, "real-only", ha="right", va="bottom",
             fontsize=TYPE["tiny"], color=C["baseline_a"])
    z = np.polyfit(auc, nsd, 1); xs = np.linspace(0.5, 1.0, 50)
    axB.plot(xs, np.poly1d(z)(xs), ls="--", lw=0.8, color=C["dark"], alpha=0.6)
    for name, a, p, col in RENDERERS:
        axB.scatter([a], [_nsd(p).mean()], s=42, color=col, edgecolor="white", linewidth=0.6, zorder=5)
        axB.annotate(name, (a, _nsd(p).mean()), textcoords="offset points", xytext=(5, 4),
                     fontsize=TYPE["tiny"], color=C["dark"])
    axB.set_xlabel("deep-field confidence AUC vs real  (→0.5 = real-like)")
    axB.set_ylabel("test NSD @ 3 mm"); axB.set_xlim(1.02, 0.5)   # reversed: real-like on the right
    axB.set_title(f"(b) physics realism predicts utility  (r = {r:.2f})")
    axB.grid(True, axis="both")

    fig.suptitle("Fixing renderer physics improves augmentation; the physics metric predicts it")
    save(fig, str(OUT / "sim2real_story"))
    print("bars:", {n: round(m, 3) for (n, *_), m in zip(BARS, means)})
    print(f"correlation deep-conf AUC vs downstream NSD: r = {r:.3f}")


if __name__ == "__main__":
    main()
