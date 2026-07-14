#!/usr/bin/env python
"""US-physics realism metric: per-frame confidence-map statistics, real vs generated.

Real US attenuates with depth (deep field / behind reflectors -> low confidence). The CUT
renderer does not reproduce this, so its confidence maps stay high everywhere. We turn that
into a PER-FRAME physics-realism score and measure how separable real vs generated are by it
(AUC). This is the one signal that matched our downstream A/B result (generated hurts), so it
is a candidate US-specific renderer-quality metric the literature currently lacks.

Per-frame scores (over the fan content):
  mean_conf       — average confidence over the fan
  deep_conf       — average confidence over the DEEP half of the fan (the discriminative part)
Separation is reported as AUC of a univariate real-vs-gen classifier on each score
(0.5 = indistinguishable = realistic; ->1.0 = trivially separable = unrealistic physics).

    module load python/3.12-base
    python renderer_training/confidence_realism.py --n-real 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renderer_training.confidence_map import confidence_map, _norm01     # noqa: E402
from style.style import figure, save, C, TYPE                            # noqa: E402


def _frame_scores(img: np.ndarray) -> tuple[float, float]:
    """(mean fan confidence, mean deep-half fan confidence) for one US frame."""
    conf = confidence_map(img)
    g = _norm01(img)
    fan = g > 0.02
    if fan.sum() < 100:
        return float("nan"), float("nan")
    rows = np.where(fan.any(1))[0]
    mid = (rows[0] + rows[-1]) // 2
    deep = fan.copy(); deep[:mid] = False
    return float(conf[fan].mean()), float(conf[deep].mean() if deep.any() else conf[fan].mean())


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that `pos` scores higher than `neg` (rank statistic). Symmetric-reported as max(auc,1-auc)."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(max(auc, 1 - auc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "train" / "images.npy")
    ap.add_argument("--gen", type=Path,
                    default=REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--n-real", type=int, default=150)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "13_confidence_realism")
    args = ap.parse_args()

    real = np.load(args.real)
    rng = np.random.default_rng(0)
    ridx = rng.permutation(len(real))[:args.n_real]
    gen = np.load(args.gen, allow_pickle=True)["generated_us"]

    r_mean, r_deep = [], []
    for k, i in enumerate(ridx):
        m, d = _frame_scores(real[i]); r_mean.append(m); r_deep.append(d)
        if (k + 1) % 40 == 0:
            print(f"  real {k+1}/{len(ridx)}", flush=True)
    g_mean, g_deep = [], []
    for k in range(len(gen)):
        m, d = _frame_scores(gen[k]); g_mean.append(m); g_deep.append(d)
        if (k + 1) % 60 == 0:
            print(f"  gen {k+1}/{len(gen)}", flush=True)
    r_mean, r_deep = np.array(r_mean), np.array(r_deep)
    g_mean, g_deep = np.array(g_mean), np.array(g_deep)

    auc_mean = _auc(g_mean, r_mean); auc_deep = _auc(g_deep, r_deep)
    print(f"\nmean-conf : real={np.nanmean(r_mean):.3f}  gen={np.nanmean(g_mean):.3f}  AUC(real vs gen)={auc_mean:.3f}")
    print(f"deep-conf : real={np.nanmean(r_deep):.3f}  gen={np.nanmean(g_deep):.3f}  AUC(real vs gen)={auc_deep:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "scores.npz", r_mean=r_mean, r_deep=r_deep, g_mean=g_mean, g_deep=g_deep,
             auc_mean=auc_mean, auc_deep=auc_deep)

    import matplotlib; matplotlib.use("Agg")
    fig, (axA, axB) = figure(ncols=2, width="double", height=2.4)
    for ax, (rv, gv, auc, name) in zip((axA, axB),
                                       [(r_mean, g_mean, auc_mean, "mean fan confidence"),
                                        (r_deep, g_deep, auc_deep, "deep-field confidence")]):
        bins = np.linspace(0, 1, 26)
        ax.hist(rv[~np.isnan(rv)], bins=bins, density=True, alpha=0.75, color=C["baseline_a"], label="real")
        ax.hist(gv[~np.isnan(gv)], bins=bins, density=True, alpha=0.65, color=C["ours"], label="generated")
        ax.set_xlabel(name); ax.set_ylabel("density")
        ax.set_title(f"AUC(real vs gen) = {auc:.2f}")
        ax.grid(True, axis="y"); ax.grid(False, axis="x")
    axA.legend(loc="upper left")
    fig.suptitle("Generated US keeps unrealistically high deep-field confidence (physics gap)")
    save(fig, str(args.out / "confidence_realism"))
    print(f"wrote {args.out}/confidence_realism.pdf + .png")


if __name__ == "__main__":
    main()
