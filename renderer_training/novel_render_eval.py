#!/usr/bin/env python
"""Evaluate the quality of pseudo-US rendered from NOVEL poses — without per-frame ground truth.

Novel (tilt) poses have no paired real US, so SSIM/PSNR/L1 (fidelity to a specific frame) do not
apply. Three GT-free families are computed instead:

  (A) Realism  — does the novel generated-US *set* match the real US *set* distributionally?
        pixel-histogram Wasserstein, Nakagami-m Wasserstein, texture-Frechet.
        Each is reported next to a real-vs-real split "floor" (the best achievable distance), so the
        absolute numbers are interpretable: novel close to the floor = realistic.
  (B) Input in-distribution — is each novel CBCT slice within the manifold the renderer trained on?
        Mahalanobis distance of the novel CBCT texture features to the training-CBCT Gaussian.
        Frames beyond the training self-distance threshold are flagged OOD (renderer may hallucinate).
  (D) Artifact prevalence — fraction of novel frames with an abnormal bright far-field band (the known
        paired bottom-rim artifact), thresholded at the real-US 95th percentile.

Outputs:
    figures/8_novel_render_quality/novel_render_quality.png
    data/novel_render_eval/novel_render_summary.json

Example:
    apptainer/run.sh python renderer_training/novel_render_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from renderer_training.renderer_eval import (  # reuse the validated metric helpers
    _central_roi,
    _fan_masks,
    _frechet,
    _nakagami_m,
    _sample_pixels,
    _texture_features,
)

DEFAULT_RENDERED = REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz"
DEFAULT_PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_FIG = REPO_ROOT / "figures" / "8_novel_render_quality" / "novel_render_quality.png"
DEFAULT_METRICS = REPO_ROOT / "data" / "novel_render_eval"


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(x)


def _feature_stack(imgs: np.ndarray, roi) -> np.ndarray:
    return np.stack([_texture_features(_norm01(im), roi) for im in imgs])


def _mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    d = x - mu
    return np.sqrt(np.einsum("ni,ij,nj->n", d, cov_inv, d))


def _bottom_excess(imgs: np.ndarray, masks: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Per-frame (mean intensity in the deepest ``frac`` of in-fan rows) − (overall in-fan mean).

    A large positive value is the paired bottom-rim artifact (a bright far-field band the real
    probe does not produce)."""
    out = np.zeros(len(imgs), dtype=np.float64)
    for i in range(len(imgs)):
        rows = np.where(masks[i].any(axis=1))[0]
        if len(rows) < 8:
            continue
        cut = rows[int(len(rows) * (1 - frac))]
        deep = imgs[i][cut:][masks[i][cut:]]
        allv = imgs[i][masks[i]]
        if deep.size and allv.size:
            out[i] = float(deep.mean() - allv.mean())
    return out


def _realism(gen: np.ndarray, gen_m: np.ndarray, real: np.ndarray, real_m: np.ndarray,
             gmask: np.ndarray, rmask: np.ndarray, groi, rroi) -> dict:
    from scipy.stats import wasserstein_distance

    gpx = _sample_pixels(gen, gmask, 200_000)
    rpx = _sample_pixels(real, rmask, 200_000)
    # real-vs-real floor (split the real set in half)
    h = len(real) // 2
    rpx_a = _sample_pixels(real[:h], rmask[:h], 100_000)
    rpx_b = _sample_pixels(real[h:], rmask[h:], 100_000)
    gen_feat = _feature_stack(gen, groi)
    real_feat = _feature_stack(real, rroi)
    rm = real_m[np.isfinite(real_m)]
    gm = gen_m[np.isfinite(gen_m)]
    return {
        "hist_w1": float(wasserstein_distance(gpx, rpx)),
        "hist_w1_floor": float(wasserstein_distance(rpx_a, rpx_b)),
        "nakagami_w1": float(wasserstein_distance(gm, rm)),
        "nakagami_w1_floor": float(wasserstein_distance(rm[:h], rm[h:])),
        "texture_frechet": _frechet(gen_feat, real_feat),
        "texture_frechet_floor": _frechet(real_feat[:h], real_feat[h:]),
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rendered", type=Path, default=DEFAULT_RENDERED)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    args = ap.parse_args()

    rd = np.load(args.rendered, allow_pickle=True)
    gen = np.asarray(rd["generated_us"], dtype=np.float32)            # [0,1]
    gen_cbct = np.asarray(rd["cbct_sector"], dtype=np.float32)        # display CBCT (pre-norm)
    pr = np.load(args.pairs, allow_pickle=True)
    real = np.asarray(pr["us"], dtype=np.float32) / 255.0             # [0,1]
    real_cbct = np.asarray(pr["cbct"], dtype=np.float32)             # [0,1]

    gmask = _fan_masks(np.stack([_norm01(c) for c in gen_cbct]))
    rmask = _fan_masks(real_cbct)
    groi = _central_roi(gmask.mean(0) > 0.99)
    rroi = _central_roi(rmask.mean(0) > 0.99)

    gen_m = np.array([_nakagami_m(gen[i], gmask[i]) for i in range(len(gen))])
    real_m = np.array([_nakagami_m(real[i], rmask[i]) for i in range(len(real))])

    print("(A) realism ...", flush=True)
    realism = _realism(gen, gen_m, real, real_m, gmask, rmask, groi, rroi)

    print("(B) input in-distribution ...", flush=True)
    train_feat = _feature_stack(real_cbct, rroi)                      # training CBCT slices
    novel_feat = _feature_stack(gen_cbct, rroi)                       # novel CBCT slices
    mu = train_feat.mean(0)
    cov = np.cov(train_feat, rowvar=False) + 1e-6 * np.eye(train_feat.shape[1])
    cov_inv = np.linalg.inv(cov)
    train_md = _mahalanobis(train_feat, mu, cov_inv)
    novel_md = _mahalanobis(novel_feat, mu, cov_inv)
    ood_thr = float(np.percentile(train_md, 99))
    in_dist_frac = float(np.mean(novel_md <= ood_thr))

    print("(D) artifact prevalence ...", flush=True)
    gen_be = _bottom_excess(gen, gmask)
    real_be = _bottom_excess(real, rmask)
    art_thr = float(np.percentile(real_be, 95))
    artifact_frac = float(np.mean(gen_be > art_thr))

    summary = {
        "n_novel": int(len(gen)), "n_real": int(len(real)),
        "A_realism": realism,
        "B_input_in_distribution": {
            "ood_threshold_train_p99": ood_thr,
            "novel_in_distribution_fraction": in_dist_frac,
            "novel_md_mean": float(novel_md.mean()), "train_md_mean": float(train_md.mean()),
        },
        "D_artifact": {
            "real_bottom_excess_p95": art_thr,
            "novel_artifact_fraction": artifact_frac,
            "novel_bottom_excess_mean": float(gen_be.mean()),
            "real_bottom_excess_mean": float(real_be.mean()),
        },
    }
    # simple pass/fail flags (realism within 3x floor; >=90% in-distribution; <=15% artifacts)
    summary["verdict"] = {
        "realism_ok": all(realism[k] <= 3.0 * realism[k + "_floor"] + 1e-9
                          for k in ("hist_w1", "nakagami_w1", "texture_frechet")),
        "input_in_distribution_ok": in_dist_frac >= 0.90,
        "artifact_ok": artifact_frac <= 0.15,
    }

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    blue, red, grey = "#2563eb", "#b91c1c", "0.55"

    # A1 realism bars vs floor
    ax = axes[0, 0]
    keys = ["hist_w1", "nakagami_w1", "texture_frechet"]
    labs = ["hist W1", "Nakagami W1", "texture\nFréchet"]
    x = np.arange(len(keys)); w = 0.38
    nov = [realism[k] / realism[k + "_floor"] for k in keys]      # relative to floor
    ax.bar(x, nov, w * 1.6, color=blue, label="novel vs real")
    ax.axhline(1.0, color=grey, ls="--", lw=1, label="real-vs-real floor (=1)")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("distance ÷ floor (↓, 1=ideal)"); ax.set_title("(A) realism vs real US (lower=more real)")
    for i, v in enumerate(nov):
        ax.text(i, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)

    # A2 intensity histogram
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 60)
    ax.hist(_sample_pixels(real, rmask, 200_000), bins=bins, density=True, histtype="step", color="0.1", lw=2, label="real US")
    ax.hist(_sample_pixels(gen, gmask, 200_000), bins=bins, density=True, histtype="step", color=blue, lw=1.6, label="novel gen")
    ax.set_yscale("log"); ax.set_xlabel("intensity"); ax.set_ylabel("density (log)")
    ax.set_title("(A) in-fan intensity distribution"); ax.legend(fontsize=8)

    # A3 Nakagami m
    ax = axes[0, 2]
    bins = np.linspace(0, max(2.0, np.nanmax(real_m)), 40)
    ax.hist(real_m[np.isfinite(real_m)], bins=bins, density=True, histtype="step", color="0.1", lw=2,
            label=f"real (m̄={np.nanmean(real_m):.2f})")
    ax.hist(gen_m[np.isfinite(gen_m)], bins=bins, density=True, histtype="step", color=blue, lw=1.6,
            label=f"novel (m̄={np.nanmean(gen_m):.2f})")
    ax.set_xlabel("Nakagami m"); ax.set_ylabel("density"); ax.set_title("(A) speckle statistic"); ax.legend(fontsize=8)

    # B input OOD
    ax = axes[1, 0]
    bins = np.linspace(0, max(novel_md.max(), train_md.max()), 40)
    ax.hist(train_md, bins=bins, density=True, histtype="step", color="0.1", lw=2, label="training CBCT")
    ax.hist(novel_md, bins=bins, density=True, histtype="stepfilled", color=blue, alpha=0.4, label="novel CBCT")
    ax.axvline(ood_thr, color=red, lw=1.5, ls="--", label=f"OOD thr (train p99)")
    ax.set_xlabel("Mahalanobis dist. to training CBCT"); ax.set_ylabel("density")
    ax.set_title(f"(B) input in-distribution: {100*in_dist_frac:.0f}% inside"); ax.legend(fontsize=8)

    # D artifact: bottom excess
    ax = axes[1, 1]
    bins = np.linspace(min(real_be.min(), gen_be.min()), max(real_be.max(), gen_be.max()), 40)
    ax.hist(real_be, bins=bins, density=True, histtype="step", color="0.1", lw=2, label="real US")
    ax.hist(gen_be, bins=bins, density=True, histtype="stepfilled", color=red, alpha=0.4, label="novel gen")
    ax.axvline(art_thr, color=red, lw=1.5, ls="--", label="real p95")
    ax.set_xlabel("far-field bright excess"); ax.set_ylabel("density")
    ax.set_title(f"(D) bottom-rim artifact: {100*artifact_frac:.0f}% of novel frames"); ax.legend(fontsize=8)

    # summary panel
    ax = axes[1, 2]; ax.axis("off")
    v = summary["verdict"]
    def mark(ok): return "PASS" if ok else "CHECK"
    def col(ok): return "#16a34a" if ok else "#ea580c"
    lines = [
        ("(A) Realism", mark(v["realism_ok"]), col(v["realism_ok"]),
         f"hist {nov[0]:.1f}× · Nakagami {nov[1]:.1f}× · texture {nov[2]:.1f}× floor"),
        ("(B) Input in-dist.", mark(v["input_in_distribution_ok"]), col(v["input_in_distribution_ok"]),
         f"{100*in_dist_frac:.0f}% of novel CBCT inside training manifold"),
        ("(D) Artifacts", mark(v["artifact_ok"]), col(v["artifact_ok"]),
         f"{100*artifact_frac:.0f}% frames with far-field artifact"),
    ]
    ax.text(0.5, 0.96, "Novel render quality (no GT needed)", ha="center", va="top",
            fontsize=13, fontweight="bold", transform=ax.transAxes)
    y = 0.78
    for name, m, c, detail in lines:
        ax.text(0.04, y, name, fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.96, y, m, fontsize=12, fontweight="bold", color=c, ha="right", transform=ax.transAxes)
        ax.text(0.04, y - 0.07, detail, fontsize=8.5, color="0.3", transform=ax.transAxes)
        y -= 0.22

    fig.suptitle(f"Quality of {len(gen)} novel-pose pseudo-US frames (paired-CUT) vs {len(real)} real US",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=130); fig.savefig(args.fig.with_suffix(".pdf"))
    plt.close(fig)

    args.metrics_out.mkdir(parents=True, exist_ok=True)
    (args.metrics_out / "novel_render_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
