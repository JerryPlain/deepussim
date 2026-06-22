#!/usr/bin/env python
"""Quantitatively compare paired vs unpaired CUT renderers on the LC2-paired eval set.

No pretrained network is required (no torchvision/lpips in the container), so realism is measured
with US-domain statistics instead of Inception-FID. Three families of metric, each answering a
different question, are computed over the fan (background pixels are masked out):

  fidelity  (generated vs the *aligned* real US frame)
      SSIM (masked), PSNR (masked), L1 (masked)   -- "does this frame match the real one?"
  realism   (generated set vs real set, no alignment)
      pixel-histogram Wasserstein-1               -- "is the brightness distribution right?"
      Nakagami m Wasserstein-1                     -- "is the speckle statistic right?"
      texture-Frechet (first-order + GLCM feats)   -- "does the texture look like US?"
  artifact
      row-wise (depth) mean-intensity profile      -- exposes the paired bottom-rim artifact

Outputs (under --out):
    renderer_eval_metrics.csv     per-frame fidelity metrics
    renderer_eval_summary.json    aggregate numbers + the verdict
    renderer_eval_comparison.png  the visual comparison figure

Example:
    apptainer/run.sh python renderer_training/renderer_eval.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepussim.renderer.cut import CUTModel  # noqa: E402

DEFAULT_PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_PAIRED = REPO_ROOT / "runs" / "renderer_cut_paired_display_ep300_b2_lp005" / "generator.pt"
DEFAULT_UNPAIRED = REPO_ROOT / "runs" / "renderer_cut_unpaired_display_ep300_b2" / "generator.pt"
DEFAULT_OUT = REPO_ROOT / "figures" / "6_renderer_eval_paired_vs_unpaired"
DEFAULT_METRICS_OUT = REPO_ROOT / "data" / "renderer_eval"


# --------------------------------------------------------------------------- generator inference
def _to_unit(x: np.ndarray) -> np.ndarray:
    """Per-image min/max normalise to [-1, 1] (the generator's training input convention)."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def _checkpoint_arg(ck: dict, name: str, default):
    return (ck.get("args") or {}).get(name, default)


def _load_generator(path: Path, device: torch.device) -> CUTModel:
    ck = torch.load(path, map_location=device)
    model = CUTModel(
        ngf=int(_checkpoint_arg(ck, "ngf", 64)),
        ndf=int(_checkpoint_arg(ck, "ndf", 64)),
        n_blocks=int(_checkpoint_arg(ck, "n_blocks", 9)),
        lambda_nce=float(_checkpoint_arg(ck, "lambda_nce", 1.0)),
        num_patches=int(_checkpoint_arg(ck, "num_patches", 256)),
    ).to(device)
    model.G.load_state_dict(ck["G"])
    model.eval()
    return model


def _run_generator(ckpt: Path, src: np.ndarray, device_name: str, batch: int) -> np.ndarray:
    """Generate US in [0, 1] for every CBCT input (src already in [-1, 1])."""
    device = torch.device(device_name)
    model = _load_generator(ckpt, device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(src), batch):
            chunk = torch.from_numpy(src[start:start + batch, None]).to(device)
            fake = model.G(chunk).cpu().numpy()[:, 0]
            out.append(fake)
            print(f"  {ckpt.parent.name}: {min(start + batch, len(src))}/{len(src)}", flush=True)
    fake = np.concatenate(out, axis=0)
    return np.clip((fake + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- metric helpers
def _fan_masks(cbct: np.ndarray, thr: float = 1e-4) -> np.ndarray:
    """Per-frame boolean fan mask (where the CBCT sector has signal)."""
    return cbct > thr


def _central_roi(mask_all: np.ndarray, frac: float = 0.4) -> tuple[slice, slice]:
    """Axis-aligned rectangle around the fan centroid, sized to sit safely inside the sector."""
    ys, xs = np.where(mask_all)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = int((ys.max() - ys.min()) * frac / 2)
    w = int((xs.max() - xs.min()) * frac / 2)
    return slice(cy - h, cy + h), slice(cx - w, cx + w)


def _ssim_psnr_l1(real: np.ndarray, fake: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    from skimage.metrics import structural_similarity

    _, smap = structural_similarity(real, fake, data_range=1.0, full=True)
    ssim = float(smap[mask].mean())
    diff = (real - fake)[mask]
    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0
    return ssim, psnr, l1


def _nakagami_m(img: np.ndarray, mask: np.ndarray) -> float:
    """Nakagami shape m = E[I^2]^2 / Var(I^2) of the envelope; ~1 is fully developed speckle."""
    v = img[mask].astype(np.float64)
    p = v ** 2
    var = float(np.var(p))
    return float(np.mean(p) ** 2 / var) if var > 1e-12 else float("nan")


def _texture_features(img: np.ndarray, roi: tuple[slice, slice]) -> np.ndarray:
    """First-order + GLCM texture descriptor of a central fan patch (for the texture-Frechet)."""
    from scipy.stats import kurtosis, skew
    from skimage.feature import graycomatrix, graycoprops

    patch = img[roi]
    flat = patch.ravel()
    first = [float(flat.mean()), float(flat.std()), float(skew(flat)), float(kurtosis(flat))]
    q = np.clip((patch * 7.0).round().astype(np.uint8), 0, 7)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 2], levels=8, symmetric=True, normed=True)
    tex = [float(graycoprops(glcm, p).mean()) for p in ("contrast", "homogeneity", "energy", "correlation")]
    return np.array(first + tex, dtype=np.float64)


def _frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two feature-set Gaussians (the FID formula, generic features)."""
    from scipy.linalg import sqrtm

    mu1, mu2 = a.mean(0), b.mean(0)
    c1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    c2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2.0 * covmean))


def _row_profile(stack: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mean intensity per image row over masked (in-fan) pixels -> depth profile (proxy for depth)."""
    out = np.zeros(stack.shape[1], dtype=np.float64)
    for r in range(stack.shape[1]):
        vals = stack[:, r, :][masks[:, r, :]]
        out[r] = float(vals.mean()) if vals.size else 0.0
    return out


def _sample_pixels(stack: np.ndarray, masks: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    pool = stack[masks]
    rng = np.random.RandomState(seed)
    if len(pool) > n:
        pool = pool[rng.choice(len(pool), n, replace=False)]
    return pool


# --------------------------------------------------------------------------- main eval
def evaluate(real: np.ndarray, gen: dict[str, np.ndarray], masks: np.ndarray,
             mask_all: np.ndarray) -> dict:
    from scipy.stats import wasserstein_distance

    roi = _central_roi(mask_all)
    n = len(real)
    methods = list(gen)

    per_frame = {m: {"ssim": [], "psnr": [], "l1": []} for m in methods}
    naka = {"real": [], **{m: [] for m in methods}}
    feats = {"real": [], **{m: [] for m in methods}}

    for i in range(n):
        mk = masks[i]
        naka["real"].append(_nakagami_m(real[i], mk))
        feats["real"].append(_texture_features(real[i], roi))
        for m in methods:
            ssim, psnr, l1 = _ssim_psnr_l1(real[i], gen[m][i], mk)
            per_frame[m]["ssim"].append(ssim)
            per_frame[m]["psnr"].append(psnr)
            per_frame[m]["l1"].append(l1)
            naka[m].append(_nakagami_m(gen[m][i], mk))
            feats[m].append(_texture_features(gen[m][i], roi))

    real_px = _sample_pixels(real, masks, 200_000)
    naka_real = np.asarray(naka["real"])
    feats_real = np.stack(feats["real"])

    summary = {"n_frames": n, "methods": {}}
    for m in methods:
        gen_px = _sample_pixels(gen[m], masks, 200_000)
        summary["methods"][m] = {
            "ssim_mean": float(np.mean(per_frame[m]["ssim"])),
            "ssim_std": float(np.std(per_frame[m]["ssim"])),
            "psnr_mean": float(np.mean(per_frame[m]["psnr"])),
            "l1_mean": float(np.mean(per_frame[m]["l1"])),
            "hist_wasserstein": float(wasserstein_distance(gen_px, real_px)),
            "nakagami_m_mean": float(np.nanmean(naka[m])),
            "nakagami_wasserstein": float(
                wasserstein_distance(np.asarray(naka[m])[~np.isnan(naka[m])],
                                     naka_real[~np.isnan(naka_real)])
            ),
            "texture_frechet": _frechet(np.stack(feats[m]), feats_real),
        }
    summary["real_nakagami_m_mean"] = float(np.nanmean(naka_real))
    return summary, per_frame, naka, roi


def _verdict(summary: dict) -> dict:
    """Lower-is-better for every realism/artifact metric; higher for ssim/psnr. Vote across them."""
    methods = list(summary["methods"])
    if len(methods) != 2:
        return {"note": "verdict only defined for two methods"}
    a, b = methods
    higher_better = {"ssim_mean", "psnr_mean"}
    lower_better = {"l1_mean", "hist_wasserstein", "nakagami_wasserstein", "texture_frechet"}
    wins = {a: 0, b: 0}
    detail = {}
    for metric in higher_better | lower_better:
        va, vb = summary["methods"][a][metric], summary["methods"][b][metric]
        if metric in higher_better:
            win = a if va > vb else b
        else:
            win = a if va < vb else b
        wins[win] += 1
        detail[metric] = {"winner": win, a: va, b: vb}
    overall = a if wins[a] > wins[b] else b
    return {"wins": wins, "overall_better": overall, "per_metric": detail}


def plot_comparison(real, gen, masks, per_frame, naka, summary, roi, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(gen)
    colors = {"paired": "#b91c1c", "unpaired": "#2563eb"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. fidelity bars (SSIM, PSNR)
    ax = axes[0, 0]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, [summary["methods"][m]["ssim_mean"] for m in methods], 0.4, label="SSIM", color="#7c3aed")
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel("SSIM (masked) ↑")
    ax.set_title("Fidelity: SSIM vs aligned real (↑ better)")
    for i, m in enumerate(methods):
        ax.text(i - 0.2, summary["methods"][m]["ssim_mean"], f'{summary["methods"][m]["ssim_mean"]:.3f}',
                ha="center", va="bottom", fontsize=9)

    # 2. fidelity bars (PSNR, L1)
    ax = axes[0, 1]
    ax.bar(x - 0.2, [summary["methods"][m]["psnr_mean"] for m in methods], 0.4, color="#0891b2", label="PSNR (dB)")
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel("PSNR dB ↑", color="#0891b2")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [summary["methods"][m]["l1_mean"] for m in methods], 0.4, color="#ea580c", label="L1")
    ax2.set_ylabel("L1 ↓", color="#ea580c")
    ax.set_title("Fidelity: PSNR (↑) & L1 (↓)")

    # 3. realism scalar bars (all lower-better, normalised to unpaired for shared axis)
    ax = axes[0, 2]
    rmetrics = ["hist_wasserstein", "nakagami_wasserstein", "texture_frechet"]
    labels = ["hist\nW1", "Nakagami\nW1", "texture\nFrechet"]
    width = 0.38
    for j, m in enumerate(methods):
        base = [summary["methods"][methods[0]][k] for k in rmetrics]
        vals = [summary["methods"][m][k] / (b if b > 1e-12 else 1.0) for k, b in zip(rmetrics, base)]
        ax.bar(np.arange(len(rmetrics)) + (j - 0.5) * width, vals, width, label=m, color=colors.get(m, None))
    ax.set_xticks(range(len(rmetrics))); ax.set_xticklabels(labels)
    ax.set_ylabel(f"relative to {methods[0]} ↓"); ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_title("Realism distances (↓ better, normalised)"); ax.legend()

    # 4. depth (row-wise) intensity profile -> bottom-rim artifact
    ax = axes[1, 0]
    rows = np.arange(real.shape[1])
    ax.plot(_row_profile(real, masks), rows, color="0.1", lw=2, label="real")
    for m in methods:
        ax.plot(_row_profile(gen[m], masks), rows, color=colors.get(m, None), lw=1.5, label=m)
    ax.invert_yaxis(); ax.set_xlabel("mean intensity (in-fan)"); ax.set_ylabel("image row (≈ depth)")
    ax.set_title("Depth profile (bottom = far field)\ndivergence at bottom = rim artifact"); ax.legend()

    # 5. intensity histograms
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 60)
    ax.hist(_sample_pixels(real, masks, 200_000), bins=bins, density=True, histtype="step",
            color="0.1", lw=2, label="real")
    for m in methods:
        ax.hist(_sample_pixels(gen[m], masks, 200_000), bins=bins, density=True, histtype="step",
                color=colors.get(m, None), lw=1.5, label=m)
    ax.set_yscale("log"); ax.set_xlabel("intensity"); ax.set_ylabel("density (log)")
    ax.set_title("In-fan intensity distribution"); ax.legend()

    # 6. Nakagami m distribution
    ax = axes[1, 2]
    bins = np.linspace(0, max(3.0, np.nanmax(naka["real"])), 40)
    ax.hist(np.asarray(naka["real"]), bins=bins, density=True, histtype="step", color="0.1", lw=2,
            label=f'real (m̄={summary["real_nakagami_m_mean"]:.2f})')
    for m in methods:
        ax.hist(np.asarray(naka[m]), bins=bins, density=True, histtype="step", color=colors.get(m, None),
                lw=1.5, label=f'{m} (m̄={summary["methods"][m]["nakagami_m_mean"]:.2f})')
    ax.axvline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("Nakagami m (1 = Rayleigh speckle)"); ax.set_ylabel("density")
    ax.set_title("Speckle statistic distribution"); ax.legend()

    v = _verdict(summary)
    fig.suptitle(
        f"Renderer eval on {summary['n_frames']} LC2-paired frames  —  "
        f"overall closer to real US: {v.get('overall_better', '?').upper()}  "
        f"(wins {v.get('wins', {})})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--paired-ckpt", type=Path, default=DEFAULT_PAIRED)
    ap.add_argument("--unpaired-ckpt", type=Path, default=DEFAULT_UNPAIRED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="figure output dir")
    ap.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT, help="csv/json data output dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None, help="debug limit")
    args = ap.parse_args()

    data = np.load(args.pairs, allow_pickle=True)
    cbct = np.asarray(data["cbct"], dtype=np.float32)
    real = np.asarray(data["us"], dtype=np.float32) / 255.0
    if args.max_frames:
        cbct, real = cbct[: args.max_frames], real[: args.max_frames]
    src = np.stack([_to_unit(x) for x in cbct]).astype(np.float32)
    masks = _fan_masks(cbct)
    mask_all = masks.mean(0) > 0.99

    print(f"generating US for {len(src)} frames on {args.device} ...", flush=True)
    gen = {
        "paired": _run_generator(args.paired_ckpt, src, args.device, args.batch),
        "unpaired": _run_generator(args.unpaired_ckpt, src, args.device, args.batch),
    }

    print("computing metrics ...", flush=True)
    summary, per_frame, naka, roi = evaluate(real, gen, masks, mask_all)
    summary["verdict"] = _verdict(summary)
    summary["paired_ckpt"] = str(args.paired_ckpt)
    summary["unpaired_ckpt"] = str(args.unpaired_ckpt)

    out_dir = args.out if args.out.suffix == "" else args.out.parent       # figures
    metrics_dir = args.metrics_out                                          # data (kept separate)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "renderer_eval_comparison.png"
    plot_comparison(real, gen, masks, per_frame, naka, summary, roi, fig_path)

    with (metrics_dir / "renderer_eval_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "method", "ssim", "psnr", "l1", "nakagami_m"])
        for m in gen:
            for i in range(len(real)):
                w.writerow([i, m, per_frame[m]["ssim"][i], per_frame[m]["psnr"][i],
                            per_frame[m]["l1"][i], naka[m][i]])
    (metrics_dir / "renderer_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
