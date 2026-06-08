#!/usr/bin/env python
"""Quantify the learned renderer (B1): turn "looks like US" into numbers.

Two metrics, the two things that matter:
  - FID / KID   — distribution realism: are the GENERATED US close to REAL US? (lower = closer)
  - structure-consistency — anti-hallucination: does G(x) keep its OWN input's layout, rather
    than inventing/averaging anatomy? Reported as low-frequency NCC for matched pairs (x_i, G(x_i))
    vs shuffled pairs (x_i, G(x_j)); a healthy renderer has matched >> shuffled. If matched ≈
    shuffled, the output ignores its input (hallucination / collapse) → unusable for encoder eval.

    python scripts/eval_renderer.py --run runs/renderer_cut --data data/renderer --n 500

FID/KID use a torchvision InceptionV3 (downloads weights once; ImageNet-trained, so treat the
absolute FID as a relative yardstick across runs, not a physical quantity). Structure-consistency
is dependency-light and is the load-bearing check for "safe to evaluate encoders on".
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "plot_script"))
from deepussim.renderer.networks import ResnetGenerator      # noqa: E402
from deepussim.renderer.data import _to_unit                 # noqa: E402


def _ncc(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def surface_depth(img, frac=0.3):
    """Per scan-line (column), the depth row where the fan first enters bright tissue.

    The probe sits at a standoff, so each column runs dark (outside) -> bright surface echo ->
    body. The surface = first row above a per-column floor. Returns ``(n_lat,)`` depth indices.
    """
    x = np.asarray(img, dtype=float)
    lo = x.min(0); hi = x.max(0)
    thr = lo + frac * (hi - lo)
    above = x > thr[None, :]
    d = above.argmax(0).astype(float)                       # first True row per column
    d[~above.any(0)] = x.shape[0]                           # column never enters tissue
    return d


def surface_alignment(src, fake, depth_mm=101.5):
    """Does the generated US put the SURFACE where its input CBCT slice has it?

    Per-column surface-depth error between (x_i, G(x_i)) (matched) vs (x_i, G(x_j)) (shuffled).
    Small matched error (and matched << shuffled) = the geometry/structure is preserved; large =
    the renderer moved/invented the surface. This is the targeted, phantom-uniformity-robust check.
    """
    ds = np.stack([surface_depth(s) for s in src])          # (N, n_lat)
    df = np.stack([surface_depth(f) for f in fake])
    n_ax = src.shape[1]; mm_px = depth_mm / n_ax
    rng = np.random.default_rng(0); perm = rng.permutation(len(src))
    matched = np.abs(ds - df).mean(1)                       # per-pair mean |Δdepth| (px)
    shuffled = np.abs(ds - df[perm]).mean(1)
    return matched * mm_px, shuffled * mm_px                # in mm


def structure_consistency(src, fake, sigma=8.0):
    """Low-freq NCC for matched (x_i,G(x_i)) vs shuffled (x_i,G(x_j)) — does G follow its input?"""
    lp_s = np.stack([gaussian_filter(s, sigma) for s in src])
    lp_f = np.stack([gaussian_filter(f, sigma) for f in fake])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(src))
    matched = np.array([_ncc(lp_s[i], lp_f[i]) for i in range(len(src))])
    shuffled = np.array([_ncc(lp_s[i], lp_f[perm[i]]) for i in range(len(src))])
    return matched, shuffled


@torch.no_grad()
def inception_features(imgs, device, batch=16):
    """2048-d InceptionV3 pool features for (N,1,H,W) images in [-1,1]."""
    import torchvision
    net = torchvision.models.inception_v3(weights="DEFAULT", aux_logits=True)
    net.fc = torch.nn.Identity(); net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(device)        # (b,1,H,W) in [-1,1]
        x = (x + 1) / 2                                           # -> [0,1]
        x = x.repeat(1, 3, 1, 1)                                  # gray -> RGB
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feats.append(net(x).cpu().numpy())
    return np.concatenate(feats)


def fid(f1, f2):
    from scipy.linalg import sqrtm
    mu1, mu2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    cov = sqrtm(c1 @ c2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * cov))


def kid(f1, f2, deg=3):
    """Unbiased polynomial-kernel MMD^2 (x1000) — robust for small samples."""
    def k(a, b):
        return (a @ b.T / a.shape[1] + 1) ** deg
    m, n = len(f1), len(f2)
    kxx, kyy, kxy = k(f1, f1), k(f2, f2), k(f1, f2)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return float((kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1))
                  - 2 * kxy.mean()) * 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/renderer_cut")
    ap.add_argument("--data", default="data/renderer")
    ap.add_argument("--n", type=int, default=500, help="subsample per pool (speed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip generation: load src/fake/real from <run>/eval_cache.npz (for the "
                         "login-side FID step after the GPU job generated the fakes)")
    args = ap.parse_args()

    run, data = Path(args.run), Path(args.data)
    if args.from_cache:
        c = np.load(run / "eval_cache.npz")
        src_n, fake, real_n = c["src"], c["fake"], c["real"]
        print(f"from cache: {len(fake)} fakes", flush=True)
    else:
        ckpt = torch.load(run / "generator.pt", map_location=args.device)
        G = ResnetGenerator(1, 1, ckpt["args"]["ngf"], ckpt["args"]["n_blocks"])
        G.load_state_dict(ckpt["G"]); G.eval().to(args.device)
        src = np.load(data / "source_cbct.npz")["images"]
        real = np.load(data / "target_us.npz")["images"]
        rng = np.random.default_rng(0)
        si = rng.permutation(len(src))[:args.n]; ri = rng.permutation(len(real))[:args.n]
        src_n = np.stack([_to_unit(src[i]) for i in si])[:, None]    # (n,1,H,W) [-1,1]
        real_n = np.stack([_to_unit(real[i]) for i in ri])[:, None]
        fake = []
        with torch.no_grad():
            for i in range(0, len(src_n), 16):
                fake.append(G(torch.from_numpy(src_n[i:i + 16]).to(args.device)).cpu().numpy())
        fake = np.concatenate(fake)
        print(f"generated {len(fake)} fakes from {len(src_n)} CBCT slices", flush=True)
        # cache the full pools so FID can be computed on the login node (torchvision + internet)
        np.savez_compressed(run / "eval_cache.npz", src=src_n, fake=fake, real=real_n)

    # structure consistency
    matched, shuffled = structure_consistency(src_n[:, 0], fake[:, 0])
    print(f"\nSTRUCTURE-CONSISTENCY (anti-hallucination):")
    print(f"  matched  (x_i, G(x_i)) low-freq NCC : mean={matched.mean():.3f}  median={np.median(matched):.3f}")
    print(f"  shuffled (x_i, G(x_j))             : mean={shuffled.mean():.3f}")
    print(f"  gap (matched - shuffled)           : {matched.mean() - shuffled.mean():+.3f}  "
          f"(large gap = G follows ITS input; ~0 = ignores input/hallucination)")

    # surface-boundary localization — the targeted, uniformity-robust structure check
    surf_m, surf_s = surface_alignment(src_n[:, 0], fake[:, 0])
    print(f"\nSURFACE LOCALIZATION (does G keep the surface where the CBCT has it):")
    print(f"  matched  (x_i, G(x_i)) surface-depth error : mean={surf_m.mean():.1f} mm  median={np.median(surf_m):.1f} mm")
    print(f"  shuffled (x_i, G(x_j))                     : mean={surf_s.mean():.1f} mm")
    print(f"  -> matched << shuffled means the surface is preserved (structure kept), not invented")

    # FID / KID
    print("\nDISTRIBUTION REALISM (vs real US):", flush=True)
    try:
        ff = inception_features(fake, args.device)
        fr = inception_features(real_n, args.device)
        print(f"  FID = {fid(ff, fr):.1f}   KID(x1000) = {kid(ff, fr):.2f}   (lower = closer to real)")
        fid_v, kid_v = fid(ff, fr), kid(ff, fr)
    except Exception as e:                                       # torchvision/inception unavailable
        print(f"  (skipped FID/KID: {e})")
        fid_v = kid_v = None

    out = Path(args.run) / "metrics.json"
    out.write_text(json.dumps({
        "n": len(fake), "from_cache": args.from_cache,
        "structure_matched_mean": matched.mean(), "structure_shuffled_mean": shuffled.mean(),
        "structure_gap": matched.mean() - shuffled.mean(),
        "surface_matched_mm": surf_m.mean(), "surface_shuffled_mm": surf_s.mean(),
        "fid": fid_v, "kid_x1000": kid_v,
    }, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
