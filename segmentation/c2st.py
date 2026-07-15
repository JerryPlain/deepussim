#!/usr/bin/env python
"""Classifier Two-Sample Test (C2ST): are real and generated US distinguishable?

Encode real + generated US with the FROZEN SAM2 encoder (the exact feature space the
downstream head sees), train a logistic-regression probe real(0) vs generated(1), and
report held-out accuracy / AUC. C2ST accuracy ~0.5 = indistinguishable = realistic;
->1.0 = trivially separable = a domain gap the segmentation head will exploit.

Also emits a PER-FRAME generated-probability for every generated frame -> a ranking/filter
score (low p_gen = most real-like; keep those, drop the obvious fakes).

    source $WORK/venvs/sam2seg/bin/activate
    python segmentation/c2st.py --sam2-ckpt $WORK/sam2/sam2.1_hiera_small.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT))
from train_sam2_head import load_sam2_encoder, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


@torch.no_grad()
def encode(embed, imgs: np.ndarray, device: str, batch: int = 16) -> np.ndarray:
    """SAM2 pooled features (N, 256) for a stack of uint8/float US frames."""
    feats = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch].astype(np.float32))
        if x.max() > 1.5:
            x = x / 255.0
        x = x[:, None]                                                  # N,1,H,W
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        f = embed(x.to(device))                                        # N,256,64,64
        feats.append(f.mean((2, 3)).cpu().numpy())                     # global avg pool -> N,256
    return np.concatenate(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam2-ckpt", required=True)
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--real", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "train" / "images.npy")
    ap.add_argument("--gen", type=Path,
                    default=REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "13_confidence_realism")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    real = np.load(args.real)
    gen = (np.load(args.gen) if str(args.gen).endswith(".npy")
           else np.load(args.gen, allow_pickle=True)["generated_us"])
    n = min(len(real), len(gen))
    real = real[rng.permutation(len(real))[:n]]                        # balance classes

    embed = load_sam2_encoder(args.sam2_cfg, args.sam2_ckpt, device)
    Xr, Xg = encode(embed, real, device), encode(embed, gen, device)
    X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd

    # 70/30 split, logistic-regression probe (torch, no sklearn dependency)
    perm = rng.permutation(len(Xs)); ntr = int(0.7 * len(Xs))
    tri, tei = perm[:ntr], perm[ntr:]
    Xt = torch.tensor(Xs[tri], dtype=torch.float32); yt = torch.tensor(y[tri], dtype=torch.float32)
    Xe = torch.tensor(Xs[tei], dtype=torch.float32); ye = y[tei]
    w = torch.zeros(Xs.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        pe = torch.sigmoid(Xe @ w + b).numpy()
    acc = float(((pe > 0.5).astype(int) == ye).mean())
    order = np.argsort(np.concatenate([pe[ye == 0], pe[ye == 1]]))     # AUC via rank
    r = np.argsort(np.argsort(pe)); npos = int((ye == 1).sum()); nneg = int((ye == 0).sum())
    auc = float((r[ye == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))

    # per-frame generated-probability for ALL gen frames (filter score)
    with torch.no_grad():
        p_gen_all = torch.sigmoid(torch.tensor((Xg - mu) / sd, dtype=torch.float32) @ w + b).numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "c2st_scores.npz", p_gen_all=p_gen_all, held_out_acc=acc, auc=auc)
    (args.out / "c2st_summary.json").write_text(json.dumps({
        "n_per_class": int(n), "held_out_accuracy": round(acc, 4), "auc": round(auc, 4),
        "interpretation": "0.5=indistinguishable(realistic); ->1.0=separable(domain gap)"}, indent=2))
    print(f"C2ST (SAM2 feature space, real vs generated):")
    print(f"  held-out accuracy = {acc:.3f}   (0.5 = indistinguishable)")
    print(f"  AUC               = {auc:.3f}")
    print(f"  per-frame p_gen: median={np.median(p_gen_all):.3f}  "
          f"[{int((p_gen_all<0.5).sum())}/{len(p_gen_all)} gen frames look real-ish (p<0.5)]")
    print(f"wrote {args.out}/c2st_scores.npz, c2st_summary.json")


if __name__ == "__main__":
    main()
