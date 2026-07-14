#!/usr/bin/env python
"""Apply a trained seg head to a split (default: test) and score the segmentation.

Loads the frozen SAM2 encoder + the trained head (head_best.pt from train_sam2_head.py),
predicts a liver mask for every frame of the split, and writes:
  - predictions.npy   (N,660,880) uint8 predicted binary masks
  - per_frame.csv      sequence, frame, is_positive, dice, iou
  - overlay.png        US + GT (red) vs prediction (lime) for a few frames
and prints the mean Dice / IoU (overall and over liver-positive frames).

    python segmentation/predict.py \
        --sam2-ckpt $WORK/sam2/sam2.1_hiera_small.pt \
        --head runs/seg_sam2_A_limited/head_best.pt --split test
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
from train_sam2_head import LiverSet, SegHead, load_sam2_encoder, dice_iou  # noqa: E402


def _surface(m: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (mask minus its erosion)."""
    m = m.astype(bool)
    return m & ~binary_erosion(m)


def surface_dice(pred: np.ndarray, gt: np.ndarray, tau_px: float) -> float:
    """Normalized Surface Dice @ tolerance tau (pixels): fraction of each boundary within
    tau of the other. Fair to a coarse GT whose exact boundary is uncertain. GT must be
    non-empty (call only on liver-positive frames)."""
    sp, sg = _surface(pred), _surface(gt)
    np_, ng = int(sp.sum()), int(sg.sum())
    if np_ == 0 and ng == 0:
        return 1.0
    if np_ == 0 or ng == 0:
        return 0.0
    dt_g = distance_transform_edt(~sg)        # distance from every pixel to nearest GT boundary
    dt_p = distance_transform_edt(~sp)
    p_close = int((dt_g[sp] <= tau_px).sum())
    g_close = int((dt_p[sg] <= tau_px).sum())
    return (p_close + g_close) / (np_ + ng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "liver_seg")
    ap.add_argument("--sam2-ckpt", required=True)
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--head", type=Path, required=True, help="trained head_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tau-mm", type=float, default=3.0, help="NSD tolerance in mm (~GT uncertainty)")
    ap.add_argument("--mm-per-px", type=float, default=0.166112957, help="US pixel spacing (mm/px)")
    ap.add_argument("--n-viz", type=int, default=8, help="frames to draw in overlay.png")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tau_px = args.tau_mm / args.mm_per_px

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or (args.head.parent / f"predict_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    embed = load_sam2_encoder(args.sam2_cfg, str(args.sam2_ckpt), device)
    head = SegHead().to(device).eval()
    head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"])

    ds = LiverSet(args.data, args.split)
    meta = list(csv.DictReader((args.data / args.split / "meta.csv").open()))

    preds = np.zeros((len(ds), *ds.masks.shape[1:]), np.uint8)
    rows, dices_pos, ious_pos, nsd_pos = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, full, j = ds[i]
            logit = head(embed(img[None].to(device)))
            pred = F.interpolate(logit, size=full.shape[-2:], mode="bilinear", align_corners=False)
            pred_bin = (torch.sigmoid(pred)[0, 0].cpu() > 0.5).float()
            preds[i] = pred_bin.numpy().astype(np.uint8)
            d, iou = dice_iou(pred_bin, full)
            is_pos = int(full.sum() > 0)
            nsd = surface_dice(preds[i], full.numpy().astype(bool), tau_px) if is_pos else float("nan")
            if is_pos:
                dices_pos.append(d); ious_pos.append(iou); nsd_pos.append(nsd)
            rows.append({"sequence": meta[j]["sequence"], "frame": meta[j]["frame"],
                         "is_positive": is_pos, "nsd": round(nsd, 4) if is_pos else "",
                         "dice": round(d, 4), "iou": round(iou, 4)})

    np.save(out / "predictions.npy", preds)
    with (out / "per_frame.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # overlay figure: prefer positive frames
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pos_i = [i for i in range(len(ds)) if ds.masks[int(ds.idx[i])].sum() > 0][: args.n_viz]
    n = len(pos_i)
    if n:
        cols = min(4, n); rowsn = (n + cols - 1) // cols
        fig, ax = plt.subplots(rowsn, cols, figsize=(4 * cols, 3.4 * rowsn)); ax = np.atleast_1d(ax).ravel()
        for k, i in enumerate(pos_i):
            j = int(ds.idx[i])
            ax[k].imshow(ds.images[j], cmap="gray")
            ax[k].contour(ds.masks[j].astype(float), [0.5], colors="red", linewidths=1.0)
            ax[k].contour(preds[i].astype(float), [0.5], colors="lime", linewidths=1.0)
            ax[k].set_title(f"{rows[i]['sequence']} f{rows[i]['frame']}  Dice={rows[i]['dice']:.2f}", fontsize=8)
            ax[k].axis("off")
        for k in range(n, len(ax)):
            ax[k].axis("off")
        fig.suptitle(f"{args.split}: GT (red) vs prediction (lime)")
        fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=115)

    npos = len(dices_pos)
    print(f"{args.split}: {len(ds)} frames  ({npos} liver-positive)  [metrics on positive frames]")
    print(f"  NSD @ {args.tau_mm:.0f}mm (primary) = {np.mean(nsd_pos) if npos else 0:.3f}   (fair to coarse GT)")
    print(f"  Dice  (reference)      = {np.mean(dices_pos) if npos else 0:.3f}")
    print(f"  IoU   (reference)      = {np.mean(ious_pos) if npos else 0:.3f}")
    import json
    (out / "summary.json").write_text(json.dumps({
        "split": args.split, "n_positive": npos, "tau_mm": args.tau_mm,
        "nsd": float(np.mean(nsd_pos)) if npos else 0.0,
        "dice": float(np.mean(dices_pos)) if npos else 0.0,
        "iou": float(np.mean(ious_pos)) if npos else 0.0}, indent=2))
    print(f"wrote {out}/predictions.npy, per_frame.csv, summary.json, overlay.png")


if __name__ == "__main__":
    main()
