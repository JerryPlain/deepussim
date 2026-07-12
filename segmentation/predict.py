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

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
from train_sam2_head import LiverSet, SegHead, load_sam2_encoder, dice_iou  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "liver_seg")
    ap.add_argument("--sam2-ckpt", required=True)
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--head", type=Path, required=True, help="trained head_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-viz", type=int, default=8, help="frames to draw in overlay.png")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or (args.head.parent / f"predict_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    embed = load_sam2_encoder(args.sam2_cfg, str(args.sam2_ckpt), device)
    head = SegHead().to(device).eval()
    head.load_state_dict(torch.load(args.head, map_location=device, weights_only=False)["head"])

    ds = LiverSet(args.data, args.split)
    meta = list(csv.DictReader((args.data / args.split / "meta.csv").open()))

    preds = np.zeros((len(ds), *ds.masks.shape[1:]), np.uint8)
    rows, dices, ious, dices_pos = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, full, j = ds[i]
            logit = head(embed(img[None].to(device)))
            pred = F.interpolate(logit, size=full.shape[-2:], mode="bilinear", align_corners=False)
            pred_bin = (torch.sigmoid(pred)[0, 0].cpu() > 0.5).float()
            preds[i] = pred_bin.numpy().astype(np.uint8)
            d, iou = dice_iou(pred_bin, full)
            dices.append(d); ious.append(iou)
            is_pos = int(full.sum() > 0)
            if is_pos:
                dices_pos.append(d)
            rows.append({"sequence": meta[j]["sequence"], "frame": meta[j]["frame"],
                         "is_positive": is_pos, "dice": round(d, 4), "iou": round(iou, 4)})

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

    print(f"{args.split}: {len(ds)} frames")
    print(f"  Dice  (all)       = {np.mean(dices):.3f}")
    print(f"  IoU   (all)       = {np.mean(ious):.3f}")
    print(f"  Dice  (liver-pos) = {np.mean(dices_pos) if dices_pos else 0:.3f}  ({len(dices_pos)} frames)")
    print(f"wrote {out}/predictions.npy, per_frame.csv, overlay.png")


if __name__ == "__main__":
    main()
