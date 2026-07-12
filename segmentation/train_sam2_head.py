#!/usr/bin/env python
"""Group A baseline: frozen SAM2 image encoder + a small trainable seg head, on LIMITED real US.

Goal for this first run is to get the pipeline working end to end and a sane Dice on the
held-out real test set -- not to win. The SAM2 Hiera image encoder is frozen; only a light
conv decoder head is trained. Generated-US augmentation (group B) comes later, same code.

Data: data/liver_seg/{train,test}/{images.npy,masks.npy,meta.csv} (built by renderer_training/
build_liver_dataset.py). "Limited real" = subsample the train split to --n-pos + --n-neg frames.

    python segmentation/train_sam2_head.py \
        --sam2-ckpt $WORK/sam2/sam2.1_hiera_small.pt \
        --sam2-cfg  configs/sam2.1/sam2.1_hiera_s.yaml \
        --n-pos 40 --n-neg 40 --epochs 60
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 1024                                  # SAM2 image encoder input
LOSS_RES = 256                                   # decoder output / loss resolution
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- data
class LiverSet(Dataset):
    """US frame -> (normalized 3x1024x1024 input, low-res mask for loss, full mask for eval)."""

    def __init__(self, root: Path, split: str, indices: np.ndarray | None = None):
        self.images = np.load(root / split / "images.npy")           # (N,660,880) uint8
        self.masks = np.load(root / split / "masks.npy")             # (N,660,880) uint8 {0,1}
        self.idx = np.arange(len(self.images)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        img = torch.from_numpy(self.images[j].astype(np.float32) / 255.0)[None, None]   # 1,1,H,W
        img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        img = img.repeat(1, 3, 1, 1)[0]                                                  # 3,1024,1024
        img = (img - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        m = torch.from_numpy(self.masks[j].astype(np.float32))[None, None]              # 1,1,H,W
        m_low = F.interpolate(m, size=(LOSS_RES, LOSS_RES), mode="nearest")[0, 0]
        return img, m_low, torch.from_numpy(self.masks[j].astype(np.float32)), j


# --------------------------------------------------------------------------- model
class SegHead(nn.Module):
    """Light decoder: SAM2 vision_features (256ch, 64x64) -> 1-channel logits at LOSS_RES."""

    def __init__(self, in_ch: int = 256):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())
        self.dec = nn.ModuleList([block(in_ch, 128), block(128, 64), block(64, 32)])
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:            # feat: B,256,64,64
        x = feat
        for blk in self.dec:                                         # 64 -> 128 -> 256 -> 512
            x = blk(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = F.interpolate(x, size=(LOSS_RES, LOSS_RES), mode="bilinear", align_corners=False)
        return self.out(x)                                           # B,1,LOSS_RES,LOSS_RES


def load_sam2_encoder(cfg: str, ckpt: str, device: str):
    """Return the frozen SAM2 image encoder as a callable x -> vision_features (B,256,64,64)."""
    from sam2.build_sam import build_sam2
    model = build_sam2(cfg, ckpt, device=device, mode="eval")
    encoder = model.image_encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def embed(x: torch.Tensor) -> torch.Tensor:
        out = encoder(x)
        return out["vision_features"] if isinstance(out, dict) else out

    return embed


# --------------------------------------------------------------------------- loss / metric
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def dice_iou(pred_bin, gt):                                          # full-res binary tensors
    inter = (pred_bin * gt).sum()
    d = (2 * inter / (pred_bin.sum() + gt.sum() + 1e-6)).item() if (pred_bin.sum() + gt.sum()) > 0 else 1.0
    union = ((pred_bin + gt) > 0).sum()
    iou = (inter / union).item() if union > 0 else 1.0
    return d, iou


# --------------------------------------------------------------------------- train / eval
def limited_indices(root: Path, n_pos: int, n_neg: int, seed: int) -> np.ndarray:
    meta = list(csv.DictReader((root / "train" / "meta.csv").open()))
    pos = np.array([i for i, r in enumerate(meta) if int(r["is_positive"]) == 1])
    neg = np.array([i for i, r in enumerate(meta) if int(r["is_positive"]) == 0])
    rng = np.random.default_rng(seed)
    sel_p = rng.permutation(pos)[:n_pos]
    sel_n = rng.permutation(neg)[:min(n_neg, len(neg))]
    return np.concatenate([sel_p, sel_n])


@torch.no_grad()
def evaluate(embed, head, loader, device):
    head.eval()
    dices, ious, dices_pos = [], [], []
    for img, _, full, _ in loader:
        img = img.to(device)
        logit = head(embed(img))
        pred = F.interpolate(logit, size=full.shape[-2:], mode="bilinear", align_corners=False)
        pred_bin = (torch.sigmoid(pred)[:, 0].cpu() > 0.5).float()
        for k in range(len(full)):
            d, iou = dice_iou(pred_bin[k], full[k])
            dices.append(d); ious.append(iou)
            if full[k].sum() > 0:
                dices_pos.append(d)
    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(dices_pos) if dices_pos else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "liver_seg")
    ap.add_argument("--sam2-ckpt", required=True, help="path to sam2.1 hiera checkpoint .pt")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--n-pos", type=int, default=40, help="limited-real positive frames")
    ap.add_argument("--n-neg", type=int, default=40, help="limited-real negative frames")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "seg_sam2_A_limited")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  torch={torch.__version__}", flush=True)

    tr_idx = limited_indices(args.data, args.n_pos, args.n_neg, args.seed)
    tr = DataLoader(LiverSet(args.data, "train", tr_idx), batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True)
    te = DataLoader(LiverSet(args.data, "test"), batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"train (limited): {len(tr_idx)} frames  |  test: {len(te.dataset)} frames", flush=True)

    embed = load_sam2_encoder(args.sam2_cfg, args.sam2_ckpt, device)
    head = SegHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    log, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        head.train(); tot = 0.0
        for img, m_low, _, _ in tr:
            img = img.to(device); m_low = m_low.to(device)[:, None]
            logit = head(embed(img))
            loss = F.binary_cross_entropy_with_logits(logit, m_low) + dice_loss(logit, m_low)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs:
            d, iou, dpos = evaluate(embed, head, te, device)
            log.append({"epoch": ep, "loss": tot / len(tr), "test_dice": d, "test_iou": iou, "test_dice_pos": dpos})
            print(f"ep {ep:3d}  loss={tot/len(tr):.3f}  test Dice={d:.3f}  IoU={iou:.3f}  Dice(pos)={dpos:.3f}", flush=True)
            if dpos > best:
                best = dpos
                torch.save({"head": head.state_dict(), "args": vars(args), "epoch": ep}, args.out / "head_best.pt")

    (args.out / "metrics.json").write_text(json.dumps({"log": log, "best_dice_pos": best,
                                                       "n_train": int(len(tr_idx))}, indent=2, default=float))
    print(f"\nbest test Dice(pos)={best:.3f}  ->  {args.out}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
