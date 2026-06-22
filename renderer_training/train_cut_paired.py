#!/usr/bin/env python
"""Train the renderer with CUT plus a weak paired loss on LC2-refined pairs.

Why this script exists
----------------------
`scripts/train_renderer.py` trains pure CUT on two unpaired pools. That is still a valid
baseline, but we now have better-aligned pairs from `pair_generation.py`:

    CBCT fan slice at multi-frame LC2 pose  <->  real US frame

Those pairs are useful, but they are not perfect pixel supervision. LC2 improves the geometry,
yet residual cross-modality and pose errors remain. So this trainer keeps the robust CUT losses:

    GAN      : make generated images look like real US.
    PatchNCE : preserve source CBCT structure without requiring exact pixel pairing.

and adds only a small paired term:

    lambda_pair * L1(G(CBCT_slice), paired_real_US)

Keep `lambda_pair` small. If it dominates, the network can learn blur or wrong structures from
remaining registration error. The paired term should gently bias the output toward the aligned
US frame, not turn the experiment into pix2pix.

Example:
    python renderer_training/train_cut_paired.py \
        --data data/renderer_lc2_pairs \
        --out runs/renderer_cut_lc2_pair \
        --epochs 200 --batch 4 --lambda-pair 0.05
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deepussim.renderer.cut import CUTModel


def _to_unit(x: np.ndarray) -> np.ndarray:
    """Per-image min/max normalisation to the generator's [-1, 1] range."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


class LC2PairDataset(Dataset):
    """LC2-refined paired renderer data.

    `cbct` and `us` are paired by index in `pairs.npz`. We still train with CUT losses, but
    the paired US is also available for the weak L1 term and for sample dumps.
    """

    def __init__(self, pairs_npz: Path):
        data = np.load(pairs_npz)
        self.cbct = data["cbct"]
        self.us = data["us"]
        self.sequence = data["sequence"] if "sequence" in data.files else None
        self.frame_index = data["frame_index"] if "frame_index" in data.files else None
        if len(self.cbct) != len(self.us):
            raise ValueError(f"cbct/us length mismatch: {len(self.cbct)} vs {len(self.us)}")

    def __len__(self) -> int:
        return len(self.cbct)

    def __getitem__(self, i: int) -> dict:
        item = {
            "src": torch.from_numpy(_to_unit(self.cbct[i]))[None],
            "paired_tgt": torch.from_numpy(_to_unit(self.us[i]))[None],
        }
        if self.sequence is not None:
            item["sequence"] = str(self.sequence[i])
        if self.frame_index is not None:
            item["frame_index"] = int(self.frame_index[i])
        return item


def paired_l1(fake: torch.Tensor, target: torch.Tensor, blur: int = 1) -> torch.Tensor:
    """Weak paired loss.

    `blur > 1` applies average pooling before L1. This makes the paired term less sensitive to
    small residual registration errors and speckle differences. Use odd blur sizes such as 5 or 9.
    """
    if blur > 1:
        if blur % 2 == 0:
            raise ValueError("--pair-blur must be odd when > 1")
        fake = F.avg_pool2d(fake, kernel_size=blur, stride=1, padding=blur // 2)
        target = F.avg_pool2d(target, kernel_size=blur, stride=1, padding=blur // 2)
    return F.l1_loss(fake, target)


def _materialize_patch_mlp(model: CUTModel, batch: dict, device: str) -> None:
    """Build CUT's lazy PatchSampleMLP before creating the optimiser."""
    with torch.no_grad():
        src0 = batch["src"].to(device)
        model.patchnce(src0, model.G(src0))


def _load_checkpoint(model: CUTModel, checkpoint: str, device: str) -> list[str]:
    ck = torch.load(checkpoint, map_location=device)
    model.G.load_state_dict(ck["G"])
    loaded = ["G"]
    for key, net in (("D", model.D), ("F", model.F)):
        if key in ck:
            net.load_state_dict(ck[key])
            loaded.append(key)
    return loaded


def _write_losses_header(csv_path: Path, keys: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("epoch," + ",".join(keys) + "\n", encoding="utf-8")


def _append_losses(csv_path: Path, epoch: int, means: dict[str, float], keys: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{epoch}," + ",".join(f"{means[k]:.6f}" for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/renderer_lc2_pairs",
                    help="directory containing pairs.npz from pair_generation.py")
    ap.add_argument("--pairs", default=None, help="explicit pairs.npz path; overrides --data")
    ap.add_argument("--out", default="runs/renderer_cut_lc2_pair")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--num-patches", type=int, default=256)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--lambda-pair", type=float, default=0.05,
                    help="small weight for paired L1; keep low because LC2 pairs are not pixel-perfect")
    ap.add_argument("--pair-blur", type=int, default=1,
                    help="odd blur kernel for paired L1; 1 = plain L1, 5/9 = smoother weak pairing")
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--resume", help="checkpoint to fine-tune from")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pairs = Path(args.pairs) if args.pairs else Path(args.data) / "pairs.npz"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = LC2PairDataset(pairs)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )
    if len(dl) == 0:
        raise SystemExit(f"not enough data for batch={args.batch}: {len(ds)} pairs")
    print(
        f"device={args.device} | pairs={len(ds)} | batches/epoch={len(dl)} | "
        f"lambda_pair={args.lambda_pair} pair_blur={args.pair_blur}",
        flush=True,
    )
    print(
        f"data={pairs} | out={out} | cbct_shape={ds.cbct.shape} | us_shape={ds.us.shape} | "
        f"batch={args.batch} lr={args.lr} epochs={args.epochs} ngf={args.ngf} ndf={args.ndf} "
        f"n_blocks={args.n_blocks} num_patches={args.num_patches}",
        flush=True,
    )

    model = CUTModel(
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        lambda_nce=args.lambda_nce,
        num_patches=args.num_patches,
    ).to(args.device)

    sample_batch = next(iter(dl))
    _materialize_patch_mlp(model, sample_batch, args.device)

    if args.resume:
        loaded = _load_checkpoint(model, args.resume, args.device)
        print(f"resumed {','.join(loaded)} from {args.resume}", flush=True)

    opt_g = torch.optim.Adam(
        list(model.G.parameters()) + list(model.F.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(model.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    loss_keys: list[str] | None = None
    csv_path = out / "losses.csv"
    for epoch in range(1, args.epochs + 1):
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        agg: dict[str, float] = {}

        for batch in dl:
            src = batch["src"].to(args.device)
            paired = batch["paired_tgt"].to(args.device)

            # Generator/F step:
            # - CUT's GAN + PatchNCE preserve the unpaired translation behaviour.
            # - The paired term is intentionally weak and uses the LC2-refined pair only as a bias.
            opt_g.zero_grad()
            g_cut, fake, parts = model.g_loss(src, paired)
            pair_raw = paired_l1(fake, paired, blur=args.pair_blur)
            pair_weighted = pair_raw * args.lambda_pair
            g_total = g_cut + pair_weighted
            g_total.backward()
            opt_g.step()

            # Discriminator step uses the paired real US batch as real-domain samples. The GAN loss
            # only cares about the US distribution, not whether the real image is paired.
            opt_d.zero_grad()
            d = model.d_loss(paired, fake)
            d.backward()
            opt_d.step()

            parts.update(
                pair_raw=float(pair_raw.detach()),
                pair=float(pair_weighted.detach()),
                total=float(g_total.detach()),
                d=float(d.detach()),
            )
            for key, value in parts.items():
                agg[key] = agg.get(key, 0.0) + value

        means = {key: agg[key] / len(dl) for key in agg}
        if loss_keys is None:
            loss_keys = sorted(means)
            _write_losses_header(csv_path, loss_keys)
        _append_losses(csv_path, epoch, means, loss_keys)

        msg = "  ".join(f"{key}={means[key]:.3f}" for key in loss_keys)
        mem = ""
        if args.device == "cuda":
            alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            mem = f"  cuda_peak={alloc:.1f}GiB reserved={reserved:.1f}GiB"
        print(f"epoch {epoch:4d}/{args.epochs}  {msg}  ({time.time() - t0:.1f}s){mem}", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                s = sample_batch["src"].to(args.device)
                p = sample_batch["paired_tgt"].to(args.device)
                f = model.G(s)
            np.savez_compressed(
                out / f"samples_ep{epoch:04d}.npz",
                src=sample_batch["src"].numpy(),
                fake=f.cpu().numpy(),
                paired_us=sample_batch["paired_tgt"].numpy(),
            )
            torch.save(
                {
                    "G": model.G.state_dict(),
                    "D": model.D.state_dict(),
                    "F": model.F.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                out / "generator.pt",
            )
            model.train()

    print(f"done -> {out / 'generator.pt'}", flush=True)


if __name__ == "__main__":
    main()
