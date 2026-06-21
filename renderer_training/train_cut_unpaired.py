#!/usr/bin/env python
"""Train the pure CUT renderer baseline on LC2-refined renderer pools.

This is the baseline for comparison against `train_cut_paired.py`.

Data source
-----------
`pair_generation.py` writes three artifacts:

    pairs.npz          paired CBCT/US data for semi-paired experiments
    source_cbct.npz    CBCT source pool, key `images`
    target_us.npz      real-US target pool, key `images`

This script deliberately ignores the pair relation and trains CUT exactly as an unpaired
translation method:

    G: CBCT slice -> US-like image
    losses: GAN(real US distribution) + PatchNCE(structure preservation)

Why keep this baseline
----------------------
The LC2-refined pairs are better than raw reslice, but they are still not pixel-perfect.
Pure CUT is therefore the conservative baseline: it learns real-US appearance from the target
pool and preserves CBCT structure through PatchNCE without trusting per-pixel supervision.

Example:
    python renderer_training/train_cut_unpaired.py \
        --data data/renderer_lc2_pairs \
        --out runs/renderer_cut_unpaired \
        --epochs 200 --batch 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepussim.renderer.cut import CUTModel
from deepussim.renderer.data import UnpairedFanDataset


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
    csv_path.write_text("epoch," + ",".join(keys) + "\n", encoding="utf-8")


def _append_losses(csv_path: Path, epoch: int, means: dict[str, float], keys: list[str]) -> None:
    with csv_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{epoch}," + ",".join(f"{means[k]:.6f}" for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/renderer_lc2_pairs",
                    help="directory with source_cbct.npz and target_us.npz from pair_generation.py")
    ap.add_argument("--source", default=None, help="explicit source_cbct.npz path; overrides --data")
    ap.add_argument("--target", default=None, help="explicit target_us.npz path; overrides --data")
    ap.add_argument("--out", default="runs/renderer_cut_unpaired")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--num-patches", type=int, default=256)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--resume", help="checkpoint to fine-tune from")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = Path(args.data)
    source = Path(args.source) if args.source else data_dir / "source_cbct.npz"
    target = Path(args.target) if args.target else data_dir / "target_us.npz"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = UnpairedFanDataset(source, target)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )
    if len(dl) == 0:
        raise SystemExit(f"not enough data for batch={args.batch}: {len(ds)} source images")
    print(
        f"device={args.device} | source={len(ds.src)} target={len(ds.tgt)} | "
        f"batches/epoch={len(dl)}",
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
        t0 = time.time()
        agg: dict[str, float] = {}

        for batch in dl:
            src = batch["src"].to(args.device)
            tgt = batch["tgt"].to(args.device)

            # Generator/F step: pure CUT, no paired supervision.
            opt_g.zero_grad()
            g_total, fake, parts = model.g_loss(src, tgt)
            g_total.backward()
            opt_g.step()

            # Discriminator step: distinguish real US target-pool samples from generated US.
            opt_d.zero_grad()
            d = model.d_loss(tgt, fake)
            d.backward()
            opt_d.step()

            parts.update(total=float(g_total.detach()), d=float(d.detach()))
            for key, value in parts.items():
                agg[key] = agg.get(key, 0.0) + value

        means = {key: agg[key] / len(dl) for key in agg}
        if loss_keys is None:
            loss_keys = sorted(means)
            _write_losses_header(csv_path, loss_keys)
        _append_losses(csv_path, epoch, means, loss_keys)

        msg = "  ".join(f"{key}={means[key]:.3f}" for key in loss_keys)
        print(f"epoch {epoch:4d}/{args.epochs}  {msg}  ({time.time() - t0:.1f}s)", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                s = sample_batch["src"].to(args.device)
                f = model.G(s)
            np.savez_compressed(
                out / f"samples_ep{epoch:04d}.npz",
                src=sample_batch["src"].numpy(),
                fake=f.cpu().numpy(),
                tgt=sample_batch["tgt"].numpy(),
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
