#!/usr/bin/env python
"""Train the learned US renderer (B1): CUT on the two fan-layout pools.

CBCT-reslice (source) -> US (target), unpaired. Adversarial + PatchNCE (see
deepussim.renderer.cut). Logs losses to stdout (SLURM-friendly), checkpoints the generator,
and dumps (src, fake, tgt) sample stacks as .npz for offline inspection with
plot_script (no matplotlib needed in the container).

    python scripts/train_renderer.py --data data/renderer --out runs/renderer_cut \
        --epochs 200 --batch 4

This is the feasibility probe: does this low-texture CBCT translate into believable US at all?
Treat the first result as go/no-go, not a final model (see docs/renderer.md).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/renderer", help="dir with source_cbct.npz + target_us.npz")
    ap.add_argument("--out", default="runs/renderer_cut", help="checkpoints + sample dumps")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--num-patches", type=int, default=256)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=10, help="dump samples every N epochs")
    ap.add_argument("--resume", help="checkpoint to fine-tune from (e.g. a previous generator.pt). "
                    "Loads G (and D/F if present) and continues training on THIS --data — point "
                    "--data at pools built from OLD + NEW sequences to avoid forgetting.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    ds = UnpairedFanDataset(data / "source_cbct.npz", data / "target_us.npz")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=(args.device == "cuda"))
    print(f"device={args.device} | {len(ds)} source imgs | {len(dl)} batches/epoch", flush=True)

    model = CUTModel(ngf=args.ngf, ndf=args.ndf, n_blocks=args.n_blocks,
                     lambda_nce=args.lambda_nce, num_patches=args.num_patches).to(args.device)

    # PatchSampleMLP (F) builds its layers lazily on first forward — run one to materialise them
    # before constructing the optimiser, then include F's params with G's (CUT convention).
    b0 = next(iter(dl))
    with torch.no_grad():
        src0 = b0["src"].to(args.device)
        model.patchnce(src0, model.G(src0))
    opt_g = torch.optim.Adam(list(model.G.parameters()) + list(model.F.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if args.resume:                                          # fine-tune: load weights, fresh optimiser
        ck = torch.load(args.resume, map_location=args.device)
        model.G.load_state_dict(ck["G"])
        loaded = ["G"]
        for k, net in (("D", model.D), ("F", model.F)):
            if k in ck:
                net.load_state_dict(ck[k]); loaded.append(k)
        print(f"resumed {','.join(loaded)} from {args.resume} (epoch {ck.get('epoch')}) "
              f"— fine-tuning on {len(ds)} imgs", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); agg = {}
        for batch in dl:
            src = batch["src"].to(args.device); tgt = batch["tgt"].to(args.device)
            # G + F step
            opt_g.zero_grad()
            g_total, fake, parts = model.g_loss(src, tgt)
            g_total.backward(); opt_g.step()
            # D step
            opt_d.zero_grad()
            d = model.d_loss(tgt, fake)
            d.backward(); opt_d.step()
            parts["d"] = float(d.detach())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        means = {k: agg[k] / len(dl) for k in agg}
        msg = "  ".join(f"{k}={means[k]:.3f}" for k in sorted(means))
        print(f"epoch {epoch:4d}/{args.epochs}  {msg}  ({time.time() - t0:.1f}s)", flush=True)
        csv = out / "losses.csv"
        if epoch == 1:
            csv.write_text("epoch," + ",".join(sorted(means)) + "\n")
        with csv.open("a") as fh:
            fh.write(f"{epoch}," + ",".join(f"{means[k]:.5f}" for k in sorted(means)) + "\n")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                s = b0["src"].to(args.device); f = model.G(s)
            np.savez_compressed(out / f"samples_ep{epoch:04d}.npz",
                                src=b0["src"].numpy(), fake=f.cpu().numpy(), tgt=b0["tgt"].numpy())
            # save G/D/F (D,F let a later --resume continue cleanly; NeuralRenderer/eval read just G+args)
            torch.save({"G": model.G.state_dict(), "D": model.D.state_dict(),
                        "F": model.F.state_dict(), "epoch": epoch, "args": vars(args)},
                       out / "generator.pt")
            model.train()
    print(f"done -> {out}/generator.pt", flush=True)


if __name__ == "__main__":
    main()
