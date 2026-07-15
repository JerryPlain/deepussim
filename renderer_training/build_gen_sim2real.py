#!/usr/bin/env python
"""Render the sim2real US and build data/liver_seg/gen_sim2real (same poses & masks as gen).

The sim2real generator (cut_phys2real) maps physics_US -> real-looking US. We already have the
physics US for the gen poses in data/liver_seg/gen_phys/images.npy, so sim2real US = G(physics US).
Reuse the identical masks -> a controlled 4th arm (A / +CUT / +physics / +sim2real).

    source $WORK/venvs/sam2seg/bin/activate ; export PYTHONPATH=$PWD/src
    python renderer_training/build_gen_sim2real.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deepussim.renderer.networks import ResnetGenerator                  # noqa: E402


def _to_unit(x):
    x = np.asarray(x, np.float32); lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-6 else (x - lo) / (hi - lo) * 2 - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-ckpt", type=Path, default=REPO_ROOT / "runs" / "renderer_cut_phys2real" / "generator.pt")
    ap.add_argument("--phys", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "gen_phys")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "gen_sim2real")
    ap.add_argument("--preview", type=Path, default=REPO_ROOT / "figures" / "16_sim2real")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.gen_ckpt, map_location=device, weights_only=False)
    G = ResnetGenerator(1, 1, ck["args"]["ngf"], ck["args"]["n_blocks"])
    G.load_state_dict(ck["G"]); G.eval().to(device)

    phys = np.load(args.phys / "images.npy")                          # physics US (uint8)
    out_imgs = []
    with torch.no_grad():
        for i in range(0, len(phys), 8):
            x = torch.from_numpy(np.stack([_to_unit(phys[k]) for k in range(i, min(i + 8, len(phys)))]))[:, None]
            y = G(x.to(device)).cpu().numpy()[:, 0]                    # [-1,1]
            out_imgs.append(y)
    sim = np.concatenate(out_imgs)
    sim_u8 = np.clip((sim + 1) / 2 * 255, 0, 255).astype(np.uint8)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "images.npy", sim_u8)
    np.save(args.out / "masks.npy", np.load(args.phys / "masks.npy"))
    shutil.copy(args.phys / "meta.csv", args.out / "meta.csv")
    print(f"wrote {args.out}: {len(sim_u8)} sim2real frames {sim_u8.shape}")

    # preview: physics -> sim2real vs CUT-gen vs real, a few frames
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from style.style import save, TYPE
    cut = np.load(REPO_ROOT / "data" / "liver_seg" / "gen" / "images.npy")
    real = np.load(REPO_ROOT / "data" / "liver_seg" / "train" / "images.npy")
    rows = 4
    fig, ax = plt.subplots(rows, 4, figsize=(11, 2.7 * rows))
    cols = ["physics US (input)", "sim2real (output)", "CUT gen", "real US (example)"]
    for r in range(rows):
        for c, im in enumerate([phys[r], sim_u8[r], cut[r], real[r]]):
            ax[r, c].imshow(im, cmap="gray"); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=TYPE["small"])
    fig.suptitle("sim2real: physics-US refined to real texture", fontsize=TYPE["body"])
    fig.tight_layout()
    args.preview.mkdir(parents=True, exist_ok=True)
    save(fig, str(args.preview / "sim2real_samples"))


if __name__ == "__main__":
    main()
