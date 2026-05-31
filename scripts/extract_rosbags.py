#!/usr/bin/env python
"""Step (doc v2 §4.3): extract the real US sequences from ROS1 rosbags.

Reads each bag's image + pose topics, syncs them by header stamp, tags dark/non-contact
frames, and (optionally) saves the synced sequence to a compressed ``.npz`` and a few
preview PNGs for a visual sanity check.

    python scripts/extract_rosbags.py data/rosbags/phantom.bag data/rosbags/phantom1.bag \
        --out data/sequences --preview 6

The saved poses are ``T_base_from_ee`` (RTE(t)) in metres — compose with ETU downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deepussim.data.rosbag import extract_sequence


def save_npz(seq, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seq.name}.npz"
    np.savez_compressed(
        path,
        images=seq.images(),
        poses=seq.poses(),
        contact=seq.contact_mask(),
        stamps=np.array([f.t for f in seq.frames]),
        mean_intensity=np.array([f.meta["mean_intensity"] for f in seq.frames]),
        sync_dt_s=np.array([f.meta["sync_dt_s"] for f in seq.frames]),
    )
    return path


def save_previews(seq, out_dir: Path, n: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  (install pillow for --preview PNGs; skipping)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(seq) - 1, num=min(n, len(seq)), dtype=int)
    for i in idx:
        f = seq.frames[i]
        tag = "contact" if f.contact else "dark"
        Image.fromarray(f.image).save(out_dir / f"{seq.name}_{i:04d}_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+", help="ROS1 .bag files")
    ap.add_argument("--out", help="dir to write synced <name>.npz (and previews)")
    ap.add_argument("--dark-threshold", type=float, default=6.0,
                    help="grayscale mean below which a frame is tagged non-contact")
    ap.add_argument("--max-sync-dt", type=float, default=None,
                    help="drop images whose nearest pose is farther than this (s)")
    ap.add_argument("--preview", type=int, default=0,
                    help="save N evenly-spaced preview PNGs per sequence")
    args = ap.parse_args()

    for bag in args.bags:
        seq = extract_sequence(bag, dark_threshold=args.dark_threshold,
                               max_sync_dt_s=args.max_sync_dt)
        m = seq.meta
        sync = np.array([f.meta["sync_dt_s"] for f in seq.frames])
        print(f"\n{seq.name}: {m['n_frames']} frames "
              f"({m['n_dark']} dark / {100 * m['n_dark'] / max(1, m['n_frames']):.0f}%), "
              f"{m['duration_s']:.1f}s")
        print(f"  images={m['n_images']} poses={m['n_poses']}  "
              f"sync_dt median={np.median(sync) * 1e3:.1f}ms max={sync.max() * 1e3:.1f}ms")
        if args.out:
            path = save_npz(seq, Path(args.out))
            print(f"  saved {path}")
            if args.preview:
                save_previews(seq, Path(args.out) / "preview", args.preview)
                print(f"  wrote {args.preview} previews to {Path(args.out) / 'preview'}")


if __name__ == "__main__":
    main()
