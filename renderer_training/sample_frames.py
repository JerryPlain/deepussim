#!/usr/bin/env python
"""Motion-based de-redundant frame sampling per sequence, tagged with liver presence.

Greedy min-separation keyframe selection: keep a frame only once the probe has moved
>= --dt mm OR >= --dr deg from the last kept frame. This removes near-duplicate adjacent
freehand frames while keeping every genuinely distinct view, adapting to sweep length
(long sweeps yield more keyframes than short ones -- unlike a fixed count per sequence).

Each kept frame is resliced against the CBCT liver label (no confidence step here) so it can
be tagged positive/negative. Output feeds the by-sequence train/test split and the mask build.

    module load python/3.12-base
    python renderer_training/sample_frames.py --dt 6 --dr 15
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SimpleITK as sitk                                                  # noqa: E402
from reslice.frame import affine_from_sitk                              # noqa: E402
from reslice.io import load_volume_data                                 # noqa: E402
from renderer_training.project_labels_to_us import sector_zoom_pair     # noqa: E402
from renderer_training.allframe_poses import all_frame_poses            # noqa: E402

LIVER = 2


def _rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def greedy_keyframes(poses: np.ndarray, dt: float, dr: float) -> list[int]:
    keep = [0]
    for j in range(1, len(poses)):
        last = poses[keep[-1]]
        if np.linalg.norm(poses[j][:3, 3] - last[:3, 3]) >= dt or _rot_deg(poses[j], last) >= dr:
            keep.append(j)
    return keep


def _key(p: Path) -> int:
    return int(p.stem.replace("scan", "").replace("_global", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences-dir", type=Path, default=REPO_ROOT / "data" / "sequences")
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--volume", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd")
    ap.add_argument("--dt", type=float, default=6.0, help="min translation (mm) between kept frames")
    ap.add_argument("--dr", type=float, default=15.0, help="min rotation (deg) between kept frames")
    ap.add_argument("--min-cov", type=float, default=0.02, help="liver area share to tag a frame positive")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "sampled_frames.csv")
    args = ap.parse_args()

    vol_i = load_volume_data(str(args.volume))
    vol_l = load_volume_data(str(args.labels))
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    seqs = sorted(args.sequences_dir.glob("scan*.npz"), key=_key)
    rows = []
    print(f"{'seq':9s} {'kept':>5s} {'pos':>5s} {'neg':>5s}")
    tot_pos = tot_neg = 0
    for sp in seqs:
        lc2 = args.lc2_dir / f"{sp.stem}_global.npz"
        if not lc2.exists():
            continue
        poses = all_frame_poses(sp, lc2)
        imgs = np.load(sp, allow_pickle=True)["images"]
        kf = greedy_keyframes(poses, args.dt, args.dr)
        pos = neg = 0
        for j in kf:
            _, lab = sector_zoom_pair(vol_i, vol_l, affine, poses[j], imgs[j].shape[:2])
            cov = float((lab == LIVER).mean())
            is_pos = int(cov >= args.min_cov)
            pos += is_pos; neg += 1 - is_pos
            rows.append({"sequence": sp.name, "frame": int(j), "liver_cov": round(cov, 4), "is_positive": is_pos})
        tot_pos += pos; tot_neg += neg
        print(f"{sp.name:9s} {len(kf):5d} {pos:5d} {neg:5d}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sequence", "frame", "liver_cov", "is_positive"])
        w.writeheader(); w.writerows(rows)
    print(f"\nTOTAL: {len(rows)} distinct frames  ({tot_pos} positive, {tot_neg} negative)  "
          f"@ dt={args.dt}mm dr={args.dr}deg")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
