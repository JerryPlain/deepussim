#!/usr/bin/env python
"""Measure the real liver pool over ALL frames of every sequence (not just the 150 LC2 subset).

Reslices the CBCT liver label onto every frame (poses from allframe_poses) and records the
liver area share -- no confidence step, so this is a fast upper bound on the positive pool.

    module load python/3.12-base
    python renderer_training/scan_liver_coverage_allframes.py

Outputs data/liver_seg/allframe_coverage.csv (sequence, frame, liver_cov) and a per-sequence
+ total summary at several coverage thresholds.
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
from renderer_training.allframe_poses import all_frame_poses, reprojection_error  # noqa: E402

LIVER = 2


def _key(p: Path) -> int:
    return int(p.stem.replace("scan", "").replace("_global", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences-dir", type=Path, default=REPO_ROOT / "data" / "sequences")
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--volume", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd")
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth frame (1 = all frames)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "allframe_coverage.csv")
    args = ap.parse_args()

    vol_i = load_volume_data(str(args.volume))
    vol_l = load_volume_data(str(args.labels))
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    seqs = sorted(args.sequences_dir.glob("scan*.npz"), key=_key)
    rows = []
    per_seq = {}
    for sp in seqs:
        lc2 = args.lc2_dir / f"{sp.stem}_global.npz"
        if not lc2.exists():
            print(f"skip {sp.name}: no {lc2.name}"); continue
        et, er = reprojection_error(sp, lc2)
        poses = all_frame_poses(sp, lc2)
        imgs = np.load(sp, allow_pickle=True)["images"]
        covs = []
        for j in range(0, len(poses), args.stride):
            _, lab = sector_zoom_pair(vol_i, vol_l, affine, poses[j], imgs[j].shape[:2])
            cov = float((lab == LIVER).mean())
            covs.append(cov)
            rows.append({"sequence": sp.name, "frame": j, "liver_cov": round(cov, 4)})
        covs = np.array(covs)
        per_seq[sp.name] = covs
        print(f"{sp.name}: frames={len(covs)} (reproj {et:.1e}mm/{er:.1e}deg)  "
              f"liver>2%={int((covs>=0.02).sum())}  >5%={int((covs>=0.05).sum())}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sequence", "frame", "liver_cov"]); w.writeheader(); w.writerows(rows)

    allc = np.concatenate(list(per_seq.values())) if per_seq else np.array([])
    print(f"\n=== TOTAL real pool (stride={args.stride}) ===")
    print(f"  frames scanned : {len(allc)}")
    for t in (0.02, 0.05, 0.10):
        n = int((allc >= t).sum())
        print(f"  liver >= {t*100:.0f}% : {n:5d} frames  ({100*n/max(len(allc),1):.1f}%)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
