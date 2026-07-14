#!/usr/bin/env python
"""Build the generated-US liver dataset (Group B augmentation) from a rendered_us.npz.

Same A1 pipeline as the real dataset: project the CBCT liver label onto each generated
frame's render pose (poses_cbct_mm), confidence-crop the shadow tail, gate frames where
nothing survives. Writes data/liver_seg/gen/{images.npy,masks.npy,meta.csv} in the SAME
format as train/ and test/ so train_sam2_head.py can concatenate it.

Note: generated US from the CUT renderer tends to lack realistic deep-field attenuation,
so confidence cropping/gating removes far less than on real US -- a known quality gap.

    module load python/3.12-base
    python renderer_training/build_generated_masks.py
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
from renderer_training.confidence_map import confidence_map            # noqa: E402

LIVER = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", type=Path,
                    default=REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--volume", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd")
    ap.add_argument("--min-cov", type=float, default=0.02)
    ap.add_argument("--crop-floor", type=float, default=0.05)
    ap.add_argument("--min-kept-cov", type=float, default=0.01)
    ap.add_argument("--tag", default="gen_tilt", help="sequence tag written to meta.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "liver_seg" / "gen")
    args = ap.parse_args()

    d = np.load(args.render, allow_pickle=True)
    gen = d["generated_us"]                                            # (N,660,880) float [0,1]
    poses = d["poses_cbct_mm"]
    vol_i = load_volume_data(str(args.volume))
    vol_l = load_volume_data(str(args.labels))
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    imgs, masks, meta = [], [], []
    npos = nneg = ngate = 0
    for i in range(len(poses)):
        _, lab = sector_zoom_pair(vol_i, vol_l, affine, poses[i], gen[i].shape[:2])
        liver = (lab == LIVER)
        cov = float(liver.mean())
        img_u8 = np.clip(gen[i] * 255.0, 0, 255).astype(np.uint8)
        if cov < args.min_cov:
            mask = np.zeros(liver.shape, np.uint8); dec = "negative"; is_pos = 0; conf_in = -1.0; nneg += 1
        else:
            conf = confidence_map(gen[i]); conf_in = float(conf[liver].mean())
            cropped = liver & (conf >= args.crop_floor)
            if float(cropped.mean()) < args.min_kept_cov:
                ngate += 1; continue
            mask = cropped.astype(np.uint8); dec = "positive"; is_pos = 1; npos += 1
        imgs.append(img_u8); masks.append(mask)
        meta.append({"sequence": args.tag, "frame": int(i), "is_positive": is_pos,
                     "liver_cov": round(cov, 4), "conf_in_liver": round(conf_in, 4), "decision": dec})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(poses)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "images.npy", np.stack(imgs).astype(np.uint8))
    np.save(args.out / "masks.npy", np.stack(masks).astype(np.uint8))
    with (args.out / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
    print(f"\ngenerated set: {npos} positive, {nneg} negative, {ngate} gated  -> {args.out}")


if __name__ == "__main__":
    main()
