#!/usr/bin/env python
"""Build cleaned binary liver masks for every LC2-registered US frame.

Pipeline per frame (A1: pixel-crop first, gate only as a fallback):
  1. project the CBCT liver label onto the US plane (project_labels_to_us.sector_zoom_pair).
  2. relative crop: drop mask pixels with confidence < --crop-floor (the deep-field shadow tail),
     so a frame keeps its VISIBLE liver even if part of the projected label sits in shadow.
  3. classify the frame:
       - liver coverage < --min-cov              -> NEGATIVE (kept, empty mask; a background frame)
       - cropped coverage >= --min-kept-cov      -> POSITIVE (kept, cropped mask)
       - cropped coverage < --min-kept-cov       -> GATED   (dropped: ~all liver was in shadow)

Confidence is the Karamalis random-walks map (confidence_map.py); it is only computed for
liver-containing frames (negatives skip it). Outputs cleaned masks + an auditable per-frame CSV.

    module load python/3.12-base
    python renderer_training/build_liver_masks.py

Outputs (under --out):
    liver_masks.npz   masks (N,H,W) uint8, sequence, frame_index, is_positive  (GATED frames excluded)
    liver_decisions.csv   per-frame: coverage, conf, decision, crop amount
    liver_gating_qc.png   a few kept vs gated frames for eyeballing the threshold
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
from renderer_training.confidence_map import confidence_map, _norm01    # noqa: E402

LIVER = 2
DEFAULT_OUT = REPO_ROOT / "data" / "liver_seg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz")
    ap.add_argument("--volume", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd")
    ap.add_argument("--min-cov", type=float, default=0.02, help="min projected-liver area share to count as a candidate")
    ap.add_argument("--crop-floor", type=float, default=0.05, help="drop mask pixels below this confidence (shadow tail)")
    ap.add_argument("--min-kept-cov", type=float, default=0.01, help="min VISIBLE (cropped) liver area share to keep a positive")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pairs = np.load(args.pairs, allow_pickle=True)
    seq = np.asarray(pairs["sequence"]).astype(str)
    fi = np.asarray(pairs["frame_index"]).astype(int)
    us = pairs["us"]; poses = pairs["refined_poses"]

    vol_i = load_volume_data(str(args.volume))
    vol_l = load_volume_data(str(args.labels))
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    rows = []
    masks = []; keep_seq = []; keep_fi = []; keep_pos = []
    gated_examples = []; kept_examples = []
    N = len(seq)
    for i in range(N):
        _, lab = sector_zoom_pair(vol_i, vol_l, affine, poses[i], us[i].shape[:2])
        liver = (lab == LIVER)
        cov = float(liver.mean())
        rec = {"sequence": seq[i], "frame_index": int(fi[i]), "liver_cov": round(cov, 4),
               "conf_in_liver": "", "kept_frac": "", "cov_kept": "", "decision": ""}

        if cov < args.min_cov:
            rec["decision"] = "negative"
            masks.append(np.zeros(liver.shape, np.uint8)); keep_seq.append(seq[i]); keep_fi.append(int(fi[i])); keep_pos.append(0)
        else:
            conf = confidence_map(us[i])
            conf_in = float(conf[liver].mean())
            cropped = liver & (conf >= args.crop_floor)          # A1: crop the shadow tail first
            cov_kept = float(cropped.mean())
            kf = float(cropped.sum() / max(liver.sum(), 1))
            rec["conf_in_liver"] = round(conf_in, 4); rec["kept_frac"] = round(kf, 3); rec["cov_kept"] = round(cov_kept, 4)
            if cov_kept < args.min_kept_cov:                     # almost nothing visible survived -> gate
                rec["decision"] = "gated"
                if len(gated_examples) < 4:
                    gated_examples.append((i, liver, conf, conf_in))
            else:
                rec["decision"] = "positive"
                masks.append(cropped.astype(np.uint8)); keep_seq.append(seq[i]); keep_fi.append(int(fi[i])); keep_pos.append(1)
                if len(kept_examples) < 4:
                    kept_examples.append((i, liver, cropped, conf, conf_in))
        rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N} processed", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "liver_masks.npz",
                        masks=np.stack(masks).astype(np.uint8),
                        sequence=np.array(keep_seq), frame_index=np.array(keep_fi, np.int32),
                        is_positive=np.array(keep_pos, np.int8))
    with (args.out / "liver_decisions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    dec = [r["decision"] for r in rows]
    n_pos = dec.count("positive"); n_neg = dec.count("negative"); n_gate = dec.count("gated")
    print(f"\n=== summary (A1: min_cov={args.min_cov}, crop_floor={args.crop_floor}, min_kept_cov={args.min_kept_cov}) ===")
    print(f"  total frames    : {N}")
    print(f"  POSITIVE (kept) : {n_pos}")
    print(f"  NEGATIVE (kept) : {n_neg}   (background, empty mask)")
    print(f"  GATED   (dropped): {n_gate}   (~all liver in shadow after crop)")
    kept_fracs = [r["kept_frac"] for r in rows if r["decision"] == "positive" and r["kept_frac"] != ""]
    if kept_fracs:
        print(f"  crop kept {100*np.median(kept_fracs):.0f}% of the raw mask (median) on positive frames")

    # min-kept-cov sensitivity: how many positives survive at each visible-area threshold
    cand = [r for r in rows if r["cov_kept"] != ""]
    covk = np.array([r["cov_kept"] for r in cand], float)
    print("\n  min-kept-cov sensitivity (positives surviving each visible-area threshold):")
    for t in (0.005, 0.01, 0.02, 0.03):
        print(f"    min_kept_cov={t:.3f} -> {(covk >= t).sum():3d} positives")

    # QC figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ncol = 4
    fig, ax = plt.subplots(2, ncol, figsize=(4 * ncol, 8))
    for c in range(ncol):
        if c < len(kept_examples):
            i, liver, cropped, conf, cin = kept_examples[c]
            ax[0, c].imshow(_norm01(us[i]), cmap="gray")
            ax[0, c].contour(liver.astype(float), [0.5], colors="red", linewidths=0.8)
            ax[0, c].contour(cropped.astype(float), [0.5], colors="lime", linewidths=1.0)
            ax[0, c].set_title(f"KEPT {seq[i]} f{int(fi[i])}\nconf={cin:.2f} red=raw green=cropped", fontsize=8)
        ax[0, c].axis("off")
        if c < len(gated_examples):
            i, liver, conf, cin = gated_examples[c]
            ax[1, c].imshow(_norm01(us[i]), cmap="gray")
            ax[1, c].contour(liver.astype(float), [0.5], colors="red", linewidths=0.9)
            ax[1, c].set_title(f"GATED {seq[i]} f{int(fi[i])}\nconf={cin:.2f} (liver in shadow)", fontsize=8)
        ax[1, c].axis("off")
    fig.tight_layout()
    fig_dir = REPO_ROOT / "figures" / "9_label_projection_check"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "liver_gating_qc.png", dpi=115)

    print(f"\nwrote {args.out / 'liver_masks.npz'}  ({len(masks)} frames: {n_pos} pos + {n_neg} neg)")
    print(f"wrote {args.out / 'liver_decisions.csv'}")
    print(f"wrote {fig_dir / 'liver_gating_qc.png'}")


if __name__ == "__main__":
    main()
