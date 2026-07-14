#!/usr/bin/env python
"""Evidence that only liver (and heart) are usable US-segmentation targets on this phantom.

Two independent checks, saved as a figure + CSV:
  (A) 3D label volume: per-class voxel count and largest-connected-component fraction
      -> which TotalSegmentator classes are real coherent structures vs scattered noise.
  (B) US-plane coverage: project each class onto the 150 LC2-registered US frames, count
      frames where the class covers >2% -> which classes the US sweeps actually image.

A class is a viable GT only if it is (A) a real coherent structure AND (B) actually imaged.

    module load python/3.12-base
    python renderer_training/analyze_label_usability.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SimpleITK as sitk                                                  # noqa: E402
from reslice.frame import affine_from_sitk                              # noqa: E402
from reslice.io import load_volume_data                                 # noqa: E402
from renderer_training.project_labels_to_us import sector_zoom_pair     # noqa: E402

OUT = REPO_ROOT / "figures" / "10_label_class_usability"


def _names(path: Path) -> dict[int, str]:
    n = {}
    for r in csv.reader(path.open()):
        if r and r[0].isdigit():
            n[int(r[0])] = r[1]
    return n


def main() -> None:
    cbct = REPO_ROOT / "data" / "cbct_20260612"
    names = _names(cbct / "labels_colortable.csv")
    lab = load_volume_data(str(cbct / "labels.nrrd")).astype(int)
    tot_vox = lab.size

    # (A) 3D structure
    vox, cc_frac = {}, {}
    for c in range(1, 18):
        m = (lab == c); n = int(m.sum()); vox[c] = n
        if n:
            l, _ = ndimage.label(m); sizes = np.bincount(l.ravel())[1:]
            cc_frac[c] = float(sizes.max()) / n
        else:
            cc_frac[c] = 0.0

    # (B) US-plane coverage over the 150 registered frames
    pairs = np.load(cbct.parent / "renderer_lc2_pairs" / "pairs.npz", allow_pickle=True)
    us = pairs["us"]; poses = pairs["refined_poses"]
    vol_i = load_volume_data(str(cbct / "CBCT.mhd"))
    affine = affine_from_sitk(sitk.ReadImage(str(cbct / "CBCT.mhd")))
    frames_2pct = {c: 0 for c in range(1, 18)}
    for i in range(len(poses)):
        _, seg = sector_zoom_pair(vol_i, lab, affine, poses[i], us[i].shape[:2])
        for c in range(1, 18):
            if (seg == c).mean() > 0.02:
                frames_2pct[c] += 1
    nfr = len(poses)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in range(1, 18):
        rows.append({"id": c, "class": names.get(c, str(c)), "voxels": vox[c],
                     "vol_pct": round(100 * vox[c] / tot_vox, 3),
                     "largest_cc_pct": round(100 * cc_frac[c], 1),
                     "frames_gt2pct": frames_2pct[c], "n_frames": nfr,
                     "usable": int(vox[c] / tot_vox > 0.02 and frames_2pct[c] >= 10)})
    rows.sort(key=lambda r: -r["voxels"])
    with (OUT / "class_usability.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # figure: two bars
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = [r["class"] for r in rows]
    volp = [r["vol_pct"] for r in rows]
    frp = [r["frames_gt2pct"] for r in rows]
    colors = ["#2ca02c" if r["usable"] else "#c0c0c0" for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    y = np.arange(len(rows))
    ax[0].barh(y, volp, color=colors); ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xscale("log"); ax[0].set_xlabel("share of CBCT volume (%, log)")
    ax[0].set_title("(A) 3D structure size — only liver/heart are large")
    for i, r in enumerate(rows):
        ax[0].text(max(volp[i], 1e-3), i, f"  {r['largest_cc_pct']:.0f}% CC", va="center", fontsize=6)
    ax[1].barh(y, frp, color=colors); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel(f"# US frames with >2% coverage (of {nfr})")
    ax[1].set_title("(B) actually imaged by the US sweeps")
    fig.suptitle("Only liver (and heart) are viable US-segmentation targets  (green = usable)")
    fig.tight_layout(); fig.savefig(OUT / "class_usability.png", dpi=120)
    print(f"wrote {OUT}/class_usability.png and class_usability.csv")
    for r in rows[:4]:
        print(f"  {r['class']:28s} vol={r['vol_pct']:.2f}%  CC={r['largest_cc_pct']:.0f}%  frames={r['frames_gt2pct']}/{nfr}")


if __name__ == "__main__":
    main()
