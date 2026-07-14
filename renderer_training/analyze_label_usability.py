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
from style.style import figure, save, C, TYPE, label_subplot           # noqa: E402

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

    # publication figure (PDF + PNG): two panels, usable classes highlighted
    import matplotlib; matplotlib.use("Agg")
    labels = [r["class"].replace("inferior ", "inf. ").replace("superior ", "sup. ")
              .replace(" lobe of", "").replace(" lung", " lung") for r in rows]
    volp = np.array([r["vol_pct"] for r in rows])
    frp = np.array([r["frames_gt2pct"] for r in rows])
    usable = [bool(r["usable"]) for r in rows]
    colors = [C["ours"] if u else C["neutral"] for u in usable]

    fig, (axA, axB) = figure(ncols=2, width="double", height=3.4)
    y = np.arange(len(rows))[::-1]                                    # largest at top

    axA.barh(y, np.maximum(volp, 5e-4), color=colors, edgecolor="white", linewidth=0.4)
    axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=TYPE["small"])
    axA.set_xscale("log"); axA.set_xlim(5e-4, 30)
    axA.set_xlabel("Share of CBCT volume (%)")
    axA.set_title("3D structure size")
    axA.grid(True, axis="x", which="major"); axA.grid(False, axis="y")
    for yi, r in zip(y, rows):
        axA.text(r["vol_pct"] * 1.4 if r["vol_pct"] > 5e-4 else 6e-4, yi,
                 f"{r['largest_cc_pct']:.0f}% CC", va="center", ha="left",
                 fontsize=TYPE["tiny"], color=C["dark"])

    axB.barh(y, frp, color=colors, edgecolor="white", linewidth=0.4)
    axB.set_yticks(y); axB.set_yticklabels([])
    axB.set_xlim(0, nfr * 1.02)
    axB.set_xlabel(f"US frames imaged (of {nfr})")
    axB.set_title("Imaged by the US sweeps")
    axB.grid(True, axis="x", which="major"); axB.grid(False, axis="y")
    for yi, v in zip(y, frp):
        if v > 0:
            axB.text(v + nfr * 0.015, yi, str(int(v)), va="center", ha="left",
                     fontsize=TYPE["tiny"], color=C["dark"])

    label_subplot(axA, "a", x=-0.55); label_subplot(axB, "b", x=-0.04)
    # legend proxies
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=C["ours"], label="usable target"),
                        Patch(facecolor=C["neutral"], label="rejected")],
               loc="lower right", fontsize=TYPE["small"])
    save(fig, str(OUT / "class_usability"))
    print(f"wrote {OUT}/class_usability.pdf + .png and class_usability.csv")
    for r in rows[:4]:
        print(f"  {r['class']:28s} vol={r['vol_pct']:.2f}%  CC={r['largest_cc_pct']:.0f}%  frames={r['frames_gt2pct']}/{nfr}")


if __name__ == "__main__":
    main()
