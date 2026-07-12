#!/usr/bin/env python
"""Project the CBCT segmentation (labels.nrrd) onto each US frame's imaging plane.

Rationale
---------
Every LC2-registered US frame has a refined probe pose in CBCT space. The intensity CBCT
sector at that pose is *known* to align with the real US (that alignment is the whole premise
of the renderer pairs). So resampling ``labels.nrrd`` on the **same** plane/geometry — but with
nearest-neighbour sampling so class ids are never blended — yields a per-frame segmentation
mask that is pixel-aligned with the US, for free. No manual US labelling.

This reuses ``plot_script.plots_reslice.compare.cbct_sector_zoom``'s geometry verbatim: the
apex/sector/crop/zoom are all computed from the *intensity* sector, then applied identically to
the label sector (``order=0``). A self-check reproduces ``pairs.npz['cbct']`` (corr ~1.0) to prove
the label plane matches the stored, alignment-validated intensity plane.

    module load python/3.12-base
    python renderer_training/project_labels_to_us.py --sequence scan1 --n 6

Outputs (under --out): per-frame overlay figure + masks.npz (label id per pixel, US-aligned).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reslice import pose as P                                              # noqa: E402
from reslice import sector as sec                                         # noqa: E402
from reslice.frame import affine_from_sitk                               # noqa: E402
from reslice.io import load_volume_data                                  # noqa: E402
from reslice.sampling import reslice_rectangular_plane                   # noqa: E402
from plot_script.plots_reslice.compare import (                          # noqa: E402
    FAN, SLICE_W, SLICE_H, AXIAL_SIGN, LATERAL_SIGN,
)

DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_LABELS = REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd"
DEFAULT_PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_OUT = REPO_ROOT / "figures" / "9_label_projection_check"


def sector_zoom_pair(vol_int, vol_lab, affine, pose, target_shape):
    """Resliced (intensity, label) sectors sharing one geometry.

    The intensity path is byte-for-byte ``cbct_sector_zoom``; the label path samples the
    same plane with ``order=0`` and inherits the identical apex/sector/crop/zoom.
    """
    plane = P.plane_from_probe_pose(pose, "probe-xz", 0.0)
    rect_i, valid = reslice_rectangular_plane(
        vol_int, affine, plane, width_mm=SLICE_W, height_mm=SLICE_H,
        n_rows=target_shape[0], n_cols=target_shape[1],
        axial_sign=AXIAL_SIGN, lateral_sign=LATERAL_SIGN, order=1)
    rect_l, _ = reslice_rectangular_plane(
        vol_lab, affine, plane, width_mm=SLICE_W, height_mm=SLICE_H,
        n_rows=target_shape[0], n_cols=target_shape[1],
        axial_sign=AXIAL_SIGN, lateral_sign=LATERAL_SIGN, order=0)
    rect_i, valid = np.rot90(rect_i, 2), np.rot90(valid, 2)
    rect_l = np.rot90(rect_l, 2)

    _, thr, _ = sec.detect_content_top_row(rect_i, valid, threshold=None, min_pixels=8)
    probe_px = sec.project_point_to_display_pixel(
        pose[:3, 3], plane, width_mm=SLICE_W, height_mm=SLICE_H,
        rows=rect_i.shape[0], cols=rect_i.shape[1], axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN, display_rot180=True)
    depth_dir = sec.project_direction_to_display_rc(
        pose[:3, 2], plane, axial_sign=AXIAL_SIGN, lateral_sign=LATERAL_SIGN,
        display_rot180=True)
    apex, _ = sec.apex_from_pose_and_edge(
        rect_i, valid, threshold=thr, probe_pixel_rc=probe_px,
        depth_direction_rc=depth_dir, max_line_distance_px=5.0)
    sector_i, mask, dbg = sec.apply_sector(
        rect_i, valid, top_margin_rows=2, apex_col_fraction=0.5,
        apex_pixel_rc=apex, depth_direction_rc=depth_dir,
        content_threshold=None, content_min_pixels=8,
        width_mm=SLICE_W, height_mm=SLICE_H, **FAN)
    crop_mask = sec.sector_mask_in_display_image(
        rect_i.shape, dbg["apex_row"], dbg["apex_col"], np.asarray(dbg["depth_direction_rc"]),
        dbg["fov_deg"], dbg["depth_mm"], 0.0, dbg["mm_per_row"], dbg["mm_per_col"])

    int_zoom, _ = sec.crop_and_zoom_sector(
        sec.normalize_image(sector_i), mask, crop_mask, tuple(target_shape), margin_px=18, order=1)
    sector_l = np.where(mask, rect_l, 0.0)
    lab_zoom, _ = sec.crop_and_zoom_sector(
        sector_l, mask, crop_mask, tuple(target_shape), margin_px=18, order=0)
    return int_zoom, np.rint(lab_zoom).astype(np.int16)


def _load_names(path: Path) -> dict[int, str]:
    names = {0: "background"}
    if not path.exists():
        return names
    for line in path.read_text().splitlines()[1:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="scan1", help="which sequence's frames to project")
    ap.add_argument("--n", type=int, default=6, help="how many frames to visualise")
    ap.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--colortable", type=Path,
                    default=REPO_ROOT / "data" / "cbct_20260612" / "labels_colortable.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    names = _load_names(args.colortable)
    pairs = np.load(args.pairs, allow_pickle=True)
    seq = np.asarray(pairs["sequence"]).astype(str)
    want = args.sequence if args.sequence.endswith(".npz") else args.sequence + ".npz"
    idx = np.where(seq == want)[0]
    if len(idx) == 0:
        raise SystemExit(f"no frames for {want}; sequences present: {sorted(set(seq))}")
    sel = idx[np.linspace(0, len(idx) - 1, min(args.n, len(idx)), dtype=int)]

    vol_int = load_volume_data(str(args.volume))
    vol_lab = load_volume_data(str(args.labels))
    import SimpleITK as sitk
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    us_all = pairs["us"]; cbct_all = pairs["cbct"]; poses = pairs["refined_poses"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(sel), 3, figsize=(11, 3.4 * len(sel)))
    axes = np.atleast_2d(axes)
    cover_accum: dict[int, int] = {}
    masks_out = []

    for row, i in enumerate(sel):
        us = np.asarray(us_all[i], dtype=float)
        int_zoom, lab = sector_zoom_pair(vol_int, vol_lab, affine, poses[i], us.shape[:2])
        masks_out.append(lab)

        # self-check: our intensity plane must reproduce the stored, alignment-validated cbct
        a = cbct_all[i].astype(float)
        def _n(x):
            lo, hi = np.nanpercentile(x, [1, 99]); return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        corr = float(np.corrcoef(_n(a).ravel(), _n(int_zoom).ravel())[0, 1])

        present = [(c, int((lab == c).sum())) for c in np.unique(lab) if c != 0]
        for c, n in present:
            cover_accum[c] = cover_accum.get(c, 0) + n
        tag = ", ".join(f"{names.get(c, c)}:{100*n/lab.size:.1f}%" for c, n in present) or "(none)"

        usn = _n(us)
        axes[row, 0].imshow(usn, cmap="gray"); axes[row, 0].set_title(f"real US  ({want} f{int(pairs['frame_index'][i])})", fontsize=8)
        axes[row, 1].imshow(usn, cmap="gray")
        # colour overlay of labels on the US
        over = np.zeros((*lab.shape, 4))
        for c in np.unique(lab):
            if c == 0:
                continue
            col = cm.tab20((c % 20) / 20.0)
            over[lab == c] = (*col[:3], 0.45)
        axes[row, 1].imshow(over)
        axes[row, 1].set_title(f"US + projected labels\ncorr(int,stored)={corr:.3f}", fontsize=8)
        axes[row, 2].imshow(lab, cmap="tab20", vmin=0, vmax=20); axes[row, 2].set_title(f"label map\n{tag}", fontsize=7)
        for a_ in axes[row]:
            a_.axis("off")

    fig.tight_layout()
    fig_path = args.out / f"label_projection_{args.sequence}.png"
    fig.savefig(fig_path, dpi=110)
    np.savez_compressed(args.out / f"masks_{args.sequence}.npz",
                        masks=np.stack(masks_out), frame_index=pairs["frame_index"][sel])

    total = np.stack(masks_out).size
    print(f"\nwrote {fig_path}")
    print(f"frames projected: {len(sel)}  (sequence {want})")
    print("class coverage across projected frames (share of all pixels):")
    for c, n in sorted(cover_accum.items(), key=lambda kv: -kv[1]):
        print(f"  {c:2d} {names.get(c, str(c)):28s} {100*n/total:5.2f}%")


if __name__ == "__main__":
    main()
