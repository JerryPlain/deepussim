#!/usr/bin/env python
"""Project the CBCT segmentation (labels.nrrd) onto each US frame's imaging plane.

Rationale
---------
Every LC2-registered US frame has a refined probe pose in CBCT space. Intensity and labels
are resliced on the same fitted convex-probe fan, then scan-converted into the fixed real
B-mode pixel geometry. Labels use nearest-neighbour sampling throughout so class ids are
never blended.

The display mapping deliberately does not detect a per-frame CBCT surface or resize a fan
bounding box. Probe intrinsics are fixed: virtual apex, face/far radii, FOV and pixel scale
were fitted once from the combined real scan1/8/15 contact envelope.

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

from reslice.frame import affine_from_sitk                               # noqa: E402
from reslice.calibration import (                                        # noqa: E402
    US_DISPLAY_FAN,
    US_DISPLAY_SHAPE,
    US_PROBE_GEOMETRY,
    US_SPACING_MM,
)
from reslice.fan import ProbeGeometry, reslice_fan, scan_convert_fan      # noqa: E402
from reslice.io import load_volume_data                                  # noqa: E402
from reslice.sector import normalize_image                               # noqa: E402

DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_LABELS = REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd"
DEFAULT_PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_OUT = REPO_ROOT / "figures" / "9_label_projection_check"

# Fixed B-mode display geometry fitted from Jerry's combined scan1/8/15 contact envelope.
# Pixel coordinates are (x, y); physical values use Feng's 0.166112957 mm/px calibration.
CALIBRATED_DISPLAY_SHAPE = US_DISPLAY_SHAPE
DEFAULT_US_SPACING_MM = US_SPACING_MM
DEFAULT_DISPLAY_FAN = US_DISPLAY_FAN
DEFAULT_PROBE_GEOMETRY = US_PROBE_GEOMETRY


def sector_zoom_pair(
    vol_int,
    vol_lab,
    affine,
    pose,
    target_shape,
    *,
    display_fan: dict | None = None,
    probe_geometry: ProbeGeometry | None = None,
):
    """Return intensity and labels in the fixed real B-mode pixel coordinate system.

    The historical function name is retained for downstream callers, but no crop/zoom is
    performed. Both volumes are sampled on one polar probe grid and scan-converted with one
    fixed display calibration. Labels use nearest-neighbour sampling in both operations.
    """
    target_shape = tuple(int(x) for x in target_shape)
    if target_shape != CALIBRATED_DISPLAY_SHAPE:
        raise ValueError(
            f"US display calibration is for {CALIBRATED_DISPLAY_SHAPE}, got {target_shape}; "
            "fit a display fan for the new scanner resolution instead of resizing it"
        )

    fan = DEFAULT_DISPLAY_FAN if display_fan is None else display_fan
    geom = DEFAULT_PROBE_GEOMETRY if probe_geometry is None else probe_geometry
    polar_i = reslice_fan(vol_int, affine, pose, geom, order=1)
    polar_l = reslice_fan(vol_lab, affine, pose, geom, order=0)

    int_display = scan_convert_fan(
        normalize_image(polar_i), target_shape, **fan, order=1, cval=0.0
    )
    lab_display = scan_convert_fan(
        polar_l, target_shape, **fan, order=0, cval=0.0
    )
    return int_display, np.rint(lab_display).astype(np.int16)


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

    us_all = pairs["us"]
    poses = pairs["refined_poses"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(sel), 3, figsize=(11, 3.4 * len(sel)))
    axes = np.atleast_2d(axes)
    cover_accum: dict[int, int] = {}
    masks_out = []

    def _n(x):
        lo, hi = np.nanpercentile(x, [1, 99])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    for row, i in enumerate(sel):
        us = np.asarray(us_all[i], dtype=float)
        _, lab = sector_zoom_pair(vol_int, vol_lab, affine, poses[i], us.shape[:2])
        masks_out.append(lab)

        present = [(c, int((lab == c).sum())) for c in np.unique(lab) if c != 0]
        for c, n in present:
            cover_accum[c] = cover_accum.get(c, 0) + n
        tag = ", ".join(f"{names.get(c, c)}:{100*n/lab.size:.1f}%" for c, n in present) or "(none)"

        usn = _n(us)
        axes[row, 0].imshow(usn, cmap="gray")
        axes[row, 0].set_title(
            f"real US  ({want} f{int(pairs['frame_index'][i])})", fontsize=8
        )
        axes[row, 1].imshow(usn, cmap="gray")
        # colour overlay of labels on the US
        over = np.zeros((*lab.shape, 4))
        for c in np.unique(lab):
            if c == 0:
                continue
            col = cm.tab20((c % 20) / 20.0)
            over[lab == c] = (*col[:3], 0.45)
        axes[row, 1].imshow(over)
        axes[row, 1].set_title(
            f"US + projected labels\nfixed fan, {DEFAULT_US_SPACING_MM:.6f} mm/px",
            fontsize=8,
        )
        axes[row, 2].imshow(lab, cmap="tab20", vmin=0, vmax=20)
        axes[row, 2].set_title(f"label map\n{tag}", fontsize=7)
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
