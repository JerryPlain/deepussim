#!/usr/bin/env python
"""Project the CBCT segmentation (labels.nrrd) onto each US frame's imaging plane.

Rationale
---------
Every LC2-registered US frame has a refined probe pose in CBCT space. Sampling the CBCT
directly onto the **real US display fan** (``reslice.us_display.render_on_us_grid``, fitted by
``calib/fit_us_fan.py``) puts the intensity pixel-aligned with the US. Sampling ``labels.nrrd``
on that *same* grid with nearest-neighbour (``order=0``, so class ids are never blended) yields
a per-frame segmentation mask aligned with the US for free. No manual US labelling.

Both channels go through the identical geometry, so intensity and labels are co-registered by
construction. Alignment to the *US itself* comes from the LC2-refined pose plus the fitted fan,
and is asserted in ``tests/test_display_alignment.py`` (support IoU vs the US fan) -- NOT by the
old ``corr(reslice, pairs['cbct'])`` self-check, which was a tautology (it only proved the
reslice reproduced the stored sector, never that the sector matched the US).

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

from reslice import sector as sec                                         # noqa: E402
from reslice.frame import affine_from_sitk                               # noqa: E402
from reslice.io import load_volume_data                                  # noqa: E402
from reslice.us_display import UsDisplayFan, render_on_us_grid           # noqa: E402

DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_LABELS = REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd"
DEFAULT_PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_OUT = REPO_ROOT / "figures" / "9_label_projection_check"


def sector_zoom_pair(vol_int, vol_lab, affine, pose, target_shape, fan=None):
    """Resliced (intensity, label) images sharing one geometry, on the real US display grid.

    Both volumes are sampled directly onto the fitted US fan (``render_on_us_grid``): the
    intensity with ``order=1`` (trilinear) and the labels with ``order=0`` (nearest, so class
    ids are never blended). Pixel-aligned to the US by construction -- no crop/resize, no apex
    detection. ``target_shape`` is kept for backward compatibility and only used to resize if
    it differs from the fan shape (it never does in the pipeline: it is the real frame size).
    """
    fan = fan or _get_fan()
    int_img, _ = render_on_us_grid(vol_int, affine, pose, fan, order=1)
    lab_img, _ = render_on_us_grid(vol_lab, affine, pose, fan, order=0, fill=0.0)
    int_img = sec.normalize_image(int_img)
    lab = np.rint(lab_img).astype(np.int16)

    if tuple(target_shape) != int_img.shape:
        from scipy.ndimage import zoom as _zoom

        zr = (target_shape[0] / int_img.shape[0], target_shape[1] / int_img.shape[1])
        int_img = _zoom(int_img, zr, order=1)
        lab = np.rint(_zoom(lab.astype(np.float32), zr, order=0)).astype(np.int16)
    return int_img.astype(np.float32), lab


_FAN: UsDisplayFan | None = None


def _get_fan() -> UsDisplayFan:
    global _FAN
    if _FAN is None:
        _FAN = UsDisplayFan.load()
    return _FAN


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

    us_all = pairs["us"]; poses = pairs["refined_poses"]

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

        # QC: correlation between the rendered CBCT intensity and the REAL US, over the fan
        # support. This is the honest alignment signal (modest, CBCT<->US modality gap); the old
        # corr-against-stored-cbct check was a tautology -- see the module docstring.
        def _n(x):
            lo, hi = np.nanpercentile(x, [1, 99]); return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        sup = _get_fan().support_mask(int_zoom.shape)
        corr = float(np.corrcoef(_n(int_zoom)[sup].ravel(), _n(us)[sup].ravel())[0, 1])

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
        axes[row, 1].set_title(f"US + projected labels\ncorr(cbct,us)={corr:.3f}", fontsize=8)
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
