#!/usr/bin/env python
"""Compare final crop windows without changing pose or CBCT reslice geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_script.plots_reslice.compare import (  # noqa: E402
    _normalize_display,
    cbct_sector_zoom,
)
from reslice import pose as P  # noqa: E402
from reslice.io import load_transform_4x4, load_volume_data  # noqa: E402


# All-15-bag real-US fan fit, used only as a red display-space reference outline.
REAL_US_FAN = {
    "apex_px": (439.4999999999998, -261.90505706969174),
    "r0_px": 307.90505706969174,
    "r1_px": 918.9050570696918,
    "fov_deg": 69.31467507611598,
}


def _real_fan_support(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    apex_x, apex_y = REAL_US_FAN["apex_px"]
    dx, dy = xx - apex_x, yy - apex_y
    radius = np.hypot(dx, dy)
    angle = np.abs(np.arctan2(dx, dy))
    return (
        (radius >= REAL_US_FAN["r0_px"])
        & (radius <= REAL_US_FAN["r1_px"])
        & (angle <= np.deg2rad(REAL_US_FAN["fov_deg"]) / 2.0)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--placement", type=Path, required=True)
    ap.add_argument("--margins", nargs="+", type=int, default=[18, 12, 8, 4, 0])
    ap.add_argument("--crop-nears", nargs="+", type=float)
    ap.add_argument("--fixed-margin", type=int, default=0)
    ap.add_argument("--col-margins", nargs="+", type=int)
    ap.add_argument("--fixed-crop-near", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    affine = np.asarray(
        report["phantom_centered_frame"]["recon_affine_centered_from_ijk_mm"], dtype=float
    )
    volume = load_volume_data(args.volume)
    world_from_phantom = load_transform_4x4(args.placement)
    sequence = np.load(args.sequence, allow_pickle=True)
    us = np.asarray(sequence["images"][args.frame])
    world_from_ee = np.asarray(sequence["poses"][args.frame], dtype=float)
    world_from_probe = world_from_ee @ P.T_EE_FROM_PROBE
    phantom_from_probe_mm = P.probe_pose_in_phantom_centered_mm(
        world_from_probe, world_from_phantom
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(None, None, None, _normalize_display(us))]
    if args.col_margins is not None:
        variants = [(args.fixed_margin, args.fixed_crop_near, col) for col in args.col_margins]
    elif args.crop_nears is not None:
        variants = [(args.fixed_margin, near, None) for near in args.crop_nears]
    else:
        variants = [(margin, 0.0, None) for margin in args.margins]
    for margin, crop_near, col_margin in variants:
        panels.append(
            (
                margin,
                crop_near,
                col_margin,
                cbct_sector_zoom(
                    volume,
                    affine,
                    phantom_from_probe_mm,
                    us.shape[:2],
                    crop_margin_px=margin,
                    crop_near_mm=crop_near,
                    crop_margin_cols_px=col_margin,
                ),
            )
        )

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 4.6 * nrows), squeeze=False)
    support = _real_fan_support(us.shape[:2])
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (margin, crop_near, col_margin, image) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        if margin is None:
            ax.set_title(f"real US · {args.sequence.stem} frame {args.frame}")
        else:
            ax.contour(support.astype(float), [0.5], colors="red", linewidths=1.4)
            col_text = "same" if col_margin is None else str(col_margin)
            ax.set_title(
                f"CBCT crop · row={margin}px · col={col_text}px · near={crop_near:g}mm"
            )
        ax.axis("off")
    fig.suptitle(
        "Only the final crop window changes · red = fitted real-US fan boundary",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
