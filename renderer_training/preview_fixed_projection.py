#!/usr/bin/env python
"""Generate a real-US / fixed-geometry projected-mask preview.

This utility is intentionally independent of a prebuilt ``pairs.npz``. It uses the fixed
scan1/8/15 probe calibration, estimates one shared LC2 correction from contact frames, and
projects CBCT intensity and labels into selected raw US frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lc2.forward import Context, FrameTargets                             # noqa: E402
from lc2.register import register_global                                 # noqa: E402
from lc2.us_fan import FanFit, unwrap_fan                                # noqa: E402
from renderer_training.project_labels_to_us import (                     # noqa: E402
    DEFAULT_DISPLAY_FAN,
    DEFAULT_PROBE_GEOMETRY,
    DEFAULT_US_SPACING_MM,
    sector_zoom_pair,
)
from reslice import pose as posemod                                      # noqa: E402
from reslice.frame import affine_from_sitk                               # noqa: E402
from reslice.io import load_volume_data                                  # noqa: E402

LIVER_ID = 2


def _normalise(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1.0, 99.0])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _initial_pose(sequence: Path, frame: int) -> np.ndarray:
    world_from_probe = posemod.pose_from_sequence(sequence, frame, "ee", 0.0)
    world_from_phantom = posemod.default_world_from_phantom_centered_m()
    return posemod.probe_pose_in_phantom_centered_mm(world_from_probe, world_from_phantom)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 9])
    ap.add_argument("--registration-frames", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures" / "9_label_projection_check" / "fixed_projection_preview.png",
    )
    args = ap.parse_args()

    sequence = np.load(args.sequence, allow_pickle=True)
    images = sequence["images"]
    contact = np.asarray(sequence["contact"], dtype=bool)
    for frame in args.frames:
        if frame < 0 or frame >= len(images):
            raise SystemExit(f"frame {frame} outside 0..{len(images) - 1}")

    fan_px = FanFit(**DEFAULT_DISPLAY_FAN, resid_px=0.2879204551442961)
    geom = DEFAULT_PROBE_GEOMETRY

    volume = load_volume_data(args.volume)
    labels = load_volume_data(args.labels)
    import SimpleITK as sitk

    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))

    contact_indices = np.flatnonzero(contact)
    reg_indices = contact_indices[
        np.linspace(
            0,
            len(contact_indices) - 1,
            min(args.registration_frames, len(contact_indices)),
            dtype=int,
        )
    ]
    targets = []
    for frame in reg_indices:
        targets.append(
            FrameTargets(
                index=int(frame),
                us_polar=unwrap_fan(images[int(frame)], fan_px, geom.n_ax, geom.n_lat),
                init_pose_mm=_initial_pose(args.sequence, int(frame)),
            )
        )
    context = Context(volume=volume, affine_centered=affine, geom=geom, frames=targets)
    registration = register_global(context, maxiter=args.maxiter)
    correction = registration["correction"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.frames), 4, figsize=(13.2, 3.4 * len(args.frames)))
    axes = np.atleast_2d(axes)
    coverages = []
    paired_us = []
    paired_cbct = []
    paired_labels = []
    paired_masks = []
    paired_poses = []
    for row, frame in enumerate(args.frames):
        us = np.asarray(images[frame])
        refined_pose = correction @ _initial_pose(args.sequence, frame)
        cbct, label_map = sector_zoom_pair(
            volume,
            labels,
            affine,
            refined_pose,
            us.shape,
            display_fan=DEFAULT_DISPLAY_FAN,
            probe_geometry=geom,
        )
        liver = label_map == LIVER_ID
        coverages.append(float(liver.mean()))
        paired_us.append(us.astype(np.uint8))
        paired_cbct.append(cbct.astype(np.float32))
        paired_labels.append(label_map.astype(np.int16))
        paired_masks.append(liver.astype(np.uint8))
        paired_poses.append(refined_pose.astype(np.float64))

        axes[row, 0].imshow(_normalise(us), cmap="gray")
        axes[row, 0].set_title(f"real US · frame {frame}")
        axes[row, 1].imshow(cbct, cmap="gray")
        axes[row, 1].set_title("CBCT · same fitted fan")
        axes[row, 2].imshow(_normalise(us), cmap="gray")
        overlay = np.zeros((*liver.shape, 4), dtype=np.float32)
        overlay[liver] = (1.0, 0.35, 0.0, 0.48)
        axes[row, 2].imshow(overlay)
        if liver.any():
            axes[row, 2].contour(liver.astype(float), [0.5], colors="yellow", linewidths=0.8)
        axes[row, 2].set_title(f"US + liver mask · {100 * liver.mean():.1f}%")
        axes[row, 3].imshow(liver, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("binary liver mask · 660×880")
        for axis in axes[row]:
            axis.axis("off")

    before = np.asarray(registration["lc2_before"])
    after = np.asarray(registration["lc2_after"])
    inside = np.asarray(registration["inside_after"])
    fig.suptitle(
        "Fixed fan projection preview · "
        f"global LC2 {before.mean():.3f}→{after.mean():.3f} · "
        f"inside {100 * inside.mean():.1f}% · {DEFAULT_US_SPACING_MM:.9f} mm/px",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)

    data_path = args.out.with_name(f"{args.out.stem}_data.npz")
    np.savez_compressed(
        data_path,
        images=np.stack(paired_us),
        cbct=np.stack(paired_cbct),
        label_maps=np.stack(paired_labels),
        masks=np.stack(paired_masks),
        frame_index=np.asarray(args.frames, dtype=np.int32),
        refined_poses=np.stack(paired_poses),
        global_correction=np.asarray(correction),
        lc2_before=before,
        lc2_after=after,
        inside_after=inside,
    )

    print(f"wrote {args.out}")
    print(f"wrote {data_path}")
    print(f"registration frames: {reg_indices.tolist()}")
    print(f"LC2 mean: {before.mean():.4f} -> {after.mean():.4f}")
    print(f"inside mean after: {100 * inside.mean():.1f}%")
    print(f"liver coverage: {[round(100 * x, 2) for x in coverages]}%")
    print(
        "fan: "
        f"apex={tuple(round(x, 3) for x in fan_px.apex_px)}, "
        f"r0={fan_px.r0_px:.3f}px, r1={fan_px.r1_px:.3f}px, fov={fan_px.fov_deg:.3f}deg"
    )


if __name__ == "__main__":
    main()
