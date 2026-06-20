"""CLI (steps 2 + 4): reslice a CBCT fan sector from a recorded probe pose.

    python -m reslice.slice                          # defaults: 2026-06-12 scan1, frame 17
    python -m reslice.slice --sequence data/sequences/scan5.npz --sequence-frame 80 \
        --depth-mm 93 --fov-deg 57 --near-mm 15

Chain: probe pose -> phantom-centred mm -> image plane -> rectangular reslice ->
ultrasound fan sector -> compare against the matched real US frame. Outputs go to
``--out-dir`` (rect, sector, zoomed sector, US frame, side-by-side, report JSON).
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "reslice"

import argparse
import json
from pathlib import Path

import numpy as np

from reslice import pose as posemod
from reslice import sector as sec
from reslice.io import load_transform_4x4, load_volume_data
from reslice.sampling import reslice_rectangular_plane

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = PKG_DIR / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_WORLD_FROM_PHANTOM = PKG_DIR / "outputs" / "world_from_phantom_liedown.txt"

# Display convention that matched the original slicer_3.0 scan1 checks.
DISPLAY_AXIAL_SIGN = -1.0
DISPLAY_LATERAL_SIGN = 1.0


def reslice_sector(
    volume: np.ndarray,
    affine_centered: np.ndarray,
    T_phantom_from_probe_mm: np.ndarray,
    us_shape: tuple[int, int],
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Full pose -> rectangular plane -> fan sector. Returns ``(rect, sector, mask, debug)``."""
    plane = posemod.plane_from_probe_pose(T_phantom_from_probe_mm, args.plane_mode, args.plane_offset_y_mm)
    rect, valid = reslice_rectangular_plane(
        volume, affine_centered, plane,
        width_mm=args.slice_width_mm, height_mm=args.slice_height_mm,
        n_rows=us_shape[0], n_cols=us_shape[1],
        axial_sign=DISPLAY_AXIAL_SIGN, lateral_sign=DISPLAY_LATERAL_SIGN,
    )

    display_rot180 = not args.no_display_rot180
    if display_rot180:
        rect, valid = np.rot90(rect, 2), np.rot90(valid, 2)

    # Pose/edge apex (unless the legacy content-centre apex is requested).
    apex_pixel_rc = depth_direction_rc = None
    if not args.legacy_centered_apex:
        _, edge_threshold, _ = sec.detect_content_top_row(
            rect, valid, threshold=args.content_threshold, min_pixels=args.content_min_pixels
        )
        probe_pixel_rc = sec.project_point_to_display_pixel(
            T_phantom_from_probe_mm[:3, 3], plane,
            width_mm=args.slice_width_mm, height_mm=args.slice_height_mm,
            rows=rect.shape[0], cols=rect.shape[1],
            axial_sign=DISPLAY_AXIAL_SIGN, lateral_sign=DISPLAY_LATERAL_SIGN,
            display_rot180=display_rot180,
        )
        depth_direction_rc = sec.project_direction_to_display_rc(
            T_phantom_from_probe_mm[:3, 2], plane,
            axial_sign=DISPLAY_AXIAL_SIGN, lateral_sign=DISPLAY_LATERAL_SIGN,
            display_rot180=display_rot180,
        )
        apex_pixel_rc, _ = sec.apex_from_pose_and_edge(
            rect, valid, threshold=edge_threshold,
            probe_pixel_rc=probe_pixel_rc, depth_direction_rc=depth_direction_rc,
            max_line_distance_px=args.edge_line_max_distance_px,
        )

    sector, mask, debug = sec.apply_sector(
        rect, valid,
        depth_mm=args.depth_mm, near_mm=args.near_mm, fov_deg=args.fov_deg,
        width_mm=args.slice_width_mm, height_mm=args.slice_height_mm,
        top_margin_rows=args.top_margin_rows, apex_col_fraction=args.apex_col_fraction,
        apex_pixel_rc=apex_pixel_rc, depth_direction_rc=depth_direction_rc,
        content_threshold=args.content_threshold, content_min_pixels=args.content_min_pixels,
    )
    debug["display_rot180"] = display_rot180
    return rect, sector, mask, debug


def save_outputs(out_dir: Path, rect, sector, mask, us_image, debug, crop_margin_px) -> None:
    """Write the rect / sector / zoom PNGs, the side-by-side compare, and the report."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rect_norm = sec.normalize_image(rect)
    sector_norm = sec.normalize_image(sector)
    sector_for_save = np.where(mask, sector_norm, 0.0)

    plt.imsave(out_dir / "cbct_rect.png", rect_norm, cmap="gray")
    plt.imsave(out_dir / "cbct_sector.png", sector_for_save, cmap="gray")

    sector_zoom = None
    if us_image is not None:
        plt.imsave(out_dir / "us_frame.png", us_image, cmap="gray")
        # Crop mask = the full fan geometry (near=0) so the zoom frames the whole sector.
        crop_mask = sec.sector_mask_in_display_image(
            rect.shape, debug["apex_row"], debug["apex_col"],
            np.asarray(debug["depth_direction_rc"], dtype=np.float32),
            debug["fov_deg"], debug["depth_mm"], 0.0, debug["mm_per_row"], debug["mm_per_col"],
        )
        sector_zoom, zoom_dbg = sec.crop_and_zoom_sector(
            sector_norm, mask, crop_mask, tuple(us_image.shape[:2]), crop_margin_px
        )
        plt.imsave(out_dir / "cbct_sector_zoom.png", sector_zoom, cmap="gray")
        debug["content_zoom"] = zoom_dbg

    ncols = 3 if us_image is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 5.0), squeeze=False)
    axes[0, 0].imshow(rect_norm, cmap="gray")
    axes[0, 0].axhline(debug["content_top_row"], color="yellow", linewidth=1.0)
    axes[0, 0].scatter([debug["apex_col"]], [debug["apex_row"]], c="red", s=14)
    axes[0, 0].set_title("CBCT rectangular plane")
    axes[0, 1].imshow(sector_for_save, cmap="gray")
    axes[0, 1].set_title("CBCT fan sector")
    if us_image is not None:
        axes[0, 2].imshow(us_image, cmap="gray")
        axes[0, 2].set_title("matched US frame")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_cbct_us.png", dpi=160)

    if sector_zoom is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10.0, 5.0), squeeze=False)
        ax2[0, 0].imshow(sector_zoom, cmap="gray"); ax2[0, 0].set_title("CBCT sector (zoom)")
        ax2[0, 1].imshow(us_image, cmap="gray"); ax2[0, 1].set_title("matched US frame")
        for ax in ax2.ravel():
            ax.axis("off")
        fig2.tight_layout()
        fig2.savefig(out_dir / "compare_cbct_zoom_us.png", dpi=160)

    (out_dir / "sector_report.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="step-1 physical_frame_report.json")
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    ap.add_argument("--sequence-frame", type=int, default=17)
    ap.add_argument("--sequence-pose-kind", choices=("ee", "probe"), default="ee")
    ap.add_argument("--probe-rotation-offset-deg", type=float, default=posemod.DEFAULT_PROBE_ROTATION_OFFSET_DEG)
    ap.add_argument("--world-from-phantom", type=Path, default=DEFAULT_WORLD_FROM_PHANTOM,
                    help="4x4 T_world_from_phantom (m); falls back to the calibrated default if missing")
    ap.add_argument("--world-from-phantom-key")
    ap.add_argument("--out-dir", type=Path, default=PKG_DIR / "outputs" / "sector")
    # Plane / display.
    ap.add_argument("--plane-mode", choices=("probe-xz", "probe-xy"), default="probe-xz")
    ap.add_argument("--plane-offset-y-mm", type=float, default=0.0)
    ap.add_argument("--slice-width-mm", type=float, default=360.0)
    ap.add_argument("--slice-height-mm", type=float, default=300.0)
    ap.add_argument("--no-display-rot180", action="store_true")
    # Fan shape (tune to your US: depth / field-of-view / near cutoff).
    ap.add_argument("--depth-mm", type=float, default=100.0)
    ap.add_argument("--near-mm", type=float, default=20.0)
    ap.add_argument("--fov-deg", type=float, default=100.0)
    # Apex / content detection.
    ap.add_argument("--legacy-centered-apex", action="store_true",
                    help="use the content-top centre apex instead of the pose/edge intersection")
    ap.add_argument("--top-margin-rows", type=int, default=2)
    ap.add_argument("--apex-col-fraction", type=float, default=0.5)
    ap.add_argument("--content-threshold", type=float, default=None)
    ap.add_argument("--content-min-pixels", type=int, default=8)
    ap.add_argument("--edge-line-max-distance-px", type=float, default=5.0)
    ap.add_argument("--crop-margin-px", type=int, default=18)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    frame_key = "recon" if args.volume_path.suffix.lower() == ".mhd" else "dicom"
    affine_centered = np.array(
        report["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"], dtype=float
    )

    T_world_from_probe_m = posemod.pose_from_sequence(
        args.sequence, args.sequence_frame, args.sequence_pose_kind, args.probe_rotation_offset_deg
    )
    if args.world_from_phantom is not None and args.world_from_phantom.exists():
        T_world_from_phantom_m = load_transform_4x4(args.world_from_phantom, args.world_from_phantom_key)
        placement_source = str(args.world_from_phantom)
    else:
        T_world_from_phantom_m = posemod.default_world_from_phantom_centered_m()
        placement_source = "calibrated default (T_WORLD_FROM_PHANTOM_MEASURED @ lie-down)"
    T_phantom_from_probe_mm = posemod.probe_pose_in_phantom_centered_mm(
        T_world_from_probe_m, T_world_from_phantom_m
    )

    seq = np.load(args.sequence, allow_pickle=True)
    us_image = np.asarray(seq["images"][args.sequence_frame])
    volume = load_volume_data(args.volume_path)

    rect, sector, mask, debug = reslice_sector(
        volume, affine_centered, T_phantom_from_probe_mm, us_image.shape[:2], args
    )
    save_outputs(args.out_dir, rect, sector, mask, us_image, debug, args.crop_margin_px)

    print(f"[slice] sequence   : {args.sequence} frame {args.sequence_frame}")
    print(f"[slice] volume     : {args.volume_path}  (frame_key={frame_key})")
    print(f"[slice] placement  : {placement_source}")
    print(f"[slice] fan        : depth {args.depth_mm}mm fov {args.fov_deg}deg near {args.near_mm}mm")
    print(f"[slice] apex source: {debug['apex_source']}  sector pixels: {debug['sector_pixel_count']}")
    print(f"[slice] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
