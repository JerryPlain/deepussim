"""Create a CBCT fan-sector slice from a tracked probe pose.

This keeps the slicer_3.0 pose chain from 02_slice_from_probe_pose.py, then
restores the older ultrasound-style fan logic:

1. sample a rectangular plane from the probe pose;
2. detect the first non-black phantom/content edge;
3. project the probe pose into the displayed slice and intersect its beam line
   with that edge, giving the red-point fan apex;
4. save the fan-masked CBCT slice beside the matched US frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
STEP02_PATH = SCRIPT_DIR / "02_slice_from_probe_pose.py"

DEFAULT_VOLUME_PATH = WORKSPACE_ROOT / "cbct_20260612" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_SEQUENCE = REPO_ROOT / "data" / "sequences_20260612" / "scan1.npz"
DEFAULT_SEQUENCE_FRAME = 17
DEFAULT_WORLD_FROM_PHANTOM = SCRIPT_DIR / "outputs" / "yesterday_T_world_from_phantom_liedown.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "step04_sector_slice_from_probe"
DEFAULT_ORIGIN000_STEP01_REPORT = (
    SCRIPT_DIR / "outputs" / "step01_physical_frame_origin000" / "physical_frame_report.json"
)


def load_step02_module():
    spec = importlib.util.spec_from_file_location("slicer3_step02", STEP02_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STEP02_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step02 = load_step02_module()


def normalize_image(img: np.ndarray) -> np.ndarray:
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)
    lo = float(np.percentile(img[finite], 0.5))
    hi = float(np.percentile(img[finite], 99.5))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def detect_content_top_row(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float | None = None,
    min_pixels: int = 8,
) -> tuple[int, float, list[int]]:
    """Return first row with enough non-background phantom pixels."""
    finite = np.isfinite(image) & valid_mask
    if threshold is None:
        values = image[finite]
        if values.size == 0:
            threshold = 0.0
        else:
            background = float(np.nanmin(values))
            threshold = background + 50.0

    row_counts = np.sum(finite & (image > threshold), axis=1)
    hits = np.flatnonzero(row_counts >= min_pixels)
    if hits.size == 0:
        top_row = 0
    else:
        top_row = int(hits[0])
    return top_row, float(threshold), row_counts.astype(int).tolist()


def top_edge_rows_by_col(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    content = np.isfinite(image) & valid_mask & (image > threshold)
    rows, cols = image.shape
    edge_rows = np.full(cols, np.nan, dtype=np.float64)
    for col in range(cols):
        hits = np.flatnonzero(content[:, col])
        if hits.size:
            edge_rows[col] = float(hits[0])
    return edge_rows


def project_point_to_display_pixel(
    point_mm: np.ndarray,
    plane: dict[str, np.ndarray],
    *,
    width_mm: float,
    height_mm: float,
    rows: int,
    cols: int,
    axial_sign: float,
    lateral_sign: float,
    display_rot180: bool,
) -> np.ndarray:
    delta = np.asarray(point_mm, dtype=np.float64) - plane["point_centered_mm"]
    u_mm = float(np.dot(delta, lateral_sign * plane["lateral_centered"]))
    v_mm = float(np.dot(delta, axial_sign * plane["axial_centered"]))
    col = (u_mm + width_mm / 2.0) / (width_mm / max(cols - 1, 1))
    row = (v_mm + height_mm * 0.25) / (height_mm / max(rows - 1, 1))
    if display_rot180:
        row = rows - 1 - row
        col = cols - 1 - col
    return np.array([row, col], dtype=np.float64)


def project_direction_to_display_rc(
    direction_mm: np.ndarray,
    plane: dict[str, np.ndarray],
    *,
    axial_sign: float,
    lateral_sign: float,
    display_rot180: bool,
) -> np.ndarray:
    direction = np.asarray(direction_mm, dtype=np.float64)
    rc = np.array(
        [
            float(np.dot(direction, axial_sign * plane["axial_centered"])),
            float(np.dot(direction, lateral_sign * plane["lateral_centered"])),
        ],
        dtype=np.float64,
    )
    if display_rot180:
        rc *= -1.0
    norm = float(np.linalg.norm(rc))
    if norm == 0.0:
        return np.array([1.0, 0.0], dtype=np.float64)
    return rc / norm


def apex_from_pose_and_edge(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    threshold: float,
    probe_pixel_rc: np.ndarray,
    depth_direction_rc: np.ndarray,
    max_line_distance_px: float = 5.0,
) -> tuple[np.ndarray, dict]:
    edge_rows = top_edge_rows_by_col(image, valid_mask, threshold)
    cols = np.flatnonzero(np.isfinite(edge_rows))
    if cols.size == 0:
        return probe_pixel_rc.copy(), {
            "method": "fallback_probe_pixel_no_edge",
            "max_line_distance_px": float(max_line_distance_px),
        }

    edge_pts = np.column_stack([edge_rows[cols], cols.astype(np.float64)])
    line_dir = np.asarray(depth_direction_rc, dtype=np.float64)
    line_dir_norm = float(np.linalg.norm(line_dir))
    if line_dir_norm == 0.0:
        line_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        line_dir /= line_dir_norm

    rel = edge_pts - probe_pixel_rc[None, :]
    t = rel @ line_dir
    closest = probe_pixel_rc[None, :] + t[:, None] * line_dir[None, :]
    dist = np.linalg.norm(edge_pts - closest, axis=1)

    candidates = np.flatnonzero(dist <= max_line_distance_px)
    if candidates.size:
        # Prefer the surface intersection in the probe depth direction. If the
        # projected probe point is already below the edge, allow the nearest
        # line hit in either direction.
        positive = candidates[t[candidates] >= 0.0]
        pool = positive if positive.size else candidates
        idx = int(pool[np.argmin(np.abs(t[pool]))])
        method = "pose_line_edge_intersection"
    else:
        idx = int(np.nanargmin(dist))
        method = "nearest_edge_to_pose_line"

    return edge_pts[idx], {
        "method": method,
        "probe_pixel_rc": probe_pixel_rc.tolist(),
        "depth_direction_rc": line_dir.tolist(),
        "edge_col": int(cols[idx]),
        "edge_row": float(edge_pts[idx, 0]),
        "line_t_px": float(t[idx]),
        "line_distance_px": float(dist[idx]),
        "max_line_distance_px": float(max_line_distance_px),
    }


def sector_mask_in_display_image(
    shape: tuple[int, int],
    apex_row: float,
    apex_col: float,
    depth_direction: np.ndarray,
    fov_deg: float,
    depth_mm: float,
    near_mm: float,
    mm_per_row: float,
    mm_per_col: float,
) -> np.ndarray:
    """Fan mask in image space with an arbitrary 2D depth direction."""
    rows, cols = shape
    rr, cc = np.indices((rows, cols), dtype=np.float32)

    # Convert image offsets to a metric display plane. +row is down, +col is right.
    offsets = np.stack(
        ((rr - float(apex_row)) * float(mm_per_row), (cc - float(apex_col)) * float(mm_per_col)),
        axis=-1,
    )

    depth_axis = np.asarray(depth_direction, dtype=np.float32)
    norm = float(np.linalg.norm(depth_axis))
    if norm == 0.0:
        depth_axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        depth_axis /= norm
    lateral_axis = np.array([-depth_axis[1], depth_axis[0]], dtype=np.float32)

    axial = offsets[..., 0] * depth_axis[0] + offsets[..., 1] * depth_axis[1]
    lateral = offsets[..., 0] * lateral_axis[0] + offsets[..., 1] * lateral_axis[1]

    radius = np.sqrt(axial * axial + lateral * lateral)
    half_angle = np.deg2rad(float(fov_deg)) / 2.0
    angle = np.abs(np.arctan2(lateral, axial))

    return (axial >= 0.0) & (radius >= near_mm) & (radius <= depth_mm) & (angle <= half_angle)


def apply_sector(
    rect: np.ndarray,
    valid_mask: np.ndarray,
    *,
    depth_mm: float,
    near_mm: float,
    fov_deg: float,
    width_mm: float,
    height_mm: float,
    top_margin_rows: int,
    apex_col_fraction: float,
    apex_pixel_rc: np.ndarray | None,
    depth_direction_rc: np.ndarray | None,
    content_threshold: float | None,
    content_min_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, cols = rect.shape
    top_row, threshold, row_counts = detect_content_top_row(
        rect,
        valid_mask,
        threshold=content_threshold,
        min_pixels=content_min_pixels,
    )
    if apex_pixel_rc is None:
        apex_row = max(0, top_row - int(top_margin_rows))
        apex_col = float(cols - 1) * float(apex_col_fraction)
        apex_source = "content_top_center_fraction"
    else:
        apex_row = float(apex_pixel_rc[0])
        apex_col = float(apex_pixel_rc[1])
        apex_source = "pose_edge_intersection"
    depth_direction = (
        np.array([1.0, 0.0], dtype=np.float32)
        if depth_direction_rc is None
        else np.asarray(depth_direction_rc, dtype=np.float32)
    )
    mm_per_row = float(height_mm) / max(rows - 1, 1)
    mm_per_col = float(width_mm) / max(cols - 1, 1)

    mask = sector_mask_in_display_image(
        rect.shape,
        apex_row=apex_row,
        apex_col=apex_col,
        depth_direction=depth_direction,
        fov_deg=fov_deg,
        depth_mm=depth_mm,
        near_mm=near_mm,
        mm_per_row=mm_per_row,
        mm_per_col=mm_per_col,
    )
    mask &= valid_mask

    sector = np.full_like(rect, np.nan, dtype=np.float32)
    sector[mask] = rect[mask]
    debug = {
        "content_top_row": int(top_row),
        "content_threshold": float(threshold),
        "content_min_pixels": int(content_min_pixels),
        "apex_row": float(apex_row),
        "apex_col": float(apex_col),
        "apex_source": apex_source,
        "depth_direction_rc": depth_direction.astype(float).tolist(),
        "apex_col_fraction": float(apex_col_fraction),
        "top_margin_rows": int(top_margin_rows),
        "fov_deg": float(fov_deg),
        "depth_mm": float(depth_mm),
        "near_mm": float(near_mm),
        "mm_per_row": float(mm_per_row),
        "mm_per_col": float(mm_per_col),
        "sector_pixel_count": int(np.count_nonzero(mask)),
        "row_counts_first_40": row_counts[:40],
    }
    return sector, mask, debug


def crop_and_zoom_sector(
    sector_norm: np.ndarray,
    mask: np.ndarray,
    crop_mask: np.ndarray,
    target_shape: tuple[int, int],
    margin_px: int,
) -> tuple[np.ndarray, dict]:
    hits = np.argwhere(crop_mask)
    if hits.size == 0:
        return np.zeros(target_shape, dtype=np.float32), {
            "crop_bbox_rc": None,
            "crop_margin_px": int(margin_px),
            "zoom_to_shape": list(target_shape),
        }

    r0, c0 = hits.min(axis=0)
    r1, c1 = hits.max(axis=0) + 1
    r0 = max(0, int(r0) - int(margin_px))
    c0 = max(0, int(c0) - int(margin_px))
    r1 = min(mask.shape[0], int(r1) + int(margin_px))
    c1 = min(mask.shape[1], int(c1) + int(margin_px))

    crop = np.where(mask[r0:r1, c0:c1], sector_norm[r0:r1, c0:c1], 0.0)
    zoom = (
        float(target_shape[0]) / max(crop.shape[0], 1),
        float(target_shape[1]) / max(crop.shape[1], 1),
    )
    zoomed = ndimage.zoom(crop, zoom, order=1)
    zoomed = zoomed[: target_shape[0], : target_shape[1]]
    if zoomed.shape != target_shape:
        out = np.zeros(target_shape, dtype=np.float32)
        out[: zoomed.shape[0], : zoomed.shape[1]] = zoomed
        zoomed = out

    return zoomed.astype(np.float32), {
        "crop_bbox_rc": [int(r0), int(c0), int(r1), int(c1)],
        "crop_margin_px": int(margin_px),
        "zoom_to_shape": list(target_shape),
        "zoom_factor_rc": [float(zoom[0]), float(zoom[1])],
    }


def save_results(
    output_dir: Path,
    rect: np.ndarray,
    sector: np.ndarray,
    mask: np.ndarray,
    us_image: np.ndarray | None,
    report: dict,
    debug: dict,
    crop_margin_px: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    rect_norm = normalize_image(rect)
    sector_norm = normalize_image(sector)

    plt.imsave(output_dir / "cbct_rect_frame.png", rect_norm, cmap="gray")
    sector_for_save = np.where(mask, sector_norm, 0.0)
    plt.imsave(output_dir / "cbct_sector_frame.png", sector_for_save, cmap="gray")

    if us_image is not None:
        plt.imsave(output_dir / "us_frame.png", us_image, cmap="gray")
        fan_geometry_crop_mask = sector_mask_in_display_image(
            rect.shape,
            apex_row=debug["apex_row"],
            apex_col=debug["apex_col"],
            depth_direction=np.asarray(debug["depth_direction_rc"], dtype=np.float32),
            fov_deg=debug["fov_deg"],
            depth_mm=debug["depth_mm"],
            near_mm=0.0,
            mm_per_row=debug["mm_per_row"],
            mm_per_col=debug["mm_per_col"],
        )
        sector_zoom, zoom_debug = crop_and_zoom_sector(
            sector_norm,
            mask,
            fan_geometry_crop_mask,
            target_shape=tuple(us_image.shape[:2]),
            margin_px=crop_margin_px,
        )
        plt.imsave(output_dir / "cbct_sector_content_zoom.png", sector_zoom, cmap="gray")
        debug["content_zoom"] = zoom_debug

    ncols = 3 if us_image is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 5.0), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(rect_norm, cmap="gray")
    ax.axhline(debug["content_top_row"], color="yellow", linewidth=1.0)
    ax.scatter([debug["apex_col"]], [debug["apex_row"]], c="red", s=14)
    ax.set_title("CBCT rectangular plane")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(sector_for_save, cmap="gray")
    ax.set_title("CBCT fan sector")
    ax.axis("off")

    if us_image is not None:
        ax = axes[0, 2]
        ax.imshow(us_image, cmap="gray")
        ax.set_title("Matched US frame")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "compare_cbct_sector_us.png", dpi=160)
    plt.close(fig)

    if us_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), squeeze=False)
        axes[0, 0].imshow(sector_zoom, cmap="gray")
        axes[0, 0].set_title("CBCT sector content zoom")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(us_image, cmap="gray")
        axes[0, 1].set_title("Matched US frame")
        axes[0, 1].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "compare_cbct_sector_zoom_us.png", dpi=160)
        plt.close(fig)

    with (output_dir / "sector_report.json").open("w", encoding="utf-8") as f:
        json.dump({"pose_report": report, "sector_debug": debug}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step01-report",
        type=Path,
        default=DEFAULT_ORIGIN000_STEP01_REPORT,
        help="step01 report defining the phantom-centered frame; defaults to phantom origin LPS [0,0,0]",
    )
    parser.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME_PATH)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--sequence-frame", type=int, default=DEFAULT_SEQUENCE_FRAME)
    parser.add_argument("--sequence-pose-kind", choices=("ee", "probe"), default="ee")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slice-width-mm", type=float, default=step02.DEFAULT_SLICE_WIDTH_MM)
    parser.add_argument("--slice-height-mm", type=float, default=step02.DEFAULT_SLICE_HEIGHT_MM)
    parser.add_argument("--slice-axial-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--slice-lateral-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--depth-mm", type=float, default=100.0)
    parser.add_argument("--near-mm", type=float, default=20.0)
    parser.add_argument("--fov-deg", type=float, default=100.0)
    parser.add_argument("--top-margin-rows", type=int, default=2)
    parser.add_argument("--apex-col-fraction", type=float, default=0.5)
    parser.add_argument("--content-threshold", type=float, default=None)
    parser.add_argument("--content-min-pixels", type=int, default=8)
    parser.add_argument("--crop-margin-px", type=int, default=18)
    parser.add_argument("--edge-line-max-distance-px", type=float, default=5.0)
    parser.add_argument(
        "--legacy-centered-apex",
        action="store_true",
        help="use the old content-top centered apex instead of pose/edge intersection",
    )
    parser.add_argument("--plane-mode", choices=("probe-xz", "probe-xy"), default="probe-xz")
    parser.add_argument("--plane-offset-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--probe-rotation-offset-deg",
        type=float,
        default=step02.DEFAULT_PROBE_ROTATION_OFFSET_DEG,
    )
    parser.add_argument("--world-from-phantom", type=Path, default=DEFAULT_WORLD_FROM_PHANTOM)
    parser.add_argument("--world-from-phantom-key")
    parser.add_argument("--no-display-rot180", action="store_true")
    args = parser.parse_args()

    step01_report = json.loads(args.step01_report.read_text(encoding="utf-8"))
    phantom_origin = np.array(
        step01_report["phantom_centered_frame"]["origin_lps_mm"],
        dtype=float,
    )

    T_world_from_probe_m = step02.pose_from_sequence(
        args.sequence,
        args.sequence_frame,
        args.sequence_pose_kind,
        args.probe_rotation_offset_deg,
    )
    if args.world_from_phantom is not None:
        T_world_from_phantom_m = step02.load_transform_4x4(
            args.world_from_phantom,
            args.world_from_phantom_key,
        )
        placement_source = str(args.world_from_phantom)
    else:
        T_world_from_phantom_m = step02.default_world_from_phantom_centered_m()
        placement_source = "default measured placement @ Rx90*Rz180"

    T_phantom_from_probe_mm = step02.probe_pose_in_phantom_centered_mm(
        T_world_from_probe_m,
        T_world_from_phantom_m,
    )
    plane = step02.plane_from_probe_pose(
        T_phantom_from_probe_mm,
        args.plane_mode,
        args.plane_offset_y_mm,
    )

    seq = np.load(args.sequence, allow_pickle=True)
    us_image = np.asarray(seq["images"][args.sequence_frame])
    volume = step02.load_volume_data(args.volume_path)
    frame_key = "recon" if args.volume_path.suffix.lower() == ".mhd" else "dicom"
    affine_key = f"{frame_key}_affine_centered_from_ijk_mm"
    affine_centered = np.array(
        step01_report["phantom_centered_frame"][affine_key],
        dtype=float,
    )

    rect, valid_mask = step02.reslice_rectangular_plane(
        volume,
        affine_centered,
        plane,
        width_mm=args.slice_width_mm,
        height_mm=args.slice_height_mm,
        n_rows=int(us_image.shape[0]),
        n_cols=int(us_image.shape[1]),
        axial_sign=args.slice_axial_sign,
        lateral_sign=args.slice_lateral_sign,
    )

    report = {
        "pose_source": (
            f"{args.sequence}, frame {args.sequence_frame}, kind {args.sequence_pose_kind}, "
            f"probe Rz {args.probe_rotation_offset_deg:g} deg"
        ),
        "placement_source": placement_source,
        "volume_path": str(args.volume_path),
        "sequence": str(args.sequence),
        "sequence_frame": int(args.sequence_frame),
        "phantom_origin_lps_mm": phantom_origin.tolist(),
        "T_world_from_probe_m": T_world_from_probe_m.tolist(),
        "T_world_from_phantom_centered_m": T_world_from_phantom_m.tolist(),
        "T_phantom_centered_from_probe_mm": T_phantom_from_probe_mm.tolist(),
        "plane_phantom_centered": {
            "point_mm": plane["point_centered_mm"].tolist(),
            "normal_unit": plane["normal_centered"].tolist(),
            "lateral_unit": plane["lateral_centered"].tolist(),
            "axial_unit": plane["axial_centered"].tolist(),
            "meaning": plane["meaning"],
            "plane_mode": args.plane_mode,
            "plane_offset_y_mm": plane["plane_offset_y_mm"],
        },
        "slice_width_mm": float(args.slice_width_mm),
        "slice_height_mm": float(args.slice_height_mm),
        "slice_axial_sign": float(args.slice_axial_sign),
        "slice_lateral_sign": float(args.slice_lateral_sign),
    }

    display_rot180 = not args.no_display_rot180
    if display_rot180:
        rect = np.rot90(rect, 2)
        valid_mask = np.rot90(valid_mask, 2)
        report["display_rot180"] = True
    else:
        report["display_rot180"] = False

    apex_pixel_rc = None
    depth_direction_rc = None
    apex_debug = {"method": "legacy_centered_apex"}
    if not args.legacy_centered_apex:
        _, edge_threshold, _ = detect_content_top_row(
            rect,
            valid_mask,
            threshold=args.content_threshold,
            min_pixels=args.content_min_pixels,
        )
        probe_point_mm = T_phantom_from_probe_mm[:3, 3]
        probe_depth_mm = T_phantom_from_probe_mm[:3, 2]
        probe_pixel_rc = project_point_to_display_pixel(
            probe_point_mm,
            plane,
            width_mm=args.slice_width_mm,
            height_mm=args.slice_height_mm,
            rows=rect.shape[0],
            cols=rect.shape[1],
            axial_sign=args.slice_axial_sign,
            lateral_sign=args.slice_lateral_sign,
            display_rot180=display_rot180,
        )
        depth_direction_rc = project_direction_to_display_rc(
            probe_depth_mm,
            plane,
            axial_sign=args.slice_axial_sign,
            lateral_sign=args.slice_lateral_sign,
            display_rot180=display_rot180,
        )
        apex_pixel_rc, apex_debug = apex_from_pose_and_edge(
            rect,
            valid_mask,
            threshold=edge_threshold,
            probe_pixel_rc=probe_pixel_rc,
            depth_direction_rc=depth_direction_rc,
            max_line_distance_px=args.edge_line_max_distance_px,
        )

    sector, mask, debug = apply_sector(
        rect,
        valid_mask,
        depth_mm=args.depth_mm,
        near_mm=args.near_mm,
        fov_deg=args.fov_deg,
        width_mm=args.slice_width_mm,
        height_mm=args.slice_height_mm,
        top_margin_rows=args.top_margin_rows,
        apex_col_fraction=args.apex_col_fraction,
        apex_pixel_rc=apex_pixel_rc,
        depth_direction_rc=depth_direction_rc,
        content_threshold=args.content_threshold,
        content_min_pixels=args.content_min_pixels,
    )
    debug["apex_from_pose_edge"] = apex_debug

    save_results(args.out_dir, rect, sector, mask, us_image, report, debug, args.crop_margin_px)
    print(json.dumps({"output_dir": str(args.out_dir), "report": report, "sector_debug": debug}, indent=2))


if __name__ == "__main__":
    main()
