#!/usr/bin/env python
"""Preview a CBCT reslice region derived only from the real-US inner arc.

The CBCT pose, plane construction, and rectangular sampling convention are the same
as the existing reslice comparison pipeline.  This preview deliberately performs no
crop-and-resize step: the US and CBCT stay on one ``rows x cols`` pixel grid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_script.plots_reslice.compare import (  # noqa: E402
    AXIAL_SIGN,
    FAN,
    LATERAL_SIGN,
    SLICE_H,
    SLICE_W,
    _normalize_display,
)
from reslice import pose as P  # noqa: E402
from reslice.io import load_transform_4x4, load_volume_data  # noqa: E402
from reslice.sampling import reslice_rectangular_plane  # noqa: E402
from reslice import sector as sec  # noqa: E402

US_SPACING_MM = 0.166112957  # measured real-US display spacing (Feng's calibration)


def _gray_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3:
        return np.mean(image[..., :3], axis=-1, dtype=np.float64)
    raise ValueError(f"expected a 2-D or RGB US image, got {image.shape}")


def fit_inner_arc(image: np.ndarray, threshold: float = 30.0) -> dict[str, np.ndarray | float]:
    """Fit the bright US inner arc in pixel coordinates.

    The fan is symmetric about the image centre, so its circle centre column is fixed
    at ``(width - 1) / 2``.  Only the circle centre row and radius are fitted.
    """
    gray = ndimage.gaussian_filter(_gray_image(image), sigma=0.8)
    rows, cols = gray.shape
    centre_col = (cols - 1.0) / 2.0

    edge_rows = np.full(cols, np.nan, dtype=np.float64)
    top_limit = min(rows, max(120, int(round(rows * 0.22))))
    for col in range(cols):
        hits = np.flatnonzero(gray[:top_limit, col] >= float(threshold))
        if hits.size:
            edge_rows[col] = float(hits[0])

    xx = np.arange(cols, dtype=np.float64)
    candidate = (
        np.isfinite(edge_rows)
        & (edge_rows >= 2.0)
        & (edge_rows <= min(110.0, rows * 0.18))
        & (xx >= cols * 0.23)
        & (xx <= cols * 0.77)
    )
    x = xx[candidate]
    y = edge_rows[candidate]
    if x.size < 80:
        raise RuntimeError(f"only {x.size} inner-arc pixels found at threshold {threshold:g}")

    def residual(params: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        centre_row, radius = params
        return np.hypot(px - centre_col, py - centre_row) - radius

    initial = np.array([-0.40 * rows, 0.47 * rows], dtype=np.float64)
    lower = np.array([-2.0 * rows, 0.12 * rows], dtype=np.float64)
    upper = np.array([-2.0, 1.8 * rows], dtype=np.float64)
    fit = least_squares(
        residual,
        initial,
        args=(x, y),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.5,
    )
    distances = residual(fit.x, x, y)
    inliers = np.abs(distances) <= 2.5
    if np.count_nonzero(inliers) >= 60:
        fit = least_squares(
            residual,
            fit.x,
            args=(x[inliers], y[inliers]),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.8,
        )
        distances = residual(fit.x, x, y)
        inliers = np.abs(distances) <= 2.5

    centre_row, inner_radius = map(float, fit.x)
    disc = inner_radius**2 - centre_row**2
    if disc <= 0:
        raise RuntimeError("fitted inner circle does not intersect the top image border")
    half_opening = float(np.sqrt(disc))
    p_left = np.array([centre_col - half_opening, 0.0])
    p_right = np.array([centre_col + half_opening, 0.0])
    centre = np.array([centre_col, centre_row])

    # A circle normal is its radius.  Each normal first crosses an image side; it then
    # continues outside the screen until it meets the bottom-tangent concentric arc.
    n_left = (p_left - centre) / inner_radius
    n_right = (p_right - centre) / inner_radius
    t_left = (0.0 - p_left[0]) / n_left[0]
    t_right = ((cols - 1.0) - p_right[0]) / n_right[0]
    side_left = p_left + t_left * n_left
    side_right = p_right + t_right * n_right
    outer_radius = float((rows - 1.0) - centre_row)
    outer_left = centre + outer_radius * n_left
    outer_right = centre + outer_radius * n_right
    bottom_tangent = np.array([centre_col, rows - 1.0])

    angle_left = float(np.arctan2(p_left[0] - centre_col, p_left[1] - centre_row))
    angle_right = float(np.arctan2(p_right[0] - centre_col, p_right[1] - centre_row))
    return {
        "centre": centre,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "p_left": p_left,
        "p_right": p_right,
        "side_left": side_left,
        "side_right": side_right,
        "outer_left": outer_left,
        "outer_right": outer_right,
        "bottom_tangent": bottom_tangent,
        "angle_left": angle_left,
        "angle_right": angle_right,
        "edge_x": x[inliers],
        "edge_y": y[inliers],
        "median_residual": float(np.median(np.abs(distances[inliers]))),
    }


def region_mask(shape: tuple[int, int], geometry: dict[str, np.ndarray | float]) -> np.ndarray:
    rows, cols = shape
    yy, xx = np.indices((rows, cols), dtype=np.float64)
    centre = np.asarray(geometry["centre"], dtype=float)
    dx, dy = xx - centre[0], yy - centre[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dx, dy)
    return (
        (radius >= float(geometry["inner_radius"]))
        & (radius <= float(geometry["outer_radius"]))
        & (angle >= float(geometry["angle_left"]))
        & (angle <= float(geometry["angle_right"]))
    )


def boundary_curves(geometry: dict[str, np.ndarray | float], n: int = 500) -> tuple[np.ndarray, ...]:
    centre = np.asarray(geometry["centre"], dtype=float)
    angles = np.linspace(float(geometry["angle_left"]), float(geometry["angle_right"]), n)
    curves = []
    for key in ("inner_radius", "outer_radius"):
        radius = float(geometry[key])
        curves.append(
            np.column_stack(
                [centre[0] + radius * np.sin(angles), centre[1] + radius * np.cos(angles)]
            )
        )
    return curves[0], curves[1]


def display_pixel_to_plane_point(
    pixel_rc: np.ndarray,
    plane: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Invert the existing 180-degree-rotated rectangular display grid."""
    rows, cols = shape
    row, col = map(float, pixel_rc)
    unrotated_row = rows - 1.0 - row
    unrotated_col = cols - 1.0 - col
    lateral_mm = -SLICE_W / 2.0 + unrotated_col * SLICE_W / max(cols - 1, 1)
    axial_mm = -SLICE_H * 0.25 + unrotated_row * SLICE_H / max(rows - 1, 1)
    return (
        np.asarray(plane["point_centered_mm"], dtype=float)
        + LATERAL_SIGN * np.asarray(plane["lateral_centered"], dtype=float) * lateral_mm
        + AXIAL_SIGN * np.asarray(plane["axial_centered"], dtype=float) * axial_mm
    )


def sample_cbct_on_us_grid(
    volume: np.ndarray,
    affine: np.ndarray,
    plane: dict[str, np.ndarray],
    surface_point_mm: np.ndarray,
    shape: tuple[int, int],
    geometry: dict[str, np.ndarray | float],
    *,
    mm_per_pixel: float = US_SPACING_MM,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Directly sample one isotropic Cartesian plane whose fan has the US pixel shape.

    The central inner-arc pixel remains anchored at the existing pipeline's ``near_mm``
    point. The supplied real-US spacing is then used identically for rows and columns;
    no already-acquired CBCT image is cropped and resized.
    """
    rows, cols = shape
    centre = np.asarray(geometry["centre"], dtype=float)
    inner_radius = float(geometry["inner_radius"])
    near_mm = float(FAN["near_mm"])
    mm_per_pixel = float(mm_per_pixel)
    if mm_per_pixel <= 0.0:
        raise ValueError("mm_per_pixel must be positive")
    if order not in (0, 1):
        raise ValueError("order must be 0 (labels) or 1 (intensity)")

    # These are the physical directions of increasing row/column after the existing
    # rectangular plane has been rotated by 180 degrees for display.
    row_direction = -AXIAL_SIGN * np.asarray(plane["axial_centered"], dtype=float)
    col_direction = -LATERAL_SIGN * np.asarray(plane["lateral_centered"], dtype=float)
    row_direction /= np.linalg.norm(row_direction)
    col_direction /= np.linalg.norm(col_direction)

    inner_centre_row = centre[1] + inner_radius
    inner_centre_point = np.asarray(surface_point_mm, dtype=float) + near_mm * row_direction
    yy, xx = np.indices((rows, cols), dtype=np.float64)
    points = (
        inner_centre_point[:, None]
        + row_direction[:, None] * ((yy.ravel() - inner_centre_row) * mm_per_pixel)[None, :]
        + col_direction[:, None] * ((xx.ravel() - centre[0]) * mm_per_pixel)[None, :]
    )
    vox = np.linalg.inv(affine) @ np.vstack([points, np.ones(points.shape[1])])
    inside = np.ones(vox.shape[1], dtype=bool)
    for axis, size in enumerate(volume.shape):
        inside &= (vox[axis] >= 0.0) & (vox[axis] <= size - 1)
    sampled = ndimage.map_coordinates(
        volume,
        vox[:3],
        order=order,
        mode="constant",
        cval=float(np.min(volume)),
    ).reshape(rows, cols)
    return sampled, inside.reshape(rows, cols), float(mm_per_pixel)


def cbct_bottom_tangent_reslice(
    volume: np.ndarray,
    affine: np.ndarray,
    phantom_from_probe_mm: np.ndarray,
    us: np.ndarray,
    *,
    inner_arc_threshold: float = 30.0,
    us_spacing_mm: float = US_SPACING_MM,
) -> tuple[np.ndarray, dict[str, np.ndarray | float], dict]:
    """Run the validated pose anchoring and directly sample the bottom-tangent fan."""
    geometry = fit_inner_arc(us, threshold=inner_arc_threshold)
    mask = region_mask(us.shape[:2], geometry)
    plane = P.plane_from_probe_pose(phantom_from_probe_mm, "probe-xz", 0.0)

    # Keep the original broad-plane path intact for the already-validated surface
    # intersection and pose anchoring.
    rect, valid = reslice_rectangular_plane(
        volume,
        affine,
        plane,
        width_mm=SLICE_W,
        height_mm=SLICE_H,
        n_rows=us.shape[0],
        n_cols=us.shape[1],
        axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN,
    )
    rect, valid = np.rot90(rect, 2), np.rot90(valid, 2)
    _, threshold, _ = sec.detect_content_top_row(rect, valid, threshold=None, min_pixels=8)
    probe_pixel = sec.project_point_to_display_pixel(
        phantom_from_probe_mm[:3, 3],
        plane,
        width_mm=SLICE_W,
        height_mm=SLICE_H,
        rows=rect.shape[0],
        cols=rect.shape[1],
        axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN,
        display_rot180=True,
    )
    depth_direction = sec.project_direction_to_display_rc(
        phantom_from_probe_mm[:3, 2],
        plane,
        axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN,
        display_rot180=True,
    )
    surface_pixel, surface_debug = sec.apex_from_pose_and_edge(
        rect,
        valid,
        threshold=threshold,
        probe_pixel_rc=probe_pixel,
        depth_direction_rc=depth_direction,
        max_line_distance_px=5.0,
    )
    surface_point = display_pixel_to_plane_point(surface_pixel, plane, us.shape[:2])
    direct, direct_valid, mm_per_pixel = sample_cbct_on_us_grid(
        volume,
        affine,
        plane,
        surface_point,
        us.shape[:2],
        geometry,
        mm_per_pixel=us_spacing_mm,
    )
    cbct = sec.normalize_image(np.where(direct_valid & mask, direct, np.nan))
    cbct = np.where(direct_valid & mask, cbct, 0.0).astype(np.float32)
    debug = {
        "surface_pixel_rc": np.asarray(surface_pixel).tolist(),
        "surface_detection": surface_debug,
        "mm_per_pixel": mm_per_pixel,
        "post_reslice_resize": False,
    }
    return cbct, geometry, debug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--placement", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--us-spacing-mm", type=float, default=US_SPACING_MM)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sequence = np.load(args.sequence, allow_pickle=True)
    us = np.asarray(sequence["images"][args.frame])

    report = json.loads(args.report.read_text(encoding="utf-8"))
    affine = np.asarray(
        report["phantom_centered_frame"]["recon_affine_centered_from_ijk_mm"], dtype=float
    )
    volume = load_volume_data(args.volume)
    world_from_phantom = load_transform_4x4(args.placement)
    world_from_ee = np.asarray(sequence["poses"][args.frame], dtype=float)
    world_from_probe = world_from_ee @ P.T_EE_FROM_PROBE
    phantom_from_probe_mm = P.probe_pose_in_phantom_centered_mm(
        world_from_probe, world_from_phantom
    )
    cbct, geometry, debug = cbct_bottom_tangent_reslice(
        volume,
        affine,
        phantom_from_probe_mm,
        us,
        inner_arc_threshold=args.threshold,
        us_spacing_mm=args.us_spacing_mm,
    )
    mm_per_pixel = float(debug["mm_per_pixel"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inner, outer = boundary_curves(geometry)
    p_left = np.asarray(geometry["p_left"])
    p_right = np.asarray(geometry["p_right"])
    side_left = np.asarray(geometry["side_left"])
    side_right = np.asarray(geometry["side_right"])
    outer_left = np.asarray(geometry["outer_left"])
    outer_right = np.asarray(geometry["outer_right"])
    bottom_tangent = np.asarray(geometry["bottom_tangent"])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.45), squeeze=False)
    ax_us, ax_cbct = axes[0]
    ax_us.imshow(_normalize_display(us), cmap="gray", vmin=0, vmax=1)
    ax_us.scatter(geometry["edge_x"], geometry["edge_y"], s=2, c="#00e5ff", alpha=0.6)
    ax_us.plot(inner[:, 0], inner[:, 1], color="#00e5ff", lw=1.8, label="fitted inner arc")
    ax_us.plot(outer[:, 0], outer[:, 1], color="#7cff00", lw=1.8, label="parallel outer arc")
    ax_us.plot(
        [p_left[0], outer_left[0]], [p_left[1], outer_left[1]], color="#ff3bd4", lw=1.8
    )
    ax_us.plot(
        [p_right[0], outer_right[0]], [p_right[1], outer_right[1]], color="#ff3bd4", lw=1.8
    )
    ax_us.scatter(
        [p_left[0], p_right[0], side_left[0], side_right[0], bottom_tangent[0]],
        [p_left[1], p_right[1], side_left[1], side_right[1], bottom_tangent[1]],
        s=30,
        c="#ffd400",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    ax_us.set_title(f"Real US geometry | {args.sequence.stem} frame {args.frame}")

    ax_cbct.imshow(cbct, cmap="gray", vmin=0, vmax=1)
    ax_cbct.plot(inner[:, 0], inner[:, 1], color="#00e5ff", lw=1.5)
    ax_cbct.plot(outer[:, 0], outer[:, 1], color="#7cff00", lw=1.5)
    ax_cbct.plot(
        [p_left[0], outer_left[0]], [p_left[1], outer_left[1]], color="#ff3bd4", lw=1.5
    )
    ax_cbct.plot(
        [p_right[0], outer_right[0]], [p_right[1], outer_right[1]], color="#ff3bd4", lw=1.5
    )
    ax_cbct.scatter(
        [bottom_tangent[0]],
        [bottom_tangent[1]],
        s=30,
        c="#ffd400",
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    ax_cbct.set_title(
        f"CBCT direct isotropic sample | {mm_per_pixel:.4f} mm/px | no resize/stretch"
    )

    for ax in (ax_us, ax_cbct):
        ax.set_xlim(-0.5, us.shape[1] - 0.5)
        ax.set_ylim(us.shape[0] - 0.5, -0.5)
        ax.axis("off")
    centre = np.asarray(geometry["centre"])
    fig.suptitle(
        "US inner arc -> perpendicular normals -> bottom-tangent concentric outer arc\n"
        f"center=({centre[0]:.2f}, {centre[1]:.2f}) px, "
        f"r_inner={float(geometry['inner_radius']):.2f}px, "
        f"r_outer={float(geometry['outer_radius']):.2f}px, "
        f"fit median error={float(geometry['median_residual']):.2f}px",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    payload = {
        "sequence": str(args.sequence),
        "frame": args.frame,
        "shape": list(us.shape[:2]),
        "centre_xy_px": np.asarray(geometry["centre"]).tolist(),
        "inner_radius_px": float(geometry["inner_radius"]),
        "outer_radius_px": float(geometry["outer_radius"]),
        "inner_top_left_xy_px": p_left.tolist(),
        "inner_top_right_xy_px": p_right.tolist(),
        "side_left_xy_px": side_left.tolist(),
        "side_right_xy_px": side_right.tolist(),
        "true_outer_left_xy_px": outer_left.tolist(),
        "true_outer_right_xy_px": outer_right.tolist(),
        "bottom_tangent_xy_px": bottom_tangent.tolist(),
        "median_fit_residual_px": float(geometry["median_residual"]),
        **debug,
    }
    print(json.dumps(payload, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
