"""Step 4: turn the rectangular CBCT plane into an ultrasound-style fan sector.

Pipeline:
1. detect the first non-background phantom row (the surface);
2. project the probe pose into the displayed image and intersect its beam line with
   that surface edge -> the fan apex;
3. mask the plane to a fan (apex, depth, field-of-view) so it matches the US sector;
4. optionally crop to the fan and zoom to the US image size for a side-by-side compare.

All coordinates here are display-image pixels (row down, col right).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Robust 0.5..99.5 percentile stretch to ``[0, 1]`` float32 (ignores non-finite)."""
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
    """Return the first row with at least ``min_pixels`` above-background pixels.

    Returns ``(top_row, threshold, per_row_counts)``. When ``threshold`` is None it is
    set to ``background + 50`` from the valid image values.
    """
    finite = np.isfinite(image) & valid_mask
    if threshold is None:
        values = image[finite]
        threshold = 0.0 if values.size == 0 else float(np.nanmin(values)) + 50.0

    row_counts = np.sum(finite & (image > threshold), axis=1)
    hits = np.flatnonzero(row_counts >= min_pixels)
    top_row = int(hits[0]) if hits.size else 0
    return top_row, float(threshold), row_counts.astype(int).tolist()


def top_edge_rows_by_col(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """For each column, the row of the topmost above-threshold pixel (NaN if none)."""
    content = np.isfinite(image) & valid_mask & (image > threshold)
    _, cols = image.shape
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
    """Project a phantom-mm point onto the displayed plane image, returning ``[row, col]``.

    Inverse of the grid in :func:`reslice.sampling.reslice_rectangular_plane`: take the point's
    offset from the plane centre, split it into lateral / axial mm, then convert mm to pixels
    using the same width / height extents (note the matching ``-W/2`` and ``+0.25*H`` offsets).
    """
    delta = np.asarray(point_mm, dtype=np.float64) - plane["point_centered_mm"]
    u_mm = float(np.dot(delta, lateral_sign * plane["lateral_centered"]))   # lateral offset (mm)
    v_mm = float(np.dot(delta, axial_sign * plane["axial_centered"]))       # axial/depth offset (mm)
    col = (u_mm + width_mm / 2.0) / (width_mm / max(cols - 1, 1))           # mm -> column
    row = (v_mm + height_mm * 0.25) / (height_mm / max(rows - 1, 1))        # mm -> row
    if display_rot180:                                                      # match the 180-rotated display
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
    """Project a phantom-mm direction onto the displayed plane as a unit ``[row, col]`` vector.

    A direction has no origin, so we only take its components along the axial (row) and lateral
    (col) plane axes — this is the probe beam direction expressed in image pixels.
    """
    direction = np.asarray(direction_mm, dtype=np.float64)
    rc = np.array(
        [
            float(np.dot(direction, axial_sign * plane["axial_centered"])),    # row component
            float(np.dot(direction, lateral_sign * plane["lateral_centered"])),  # col component
        ],
        dtype=np.float64,
    )
    if display_rot180:                          # the 180-rotated display flips both axes
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
    """Fan apex = where the probe beam line crosses the detected surface edge.

    Falls back to the probe pixel if no edge is found, or to the nearest edge point to
    the beam line if none lies within ``max_line_distance_px``.
    """
    edge_rows = top_edge_rows_by_col(image, valid_mask, threshold)
    cols = np.flatnonzero(np.isfinite(edge_rows))
    if cols.size == 0:
        return probe_pixel_rc.copy(), {
            "method": "fallback_probe_pixel_no_edge",
            "max_line_distance_px": float(max_line_distance_px),
        }

    edge_pts = np.column_stack([edge_rows[cols], cols.astype(np.float64)])
    line_dir = np.asarray(depth_direction_rc, dtype=np.float64)
    norm = float(np.linalg.norm(line_dir))
    line_dir = np.array([1.0, 0.0]) if norm == 0.0 else line_dir / norm

    # Distance of each edge point to the beam line through the probe pixel.
    rel = edge_pts - probe_pixel_rc[None, :]
    t = rel @ line_dir
    closest = probe_pixel_rc[None, :] + t[:, None] * line_dir[None, :]
    dist = np.linalg.norm(edge_pts - closest, axis=1)

    candidates = np.flatnonzero(dist <= max_line_distance_px)
    if candidates.size:
        # Prefer the surface crossing in the forward (depth) direction.
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
    """Boolean fan mask: apex + arbitrary 2D beam direction + FOV + near/far radius."""
    rows, cols = shape
    rr, cc = np.indices((rows, cols), dtype=np.float32)

    # Pixel offsets from the apex, in metric display-plane mm.
    offsets = np.stack(
        ((rr - float(apex_row)) * float(mm_per_row), (cc - float(apex_col)) * float(mm_per_col)),
        axis=-1,
    )

    depth_axis = np.asarray(depth_direction, dtype=np.float32)
    norm = float(np.linalg.norm(depth_axis))
    depth_axis = np.array([1.0, 0.0], dtype=np.float32) if norm == 0.0 else depth_axis / norm
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
    """Mask the rectangular plane down to the US fan; returns ``(sector, mask, debug)``.

    With ``apex_pixel_rc`` None, the apex falls back to the content-top centre (legacy);
    otherwise it uses the supplied pose/edge apex and beam direction.
    """
    rows, cols = rect.shape
    top_row, threshold, row_counts = detect_content_top_row(
        rect, valid_mask, threshold=content_threshold, min_pixels=content_min_pixels
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
        rect.shape, apex_row, apex_col, depth_direction,
        fov_deg, depth_mm, near_mm, mm_per_row, mm_per_col,
    )
    mask &= valid_mask

    sector = np.full_like(rect, np.nan, dtype=np.float32)
    sector[mask] = rect[mask]
    debug = {
        "content_top_row": int(top_row),
        "content_threshold": float(threshold),
        "apex_row": float(apex_row),
        "apex_col": float(apex_col),
        "apex_source": apex_source,
        "depth_direction_rc": depth_direction.astype(float).tolist(),
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
    order: int = 1,
) -> tuple[np.ndarray, dict]:
    """Crop to the fan bounding box (+margin) and resize to ``target_shape`` (US size).

    ``order=1`` (bilinear) for an intensity sector, ``order=0`` (nearest) for a label
    sector so class ids are never blended by the resize.
    """
    hits = np.argwhere(crop_mask)
    if hits.size == 0:
        return np.zeros(target_shape, dtype=np.float32), {
            "crop_bbox_rc": None, "crop_margin_px": int(margin_px),
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
    zoomed = ndimage.zoom(crop, zoom, order=order)[: target_shape[0], : target_shape[1]]
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
