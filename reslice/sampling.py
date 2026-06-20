"""Step 2 (sampling half): cut a rectangular plane out of the 3D volume.

For each output pixel we (1) find its physical mm position on the plane, (2) convert
that to a fractional voxel index with the inverse affine, and (3) trilinearly sample
the volume there. The plane's vertical (axial) extent runs from ``-0.25*height`` to
``+0.75*height`` so the probe origin sits near the top quarter, matching the US layout.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def reslice_rectangular_plane(
    data: np.ndarray,
    affine_centered_from_ijk_mm: np.ndarray,
    plane: dict[str, np.ndarray],
    *,
    width_mm: float,
    height_mm: float,
    n_rows: int,
    n_cols: int,
    axial_sign: float,
    lateral_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a ``width_mm x height_mm`` plane into an ``(n_rows, n_cols)`` image.

    ``axial_sign`` / ``lateral_sign`` flip the in-plane axes to match the display
    convention. Returns ``(image, valid_mask)`` where ``valid_mask`` marks pixels that
    fell inside the volume bounds.
    """
    # Metric grid on the plane: columns span lateral, rows span axial (with the top-quarter offset).
    lateral_mm = np.linspace(-width_mm / 2.0, width_mm / 2.0, n_cols)
    axial_mm = np.linspace(-height_mm * 0.25, height_mm * 0.75, n_rows)
    uu, vv = np.meshgrid(lateral_mm, axial_mm)

    center = plane["point_centered_mm"]
    lateral = lateral_sign * plane["lateral_centered"]
    axial = axial_sign * plane["axial_centered"]

    # Each pixel's physical mm position on the plane.
    pts = (
        center[:, None]
        + lateral[:, None] * uu.ravel()[None, :]
        + axial[:, None] * vv.ravel()[None, :]
    )
    pts_h = np.vstack([pts, np.ones(pts.shape[1])])

    # mm -> fractional voxel index, then trilinear sample.
    vox = np.linalg.inv(affine_centered_from_ijk_mm) @ pts_h
    inside = np.ones(vox.shape[1], dtype=bool)
    for axis, size in enumerate(data.shape):
        inside &= (vox[axis] >= 0.0) & (vox[axis] <= size - 1)

    cval = float(np.min(data))
    sampled = map_coordinates(data, vox[:3], order=1, mode="constant", cval=cval)
    return sampled.reshape(n_rows, n_cols), inside.reshape(n_rows, n_cols)
