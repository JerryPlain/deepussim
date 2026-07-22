"""Curvilinear (convex) fan reslice — the sampling layout the US comparison needs.

The rectangular reslice in :mod:`reslice.sampling` is for inspection. LC2 scoring instead
needs the CBCT sampled on the *same fan* as the ultrasound, so it lines up with the
unwrapped US scan-line image. This module lays out the convex fan in the probe frame and
reuses the identical "plane point -> voxel -> trilinear sample" core.

Probe frame convention: origin at the transducer face centre, +x lateral, +z axial (into
tissue). Scan lines fan out from a virtual apex at ``z = -radius_mm`` and are sampled from
the face (radial distance ``radius_mm``) out to ``radius_mm + depth_mm``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class ProbeGeometry:
    """Convex probe geometry; fit it from the real US with the project tooling."""

    radius_mm: float = 60.0   # virtual apex -> transducer face
    fov_deg: float = 60.0     # total sector angle
    depth_mm: float = 120.0   # imaging depth along each beam, from the face outward
    n_lat: int = 256          # scan lines (image width)
    n_ax: int = 512           # samples per scan line (image height)

    def plane_grid(self) -> np.ndarray:
        """Homogeneous probe-frame coordinates of every fan pixel, shape ``(4, n_ax*n_lat)``.

        Row index runs along the beam (depth), column index across scan lines, matching the
        ``(n_ax, n_lat)`` image that :func:`reslice_fan` returns.
        """
        thetas = np.deg2rad(np.linspace(-self.fov_deg / 2.0, self.fov_deg / 2.0, self.n_lat))
        s = self.radius_mm + np.linspace(0.0, self.depth_mm, self.n_ax)   # radial dist from apex
        ss, th = np.meshgrid(s, thetas, indexing="ij")                    # (n_ax, n_lat)
        x = ss * np.sin(th)
        z = ss * np.cos(th) - self.radius_mm                              # apex at z = -radius
        return np.stack([x.ravel(), np.zeros(x.size), z.ravel(), np.ones(x.size)], axis=0)


def reslice_fan(
    data: np.ndarray,
    affine_centered_from_ijk_mm: np.ndarray,
    T_phantom_from_probe_mm: np.ndarray,
    geom: ProbeGeometry,
    order: int = 1,
) -> np.ndarray:
    """Sample the imaging fan out of ``data``; returns an ``(n_ax, n_lat)`` image.

    Same frame as the rectangular reslice: ``T_phantom_from_probe_mm`` is the probe pose in
    phantom-centred mm and ``affine_centered_from_ijk_mm`` maps voxels to that same frame.
    """
    pts_probe = geom.plane_grid()                                  # (4, N) in probe frame
    pts_phantom = np.asarray(T_phantom_from_probe_mm, dtype=float) @ pts_probe
    vox = np.linalg.inv(np.asarray(affine_centered_from_ijk_mm, dtype=float)) @ pts_phantom
    cval = float(np.min(data))
    sampled = map_coordinates(data, vox[:3], order=order, mode="constant", cval=cval)
    return sampled.reshape(geom.n_ax, geom.n_lat)


def scan_convert_fan(
    polar: np.ndarray,
    output_shape: tuple[int, int],
    *,
    apex_px: tuple[float, float],
    r0_px: float,
    r1_px: float,
    fov_deg: float,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Map a polar fan image back to the fixed B-mode display pixel grid.

    ``polar`` follows :func:`reslice_fan`'s ``(depth, scan-line)`` layout.  The display
    geometry is measured from the real B-mode image: ``apex_px`` is ``(x, y)`` of the
    virtual apex, ``r0_px`` / ``r1_px`` are the face and far radii, and ``fov_deg`` is
    the total fan angle.  Mapping directly into ``output_shape`` preserves the scanner's
    pixel geometry; unlike a fan-bounding-box resize, it never stretches rows and columns
    by different factors.

    Use ``order=1`` for intensity and ``order=0`` for discrete labels.  Nearest-neighbour
    conversion returns the input dtype so label ids remain exact.
    """
    from scipy.ndimage import map_coordinates

    source = np.asarray(polar)
    if source.ndim != 2:
        raise ValueError(f"polar fan must be 2D, got shape {source.shape}")
    if r1_px <= r0_px:
        raise ValueError(f"r1_px must exceed r0_px, got {r0_px} and {r1_px}")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    if order not in (0, 1):
        raise ValueError(f"scan conversion supports order 0 or 1, got {order}")

    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"output_shape must be positive, got {output_shape}")

    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    apex_x, apex_y = float(apex_px[0]), float(apex_px[1])
    dx = xx - apex_x
    dy = yy - apex_y
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dx, dy)
    half_angle = np.deg2rad(float(fov_deg)) / 2.0

    polar_row = (radius - float(r0_px)) / float(r1_px - r0_px) * (source.shape[0] - 1)
    polar_col = (theta + half_angle) / (2.0 * half_angle) * (source.shape[1] - 1)
    valid = (
        (radius >= float(r0_px))
        & (radius <= float(r1_px))
        & (np.abs(theta) <= half_angle)
    )

    sample_source = source if order == 0 else source.astype(np.float32, copy=False)
    converted = map_coordinates(
        sample_source,
        [polar_row.ravel(), polar_col.ravel()],
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    ).reshape(height, width)
    converted[~valid] = cval

    if order == 0:
        return converted.astype(source.dtype, copy=False)
    return converted.astype(np.float32, copy=False)


def fraction_inside(fan_image: np.ndarray, cval: float | None = None) -> float:
    """Fraction of fan pixels that fell inside the volume (anti-graze guard for LC2)."""
    fan = np.asarray(fan_image, dtype=float)
    floor = float(np.min(fan)) if cval is None else cval
    return float((fan > floor + 1e-6).mean())
