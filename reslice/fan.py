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


def fraction_inside(fan_image: np.ndarray, cval: float | None = None) -> float:
    """Fraction of fan pixels that fell inside the volume (anti-graze guard for LC2)."""
    fan = np.asarray(fan_image, dtype=float)
    floor = float(np.min(fan)) if cval is None else cval
    return float((fan > floor + 1e-6).mean())
