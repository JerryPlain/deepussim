"""Reslice a 3D volume along a US imaging plane (the geometric half of Step 5).

Probe / image-plane frame convention (right-handed):
  - origin at the centre of the probe face,
  - +x = lateral  (image width / scan-line direction),
  - +y = elevation (out-of-plane thickness; 0 for a thin 2D slice),
  - +z = axial    (imaging depth, into the tissue).

Given ``T_world_from_probe`` we lay out a regular grid on the (x, z) plane, map it
into world millimetres, then into fractional voxel indices, and trilinearly sample
the volume with ``scipy.ndimage.map_coordinates``.

Use ``order=1`` (trilinear) for the intensity volume and ``order=0`` (nearest) for a
label volume so anatomy classes are never blended.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from ..data.volume import Volume


@dataclass
class ProbeGeometry:
    """Linear-probe imaging geometry (physical extent + pixel sampling)."""

    width_mm: float = 40.0
    depth_mm: float = 80.0
    n_lat: int = 256
    n_ax: int = 512

    def axial_depths_mm(self) -> np.ndarray:
        """Depth (mm) of each image row, from the probe face into tissue."""
        return np.linspace(0.0, self.depth_mm, self.n_ax)

    def plane_grid(self) -> np.ndarray:
        """Homogeneous plane-frame coordinates of every pixel, shape (4, n_ax*n_lat)."""
        xs = np.linspace(-self.width_mm / 2.0, self.width_mm / 2.0, self.n_lat)
        zs = self.axial_depths_mm()
        zz, xx = np.meshgrid(zs, xs, indexing="ij")  # both (n_ax, n_lat)
        flat = np.stack(
            [xx.ravel(), np.zeros(xx.size), zz.ravel(), np.ones(xx.size)], axis=0
        )
        return flat  # (4, N)


def reslice(
    data: np.ndarray,
    affine: np.ndarray,
    T_world_from_probe: np.ndarray,
    geom: ProbeGeometry,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Sample the imaging plane out of ``data``; returns (n_ax, n_lat)."""
    pts_plane = geom.plane_grid()  # (4, N) in probe frame
    pts_world = np.asarray(T_world_from_probe, dtype=float) @ pts_plane  # (4, N)
    vox = np.linalg.inv(np.asarray(affine, dtype=float)) @ pts_world  # (4, N)
    coords = vox[:3]  # (3, N) -> (i, j, k)
    sampled = map_coordinates(data, coords, order=order, cval=cval, mode="constant")
    return sampled.reshape(geom.n_ax, geom.n_lat)


def reslice_volume(
    volume: Volume,
    T_world_from_probe: np.ndarray,
    geom: ProbeGeometry,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper around :func:`reslice` for a :class:`Volume`."""
    return reslice(volume.data, volume.affine, T_world_from_probe, geom, order, cval)
