"""Reslice a 3D volume along the US imaging fan (the geometric half of Step 5).

The probe is curvilinear (convex) — see :class:`ProbeGeometry`. Given ``T_world_from_probe``
we lay out the fan of sample points in the probe frame, map them into world millimetres, then
into fractional voxel indices, and trilinearly sample the volume with
``scipy.ndimage.map_coordinates``.

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
    """Curvilinear (convex) probe geometry — the real DeepUSSim probe.

    Scan lines fan out from a virtual apex behind the transducer face. Frame convention
    (right-handed): origin at the face centre, +x lateral, +y elevation (0 for a thin 2D
    slice), +z axial (into tissue). The apex sits at ``z = -radius_mm``; scan lines span
    ``+/- fov_deg/2`` about it; each line is sampled from the face (radial distance
    ``radius_mm``) out to ``radius_mm + depth_mm``. The centre line (theta = 0) is a straight
    +z beam, so a pose places the face centre and aims the central beam into the tissue.
    """

    radius_mm: float = 60.0   # convex-array radius (virtual apex -> transducer face)
    fov_deg: float = 60.0     # total sector angle (lateral field of view)
    depth_mm: float = 120.0   # imaging depth along each beam, from the face outward
    n_lat: int = 256          # number of scan lines (image width, px)
    n_ax: int = 512           # samples per scan line (image height, px)

    def axial_depths_mm(self) -> np.ndarray:
        """Depth (mm) along each beam from the face — identical for every scan line."""
        return np.linspace(0.0, self.depth_mm, self.n_ax)

    def plane_grid(self) -> np.ndarray:
        """Homogeneous probe-frame coordinates of every pixel, shape (4, n_ax*n_lat).

        Row index = depth along the beam, column index = scan line — matching the (n_ax,
        n_lat) layout :func:`reslice` reshapes to.
        """
        thetas = np.deg2rad(np.linspace(-self.fov_deg / 2.0, self.fov_deg / 2.0, self.n_lat))
        s = self.radius_mm + self.axial_depths_mm()        # (n_ax,) radial dist from apex
        ss, th = np.meshgrid(s, thetas, indexing="ij")      # both (n_ax, n_lat)
        x = ss * np.sin(th)
        z = ss * np.cos(th) - self.radius_mm                # apex at z = -radius_mm
        flat = np.stack(
            [x.ravel(), np.zeros(x.size), z.ravel(), np.ones(x.size)], axis=0
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
    """Sample the imaging fan out of ``data``; returns (n_ax, n_lat)."""
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
