"""The real US display geometry: convex-sector fan in B-mode pixel coordinates.

This is the single authoritative description of *where the US image lives*. Anything that
wants to draw CBCT (or labels) on top of a US frame must go through here, so that the result
is pixel-aligned by construction rather than by resizing.

Conventions (identical to :func:`lc2.us_fan.unwrap_fan` and :meth:`reslice.fan.ProbeGeometry.plane_grid`)::

    col = apex_x + r_px * sin(theta)          display pixels, theta measured from +row (down)
    row = apex_y + r_px * cos(theta)
    x_probe = s_mm * sin(theta)               probe frame, mm
    z_probe = s_mm * cos(theta) - radius_mm   probe origin at the face centre, apex at -radius

Both decompositions share the apex and the same ``(sin, cos)`` split, so eliminating
``theta`` collapses display -> probe into a pure scale-and-translate::

    x_probe = (col - apex_x) * spacing_mm
    z_probe = (row - apex_y) * spacing_mm - radius_mm

No trigonometry is involved in the mapping; the fan angles/radii only bound the *valid*
region. That is why :meth:`UsDisplayFan.pixel_grid_probe_mm` is exact and cheap.

Fit this once per probe with ``calib/fit_us_fan.py`` and load the JSON it writes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_FAN_JSON = Path(__file__).resolve().parents[1] / "calib" / "us_fan.json"


def render_on_us_grid(
    volume: np.ndarray,
    affine_centered_from_ijk_mm: np.ndarray,
    T_phantom_from_probe_mm: np.ndarray,
    fan: "UsDisplayFan",
    *,
    order: int = 1,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``volume`` directly onto the real US display grid. Pixel-aligned by construction.

    Every US pixel ``(row, col)`` is mapped to probe-frame mm via ``fan.pixel_grid_probe_mm``,
    then to phantom mm by the pose, then to a fractional voxel by the inverse affine, and
    sampled with ``map_coordinates`` (``order=1`` intensity, ``order=0`` labels). Pixels outside
    the fan sector or outside the volume are set to ``fill``.

    Returns ``(image, valid)`` where ``image`` has ``fan.shape`` and ``valid`` marks pixels that
    are both inside the fan sector and inside the volume bounds. There is NO crop/resize: the
    output already lives in US pixel coordinates, which is the whole point of this path.
    """
    from scipy.ndimage import map_coordinates

    volume = np.asarray(volume)
    grid_probe = fan.pixel_grid_probe_mm()                              # (4, rows*cols)
    pts_phantom = np.asarray(T_phantom_from_probe_mm, dtype=float) @ grid_probe
    vox = np.linalg.inv(np.asarray(affine_centered_from_ijk_mm, dtype=float)) @ pts_phantom

    inside = np.ones(vox.shape[1], dtype=bool)
    for axis, size in enumerate(volume.shape):
        inside &= (vox[axis] >= 0.0) & (vox[axis] <= size - 1)

    cval = float(np.min(volume))
    sampled = map_coordinates(volume, vox[:3], order=order, mode="constant", cval=cval)

    support = fan.support_mask().ravel()
    valid = inside & support
    out = np.where(valid, sampled, float(fill))
    return out.reshape(fan.shape), valid.reshape(fan.shape)


@dataclass(frozen=True)
class UsDisplayFan:
    """Convex sector of a B-mode frame, in display pixels plus the mm scale."""

    apex_x_px: float          # virtual apex, column coordinate
    apex_y_px: float          # virtual apex, row coordinate (negative = above the frame)
    r0_px: float              # apex -> transducer-face arc
    r1_px: float              # apex -> outer (depth) arc
    fov_deg: float            # total sector angle
    spacing_mm: float         # mm per display pixel (isotropic)
    rows: int                 # frame height the fit is valid for
    cols: int                 # frame width
    resid_px: float = 0.0     # side-line fit residual, for provenance
    source: str = ""          # which sequence//envelope this was fitted from

    # -- derived physical quantities -----------------------------------------------------
    @property
    def radius_mm(self) -> float:
        """Virtual apex -> transducer face. This is ``ProbeGeometry.radius_mm``."""
        return self.r0_px * self.spacing_mm

    @property
    def depth_mm(self) -> float:
        """Imaging depth along each beam, from the face outward."""
        return (self.r1_px - self.r0_px) * self.spacing_mm

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def to_probe_geometry(self, n_lat: int = 256, n_ax: int = 512):
        """The equivalent polar :class:`reslice.fan.ProbeGeometry` used by LC2 scoring."""
        from reslice.fan import ProbeGeometry

        return ProbeGeometry(radius_mm=self.radius_mm, fov_deg=self.fov_deg,
                             depth_mm=self.depth_mm, n_lat=n_lat, n_ax=n_ax)

    # -- geometry ------------------------------------------------------------------------
    def polar_of_pixels(self, shape: tuple[int, int] | None = None):
        """``(r_px, theta_deg)`` of every display pixel."""
        rows, cols = shape or self.shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        dx = cc - self.apex_x_px
        dy = rr - self.apex_y_px
        return np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy))

    def support_mask(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Boolean mask of the pixels the probe actually images."""
        r, theta = self.polar_of_pixels(shape)
        return (r >= self.r0_px) & (r <= self.r1_px) & (np.abs(theta) <= self.fov_deg / 2.0)

    def pixel_grid_probe_mm(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Homogeneous probe-frame mm coordinates of every display pixel, shape ``(4, rows*cols)``.

        The imaging plane is the probe's x-z plane (y = 0), matching ``plane_grid``. Feed the
        result through ``T_phantom_from_probe_mm`` to land in phantom-centred mm.
        """
        rows, cols = shape or self.shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        x = (cc - self.apex_x_px) * self.spacing_mm
        z = (rr - self.apex_y_px) * self.spacing_mm - self.radius_mm
        return np.stack([x.ravel(), np.zeros(x.size), z.ravel(), np.ones(x.size)], axis=0)

    # -- persistence ---------------------------------------------------------------------
    def save(self, path: Path | str = DEFAULT_FAN_JSON) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_FAN_JSON) -> "UsDisplayFan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def from_fan_fit(cls, fit, spacing_mm: float, shape: tuple[int, int],
                     source: str = "") -> "UsDisplayFan":
        """Adapt a :class:`lc2.us_fan.FanFit` (pixel-space fit) into a display fan."""
        return cls(
            apex_x_px=float(fit.apex_px[0]), apex_y_px=float(fit.apex_px[1]),
            r0_px=float(fit.r0_px), r1_px=float(fit.r1_px), fov_deg=float(fit.fov_deg),
            spacing_mm=float(spacing_mm), rows=int(shape[0]), cols=int(shape[1]),
            resid_px=float(getattr(fit, "resid_px", 0.0)), source=source,
        )
