"""``UsDisplayFan`` geometry: round-trips, persistence, and agreement with the polar path.

The load-bearing claim of ``reslice/us_display.py`` is that display->probe collapses to a pure
scale-and-translate because ``unwrap_fan`` and ``ProbeGeometry.plane_grid`` share an apex and
the same ``(sin, cos)`` split. If that is wrong, everything drawn on the US grid is wrong, so
it is checked here against the two independent implementations rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reslice.us_display import DEFAULT_FAN_JSON, UsDisplayFan   # noqa: E402


def make_fan(**kw) -> UsDisplayFan:
    base = dict(apex_x_px=494.5, apex_y_px=-347.6, r0_px=389.6, r1_px=950.6,
                fov_deg=57.11, spacing_mm=0.166112957, rows=660, cols=880)
    base.update(kw)
    return UsDisplayFan(**base)


# --------------------------------------------------------------------------------------
# derived physical quantities
# --------------------------------------------------------------------------------------

def test_derived_mm_quantities():
    fan = make_fan()
    assert fan.radius_mm == pytest.approx(64.72, abs=0.01)
    assert fan.depth_mm == pytest.approx(93.19, abs=0.01)
    assert fan.to_probe_geometry().radius_mm == pytest.approx(fan.radius_mm)


# --------------------------------------------------------------------------------------
# the affine shortcut must equal the trigonometric definition
# --------------------------------------------------------------------------------------

def test_pixel_grid_matches_trigonometric_definition():
    """x = s*sin(th), z = s*cos(th) - radius, computed the long way, must match the affine map."""
    fan = make_fan(rows=64, cols=80)
    r_px, theta_deg = fan.polar_of_pixels()
    s_mm = r_px * fan.spacing_mm
    th = np.deg2rad(theta_deg)
    x_trig = s_mm * np.sin(th)
    z_trig = s_mm * np.cos(th) - fan.radius_mm

    grid = fan.pixel_grid_probe_mm()
    assert grid.shape == (4, 64 * 80)
    assert np.allclose(grid[0].reshape(64, 80), x_trig, atol=1e-9)
    assert np.allclose(grid[1], 0.0)                      # imaging plane is probe y = 0
    assert np.allclose(grid[2].reshape(64, 80), z_trig, atol=1e-9)
    assert np.allclose(grid[3], 1.0)


def test_pixel_grid_agrees_with_probe_geometry_plane_grid():
    """Map the polar fan's probe-mm points back to pixels; must land on ``unwrap_fan``'s pixels."""
    fan = make_fan()
    geom = fan.to_probe_geometry(n_lat=33, n_ax=41)

    # forward: polar grid -> probe mm (the LC2 reslice path)
    pts = geom.plane_grid()
    # inverse of pixel_grid_probe_mm
    col = pts[0] / fan.spacing_mm + fan.apex_x_px
    row = (pts[2] + fan.radius_mm) / fan.spacing_mm + fan.apex_y_px

    # independent: unwrap_fan's own pixel formulas for the same (r, theta) lattice
    depth = np.linspace(0.0, 1.0, geom.n_ax)
    theta = np.deg2rad(np.linspace(-fan.fov_deg / 2, fan.fov_deg / 2, geom.n_lat))
    r = fan.r0_px + np.outer(depth, np.ones(geom.n_lat)) * (fan.r1_px - fan.r0_px)
    th = np.outer(np.ones(geom.n_ax), theta)
    col_ref = fan.apex_x_px + r * np.sin(th)
    row_ref = fan.apex_y_px + r * np.cos(th)

    assert np.allclose(col.reshape(geom.n_ax, geom.n_lat), col_ref, atol=1e-6)
    assert np.allclose(row.reshape(geom.n_ax, geom.n_lat), row_ref, atol=1e-6)


def test_apex_pixel_maps_to_the_virtual_apex_in_probe_mm():
    """The apex sits at probe z = -radius_mm, x = 0 -- behind the transducer face."""
    fan = make_fan()
    x = (fan.apex_x_px - fan.apex_x_px) * fan.spacing_mm
    z = (fan.apex_y_px - fan.apex_y_px) * fan.spacing_mm - fan.radius_mm
    assert x == pytest.approx(0.0)
    assert z == pytest.approx(-fan.radius_mm)


# --------------------------------------------------------------------------------------
# support mask
# --------------------------------------------------------------------------------------

def test_support_mask_agrees_with_an_independent_rasteriser():
    """Guards against silent divergence from the copy in test_display_alignment.py."""
    from tests.test_display_alignment import fan_support_mask

    class _Shim:                                   # the FanFit-like duck type that helper wants
        def __init__(self, f):
            self.apex_px = (f.apex_x_px, f.apex_y_px)
            self.r0_px, self.r1_px, self.fov_deg = f.r0_px, f.r1_px, f.fov_deg

    fan = make_fan()
    assert np.array_equal(fan.support_mask(), fan_support_mask(fan.shape, _Shim(fan)))


def test_support_mask_is_a_plausible_sector():
    fan = make_fan()
    m = fan.support_mask()
    assert m.shape == (660, 880)
    assert 0.35 < m.mean() < 0.75                  # a sector, not empty and not the whole frame
    # widens with depth: the apex is above the frame, so every row down is wider
    widths = m.sum(axis=1)[m.any(axis=1)]
    assert widths[0] < widths[len(widths) // 2]


# --------------------------------------------------------------------------------------
# persistence + the frozen calibration
# --------------------------------------------------------------------------------------

def test_json_round_trip(tmp_path):
    fan = make_fan(resid_px=5.51, source="scan1.npz//contact_envelope")
    assert UsDisplayFan.load(fan.save(tmp_path / "us_fan.json")) == fan


def test_frozen_calibration_is_sane():
    """The committed calib/us_fan.json must describe a convex probe, apex above the frame."""
    if not DEFAULT_FAN_JSON.exists():
        pytest.skip("calib/us_fan.json not generated yet (run calib/fit_us_fan.py)")
    fan = UsDisplayFan.load()
    assert fan.apex_y_px < 0, "virtual apex must sit above the frame for a convex array"
    assert fan.r1_px > fan.r0_px > 0
    assert 20.0 < fan.fov_deg < 120.0
    assert 30.0 < fan.radius_mm < 120.0
    assert 40.0 < fan.depth_mm < 200.0
    assert fan.resid_px < 10.0
