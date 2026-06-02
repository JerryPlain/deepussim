"""Fan-geometry fit: recover a known convex sector from a rasterized B-mode image."""
import numpy as np
import pytest

from deepussim.calib.us_geometry import (
    FanFit, contact_envelope, fit_fan_pixels, fit_fan_geometry, unwrap_fan,
)


def _render_fan(apex, r0_px, r1_px, fov_deg, H=660, W=880):
    """Rasterize a solid convex sector (apex above the frame) into a uint8 image."""
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = xx - apex[0], yy - apex[1]
    r = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dx, dy))                 # angle from +y (down) axis
    mask = (r >= r0_px) & (r <= r1_px) & (np.abs(th) <= fov_deg / 2.0)
    return (mask * 255).astype(np.uint8)


def test_fit_fan_pixels_recovers_known_sector():
    apex, r0, r1, fov = (440.0, -270.0), 316.0, 927.0, 68.0
    img = _render_fan(apex, r0, r1, fov)
    fit = fit_fan_pixels(img)
    assert fit.apex_px[0] == pytest.approx(apex[0], abs=8)
    assert fit.apex_px[1] == pytest.approx(apex[1], abs=15)
    assert fit.r0_px == pytest.approx(r0, abs=12)
    assert fit.r1_px == pytest.approx(r1, abs=12)
    assert fit.fov_deg == pytest.approx(fov, abs=3)
    assert fit.resid_px < 3.0


def test_fit_fan_geometry_applies_mm_scale():
    spacing = 0.166112957
    apex, r0, r1, fov = (440.0, -270.0), 316.0, 927.0, 68.0
    geom = fit_fan_geometry(_render_fan(apex, r0, r1, fov), spacing, n_lat=200, n_ax=400)
    assert geom.radius_mm == pytest.approx(r0 * spacing, abs=2.5)
    assert geom.depth_mm == pytest.approx((r1 - r0) * spacing, abs=2.5)
    assert geom.fov_deg == pytest.approx(fov, abs=3)
    assert (geom.n_lat, geom.n_ax) == (200, 400)        # sampling resolution is pass-through


def test_contact_envelope_is_max_over_selected_frames():
    a = np.zeros((3, 4, 4), np.uint8)
    a[0, 0, 0] = 200; a[1, 1, 1] = 100; a[2] = 50       # frame 2 uniformly 50
    env = contact_envelope(a, contact=[True, True, False])
    assert env[0, 0] == 200 and env[1, 1] == 100 and env[3, 3] == 0  # frame 2 excluded


def test_degenerate_fan_raises():
    with pytest.raises(ValueError):
        fit_fan_pixels(np.zeros((660, 880), np.uint8))   # empty mask -> no fan


def test_unwrap_fan_maps_depth_to_radius():
    # a display image whose value == radius from the apex; after unwrap each scan-line column
    # must rise monotonically from r0 (face) to r1 (depth limit).
    fan = FanFit(apex_px=(440.0, -270.0), r0_px=316.0, r1_px=927.0, fov_deg=68.0, resid_px=0.0)
    H, W = 660, 880
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.hypot(xx - fan.apex_px[0], yy - fan.apex_px[1])     # pixel value = radius
    pol = unwrap_fan(img, fan, n_ax=100, n_lat=40)
    col = pol[:, 20]                                             # a central scan line
    assert col[0] == pytest.approx(316.0, abs=5)
    assert col[-1] == pytest.approx(927.0, abs=5)
    assert np.all(np.diff(col) > 0)
