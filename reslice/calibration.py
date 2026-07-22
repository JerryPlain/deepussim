"""Fixed 2026-06-12 convex-probe calibration in physical and display coordinates.

The display fan was fitted once from the contact-frame maximum envelope of all 15
``scan*.bag`` acquisitions.  Keeping these values in one module prevents the renderer,
label projection and comparison plots from silently using different crop geometries.
"""
from __future__ import annotations

from .fan import ProbeGeometry


US_DISPLAY_SHAPE = (660, 880)
US_SPACING_MM = 0.166112957

# Pixel coordinates are (x, y).  The virtual apex is above the visible image.
US_DISPLAY_FAN = {
    "apex_px": (439.4999999999998, -261.90505706969174),
    "r0_px": 307.90505706969174,
    "r1_px": 918.9050570696918,
    "fov_deg": 69.31467507611598,
}

US_PROBE_GEOMETRY = ProbeGeometry(
    radius_mm=US_DISPLAY_FAN["r0_px"] * US_SPACING_MM,
    fov_deg=US_DISPLAY_FAN["fov_deg"],
    depth_mm=(US_DISPLAY_FAN["r1_px"] - US_DISPLAY_FAN["r0_px"]) * US_SPACING_MM,
    n_ax=512,
    n_lat=256,
)
