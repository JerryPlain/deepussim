"""Calibration: Step 3 (coordinate registration) and Step 4 (renderer fitting)."""

from .registration import rigid_register, build_world_to_cbct
from .renderer_fit import ncc, render_loss, fit_renderer
from .placement import (
    meters_to_mm,
    align_points_placement,
    align_centers_placement,
    sim_pose_to_cbct,
)

__all__ = [
    "rigid_register",
    "build_world_to_cbct",
    "ncc",
    "render_loss",
    "fit_renderer",
    "meters_to_mm",
    "align_points_placement",
    "align_centers_placement",
    "sim_pose_to_cbct",
]
