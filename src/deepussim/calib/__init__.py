"""Calibration: Step 3 (coordinate registration) and Step 4 (renderer fitting)."""

from .registration import rigid_register, build_world_to_cbct
from .renderer_fit import ncc, render_loss, fit_renderer
from .placement import (
    meters_to_mm,
    mm_to_meters,
    align_points_placement,
    align_centers_placement,
    sim_pose_to_cbct,
    seat_phantom_placement,
)
from .transforms import (
    T_PROBE_FROM_EE,
    T_EE_FROM_PROBE,
    T_PHANTOM_FROM_ROBOT,
    T_ROBOT_FROM_PHANTOM,
    T_WORLD_FROM_CBCT,
)
from .us_geometry import (
    FanFit, contact_envelope, fit_fan_pixels, fit_fan_geometry, unwrap_fan,
)
from .lc2 import lc2_similarity, lc2_map, gradient_magnitude, register_frame_lc2

__all__ = [
    "rigid_register",
    "build_world_to_cbct",
    "ncc",
    "render_loss",
    "fit_renderer",
    "FanFit",
    "contact_envelope",
    "fit_fan_pixels",
    "fit_fan_geometry",
    "unwrap_fan",
    "lc2_similarity",
    "lc2_map",
    "gradient_magnitude",
    "register_frame_lc2",
    "meters_to_mm",
    "mm_to_meters",
    "align_points_placement",
    "align_centers_placement",
    "sim_pose_to_cbct",
    "seat_phantom_placement",
    "T_PROBE_FROM_EE",
    "T_EE_FROM_PROBE",
    "T_PHANTOM_FROM_ROBOT",
    "T_ROBOT_FROM_PHANTOM",
    "T_WORLD_FROM_CBCT",
]
