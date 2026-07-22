import numpy as np

from reslice.fan import scan_convert_fan
from renderer_training.project_labels_to_us import (
    CALIBRATED_DISPLAY_SHAPE,
    DEFAULT_DISPLAY_FAN,
    DEFAULT_PROBE_GEOMETRY,
    DEFAULT_US_SPACING_MM,
)


def test_scan_convert_label_preserves_shape_dtype_and_ids():
    polar = np.zeros((6, 5), dtype=np.int16)
    polar[:2] = 2
    polar[2:4] = 7
    polar[4:] = 11

    display = scan_convert_fan(
        polar,
        (31, 31),
        apex_px=(15.0, -10.0),
        r0_px=10.0,
        r1_px=30.0,
        fov_deg=60.0,
        order=0,
    )

    assert display.shape == (31, 31)
    assert display.dtype == polar.dtype
    assert set(np.unique(display)).issubset({0, 2, 7, 11})


def test_scan_convert_maps_near_and_far_centerline_without_resize():
    polar = np.broadcast_to(np.arange(1, 7, dtype=np.int16)[:, None], (6, 5)).copy()

    display = scan_convert_fan(
        polar,
        (31, 31),
        apex_px=(15.0, -10.0),
        r0_px=10.0,
        r1_px=30.0,
        fov_deg=60.0,
        order=0,
    )

    # On the centre scan line, y=0 is exactly r0 and y=20 is exactly r1.
    assert display[0, 15] == 1
    assert display[20, 15] == 6
    assert display[21, 15] == 0


def test_scan_convert_intensity_returns_float32():
    polar = np.linspace(0.0, 1.0, 30, dtype=np.float64).reshape(6, 5)
    display = scan_convert_fan(
        polar,
        (31, 31),
        apex_px=(15.0, -10.0),
        r0_px=10.0,
        r1_px=30.0,
        fov_deg=60.0,
        order=1,
    )

    assert display.dtype == np.float32
    assert np.isfinite(display).all()
    assert display.min() >= 0.0 and display.max() <= 1.0


def test_fixed_display_and_physical_fan_calibrations_agree():
    fan = DEFAULT_DISPLAY_FAN
    assert CALIBRATED_DISPLAY_SHAPE == (660, 880)
    assert np.isclose(
        fan["r0_px"] * DEFAULT_US_SPACING_MM,
        DEFAULT_PROBE_GEOMETRY.radius_mm,
    )
    assert np.isclose(
        (fan["r1_px"] - fan["r0_px"]) * DEFAULT_US_SPACING_MM,
        DEFAULT_PROBE_GEOMETRY.depth_mm,
    )
    assert np.isclose(fan["fov_deg"], DEFAULT_PROBE_GEOMETRY.fov_deg)

    # The all-15-bag fitted central scan line occupies y=46..657 in the real B-mode image.
    assert np.isclose(fan["apex_px"][1] + fan["r0_px"], 46.0)
    assert np.isclose(fan["apex_px"][1] + fan["r1_px"], 657.0)


def test_calibrated_fan_occupies_the_measured_us_pixel_extent():
    support = scan_convert_fan(
        np.ones(
            (DEFAULT_PROBE_GEOMETRY.n_ax, DEFAULT_PROBE_GEOMETRY.n_lat),
            dtype=np.uint8,
        ),
        CALIBRATED_DISPLAY_SHAPE,
        **DEFAULT_DISPLAY_FAN,
        order=0,
        cval=0,
    ).astype(bool)

    # The virtual apex is above the image, so the diverging side beams already enter
    # the canvas at row 0.  The nominal central-line face/far coordinates are checked
    # separately above; rasterisation places the final supported pixel at row 656.
    rows, cols = np.where(support)
    assert (int(rows.min()), int(rows.max())) == (0, 656)
    assert (int(cols.min()), int(cols.max())) == (0, 879)
