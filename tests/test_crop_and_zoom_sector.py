import numpy as np

from reslice.sector import crop_and_zoom_sector


def test_crop_and_zoom_supports_independent_row_and_column_margins():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.ones_like(image, dtype=bool)
    crop_mask = np.zeros_like(mask)
    crop_mask[2:8, 1:9] = True

    out, debug = crop_and_zoom_sector(
        image,
        mask,
        crop_mask,
        target_shape=(12, 12),
        margin_px=0,
        margin_rows_px=0,
        margin_cols_px=-1,
    )

    assert out.shape == (12, 12)
    assert debug["crop_bbox_rc"] == [2, 2, 8, 8]
    assert debug["crop_margin_rows_px"] == 0
    assert debug["crop_margin_cols_px"] == -1
