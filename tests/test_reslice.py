import numpy as np

from deepussim.data.volume import Volume
from deepussim.geometry import make_transform, rot_y
from deepussim.us.reslice import ProbeGeometry, reslice_volume


def _ramp_volume(n=32):
    # Intensity increases linearly along the k (z-world) axis; identity affine (1mm).
    data = np.broadcast_to(np.arange(n, dtype=float), (n, n, n)).copy()
    return Volume(data, np.eye(4))


def test_reslice_recovers_known_plane():
    vol = _ramp_volume(32)
    geom = ProbeGeometry(width_mm=10, depth_mm=10, n_lat=11, n_ax=11)
    # Probe at world (16,16,5), +z axial pointing along +world-z (identity rotation).
    T = make_transform(np.eye(3), [16.0, 16.0, 5.0])
    img = reslice_volume(vol, T, geom, order=1)
    assert img.shape == (11, 11)
    # Along depth (rows) intensity should increase (volume ramps with k = world z).
    col = img[:, 5]
    assert np.all(np.diff(col) > 0)


def test_label_reslice_is_integer_preserving():
    n = 32
    labels = np.zeros((n, n, n), dtype=np.int16)
    labels[:, :, n // 2 :] = 7  # half-space label
    vol = Volume(labels, np.eye(4))
    geom = ProbeGeometry(width_mm=8, depth_mm=20, n_lat=9, n_ax=21)
    T = make_transform(np.eye(3), [16.0, 16.0, 6.0])
    mask = reslice_volume(vol, T, geom, order=0).astype(int)
    assert set(np.unique(mask)).issubset({0, 7})  # nearest-neighbour, no blending


def test_oblique_pose_runs():
    vol = _ramp_volume(32)
    geom = ProbeGeometry(width_mm=10, depth_mm=10, n_lat=8, n_ax=8)
    T = make_transform(rot_y(0.4), [16.0, 16.0, 16.0])
    img = reslice_volume(vol, T, geom)
    assert img.shape == (8, 8) and np.isfinite(img).all()
