"""LC2 similarity: high when US is a local linear combination of CT + |∇CT|, low otherwise."""
import numpy as np
import pytest

from deepussim.calib.lc2 import lc2_similarity, gradient_magnitude


def _smooth_field(seed, shape=(128, 128)):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal(shape), sigma=3.0)


def test_lc2_high_for_linear_combination():
    ct = _smooth_field(0)
    us = 0.7 * ct + 0.3 * gradient_magnitude(ct) + 0.1      # exactly the LC2 model
    assert lc2_similarity(us, ct) > 0.95


def test_lc2_low_for_independent_image():
    ct = _smooth_field(1)
    matched = lc2_similarity(0.7 * ct + 0.3 * gradient_magnitude(ct), ct)
    indep = lc2_similarity(_smooth_field(99), ct)           # unrelated structure
    # smooth synthetic fields have a nonzero spurious floor; what matters is matched >> independent
    assert indep < 0.5 and indep < matched - 0.4


def test_lc2_drops_when_misaligned():
    ct = _smooth_field(2)
    us = 0.6 * ct + 0.4 * gradient_magnitude(ct)
    aligned = lc2_similarity(us, ct)
    shifted = lc2_similarity(us, np.roll(ct, 20, axis=1))   # break correspondence
    assert aligned > 0.9 and aligned > shifted + 0.3


def test_lc2_mask_restricts_region():
    ct = _smooth_field(3)
    us = 0.5 * ct + 0.5 * gradient_magnitude(ct)
    mask = np.zeros(ct.shape, bool); mask[32:96, 32:96] = True
    assert lc2_similarity(us, ct, mask=mask) > 0.9          # still explained inside the mask


def test_lc2_shape_mismatch_raises():
    with pytest.raises(ValueError):
        lc2_similarity(np.zeros((10, 10)), np.zeros((10, 12)))


def _synth_volume(n=80):
    """A discriminative volume: smooth base + sharp asymmetric blobs giving strong, localized
    gradients so the LC2 landscape is peaked at the true pose. 1 mm spacing, centred at origin."""
    from scipy.ndimage import gaussian_filter
    from deepussim.data.volume import Volume
    rng = np.random.default_rng(0)
    v = gaussian_filter(rng.standard_normal((n, n, n)), 3.0)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
    for cx, cy, cz, amp in [(0.40, 0.45, 0.50, 4.0), (0.62, 0.55, 0.48, -3.0),
                            (0.50, 0.40, 0.60, 3.0)]:
        v += amp * np.exp(-((xx - n * cx) ** 2 + (yy - n * cy) ** 2 + (zz - n * cz) ** 2)
                          / (2 * 4.0 ** 2))
    affine = np.eye(4); affine[:3, 3] = -(n - 1) / 2.0          # 1 mm spacing, centred
    return Volume(v.astype(float), affine)


def test_register_frame_lc2_recovers_perturbed_pose():
    from scipy.spatial.transform import Rotation as Rot
    from deepussim.us.reslice import ProbeGeometry, reslice_volume
    from deepussim.calib.lc2 import register_frame_lc2, gradient_magnitude

    volume = _synth_volume()
    geom = ProbeGeometry(radius_mm=20.0, fov_deg=60.0, depth_mm=40.0, n_lat=48, n_ax=64)
    true_pose = np.eye(4); true_pose[:3, 3] = [0.0, 0.0, -20.0]   # face at z=-20, axial +z

    sl = reslice_volume(volume, true_pose, geom)
    us = 0.7 * sl + 0.3 * gradient_magnitude(sl) + 0.1           # exact LC2 model of the slice

    delta = np.eye(4)
    delta[:3, :3] = Rot.from_euler("xyz", [2.0, 2.0, 2.0], degrees=True).as_matrix()
    delta[:3, 3] = [3.0, -2.0, 2.0]                              # ~4 mm / ~3.5 deg off (good init)
    init_pose = true_pose @ delta

    lc2_init = lc2_similarity(us, reslice_volume(volume, init_pose, geom))
    refined, lc2_ref = register_frame_lc2(us, volume, geom, init_pose,
                                          max_trans_mm=8.0, max_rot_deg=8.0)   # constrained range

    err_init = np.linalg.norm(init_pose[:3, 3] - true_pose[:3, 3])
    err_ref = np.linalg.norm(refined[:3, 3] - true_pose[:3, 3])
    assert lc2_ref > lc2_init             # LC2 improved
    assert lc2_ref > 0.9                  # climbed close to the true alignment
    assert err_ref < err_init             # pose moved toward truth
