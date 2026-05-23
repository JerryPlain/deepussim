import numpy as np

from deepussim.us.reslice import ProbeGeometry
from deepussim.us.renderer import RendererParams, render
from deepussim.calib.renderer_fit import ncc


def test_render_output_range_and_shape():
    geom = ProbeGeometry(n_lat=32, n_ax=64)
    rng = np.random.default_rng(0)
    intensity = rng.random((geom.n_ax, geom.n_lat))
    img = render(intensity, geom, RendererParams(seed=0))
    assert img.shape == (geom.n_ax, geom.n_lat)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_interfaces_produce_echoes():
    geom = ProbeGeometry(n_lat=16, n_ax=64)
    intensity = np.zeros((geom.n_ax, geom.n_lat))
    intensity[32:, :] = 1.0  # a single horizontal interface at mid-depth
    img = render(intensity, geom, RendererParams(speckle_sigma=0.0, seed=0))
    # The brightest row should sit at the interface, not in the flat regions.
    assert 28 <= int(np.argmax(img.mean(axis=1))) <= 36


def test_render_deterministic_with_seed():
    geom = ProbeGeometry(n_lat=16, n_ax=32)
    intensity = np.random.default_rng(1).random((geom.n_ax, geom.n_lat))
    a = render(intensity, geom, RendererParams(seed=42))
    b = render(intensity, geom, RendererParams(seed=42))
    assert np.allclose(a, b)
    assert ncc(a, b) > 0.999
