"""Learned-renderer (CUT) sanity: generator is shape-preserving and the losses compute +
backprop. Skipped where torch isn't installed (CPU-only core / no GPU stack)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepussim.renderer.cut import CUTModel
from deepussim.renderer.data import _to_unit


def test_generator_is_shape_preserving():
    m = CUTModel(ngf=16, ndf=16, n_blocks=2, num_patches=32)
    x = torch.randn(2, 1, 128, 64)
    assert m.G(x).shape == x.shape          # CBCT-slice -> US, same fan layout


def test_cut_losses_compute_and_backprop():
    torch.manual_seed(0)
    m = CUTModel(ngf=16, ndf=16, n_blocks=2, num_patches=32)
    src, us = torch.randn(2, 1, 128, 64), torch.randn(2, 1, 128, 64)
    g_total, fake, parts = m.g_loss(src, us)
    assert {"gan", "nce", "nce_idt"} <= set(parts)
    g_total.backward()                      # generator path differentiable
    d = m.d_loss(us, fake.detach())
    d.backward()                            # discriminator path differentiable
    assert torch.isfinite(g_total) and torch.isfinite(d)


def test_to_unit_normalises_to_pm1():
    x = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
    u = _to_unit(x)
    assert u.min() == pytest.approx(-1.0) and u.max() == pytest.approx(1.0)
    assert _to_unit(np.full((3, 3), 7.0)).max() == 0.0   # uniform -> zeros


def _save_tiny_ckpt(path):
    m = CUTModel(ngf=16, ndf=16, n_blocks=2)
    torch.save({"G": m.G.state_dict(), "args": {"ngf": 16, "n_blocks": 2}, "epoch": 3}, path)


def test_neural_renderer_matches_physics_interface(tmp_path):
    from deepussim.renderer.neural import NeuralRenderer
    ckpt = tmp_path / "generator.pt"; _save_tiny_ckpt(ckpt)
    r = NeuralRenderer(ckpt, device="cpu")
    img = r(np.random.default_rng(0).random((128, 64)))   # CBCT slice -> US image
    assert img.shape == (128, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0          # [0,1], like render()


def test_generate_dataset_uses_learned_renderer(tmp_path):
    from deepussim.renderer.neural import NeuralRenderer
    from deepussim.data.volume import Volume
    from deepussim.us.reslice import ProbeGeometry
    from deepussim.pipeline.scaleup import generate_dataset
    from deepussim.data.record import load_sample
    from deepussim import geometry as g

    ckpt = tmp_path / "generator.pt"; _save_tiny_ckpt(ckpt)
    r = NeuralRenderer(ckpt, device="cpu")
    vol = Volume(np.random.default_rng(0).random((80, 80, 80)) + 0.5, np.eye(4))
    geom = ProbeGeometry(radius_mm=20.0, fov_deg=60.0, depth_mm=15.0, n_lat=16, n_ax=24)
    n = generate_dataset(tmp_path / "ds", vol, [g.from_translation([40.0, 40.0, 20.0])],
                         geom, renderer=r, progress=False)
    assert n == 1
    img = load_sample(sorted((tmp_path / "ds").glob("sample_*.npz"))[0]).image
    assert img.shape == (24, 16) and 0.0 <= img.min() and img.max() <= 1.0
