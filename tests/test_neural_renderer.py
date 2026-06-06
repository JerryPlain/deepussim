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
