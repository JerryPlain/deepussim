"""Ultrasound image formation: reslice a volume at a probe pose, then render.

``reslice`` is the geometric step (sample a 2D plane out of the 3D CBCT); ``render``
is the acoustic step (turn the resliced intensities into a B-mode-like image). The
renderer's parameters are exactly what Step 4 calibrates against real US.
"""

from .reslice import ProbeGeometry, reslice, reslice_volume
from .renderer import RendererParams, render

__all__ = ["ProbeGeometry", "reslice", "reslice_volume", "RendererParams", "render"]
