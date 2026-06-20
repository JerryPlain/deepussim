"""Clean reimplementation of the CBCT->US reslicing workflow (formerly slicer_3.0).

Geometry-faithful refactor: same maths, clearer structure, English docstrings, no cruft.

Pipeline:
    build_frame  (step 1)  DICOM + recon  -> phantom-centred voxel->mm affine (report)
    pose         (step 2a) recorded probe pose -> image plane in phantom-centred mm
    sampling     (step 2b) reslice a rectangular plane out of the volume
    sector       (step 4)  mask the plane to the ultrasound fan + crop/zoom to US size

CLIs: ``python -m reslice.build_frame`` and ``python -m reslice.slice``.
"""
from __future__ import annotations

from reslice import frame, io, pose, sampling, sector

__all__ = ["frame", "io", "pose", "sampling", "sector"]
