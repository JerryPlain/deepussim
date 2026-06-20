"""Loading helpers: CBCT volumes and 4x4 transform files.

A volume may be a single image file (``.mhd`` / ``.nrrd`` / ``.nii.gz``) or a directory
holding a DICOM series. Either way :func:`load_volume_data` returns the voxel array in
``(x, y, z)`` order, matching the ``(i, j, k)`` indexing used by the affine in
:mod:`reslice.frame`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_volume_data(path: Path) -> np.ndarray:
    """Read a CBCT volume and return its voxels as a float32 ``(x, y, z)`` array.

    SimpleITK returns arrays in ``(z, y, x)`` order, so we transpose to ``(x, y, z)``
    to match the SimpleITK ``GetSize()`` / affine convention (``i == x``).
    """
    import SimpleITK as sitk

    path = Path(path)
    if path.is_dir():
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if not ids:
            raise FileNotFoundError(f"no DICOM series found under {path}")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(str(path))

    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    return np.transpose(arr, (2, 1, 0)).astype(np.float32)   # (x, y, z)


def load_transform_4x4(path: Path, key: str | None = None) -> np.ndarray:
    """Load a 4x4 transform from ``.npy`` / ``.npz`` / ``.txt`` / ``.csv``.

    For ``.npz`` either give an explicit ``key`` or rely on the first 4x4 array found.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        T = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if key:
            T = data[key]
        else:
            candidates = [n for n in data.files if np.asarray(data[n]).shape == (4, 4)]
            if not candidates:
                raise ValueError(f"no 4x4 matrix found in {path}")
            T = data[candidates[0]]
    else:
        T = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{path} must contain a 4x4 transform, got {T.shape}")
    return T
