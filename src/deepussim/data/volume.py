"""3D volume container + IO (Step 1: CBCT intensity volume and label volume).

A ``Volume`` is a voxel array plus an affine mapping voxel indices ``(i, j, k)`` to
world coordinates in millimetres:

    p_world = affine @ [i, j, k, 1]^T.

Both the intensity volume and the segmented label volume share this representation;
reslicing them with the *same* probe pose yields a paired (US image, anatomy mask).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Volume:
    data: np.ndarray  # (ni, nj, nk)
    affine: np.ndarray  # 4x4 voxel(i,j,k)->world(mm)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.affine = np.asarray(self.affine, dtype=float)
        if self.affine.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got {self.affine.shape}")
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3D, got shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def spacing(self) -> np.ndarray:
        """Voxel size (mm) along each axis = column norms of the affine."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def world_to_voxel(self, pts_world: np.ndarray) -> np.ndarray:
        """Map world points (N,3) mm to fractional voxel indices (N,3)."""
        inv = np.linalg.inv(self.affine)
        pts = np.atleast_2d(np.asarray(pts_world, dtype=float))
        h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        return (h @ inv.T)[:, :3]

    def voxel_to_world(self, vox: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(np.asarray(vox, dtype=float))
        h = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
        return (h @ self.affine.T)[:, :3]

    def center_world(self) -> np.ndarray:
        """World coordinate of the volume centre — handy for aiming a probe."""
        c = (np.array(self.data.shape) - 1) / 2.0
        return self.voxel_to_world(c)[0]


def load_nifti(path) -> Volume:
    """Load a NIfTI file (.nii / .nii.gz) into a Volume."""
    import nibabel as nib

    img = nib.load(str(path))
    return Volume(data=np.asarray(img.get_fdata()), affine=np.asarray(img.affine))


def save_nifti(path, volume: Volume) -> None:
    import nibabel as nib

    nib.save(nib.Nifti1Image(volume.data, volume.affine), str(path))


def _volume_from_sitk(img) -> Volume:
    """Build a Volume from a SimpleITK image (shared by DICOM and NRRD loaders).

    SimpleITK indexes (x, y, z) while ``GetArrayFromImage`` returns (z, y, x) — we
    transpose to (i, j, k) = (x, y, z) so the voxel->world affine matches the array.
    """
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    data = np.transpose(arr, (2, 1, 0))  # (x, y, z) == (i, j, k)

    spacing = np.array(img.GetSpacing())  # (sx, sy, sz)
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return Volume(data=data, affine=affine)


def load_nrrd(path) -> Volume:
    """Load a NRRD volume (3D Slicer's default scalar-volume export) via SimpleITK."""
    import SimpleITK as sitk

    return _volume_from_sitk(sitk.ReadImage(str(path)))


def load_dicom_series(directory) -> Volume:
    """Load a DICOM series via SimpleITK and convert to a Volume (mm world frame)."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(directory))
    if not files:
        raise FileNotFoundError(f"no DICOM series found under {directory}")
    reader.SetFileNames(files)
    return _volume_from_sitk(reader.Execute())


def load_volume(path) -> Volume:
    """Load a volume by extension: ``.nii/.nii.gz`` (NIfTI) or ``.nrrd`` (NRRD)."""
    s = str(path).lower()
    if s.endswith(".nrrd") or s.endswith(".nhdr"):
        return load_nrrd(path)
    return load_nifti(path)
