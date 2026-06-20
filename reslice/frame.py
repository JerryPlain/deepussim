"""Step 1 of the workflow: build the CBCT physical coordinate frame.

For a DICOM/recon pair this computes the voxel-index -> physical affine in DICOM-LPS
millimetres, plus a *phantom-centred* frame whose origin is the simulator's phantom
local origin (normally ``[0, 0, 0]`` in DICOM-LPS mm). Downstream reslicing samples the
volume through this phantom-centred affine.

Coordinate convention:

* voxel indices ``(i, j, k)`` == SimpleITK ``(x, y, z)``;
* physical coordinates are DICOM/LPS millimetres: ``+X = Left``, ``+Y = Posterior``,
  ``+Z = Superior``;
* the phantom-centred frame keeps the same axis directions and only shifts the origin.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from reslice._geometry import translation


@dataclass
class VolumeFrame:
    """Geometry of one volume (DICOM series or recon image) in DICOM-LPS mm."""

    name: str
    path: str
    size_ijk: list[int]
    spacing_mm: list[float]
    origin_lps_mm: list[float]
    direction_lps_from_ijk: list[list[float]]
    affine_lps_from_ijk_mm: list[list[float]]
    center_lps_mm: list[float]
    det_direction: float
    max_orthogonality_error: float


def affine_from_sitk(img) -> np.ndarray:
    """Voxel-index -> DICOM-LPS mm affine for a SimpleITK image: ``direction @ diag(spacing)``."""
    spacing = np.array(img.GetSpacing(), dtype=float)
    origin = np.array(img.GetOrigin(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def frame_from_image(name: str, path: Path, img) -> VolumeFrame:
    """Describe a SimpleITK image's geometry, with determinant / orthogonality checks."""
    size = np.array(img.GetSize(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    affine = affine_from_sitk(img)
    center_ijk = (size - 1.0) / 2.0
    center_lps = (affine @ np.r_[center_ijk, 1.0])[:3]
    orth_error = float(np.max(np.abs(direction.T @ direction - np.eye(3))))
    return VolumeFrame(
        name=name,
        path=str(path),
        size_ijk=[int(x) for x in img.GetSize()],
        spacing_mm=[float(x) for x in img.GetSpacing()],
        origin_lps_mm=[float(x) for x in img.GetOrigin()],
        direction_lps_from_ijk=direction.tolist(),
        affine_lps_from_ijk_mm=affine.tolist(),
        center_lps_mm=center_lps.tolist(),
        det_direction=float(np.linalg.det(direction)),
        max_orthogonality_error=orth_error,
    )


def centered_affine(frame: VolumeFrame, origin_lps_mm: np.ndarray) -> np.ndarray:
    """Voxel ijk -> phantom-centred mm: the LPS affine shifted so ``origin_lps_mm`` is 0."""
    affine_lps = np.array(frame.affine_lps_from_ijk_mm, dtype=float)
    return translation(-np.asarray(origin_lps_mm, dtype=float)) @ affine_lps


def read_dicom_series(directory: Path):
    """Read the first DICOM series under ``directory``; returns ``(image, series_id, n_files)``."""
    import SimpleITK as sitk

    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
    if not ids:
        raise FileNotFoundError(f"no DICOM series found under {directory}")
    series_id = ids[0]
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), series_id, len(files)


def build_report(
    dicom_dir: Path,
    recon_path: Path,
    phantom_origin_lps_mm: np.ndarray,
) -> dict:
    """Build the metadata bundle consumed by :mod:`reslice.slice`.

    Holds, for both the DICOM series and the recon image, the phantom-centred
    voxel->mm affine (the key downstream reslicing needs).
    """
    import SimpleITK as sitk

    dicom_img, series_id, n_files = read_dicom_series(dicom_dir)
    recon_img = sitk.ReadImage(str(recon_path))

    dicom = frame_from_image("dicom", dicom_dir, dicom_img)
    recon = frame_from_image("recon", recon_path, recon_img)
    phantom_origin = np.asarray(phantom_origin_lps_mm, dtype=float)

    return {
        "coordinate_convention": {
            "raw_volume_frame": "DICOM LPS, millimetres",
            "working_frame": "phantom-centred CBCT, millimetres",
            "positive_x": "Left",
            "positive_y": "Posterior",
            "positive_z": "Superior",
            "voxel_order": "ijk == SimpleITK xyz",
        },
        "phantom_centered_frame": {
            "origin_lps_mm": phantom_origin.tolist(),
            "dicom_affine_centered_from_ijk_mm": centered_affine(dicom, phantom_origin).tolist(),
            "recon_affine_centered_from_ijk_mm": centered_affine(recon, phantom_origin).tolist(),
        },
        "dicom_series_id": series_id,
        "dicom_file_count": n_files,
        "dicom": asdict(dicom),
        "recon": asdict(recon),
    }


def write_report(report: dict, out_dir: Path) -> Path:
    """Write ``physical_frame_report.json`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "physical_frame_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
