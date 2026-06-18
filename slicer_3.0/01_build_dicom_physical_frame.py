"""Build the physical CBCT coordinate frame for a DICOM/recon pair.

This is slicer 3.0 step 1:

* read a CBCT DICOM series and its reconstructed ``CBCT.mhd``;
* build the voxel-index -> DICOM-LPS physical affine in millimetres;
* build a phantom-centred millimetre frame whose origin is the simulator's
  phantom local origin, normally ``[0, 0, 0]`` in DICOM-LPS millimetres;
* write a small metadata bundle for later slicer/simulation steps;
* print direction checks so the DICOM volume frame stays aligned with the
  frame used by the simulator's CBCT reslicing code.

Coordinate convention:

* voxel indices are ``(i, j, k)`` = SimpleITK ``(x, y, z)``;
* physical coordinates are DICOM/LPS millimetres:
  ``+X = Left``, ``+Y = Posterior``, ``+Z = Superior``;
* slicer 3.0 downstream code should use the centred CBCT frame in millimetres.

Run from the repository root in the conda env:

    python slicer_3.0/01_build_dicom_physical_frame.py
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deepussim-main"

DEFAULT_DICOM_DIR = ROOT / "cbct_20260612" / "cbct_20260612" / "DICOM1"
DEFAULT_RECON_PATH = ROOT / "cbct_20260612" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_OUT_DIR = PROJECT / "slicer_3.0" / "outputs" / "step01_physical_frame_origin000"
DEFAULT_PHANTOM_ORIGIN_LPS_MM = np.array([0.0, 0.0, 0.0], dtype=float)


@dataclass
class VolumeFrame:
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


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray | list[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def translation(offset: np.ndarray | list[float]) -> np.ndarray:
    return make_transform(np.eye(3), offset)


def affine_from_sitk(img: sitk.Image) -> np.ndarray:
    spacing = np.array(img.GetSpacing(), dtype=float)
    origin = np.array(img.GetOrigin(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def frame_from_image(name: str, path: Path, img: sitk.Image) -> VolumeFrame:
    size = np.array(img.GetSize(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    affine = affine_from_sitk(img)
    center_ijk = (size - 1.0) / 2.0
    center_lps = (affine @ np.r_[center_ijk, 1.0])[:3]
    orth_error = np.max(np.abs(direction.T @ direction - np.eye(3)))

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
        max_orthogonality_error=float(orth_error),
    )


def read_dicom_series(directory: Path) -> tuple[sitk.Image, str, int]:
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
    if not ids:
        raise FileNotFoundError(f"no DICOM series found under {directory}")

    # The June 12 folder is expected to contain one CBCT series. If there are
    # multiple, using the first keeps the script deterministic and reports it.
    series_id = ids[0]
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), series_id, len(files)


def compare_frames(a: VolumeFrame, b: VolumeFrame) -> dict[str, object]:
    aa = np.array(a.affine_lps_from_ijk_mm, dtype=float)
    bb = np.array(b.affine_lps_from_ijk_mm, dtype=float)
    T_a_ijk_from_b_ijk = np.linalg.inv(aa) @ bb
    return {
        "same_size": a.size_ijk == b.size_ijk,
        "spacing_abs_diff_mm": (
            np.abs(np.array(a.spacing_mm) - np.array(b.spacing_mm)).tolist()
        ),
        "origin_abs_diff_mm": (
            np.abs(np.array(a.origin_lps_mm) - np.array(b.origin_lps_mm)).tolist()
        ),
        "direction_abs_diff": (
            np.abs(
                np.array(a.direction_lps_from_ijk) - np.array(b.direction_lps_from_ijk)
            ).tolist()
        ),
        "affine_max_abs_diff": float(np.max(np.abs(aa - bb))),
        "center_abs_diff_mm": (
            np.abs(np.array(a.center_lps_mm) - np.array(b.center_lps_mm)).tolist()
        ),
        "T_dicom_ijk_from_recon_ijk": T_a_ijk_from_b_ijk.tolist(),
    }


def centered_affine(frame: VolumeFrame, origin_lps_mm: np.ndarray) -> np.ndarray:
    """Voxel ijk -> phantom-centred CBCT coordinates in millimetres."""
    affine_lps = np.array(frame.affine_lps_from_ijk_mm, dtype=float)
    return translation(-origin_lps_mm) @ affine_lps


def print_frame(frame: VolumeFrame) -> None:
    print(f"\n[{frame.name}]")
    print(f"  path      : {frame.path}")
    print(f"  size ijk  : {frame.size_ijk}")
    print(f"  spacing mm: {np.array(frame.spacing_mm)}")
    print(f"  origin LPS: {np.array(frame.origin_lps_mm)}")
    print("  direction columns in DICOM-LPS:")
    print(np.array(frame.direction_lps_from_ijk))
    print("  affine LPS(mm) <- voxel ijk:")
    print(np.array(frame.affine_lps_from_ijk_mm))
    print(f"  center LPS mm : {np.array(frame.center_lps_mm)}")
    print(f"  det(direction): {frame.det_direction:.6f}")
    print(f"  orth err      : {frame.max_orthogonality_error:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--recon", type=Path, default=DEFAULT_RECON_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--phantom-origin-lps-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "phantom local origin in DICOM-LPS mm. Defaults to 0 0 0, matching "
            "the simulator phantom local origin."
        ),
    )
    args = parser.parse_args()

    dicom_img, series_id, n_files = read_dicom_series(args.dicom_dir)
    recon_img = sitk.ReadImage(str(args.recon))

    dicom = frame_from_image("dicom", args.dicom_dir, dicom_img)
    recon = frame_from_image("recon", args.recon, recon_img)
    if args.phantom_origin_lps_mm is None:
        phantom_origin_lps_mm = DEFAULT_PHANTOM_ORIGIN_LPS_MM.copy()
        phantom_origin_source = "default_origin_000"
    else:
        phantom_origin_lps_mm = np.array(args.phantom_origin_lps_mm, dtype=float)
        phantom_origin_source = "command_line"

    T_centered_from_lps = translation(-phantom_origin_lps_mm)
    dicom_affine_centered = centered_affine(dicom, phantom_origin_lps_mm)
    recon_affine_centered = centered_affine(recon, phantom_origin_lps_mm)

    # This is the frame conversion already used by the simulator bridge:
    # raw measured scan/optical CBCT frame <- DICOM-LPS volume frame.
    # The volume/reslice frame itself remains DICOM-LPS millimetres.
    R_scan_from_dicom_lps = rot_x(math.pi / 2.0) @ rot_z(math.pi)
    T_scan_from_dicom_lps = make_transform(R_scan_from_dicom_lps, [0.0, 0.0, 0.0])
    recon_direction = np.array(recon.direction_lps_from_ijk, dtype=float)
    recon_matches_lie_down = float(np.max(np.abs(recon_direction - R_scan_from_dicom_lps)))

    report = {
        "coordinate_convention": {
            "raw_volume_frame": "DICOM LPS, millimetres",
            "default_working_frame": "phantom-centred CBCT, millimetres",
            "positive_x": "Left",
            "positive_y": "Posterior",
            "positive_z": "Superior",
            "voxel_order": "ijk == SimpleITK xyz",
            "sim_phantom_origin": "phantom local origin / centred mesh origin",
            "sim_reslice_frame": (
                "same axis directions as DICOM-LPS, translated so the phantom "
                "origin is (0,0,0)"
            ),
            "slicer_display_note": "3D Slicer uses RAS internally: RAS = diag(-1,-1,1) @ LPS",
        },
        "phantom_centered_frame": {
            "origin_source": phantom_origin_source,
            "origin_lps_mm": phantom_origin_lps_mm.tolist(),
            "axis_alignment_with_dicom_lps": {
                "phantom_local_x": "+DICOM_LPS_X / Left",
                "phantom_local_y": "+DICOM_LPS_Y / Posterior",
                "phantom_local_z": "+DICOM_LPS_Z / Superior",
                "rotation_centered_from_lps": np.eye(3).tolist(),
                "note": (
                    "The centered frame changes only the origin. It does not "
                    "rotate or flip the DICOM-LPS axes, so simulator phantom "
                    "local axes match DICOM-LPS before world placement."
                ),
            },
            "T_centered_from_lps_mm": T_centered_from_lps.tolist(),
            "dicom_affine_centered_from_ijk_mm": dicom_affine_centered.tolist(),
            "recon_affine_centered_from_ijk_mm": recon_affine_centered.tolist(),
            "dicom_center_in_centered_mm": (
                T_centered_from_lps @ np.r_[dicom.center_lps_mm, 1.0]
            )[:3].tolist(),
            "recon_center_in_centered_mm": (
                T_centered_from_lps @ np.r_[recon.center_lps_mm, 1.0]
            )[:3].tolist(),
        },
        "dicom_series_id": series_id,
        "dicom_file_count": n_files,
        "dicom": asdict(dicom),
        "recon": asdict(recon),
        "dicom_vs_recon": compare_frames(dicom, recon),
        "sim_alignment": {
            "T_scan_optical_from_dicom_lps": T_scan_from_dicom_lps.tolist(),
            "recon_direction_matches_T_scan_optical_from_dicom_lps_max_abs_diff": (
                recon_matches_lie_down
            ),
            "note": (
                "Use the phantom-centred CBCT mm frame for slicer 3.0 downstream work. "
                "It keeps the same axis directions as DICOM-LPS but moves the "
                "sim phantom origin to (0,0,0). "
                "Only apply this Rx(90)*Rz(180) lie-down rotation when composing "
                "the measured scan/optical CBCT placement into the robot/sim world."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "physical_frame_report.json"
    npz_path = args.out_dir / "physical_frame_matrices.npz"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        dicom_affine_lps_from_ijk_mm=np.array(dicom.affine_lps_from_ijk_mm),
        recon_affine_lps_from_ijk_mm=np.array(recon.affine_lps_from_ijk_mm),
        T_centered_from_lps_mm=T_centered_from_lps,
        phantom_origin_lps_mm=phantom_origin_lps_mm,
        dicom_affine_centered_from_ijk_mm=dicom_affine_centered,
        recon_affine_centered_from_ijk_mm=recon_affine_centered,
        T_dicom_ijk_from_recon_ijk=np.array(
            report["dicom_vs_recon"]["T_dicom_ijk_from_recon_ijk"]
        ),
        T_scan_optical_from_dicom_lps=T_scan_from_dicom_lps,
    )

    print_frame(dicom)
    print_frame(recon)
    print("\n[dicom vs recon]")
    print(json.dumps(report["dicom_vs_recon"], indent=2))
    print("\n[sim alignment]")
    print("  raw CBCT volume frame: DICOM-LPS millimetres")
    print("  slicer 3.0 working frame: phantom-centred CBCT millimetres")
    print("  phantom local axes before world placement:")
    print("    +X = +DICOM LPS X = Left")
    print("    +Y = +DICOM LPS Y = Posterior")
    print("    +Z = +DICOM LPS Z = Superior")
    print("  phantom origin LPS mm:", phantom_origin_lps_mm)
    print("  DICOM center in centred frame mm:",
          np.array(report["phantom_centered_frame"]["dicom_center_in_centered_mm"]))
    print("  recon center in centred frame mm:",
          np.array(report["phantom_centered_frame"]["recon_center_in_centered_mm"]))
    print(
        "  recon direction vs sim lie-down max abs diff:",
        f"{recon_matches_lie_down:.3e}",
    )
    print("  T_scan_optical_from_dicom_lps:")
    print(T_scan_from_dicom_lps)
    print(f"\nwrote {report_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
