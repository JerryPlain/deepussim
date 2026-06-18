"""Build a Slicer/VTK slice plane from the final robot probe pose.

This script connects the simulation pose chain to the slicer 3.0 CBCT frame:

    T_world_from_probe_m
        -> T_phantom_centered_from_probe_mm
        -> point + normal in phantom-centred CBCT millimetres
        -> optional 3D Slicer SliceToRAS matrix

The working CBCT frame is the one produced by step 1:

* origin at the simulator phantom local origin;
* axes aligned with DICOM/LPS;
* units are millimetres.

The ultrasound image plane is taken as the probe local X-Z plane. Therefore:

* plane point  = probe origin;
* plane normal = probe local +Y (elevation);
* image lateral direction = probe local X;
* image axial direction   = probe local Z.

Run outside Slicer to print/export the math:

    python slicer_3.0/02_slice_from_probe_pose.py --sequence-frame 21

Run inside 3D Slicer's Python interactor to show the Red slice:

    exec(open(r"D:\\Master1\\Praktikum\\deepussim-main\\deepussim-main\\slicer_3.0\\02_slice_from_probe_pose.py").read())
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deepussim-main"

STEP01_REPORT = PROJECT / "slicer_3.0" / "outputs" / "step01_physical_frame" / "physical_frame_report.json"
DEFAULT_SEQUENCE = PROJECT / "data" / "sequences_20260618" / "scan2.npz"
DEFAULT_SEQUENCE_FRAME = 21
DEFAULT_OUT_DIR = PROJECT / "slicer_3.0" / "outputs" / "step02_slice_from_probe"
DEFAULT_VOLUME_PATH = ROOT / "data_0618" / "cbct" / "0618" / "Series10" / "recon" / "CBCT.mhd"
DEFAULT_SLICE_WIDTH_MM = 360.0
DEFAULT_SLICE_HEIGHT_MM = 300.0

# Project hand-eye: probe frame -> robot end-effector frame.
T_EE_FROM_PROBE = np.array([
    [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
    [-0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.183],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

# Measured placement before the lie-down roll. Translation is metres.
T_WORLD_FROM_PHANTOM_MEASURED_M = np.array([
    [-0.9993213267731319, -0.004748964835081844, -0.036528525694109644, 0.5618980642296896],
    [0.004894960353026591, -0.9999803818526938, -0.0039083593598543695, 0.3711799666308714],
    [-0.03650924841094883, -0.004084512546023032, 0.9993249679347199, 0.3154777116310868],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

# Keep the display convention that matched the slicer 2.0 scan1 checks.
DISPLAY_LATERAL_SIGN = -1.0
DISPLAY_AXIAL_SIGN = 1.0
DEFAULT_PROBE_ROTATION_OFFSET_DEG = 0.0


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def invert_rigid(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction vector")
    return v / n


def load_transform_4x4(path: Path, key: str | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        T = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if key:
            T = data[key]
        else:
            candidates = [name for name in data.files if np.asarray(data[name]).shape == (4, 4)]
            if not candidates:
                raise ValueError(f"no 4x4 matrix found in {path}")
            T = data[candidates[0]]
    else:
        T = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{path} must contain a 4x4 transform, got {T.shape}")
    return T


def probe_rotation_offset(degrees: float) -> np.ndarray:
    return make_transform(rot_z(math.radians(degrees)), [0.0, 0.0, 0.0])


def default_world_from_phantom_centered_m() -> np.ndarray:
    """Simulator placement for the centred phantom mesh, in metres."""
    lie = make_transform(rot_x(math.pi / 2.0) @ rot_z(math.pi), [0.0, 0.0, 0.0])
    return T_WORLD_FROM_PHANTOM_MEASURED_M @ lie


def pose_from_sequence(
    sequence: Path,
    frame: int,
    pose_kind: str,
    probe_rotation_deg: float,
) -> np.ndarray:
    seq = np.load(sequence, allow_pickle=True)
    poses = np.asarray(seq["poses"], dtype=float)
    if frame < 0 or frame >= len(poses):
        raise IndexError(f"frame {frame} out of range 0..{len(poses) - 1}")

    T_world_from_ee = poses[frame]
    if pose_kind == "ee":
        return T_world_from_ee @ T_EE_FROM_PROBE @ probe_rotation_offset(probe_rotation_deg)
    if pose_kind == "probe":
        return T_world_from_ee @ probe_rotation_offset(probe_rotation_deg)
    raise ValueError("--sequence-pose-kind must be 'ee' or 'probe'")


def meters_pose_to_mm(T_m: np.ndarray) -> np.ndarray:
    out = np.asarray(T_m, dtype=float).copy()
    out[:3, 3] *= 1000.0
    return out


def probe_pose_in_phantom_centered_mm(
    T_world_from_probe_m: np.ndarray,
    T_world_from_phantom_m: np.ndarray,
) -> np.ndarray:
    T_phantom_from_world_m = invert_rigid(T_world_from_phantom_m)
    return meters_pose_to_mm(T_phantom_from_world_m @ T_world_from_probe_m)


def plane_from_probe_pose(
    T_phantom_from_probe_mm: np.ndarray,
    plane_mode: str,
    plane_offset_y_mm: float,
) -> dict[str, np.ndarray]:
    R = T_phantom_from_probe_mm[:3, :3]
    point = T_phantom_from_probe_mm[:3, 3] + float(plane_offset_y_mm) * normalize(R[:, 1])
    lateral = normalize(R[:, 0])
    if plane_mode == "probe-xz":
        normal = normalize(R[:, 1])
        axial = normalize(R[:, 2])
        meaning = "plane is probe local X-Z; normal is probe local +Y"
    elif plane_mode == "probe-xy":
        normal = normalize(R[:, 2])
        axial = normalize(R[:, 1])
        meaning = "plane is probe local X-Y; normal is probe local +Z"
    else:
        raise ValueError("--plane-mode must be probe-xz or probe-xy")
    return {
        "point_centered_mm": point,
        "normal_centered": normal,
        "lateral_centered": lateral,
        "axial_centered": axial,
        "meaning": meaning,
        "plane_offset_y_mm": float(plane_offset_y_mm),
    }


def centered_to_lps(point_or_direction: np.ndarray, origin_lps_mm: np.ndarray, is_point: bool) -> np.ndarray:
    x = np.asarray(point_or_direction, dtype=float).reshape(3)
    return x + origin_lps_mm if is_point else x


def lps_to_ras(point_or_direction: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(point_or_direction, dtype=float).reshape(3)
    return np.array([-x, -y, z], dtype=float)


def slicer_axes_ras(plane: dict[str, np.ndarray], phantom_origin_lps_mm: np.ndarray) -> dict[str, np.ndarray]:
    point_lps = centered_to_lps(plane["point_centered_mm"], phantom_origin_lps_mm, True)
    lateral_lps = centered_to_lps(plane["lateral_centered"], phantom_origin_lps_mm, False)
    normal_lps = centered_to_lps(plane["normal_centered"], phantom_origin_lps_mm, False)
    axial_lps = centered_to_lps(plane["axial_centered"], phantom_origin_lps_mm, False)

    return {
        "point_ras_mm": lps_to_ras(point_lps),
        "x_axis_ras": normalize(DISPLAY_LATERAL_SIGN * lps_to_ras(lateral_lps)),
        "y_axis_ras": normalize(DISPLAY_AXIAL_SIGN * lps_to_ras(axial_lps)),
        "z_axis_ras": normalize(lps_to_ras(normal_lps)),
    }


def slice_to_ras_numpy(axes: dict[str, np.ndarray]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 0] = axes["x_axis_ras"]
    T[:3, 1] = axes["y_axis_ras"]
    T[:3, 2] = axes["z_axis_ras"]
    T[:3, 3] = axes["point_ras_mm"]
    return T


def load_volume_data(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    if path.is_dir():
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if not ids:
            raise FileNotFoundError(f"no DICOM series under {path}")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(str(path))

    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return np.transpose(arr, (2, 1, 0)).astype(np.float32)  # x, y, z


def reslice_rectangular_plane(
    data: np.ndarray,
    affine_centered_from_ijk_mm: np.ndarray,
    plane: dict[str, np.ndarray],
    *,
    width_mm: float,
    height_mm: float,
    n_rows: int,
    n_cols: int,
    axial_sign: float,
    lateral_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import map_coordinates

    lateral_mm = np.linspace(-width_mm / 2.0, width_mm / 2.0, n_cols)
    axial_mm = np.linspace(-height_mm * 0.25, height_mm * 0.75, n_rows)
    uu, vv = np.meshgrid(lateral_mm, axial_mm)

    center = plane["point_centered_mm"]
    lateral = lateral_sign * plane["lateral_centered"]
    axial = axial_sign * plane["axial_centered"]
    pts = center[:, None] + lateral[:, None] * uu.ravel()[None, :] + axial[:, None] * vv.ravel()[None, :]
    pts_h = np.vstack([pts, np.ones(pts.shape[1])])
    vox = np.linalg.inv(affine_centered_from_ijk_mm) @ pts_h

    inside = np.ones(vox.shape[1], dtype=bool)
    for axis, size in enumerate(data.shape):
        inside &= (vox[axis] >= 0.0) & (vox[axis] <= size - 1)

    cval = float(np.min(data))
    sampled = map_coordinates(data, vox[:3], order=1, mode="constant", cval=cval)
    return sampled.reshape(n_rows, n_cols), inside.reshape(n_rows, n_cols)


def robust_uint8(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = x[np.isfinite(x)]
    if mask is not None:
        valid = x[mask & np.isfinite(x)]
    if valid.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [0.5, 99.5])
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = (255.0 * y).astype(np.uint8)
    if mask is not None:
        out[~mask] = 0
    return out


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(image)).save(path)


def save_comparison(path: Path, cbct_u8: np.ndarray, us_u8: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(cbct_u8, cmap="gray")
    axes[0].set_title("CBCT slice from probe pose")
    axes[1].imshow(us_u8, cmap="gray")
    axes[1].set_title("Real US frame")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_slice_images(
    *,
    volume_path: Path,
    report: dict,
    plane: dict[str, np.ndarray],
    sequence: Path,
    sequence_frame: int,
    out_dir: Path,
    width_mm: float,
    height_mm: float,
    axial_sign: float,
    lateral_sign: float,
    display_rot180: bool,
) -> dict[str, str]:
    seq = np.load(sequence, allow_pickle=True)
    us = np.asarray(seq["images"][sequence_frame])
    data = load_volume_data(volume_path)

    frame_key = "recon" if volume_path.suffix.lower() == ".mhd" else "dicom"
    affine_key = f"{frame_key}_affine_centered_from_ijk_mm"
    affine = np.array(
        report["phantom_centered_frame"][affine_key],
        dtype=float,
    )

    sl, mask = reslice_rectangular_plane(
        data,
        affine,
        plane,
        width_mm=width_mm,
        height_mm=height_mm,
        n_rows=int(us.shape[0]),
        n_cols=int(us.shape[1]),
        axial_sign=axial_sign,
        lateral_sign=lateral_sign,
    )
    cbct_u8 = robust_uint8(sl, mask)
    us_u8 = robust_uint8(us)
    if display_rot180:
        cbct_u8 = np.rot90(cbct_u8, 2)

    slice_path = out_dir / f"cbct_slice_frame{sequence_frame:04d}.png"
    us_path = out_dir / f"us_frame{sequence_frame:04d}.png"
    compare_path = out_dir / f"compare_cbct_us_frame{sequence_frame:04d}.png"
    save_png(slice_path, cbct_u8)
    save_png(us_path, us_u8)
    save_comparison(compare_path, cbct_u8, us_u8)
    return {
        "cbct_slice_png": str(slice_path),
        "us_frame_png": str(us_path),
        "comparison_png": str(compare_path),
    }


def vtk_matrix_from_numpy(T: np.ndarray):
    import vtk  # type: ignore

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for r in range(4):
        for c in range(4):
            matrix.SetElement(r, c, float(T[r, c]))
    return matrix


def maybe_show_in_slicer(T_slice_to_ras: np.ndarray, volume_path: Path | None, slice_view: str) -> None:
    try:
        import slicer  # type: ignore
    except ImportError:
        return

    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    volume = volumes[0] if volumes else None
    if volume is None and volume_path is not None:
        volume = slicer.util.loadVolume(str(volume_path))
    if volume is None:
        raise RuntimeError("No scalar volume loaded and --volume-path was not usable")

    selection_node = slicer.app.applicationLogic().GetSelectionNode()
    selection_node.SetReferenceActiveVolumeID(volume.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection()

    layout_manager = slicer.app.layoutManager()
    slice_widget = layout_manager.sliceWidget(slice_view)
    if slice_widget is None:
        raise RuntimeError(f"slice view not found: {slice_view}")

    slice_node = slice_widget.mrmlSliceNode()
    slice_node.SetOrientation("Reformat")
    slice_node.SetSliceToRAS(vtk_matrix_from_numpy(T_slice_to_ras))
    slice_node.UpdateMatrices()
    slice_widget.sliceLogic().FitSliceToAll()
    slicer.util.forceRenderAllViews()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step01-report", type=Path, default=STEP01_REPORT)
    parser.add_argument("--probe-pose", type=Path, help="4x4 T_world_from_probe_m (.npy/.npz/.txt/.csv)")
    parser.add_argument("--probe-pose-key", help="key for --probe-pose .npz")
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument(
        "--sequence-frame",
        type=int,
        default=DEFAULT_SEQUENCE_FRAME,
        help="read final pose from a scan npz frame",
    )
    parser.add_argument("--sequence-pose-kind", choices=("ee", "probe"), default="ee")
    parser.add_argument(
        "--probe-rotation-offset-deg",
        type=float,
        default=DEFAULT_PROBE_ROTATION_OFFSET_DEG,
        help="post-multiply the probe pose by this local Z rotation",
    )
    parser.add_argument("--world-from-phantom", type=Path, help="optional 4x4 T_world_from_phantom_m")
    parser.add_argument("--world-from-phantom-key", help="key for --world-from-phantom .npz")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME_PATH)
    parser.add_argument("--slice-view", default="Red")
    parser.add_argument(
        "--plane-mode",
        choices=("probe-xz", "probe-xy"),
        default="probe-xz",
        help="which probe-local plane is the ultrasound image plane",
    )
    parser.add_argument(
        "--plane-offset-y-mm",
        type=float,
        default=0.0,
        help="shift the slice point along probe local +Y before sampling",
    )
    parser.add_argument("--slice-width-mm", type=float, default=DEFAULT_SLICE_WIDTH_MM)
    parser.add_argument("--slice-height-mm", type=float, default=DEFAULT_SLICE_HEIGHT_MM)
    parser.add_argument("--slice-axial-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--slice-lateral-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--no-display-rot180",
        action="store_true",
        help="do not rotate the exported CBCT slice PNG by 180 degrees",
    )
    parser.add_argument("--no-image-export", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("  ignored unknown args:", unknown)

    report = json.loads(args.step01_report.read_text(encoding="utf-8"))
    phantom_origin_lps_mm = np.array(
        report["phantom_centered_frame"]["origin_lps_mm"],
        dtype=float,
    )

    if args.probe_pose is not None:
        T_world_from_probe_m = load_transform_4x4(args.probe_pose, args.probe_pose_key)
        T_world_from_probe_m = T_world_from_probe_m @ probe_rotation_offset(
            args.probe_rotation_offset_deg
        )
        pose_source = str(args.probe_pose)
    else:
        T_world_from_probe_m = pose_from_sequence(
            args.sequence,
            args.sequence_frame,
            args.sequence_pose_kind,
            args.probe_rotation_offset_deg,
        )
        pose_source = (
            f"{args.sequence}, frame {args.sequence_frame}, kind {args.sequence_pose_kind}, "
            f"probe Rz {args.probe_rotation_offset_deg:g} deg"
        )

    if args.world_from_phantom is not None:
        T_world_from_phantom_m = load_transform_4x4(
            args.world_from_phantom,
            args.world_from_phantom_key,
        )
        placement_source = str(args.world_from_phantom)
    else:
        T_world_from_phantom_m = default_world_from_phantom_centered_m()
        placement_source = "default measured placement @ Rx90*Rz180"

    T_phantom_from_probe_mm = probe_pose_in_phantom_centered_mm(
        T_world_from_probe_m,
        T_world_from_phantom_m,
    )
    plane = plane_from_probe_pose(T_phantom_from_probe_mm, args.plane_mode, args.plane_offset_y_mm)
    axes = slicer_axes_ras(plane, phantom_origin_lps_mm)
    T_slice_to_ras = slice_to_ras_numpy(axes)

    out = {
        "pose_source": pose_source,
        "placement_source": placement_source,
        "T_world_from_probe_m": T_world_from_probe_m.tolist(),
        "probe_rotation_offset_deg": args.probe_rotation_offset_deg,
        "T_world_from_phantom_centered_m": T_world_from_phantom_m.tolist(),
        "T_phantom_centered_from_probe_mm": T_phantom_from_probe_mm.tolist(),
        "plane_phantom_centered": {
            "point_mm": plane["point_centered_mm"].tolist(),
            "normal_unit": plane["normal_centered"].tolist(),
            "lateral_unit": plane["lateral_centered"].tolist(),
            "axial_unit": plane["axial_centered"].tolist(),
            "meaning": plane["meaning"],
            "plane_mode": args.plane_mode,
            "plane_offset_y_mm": plane["plane_offset_y_mm"],
        },
        "slicer_ras": {
            "point_ras_mm": axes["point_ras_mm"].tolist(),
            "x_axis_ras": axes["x_axis_ras"].tolist(),
            "y_axis_ras": axes["y_axis_ras"].tolist(),
            "z_axis_ras": axes["z_axis_ras"].tolist(),
            "SliceToRAS": T_slice_to_ras.tolist(),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_outputs = {}
    if not args.no_image_export and args.sequence_frame is not None:
        image_outputs = export_slice_images(
            volume_path=args.volume_path,
            report=report,
            plane=plane,
            sequence=args.sequence,
            sequence_frame=args.sequence_frame,
            out_dir=args.out_dir,
            width_mm=args.slice_width_mm,
            height_mm=args.slice_height_mm,
            axial_sign=args.slice_axial_sign,
            lateral_sign=args.slice_lateral_sign,
            display_rot180=not args.no_display_rot180,
        )
        out["image_export"] = {
            **image_outputs,
            "width_mm": args.slice_width_mm,
            "height_mm": args.slice_height_mm,
            "axial_sign": args.slice_axial_sign,
            "lateral_sign": args.slice_lateral_sign,
            "display_rot180": not args.no_display_rot180,
            "volume_path": str(args.volume_path),
        }

    out_path = args.out_dir / "slice_plane_from_probe.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[slicer 3.0 step 2] probe pose -> phantom-centred slice plane")
    print("  pose source:", pose_source)
    print("  placement  :", placement_source)
    print("  point centered mm :", plane["point_centered_mm"])
    print("  normal centered   :", plane["normal_centered"])
    print("  lateral centered  :", plane["lateral_centered"])
    print("  axial centered    :", plane["axial_centered"])
    print("  SliceToRAS:")
    print(T_slice_to_ras)
    if image_outputs:
        print("  wrote CBCT slice:", image_outputs["cbct_slice_png"])
        print("  wrote US frame  :", image_outputs["us_frame_png"])
        print("  wrote compare   :", image_outputs["comparison_png"])
    print(f"  wrote {out_path}")

    maybe_show_in_slicer(T_slice_to_ras, args.volume_path, args.slice_view)


if __name__ == "__main__":
    main()
