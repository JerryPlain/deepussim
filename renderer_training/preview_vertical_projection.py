#!/usr/bin/env python
"""Project CBCT labels for a probe placed above the phantom and pointing vertically down."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renderer_training.project_labels_to_us import (                     # noqa: E402
    CALIBRATED_DISPLAY_SHAPE,
    DEFAULT_DISPLAY_FAN,
    DEFAULT_PROBE_GEOMETRY,
    sector_zoom_pair,
)
from reslice import pose as posemod                                      # noqa: E402
from reslice.fan import reslice_fan, scan_convert_fan                    # noqa: E402
from reslice.frame import affine_from_sitk                               # noqa: E402
from reslice.io import load_volume_data                                  # noqa: E402

LIVER_ID = 2


def _world_vertical_pose(x_m: float, y_m: float, surface_z_m: float) -> np.ndarray:
    """Probe +Z is world -Z; probe origin is placed on the top surface."""
    world_from_probe = np.eye(4, dtype=np.float64)
    world_from_probe[:3, :3] = np.diag([1.0, -1.0, -1.0])
    world_from_probe[:3, 3] = [x_m, y_m, surface_z_m]
    return posemod.probe_pose_in_phantom_centered_mm(
        world_from_probe, posemod.default_world_from_phantom_centered_m()
    )


def _surface_points_world(mesh_path: Path) -> np.ndarray:
    import trimesh

    vertices_mm = np.asarray(trimesh.load(mesh_path, process=False).vertices, dtype=np.float64)
    world_from_phantom = posemod.default_world_from_phantom_centered_m()
    return (
        vertices_mm @ world_from_phantom[:3, :3].T / 1000.0
        + world_from_phantom[:3, 3]
    )


def _normalise(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1.0, 99.0])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--surface", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--surface-radius-mm", type=float, default=8.0)
    ap.add_argument(
        "--position-mode",
        choices=("center", "liver-max"),
        default="center",
        help="center = top geometric centre; liver-max = vertical top pose with most liver",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures" / "9_label_projection_check" / "vertical_probe_mask.png",
    )
    args = ap.parse_args()

    intensity = load_volume_data(args.volume)
    labels = load_volume_data(args.labels)
    import SimpleITK as sitk

    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))
    surface_world = _surface_points_world(args.surface)

    # A decimated XY cloud is sufficient for finding the local top surface and avoids building
    # a multi-million-point tree from the binary STL's repeated triangle vertices.
    decimated = surface_world[::8]
    from scipy.spatial import cKDTree

    tree = cKDTree(decimated[:, :2])
    lower = np.quantile(surface_world[:, :2], 0.18, axis=0)
    upper = np.quantile(surface_world[:, :2], 0.82, axis=0)
    xs = np.linspace(lower[0], upper[0], args.grid)
    ys = np.linspace(lower[1], upper[1], args.grid)
    radius_m = args.surface_radius_mm / 1000.0

    def evaluate_position(x_m: float, y_m: float):
        neighbours = tree.query_ball_point([x_m, y_m], radius_m)
        if not neighbours:
            return None
        surface_z_m = float(decimated[neighbours, 2].max())
        pose = _world_vertical_pose(float(x_m), float(y_m), surface_z_m)
        polar_label = reslice_fan(labels, affine, pose, DEFAULT_PROBE_GEOMETRY, order=0)
        label_map = scan_convert_fan(
            polar_label,
            CALIBRATED_DISPLAY_SHAPE,
            **DEFAULT_DISPLAY_FAN,
            order=0,
            cval=0.0,
        )
        coverage = float(np.mean(label_map == LIVER_ID))
        return (coverage, float(x_m), float(y_m), surface_z_m, pose)

    if args.position_mode == "center":
        centre_xy = (lower + upper) / 2.0
        result = evaluate_position(float(centre_xy[0]), float(centre_xy[1]))
        evaluated = [] if result is None else [result]
        best = result
    else:
        best = None
        evaluated = []
        for x_m in xs:
            for y_m in ys:
                result = evaluate_position(float(x_m), float(y_m))
                if result is None:
                    continue
                evaluated.append(result)
                if best is None or result[0] > best[0]:
                    best = result

    if best is None:
        raise SystemExit("no top-surface candidate found; increase --surface-radius-mm")

    coverage, x_m, y_m, z_m, pose = best
    cbct, label_map = sector_zoom_pair(intensity, labels, affine, pose, CALIBRATED_DISPLAY_SHAPE)
    liver = label_map == LIVER_ID
    fan_support = scan_convert_fan(
        np.ones(
            (DEFAULT_PROBE_GEOMETRY.n_ax, DEFAULT_PROBE_GEOMETRY.n_lat), dtype=np.uint8
        ),
        CALIBRATED_DISPLAY_SHAPE,
        **DEFAULT_DISPLAY_FAN,
        order=0,
        cval=0.0,
    ).astype(bool)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.7))
    axes[0].imshow(cbct, cmap="gray")
    axes[0].contour(fan_support.astype(float), [0.5], colors="cyan", linewidths=0.9)
    axes[0].set_title("CBCT content · cyan = full fan")
    axes[1].imshow(label_map, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("all projected labels")
    axes[2].imshow(_normalise(cbct), cmap="gray")
    overlay = np.zeros((*liver.shape, 4), dtype=np.float32)
    overlay[liver] = (1.0, 0.35, 0.0, 0.52)
    axes[2].imshow(overlay)
    if liver.any():
        axes[2].contour(liver.astype(float), [0.5], colors="yellow", linewidths=0.8)
    axes[2].set_title(f"CBCT + liver · {100 * coverage:.1f}%")
    axes[3].imshow(liver, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("binary liver mask · 660×880")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"Probe vertical above phantom ({args.position_mode}) · beam = world −Z · "
        f"contact xyz=({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) m",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    np.savez_compressed(
        args.out.with_name(f"{args.out.stem}_data.npz"),
        cbct=cbct.astype(np.float32),
        label_map=label_map.astype(np.int16),
        liver_mask=liver.astype(np.uint8),
        fan_support=fan_support.astype(np.uint8),
        pose_cbct_mm=pose,
        pose_world_xyz_m=np.array([x_m, y_m, z_m]),
    )
    evaluated.sort(reverse=True, key=lambda item: item[0])
    print(f"wrote {args.out}")
    print(f"pose world xyz m: {[round(v, 6) for v in (x_m, y_m, z_m)]}")
    print("beam direction world: [0, 0, -1]")
    print(f"position mode: {args.position_mode}")
    print(f"liver coverage: {100 * coverage:.2f}%")
    print(f"candidates evaluated: {len(evaluated)}")
    print(f"top coverages: {[round(100 * item[0], 2) for item in evaluated[:5]]}%")


if __name__ == "__main__":
    main()
