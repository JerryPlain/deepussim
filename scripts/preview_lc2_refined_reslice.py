#!/usr/bin/env python
"""LC2-refine one recorded frame and compare its initial/refined CBCT fan reslices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lc2.forward import Context, FrameTargets  # noqa: E402
from lc2.register import register_frame  # noqa: E402
from lc2.us_fan import FanFit, unwrap_fan  # noqa: E402
from plot_script.plots_reslice.compare import _normalize_display  # noqa: E402
from reslice import pose as P  # noqa: E402
from reslice.fan import ProbeGeometry  # noqa: E402
from reslice.io import load_transform_4x4, load_volume_data  # noqa: E402
from preview_us_normal_reslice_region import (  # noqa: E402
    US_SPACING_MM,
    boundary_curves,
    cbct_bottom_tangent_reslice,
    fit_inner_arc,
)


def _display_fan_from_geometry(geometry: dict[str, np.ndarray | float]) -> FanFit:
    centre = np.asarray(geometry["centre"], dtype=float)
    fov_deg = np.rad2deg(
        float(geometry["angle_right"]) - float(geometry["angle_left"])
    )
    return FanFit(
        apex_px=(float(centre[0]), float(centre[1])),
        r0_px=float(geometry["inner_radius"]),
        r1_px=float(geometry["outer_radius"]),
        fov_deg=float(fov_deg),
        resid_px=float(geometry["median_residual"]),
    )


def _draw_boundary(ax, geometry: dict[str, np.ndarray | float]) -> None:
    inner, outer = boundary_curves(geometry)
    p_left = np.asarray(geometry["p_left"])
    p_right = np.asarray(geometry["p_right"])
    outer_left = np.asarray(geometry["outer_left"])
    outer_right = np.asarray(geometry["outer_right"])
    ax.plot(inner[:, 0], inner[:, 1], color="#00e5ff", lw=1.2)
    ax.plot(outer[:, 0], outer[:, 1], color="#7cff00", lw=1.2)
    ax.plot(
        [p_left[0], outer_left[0]], [p_left[1], outer_left[1]], color="#ff3bd4", lw=1.2
    )
    ax.plot(
        [p_right[0], outer_right[0]],
        [p_right[1], outer_right[1]],
        color="#ff3bd4",
        lw=1.2,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--placement", type=Path, required=True)
    ap.add_argument("--us-spacing-mm", type=float, default=US_SPACING_MM)
    ap.add_argument("--inner-arc-threshold", type=float, default=30.0)
    ap.add_argument("--max-trans-mm", type=float, default=8.0)
    ap.add_argument("--max-rot-deg", type=float, default=8.0)
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sequence = np.load(args.sequence, allow_pickle=True)
    us = np.asarray(sequence["images"][args.frame])
    world_from_ee = np.asarray(sequence["poses"][args.frame], dtype=float)
    world_from_probe = world_from_ee @ P.T_EE_FROM_PROBE
    world_from_phantom = load_transform_4x4(args.placement)
    init_pose = P.probe_pose_in_phantom_centered_mm(world_from_probe, world_from_phantom)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    affine = np.asarray(
        report["phantom_centered_frame"]["recon_affine_centered_from_ijk_mm"], dtype=float
    )
    volume = load_volume_data(args.volume)

    display_geometry = fit_inner_arc(us, threshold=args.inner_arc_threshold)
    fan = _display_fan_from_geometry(display_geometry)
    polar_geometry = ProbeGeometry(
        radius_mm=fan.r0_px * args.us_spacing_mm,
        fov_deg=fan.fov_deg,
        depth_mm=(fan.r1_px - fan.r0_px) * args.us_spacing_mm,
        n_lat=256,
        n_ax=512,
    )
    us_polar = unwrap_fan(us, fan, n_ax=polar_geometry.n_ax, n_lat=polar_geometry.n_lat)
    target = FrameTargets(index=args.frame, us_polar=us_polar, init_pose_mm=init_pose)
    context = Context(volume=volume, affine_centered=affine, geom=polar_geometry, frames=[target])

    result = register_frame(
        context,
        target,
        max_trans_mm=args.max_trans_mm,
        max_rot_deg=args.max_rot_deg,
        maxiter=args.maxiter,
    )
    refined_pose = np.asarray(result["refined_pose"], dtype=float)

    initial_cbct, initial_geometry, initial_debug = cbct_bottom_tangent_reslice(
        volume,
        affine,
        init_pose,
        us,
        inner_arc_threshold=args.inner_arc_threshold,
        us_spacing_mm=args.us_spacing_mm,
    )
    refined_cbct, refined_geometry, refined_debug = cbct_bottom_tangent_reslice(
        volume,
        affine,
        refined_pose,
        us,
        inner_arc_threshold=args.inner_arc_threshold,
        us_spacing_mm=args.us_spacing_mm,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.25), squeeze=False)
    ax_us, ax_init, ax_refined = axes[0]
    ax_us.imshow(_normalize_display(us), cmap="gray", vmin=0, vmax=1)
    ax_init.imshow(initial_cbct, cmap="gray", vmin=0, vmax=1)
    ax_refined.imshow(refined_cbct, cmap="gray", vmin=0, vmax=1)
    for ax, geometry in (
        (ax_us, display_geometry),
        (ax_init, initial_geometry),
        (ax_refined, refined_geometry),
    ):
        _draw_boundary(ax, geometry)
        ax.set_xlim(-0.5, us.shape[1] - 0.5)
        ax.set_ylim(us.shape[0] - 0.5, -0.5)
        ax.axis("off")

    ax_us.set_title(f"Real US | {args.sequence.stem} frame {args.frame}")
    ax_init.set_title(
        f"Robot-pose CBCT | LC2={float(result['lc2_before']):.4f} | "
        f"inside={100.0 * float(result['inside_before']):.1f}%"
    )
    ax_refined.set_title(
        f"LC2-refined CBCT | LC2={float(result['lc2_after']):.4f} | "
        f"inside={100.0 * float(result['inside_after']):.1f}%"
    )
    params = np.asarray(result["params"], dtype=float)
    fig.suptitle(
        f"Per-frame bounded LC2 | rotation xyz=({params[0]:+.2f}, {params[1]:+.2f}, "
        f"{params[2]:+.2f}) deg | translation xyz=({params[3]:+.2f}, {params[4]:+.2f}, "
        f"{params[5]:+.2f}) mm\n"
        f"US/CBCT grid={args.us_spacing_mm:.9f} mm/px | same pose chain and "
        "bottom-tangent fan logic | no resize/stretch",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    serializable = {
        "sequence": str(args.sequence),
        "frame": args.frame,
        "placement": str(args.placement),
        "us_spacing_mm": args.us_spacing_mm,
        "fan": {
            "apex_xy_px": list(fan.apex_px),
            "r0_px": fan.r0_px,
            "r1_px": fan.r1_px,
            "fov_deg": fan.fov_deg,
            "radius_mm": polar_geometry.radius_mm,
            "depth_mm": polar_geometry.depth_mm,
        },
        "bounds": {
            "max_trans_mm": args.max_trans_mm,
            "max_rot_deg": args.max_rot_deg,
            "maxiter": args.maxiter,
        },
        "lc2_before": float(result["lc2_before"]),
        "lc2_after": float(result["lc2_after"]),
        "inside_before": float(result["inside_before"]),
        "inside_after": float(result["inside_after"]),
        "correction_rx_ry_rz_deg_tx_ty_tz_mm": result["params"],
        "init_T_phantom_from_probe_mm": init_pose.tolist(),
        "refined_T_phantom_from_probe_mm": refined_pose.tolist(),
        "initial_surface_pixel_rc": initial_debug["surface_pixel_rc"],
        "refined_surface_pixel_rc": refined_debug["surface_pixel_rc"],
        "post_reslice_resize": False,
    }
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
