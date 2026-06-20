"""Forward model + scoring context for LC2 refinement.

For a sequence we fit the US fan once, then for each selected in-contact frame keep its
unwrapped (polar) US image and its calibration init pose. Scoring a pose = reslice the
CBCT on the same fan (``reslice.fan``) and compute LC2 against the unwrapped US.

The CBCT slicing is the `reslice` logic; the US fan-fit / unwrap (`lc2.us_fan`) and the
LC2 metric (`lc2.metric`) are reimplemented in this package — no `deepussim` dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import lc2  # noqa: F401  (sets up sys.path for reslice)
from reslice import pose as posemod
from reslice.fan import ProbeGeometry, fraction_inside, reslice_fan
from reslice.io import load_transform_4x4, load_volume_data

from lc2.metric import lc2_similarity
from lc2.us_fan import contact_envelope, fit_fan_geometry, fit_fan_pixels, unwrap_fan


@dataclass
class FrameTargets:
    """Everything needed to score one frame: its unwrapped US and its init pose."""

    index: int
    us_polar: np.ndarray
    init_pose_mm: np.ndarray


@dataclass
class Context:
    """Shared state for scoring poses against a sequence's US frames."""

    volume: np.ndarray
    affine_centered: np.ndarray
    geom: ProbeGeometry
    frames: list[FrameTargets] = field(default_factory=list)

    def cbct_fan(self, pose_mm: np.ndarray) -> np.ndarray:
        """Reslice the CBCT on the probe fan for one pose (the forward model)."""
        return reslice_fan(self.volume, self.affine_centered, pose_mm, self.geom)

    def score(self, pose_mm: np.ndarray, us_polar: np.ndarray) -> float:
        """LC2 similarity between the resliced CBCT fan and the unwrapped US."""
        return float(lc2_similarity(us_polar, self.cbct_fan(pose_mm)))

    def fraction_inside(self, pose_mm: np.ndarray) -> float:
        """Anti-graze guard: fraction of the resliced fan that landed inside the volume."""
        return fraction_inside(self.cbct_fan(pose_mm))


def fit_us_fan(images, contact, us_spacing_mm: float, n_lat: int = 256, n_ax: int = 512):
    """Fit the probe fan from a sequence: returns ``(fan_pixels, ProbeGeometry)``.

    The fan is a property of the probe, so it is normally fitted ONCE from a clean
    sequence (e.g. scan1) and reused for all sequences via :func:`prepare`'s ``fan`` /
    ``geom`` arguments — per-sequence fitting can be degenerate on some envelopes.
    """
    env = contact_envelope(images, np.asarray(contact, dtype=bool))
    fan = fit_fan_pixels(env)
    fan_geom = fit_fan_geometry(env, us_spacing_mm, n_lat=n_lat, n_ax=n_ax)
    geom = ProbeGeometry(
        radius_mm=fan_geom.radius_mm, fov_deg=fan_geom.fov_deg, depth_mm=fan_geom.depth_mm,
        n_lat=n_lat, n_ax=n_ax,
    )
    return fan, geom


def prepare(
    sequence: Path,
    report: dict,
    volume_path: Path,
    world_from_phantom: Path,
    us_spacing_mm: float,
    n_frames: int,
    n_lat: int = 256,
    n_ax: int = 512,
    probe_rotation_deg: float = 0.0,
    fan=None,
    geom: ProbeGeometry | None = None,
) -> Context:
    """Load the volume and build the per-frame targets + init poses.

    Pass a precomputed ``fan`` + ``geom`` (from :func:`fit_us_fan`) to reuse one probe
    fan across sequences; otherwise the fan is fitted from this sequence.
    """
    frame_key = "recon" if Path(volume_path).suffix.lower() == ".mhd" else "dicom"
    affine_centered = np.array(
        report["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"], dtype=float
    )
    volume = load_volume_data(Path(volume_path))
    T_world_from_phantom = load_transform_4x4(Path(world_from_phantom))

    seq = np.load(sequence, allow_pickle=True)
    images, contact = seq["images"], np.asarray(seq["contact"], dtype=bool)

    if fan is None or geom is None:
        fan, geom = fit_us_fan(images, contact, us_spacing_mm, n_lat=n_lat, n_ax=n_ax)

    contact_idx = np.where(contact)[0]
    sel = contact_idx[np.linspace(0, len(contact_idx) - 1, min(n_frames, len(contact_idx)), dtype=int)]

    frames: list[FrameTargets] = []
    for i in sel:
        us_polar = unwrap_fan(images[int(i)], fan, n_ax=n_ax, n_lat=n_lat)
        T_world_from_probe = posemod.pose_from_sequence(Path(sequence), int(i), "ee", probe_rotation_deg)
        init = posemod.probe_pose_in_phantom_centered_mm(T_world_from_probe, T_world_from_phantom)
        frames.append(FrameTargets(index=int(i), us_polar=us_polar, init_pose_mm=init))

    return Context(volume=volume, affine_centered=affine_centered, geom=geom, frames=frames)
