"""Extract the real US sequences from ROS1 rosbags (Stage 1, doc v2 §4.3).

Each bag carries exactly two topics (confirmed on phantom.bag / phantom1.bag):

  /frame_grabber/us_img    sensor_msgs/Image          8UC3, 660x880   -> US B-mode frame
  /fr3/state/current_pose  geometry_msgs/PoseStamped  frame 'base', m -> R T_E(t)

The pose stream (~94 Hz) is denser than the image stream (~15 Hz), so each image is
matched to the **nearest pose by header stamp** (image and pose rates differ). The
matched pose is ``T_base_from_ee`` in metres — exactly the ``RTE(t)`` segment of the
US-pixel -> CBCT-voxel chain (doc §4.2); compose it with ``T_EE_FROM_PROBE`` downstream.

Dark / non-contact frames (probe lifted off, no acoustic coupling) are **tagged**
(``contact=False``), never silently dropped (doc §4.3): they are no-contact negatives for
the dataset, lift-off supervision for the renderer, and should be filtered before LC2.
Each frame keeps its grayscale mean so the dark threshold can be re-tuned without re-reading.

Requires the pure-Python ``rosbags`` library (no ROS1 install): ``pip install rosbags``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import pose_from_pos_quat

IMAGE_TOPIC = "/frame_grabber/us_img"
POSE_TOPIC = "/fr3/state/current_pose"


# --- pure helpers (unit-tested without a bag) -----------------------------

def pose_matrix_from_ros(pos_xyz, quat_xyzw) -> np.ndarray:
    """4x4 ``T_base_from_ee`` from a ROS position + quaternion.

    ROS quaternions are ``(x, y, z, w)``; :mod:`deepussim.geometry` uses scalar-first
    ``(w, x, y, z)``, so we reorder before assembling the transform.
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    return pose_from_pos_quat(np.asarray(pos_xyz, dtype=float), (w, x, y, z))


def nearest_indices(query_t: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    """For each ``query_t``, the index of the closest value in sorted ``ref_t``."""
    ref_t = np.asarray(ref_t, dtype=float)
    query_t = np.asarray(query_t, dtype=float)
    pos = np.searchsorted(ref_t, query_t)
    pos = np.clip(pos, 1, len(ref_t) - 1)
    left, right = ref_t[pos - 1], ref_t[pos]
    return np.where(query_t - left <= right - query_t, pos - 1, pos)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse a decoded frame to a single-channel uint8 image.

    Channels are not identical on this hardware (the B-mode carries coloured overlay),
    so use Rec.601 luminance rather than picking one channel.
    """
    a = np.asarray(image)
    if a.ndim == 2:
        return a.astype(np.uint8)
    r, g, b = a[..., 0].astype(float), a[..., 1].astype(float), a[..., 2].astype(float)
    return (0.299 * r + 0.587 * g + 0.114 * b).round().clip(0, 255).astype(np.uint8)


# --- records --------------------------------------------------------------

@dataclass
class UsFrame:
    """One image matched to its nearest pose."""

    t: float                 # image header stamp (seconds)
    image: np.ndarray        # (H, W) grayscale uint8
    pose: np.ndarray         # 4x4 T_base_from_ee (RTE(t)), metres
    contact: bool            # False => dark / non-contact (lift-off)
    meta: dict = field(default_factory=dict)  # mean_intensity, sync_dt_s, pose_t


@dataclass
class UsSequence:
    name: str
    frames: list[UsFrame]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def images(self) -> np.ndarray:
        return np.stack([f.image for f in self.frames])

    def poses(self) -> np.ndarray:
        return np.stack([f.pose for f in self.frames])

    def contact_mask(self) -> np.ndarray:
        return np.array([f.contact for f in self.frames], dtype=bool)


# --- decoding -------------------------------------------------------------

def _decode_image_msg(msg) -> np.ndarray:
    """sensor_msgs/Image -> grayscale uint8 (H, W). Handles mono8 / rgb8 / 8UC3."""
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    channels = max(1, buf.size // (h * w))
    arr = buf.reshape(h, w, channels) if channels > 1 else buf.reshape(h, w)
    return to_grayscale(arr)


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec if hasattr(stamp, "nanosec") else stamp.nsec) * 1e-9


# --- main entry point -----------------------------------------------------

def extract_sequence(
    bag_path,
    dark_threshold: float = 6.0,
    image_topic: str = IMAGE_TOPIC,
    pose_topic: str = POSE_TOPIC,
    max_sync_dt_s: float | None = None,
) -> UsSequence:
    """Read one bag into a pose-synced, dark-tagged :class:`UsSequence`.

    Args:
        bag_path: path to a ROS1 ``.bag`` (or a ``rosbags`` directory).
        dark_threshold: a frame is tagged ``contact=False`` when its grayscale mean is
            below this (lift-off / no coupling). Each frame keeps ``mean_intensity`` so
            the cut can be re-tuned without re-reading the bag.
        image_topic, pose_topic: override if a future bag renames the topics.
        max_sync_dt_s: if set, drop images whose nearest pose is farther than this in
            time (otherwise keep every image and record ``sync_dt_s`` in its meta).
    """
    from rosbags.highlevel import AnyReader  # local import: optional dependency

    bag_path = Path(bag_path)
    img_t: list[float] = []
    img_arr: list[np.ndarray] = []
    pose_t: list[float] = []
    pose_mat: list[np.ndarray] = []

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic in (image_topic, pose_topic)]
        for conn, _ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = _stamp_seconds(msg.header.stamp)
            if conn.topic == image_topic:
                img_t.append(t)
                img_arr.append(_decode_image_msg(msg))
            else:
                p, q = msg.pose.position, msg.pose.orientation
                pose_t.append(t)
                pose_mat.append(pose_matrix_from_ros((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))

    if not img_t:
        raise ValueError(f"no messages on {image_topic!r} in {bag_path}")
    if not pose_t:
        raise ValueError(f"no messages on {pose_topic!r} in {bag_path}")

    # Poses arrive time-ordered, but sort defensively before nearest-neighbour matching.
    pose_t_arr = np.asarray(pose_t)
    order = np.argsort(pose_t_arr)
    pose_t_arr = pose_t_arr[order]
    pose_mat = [pose_mat[i] for i in order]

    img_t_arr = np.asarray(img_t)
    nn = nearest_indices(img_t_arr, pose_t_arr)

    frames: list[UsFrame] = []
    n_dark = n_dropped = 0
    for k, (t, image, j) in enumerate(zip(img_t_arr, img_arr, nn)):
        sync_dt = float(abs(t - pose_t_arr[j]))
        if max_sync_dt_s is not None and sync_dt > max_sync_dt_s:
            n_dropped += 1
            continue
        mean_intensity = float(image.mean())
        contact = mean_intensity >= dark_threshold
        n_dark += not contact
        frames.append(UsFrame(
            t=float(t), image=image, pose=pose_mat[j], contact=contact,
            meta={"mean_intensity": mean_intensity, "sync_dt_s": sync_dt,
                  "pose_t": float(pose_t_arr[j])},
        ))

    meta = {
        "bag": str(bag_path),
        "image_topic": image_topic,
        "pose_topic": pose_topic,
        "n_images": len(img_t),
        "n_poses": len(pose_t),
        "n_frames": len(frames),
        "n_dark": int(n_dark),
        "n_dropped_unsynced": int(n_dropped),
        "dark_threshold": dark_threshold,
        "duration_s": float(img_t_arr.max() - img_t_arr.min()),
        "pose_frame": "base", "pose_units": "m", "pose_is": "T_base_from_ee (RTE(t))",
    }
    return UsSequence(name=bag_path.stem, frames=frames, meta=meta)
