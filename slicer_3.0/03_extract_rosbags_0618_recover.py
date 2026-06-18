"""Recover and extract 2026-06-18 ROS1 bags with damaged indexes.

The normal ``scripts/extract_rosbags.py`` path uses ``rosbags.highlevel.AnyReader``,
which requires a valid ROS1 bag index. The 0618 bags currently have damaged index
records, so this script scans chunk records sequentially and ignores the tail index.

Outputs match the existing sequence format:

    data/sequences_20260618/scan2.npz
    data/sequences_20260618/scan2_us_png/frame_0000.png

Each ``.npz`` contains ``images``, ``poses``, ``contact``, ``stamps``,
``mean_intensity`` and ``sync_dt_s``.
"""
from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag1,
    MessageDefinition,
    MessageDefinitionFormat,
)
from rosbags.rosbag1.reader import (
    Header,
    RecordType,
    decompressors,
    normalize,
    normalize_msgtype,
    read_uint32,
)
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from deepussim.data.rosbag import (
    IMAGE_TOPIC,
    POSE_TOPIC,
    _decode_image_msg,
    _stamp_seconds,
    pose_matrix_from_ros,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "deepussim-main"
DEFAULT_BAGS = [
    ROOT / "data_0618" / "us" / "0618" / "scan2.bag",
    ROOT / "data_0618" / "us" / "0618" / "scan3.bag",
]
DEFAULT_OUT = PROJECT / "data" / "sequences_20260618"


@dataclass
class RawMessage:
    conn: int
    time_ns: int
    raw: bytes


def read_data_record_payload(src) -> bytes:
    size = read_uint32(src)
    return src.read(size)


def read_connection_record(src, header: Header, owner) -> Connection:
    conn = header.get_uint32("conn")
    topic = normalize(header.get_string("topic"))
    data = Header.read(src)
    msgtype = normalize_msgtype(data.get_string("type"))
    msgdef = MessageDefinition(MessageDefinitionFormat.MSG, data.get_string("message_definition"))
    md5sum = data.get_string("md5sum")
    callerid = data.get_string("callerid") if "callerid" in data else None
    latching_raw = data.get_string("latching") if "latching" in data else None
    latching = int(latching_raw) if latching_raw not in (None, "") else None
    return Connection(
        conn,
        topic,
        msgtype,
        msgdef,
        md5sum,
        0,
        ConnectionExtRosbag1(callerid, latching),
        owner,
    )


def scan_chunk(src, chunk_data: bytes, connections: dict[int, Connection]) -> list[RawMessage]:
    messages: list[RawMessage] = []
    bio = io.BytesIO(chunk_data)
    while bio.tell() < len(chunk_data):
        try:
            header = Header.read(bio)
        except Exception:
            break
        op = header.get_uint8("op")
        if op == RecordType.CONNECTION:
            conn = read_connection_record(bio, header, None)
            connections[conn.id] = conn
        elif op == RecordType.MSGDATA:
            payload = read_data_record_payload(bio)
            messages.append(
                RawMessage(
                    conn=header.get_uint32("conn"),
                    time_ns=header.get_time("time"),
                    raw=payload,
                )
            )
        else:
            _ = bio.seek(read_uint32(bio), os.SEEK_CUR)
    return messages


def scan_bag(path: Path) -> tuple[dict[int, Connection], list[RawMessage]]:
    connections: dict[int, Connection] = {}
    messages: list[RawMessage] = []

    with path.open("rb") as src:
        magic = src.readline().decode(errors="replace")
        if not magic.startswith("#ROSBAG V2.0"):
            raise ValueError(f"{path} is not a ROS1 V2.0 bag")
        _bag_header = Header.read(src, RecordType.BAGHEADER)
        _ = src.seek(read_uint32(src), os.SEEK_CUR)

        while True:
            pos = src.tell()
            try:
                header = Header.read(src)
            except Exception:
                break
            try:
                op = header.get_uint8("op")
            except Exception:
                break

            if op == RecordType.CHUNK:
                compression = header.get_string("compression")
                datasize = read_uint32(src)
                raw = src.read(datasize)
                chunk = decompressors[compression](raw)
                messages.extend(scan_chunk(src, chunk, connections))
            elif op == RecordType.CONNECTION:
                conn = read_connection_record(src, header, None)
                connections[conn.id] = conn
            else:
                try:
                    _ = src.seek(read_uint32(src), os.SEEK_CUR)
                except Exception:
                    src.seek(pos)
                    break

    return connections, messages


def make_typestore(connections: dict[int, Connection]):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    types = {}
    for conn in connections.values():
        if conn.msgdef.data:
            types.update(get_types_from_msg(conn.msgdef.data, conn.msgtype))
    typestore.register(types)
    return typestore


def to_sequence(path: Path, dark_threshold: float, max_sync_dt_s: float | None):
    connections, raw_messages = scan_bag(path)
    typestore = make_typestore(connections)
    conn_by_topic = {conn.topic.lstrip("/"): conn for conn in connections.values()}
    image_topic_key = IMAGE_TOPIC.lstrip("/")
    pose_topic_key = POSE_TOPIC.lstrip("/")
    if image_topic_key not in conn_by_topic:
        raise ValueError(f"{path}: missing {IMAGE_TOPIC}; topics={list(conn_by_topic)}")
    if pose_topic_key not in conn_by_topic:
        raise ValueError(f"{path}: missing {POSE_TOPIC}; topics={list(conn_by_topic)}")

    img_conn = conn_by_topic[image_topic_key]
    pose_conn = conn_by_topic[pose_topic_key]

    img_t: list[float] = []
    img_arr: list[np.ndarray] = []
    pose_t: list[float] = []
    pose_mat: list[np.ndarray] = []

    for raw in raw_messages:
        conn = connections.get(raw.conn)
        if conn is None or conn.topic.lstrip("/") not in (image_topic_key, pose_topic_key):
            continue
        msg = typestore.deserialize_ros1(raw.raw, conn.msgtype)
        t = _stamp_seconds(msg.header.stamp)
        if conn.id == img_conn.id:
            img_t.append(t)
            img_arr.append(_decode_image_msg(msg))
        elif conn.id == pose_conn.id:
            p, q = msg.pose.position, msg.pose.orientation
            pose_t.append(t)
            pose_mat.append(pose_matrix_from_ros((p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))

    if not img_t or not pose_t:
        raise ValueError(f"{path}: recovered {len(img_t)} images and {len(pose_t)} poses")

    pose_t_arr = np.asarray(pose_t)
    order = np.argsort(pose_t_arr)
    pose_t_arr = pose_t_arr[order]
    pose_mat = [pose_mat[i] for i in order]
    img_t_arr = np.asarray(img_t)
    nn = np.searchsorted(pose_t_arr, img_t_arr)
    nn = np.clip(nn, 1, len(pose_t_arr) - 1)
    left = pose_t_arr[nn - 1]
    right = pose_t_arr[nn]
    nn = np.where(img_t_arr - left <= right - img_t_arr, nn - 1, nn)

    images = []
    poses = []
    stamps = []
    means = []
    sync_dt = []
    contact = []
    dropped = 0
    for t, image, j in zip(img_t_arr, img_arr, nn):
        dt = float(abs(t - pose_t_arr[j]))
        if max_sync_dt_s is not None and dt > max_sync_dt_s:
            dropped += 1
            continue
        mean = float(image.mean())
        images.append(image)
        poses.append(pose_mat[j])
        stamps.append(float(t))
        means.append(mean)
        sync_dt.append(dt)
        contact.append(mean >= dark_threshold)

    meta = {
        "bag": str(path),
        "n_images": len(img_t),
        "n_poses": len(pose_t),
        "n_frames": len(images),
        "n_dark": int(np.size(contact) - np.count_nonzero(contact)),
        "n_dropped_unsynced": dropped,
        "duration_s": float(img_t_arr.max() - img_t_arr.min()),
    }
    return {
        "name": path.stem,
        "images": np.stack(images),
        "poses": np.stack(poses),
        "contact": np.asarray(contact, dtype=bool),
        "stamps": np.asarray(stamps, dtype=float),
        "mean_intensity": np.asarray(means, dtype=float),
        "sync_dt_s": np.asarray(sync_dt, dtype=float),
        "meta": meta,
    }


def save_sequence(seq: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seq['name']}.npz"
    np.savez_compressed(
        path,
        images=seq["images"],
        poses=seq["poses"],
        contact=seq["contact"],
        stamps=seq["stamps"],
        mean_intensity=seq["mean_intensity"],
        sync_dt_s=seq["sync_dt_s"],
    )
    return path


def save_all_png(seq: dict, out_dir: Path) -> Path:
    png_dir = out_dir / f"{seq['name']}_us_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(seq["images"]):
        tag = "contact" if seq["contact"][i] else "dark"
        Image.fromarray(image).save(png_dir / f"frame_{i:04d}_{tag}.png")
    return png_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", nargs="*", type=Path, default=DEFAULT_BAGS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dark-threshold", type=float, default=6.0)
    parser.add_argument("--max-sync-dt", type=float, default=None)
    args = parser.parse_args()

    for bag in args.bags:
        seq = to_sequence(bag, args.dark_threshold, args.max_sync_dt)
        npz = save_sequence(seq, args.out)
        png_dir = save_all_png(seq, args.out)
        sync = seq["sync_dt_s"]
        meta = seq["meta"]
        print(f"\n{seq['name']}: {meta['n_frames']} frames "
              f"({meta['n_dark']} dark), {meta['duration_s']:.1f}s")
        print(f"  images={meta['n_images']} poses={meta['n_poses']} "
              f"sync_dt median={np.median(sync) * 1e3:.1f}ms max={sync.max() * 1e3:.1f}ms")
        print(f"  saved {npz}")
        print(f"  saved all PNGs to {png_dir}")


if __name__ == "__main__":
    main()
