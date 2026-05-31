"""Dataset records — one sample = the aligned multi-modal tuple from Step 5.

A ``Sample`` bundles everything produced for a single probe pose:
  - ``image``: rendered (or real) US image, (n_ax, n_lat) float in [0, 1]
  - ``pose``:  4x4 T_world_from_probe
  - ``mask``:  anatomy label map resliced from the CBCT label volume (or None)
  - ``force``: contact force vector/scalar from Genesis physics (or None for no-sim)
  - ``meta``:  free-form dict (sample index, source, geometry, ...)

``DatasetWriter`` streams samples to a directory as ``.npz`` files plus an
``index.json`` manifest, so a scale-up run can be resumed/inspected incrementally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    pose: np.ndarray
    mask: np.ndarray | None = None
    force: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def save_sample(path, sample: Sample) -> None:
    path = Path(path)
    arrays = {
        "image": np.asarray(sample.image, dtype=np.float32),
        "pose": np.asarray(sample.pose, dtype=np.float64),
        "meta_json": np.array(json.dumps(sample.meta)),
    }
    if sample.mask is not None:
        arrays["mask"] = np.asarray(sample.mask, dtype=np.int16)
    if sample.force is not None:
        arrays["force"] = np.asarray(sample.force, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_sample(path) -> Sample:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        return Sample(
            image=z["image"],
            pose=z["pose"],
            mask=z["mask"] if "mask" in z else None,
            force=z["force"] if "force" in z else None,
            meta=meta,
        )


class DatasetWriter:
    """Append-only writer producing ``<out_dir>/sample_000000.npz`` + ``index.json``."""

    def __init__(self, out_dir, meta: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []
        self.dataset_meta = meta or {}
        self._n = 0

    def add(self, sample: Sample) -> Path:
        fname = f"sample_{self._n:06d}.npz"
        save_sample(self.out_dir / fname, sample)
        self.index.append({"file": fname, **sample.meta})
        self._n += 1
        return self.out_dir / fname

    def close(self) -> None:
        manifest = {"count": self._n, "meta": self.dataset_meta, "samples": self.index}
        (self.out_dir / "index.json").write_text(json.dumps(manifest, indent=2))

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n
