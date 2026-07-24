#!/usr/bin/env python
"""Verify the files locked by the validated CBCT slice baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "cbct_slice_bottom_tangent_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--include-reference-inputs",
        action="store_true",
        help="also verify the dataset-specific 0612 geometry and placement files",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = [("implementation", manifest["locked_files"])]
    if args.include_reference_inputs:
        groups.append(("reference input", manifest["locked_reference_inputs"]))

    failures = 0
    print(f"Baseline: {manifest['baseline_id']} ({manifest['status']})")
    for group, entries in groups:
        for entry in entries:
            path = (REPO_ROOT / entry["path"]).resolve()
            if not path.is_file():
                print(f"MISSING  [{group}] {entry['path']}")
                failures += 1
                continue
            actual = _sha256(path)
            if actual.lower() != entry["sha256"].lower():
                print(f"CHANGED  [{group}] {entry['path']}")
                failures += 1
            else:
                print(f"OK       [{group}] {entry['path']}")

    if failures:
        print(f"Result: {failures} locked file(s) differ from the baseline.")
        return 1
    print("Result: baseline is intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
