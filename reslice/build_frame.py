"""CLI (step 1): build the phantom-centred CBCT frame for a DICOM/recon pair.

    python -m reslice.build_frame \
        --dicom-dir data/cbct_20260612/DICOM1 \
        --recon data/cbct_20260612/CBCT.mhd \
        --phantom-origin-lps-mm 0 0 0 \
        --out-dir reslice/outputs/frame_origin000

Writes ``physical_frame_report.json`` (the phantom-centred affines that
``reslice.slice`` reads). Re-run once per DICOM/recon pair.
"""
from __future__ import annotations

# Allow running both as a module (`python -m reslice.build_frame`) and as a script.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "reslice"

import argparse
import json
from pathlib import Path

import numpy as np

from reslice.frame import build_report, write_report

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICOM_DIR = REPO_ROOT / "data" / "cbct_20260612" / "DICOM1"
DEFAULT_RECON = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "frame_origin000"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    ap.add_argument("--recon", type=Path, default=DEFAULT_RECON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--phantom-origin-lps-mm", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="phantom local origin in DICOM-LPS mm (default 0 0 0 = simulator phantom origin)",
    )
    args = ap.parse_args()

    report = build_report(args.dicom_dir, args.recon, np.array(args.phantom_origin_lps_mm))
    path = write_report(report, args.out_dir)

    pcf = report["phantom_centered_frame"]
    print(f"[build_frame] dicom : {args.dicom_dir}")
    print(f"[build_frame] recon : {args.recon}")
    print(f"[build_frame] phantom origin LPS mm: {pcf['origin_lps_mm']}")
    print(f"[build_frame] recon size ijk       : {report['recon']['size_ijk']}")
    print(f"[build_frame] recon det / orth err : "
          f"{report['recon']['det_direction']:.4f} / {report['recon']['max_orthogonality_error']:.2e}")
    print("[build_frame] recon affine centred (mm <- ijk):")
    print(np.array(pcf["recon_affine_centered_from_ijk_mm"]))
    print(f"[build_frame] wrote {path}")


if __name__ == "__main__":
    main()
