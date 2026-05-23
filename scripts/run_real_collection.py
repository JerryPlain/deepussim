#!/usr/bin/env python
"""Step 2 entrypoint (STUB): collect real (US image, probe pose, force) on hardware.

This is hardware-specific and intentionally unimplemented. A real implementation wires
together, in a synchronised loop:
  - the Franka FCI (e.g. via franky / libfranka / polymetis) for FK pose + force/torque,
  - the ultrasound machine's frame grabber for B-mode images,
  - timestamp alignment so each US frame carries the pose+force from the same instant,
and writes records compatible with ``deepussim.data.record.Sample`` (pose in the world
frame; convert to the CBCT frame later with calib.registration).

Run order downstream: collect here -> calib.registration (Step 3) to put poses in the
CBCT frame -> calib.renderer_fit (Step 4) -> scripts/run_scaleup.py (Step 5).
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Real data collection is hardware-specific; implement against your Franka + "
        "US-machine stack. See the module docstring for the required loop."
    )


if __name__ == "__main__":
    main()
