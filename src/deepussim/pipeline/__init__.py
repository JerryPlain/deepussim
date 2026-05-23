"""Step 5 — scale-up: turn many probe poses into aligned multi-modal samples."""

from .sampling import linear_sweep, tilt_fan
from .scaleup import generate_dataset

__all__ = ["linear_sweep", "tilt_fan", "generate_dataset"]
