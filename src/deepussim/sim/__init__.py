"""Genesis simulation: Franka + phantom, contact force, trajectory control.

Genesis is imported lazily (see :func:`require_genesis`) so that importing
``deepussim`` and using the geometric/rendering core never requires Genesis. This is
the source of the *force* and *reachable pose* signals in Step 5 — CBCT supplies only
image + mask.
"""

from .scene import SceneConfig, UltrasoundScene, require_genesis

__all__ = ["SceneConfig", "UltrasoundScene", "require_genesis"]
