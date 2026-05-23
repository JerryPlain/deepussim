"""Genesis scene: Franka arm with a US probe pressing on a phantom.

WHY this exists: scale-up (Step 5) needs realistic *(pose, force)* pairs that are
actually reachable by the arm and maintain plausible contact — not arbitrary planes.
The Franka + contact physics produce those; the CBCT only supplies image + mask.

STATUS: skeleton. The method *contracts* (signatures + docstrings) are stable and are
what ``pipeline.scaleup`` depends on. The bodies call Genesis through a lazy import and
are marked TODO where the exact API must be verified against your installed
``genesis-world`` version (the package targets Python 3.10–3.12; the API has moved
between releases, so confirm names like ``gs.init``, ``gs.Scene``, entity loaders, the
IK helper, and contact-force accessors before relying on them).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def require_genesis():
    """Import and return the ``genesis`` module, with an actionable error if absent."""
    try:
        import genesis as gs  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "Genesis is not installed in this environment. Genesis (genesis-world) "
            "supports Python 3.10-3.12 only.\n"
            "  conda create -n deepussim python=3.11 -y && conda activate deepussim\n"
            "  pip install -e \".[dev]\"\n"
            "Then pin the version: pip freeze | grep genesis-world"
        ) from e
    return gs


@dataclass
class SceneConfig:
    franka_urdf: str | None = None        # None -> use Genesis' bundled Franka asset
    phantom_mesh: str | None = None       # surface mesh of the phantom (mm)
    phantom_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    dt: float = 1e-2
    substeps: int = 10
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    show_viewer: bool = False
    target_force_n: float = 5.0           # contact force to servo the probe toward


class UltrasoundScene:
    """Thin wrapper over a Genesis scene exposing the pose/contact API scale-up needs.

    Expected lifecycle:
        scene = UltrasoundScene(cfg); scene.build(); scene.reset()
        scene.set_probe_pose(T); scene.step(); f = scene.contact_force()
    """

    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self._gs = None
        self._scene = None
        self._franka = None
        self._phantom = None
        self._ee_link = None  # end-effector / probe link handle

    # --- construction -----------------------------------------------------
    def build(self) -> "UltrasoundScene":
        """Create the Genesis scene and add the Franka + phantom.

        TODO(api): verify against installed Genesis. Sketch:
            gs = require_genesis(); gs.init(backend=gs.gpu)
            self._scene = gs.Scene(sim_options=gs.options.SimOptions(dt=cfg.dt, ...),
                                   show_viewer=cfg.show_viewer)
            self._franka = self._scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
            self._phantom = self._scene.add_entity(gs.morphs.Mesh(file=cfg.phantom_mesh, fixed=True, pos=...))
            self._scene.build()
            self._ee_link = self._franka.get_link("hand")  # probe mounts here
        """
        self._gs = require_genesis()
        raise NotImplementedError(
            "Implement build() against your installed Genesis API (see docstring sketch)."
        )

    # --- control & stepping ----------------------------------------------
    def reset(self) -> None:
        """Reset arm to a home configuration and settle the phantom contact."""
        raise NotImplementedError

    def set_probe_pose(self, T_world_from_probe: np.ndarray) -> None:
        """Drive the probe to a world-frame pose via IK on the EE link.

        TODO(api): q = self._franka.inverse_kinematics(link=self._ee_link,
                       pos=T[:3,3], quat=mat_to_quat(T[:3,:3]))
                   self._franka.control_dofs_position(q)
        """
        raise NotImplementedError

    def step(self, n: int = 1) -> None:
        """Advance the simulation ``n`` steps (``self._scene.step()``)."""
        raise NotImplementedError

    # --- readouts ---------------------------------------------------------
    def probe_pose(self) -> np.ndarray:
        """Achieved world-frame probe pose as 4x4 (FK of the EE link)."""
        raise NotImplementedError

    def contact_force(self) -> np.ndarray:
        """Net contact force (N) on the probe from the phantom, as a 3-vector.

        TODO(api): read the probe<->phantom contact (e.g. self._scene.get_contacts()
        or the link's net external force) and sum into the world frame.
        """
        raise NotImplementedError

    def in_contact(self, threshold_n: float = 0.1) -> bool:
        """Whether the probe currently touches the phantom above ``threshold_n``."""
        return bool(np.linalg.norm(self.contact_force()) >= threshold_n)
