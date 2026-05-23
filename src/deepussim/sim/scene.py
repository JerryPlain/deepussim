"""Genesis scene: Franka arm with a US probe pressing on a phantom.

WHY this exists: scale-up (Step 5) needs realistic *(pose, force)* pairs that are
actually reachable by the arm and maintain plausible contact — not arbitrary planes.
The Franka + contact physics produce those; the CBCT only supplies image + mask.

UNITS / FRAMES (read before wiring sim -> reslice): Genesis works in **metres** in the
*sim world* frame. The CBCT volume is in **millimetres** in the *CBCT* frame. A pose
read from this scene is therefore NOT directly a reslice pose — it must be mapped by
the phantom-placement transform ``T_cbct_from_simworld`` (the sim analogue of the
Step-3 calibration) and converted mm<->m. This class deliberately stays in sim units;
the mapping belongs in the pipeline once you fix the phantom placement.

Implemented against the Genesis API (gs.init / gs.Scene / morphs / inverse_kinematics /
control_dofs_position / get_links_net_contact_force / link.get_pos|get_quat). Verified
on genesis-world; if you upgrade and a name moves, the call sites are localised here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import pose_from_pos_quat, mat_to_quat, invert, compose

_GS_INITIALIZED = False


def require_genesis():
    """Import and return the ``genesis`` module, with an actionable error if absent."""
    try:
        import genesis as gs  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "Genesis is not installed in this environment. Genesis (genesis-world) "
            "supports Python 3.10-3.12 only.\n"
            "  conda activate deepussim && pip install genesis-world\n"
            "Then pin the version: pip freeze | grep genesis-world"
        ) from e
    return gs


def _np(x) -> np.ndarray:
    """Convert a Genesis return (often a torch tensor) to a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


# Franka Panda home configuration (7 arm joints + 2 fingers).
_FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])


@dataclass
class SceneConfig:
    backend: str = "gpu"                  # "gpu" | "cpu"
    franka_mjcf: str = "xml/franka_emika_panda/panda.xml"  # Genesis bundled asset
    phantom_mesh: str | None = None       # surface mesh (m); None -> a Box phantom
    phantom_size: tuple[float, float, float] = (0.2, 0.2, 0.08)  # box phantom (m)
    phantom_pos: tuple[float, float, float] = (0.55, 0.0, 0.04)  # within reach (m)
    dt: float = 1e-2
    show_viewer: bool = False
    camera_pos: tuple[float, float, float] = (1.4, -1.0, 0.9)   # viewer camera (m)
    camera_lookat: tuple[float, float, float] = (0.45, 0.0, 0.1)
    camera_fov: float = 40.0
    ee_link_name: str = "hand"
    n_arm_dofs: int = 7
    # T_ee_from_probe: hand-eye offset placing the US image plane relative to the EE.
    probe_offset: np.ndarray = field(default_factory=lambda: np.eye(4))
    home_qpos: np.ndarray = field(default_factory=lambda: _FRANKA_HOME.copy())


class UltrasoundScene:
    """Genesis scene exposing the pose/contact API that scale-up needs.

    Lifecycle:
        scene = UltrasoundScene(cfg).build()
        scene.reset()
        scene.set_probe_pose(T); scene.step(50)
        f = scene.contact_force(); pose = scene.probe_pose()
    """

    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self._gs = None
        self._scene = None
        self._franka = None
        self._phantom = None
        self._ee_link = None
        self._ee_local_idx: int | None = None
        self._arm_dofs = np.arange(self.cfg.n_arm_dofs)

    # --- construction -----------------------------------------------------
    def build(self) -> "UltrasoundScene":
        global _GS_INITIALIZED
        gs = require_genesis()
        self._gs = gs

        if not _GS_INITIALIZED:
            backend = gs.gpu if self.cfg.backend == "gpu" else gs.cpu
            try:
                gs.init(backend=backend)
            except Exception:  # pragma: no cover - fall back to CPU if GPU init fails
                gs.init(backend=gs.cpu)
            _GS_INITIALIZED = True

        viewer_options = None
        if self.cfg.show_viewer:
            viewer_options = gs.options.ViewerOptions(
                camera_pos=self.cfg.camera_pos,
                camera_lookat=self.cfg.camera_lookat,
                camera_fov=self.cfg.camera_fov,
            )
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt),
            viewer_options=viewer_options,
            show_viewer=self.cfg.show_viewer,
        )
        self._scene.add_entity(gs.morphs.Plane())

        if self.cfg.phantom_mesh:
            self._phantom = self._scene.add_entity(
                gs.morphs.Mesh(file=self.cfg.phantom_mesh, fixed=True,
                               pos=self.cfg.phantom_pos)
            )
        else:
            self._phantom = self._scene.add_entity(
                gs.morphs.Box(size=self.cfg.phantom_size, pos=self.cfg.phantom_pos,
                              fixed=True)
            )

        self._franka = self._scene.add_entity(gs.morphs.MJCF(file=self.cfg.franka_mjcf))
        self._scene.build()

        self._ee_link = self._franka.get_link(self.cfg.ee_link_name)
        self._ee_local_idx = self._find_link_index(self.cfg.ee_link_name)
        return self

    def _find_link_index(self, name: str) -> int:
        for i, link in enumerate(self._franka.links):
            if getattr(link, "name", None) == name:
                return i
        raise KeyError(f"link {name!r} not found on the Franka entity")

    # --- control & stepping ----------------------------------------------
    def reset(self) -> None:
        """Set the arm to its home configuration and settle."""
        self._franka.set_dofs_position(np.asarray(self.cfg.home_qpos, dtype=float))
        self.step(20)

    def set_probe_pose(self, T_world_from_probe: np.ndarray) -> None:
        """Drive the probe to a sim-world pose via IK on the EE link.

        The IK target is the EE pose implied by the probe pose and the hand-eye
        offset: ``T_world_from_ee = T_world_from_probe @ inv(T_ee_from_probe)``.
        """
        T_world_from_ee = compose(np.asarray(T_world_from_probe, dtype=float),
                                  invert(self.cfg.probe_offset))
        pos = T_world_from_ee[:3, 3]
        quat = mat_to_quat(T_world_from_ee[:3, :3])  # scalar-first (w, x, y, z)
        qpos = self._franka.inverse_kinematics(link=self._ee_link, pos=pos, quat=quat)
        qpos = _np(qpos)
        self._franka.control_dofs_position(qpos[self._arm_dofs], self._arm_dofs)

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._scene.step()

    # --- readouts ---------------------------------------------------------
    def probe_pose(self) -> np.ndarray:
        """Achieved sim-world probe pose as 4x4 = T_world_from_ee @ T_ee_from_probe."""
        T_world_from_ee = pose_from_pos_quat(_np(self._ee_link.get_pos()),
                                             _np(self._ee_link.get_quat()))
        return compose(T_world_from_ee, self.cfg.probe_offset)

    def contact_force(self) -> np.ndarray:
        """Net contact force (N) on the probe as a sim-world 3-vector.

        Summed over all of the robot's links: with the bare Panda gripper the contact
        lands on the *finger* links, not the ``hand`` link, so summing is the robust
        signal (the only thing the arm can touch is the fixed phantom). Once you mount
        a dedicated rigid probe geometry, switch to that link's row (``_ee_local_idx``).
        """
        forces = _np(self._franka.get_links_net_contact_force()).reshape(-1, 3)
        return forces.sum(axis=0).astype(float)

    def in_contact(self, threshold_n: float = 0.1) -> bool:
        """Whether the probe currently touches the phantom above ``threshold_n``."""
        return bool(np.linalg.norm(self.contact_force()) >= threshold_n)
