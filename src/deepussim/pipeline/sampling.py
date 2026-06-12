"""Probe-pose samplers (in the volume/CBCT frame) for scale-up.

These produce *geometric* poses for the no-sim path. When the Genesis scene is in the
loop, treat these as nominal targets — the achieved (reachable, in-contact) pose comes
back from ``UltrasoundScene.probe_pose()``.

Pose convention matches us.reslice: probe +z is axial (into tissue), +x lateral.
"""
from __future__ import annotations

import numpy as np

from ..geometry import make_transform, rot_x


def _aim_into_tissue(position, axial_dir) -> np.ndarray:
    """Build T_vol_from_probe with probe +z along ``axial_dir`` (into the volume)."""
    z = np.asarray(axial_dir, dtype=float)
    z = z / (np.linalg.norm(z) + 1e-12)
    # Pick a lateral axis not parallel to z.
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return make_transform(R, position)


def linear_sweep(start, end, n: int, axial_dir=(0.0, 0.0, -1.0)) -> list[np.ndarray]:
    """``n`` probe poses translating from ``start`` to ``end`` (mm), fixed orientation."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    return [_aim_into_tissue(start + t * (end - start), axial_dir) for t in ts]


def tilt_fan(position, n: int, max_tilt_deg: float = 20.0,
             axial_dir=(0.0, 0.0, -1.0)) -> list[np.ndarray]:
    """``n`` poses at a fixed point, fanning the probe through +/- ``max_tilt_deg``."""
    base = _aim_into_tissue(position, axial_dir)
    angles = np.deg2rad(np.linspace(-max_tilt_deg, max_tilt_deg, n))
    out = []
    for a in angles:
        R = base.copy()
        R[:3, :3] = base[:3, :3] @ rot_x(a)
        out.append(R)
    return out


# --- surface-constrained trajectory (the real generator) ------------------
# The probe can only glide *along* the phantom surface, so poses are placed on the
# CBCT surface mesh: sample a point, estimate a smoothed normal, rest the probe on the
# surface with its axial (+z) axis pointing inward (perpendicular contact). The same
# pose stream then drives both the arm and the volume reslice (aligned by construction).


def _pose_on_surface(surface_pt, normal_out, sweep_dir, standoff_mm) -> np.ndarray:
    """Probe pose resting on the surface: +z axial inward, +x along the sweep direction.

    ``normal_out`` is the unit outward surface normal. The probe sits ``standoff_mm`` out
    along it; its axial axis points inward (``-normal_out``); the lateral (+x) axis follows
    ``sweep_dir`` projected into the imaging plane so the image width tracks the scan.
    """
    z = -np.asarray(normal_out, dtype=float)
    z = z / (np.linalg.norm(z) + 1e-12)
    d = np.asarray(sweep_dir, dtype=float)
    x = d - (d @ z) * z
    if np.linalg.norm(x) < 1e-6:  # sweep parallel to the normal: pick any in-plane axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - (ref @ z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    pos = np.asarray(surface_pt, dtype=float) + standoff_mm * np.asarray(normal_out, dtype=float)
    return make_transform(R, pos)


def _smoothed_normal(tree, vertices, point, radius, n_fallback=30):
    """Outward-ish unit normal at ``point`` via PCA over neighbouring mesh vertices."""
    idx = tree.query_ball_point(point, radius)
    if len(idx) < 8:
        idx = tree.query(point, n_fallback)[1]
    Q = vertices[idx] - vertices[idx].mean(axis=0)
    _, vecs = np.linalg.eigh(Q.T @ Q)
    n = vecs[:, 0]  # smallest-variance direction = surface normal
    return n / (np.linalg.norm(n) + 1e-12)


def _pose_from_guide(tree, V, guide, sweep_dir, standoff_mm, smooth_radius_mm) -> np.ndarray:
    """One surface pose for a guide point: project to the surface, smooth normal, rest probe."""
    surface_pt = V[tree.query(guide)[1]]
    normal = _smoothed_normal(tree, V, surface_pt, smooth_radius_mm)
    if normal @ (guide - surface_pt) < 0:  # orient outward (toward the guide point)
        normal = -normal
    return _pose_on_surface(surface_pt, normal, sweep_dir, standoff_mm)


def _line_poses(tree, V, start, end, n, standoff_mm, smooth_radius_mm) -> list[np.ndarray]:
    sweep_dir = end - start
    return [_pose_from_guide(tree, V, start + t * sweep_dir, sweep_dir, standoff_mm,
                             smooth_radius_mm) for t in np.linspace(0.0, 1.0, n)]


def surface_sweep(mesh, start, end, n: int, standoff_mm: float = 2.0,
                  smooth_radius_mm: float = 8.0) -> list[np.ndarray]:
    """``n`` probe poses (``T_cbct_from_probe``, mm) gliding along the phantom surface.

    ``start``/``end`` are guide points in the CBCT mm frame placed *outside* the phantom on
    the approach side. Each is projected to the nearest surface point; a smoothed normal is
    estimated there (PCA over a ``smooth_radius_mm`` neighbourhood, oriented toward the guide
    point so it points outward) and a probe pose is built resting on the surface with the
    axial axis pointing inward. Returns the poses in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    return _line_poses(tree, V, np.asarray(start, float), np.asarray(end, float),
                       n, standoff_mm, smooth_radius_mm)


def surface_raster(mesh, axis: int = 0, span_frac: float = 0.6, cross_frac: float = 0.4,
                   n_lines: int = 5, n_per_line: int = 24, standoff_mm: float = 2.0,
                   smooth_radius_mm: float = 8.0, serpentine: bool = True,
                   clearance_mm: float = 20.0) -> list[np.ndarray]:
    """Cover a patch of the phantom's +z-top with ``n_lines`` parallel surface sweeps.

    Sweeps run along ``axis`` (0=x, 1=y) spanning ``span_frac`` of that extent, stepped across
    the other in-plane axis over ``cross_frac`` of its extent in ``n_lines`` lines of
    ``n_per_line`` poses each. With ``serpentine`` the lines alternate direction so the pose
    stream is a continuous lawnmower path. The KD-tree is built once and shared across lines.
    Returns ``n_lines * n_per_line`` poses (``T_cbct_from_probe``, mm) in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    centre = (lo + hi) / 2.0
    cross_axis = 1 - axis  # the other in-plane horizontal axis (0<->1); up is z (axis 2)
    half_along = (hi[axis] - lo[axis]) * span_frac / 2.0
    half_cross = (hi[cross_axis] - lo[cross_axis]) * cross_frac / 2.0
    z_top = hi[2] + clearance_mm
    cross_offsets = np.linspace(-half_cross, half_cross, n_lines) if n_lines > 1 else [0.0]

    poses: list[np.ndarray] = []
    for j, c in enumerate(cross_offsets):
        start = centre.copy(); end = centre.copy()
        start[2] = end[2] = z_top
        start[axis], end[axis] = centre[axis] - half_along, centre[axis] + half_along
        start[cross_axis] = end[cross_axis] = centre[cross_axis] + c
        line = _line_poses(tree, V, start, end, n_per_line, standoff_mm, smooth_radius_mm)
        if serpentine and j % 2 == 1:
            line = line[::-1]
        poses.extend(line)
    return poses


def contact_raster(mesh, contact_points, n_lines: int = 6, n_per_line: int = 16,
                   expand: float = 1.25, lift_mm: float = 25.0, standoff_mm: float = 2.0,
                   smooth_radius_mm: float = 8.0, serpentine: bool = True) -> list[np.ndarray]:
    """Raster the patch of the phantom surface that the real probe actually scanned.

    Unlike :func:`surface_raster` (which targets the mesh's geometric +z top), this is anchored
    to ``contact_points`` — the real in-contact probe positions, in the **CBCT mm** frame — so the
    generated poses land on the same face the arm reached (and are reachable). A local tangent
    frame is fit to the contact cloud by PCA (the two high-variance axes ``u, v`` span the scan
    plane, the low-variance axis ``w`` is the surface normal, oriented outward); a serpentine grid
    in that plane — sized to the contact extent times ``expand`` — is lifted ``lift_mm`` above the
    surface and each guide point projected back onto the mesh with an inward-axial pose. Returns
    ``n_lines * n_per_line`` poses (``T_cbct_from_probe``, mm) in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    pts = np.asarray(contact_points, dtype=float)
    c = pts.mean(0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u, v, w = Vt[0], Vt[1], Vt[2]
    if w @ (c - V.mean(0)) < 0:                          # orient w outward (away from the interior)
        w = -w
    su, sv = X @ u, X @ v
    half_u = (su.max() - su.min()) / 2.0 * expand
    half_v = (sv.max() - sv.min()) / 2.0 * expand
    poses: list[np.ndarray] = []
    for j, dv in enumerate(np.linspace(-half_v, half_v, n_lines)):
        row = np.linspace(-half_u, half_u, n_per_line)
        if serpentine and j % 2 == 1:
            row = row[::-1]
        for du in row:
            guide = c + du * u + dv * v + lift_mm * w
            poses.append(_pose_from_guide(tree, V, guide, u, standoff_mm, smooth_radius_mm))
    return poses


# --- EE-oriented surface raster (orientation from real poses, NOT mesh normals) ----------
# WHY: on the real rig the probe presses straight down (its axial varies <0.6deg over a whole
# sweep) — it does NOT follow the local surface normal. Measured against the real EE poses,
# both the per-point PCA normal and the mesh-native normal disagree with the real probe axial
# (|dot|~0.6, only ~30% within ~37deg), so a mesh-normal-oriented raster aims the probe wrongly.
# Here orientation is borrowed from the nearest real probe pose; the mesh is used ONLY to
# project each guide onto the surface and set the standoff. With today's near-constant data this
# is ~one fixed downward orientation; once tilt-diverse poses are collected, generated poses
# near a tilted region inherit that tilt automatically.


def _nearest_rotation(pose_tree, rotations, query):
    """Rotation (3x3) of the real probe pose nearest ``query`` (position-space k=1)."""
    return rotations[pose_tree.query(query)[1]]


def contact_raster_ee(mesh, contact_poses, n_lines: int = 6, n_per_line: int = 16,
                      expand: float = 1.0, lift_mm: float = 25.0, standoff_mm: float = 2.0,
                      serpentine: bool = True) -> list[np.ndarray]:
    """Raster the scanned patch, orienting each pose from the nearest **real EE pose**.

    ``contact_poses`` are the real in-contact probe poses (``T_cbct_from_probe``, 4x4, mm) —
    e.g. ``sim_pose_to_cbct(compose(frame.pose, T_EE_FROM_PROBE), sim_to_cbct)`` per contact
    frame. Their positions define the scan plane (PCA, as in :func:`contact_raster`); their
    rotations define the probe orientation (the mesh normal is deliberately not used — see the
    note above). A serpentine grid in the plane is lifted ``lift_mm`` above the surface, each
    guide projected onto the mesh, and a pose built there with the nearest real pose's rotation,
    sitting ``standoff_mm`` outside the surface along its (outward) axial. Returns
    ``n_lines * n_per_line`` poses (``T_cbct_from_probe``, mm) in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    surf_tree = cKDTree(V)
    poses_in = np.asarray(contact_poses, dtype=float)
    if poses_in.ndim != 3 or poses_in.shape[1:] != (4, 4):
        raise ValueError(f"contact_poses must be (N,4,4), got {poses_in.shape}")
    pts = poses_in[:, :3, 3]
    rots = poses_in[:, :3, :3]
    pose_tree = cKDTree(pts)

    c = pts.mean(0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u, v, w = Vt[0], Vt[1], Vt[2]
    if w @ (c - V.mean(0)) < 0:                          # orient w outward (away from interior)
        w = -w
    su, sv = X @ u, X @ v
    half_u = (su.max() - su.min()) / 2.0 * expand
    half_v = (sv.max() - sv.min()) / 2.0 * expand

    poses: list[np.ndarray] = []
    for j, dv in enumerate(np.linspace(-half_v, half_v, n_lines)):
        row = np.linspace(-half_u, half_u, n_per_line)
        if serpentine and j % 2 == 1:
            row = row[::-1]
        for du in row:
            guide = c + du * u + dv * v + lift_mm * w
            surface_pt = V[surf_tree.query(guide)[1]]
            R = _nearest_rotation(pose_tree, rots, surface_pt)
            axial = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)   # +z into tissue
            pos = surface_pt - standoff_mm * axial                # rest standoff outside surface
            poses.append(make_transform(R, pos))
    return poses


def top_sweep_endpoints(mesh, axis: int = 0, span_frac: float = 0.6,
                        clearance_mm: float = 20.0):
    """Convenience guide endpoints for a sweep across the mesh's +z-top along ``axis``.

    Returns ``(start, end)`` points held ``clearance_mm`` above the top of the bounding box,
    spanning ``span_frac`` of the extent along ``axis`` (0=x, 1=y), centred otherwise.
    """
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    centre = (lo + hi) / 2.0
    half = (hi[axis] - lo[axis]) * span_frac / 2.0
    z_top = hi[2] + clearance_mm
    start = centre.copy(); end = centre.copy()
    start[2] = end[2] = z_top
    start[axis] = centre[axis] - half
    end[axis] = centre[axis] + half
    return start, end


# --- surface curves through hand-picked points (open the *coverage* axis) ------------------
# WHY: contact_raster_ee densifies the real footprint only (its grid is bounded by half_u/half_v
# = the contact extent), so it scales up volume/labels but NOT *where* the probe goes. These
# samplers leave the scanned patch: drape smooth curves through arbitrary surface anchor points —
# including the lateral sides a real probe can't reach (sim's exclusive advantage). Orientation is
# still borrowed from the nearest real EE pose (the single trusted down-press), so we extrapolate
# on the cheap *position* axis while staying in-regime on the *orientation* axis. Points whose
# local surface normal is far from that down-press axial are the "Tier B" frontier
# (pose_surface_deviation): generate them, but tag/quarantine before training.


def _curve_points(anchors, n: int, smooth: bool = True) -> np.ndarray:
    """``n`` points along a smooth path through the ``(k>=2, 3)`` anchor points (CBCT mm).

    A spline is fit *through* the anchors (Catmull-Rom-like, ``k=min(3, len-1)``); with <3
    anchors (or ``smooth=False``) it falls back to an arc-length-uniform polyline.
    """
    A = np.asarray(anchors, dtype=float)
    if A.ndim != 2 or A.shape[1] != 3 or len(A) < 2:
        raise ValueError(f"each curve needs >=2 anchor points (k,3); got {A.shape}")
    if not smooth or len(A) < 3:
        seg = np.linalg.norm(np.diff(A, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        ts = np.linspace(0.0, s[-1], n)
        return np.column_stack([np.interp(ts, s, A[:, k]) for k in range(3)])
    from scipy.interpolate import splev, splprep
    tck, _ = splprep([A[:, 0], A[:, 1], A[:, 2]], s=0, k=min(3, len(A) - 1))
    x, y, z = splev(np.linspace(0.0, 1.0, n), tck)
    return np.column_stack([x, y, z])


def surface_curves_from_points(mesh, curves, contact_poses=None, points_per_curve: int = 24,
                               standoff_mm: float = 2.0, smooth_radius_mm: float = 8.0,
                               smooth: bool = True, serpentine: bool = True) -> list[np.ndarray]:
    """Drape smooth curves through hand-picked surface points and build a pose at each.

    Unlike :func:`contact_raster_ee` (a grid bounded by the *real* contact footprint), the
    ``curves`` anchors may sit anywhere on the phantom — including the lateral sides the real probe
    can't reach. Each curve is smoothed (:func:`_curve_points`), sampled to ``points_per_curve``
    points, projected onto the surface, and turned into a pose resting ``standoff_mm`` outside it.

    Orientation:
      * ``contact_poses`` given (real in-contact probe poses, ``T_cbct_from_probe`` mm, ``(N,4,4)``)
        -> each pose borrows the rotation of the *nearest real EE pose*, i.e. the single trusted
        down-press (same rule as :func:`contact_raster_ee`). Position leaves the patch; orientation
        stays in-regime. Tag the lateral frontier with :func:`pose_surface_deviation`.
      * ``contact_poses`` None -> a perpendicular press along the smoothed surface normal (same rule
        as :func:`surface_sweep`); honest geometry, but off the renderer's orientation regime.

    ``curves`` is a list of ``(k>=2, 3)`` anchor arrays (CBCT mm). Returns the poses
    (``T_cbct_from_probe``, mm) of all curves concatenated in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    surf_tree = cKDTree(V)
    pose_tree = rots = None
    if contact_poses is not None:
        cp = np.asarray(contact_poses, dtype=float)
        if cp.ndim != 3 or cp.shape[1:] != (4, 4):
            raise ValueError(f"contact_poses must be (N,4,4), got {cp.shape}")
        pose_tree = cKDTree(cp[:, :3, 3])
        rots = cp[:, :3, :3]

    poses: list[np.ndarray] = []
    for j, anchors in enumerate(curves):
        pts = _curve_points(anchors, points_per_curve, smooth=smooth)
        if serpentine and j % 2 == 1:
            pts = pts[::-1]
        for q, guide in enumerate(pts):
            if pose_tree is None:                            # perpendicular press (mesh normal)
                tangent = pts[min(q + 1, len(pts) - 1)] - pts[max(q - 1, 0)]
                poses.append(_pose_from_guide(surf_tree, V, guide, tangent, standoff_mm,
                                              smooth_radius_mm))
                continue
            surface_pt = V[surf_tree.query(guide)[1]]        # down-press from nearest real EE pose
            R = _nearest_rotation(pose_tree, rots, surface_pt)
            axial = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)
            poses.append(make_transform(R, surface_pt - standoff_mm * axial))
    return poses


def side_anchor_curves(mesh, contact_poses, n_curves: int = 5, n_anchors: int = 6,
                       along_frac: float = 1.0, reach: float = 2.0, lift_mm: float = 25.0,
                       drop_mm: float = 40.0):
    """Auto-build anchor curves that sweep from the scanned patch out onto the lateral sides.

    A convenience seed for :func:`surface_curves_from_points` so the whole thing runs end-to-end
    without hand-picking points. Uses the same PCA frame as :func:`contact_raster_ee` (``u, v``
    span the scan plane, ``w`` the outward normal). Each curve runs along ``u`` (the long scan
    axis) at a fixed cross-offset ``dv``; ``n_curves`` offsets fan out to ``reach`` x the patch
    half-width, so the outer curves leave the patch. As ``|dv|`` exceeds the patch, the guide
    descends (``+lift_mm`` -> ``-drop_mm``) so projecting onto the mesh wraps it onto the side
    faces. Inner curves stay on top (Tier A); outer curves drape the sides (Tier B). Returns a
    list of ``(n_anchors, 3)`` anchor arrays (CBCT mm).
    """
    V = np.asarray(mesh.vertices, dtype=float)
    pts = np.asarray(contact_poses, dtype=float)[:, :3, 3]
    c = pts.mean(0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u, v, w = Vt[0], Vt[1], Vt[2]
    if w @ (c - V.mean(0)) < 0:                              # orient w outward (away from interior)
        w = -w
    su, sv = X @ u, X @ v
    half_u = (su.max() - su.min()) / 2.0 * along_frac
    half_v = (sv.max() - sv.min()) / 2.0
    curves = []
    for dv in np.linspace(-reach * half_v, reach * half_v, n_curves):
        over = min(1.0, max(0.0, abs(dv) - half_v) / (half_v + 1e-9))   # 0 on patch, ->1 past it
        z = lift_mm - (lift_mm + drop_mm) * over                        # descend onto the side
        curves.append(np.array([c + du * u + dv * v + z * w
                                for du in np.linspace(-half_u, half_u, n_anchors)]))
    return curves


def pose_surface_deviation(mesh, poses, contact_poses, smooth_radius_mm: float = 8.0):
    """Per-pose ``(standoff_mm, surface_turn_deg)`` for Tier A/B tagging — winding-independent.

    ``surface_turn_deg`` is the angle between the local surface-normal *line* at the pose and at the
    nearest real *contact* point (sign-free PCA normals): ~0 deg means the surface there is oriented
    like the scanned patch, so the borrowed down-press relates to it as it did in training
    (renderer-trusted "Tier A"); large means the surface has turned away — the lateral "Tier B"
    frontier to tag or quarantine.

    Why turn-vs-patch and not axial-vs-normal: the real probe presses *straight down*, ~50 deg off
    the local surface normal (it does not press perpendicular — see :func:`contact_raster_ee`), so
    an axial-vs-normal metric would flag even the trusted real poses as Tier B. Comparing normal to
    normal cancels that constant offset, so the real contact poses score ~0 deg by construction and
    the metric isolates how far the *surface* (hence the contact geometry) has drifted. Mesh
    ``face_normals`` are avoided (unreliable sign on segmented STLs); ``standoff_mm`` is signed
    (+ outside the surface), with the normal oriented by the real approach ``-mean(contact axial)``.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    cp = np.asarray(contact_poses, dtype=float)
    cpt = cp[:, :3, 3]
    cp_tree = cKDTree(cpt)
    up = -cp[:, :3, 2].mean(0)                               # outward: the side the probe comes from
    up = up / (np.linalg.norm(up) + 1e-12)
    ref_cache: dict[int, np.ndarray] = {}                    # patch-point normal per contact index

    P = np.asarray(poses, dtype=float)
    turn = np.empty(len(P)); standoff = np.empty(len(P))
    for i in range(len(P)):
        p = P[i, :3, 3]
        sp = V[tree.query(p)[1]]
        n = _smoothed_normal(tree, V, sp, smooth_radius_mm)
        j = int(cp_tree.query(p)[1])                         # nearest real contact point
        if j not in ref_cache:
            ref_cache[j] = _smoothed_normal(tree, V, V[tree.query(cpt[j])[1]], smooth_radius_mm)
        turn[i] = np.degrees(np.arccos(np.clip(abs(n @ ref_cache[j]), 0.0, 1.0)))
        standoff[i] = (p - sp) @ (n if n @ up >= 0 else -n)  # signed, + outside the surface
    return standoff, turn
