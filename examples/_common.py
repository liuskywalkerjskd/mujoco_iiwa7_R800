"""Shared helpers for demo scripts. Keep demos focused on trajectory
design — precompute + spline + render boilerplate lives here."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.interpolate import CubicSpline


REPO_ROOT = Path(__file__).resolve().parent.parent


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def precompute_joint_trajectory(
    model: mujoco.MjModel,
    ctrl,  # iiwa7_controller.IiwaEEController (avoids circular import)
    times: np.ndarray,
    ee_pos_fn: Callable[[float], Optional[np.ndarray]],
    ee_quat_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
    warm_start_q: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list]:
    """For every time sample, resolve the desired joint configuration by
    running IK (through the controller). Returns (q_frames, ref_ee_list).

    If `ee_pos_fn(t)` returns None for a given t, the controller's previous
    q_target is carried over (useful for joint-space ramps / hold segments
    handled elsewhere).
    """
    nq = model.nq
    if warm_start_q is not None:
        ctrl.set_joint_target(warm_start_q)

    q_frames = np.zeros((len(times), nq))
    ref_ee = [None] * len(times)
    for i, t in enumerate(times):
        pos = ee_pos_fn(t)
        if pos is None:
            q_frames[i] = ctrl.q_target
            continue
        quat = ee_quat_fn(t) if ee_quat_fn is not None else None
        ctrl.set_ee_target(pos=pos, quat=quat)
        q_frames[i] = ctrl.q_target
        ref_ee[i] = pos
    return q_frames, ref_ee


def fit_splines(times: np.ndarray, q_frames: np.ndarray) -> tuple[list, list, list]:
    """Fit one cubic spline per joint over the full trajectory.
    Returns (q_splines, qd_splines, qdd_splines)."""
    sq = [CubicSpline(times, q_frames[:, j], bc_type="clamped") for j in range(q_frames.shape[1])]
    sqd = [sp.derivative(1) for sp in sq]
    sqdd = [sp.derivative(2) for sp in sq]
    return sq, sqd, sqdd


def eval_spline(splines: Sequence, t: float) -> np.ndarray:
    return np.array([sp(t) for sp in splines])


def make_default_camera(
    azimuth: float = 145.0,
    elevation: float = -28.0,
    distance: float = 2.0,
    lookat: Sequence[float] = (0.4, 0.0, 0.55),
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = list(lookat)
    return cam


def write_video(out_path: Path, frames: list, fps: int = 30, quality: int = 7) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps, quality=quality, macro_block_size=1)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.1f} KiB)")


def draw_polyline(scn: mujoco.MjvScene, points, rgba, radius: float = 0.003) -> None:
    """Append a polyline overlay to an mjvScene as connected capsules."""
    for i in range(len(points) - 1):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.array(points[i]),
            np.array(points[i + 1]),
        )
        scn.ngeom += 1
