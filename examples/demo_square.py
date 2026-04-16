#!/usr/bin/env python3
"""End-effector draws a horizontal square — MuJoCo headless demo.

Pipeline:
  1. Define a 30 cm x 30 cm square in world frame at Z=0.55 m.
  2. Pre-solve damped-least-squares (DLS) IK offline along the path to get a
     list of joint-space waypoints (warm-start from previous solution).
  3. Play those waypoints back through position actuators in the sim, render
     at 30 fps, save MP4.

Trajectory timing (total 18 s):
  0.0 - 2.0 s   home -> ready keyframe (smoothstep)
  2.0 - 3.5 s   ready -> square corner A (IK waypoint)
  3.5 - 15.5 s  trace A->B->C->D->A twice (6 s per loop)
  15.5 - 18.0 s hold + return to ready

The reference square (cyan spheres + capsules) is rendered by
iiwa7_square_scene.xml, so the video shows the target and the robot both.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path


# Unified controller — one class drives all demos.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iiwa7_controller import IiwaEEController
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
MJCF = HERE / "scenes" / "iiwa7_square_scene.xml"
OUT = HERE.parent / "media" / "videos" / "demo_square.mp4"

WIDTH, HEIGHT = 720, 540
FPS = 30

# Square corners (world frame, meters)
CORNERS = np.array([
    [0.35, -0.15, 0.55],  # A
    [0.65, -0.15, 0.55],  # B
    [0.65,  0.15, 0.55],  # C
    [0.35,  0.15, 0.55],  # D
])

# Timing (seconds)
T_HOME_READY     = 2.0
T_READY_APPROACH = 1.5
T_EDGE           = 1.5   # per edge of the square
N_LOOPS          = 2
T_RETURN         = 2.5
TOTAL_S = (
    T_HOME_READY + T_READY_APPROACH
    + T_EDGE * 4 * N_LOOPS
    + T_RETURN
)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ik_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target: np.ndarray,
    q0: np.ndarray,
    tool_offset: np.ndarray = np.zeros(3),
    max_iter: int = 200,
    tol: float = 5e-4,
    damping: float = 0.05,
    step: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Damped least squares IK, position only.

    Returns (q, residual_norm). Modifies data.qpos during iteration but
    restores-nothing at the end; caller should reset if needed.
    """
    data.qpos[:] = q0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        xpos = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ tool_offset
        err = target - xpos
        norm = np.linalg.norm(err)
        if norm < tol:
            return data.qpos.copy(), norm
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body_id)
        # damped least squares
        JJT = jac_pos @ jac_pos.T + damping**2 * np.eye(3)
        dq = jac_pos.T @ np.linalg.solve(JJT, err)
        data.qpos[:] = data.qpos + step * dq
        # respect joint limits
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)
    return data.qpos.copy(), norm


def main() -> int:
    if not MJCF.exists():
        print(f"[FAIL] not found: {MJCF}", file=sys.stderr)
        return 1

    print(f"loading {MJCF.name}")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)

    home_id = model.key("home").id
    ready_id = model.key("ready").id
    home_q = model.key_qpos[ready_id].copy()  # start at ready (no home ramp)
    ready_q = model.key_qpos[ready_id].copy()

    ee_body = model.body("iiwa_link_7").id
    tool_offset = np.array([0.0, 0.0, 0.05])  # 5 cm along link_7 local Z

    # --- precompute reference square path (waypoints in task space) ---
    n_frames = int(TOTAL_S * FPS)
    ref_path: list[np.ndarray | None] = [None] * n_frames  # EE target per frame (for logging)
    joint_target: list[np.ndarray] = [None] * n_frames  # actuator ctrl per frame

    # Cumulative time boundaries (sec)
    t_ready_end   = T_HOME_READY
    t_approach_end = t_ready_end + T_READY_APPROACH
    t_loop_start  = t_approach_end

    print(
        f"precomputing {n_frames} waypoints over {TOTAL_S:.1f} s "
        f"(edge time {T_EDGE}s, loops {N_LOOPS})"
    )

    # IK warm-start state lives in its own data object so we don't disturb sim
    ik_data = mujoco.MjData(model)
    q_prev = ready_q.copy()

    for f in range(n_frames):
        t = f / FPS

        if t < t_ready_end:
            # Phase 1: home -> ready (joint space)
            alpha = smoothstep(t / T_HOME_READY)
            joint_target[f] = home_q + alpha * (ready_q - home_q)
            ref_path[f] = None  # no EE target yet

        elif t < t_approach_end:
            # Phase 2: ready -> corner A (IK in task space)
            alpha = smoothstep((t - t_ready_end) / T_READY_APPROACH)
            # starting EE: re-evaluate at ready pose
            mujoco.mj_resetData(model, ik_data)
            ik_data.qpos[:] = ready_q
            mujoco.mj_forward(model, ik_data)
            ready_ee = ik_data.xpos[ee_body] + ik_data.xmat[ee_body].reshape(3,3) @ tool_offset
            target = (1 - alpha) * ready_ee + alpha * CORNERS[0]
            q, res = ik_dls(model, ik_data, ee_body, target, q_prev, tool_offset=tool_offset)
            joint_target[f] = q
            ref_path[f] = target
            q_prev = q

        else:
            # Phase 3 or 4: tracing square, then return
            tau = t - t_loop_start
            loop_duration = T_EDGE * 4 * N_LOOPS
            if tau < loop_duration:
                # which edge within which loop
                edge_t = tau / T_EDGE
                edge_idx = int(edge_t) % 4
                edge_frac = smoothstep(edge_t - int(edge_t))
                p0 = CORNERS[edge_idx]
                p1 = CORNERS[(edge_idx + 1) % 4]
                target = (1 - edge_frac) * p0 + edge_frac * p1
                q, res = ik_dls(model, ik_data, ee_body, target, q_prev, tool_offset=tool_offset)
                joint_target[f] = q
                ref_path[f] = target
                q_prev = q
            else:
                # Phase 4: return to ready
                tau2 = tau - loop_duration
                alpha = smoothstep(tau2 / T_RETURN)
                joint_target[f] = q_prev * (1 - alpha) + ready_q * alpha
                ref_path[f] = None

        if (f + 1) % 60 == 0:
            print(f"  precomputed {f+1}/{n_frames} frames")

    # --- kinematic playback + render ---
    # This demo visualizes IK correctness (EE on the square), not actuator
    # tracking dynamics. So we set qpos directly each frame and call
    # mj_forward. For a dynamics-under-actuator demo, use iiwa7_scene.xml
    # with demo_record.py instead.
    print("running kinematic playback and rendering...")
    ctrl = IiwaEEController(model, data, mode="kinematic")

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.azimuth = 145.0
    camera.elevation = -28.0
    camera.distance = 2.0
    camera.lookat[:] = [0.4, 0.0, 0.55]

    frames = []
    ee_actual_log = []

    for f in range(n_frames):
        ctrl.set_joint_target(joint_target[f])
        ctrl.update(model, data)
        mujoco.mj_forward(model, data)
        ee_actual = data.xpos[ee_body] + data.xmat[ee_body].reshape(3, 3) @ tool_offset
        ee_actual_log.append(ee_actual.copy())

        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if (f + 1) % 60 == 0:
            ref = ref_path[f]
            err_str = ""
            if ref is not None:
                err_str = f"  |EE-target|={np.linalg.norm(ref - ee_actual)*1000:.2f}mm"
            print(f"  frame {f+1}/{n_frames}  EE={ee_actual}{err_str}")

    if np.any(np.isnan(data.qpos)):
        print("[FAIL] NaN in qpos", file=sys.stderr)
        return 1

    # Summary: tracking error during square-tracing phase
    phase3_frames = [
        i for i, r in enumerate(ref_path)
        if r is not None and i / FPS >= t_approach_end and i / FPS < t_approach_end + T_EDGE * 4 * N_LOOPS
    ]
    errs = [np.linalg.norm(ref_path[i] - ee_actual_log[i]) for i in phase3_frames]
    print(
        f"\ntracking error along square path: "
        f"mean={np.mean(errs)*1000:.2f} mm  max={np.max(errs)*1000:.2f} mm "
        f"over {len(phase3_frames)} frames"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"[OK] size={OUT.stat().st_size/1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
