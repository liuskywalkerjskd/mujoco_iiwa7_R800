#!/usr/bin/env python3
"""Drive the iiwa7 with the canonical robotics 7-tuple interface.

Input contract:
    pose7 = [x, y, z, qx, qy, qz, qw]

    (x, y, z): end-effector origin expressed in the BASE frame (meters)
    (qx, qy, qz, qw): end-effector orientation, quaternion in xyzw order
                      (ROS / Eigen / scipy convention)

The user's side boils down to two calls per sim step:

    controller.set_ee_pose(pose7)        # runs IK (warm-started)
    controller.update(model, data)       # writes ctrl + qfrc_applied

This demo traces a 3D Lissajous figure with the end-effector locked to
"tool pointing down" (tool axis = world -Z). 18 seconds of video render.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iiwa7_controller import IiwaEEController  # noqa: E402

SCENE = REPO / "examples" / "scenes" / "iiwa7_tuned_square_scene.xml"
OUT = REPO / "media" / "videos" / "demo_ee_control.mp4"

FPS = 30
DURATION_S = 18.0
WIDTH, HEIGHT = 720, 540


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# "Tool pointing down" orientation in base frame, xyzw convention.
# quaternion for rotation 180 deg about X: (qx=1, qy=0, qz=0, qw=0)
#   -> maps local +Z to world -Z  (tool axis points down)
POSE_DOWN_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])


def desired_ee_pose7(t: float) -> np.ndarray:
    """Return a 7-vector [x, y, z, qx, qy, qz, qw] in BASE frame.

    Users replace this with their own planner / teleop stream / RL
    policy output — the controller does not care where the pose comes
    from as long as the layout is [pos, quat_xyzw].
    """
    ramp_end = 3.0
    if t < ramp_end:
        a = smoothstep(t / ramp_end)
        neutral = np.array([0.5, 0.0, 0.75])
        start = np.array([0.5, 0.15, 0.55])
        pos = (1 - a) * neutral + a * start
    else:
        tt = t - ramp_end
        cx, cy, cz = 0.5, 0.0, 0.55
        ax, ay, az = 0.18, 0.15, 0.08
        w = 2 * np.pi * 0.1
        pos = np.array([
            cx + ax * np.cos(w * tt),
            cy + ay * np.sin(2 * w * tt),
            cz + az * np.sin(w * tt),
        ])
    return np.concatenate([pos, POSE_DOWN_QUAT_XYZW])


def main() -> int:
    print(f"loading {SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # === one-line controller init, seven-vector pose interface ===
    ctrl = IiwaEEController(model, data)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 145.0
    cam.elevation = -28.0
    cam.distance = 2.0
    cam.lookat[:] = [0.4, 0.0, 0.55]

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / dt)))
    n_frames = int(DURATION_S * FPS)
    print(f"running {n_frames} frames ({sim_steps_per_frame} sim steps/frame)")

    frames = []
    pos_errs = []
    ori_errs_deg = []
    ee_body = model.body("iiwa_link_7").id
    tool_offset = np.array([0.0, 0.0, 0.05])

    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            pose7 = desired_ee_pose7(t_sim)

            # === the only two calls the user needs to drive the arm ===
            ctrl.set_ee_pose(pose7)
            ctrl.update(model, data)

            mujoco.mj_step(model, data)

        # Log per-frame position + orientation error
        pose7_f = desired_ee_pose7(f / FPS)
        target_pos = pose7_f[:3]
        target_quat_xyzw = pose7_f[3:]
        ee_pos = data.xpos[ee_body] + data.xmat[ee_body].reshape(3, 3) @ tool_offset
        pos_errs.append(np.linalg.norm(target_pos - ee_pos))

        ee_quat_wxyz = data.xquat[ee_body].copy()
        target_quat_wxyz = np.array([target_quat_xyzw[3], *target_quat_xyzw[:3]])
        q_diff = np.empty(4)
        q_inv = np.empty(4)
        mujoco.mju_negQuat(q_inv, ee_quat_wxyz)
        mujoco.mju_mulQuat(q_diff, target_quat_wxyz, q_inv)
        angle_rad = 2.0 * np.arccos(np.clip(abs(q_diff[0]), 0.0, 1.0))
        ori_errs_deg.append(np.degrees(angle_rad))

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

        if (f + 1) % 90 == 0:
            print(
                f"  frame {f+1}/{n_frames}  pos_err={pos_errs[-1]*1000:.2f} mm  "
                f"ori_err={ori_errs_deg[-1]:.2f} deg"
            )

    pos_errs = np.array(pos_errs) * 1000
    ori_errs_deg = np.array(ori_errs_deg)
    print(
        f"\nEE position tracking: mean={pos_errs.mean():.2f} mm  "
        f"p95={np.percentile(pos_errs, 95):.2f} mm  max={pos_errs.max():.2f} mm"
    )
    print(
        f"EE orientation tracking: mean={ori_errs_deg.mean():.2f} deg  "
        f"p95={np.percentile(ori_errs_deg, 95):.2f} deg  max={ori_errs_deg.max():.2f} deg"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
