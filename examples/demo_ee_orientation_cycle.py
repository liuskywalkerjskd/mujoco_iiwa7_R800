#!/usr/bin/env python3
"""Keep the end-effector at a fixed Cartesian point while continuously
cycling its orientation through a tour of attitudes.

Shows off 6-DOF IK + controller precision: the arm has to rearrange all
7 joints to hold the wrist exactly still while rotating the tool. Uses
the canonical 7-tuple pose API:

    pose7 = [x, y, z, qx, qy, qz, qw]   (base frame, quat xyzw)

The sequence hits these anchor orientations (roll-pitch-yaw about the
base frame, applied on top of "tool pointing down"):

    down (neutral) -> forward tilt -> down -> backward tilt
    -> down -> left tilt -> down -> right tilt
    -> down -> roll +60 deg -> down -> roll -60 deg
    -> down (loop)

Orientations are interpolated with a great-circle SLERP so the path is
smooth and the controller sees C^1-continuous angular velocity.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R, Slerp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iiwa7_controller import IiwaEEController  # noqa: E402

SCENE = REPO / "examples" / "scenes" / "iiwa7_tuned_square_scene.xml"
OUT = REPO / "media" / "videos" / "demo_ee_orientation_cycle.mp4"

FPS = 30
WIDTH, HEIGHT = 720, 540

# Fixed EE position (base frame, meters)
FIXED_POS = np.array([0.50, 0.00, 0.55])

# Anchor orientations expressed as (roll_x, pitch_y, yaw_z) Euler degrees.
# All anchors start from "tool pointing down" (180 deg rotation about X).
# Extra roll/pitch/yaw is applied ON TOP to tilt or rotate the tool.
TILT = 30.0   # degrees
ROLL = 60.0   # degrees around tool axis
ANCHORS_RPY_DEG = [
    (180.0,  0.0, 0.0),              # neutral: tool straight down
    (180.0,  TILT, 0.0),             # tilt forward (+pitch)
    (180.0,  0.0, 0.0),              # back to neutral
    (180.0, -TILT, 0.0),             # tilt backward
    (180.0,  0.0, 0.0),
    (180.0 - TILT, 0.0, 0.0),        # tilt left (less roll -> side)
    (180.0,  0.0, 0.0),
    (180.0 + TILT, 0.0, 0.0),        # tilt right
    (180.0,  0.0, 0.0),
    (180.0,  0.0,  ROLL),            # roll CCW about tool axis
    (180.0,  0.0, 0.0),
    (180.0,  0.0, -ROLL),            # roll CW
    (180.0,  0.0, 0.0),              # back to neutral (close the loop)
]
SEG_TIME = 1.5                        # seconds between adjacent anchors
RAMP_IN = 3.0                          # ramp from ready pose to fixed position + first anchor
DURATION_S = RAMP_IN + SEG_TIME * (len(ANCHORS_RPY_DEG) - 1)


def build_slerp() -> Slerp:
    rots = R.from_euler("xyz", ANCHORS_RPY_DEG, degrees=True)
    times = RAMP_IN + np.arange(len(ANCHORS_RPY_DEG)) * SEG_TIME
    return Slerp(times, rots)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def desired_pose7(t: float, slerp: Slerp, start_pos: np.ndarray) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw] in base frame for time `t`."""
    # Phase 1: ramp from `start_pos` to FIXED_POS while locking to first anchor
    if t < RAMP_IN:
        a = smoothstep(t / RAMP_IN)
        pos = (1 - a) * start_pos + a * FIXED_POS
        quat_xyzw = R.from_euler("xyz", ANCHORS_RPY_DEG[0], degrees=True).as_quat()
    else:
        pos = FIXED_POS
        # SLERP between anchors; clamp to last anchor once past it
        t_eff = min(t, RAMP_IN + SEG_TIME * (len(ANCHORS_RPY_DEG) - 1) - 1e-6)
        quat_xyzw = slerp(t_eff).as_quat()
    return np.concatenate([pos, quat_xyzw])


def main() -> int:
    print(f"loading {SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)

    # Figure out the current EE position to ramp from
    ee_body = model.body("iiwa_link_7").id
    tool_offset = np.array([0.0, 0.0, 0.05])
    start_pos = (
        data.xpos[ee_body]
        + data.xmat[ee_body].reshape(3, 3) @ tool_offset
    ).copy()

    slerp = build_slerp()
    ctrl = IiwaEEController(model, data)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 120.0
    cam.elevation = -22.0
    cam.distance = 1.6
    cam.lookat[:] = FIXED_POS + np.array([0.0, 0.0, -0.05])

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / dt)))
    n_frames = int(DURATION_S * FPS)
    print(
        f"running {n_frames} frames  ({sim_steps_per_frame} sim steps/frame, "
        f"duration {DURATION_S:.1f} s)"
    )

    frames = []
    pos_errs = []
    ori_errs_deg = []
    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            ctrl.set_ee_pose(desired_pose7(t_sim, slerp, start_pos))
            ctrl.update(model, data)
            mujoco.mj_step(model, data)

        pose7 = desired_pose7(f / FPS, slerp, start_pos)
        ee_pos = data.xpos[ee_body] + data.xmat[ee_body].reshape(3, 3) @ tool_offset
        pos_errs.append(np.linalg.norm(pose7[:3] - ee_pos))

        ee_quat_wxyz = data.xquat[ee_body].copy()
        target_quat_wxyz = np.array([pose7[6], pose7[3], pose7[4], pose7[5]])
        q_inv = np.empty(4); q_diff = np.empty(4)
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
        f"\nposition (should be ~0): "
        f"mean={pos_errs.mean():.3f} mm  p95={np.percentile(pos_errs, 95):.3f} mm  "
        f"max={pos_errs.max():.3f} mm"
    )
    print(
        f"orientation tracking:    "
        f"mean={ori_errs_deg.mean():.3f} deg  p95={np.percentile(ori_errs_deg, 95):.3f} deg  "
        f"max={ori_errs_deg.max():.3f} deg"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
