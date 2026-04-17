#!/usr/bin/env python3
"""Full pick-and-place demo with the iiwa7 + Robotiq 2F-85 composite.

Control architecture (mirrors the real robot):
  * ARM     ctrl[0..6]  driven by `IiwaEEController` via Cartesian IK
  * GRIPPER ctrl[7]     driven independently by a scripted setpoint

Waypoint timeline (all positions in BASE = world frame for this scene):

  t_s   phase          TCP target (x, y, z)          gripper ctrl
  0.0   home           (arm keyframe home)            0.00 (open)
  1.5   pre_grasp      (0.50, -0.20, 0.18)            0.00
  3.0   grasp_descend  (0.50, -0.20, 0.02)            0.00
  3.5   close_on_cube  (hold)                         0.55 (close, ~40 mm gap)
  4.5   lift           (0.50, -0.20, 0.22)            0.55
  6.0   transit        (0.50,  0.20, 0.22)            0.55
  7.5   place_descend  (0.50,  0.20, 0.05)            0.55
  8.0   release        (hold)                         0.00 (open)
  8.8   retreat        (0.50,  0.20, 0.22)            0.00
 10.0   return_home    (arm home)                     0.00

Video output: media/videos/pickplace_gripper.mp4
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

SCENE = REPO / "examples" / "scenes" / "iiwa7_pickplace_gripper_scene.xml"
OUT = REPO / "media" / "videos" / "pickplace_gripper.mp4"

FPS = 30
WIDTH, HEIGHT = 960, 720

# Tool offset: link_7 origin -> pad midpoint (measured in the MJCF frame).
#   attachment_site is at link_7 z=0.05, so gripper base at z=0.05
#   pad midpoint is at z=0.144 inside the gripper base (measured empirically)
#   -> link_7 to pad midpoint = 0.05 + 0.144 = 0.194
TOOL_OFFSET = np.array([0.0, 0.0, 0.194])

# Tool-down orientation matched to the home keyframe of iiwa7 so there is
# no 180-deg spurious yaw flip at t=0.
#
#   At keyframe home, link_7 body axes in WORLD frame are:
#       local +X -> world -X,  local +Y -> world +Y,  local +Z -> world -Z
#   This is a 180 deg rotation about the WORLD Y axis, i.e.
#       quaternion (wxyz) = (0, 0, 1, 0)  <=>  (xyzw) = (0, 1, 0, 0).
#   Picking this as the "tool-down" target means the arm starts already on
#   the target orientation manifold and only has to translate.
#
# (Gripper jaws close along local +/-Y, which maps to world +/-Y here, so
# the cube — lined up along the world Y axis in this scene — is grasped
# perpendicular to the approach direction, as intended.)
Q_DOWN_XYZW = np.array([0.0, 1.0, 0.0, 0.0])

# Gripper setpoints (rad):  0.00 = fully open, 0.55 ~ grip 40-mm cube,
# 0.80 = fully closed.
G_OPEN = 0.0
G_GRIP = 0.55


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp_pose(t: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear interpolation between 7-vectors with smoothstep timing."""
    s = smoothstep(t)
    return (1 - s) * a + s * b


# ---------------------------------------------------------------- waypoints

HOME_POS = np.array([0.40, 0.00, 0.45])           # approximate home TCP location
PRE_GRASP = np.array([0.50, -0.20, 0.18])
GRASP = np.array([0.50, -0.20, 0.02])             # TCP = cube center
LIFT = np.array([0.50, -0.20, 0.22])
TRANSIT = np.array([0.50, 0.20, 0.22])
PLACE = np.array([0.50, 0.20, 0.05])              # release just above the pad
RETREAT = np.array([0.50, 0.20, 0.22])


def pose7(pos: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, Q_DOWN_XYZW])


# (t_start, t_end, ee_pose_fn, gripper_ctrl)
SCHEDULE = [
    (0.0, 1.5,  lambda t: lerp_pose(t, pose7(HOME_POS),  pose7(PRE_GRASP)),  G_OPEN),
    (1.5, 3.0,  lambda t: lerp_pose(t, pose7(PRE_GRASP), pose7(GRASP)),      G_OPEN),
    (3.0, 3.8,  lambda t: pose7(GRASP),                                       G_GRIP),  # close on cube
    (3.8, 5.0,  lambda t: lerp_pose(t, pose7(GRASP),     pose7(LIFT)),       G_GRIP),
    (5.0, 6.5,  lambda t: lerp_pose(t, pose7(LIFT),      pose7(TRANSIT)),    G_GRIP),
    (6.5, 8.0,  lambda t: lerp_pose(t, pose7(TRANSIT),   pose7(PLACE)),      G_GRIP),
    (8.0, 8.8,  lambda t: pose7(PLACE),                                       G_OPEN),  # release
    (8.8, 10.0, lambda t: lerp_pose(t, pose7(PLACE),     pose7(RETREAT)),    G_OPEN),
    (10.0, 12.0, lambda t: lerp_pose(t, pose7(RETREAT),  pose7(HOME_POS)),   G_OPEN),
]
DURATION_S = SCHEDULE[-1][1]


def current_waypoint(t: float):
    for t0, t1, fn, g in SCHEDULE:
        if t <= t1:
            s = (t - t0) / max(t1 - t0, 1e-6)
            return fn(s), g
    last_t0, last_t1, last_fn, last_g = SCHEDULE[-1]
    return last_fn(1.0), last_g


def main() -> int:
    print(f"loading {SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # Controller drives ARM only (ctrl[0..6]); ctrl[7] is overwritten every
    # sim step below. tool_offset points from link_7 origin to the pad midpoint.
    # mode="gravity_ff" is used (instead of the repo default full-ID mode)
    # because the scene contains a cube freejoint (nq != nv), which breaks
    # the full-ID finite-difference math. Gravity feed-forward still gives
    # ~1-cm tracking, plenty for grasping a 40-mm cube.
    ctrl = IiwaEEController(model, data, mode="gravity_ff", tool_offset=TOOL_OFFSET)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 140.0
    cam.elevation = -22.0
    cam.distance = 1.6
    cam.lookat[:] = [0.45, 0.0, 0.25]

    cube_body = model.body("cube").id
    cube_qadr = model.jnt_qposadr[model.joint("cube_free").id]
    tcp_site = model.site("2f85_tcp").id
    gripper_actuator_idx = model.actuator("2f85_fingers_actuator").id  # = 7
    assert gripper_actuator_idx == 7, f"expected gripper ctrl idx 7, got {gripper_actuator_idx}"

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / dt)))
    n_frames = int(DURATION_S * FPS)
    print(f"running {n_frames} frames ({sim_steps_per_frame} sim steps/frame)")

    cube_at_start = data.xpos[cube_body].copy()

    frames = []
    milestones = []
    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            pose, g_cmd = current_waypoint(t_sim)
            ctrl.set_ee_pose(pose)
            ctrl.update(model, data)
            # IMPORTANT: separate-channel semantics. The arm controller writes
            # data.ctrl[: model.nu] (nu=8), which would clobber the gripper
            # slot. We reassert the gripper setpoint on every sim step so the
            # two channels stay logically independent.
            data.ctrl[gripper_actuator_idx] = g_cmd
            # Also confine the inverse-dynamics feed-forward to the 7 arm DOFs
            # (qvel indices 0..6). Without this, the full-ID mode tries to
            # control the cube's freejoint DOFs back toward their initial
            # pose, which directly cancels the gripper's physical grasp.
            data.qfrc_applied[7:] = 0.0
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

        t_now = (f + 1) / FPS
        if len(milestones) < 5 and t_now >= [1.5, 3.8, 5.0, 8.0, 10.0][len(milestones)]:
            tag = ["pre_grasp", "closed_on_cube", "lifted", "released", "retreated"][len(milestones)]
            cube_now = data.xpos[cube_body].copy()
            tcp_now = data.site_xpos[tcp_site].copy()
            print(f"  t={t_now:4.1f}s  {tag:16s}  cube={cube_now.round(3).tolist()}  tcp={tcp_now.round(3).tolist()}")
            milestones.append((tag, cube_now))

    cube_final = data.xpos[cube_body].copy()
    tcp_final = data.site_xpos[tcp_site].copy()

    print(f"\ncube start    : {cube_at_start.round(3).tolist()}  (expect ~pickup pad)")
    print(f"cube final    : {cube_final.round(3).tolist()}  (expect ~place pad)")
    displacement = np.linalg.norm(cube_final[:2] - cube_at_start[:2])
    print(f"XY displacement: {displacement*1000:.1f} mm  (expected ~400 mm along +Y)")

    # Success criteria
    place_target = np.array([0.50, 0.20])
    landing_err = np.linalg.norm(cube_final[:2] - place_target)
    above_floor = cube_final[2] > 0.005
    print(f"landing error : {landing_err*1000:.1f} mm from place pad center")
    success = (displacement > 0.30) and (landing_err < 0.08) and above_floor
    print(f"RESULT        : {'SUCCESS' if success else 'FAILURE'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KiB)")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
