#!/usr/bin/env python3
"""Minimal 'send EE pose, watch iiwa7 track it' demo.

Shows how a user drops iiwa7_mjcf into their own scene and needs nothing
more than the end-effector target pose to drive the arm. Internally the
IiwaEEController handles IK, feed-forward inverse dynamics, and
task-space PD — the caller never sees joint-space targets.

Renders an 18-second headless video (MUJOCO_GL=egl) showing the arm
following a 3D Lissajous path plus a hold segment.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

# Make sibling package importable when running from examples/
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


def desired_ee_pose(t: float) -> np.ndarray:
    """User-defined EE trajectory in world frame. Replace this with your
    own planner / teleop input / RL policy output — the controller does
    not care where the pose comes from."""
    ramp_end = 3.0
    if t < ramp_end:
        # smooth ramp from a neutral point to the Lissajous starting point
        a = smoothstep(t / ramp_end)
        neutral = np.array([0.5, 0.0, 0.75])
        start = np.array([0.5, 0.15, 0.55])
        return (1 - a) * neutral + a * start
    # Lissajous
    tt = t - ramp_end
    cx, cy, cz = 0.5, 0.0, 0.55
    ax, ay, az = 0.18, 0.15, 0.08
    w = 2 * np.pi * 0.1
    x = cx + ax * np.cos(w * tt)
    y = cy + ay * np.sin(2 * w * tt)
    z = cz + az * np.sin(w * tt)
    return np.array([x, y, z])


def main() -> int:
    print(f"loading {SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)

    # === One-liner controller init ===
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
    err_log = []
    ee_body = model.body("iiwa_link_7").id
    tool_offset = np.array([0.0, 0.0, 0.05])

    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            ee_target = desired_ee_pose(t_sim)

            # === The only two calls the user needs to drive the arm ===
            ctrl.set_ee_target(pos=ee_target)
            ctrl.update(model, data)

            mujoco.mj_step(model, data)

        # Log tracking error at frame boundary
        ee_actual = data.xpos[ee_body] + data.xmat[ee_body].reshape(3, 3) @ tool_offset
        err_log.append(np.linalg.norm(desired_ee_pose(f / FPS) - ee_actual))

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

        if (f + 1) % 90 == 0:
            print(f"  frame {f+1}/{n_frames}  current EE err={err_log[-1]*1000:.2f} mm")

    errs = np.array(err_log) * 1000
    print(
        f"\nEE tracking error: mean={errs.mean():.2f} mm  "
        f"p95={np.percentile(errs, 95):.2f} mm  max={errs.max():.2f} mm"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
