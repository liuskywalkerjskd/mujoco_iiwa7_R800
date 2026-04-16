#!/usr/bin/env python3
"""Headless MuJoCo demo for iiwa7 — records an MP4 without a display.

Trajectory (total 10 s at 30 fps):
  - 0.0 - 3.0 s : smooth ramp from 'home' keyframe to 'ready' keyframe
                  (position actuators drive the motion)
  - 3.0 - 10.0 s: sinusoidal sweep on J1/J2/J4/J6 to show multi-axis
                  motion + dynamics-consistent damping

Writes:
  media/videos/demo.mp4
"""
from __future__ import annotations

import os
# Pick a headless backend before importing mujoco.
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
MJCF = HERE / "scenes" / "iiwa7_scene.xml"
OUT = HERE.parent / "media" / "videos" / "demo.mp4"

WIDTH, HEIGHT = 640, 480
FPS = 30
DURATION_S = 10.0
RAMP_S = 3.0


def smooth(t: float) -> float:
    # smoothstep: 0 at t=0, 1 at t=1, zero first-derivative at endpoints
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def main() -> int:
    if not MJCF.exists():
        print(f"[FAIL] not found: {MJCF}", file=sys.stderr)
        return 1

    print(f"loading {MJCF.name}")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)

    home_id = model.key("home").id
    home_q = model.key_qpos[home_id].copy()  # start at ready (no home ramp)
    ready_q = model.key_qpos[home_id].copy()

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    ctrl = IiwaEEController(model, data, mode="pd_only")

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = 2.2
    camera.lookat[:] = [0.0, 0.0, 0.55]

    n_frames = int(DURATION_S * FPS)
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / model.opt.timestep)))

    print(
        f"recording {n_frames} frames @ {FPS}fps "
        f"({sim_steps_per_frame} sim steps/frame, dt={model.opt.timestep})"
    )

    frames: list[np.ndarray] = []
    for f in range(n_frames):
        t = f / FPS
        if t <= RAMP_S:
            alpha = smooth(t / RAMP_S)
            target = home_q + alpha * (ready_q - home_q)
        else:
            # sinusoidal sweep around ready pose
            w = 2 * np.pi * 0.25  # 0.25 Hz
            phase = t - RAMP_S
            offset = np.zeros(7)
            offset[0] = 0.5 * np.sin(w * phase)
            offset[1] = 0.25 * np.sin(w * phase + 0.5)
            offset[3] = 0.25 * np.sin(w * phase + 1.0)
            offset[5] = 0.5 * np.sin(w * phase + 1.5)
            target = ready_q + offset

        ctrl.set_joint_target(target)
        for _ in range(sim_steps_per_frame):
            ctrl.update(model, data)
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        if (f + 1) % 30 == 0:
            print(f"  frame {f+1}/{n_frames}  qpos[0:3]={data.qpos[0:3]}")

    if np.any(np.isnan(data.qpos)):
        print("[FAIL] NaN in qpos — sim diverged", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"[OK] size={OUT.stat().st_size/1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
