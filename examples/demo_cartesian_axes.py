#!/usr/bin/env python3
"""Cartesian six-axis EE motion demo.

From the ready-pose home configuration the end-effector is driven
5 cm along each of +X, -X, +Y, -Y, +Z, -Z in the BASE frame, in
turn. After every probe the EE returns to the home position before
the next probe starts. Orientation is held constant (the home
orientation) throughout, so only translation is exercised.

Pipeline: `IiwaEEController.set_ee_pose(pose7)` per sim step
(damped-least-squares IK + full-ID feedforward). The demo renders
an MP4 and prints the Cartesian tracking stats.

Output:  media/videos/demo_cartesian_axes.mp4
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

SCENE = REPO / "examples" / "scenes" / "iiwa7_clean_scene.xml"
OUT = REPO / "media" / "videos" / "demo_cartesian_axes.mp4"

FPS = 30
WIDTH, HEIGHT = 720, 540
STEP = 0.05          # 5 cm probe amplitude
MOVE_S = 1.0         # out-and-back leg duration
HOLD_S = 0.4         # pause at each extreme / at home

AXES = [
    ("+X", np.array([ 1.0,  0.0,  0.0])),
    ("-X", np.array([-1.0,  0.0,  0.0])),
    ("+Y", np.array([ 0.0,  1.0,  0.0])),
    ("-Y", np.array([ 0.0, -1.0,  0.0])),
    ("+Z", np.array([ 0.0,  0.0,  1.0])),
    ("-Z", np.array([ 0.0,  0.0, -1.0])),
]


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(s: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = smoothstep(s)
    return (1 - k) * a + k * b


def build_schedule(p_home: np.ndarray):
    """Return a list of (t0, t1, pos_fn) segments, and total duration."""
    schedule = []
    t = 0.0

    # Initial settle at home
    schedule.append((t, t + HOLD_S, lambda s, h=p_home: h))
    t += HOLD_S

    for name, dvec in AXES:
        p_far = p_home + STEP * dvec
        schedule.append((t, t + MOVE_S, lambda s, a=p_home, b=p_far: lerp(s, a, b)))
        t += MOVE_S
        schedule.append((t, t + HOLD_S, lambda s, b=p_far: b))
        t += HOLD_S
        schedule.append((t, t + MOVE_S, lambda s, a=p_far, b=p_home: lerp(s, a, b)))
        t += MOVE_S
        schedule.append((t, t + HOLD_S, lambda s, h=p_home: h))
        t += HOLD_S

    return schedule, t


def pose_at(schedule, t_sim: float, p_home: np.ndarray) -> np.ndarray:
    for t0, t1, fn in schedule:
        if t_sim <= t1:
            s = (t_sim - t0) / max(t1 - t0, 1e-6)
            return fn(s)
    return p_home


def current_axis_label(t_sim: float) -> str:
    # Each axis block takes 2*MOVE_S + 2*HOLD_S; the first HOLD_S is pre-phase
    block = 2 * MOVE_S + 2 * HOLD_S
    if t_sim < HOLD_S:
        return "HOME"
    idx = int((t_sim - HOLD_S) // block)
    if idx >= len(AXES):
        return "HOME"
    return AXES[idx][0]


def main() -> int:
    print(f"loading {SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    # reset to the home keyframe (inherited from iiwa7_tuned.xml)
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    except Exception:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ctrl = IiwaEEController(model, data)

    # Capture the home EE pose to use as the fixed orientation target and
    # the return anchor for every axis probe.
    link7 = model.body("iiwa_link_7").id
    R = data.xmat[link7].reshape(3, 3)
    p_home = data.xpos[link7] + R @ ctrl.tool_offset  # world = base here
    q_home_wxyz = data.xquat[link7].copy()
    q_home_xyzw = np.array([q_home_wxyz[1], q_home_wxyz[2], q_home_wxyz[3], q_home_wxyz[0]])
    print(f"home TCP pos  (base frame): {p_home.round(4).tolist()}")
    print(f"home TCP quat (xyzw)      : {q_home_xyzw.round(4).tolist()}")

    schedule, DURATION_S = build_schedule(p_home)
    print(f"schedule total: {DURATION_S:.2f} s  ({len(AXES)} axes x (out+hold+back+hold))")

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 140.0
    cam.elevation = -22.0
    cam.distance = 1.7
    cam.lookat[:] = [0.4, 0.0, 0.5]

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / dt)))
    n_frames = int(DURATION_S * FPS) + 1
    print(f"rendering {n_frames} frames ({sim_steps_per_frame} sim steps/frame)")

    frames = []
    tracking_errs_mm = []
    seen_axes = set()
    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            pos_target = pose_at(schedule, t_sim, p_home)
            pose7 = np.concatenate([pos_target, q_home_xyzw])
            ctrl.set_ee_pose(pose7)
            ctrl.update(model, data)
            mujoco.mj_step(model, data)

        # Log frame-level tracking (at frame time)
        t_now = f / FPS
        pos_target = pose_at(schedule, t_now, p_home)
        R_now = data.xmat[link7].reshape(3, 3)
        tcp_now = data.xpos[link7] + R_now @ ctrl.tool_offset
        err_mm = np.linalg.norm(tcp_now - pos_target) * 1000
        tracking_errs_mm.append(err_mm)

        # Progress log once per axis transition
        lbl = current_axis_label(t_now)
        if lbl not in seen_axes and lbl != "HOME":
            seen_axes.add(lbl)
            print(f"  t={t_now:5.2f}s  probing {lbl}   TCP={tcp_now.round(4).tolist()}  err={err_mm:.2f} mm")

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    errs = np.array(tracking_errs_mm)
    print(
        f"\nTCP tracking: mean={errs.mean():.2f} mm  "
        f"p95={np.percentile(errs, 95):.2f} mm  max={errs.max():.2f} mm"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
