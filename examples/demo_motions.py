#!/usr/bin/env python3
"""Multi-motion demo suite for iiwa7 MJCF using the best-tuned controller
(current-state inverse-dynamics FF + task-space PD).

Motions:
  - vertical circle in XZ plane  (radius 0.15 m, center x=0.5, z=0.55)
  - vertical rectangle in XZ plane (0.30 m wide x 0.20 m tall, same center)
  - pick-and-place                (pickup at (0.5,-0.2,0.4), place at
                                   (0.5,+0.2,0.4), cube visibly tracks EE
                                   during the grab phase via mocap)

All motions use the same controller stack:
  cubic-spline C^2 reference -> mj_inverse at (data.qpos, data.qvel) with
  qdd_cmd = qdd_d + Kp_tsk*(q_d-q) + Kd_tsk*(qd_d-qvel) -> qfrc_applied.
  Actuator PD (kp=2000 kd=200) is still active, ctrl=q_d.

Outputs MP4 for each motion plus a combined metrics summary.
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
from typing import Callable

import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
CLEAN_SCENE     = HERE / "scenes" / "iiwa7_clean_scene.xml"
PICKPLACE_SCENE = HERE / "scenes" / "iiwa7_pickplace_scene.xml"

WIDTH, HEIGHT = 720, 540
FPS = 30
KP_TSK = 400.0
KD_TSK = 40.0
IK_TOOL_OFFSET = np.array([0.0, 0.0, 0.05])


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ik_dls(model, data, body_id, target, q0,
           max_iter=200, tol=5e-4, damping=0.05, step=0.5):
    data.qpos[:] = q0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        xpos = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ IK_TOOL_OFFSET
        err = target - xpos
        if np.linalg.norm(err) < tol:
            return data.qpos.copy()
        jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, body_id)
        dq = jp.T @ np.linalg.solve(jp @ jp.T + damping**2 * np.eye(3), err)
        data.qpos[:] = data.qpos + step * dq
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)
    return data.qpos.copy()


# ---- motion generators: return list of (t, target_ee, grab_flag) ----

def motion_vertical_circle(t_total=16.0, n_loops=2, center=(0.5, 0, 0.55), radius=0.15):
    """XZ-plane circle traced counter-clockwise starting at 3 o'clock."""
    wp = []
    # Phase A: home -> ready -> starting point (top of circle)
    start_point = (center[0] + radius, center[1], center[2])
    t_preroll = 3.0
    wp.append((0.0, None, False, "home"))
    wp.append((2.0, None, False, "ready"))
    wp.append((t_preroll, np.array(start_point), False, "approach"))

    n_pts = int((t_total - t_preroll) * FPS)
    t_loops_total = t_total - t_preroll
    for i in range(1, n_pts + 1):
        t = t_preroll + i * t_loops_total / n_pts
        theta = 2 * np.pi * n_loops * (i / n_pts)
        x = center[0] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
        wp.append((t, np.array([x, center[1], z]), False, None))
    return wp, t_total


def motion_vertical_rectangle(t_total=15.0, center=(0.5, 0, 0.55), size=(0.30, 0.20)):
    """XZ-plane rectangle, 2 loops."""
    w, h = size
    corners = np.array([
        [center[0] - w/2, center[1], center[2] - h/2],  # bottom-left
        [center[0] + w/2, center[1], center[2] - h/2],  # bottom-right
        [center[0] + w/2, center[1], center[2] + h/2],  # top-right
        [center[0] - w/2, center[1], center[2] + h/2],  # top-left
    ])
    n_loops = 2
    edge_time = 1.25

    wp = []
    wp.append((0.0, None, False, "home"))
    wp.append((2.0, None, False, "ready"))
    wp.append((3.5, corners[0], False, "approach"))

    for loop in range(n_loops):
        for e in range(4):
            p0 = corners[e]
            p1 = corners[(e+1) % 4]
            n_steps = int(edge_time * FPS)
            for s in range(1, n_steps + 1):
                frac = smoothstep(s / n_steps)
                t = 3.5 + (loop * 4 + e) * edge_time + s * edge_time / n_steps
                wp.append((t, (1-frac)*p0 + frac*p1, False, None))
    return wp, 3.5 + n_loops * 4 * edge_time + 0.5


def motion_pick_place(t_total=14.0):
    """Pick cube at (0.5,-0.2,0.4), place at (0.5,+0.2,0.4)."""
    approach_h = 0.6
    target_h   = 0.40
    pickup  = np.array([0.50, -0.20, target_h])
    above_pu = pickup + np.array([0, 0, approach_h - target_h])
    place   = np.array([0.50,  0.20, target_h])
    above_pl = place + np.array([0, 0, approach_h - target_h])

    wp = []
    wp.append((0.0, None, False, "home"))
    wp.append((2.0, None, False, "ready"))
    wp.append((3.5, above_pu, False, "approach pickup"))
    # descend to pickup
    n = int(1.0 * FPS)
    for i in range(1, n+1):
        f = smoothstep(i / n)
        t = 3.5 + i / FPS
        wp.append((t, (1-f)*above_pu + f*pickup, False, None))
    # grab pause (cube attaches)
    for i in range(int(0.5 * FPS)):
        t = 4.5 + i / FPS
        wp.append((t, pickup, True, "grab"))
    # lift up with cube
    for i in range(1, n+1):
        f = smoothstep(i / n)
        t = 5.0 + i / FPS
        wp.append((t, (1-f)*pickup + f*above_pu, True, None))
    # translate to above place
    n2 = int(1.5 * FPS)
    for i in range(1, n2+1):
        f = smoothstep(i / n2)
        t = 6.0 + i / FPS
        wp.append((t, (1-f)*above_pu + f*above_pl, True, None))
    # descend
    for i in range(1, n+1):
        f = smoothstep(i / n)
        t = 7.5 + i / FPS
        wp.append((t, (1-f)*above_pl + f*place, True, None))
    # release (cube detaches)
    for i in range(int(0.5 * FPS)):
        t = 8.5 + i / FPS
        wp.append((t, place, False, "release"))
    # retreat
    for i in range(1, n+1):
        f = smoothstep(i / n)
        t = 9.0 + i / FPS
        wp.append((t, (1-f)*place + f*above_pl, False, None))
    # hold
    for i in range(int(1.0 * FPS)):
        t = 10.0 + i / FPS
        wp.append((t, above_pl, False, "done"))
    return wp, 11.5


# ---- controller + render ----

def precompute_joint_targets(model, wp, t_total):
    """Given waypoints list of (t, target_ee or None for key, grab, tag),
    solve IK per frame and return (times, q_frames, ref_ee, grab_flags)."""
    ee_body = model.body("iiwa_link_7").id
    home_q  = model.key_qpos[model.key("home").id].copy()
    ready_q = model.key_qpos[model.key("ready").id].copy()

    n_frames = int(t_total * FPS)
    times = np.arange(n_frames) / FPS
    q_out = np.zeros((n_frames, model.nq))
    ref_ee = [None] * n_frames
    grab_flags = np.zeros(n_frames, dtype=bool)

    # Split waypoints: key-based phases vs EE-target-based
    # Interpolate between sequential waypoint (t, pos_or_None).
    # Build separate timeline: EE target t -> pos. For frames before the
    # first EE waypoint, ramp home -> ready based on key waypoints.

    # Identify key events
    t_home  = next((w[0] for w in wp if w[3] == "home"), 0.0)
    t_ready = next((w[0] for w in wp if w[3] == "ready"), None)
    ee_wps = [(w[0], w[1], w[2]) for w in wp if w[1] is not None]

    # Sort EE waypoints by time for lerp lookup
    ee_wps.sort(key=lambda x: x[0])
    ee_times = np.array([w[0] for w in ee_wps])
    ee_poss  = np.array([w[1] for w in ee_wps])
    ee_grabs = np.array([w[2] for w in ee_wps])

    ik_data = mujoco.MjData(model)
    q_prev = ready_q.copy()

    for f in range(n_frames):
        t = times[f]
        if t_ready is not None and t < t_ready:
            # Home -> ready joint-space ramp
            a = smoothstep((t - t_home) / (t_ready - t_home))
            q_out[f] = home_q + a * (ready_q - home_q)
        elif ee_times.size and t < ee_times[0]:
            # ramp ready -> first EE target (in task space via IK at progressing target)
            a = smoothstep((t - t_ready) / (ee_times[0] - t_ready))
            ik_data.qpos[:] = ready_q
            mujoco.mj_forward(model, ik_data)
            ee0 = ik_data.xpos[ee_body] + ik_data.xmat[ee_body].reshape(3,3) @ IK_TOOL_OFFSET
            target = (1-a)*ee0 + a*ee_poss[0]
            q = ik_dls(model, ik_data, ee_body, target, q_prev)
            q_out[f] = q; ref_ee[f] = target; q_prev = q
        elif ee_times.size and t <= ee_times[-1]:
            # Linear interp between bracketing EE waypoints
            idx = np.searchsorted(ee_times, t) - 1
            idx = max(0, min(idx, len(ee_times) - 2))
            t0, t1 = ee_times[idx], ee_times[idx+1]
            p0, p1 = ee_poss[idx], ee_poss[idx+1]
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            target = (1-frac)*p0 + frac*p1
            q = ik_dls(model, ik_data, ee_body, target, q_prev)
            q_out[f] = q; ref_ee[f] = target
            grab_flags[f] = ee_grabs[idx] or ee_grabs[idx+1]
            q_prev = q
        else:
            q_out[f] = q_prev

    return times, q_out, ref_ee, grab_flags, ee_body


def draw_polyline(scn, points, rgba, radius=0.003):
    """Add a polyline overlay as connected capsules to mjvScene."""
    n = len(points)
    for i in range(n - 1):
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3), pos=np.zeros(3),
                mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                np.array(points[i]), np.array(points[i+1]),
            )
            scn.ngeom += 1


def run(scene_path: Path, wp, t_total: float, out_mp4: Path,
        overlay_builder=None, has_mocap_cube=False, label=""):
    print(f"\n=== {label}: {scene_path.name} -> {out_mp4.name} ===")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data  = mujoco.MjData(model)
    times, q_frames, ref_ee, grab_flags, ee_body = precompute_joint_targets(model, wp, t_total)
    n_frames = len(times)

    splines_q   = [CubicSpline(times, q_frames[:, j], bc_type="clamped") for j in range(model.nq)]
    splines_qd  = [sp.derivative(1) for sp in splines_q]
    splines_qdd = [sp.derivative(2) for sp in splines_q]

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    ctrl = IiwaEEController(model, data, mode="full_id_ff_current", kp_tsk=KP_TSK, kd_tsk=KD_TSK)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 125.0; cam.elevation = -22.0
    cam.distance = 2.2; cam.lookat[:] = [0.45, 0.0, 0.5]

    cube_mocap_id = None
    cube_initial_pos = None
    if has_mocap_cube:
        cube_body_id = model.body("cube").id
        cube_mocap_id = model.body_mocapid[cube_body_id]
        cube_initial_pos = data.mocap_pos[cube_mocap_id].copy()

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0/FPS / dt)))

    frames = []
    errs = []

    for f in range(n_frames):
        for s in range(sim_steps_per_frame):
            t_sim = times[f] + s * dt
            q_d = np.array([sp(t_sim) for sp in splines_q])
            ctrl.set_joint_target(q_d)
            ctrl.update(model, data)
            mujoco.mj_step(model, data)
        # mocap cube: stick to EE during grab phase
        if has_mocap_cube and grab_flags[f]:
            ee_pos = data.xpos[ee_body] + data.xmat[ee_body].reshape(3,3) @ IK_TOOL_OFFSET
            # place cube just below EE (minus tool offset height, so it looks grabbed)
            data.mocap_pos[cube_mocap_id] = ee_pos - np.array([0, 0, 0.055])

        ee_now = data.xpos[ee_body] + data.xmat[ee_body].reshape(3,3) @ IK_TOOL_OFFSET
        if ref_ee[f] is not None:
            errs.append(np.linalg.norm(ref_ee[f] - ee_now))

        renderer.update_scene(data, camera=cam)
        if overlay_builder is not None:
            overlay_builder(renderer.scene)
        frames.append(renderer.render())

        if (f + 1) % 150 == 0:
            print(f"  frame {f+1}/{n_frames}")

    errs = np.array(errs) * 1000 if errs else np.array([0.0])
    mean, p95, mx = float(errs.mean()), float(np.percentile(errs, 95)), float(errs.max())
    print(f"tracking: mean={mean:.2f} mm  p95={p95:.2f} mm  max={mx:.2f} mm  ({len(errs)} frames)")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_mp4), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {out_mp4} ({out_mp4.stat().st_size/1024:.1f} KiB)")
    return mean, p95, mx


def main():
    # --- Motion 1: vertical circle ---
    wp_c, tt_c = motion_vertical_circle()
    circle_pts = []
    center_c = (0.5, 0, 0.55); r_c = 0.15
    for i in range(72 + 1):
        theta = 2 * np.pi * i / 72
        circle_pts.append([center_c[0] + r_c*np.cos(theta), center_c[1], center_c[2] + r_c*np.sin(theta)])
    def overlay_circle(scn):
        draw_polyline(scn, circle_pts, (0, 1, 1, 0.9), radius=0.004)

    m1, _, x1 = run(CLEAN_SCENE, wp_c, tt_c,
                     HERE.parent / "media" / "videos" / "demo_motion_vcircle.mp4",
                     overlay_builder=overlay_circle,
                     label="vertical circle")

    # --- Motion 2: vertical rectangle ---
    wp_r, tt_r = motion_vertical_rectangle()
    rect_corners = [
        [0.35, 0, 0.45], [0.65, 0, 0.45], [0.65, 0, 0.65], [0.35, 0, 0.65], [0.35, 0, 0.45]
    ]
    def overlay_rect(scn):
        draw_polyline(scn, rect_corners, (0, 1, 1, 0.9), radius=0.005)

    m2, _, x2 = run(CLEAN_SCENE, wp_r, tt_r,
                     HERE.parent / "media" / "videos" / "demo_motion_vrect.mp4",
                     overlay_builder=overlay_rect,
                     label="vertical rectangle")

    # --- Motion 3: pick and place ---
    wp_p, tt_p = motion_pick_place()
    m3, _, x3 = run(PICKPLACE_SCENE, wp_p, tt_p,
                     HERE.parent / "media" / "videos" / "demo_motion_pickplace.mp4",
                     overlay_builder=None,
                     has_mocap_cube=True,
                     label="pick and place")

    print("\n=== summary ===")
    print(f"vertical circle:    mean {m1:.2f} mm  max {x1:.2f} mm")
    print(f"vertical rectangle: mean {m2:.2f} mm  max {x2:.2f} mm")
    print(f"pick and place:     mean {m3:.2f} mm  max {x3:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
