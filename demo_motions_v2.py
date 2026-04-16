#!/usr/bin/env python3
"""Extended motion demo suite — figure-8, spiral, obstacle avoidance, cube
stacking, 6-DOF square (tool pointing down). Reuses the best-tuned
current-state inverse-dynamics FF + task-space PD controller.

Generates 5 MP4s:
  iiwa7/demo_motion_figure8.mp4       horizontal Lissajous figure-8
  iiwa7/demo_motion_spiral.mp4        rising conical spiral
  iiwa7/demo_motion_obstacle.mp4      arc over a vertical pillar
  iiwa7/demo_motion_stack.mp4         stack 2 cubes on a base pad
  iiwa7/demo_motion_sq6dof.mp4        horizontal square, EE locked pointing down
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
CLEAN_SCENE    = HERE / "iiwa7" / "iiwa7_clean_scene.xml"
OBSTACLE_SCENE = HERE / "iiwa7" / "iiwa7_obstacle_scene.xml"
STACK_SCENE    = HERE / "iiwa7" / "iiwa7_stack_scene.xml"
SQUARE_SCENE   = HERE / "iiwa7" / "iiwa7_tuned_square_scene.xml"

WIDTH, HEIGHT = 720, 540
FPS = 30
KP_TSK, KD_TSK = 400.0, 40.0
TOOL_OFFSET = np.array([0.0, 0.0, 0.05])

# Pointing-down quaternion: rotate 180 deg about X axis, maps local +Z to world -Z
QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t*t*(3-2*t)


# --------- IK (3-DOF and 6-DOF variants) ----------

def ik_dls_pos(model, data, body_id, target_pos, q0,
               max_iter=200, tol=5e-4, damping=0.05, step=0.5):
    data.qpos[:] = q0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        xpos = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ TOOL_OFFSET
        err = target_pos - xpos
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


def ik_dls_6dof(model, data, body_id, target_pos, target_quat, q0,
                max_iter=300, tol_pos=5e-4, tol_ori=5e-3,
                damping_pos=0.05, damping_ori=0.1, step=0.5,
                pos_weight=1.0, ori_weight=0.5):
    data.qpos[:] = q0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        xpos = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ TOOL_OFFSET
        pos_err = target_pos - xpos

        xquat = data.xquat[body_id].copy()
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, xquat)
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, target_quat, q_inv)
        sgn = 1.0 if q_diff[0] >= 0 else -1.0
        ori_err = 2 * sgn * q_diff[1:4]

        if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(ori_err) < tol_ori:
            return data.qpos.copy()

        jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, body_id)
        J = np.vstack([pos_weight * jp, ori_weight * jr])
        err = np.concatenate([pos_weight * pos_err, ori_weight * ori_err])
        lam2 = np.diag([damping_pos**2]*3 + [damping_ori**2]*3)
        dq = J.T @ np.linalg.solve(J @ J.T + lam2, err)
        data.qpos[:] = data.qpos + step * dq
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)
    return data.qpos.copy()


# --------- motion generators: return list of Waypoint dicts ----------
# Each WP: {"t": float, "pos": [3] or None, "quat": [4] or None, "tags": set}

def mk_wp(t, pos=None, quat=None, tags=()):
    return {"t": t, "pos": pos, "quat": quat, "tags": set(tags)}


def motion_figure8():
    """Horizontal XY plane Lissajous figure-8 at Z=0.55, 2 loops."""
    t_total = 14.0
    t_pre = 3.0
    ax, ay = 0.18, 0.12   # x amplitude, y amplitude
    cx, cy, cz = 0.5, 0.0, 0.55
    wps = [mk_wp(0.0, tags=["home"]), mk_wp(2.0, tags=["ready"])]
    # approach: go to start point
    start = np.array([cx + ax, cy, cz])
    wps.append(mk_wp(t_pre, pos=start, tags=["approach"]))
    n_loops = 2
    t_play = t_total - t_pre
    n_pts = int(t_play * FPS)
    for i in range(1, n_pts + 1):
        t = t_pre + i / FPS
        theta = 2*np.pi*n_loops * (i / n_pts)
        x = cx + ax * np.cos(theta)
        y = cy + ay * np.sin(2 * theta)
        wps.append(mk_wp(t, pos=np.array([x, y, cz])))
    return wps, t_total


def motion_spiral():
    """Rising conical spiral at x-y plane center (0.5,0), z from 0.35 to 0.75,
    radius shrinks from 0.20 to 0.05."""
    t_total = 13.0
    t_pre = 3.0
    cx, cy = 0.5, 0.0
    wps = [mk_wp(0.0, tags=["home"]), mk_wp(2.0, tags=["ready"])]
    wps.append(mk_wp(t_pre, pos=np.array([cx + 0.20, cy, 0.35]), tags=["approach"]))
    n_turns = 3.0
    t_play = t_total - t_pre
    n_pts = int(t_play * FPS)
    for i in range(1, n_pts + 1):
        t = t_pre + i / FPS
        u = i / n_pts  # 0..1
        theta = 2*np.pi * n_turns * u
        r = 0.20 * (1 - u) + 0.05 * u
        z = 0.35 + (0.75 - 0.35) * u
        wps.append(mk_wp(t, pos=np.array([cx + r*np.cos(theta), cy + r*np.sin(theta), z])))
    return wps, t_total


def motion_obstacle():
    """Up-and-over avoidance of a vertical pillar at (0.5, 0). Start (-0.28)
    end (+0.28) in Y, Z=0.55. Lift over the top of the pillar at Z=0.85."""
    wps = [mk_wp(0.0, tags=["home"]), mk_wp(2.0, tags=["ready"])]
    start = np.array([0.50, -0.28, 0.55])
    goal  = np.array([0.50,  0.28, 0.55])
    top_L = np.array([0.50, -0.15, 0.85])   # after lift
    top_R = np.array([0.50,  0.15, 0.85])   # before descent
    # timing
    wps.append(mk_wp(3.5, pos=start, tags=["approach"]))
    # hold at start briefly
    wps.append(mk_wp(4.0, pos=start))
    # lift to top_L
    for i in range(1, 31):
        wps.append(mk_wp(4.0 + i/FPS, pos=start + smoothstep(i/30)*(top_L - start)))
    # cross
    for i in range(1, 46):
        wps.append(mk_wp(5.0 + i/FPS, pos=top_L + smoothstep(i/45)*(top_R - top_L)))
    # descend to goal
    for i in range(1, 31):
        wps.append(mk_wp(6.5 + i/FPS, pos=top_R + smoothstep(i/30)*(goal - top_R)))
    # hold
    for i in range(1, 16):
        wps.append(mk_wp(7.5 + i/FPS, pos=goal))
    # return: straight shot back over (reverse)
    for i in range(1, 31):
        wps.append(mk_wp(8.0 + i/FPS, pos=goal + smoothstep(i/30)*(top_R - goal)))
    for i in range(1, 46):
        wps.append(mk_wp(9.0 + i/FPS, pos=top_R + smoothstep(i/45)*(top_L - top_R)))
    for i in range(1, 31):
        wps.append(mk_wp(10.5 + i/FPS, pos=top_L + smoothstep(i/30)*(start - top_L)))
    return wps, 11.5


def motion_stack():
    """Pick cube_A, stack on base pad. Pick cube_B, stack on cube_A."""
    wps = [mk_wp(0.0, tags=["home"]), mk_wp(2.0, tags=["ready"])]

    cube_A_start = np.array([0.50, -0.22, 0.335])
    cube_B_start = np.array([0.60, -0.22, 0.335])
    pad_pos      = np.array([0.55,  0.20, 0.304])
    stack_z0     = pad_pos[2] + 0.030     # top of base pad + half cube = rough visual
    stack_z1     = stack_z0 + 0.060       # on top of cube_A

    approach_dz = 0.15

    def segment(wps, t_start, p_start, p_end, duration, n_steps, tag_active=None):
        for i in range(1, n_steps + 1):
            frac = smoothstep(i / n_steps)
            tags = [tag_active] if tag_active else []
            wps.append(mk_wp(t_start + i * duration / n_steps,
                              pos=(1-frac)*p_start + frac*p_end, tags=tags))

    # Timeline:
    # t=0..2  home->ready
    # t=2..3  approach above cube_A
    # t=3..4  descend on cube_A
    # t=4..4.5  grab A (hold)
    # t=4.5..5.5  lift A
    # t=5.5..7.0 translate A to above pad
    # t=7.0..8.0 descend A onto pad
    # t=8.0..8.5 release A
    # t=8.5..9.5 lift off
    # t=9.5..10.5 translate to above cube_B
    # t=10.5..11.5 descend to cube_B
    # t=11.5..12.0 grab B
    # t=12.0..13.0 lift B
    # t=13.0..14.5 translate B above stacked position
    # t=14.5..15.5 descend onto cube_A
    # t=15.5..16.0 release B
    # t=16.0..17.0 retreat up

    above_A = cube_A_start + np.array([0, 0, approach_dz])
    above_pad = pad_pos + np.array([0, 0, approach_dz])
    above_B = cube_B_start + np.array([0, 0, approach_dz])
    stacked_A_pos = np.array([pad_pos[0], pad_pos[1], stack_z0])
    stacked_B_pos = np.array([pad_pos[0], pad_pos[1], stack_z1])
    above_stack = stacked_B_pos + np.array([0, 0, approach_dz])

    wps.append(mk_wp(3.0, pos=above_A, tags=["approach"]))
    segment(wps, 3.0, above_A, cube_A_start, 1.0, 30)          # descend to A
    for i in range(int(0.5 * FPS)):                             # grab A
        wps.append(mk_wp(4.0 + i/FPS, pos=cube_A_start, tags=["grab_A"]))
    segment(wps, 4.5, cube_A_start, above_A, 1.0, 30, "grab_A")  # lift A
    segment(wps, 5.5, above_A, above_pad, 1.5, 45, "grab_A")   # move over
    segment(wps, 7.0, above_pad, stacked_A_pos, 1.0, 30, "grab_A")  # descend
    for i in range(int(0.5 * FPS)):                             # release A
        wps.append(mk_wp(8.0 + i/FPS, pos=stacked_A_pos, tags=["release_A"]))
    segment(wps, 8.5, stacked_A_pos, above_pad, 1.0, 30)       # retreat up
    segment(wps, 9.5, above_pad, above_B, 1.0, 30)             # move to above B
    segment(wps, 10.5, above_B, cube_B_start, 1.0, 30)         # descend B
    for i in range(int(0.5 * FPS)):                             # grab B
        wps.append(mk_wp(11.5 + i/FPS, pos=cube_B_start, tags=["grab_B"]))
    segment(wps, 12.0, cube_B_start, above_B, 1.0, 30, "grab_B")  # lift B
    segment(wps, 13.0, above_B, above_stack, 1.5, 45, "grab_B")  # over stack
    segment(wps, 14.5, above_stack, stacked_B_pos, 1.0, 30, "grab_B")  # descend
    for i in range(int(0.5 * FPS)):                             # release B
        wps.append(mk_wp(15.5 + i/FPS, pos=stacked_B_pos, tags=["release_B"]))
    segment(wps, 16.0, stacked_B_pos, above_stack, 1.0, 30)    # retreat
    return wps, 17.0


def motion_square_6dof():
    """Horizontal 20x20 cm square at Z=0.50, EE quat locked to point down."""
    t_total = 12.0
    t_pre = 3.0
    cx, cy, cz = 0.50, 0.0, 0.50
    w, h = 0.20, 0.20
    corners = [
        np.array([cx - w/2, cy - h/2, cz]),
        np.array([cx + w/2, cy - h/2, cz]),
        np.array([cx + w/2, cy + h/2, cz]),
        np.array([cx - w/2, cy + h/2, cz]),
    ]
    wps = [mk_wp(0.0, tags=["home"]), mk_wp(2.0, tags=["ready"])]
    wps.append(mk_wp(t_pre, pos=corners[0], quat=QUAT_DOWN, tags=["approach", "6dof"]))
    n_loops = 1
    edge_time = (t_total - t_pre) / (4 * n_loops)
    t = t_pre
    for loop in range(n_loops):
        for e in range(4):
            p0 = corners[e]
            p1 = corners[(e+1) % 4]
            n = int(edge_time * FPS)
            for i in range(1, n+1):
                f = smoothstep(i / n)
                t_i = t + i * edge_time / n
                wps.append(mk_wp(t_i, pos=(1-f)*p0 + f*p1, quat=QUAT_DOWN, tags=["6dof"]))
            t += edge_time
    return wps, t_total


# --------- precompute joint targets ----------

def precompute_q(model, wps, t_total):
    ee_body = model.body("iiwa_link_7").id
    home_q  = model.key_qpos[model.key("home").id].copy()
    ready_q = model.key_qpos[model.key("ready").id].copy()

    n_frames = int(t_total * FPS)
    times = np.arange(n_frames) / FPS
    q_out = np.zeros((n_frames, model.nq))
    ref_ee = [None] * n_frames
    tags_per_frame = [set() for _ in range(n_frames)]

    t_home = 0.0
    t_ready = next((w["t"] for w in wps if "ready" in w["tags"]), None)

    ee_wps = [w for w in wps if w["pos"] is not None]
    ee_wps.sort(key=lambda w: w["t"])
    ee_ts = np.array([w["t"] for w in ee_wps])
    ee_ps = np.array([w["pos"] for w in ee_wps])
    ee_qs = [w["quat"] for w in ee_wps]
    ee_tg = [w["tags"] for w in ee_wps]

    ik_data = mujoco.MjData(model)
    q_prev = ready_q.copy()

    for f in range(n_frames):
        t = times[f]
        if t_ready is not None and t < t_ready:
            a = smoothstep((t - t_home) / (t_ready - t_home))
            q_out[f] = home_q + a * (ready_q - home_q)
        elif ee_ts.size and t < ee_ts[0]:
            a = smoothstep((t - t_ready) / (ee_ts[0] - t_ready))
            ik_data.qpos[:] = ready_q
            mujoco.mj_forward(model, ik_data)
            ee0 = ik_data.xpos[ee_body] + ik_data.xmat[ee_body].reshape(3,3) @ TOOL_OFFSET
            target = (1-a)*ee0 + a*ee_ps[0]
            target_q = ee_qs[0]
            if target_q is not None:
                q = ik_dls_6dof(model, ik_data, ee_body, target, np.array(target_q), q_prev)
            else:
                q = ik_dls_pos(model, ik_data, ee_body, target, q_prev)
            q_out[f] = q; ref_ee[f] = target; q_prev = q
            tags_per_frame[f] = set(ee_tg[0])
        elif ee_ts.size and t <= ee_ts[-1]:
            idx = np.searchsorted(ee_ts, t) - 1
            idx = max(0, min(idx, len(ee_ts) - 2))
            t0, t1 = ee_ts[idx], ee_ts[idx+1]
            p0, p1 = ee_ps[idx], ee_ps[idx+1]
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            target = (1-frac)*p0 + frac*p1
            q_tgt = ee_qs[idx] if ee_qs[idx] is not None else ee_qs[idx+1]
            if q_tgt is not None:
                q = ik_dls_6dof(model, ik_data, ee_body, target, np.array(q_tgt), q_prev)
            else:
                q = ik_dls_pos(model, ik_data, ee_body, target, q_prev)
            q_out[f] = q; ref_ee[f] = target
            tags_per_frame[f] = ee_tg[idx] | ee_tg[idx+1]
            q_prev = q
        else:
            q_out[f] = q_prev

    return times, q_out, ref_ee, tags_per_frame, ee_body


# --------- runtime overlay helpers ----------

def draw_polyline(scn, points, rgba, radius=0.003):
    for i in range(len(points) - 1):
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                size=np.zeros(3), pos=np.zeros(3),
                                mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                 np.array(points[i]), np.array(points[i+1]))
            scn.ngeom += 1


# --------- main runner ----------

def run(scene_path: Path, wps, t_total: float, out_mp4: Path,
        overlay_builder=None, mocap_handler=None, camera_cfg=None, label=""):
    print(f"\n=== {label}: {scene_path.name} -> {out_mp4.name} ===")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    times, q_frames, ref_ee, tags_per_frame, ee_body = precompute_q(model, wps, t_total)
    n_frames = len(times)

    splines_q   = [CubicSpline(times, q_frames[:, j], bc_type="clamped") for j in range(model.nq)]
    splines_qd  = [sp.derivative(1) for sp in splines_q]
    splines_qdd = [sp.derivative(2) for sp in splines_q]

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    if camera_cfg is None:
        cam.azimuth = 125.0; cam.elevation = -22.0; cam.distance = 2.2; cam.lookat[:] = [0.45, 0, 0.5]
    else:
        cam.azimuth = camera_cfg.get("az", 125.0)
        cam.elevation = camera_cfg.get("el", -22.0)
        cam.distance = camera_cfg.get("d", 2.2)
        cam.lookat[:] = camera_cfg.get("lookat", [0.45, 0, 0.5])

    dt = model.opt.timestep
    sps = max(1, int(round(1.0/FPS / dt)))

    frames = []
    errs = []

    for f in range(n_frames):
        for s in range(sps):
            t_sim = times[f] + s * dt
            q_d   = np.array([sp(t_sim) for sp in splines_q])
            qd_d  = np.array([sp(t_sim) for sp in splines_qd])
            qdd_d = np.array([sp(t_sim) for sp in splines_qdd])
            qdd_cmd = qdd_d + KP_TSK*(q_d - data.qpos) + KD_TSK*(qd_d - data.qvel)
            scratch.qpos[:] = data.qpos
            scratch.qvel[:] = data.qvel
            scratch.qacc[:] = qdd_cmd
            mujoco.mj_inverse(model, scratch)
            data.ctrl[:] = q_d
            data.qfrc_applied[:] = scratch.qfrc_inverse
            mujoco.mj_step(model, data)

        ee_now = data.xpos[ee_body] + data.xmat[ee_body].reshape(3,3) @ TOOL_OFFSET
        if ref_ee[f] is not None:
            errs.append(np.linalg.norm(ref_ee[f] - ee_now))

        if mocap_handler is not None:
            mocap_handler(model, data, tags_per_frame[f], ee_now)

        renderer.update_scene(data, camera=cam)
        if overlay_builder is not None:
            overlay_builder(renderer.scene)
        frames.append(renderer.render())

        if (f + 1) % 180 == 0:
            print(f"  frame {f+1}/{n_frames}")

    errs = np.array(errs)*1000 if errs else np.array([0.0])
    mean, p95, mx = float(errs.mean()), float(np.percentile(errs, 95)), float(errs.max())
    print(f"tracking: mean={mean:.2f} mm  p95={p95:.2f} mm  max={mx:.2f} mm  ({len(errs)} frames)")
    imageio.mimwrite(str(out_mp4), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {out_mp4} ({out_mp4.stat().st_size/1024:.1f} KiB)")
    return mean, p95, mx


# --------- per-demo overlay/mocap specs + main ----------

def main():
    results = {}

    # 1) Figure-8
    def overlay_fig8(scn):
        pts = []
        cx, cy, cz = 0.5, 0, 0.55
        ax, ay = 0.18, 0.12
        for i in range(241):
            theta = 2*np.pi * (i/240)
            pts.append([cx + ax*np.cos(theta), cy + ay*np.sin(2*theta), cz])
        draw_polyline(scn, pts, (0, 1, 1, 0.9), radius=0.004)
    wps, tt = motion_figure8()
    results["figure8"] = run(CLEAN_SCENE, wps, tt,
                              HERE/"iiwa7"/"demo_motion_figure8.mp4",
                              overlay_builder=overlay_fig8,
                              camera_cfg={"az":135, "el":-55, "d":1.8, "lookat":[0.5, 0, 0.55]},
                              label="figure-8")

    # 2) Spiral
    def overlay_spiral(scn):
        pts = []
        cx, cy = 0.5, 0
        n_turns = 3.0
        for i in range(361):
            u = i/360
            th = 2*np.pi*n_turns*u
            r = 0.20*(1-u) + 0.05*u
            z = 0.35 + (0.75-0.35)*u
            pts.append([cx + r*np.cos(th), cy + r*np.sin(th), z])
        draw_polyline(scn, pts, (1, 0.8, 0.2, 0.9), radius=0.004)
    wps, tt = motion_spiral()
    results["spiral"] = run(CLEAN_SCENE, wps, tt,
                             HERE/"iiwa7"/"demo_motion_spiral.mp4",
                             overlay_builder=overlay_spiral,
                             camera_cfg={"az":140, "el":-20, "d":2.2, "lookat":[0.5, 0, 0.55]},
                             label="spiral")

    # 3) Obstacle avoidance (pre-planned over-the-top)
    def overlay_obstacle(scn):
        # re-plot the same up-and-over path used by the waypoint generator
        path = []
        start = [0.50, -0.28, 0.55]
        top_L = [0.50, -0.15, 0.85]
        top_R = [0.50,  0.15, 0.85]
        goal  = [0.50,  0.28, 0.55]
        for i in range(30+1):
            f = smoothstep(i/30); path.append([(1-f)*start[0]+f*top_L[0], (1-f)*start[1]+f*top_L[1], (1-f)*start[2]+f*top_L[2]])
        for i in range(45+1):
            f = smoothstep(i/45); path.append([(1-f)*top_L[0]+f*top_R[0], (1-f)*top_L[1]+f*top_R[1], (1-f)*top_L[2]+f*top_R[2]])
        for i in range(30+1):
            f = smoothstep(i/30); path.append([(1-f)*top_R[0]+f*goal[0], (1-f)*top_R[1]+f*goal[1], (1-f)*top_R[2]+f*goal[2]])
        draw_polyline(scn, path, (0, 1, 0.4, 0.85), radius=0.004)
    wps, tt = motion_obstacle()
    results["obstacle"] = run(OBSTACLE_SCENE, wps, tt,
                               HERE/"iiwa7"/"demo_motion_obstacle.mp4",
                               overlay_builder=overlay_obstacle,
                               camera_cfg={"az":180, "el":-15, "d":2.0, "lookat":[0.5, 0, 0.6]},
                               label="obstacle avoidance")

    # 4) Stacking
    def stack_mocap_handler(model, data, tags, ee_now):
        ida = model.body_mocapid[model.body("cube_A").id]
        idb = model.body_mocapid[model.body("cube_B").id]
        if "grab_A" in tags:
            data.mocap_pos[ida] = ee_now - np.array([0, 0, 0.055])
        if "grab_B" in tags:
            data.mocap_pos[idb] = ee_now - np.array([0, 0, 0.055])
    wps, tt = motion_stack()
    results["stack"] = run(STACK_SCENE, wps, tt,
                            HERE/"iiwa7"/"demo_motion_stack.mp4",
                            mocap_handler=stack_mocap_handler,
                            camera_cfg={"az":135, "el":-28, "d":2.2, "lookat":[0.5, 0, 0.45]},
                            label="stack 2 cubes")

    # 5) 6-DOF square (EE pointing down)
    # Track EE pose error via both position and orientation, but only position
    # goes into the error metric (existing framework). Orientation is enforced
    # by the 6-DOF IK during precomputation.
    def overlay_sq6dof(scn):
        corners = [[0.40, -0.10, 0.50], [0.60, -0.10, 0.50], [0.60, 0.10, 0.50], [0.40, 0.10, 0.50], [0.40, -0.10, 0.50]]
        draw_polyline(scn, corners, (1, 0.4, 1, 0.9), radius=0.005)
    wps, tt = motion_square_6dof()
    results["sq6dof"] = run(CLEAN_SCENE, wps, tt,
                             HERE/"iiwa7"/"demo_motion_sq6dof.mp4",
                             overlay_builder=overlay_sq6dof,
                             camera_cfg={"az":145, "el":-30, "d":2.0, "lookat":[0.5, 0, 0.5]},
                             label="6-DOF square (tool pointing down)")

    print("\n=== summary ===")
    for name, (m, _, x) in results.items():
        print(f"  {name:12s}: mean={m:6.2f} mm  max={x:6.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
