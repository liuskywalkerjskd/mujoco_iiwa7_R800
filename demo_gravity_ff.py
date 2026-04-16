#!/usr/bin/env python3
"""Gravity feedforward demo — adds inverse-dynamics gravity torque on top of
the tuned (menagerie-style) PD actuator.

Compensation recipe per frame:
  1. Load target qpos from IK precomputation (same as demo_tuning_compare.py).
  2. In a scratch MjData, set qpos=target, qvel=0, call mj_forward.
     qfrc_bias at (q, qvel=0) reduces to pure gravity torque g(q).
  3. In the main sim, set data.ctrl=target (PD tracks) and
     data.qfrc_applied=g(q) (gravity cancelled).
  4. mj_step.

Equation of motion after compensation:
  M(q) qddot = tau_PD(q, v) + qfrc_applied - qfrc_bias
             ~ tau_PD(q, v)            (since qfrc_applied == g(q) ~ qfrc_bias|v=0)
  -> PD operates on a (near-)gravity-free arm, steady-state error collapses.

Compares against iiwa7_tuned_square_scene.xml (same PD gains, no FF).

Outputs:
  media/videos/demo_square_gravity_ff.mp4
  TUNING_REPORT.md (appended)
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
TUNED_SCENE = HERE / "iiwa7" / "iiwa7_tuned_square_scene.xml"
REPORT = HERE / "TUNING_REPORT.md"
OUT_MP4 = HERE / "media" / "videos" / "demo_square_gravity_ff.mp4"

WIDTH, HEIGHT = 720, 540
FPS = 30

CORNERS = np.array([
    [0.35, -0.15, 0.55],
    [0.65, -0.15, 0.55],
    [0.65,  0.15, 0.55],
    [0.35,  0.15, 0.55],
])

T_HOME_READY     = 2.0
T_READY_APPROACH = 2.0
T_EDGE           = 2.5
N_LOOPS          = 2
T_RETURN         = 2.5
TOTAL_S = T_HOME_READY + T_READY_APPROACH + T_EDGE * 4 * N_LOOPS + T_RETURN


def smoothstep(t): t = max(0.0, min(1.0, t)); return t*t*(3-2*t)


def ik_dls(model, data, body_id, target, q0, tool_offset,
           max_iter=200, tol=5e-4, damping=0.05, step=0.5):
    data.qpos[:] = q0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        xpos = data.xpos[body_id] + data.xmat[body_id].reshape(3,3) @ tool_offset
        err = target - xpos
        if np.linalg.norm(err) < tol:
            return data.qpos.copy()
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, body_id)
        dq = jp.T @ np.linalg.solve(jp @ jp.T + damping**2 * np.eye(3), err)
        data.qpos[:] = data.qpos + step * dq
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                data.qpos[j] = np.clip(data.qpos[j], lo, hi)
    return data.qpos.copy()


def precompute(model):
    ee = model.body("iiwa_link_7").id
    tool = np.array([0.0, 0.0, 0.05])
    home_q = model.key_qpos[model.key("home").id].copy()
    ready_q = model.key_qpos[model.key("ready").id].copy()

    n = int(TOTAL_S * FPS)
    jt, rp = [None]*n, [None]*n
    t_ready_end    = T_HOME_READY
    t_approach_end = t_ready_end + T_READY_APPROACH
    t_loop         = t_approach_end
    loop_dur       = T_EDGE * 4 * N_LOOPS

    ik_d = mujoco.MjData(model)
    q_prev = ready_q.copy()

    for f in range(n):
        t = f / FPS
        if t < t_ready_end:
            a = smoothstep(t / T_HOME_READY)
            jt[f] = home_q + a * (ready_q - home_q)
        elif t < t_approach_end:
            a = smoothstep((t - t_ready_end) / T_READY_APPROACH)
            ik_d.qpos[:] = ready_q
            mujoco.mj_forward(model, ik_d)
            p0 = ik_d.xpos[ee] + ik_d.xmat[ee].reshape(3,3) @ tool
            target = (1-a)*p0 + a*CORNERS[0]
            q = ik_dls(model, ik_d, ee, target, q_prev, tool)
            jt[f], rp[f] = q, target; q_prev = q
        else:
            tau = t - t_loop
            if tau < loop_dur:
                et = tau / T_EDGE
                idx = int(et) % 4
                frac = smoothstep(et - int(et))
                target = (1-frac)*CORNERS[idx] + frac*CORNERS[(idx+1)%4]
                q = ik_dls(model, ik_d, ee, target, q_prev, tool)
                jt[f], rp[f] = q, target; q_prev = q
            else:
                a = smoothstep((tau - loop_dur) / T_RETURN)
                jt[f] = q_prev*(1-a) + ready_q*a
    return jt, rp, ee, tool


def main():
    print(f"loading {TUNED_SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(TUNED_SCENE))
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    print("precomputing IK waypoints...")
    jt, rp, ee, tool = precompute(model)
    n = len(jt)

    # precompute gravity FF vector for every frame
    print("precomputing gravity feedforward torques...")
    grav_ff = np.zeros((n, model.nv))
    for f in range(n):
        scratch.qpos[:] = jt[f]
        scratch.qvel[:] = 0
        mujoco.mj_forward(model, scratch)
        grav_ff[f] = scratch.qfrc_bias.copy()

    # report max FF torque magnitude per joint (debug info)
    ff_peak = np.abs(grav_ff).max(axis=0)
    print(f"  FF peak torque per joint (Nm): "
          + ", ".join(f"J{i+1}={v:.1f}" for i, v in enumerate(ff_peak)))

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 145.0; cam.elevation = -28.0
    cam.distance = 2.0; cam.lookat[:] = [0.4, 0.0, 0.55]

    sim_steps_per_frame = max(1, int(round(1.0/FPS / model.opt.timestep)))
    print(f"running sim + render ({n} frames, {sim_steps_per_frame} sim steps/frame)...")

    frames, errs = [], []
    for f in range(n):
        data.ctrl[:] = jt[f]
        data.qfrc_applied[:] = grav_ff[f]
        for _ in range(sim_steps_per_frame):
            mujoco.mj_step(model, data)
        ee_now = data.xpos[ee] + data.xmat[ee].reshape(3,3) @ tool
        if rp[f] is not None:
            errs.append(np.linalg.norm(rp[f] - ee_now))
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    errs = np.array(errs) * 1000
    mean, p95, mx = float(errs.mean()), float(np.percentile(errs, 95)), float(errs.max())
    print(f"tracking error along reference: mean={mean:.2f} mm  p95={p95:.2f} mm  max={mx:.2f} mm  ({len(errs)} frames)")

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT_MP4), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT_MP4} ({OUT_MP4.stat().st_size/1024:.1f} KiB)")

    # append row to existing TUNING_REPORT.md
    if REPORT.exists():
        content = REPORT.read_text()
        new_row = (
            f"| **tuned + gravity FF (PD + qfrc_applied)** | "
            f"**{mean:.2f}** | **{p95:.2f}** | **{mx:.2f}** |\n"
        )
        marker = "| **tuned (general, kp=2000 kd=200)**"
        if marker in content and new_row not in content:
            content = content.replace(
                marker + f"| **{22.28:.2f}** | **{30.37:.2f}** | **{31.90:.2f}** |\n",
                marker + f"| **{22.28:.2f}** | **{30.37:.2f}** | **{31.90:.2f}** |\n" + new_row,
                1,
            )
            REPORT.write_text(content)
            print(f"appended gravity-FF row to {REPORT.name}")
        else:
            # fallback: just append a note at the end
            with REPORT.open("a") as fh:
                fh.write(f"\n### Update (gravity FF added)\n\n{new_row}\n")
            print(f"appended (fallback) to {REPORT.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
