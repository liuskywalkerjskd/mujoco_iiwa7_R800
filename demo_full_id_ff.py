#!/usr/bin/env python3
"""Full inverse-dynamics feedforward — targets <5 mm EE tracking error.

Upgrade over demo_gravity_ff.py:
  - Frame-rate IK targets are fitted with a cubic spline per joint, so the
    reference is C^2 (continuous acceleration) rather than stepped at 30 Hz.
  - At every sim step (500 Hz), the spline is evaluated for q_d, qd_d, qdd_d.
  - mj_inverse is called on a scratch MjData with those values, giving
    tau_ff = M(q_d) qdd_d + C(q_d, qd_d) qd_d + G(q_d) - F_passive(q_d, qd_d).
  - tau_ff is injected via data.qfrc_applied; the PD actuator keeps ctrl=q_d
    so it still rejects residual model error.

Expected: gravity + Coriolis + inertial terms all compensated, residual
error dominated by actuator force-range saturation, mesh-approximation
inertia mismatch, and integration error.

Outputs:
  media/videos/demo_square_full_id_ff.mp4
  TUNING_REPORT.md  (appended)
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
TUNED_SCENE = HERE / "iiwa7" / "iiwa7_tuned_square_scene.xml"
REPORT = HERE / "TUNING_REPORT.md"
OUT_MP4 = HERE / "media" / "videos" / "demo_square_full_id_ff.mp4"

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


def precompute_q_at_frames(model):
    """Return (n_frames, nq) array of joint targets at frame rate."""
    ee = model.body("iiwa_link_7").id
    tool = np.array([0.0, 0.0, 0.05])
    home_q = model.key_qpos[model.key("home").id].copy()
    ready_q = model.key_qpos[model.key("ready").id].copy()

    n = int(TOTAL_S * FPS)
    jt = np.zeros((n, model.nq))
    rp = [None] * n
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
            jt[f] = q; rp[f] = target; q_prev = q
        else:
            tau = t - t_loop
            if tau < loop_dur:
                et = tau / T_EDGE
                idx = int(et) % 4
                frac = smoothstep(et - int(et))
                target = (1-frac)*CORNERS[idx] + frac*CORNERS[(idx+1)%4]
                q = ik_dls(model, ik_d, ee, target, q_prev, tool)
                jt[f] = q; rp[f] = target; q_prev = q
            else:
                a = smoothstep((tau - loop_dur) / T_RETURN)
                jt[f] = q_prev*(1-a) + ready_q*a

    return jt, rp, ee, tool


def main():
    print(f"loading {TUNED_SCENE.name}")
    model = mujoco.MjModel.from_xml_path(str(TUNED_SCENE))
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    print("precomputing IK targets at frame rate...")
    q_frames, ref_path, ee, tool = precompute_q_at_frames(model)
    n_frames = q_frames.shape[0]
    t_frames = np.arange(n_frames) / FPS

    print("fitting cubic splines per joint over trajectory...")
    splines_q = [CubicSpline(t_frames, q_frames[:, j], bc_type="clamped") for j in range(model.nq)]

    # derivative splines (evaluate via CubicSpline.derivative)
    splines_qd  = [sp.derivative(1) for sp in splines_q]
    splines_qdd = [sp.derivative(2) for sp in splines_q]

    def eval_ref(t: float):
        q   = np.array([sp(t)   for sp in splines_q])
        qd  = np.array([sp(t)   for sp in splines_qd])
        qdd = np.array([sp(t)   for sp in splines_qdd])
        return q, qd, qdd

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.azimuth = 145.0; cam.elevation = -28.0
    cam.distance = 2.0; cam.lookat[:] = [0.4, 0.0, 0.55]

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0/FPS / dt)))
    total_sim_steps = n_frames * sim_steps_per_frame
    print(f"running sim + render ({n_frames} frames, {sim_steps_per_frame} "
          f"sim steps/frame, dt={dt}, total {total_sim_steps} sim steps)...")

    ff_peaks = np.zeros(model.nv)
    frames = []
    errs = []

    for f in range(n_frames):
        # render-frame boundary: run sim_steps_per_frame sim steps
        for s in range(sim_steps_per_frame):
            t_sim = f / FPS + s * dt
            q_d, qd_d, qdd_d = eval_ref(t_sim)

            # full inverse dynamics at reference state
            scratch.qpos[:] = q_d
            scratch.qvel[:] = qd_d
            scratch.qacc[:] = qdd_d
            mujoco.mj_inverse(model, scratch)
            tau_ff = scratch.qfrc_inverse.copy()
            ff_peaks = np.maximum(ff_peaks, np.abs(tau_ff))

            data.ctrl[:] = q_d
            data.qfrc_applied[:] = tau_ff
            mujoco.mj_step(model, data)

        ee_now = data.xpos[ee] + data.xmat[ee].reshape(3,3) @ tool
        if ref_path[f] is not None:
            errs.append(np.linalg.norm(ref_path[f] - ee_now))
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

        if (f + 1) % 120 == 0:
            print(f"  frame {f+1}/{n_frames}")

    errs = np.array(errs) * 1000
    mean, p95, mx = float(errs.mean()), float(np.percentile(errs, 95)), float(errs.max())
    print(f"\ntracking error: mean={mean:.2f} mm  p95={p95:.2f} mm  max={mx:.2f} mm  ({len(errs)} frames)")
    print(f"FF peak torque per joint (N·m): "
          + ", ".join(f"J{i+1}={v:.1f}" for i, v in enumerate(ff_peaks)))

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(OUT_MP4), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"wrote {OUT_MP4} ({OUT_MP4.stat().st_size/1024:.1f} KiB)")

    # append new row to TUNING_REPORT.md
    if REPORT.exists():
        content = REPORT.read_text()
        new_row = (
            f"| **tuned + full ID FF (mj_inverse)** | "
            f"**{mean:.2f}** | **{p95:.2f}** | **{mx:.2f}** |\n"
        )
        if new_row not in content:
            anchor = "| **tuned + gravity FF (PD + qfrc_applied)** |"
            if anchor in content:
                # insert right after gravity FF row
                lines = content.splitlines(keepends=True)
                out = []
                for i, ln in enumerate(lines):
                    out.append(ln)
                    if ln.startswith(anchor):
                        out.append(new_row)
                REPORT.write_text("".join(out))
                print(f"appended full-ID-FF row to {REPORT.name}")
            else:
                with REPORT.open("a") as fh:
                    fh.write(f"\n### Update (full ID FF added)\n\n{new_row}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
