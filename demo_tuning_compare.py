#!/usr/bin/env python3
"""Compare legacy (position actuator kp=400/200/100) vs tuned (menagerie-style
general actuator kp=2000 kd=200) — both driving the same square trajectory
under full actuator-in-loop dynamics.

Outputs:
  iiwa7/demo_square_actuated_legacy.mp4
  iiwa7/demo_square_actuated_tuned.mp4
  TUNING_REPORT.md  (summary table with mean/max EE tracking error)
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
LEGACY_SCENE = HERE / "iiwa7" / "iiwa7_square_scene.xml"
TUNED_SCENE  = HERE / "iiwa7" / "iiwa7_tuned_square_scene.xml"
REPORT_PATH  = HERE / "TUNING_REPORT.md"

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


def precompute_trajectory(model):
    """Solve IK per frame for the reference path. Returns (joint_targets,
    ref_path_or_None per frame)."""
    ee_body = model.body("iiwa_link_7").id
    tool_offset = np.array([0.0, 0.0, 0.05])
    home_q = model.key_qpos[model.key("home").id].copy()
    ready_q = model.key_qpos[model.key("ready").id].copy()

    n_frames = int(TOTAL_S * FPS)
    joint_target = [None] * n_frames
    ref_path = [None] * n_frames

    t_ready_end    = T_HOME_READY
    t_approach_end = t_ready_end + T_READY_APPROACH
    t_loop_start   = t_approach_end
    loop_duration  = T_EDGE * 4 * N_LOOPS

    ik_data = mujoco.MjData(model)
    q_prev = ready_q.copy()

    for f in range(n_frames):
        t = f / FPS
        if t < t_ready_end:
            a = smoothstep(t / T_HOME_READY)
            joint_target[f] = home_q + a * (ready_q - home_q)
        elif t < t_approach_end:
            a = smoothstep((t - t_ready_end) / T_READY_APPROACH)
            ik_data.qpos[:] = ready_q
            mujoco.mj_forward(model, ik_data)
            p0 = ik_data.xpos[ee_body] + ik_data.xmat[ee_body].reshape(3,3) @ tool_offset
            target = (1 - a) * p0 + a * CORNERS[0]
            q = ik_dls(model, ik_data, ee_body, target, q_prev, tool_offset)
            joint_target[f] = q
            ref_path[f] = target
            q_prev = q
        else:
            tau = t - t_loop_start
            if tau < loop_duration:
                et = tau / T_EDGE
                idx = int(et) % 4
                frac = smoothstep(et - int(et))
                p0, p1 = CORNERS[idx], CORNERS[(idx+1)%4]
                target = (1-frac)*p0 + frac*p1
                q = ik_dls(model, ik_data, ee_body, target, q_prev, tool_offset)
                joint_target[f] = q
                ref_path[f] = target
                q_prev = q
            else:
                a = smoothstep((tau - loop_duration) / T_RETURN)
                joint_target[f] = q_prev * (1-a) + ready_q * a

    return joint_target, ref_path, ee_body, tool_offset


def run_actuated(scene_path: Path, label: str, out_mp4: Path):
    print(f"\n=== {label}: {scene_path.name} ===")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    joint_target, ref_path, ee_body, tool_offset = precompute_trajectory(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.azimuth = 145.0; camera.elevation = -28.0
    camera.distance = 2.0; camera.lookat[:] = [0.4, 0.0, 0.55]

    dt = model.opt.timestep
    sim_steps_per_frame = max(1, int(round(1.0 / FPS / dt)))

    frames = []
    errs = []
    n_frames = len(joint_target)
    for f in range(n_frames):
        data.ctrl[:] = joint_target[f]
        for _ in range(sim_steps_per_frame):
            mujoco.mj_step(model, data)
        ee_now = data.xpos[ee_body] + data.xmat[ee_body].reshape(3,3) @ tool_offset
        if ref_path[f] is not None:
            errs.append(np.linalg.norm(ref_path[f] - ee_now))
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

    errs = np.array(errs) * 1000  # mm
    print(f"  tracking error along reference: mean={errs.mean():.2f} mm  "
          f"max={errs.max():.2f} mm  p95={np.percentile(errs, 95):.2f} mm  "
          f"({len(errs)} frames)")

    imageio.mimwrite(str(out_mp4), frames, fps=FPS, quality=7, macro_block_size=1)
    print(f"  wrote {out_mp4} ({out_mp4.stat().st_size/1024:.1f} KiB)")

    return float(errs.mean()), float(errs.max()), float(np.percentile(errs, 95))


def main():
    legacy_out = HERE / "iiwa7" / "demo_square_actuated_legacy.mp4"
    tuned_out  = HERE / "iiwa7" / "demo_square_actuated_tuned.mp4"

    lm, lx, l95 = run_actuated(LEGACY_SCENE, "legacy (position, kp=400/200/100)", legacy_out)
    tm, tx, t95 = run_actuated(TUNED_SCENE,  "tuned (general, kp=2000 kd=200)",    tuned_out)

    with REPORT_PATH.open("w") as f:
        f.write("# iiwa7 MJCF Tuning Report\n\n")
        f.write("Cross-referenced against "
                "[mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) "
                "iiwa14.xml. Same reference path "
                "(30 cm square at Z=0.55 m, 2 loops) driven through actuators "
                "in closed-loop sim on both models.\n\n")
        f.write("## Actuator-in-loop EE tracking error\n\n")
        f.write("| model                            | mean (mm) | p95 (mm) | max (mm) |\n")
        f.write("|---                                |---        |---       |---       |\n")
        f.write(f"| legacy (position, kp=400/200/100) | {lm:>7.2f} | {l95:>6.2f} | {lx:>6.2f} |\n")
        f.write(f"| **tuned (general, kp=2000 kd=200)**| **{tm:>5.2f}** | **{t95:>4.2f}** | **{tx:>4.2f}** |\n")
        f.write(f"\nimprovement: mean {lm/tm:.1f}x, max {lx/tx:.1f}x\n\n")
        f.write("## Parameter diff adopted from menagerie\n\n")
        f.write("| parameter | legacy iiwa7.xml | tuned iiwa7_tuned.xml | menagerie iiwa14 | rationale |\n")
        f.write("|---|---|---|---|---|\n")
        f.write("| actuator type | `<position>` | `<general>` gaintype=fixed biastype=affine | `<general>` same | full PD with kd term |\n")
        f.write("| gainprm / kp | 400/200/100 (tiered) | 2000 uniform | 2000 uniform | stiffer gravity rejection |\n")
        f.write("| biasprm kd | implicit 0 | -200 | -200 | velocity damping via actuator |\n")
        f.write("| forcerange | none | J1/2:±176, J3-5:±110, J6/7:±40 Nm | J1-5:±various, J6/7:±40 | iiwa7 datasheet torque limits |\n")
        f.write("| joint damping | 0.5 (URDF) | 0.5 (kept) | 0 | let passive sim still settle |\n")
        f.write("| armature | none | 0.1 | none | reflect motor rotor inertia, stabilise at high kp |\n")
        f.write("| contact exclude | none | 7 pairs | 7 pairs | avoid false self-contact |\n")
        f.write("| attachment_site | none | on link7 at (0,0,0.05) | same | standard tool mount point |\n")
        f.write("| default classes | none | iiwa/joint1..7 classes | same | dedupe joint/actuator config |\n")
        f.write("| inertial values | URDF iiwa7 (masses 3.45/3.48/4.06...) | URDF iiwa7 (unchanged) | menagerie iiwa14 (5.76/6.35/3.5...) | iiwa7 ≠ iiwa14 — do not copy |\n")

    print(f"\nwrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
