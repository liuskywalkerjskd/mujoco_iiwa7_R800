#!/usr/bin/env python3
"""6-DoF SpaceMouse teleoperation of the iiwa7 + Robotiq 2F-85 simulation.

Design lineage: https://github.com/chad-yao/KUKA-Controller  (specifically
``demo_real_robot.py`` and ``common/spacemouse_shared_memory.py``).

The pipeline mirrors that reference:

    1.  A device thread polls the SpaceMouse at ~200 Hz and exposes the
        latest axis / button state through a shared ``TeleopState``.
    2.  The control loop runs at ``CTRL_HZ`` (default 30 Hz).  Each tick:
          - read the latest 6-axis state normalised to [-1, 1],
          - apply a dead-zone,
          - map spnav axes -> robot base frame via ``TX_ZUP_SPNAV``,
          - scale by ``dt * max_pos_speed`` / ``dt * max_rot_speed``,
          - integrate (translation += dpos, rotation = drot * R)
            the target pose in base frame,
          - send ``pose7 = [x, y, z, qx, qy, qz, qw]`` through
            ``IiwaEEController.set_ee_pose`` (damped-least-squares IK +
            full inverse-dynamics feed-forward).
          - button 0 toggles the gripper (open / closed).
          - button 1 resets target pose to home.

Two entry modes:

    --live      : opens ``mujoco.viewer.launch_passive``; requires a real
                  3Dconnexion SpaceMouse reachable via pyspacemouse (hidapi).
    --record    : feeds a scripted axis stream instead of the device and
                  writes an MP4 to ``media/videos/spacemouse_teleop.mp4`` —
                  lets you verify the pipeline without hardware.

Install (once):
    pip install pyspacemouse      # pulls in easyhid / hidapi

On Linux you may need:
    sudo usermod -a -G plugdev $USER
    sudo cp <repo>/pyspacemouse.rules /etc/udev/rules.d/   # optional
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iiwa7_controller import IiwaEEController  # noqa: E402


# ---------------------------------------------------------------- scene
SCENE = REPO / "examples" / "scenes" / "iiwa7_with_gripper_scene.xml"

# ---------------------------------------------------------------- teleop params
CTRL_HZ = 30.0
DEVICE_HZ = 200.0

# Copied verbatim from KUKA-Controller so the operator feel matches.
# Row i of TX maps spnav axis i onto a robot-base axis.
#   spnav +x (right)   -> base -z (up)         [yes, sign -1]
#   spnav +y (toward)  -> base +x (forward)
#   spnav +z (up)      -> base +y (left)
TX_ZUP_SPNAV = np.array(
    [[0.0, 0.0, -1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]]
)
# Axis signs on the rotation channel (reference uses [-1, +1, -1]).
ROT_SIGN = np.array([-1.0, 1.0, -1.0])

MAX_POS_SPEED = 0.25   # m/s at full-stick translation
MAX_ROT_SPEED = 0.9    # rad/s at full-stick rotation
DEADZONE_POS = 0.05
DEADZONE_ROT = 0.10

# Gripper ctrl[7] setpoints (rad; 0 = open 85 mm, 0.55 ~ 40 mm grip).
G_OPEN = 0.0
G_CLOSE = 0.55


# ---------------------------------------------------------------- device layer


@dataclass
class TeleopState:
    """Latest device sample. Writer: device thread. Reader: control loop."""
    axes: np.ndarray = field(default_factory=lambda: np.zeros(6))  # tx ty tz rx ry rz
    buttons: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    t_stamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, axes: np.ndarray, buttons: np.ndarray) -> None:
        with self.lock:
            self.axes = np.asarray(axes, dtype=float).copy()
            self.buttons = np.asarray(buttons, dtype=bool).copy()
            self.t_stamp = time.time()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, float]:
        with self.lock:
            return self.axes.copy(), self.buttons.copy(), self.t_stamp


class PySpaceMouseSource:
    """Real 3Dconnexion device via pyspacemouse (hidapi backend)."""

    def __init__(self, state: TeleopState, rate_hz: float = DEVICE_HZ) -> None:
        import pyspacemouse  # local import so --record works without hidapi
        if not pyspacemouse.open():
            raise RuntimeError(
                "pyspacemouse.open() failed. Check that a 3Dconnexion device "
                "is plugged in and that the current user can read it (udev)."
            )
        self.pyspacemouse = pyspacemouse
        self.state = state
        self.dt = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="spacemouse", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.pyspacemouse.close()
        except Exception:
            pass

    def _loop(self) -> None:
        # pyspacemouse.read() returns a namespace with .x .y .z .roll .pitch .yaw
        # (all in [-1, 1]) and .buttons (list of 0/1). Convention matches spnav.
        while not self._stop.is_set():
            s = self.pyspacemouse.read()
            axes = np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw], dtype=float)
            btns = np.array(s.buttons[:2] if len(s.buttons) >= 2 else [0, 0], dtype=bool)
            self.state.update(axes, btns)
            time.sleep(self.dt)


class ScriptedSource:
    """Replay a programmed 6-axis stream — for headless verification / video.

    The script is a list of ``(duration_s, axes6, buttons2)`` tuples.  Axis
    values linearly interpolate between entries; buttons flip at the segment
    boundary.  Unlike the device thread this source is sampled deterministically
    by ``sample_at(t_sim)`` so the headless recorder stays in sync with the
    MuJoCo clock (wall-clock polling would race with the fast sim loop).
    """

    def __init__(self, script: list) -> None:
        cum: list[tuple[float, np.ndarray, np.ndarray]] = []
        t = 0.0
        for dur, axes, btns in script:
            t += float(dur)
            cum.append((t, np.asarray(axes, float), np.asarray(btns, bool)))
        self._cum = cum
        self._total = cum[-1][0] if cum else 0.0

    @property
    def total_s(self) -> float:
        return self._total

    def sample_at(self, t_sim: float) -> tuple[np.ndarray, np.ndarray]:
        if t_sim <= 0.0 or not self._cum:
            return np.zeros(6), np.zeros(2, dtype=bool)
        if t_sim >= self._total:
            return np.zeros(6), self._cum[-1][2].copy()
        prev_t, prev_axes = 0.0, np.zeros(6)
        for end_t, axes, btn in self._cum:
            if t_sim <= end_t:
                frac = (t_sim - prev_t) / max(end_t - prev_t, 1e-6)
                cur = prev_axes + (axes - prev_axes) * frac
                return cur, btn.copy()
            prev_t, prev_axes = end_t, axes
        return np.zeros(6), np.zeros(2, dtype=bool)


# ---------------------------------------------------------------- math


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return quaternion (wxyz) for rotation of `angle` rad about `axis`."""
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = axis / n
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), ax[0] * s, ax[1] * s, ax[2] * s])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (wxyz) q1 * q2 (left-multiplies q2 by q1)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def deadzone(v: np.ndarray, dz_pos: float, dz_rot: float) -> np.ndarray:
    out = v.copy()
    for i in range(3):
        out[i] = 0.0 if abs(v[i]) < dz_pos else (v[i] - np.sign(v[i]) * dz_pos)
    for i in range(3, 6):
        out[i] = 0.0 if abs(v[i]) < dz_rot else (v[i] - np.sign(v[i]) * dz_rot)
    return out


# ---------------------------------------------------------------- teleop kernel


class TeleopController:
    """Holds the integrated EE target in BASE frame + gripper setpoint."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 ctrl: IiwaEEController) -> None:
        self.model = model
        self.data = data
        self.ctrl = ctrl
        # Snapshot base-frame pose at the current configuration. We take the
        # pose of link_7+tool_offset expressed relative to iiwa_link_0.
        mujoco.mj_forward(model, data)
        ee_body = ctrl.ee_body_id
        base_body = ctrl.base_body_id
        Rw = data.xmat[ee_body].reshape(3, 3)
        pos_w = data.xpos[ee_body] + Rw @ ctrl.tool_offset
        base_pos_w = data.xpos[base_body].copy()
        base_quat_w = data.xquat[base_body].copy()
        # pos in base frame = R_base^T (pos_w - base_pos_w)
        inv_qw = np.empty(4)
        mujoco.mju_negQuat(inv_qw, base_quat_w)
        rel = pos_w - base_pos_w
        pos_base = np.empty(3)
        mujoco.mju_rotVecQuat(pos_base, rel, inv_qw)
        # quat in base frame = inv(base_quat) * world_quat
        world_quat_wxyz = data.xquat[ee_body].copy()
        quat_base = np.empty(4)
        mujoco.mju_mulQuat(quat_base, inv_qw, world_quat_wxyz)

        self._home_pos = pos_base.copy()
        self._home_quat_wxyz = quat_base.copy()
        self.target_pos = pos_base.copy()
        self.target_quat_wxyz = quat_base.copy()
        self.gripper = G_OPEN

        # Debouncing for button rising-edges.
        self._btn_prev = np.zeros(2, dtype=bool)

    def reset_to_home(self) -> None:
        self.target_pos = self._home_pos.copy()
        self.target_quat_wxyz = self._home_quat_wxyz.copy()

    def step(self, axes: np.ndarray, buttons: np.ndarray, dt: float) -> np.ndarray:
        # --- axes -> base-frame deltas (KUKA-Controller convention) ---
        a = deadzone(axes, DEADZONE_POS, DEADZONE_ROT)
        a_pos = TX_ZUP_SPNAV @ a[:3]
        a_rot = (TX_ZUP_SPNAV @ a[3:]) * ROT_SIGN
        dpos = a_pos * (MAX_POS_SPEED * dt)
        drot = a_rot * (MAX_ROT_SPEED * dt)

        # --- integrate in base frame ---
        self.target_pos = self.target_pos + dpos
        # Left-multiply in base frame: R_new = R_delta * R_target
        rot_angle = float(np.linalg.norm(drot))
        if rot_angle > 1e-9:
            q_delta = quat_from_axis_angle(drot, rot_angle)
            self.target_quat_wxyz = quat_mul(q_delta, self.target_quat_wxyz)
            # renormalize
            self.target_quat_wxyz /= np.linalg.norm(self.target_quat_wxyz)

        # --- buttons (rising-edge) ---
        rising = buttons & ~self._btn_prev
        if rising[0]:
            self.gripper = G_CLOSE if self.gripper < 0.1 else G_OPEN
        if rising[1]:
            self.reset_to_home()
        self._btn_prev = buttons.copy()

        # --- dispatch to the arm (base frame, xyzw quat ordering) ---
        qw, qx, qy, qz = self.target_quat_wxyz
        pose7 = np.array([
            self.target_pos[0], self.target_pos[1], self.target_pos[2],
            qx, qy, qz, qw,
        ])
        self.ctrl.set_ee_pose(pose7)
        return pose7


# ---------------------------------------------------------------- runtime


def _scripted_script() -> list:
    """A ~18 s canned motion that exercises every axis + the gripper."""
    # axes order: [x, y, z, rx, ry, rz], values in [-1, 1]
    zero = [0.0] * 6
    return [
        (1.0, zero,                    [0, 0]),   # settle
        (2.0, [0.0,  0.7, 0.0,  0, 0, 0], [0, 0]),  # push forward (spnav +y -> base +x)
        (0.5, zero,                    [0, 0]),
        (1.5, [0.0,  0.0, 0.7,  0, 0, 0], [0, 0]),  # lift (+Z base)
        (0.5, zero,                    [0, 0]),
        (1.5, [0.7,  0.0, 0.0,  0, 0, 0], [0, 0]),  # lateral (spnav +x -> base -z down)
        (0.5, zero,                    [0, 0]),
        (1.5, [0.0,  0.0, 0.0,  0.6, 0, 0], [0, 0]),  # roll
        (0.5, zero,                    [0, 0]),
        (1.5, [0.0,  0.0, 0.0,  0.0, 0.6, 0.0], [0, 0]),  # pitch
        (0.5, zero,                    [0, 0]),
        (1.5, [0.0,  0.0, 0.0,  0.0, 0.0, 0.6], [0, 0]),  # yaw
        (0.5, zero,                    [0, 0]),
        (0.1, zero,                    [1, 0]),   # close gripper (rising edge)
        (1.5, zero,                    [0, 0]),
        (0.1, zero,                    [0, 1]),   # reset to home
        (1.5, zero,                    [0, 0]),
    ]


def run_live(model, data, ctrl: IiwaEEController, teleop: TeleopController,
             state: TeleopState, source) -> int:
    """Interactive passive viewer driven by the SpaceMouse."""
    import mujoco.viewer  # only when truly live

    source.start()
    dt_ctrl = 1.0 / CTRL_HZ
    gripper_act = model.actuator("2f85_fingers_actuator").id
    print(f"[live] {SCENE.name}  ctrl_hz={CTRL_HZ}  max_pos={MAX_POS_SPEED}m/s  "
          f"max_rot={MAX_ROT_SPEED}rad/s")
    print("[live] button 0 = toggle gripper ; button 1 = reset to home")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            ctrl.reset(data)
            teleop.reset_to_home()
            t_next = time.time()
            while viewer.is_running():
                axes, buttons, _ = state.snapshot()
                teleop.step(axes, buttons, dt_ctrl)
                # Run one control tick worth of sim steps.
                sim_steps = max(1, int(round(dt_ctrl / model.opt.timestep)))
                for _ in range(sim_steps):
                    ctrl.update(model, data)
                    data.ctrl[gripper_act] = teleop.gripper
                    data.qfrc_applied[7:] = 0.0  # keep gripper / freejoint free
                    mujoco.mj_step(model, data)
                viewer.sync()
                t_next += dt_ctrl
                time.sleep(max(0.0, t_next - time.time()))
    finally:
        source.stop()
    return 0


def run_record(model, data, ctrl: IiwaEEController, teleop: TeleopController,
               source: ScriptedSource,
               out_mp4: Path, duration_s: float = 18.0,
               width: int = 640, height: int = 480) -> int:
    """Headless scripted input -> MP4, so you can verify without hardware."""
    import imageio.v2 as imageio

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    ctrl.reset(data)
    teleop.reset_to_home()

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.azimuth = 140.0
    cam.elevation = -22.0
    cam.distance = 1.7
    cam.lookat[:] = [0.4, 0.0, 0.45]

    gripper_act = model.actuator("2f85_fingers_actuator").id
    fps = 30
    n_frames = int(duration_s * fps)
    dt_ctrl = 1.0 / CTRL_HZ  # 1/30 -- equal to video frame rate
    sim_steps_per_frame = max(1, int(round(dt_ctrl / model.opt.timestep)))

    frames = []
    print(f"[record] scripted teleop -> {out_mp4.name}  "
          f"({n_frames} frames, {sim_steps_per_frame} sim steps / frame)")
    for f in range(n_frames):
        t_sim = f / fps
        axes, buttons = source.sample_at(t_sim)
        teleop.step(axes, buttons, dt_ctrl)
        for _ in range(sim_steps_per_frame):
            ctrl.update(model, data)
            data.ctrl[gripper_act] = teleop.gripper
            data.qfrc_applied[7:] = 0.0
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
        if (f + 1) % 30 == 0:
            ee_body = ctrl.ee_body_id
            R = data.xmat[ee_body].reshape(3, 3)
            ee = data.xpos[ee_body] + R @ ctrl.tool_offset
            active = " ".join(f"{x:+.2f}" for x in axes)
            print(f"  t={(f+1)/fps:4.1f}s  axes=[{active}]  "
                  f"TCP_world={ee.round(3).tolist()}  "
                  f"gripper_ctrl={teleop.gripper:.2f}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_mp4), frames, fps=fps, quality=7, macro_block_size=1)
    print(f"wrote {out_mp4}  ({out_mp4.stat().st_size/1024:.1f} KiB)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true",
                   help="interactive passive viewer + real 3Dconnexion SpaceMouse")
    p.add_argument("--record", action="store_true",
                   help="headless scripted input -> MP4 (default if no flag)")
    p.add_argument("--scene", type=Path, default=SCENE)
    p.add_argument("--out", type=Path,
                   default=REPO / "media" / "videos" / "spacemouse_teleop.mp4")
    p.add_argument("--duration", type=float, default=18.0,
                   help="(record mode) seconds of video to render")
    args = p.parse_args()

    if not args.live and not args.record:
        args.record = True  # default safe path

    print(f"loading {args.scene}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    ctrl = IiwaEEController(model, data, mode="full_id_ff_current")
    teleop = TeleopController(model, data, ctrl)
    state = TeleopState()

    if args.live:
        source = PySpaceMouseSource(state)
        return run_live(model, data, ctrl, teleop, state, source)
    else:
        source = ScriptedSource(_scripted_script())
        return run_record(model, data, ctrl, teleop, source,
                          out_mp4=args.out, duration_s=args.duration)


if __name__ == "__main__":
    sys.exit(main())
