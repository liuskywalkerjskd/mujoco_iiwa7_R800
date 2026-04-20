"""Unified EE-pose controller for iiwa7 in MuJoCo.

One class, five control modes — all demos in the `examples/` suite drive
the arm through this single API so users can pick the behaviour that
matches their application:

    mode                    pipeline                                               tracking
    --------------------    ---------------------------------------------------    ---------
    "kinematic"             qpos := q_target; no dynamics, no actuator             ~0 mm (ideal)
    "pd_only"               data.ctrl := q_target (actuator PD only)               ~90 mm on square
    "gravity_ff"            PD + qfrc_applied = gravity(q_target)                  ~11 mm
    "full_id_ff_ref"        PD + qfrc_applied = mj_inverse(q_d, qd_d, qdd_d)       ~9 mm
    "full_id_ff_current"    PD + mj_inverse(q_actual, qd_actual, qdd_cmd) with     ~5 mm
     (default, best)         qdd_cmd = qdd_d + Kp(q_d-q) + Kd(qd_d-qd)

Input is always a joint target. Two ways to set it:
    - set_ee_target(pos, quat=None)  : solve damped-least-squares IK
    - set_joint_target(q)            : bypass IK entirely

Output: `update(model, data)` writes `data.ctrl` and/or
`data.qfrc_applied`. The caller then invokes `mujoco.mj_step(model, data)`
(or `mujoco.mj_forward` for the kinematic mode).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco


MODES = (
    "kinematic",
    "pd_only",
    "gravity_ff",
    "full_id_ff_ref",
    "full_id_ff_current",
)


class IiwaEEController:
    """Pose-in / control-signals-out controller for iiwa7 in MuJoCo."""

    DEFAULT_TOOL_OFFSET = np.array([0.0, 0.0, 0.05])  # attachment_site on link7

    # The arm has 7 revolute joints; any DoFs beyond that (e.g. a 2F-85
    # gripper) must be excluded from IK, PD tracking and feedforward so the
    # gripper's own actuator channel stays independent.
    ARM_NV = 7
    ARM_NU = 7

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mode: str = "full_id_ff_current",
        ee_body_name: str = "iiwa_link_7",
        base_body_name: str = "iiwa_link_0",
        tool_offset: np.ndarray = DEFAULT_TOOL_OFFSET,
        kp_tsk: float = 400.0,
        kd_tsk: float = 40.0,
        ik_pos_tol: float = 5e-4,
        ik_ori_tol: float = 5e-3,
        ik_damping: float = 0.05,
        ik_max_iter: int = 200,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.model = model
        self.ee_body_id = model.body(ee_body_name).id
        self.base_body_id = model.body(base_body_name).id
        self.tool_offset = np.asarray(tool_offset, dtype=float).copy()

        self.kp_tsk = float(kp_tsk)
        self.kd_tsk = float(kd_tsk)

        self.ik_pos_tol = float(ik_pos_tol)
        self.ik_ori_tol = float(ik_ori_tol)
        self.ik_damping = float(ik_damping)
        self.ik_max_iter = int(ik_max_iter)

        # scratch MjData for IK and inverse-dynamics (never disturbs live data)
        self._ik_data = mujoco.MjData(model)
        self._id_data = mujoco.MjData(model)

        # Cache base-frame pose in the world (assumed static after init).
        # Ensures set_ee_pose works whether the base is at the origin or
        # mounted somewhere else in the scene.
        mujoco.mj_forward(model, data)
        self._base_pos_world = data.xpos[self.base_body_id].copy()
        self._base_quat_world = data.xquat[self.base_body_id].copy()

        nq = model.nq
        self._q_target = data.qpos[:nq].copy()
        # history for finite-difference derivatives
        self._q_hist = np.tile(self._q_target, (3, 1))

    # -------------------------------------------------------------- API

    @property
    def q_target(self) -> np.ndarray:
        return self._q_target.copy()

    def reset(self, data: mujoco.MjData) -> None:
        """Re-sync internal state to current `data.qpos`. Call after
        `mj_resetData` / keyframe reset / any discontinuity."""
        q = data.qpos[: self.model.nq].copy()
        self._q_target = q
        self._q_hist = np.tile(q, (3, 1))

    def set_ee_target(
        self,
        pos: np.ndarray,
        quat: Optional[np.ndarray] = None,
    ) -> float:
        """Solve IK and update the internal joint target.

        Args:
            pos:  desired EE position in WORLD frame, shape (3,).
            quat: optional desired EE orientation in MuJoCo convention
                  (w, x, y, z), shape (4,). If None, only position is
                  constrained (3-DOF IK).

        Returns:
            residual: final pose error norm at IK convergence.
        """
        pos = np.asarray(pos, dtype=float).reshape(3)
        if quat is None:
            return self._ik_pos_only(pos)
        quat = np.asarray(quat, dtype=float).reshape(4)
        return self._ik_6dof(pos, quat)

    def set_ee_pose(self, pose7: np.ndarray) -> float:
        """High-level pose API matching the common robotics convention:

            pose7 = [x, y, z, qx, qy, qz, qw]

        where:
          - (x, y, z) is the end-effector origin expressed in the BASE
            frame (iiwa_link_0) in meters,
          - (qx, qy, qz, qw) is the end-effector orientation as a
            quaternion in xyzw order (ROS / Eigen / scipy convention).

        Internally the pose is transformed into the world frame (using
        the base body's cached world pose) and the quaternion is
        re-ordered to MuJoCo's wxyz convention before being handed to the
        IK solver.

        Returns:
            residual: final pose error norm at IK convergence.
        """
        pose7 = np.asarray(pose7, dtype=float).reshape(7)
        pos_base = pose7[:3]
        qx, qy, qz, qw = pose7[3:]
        quat_base_wxyz = np.array([qw, qx, qy, qz])

        # World = base * local
        pos_world = np.empty(3)
        mujoco.mju_rotVecQuat(pos_world, pos_base, self._base_quat_world)
        pos_world += self._base_pos_world

        quat_world_wxyz = np.empty(4)
        mujoco.mju_mulQuat(quat_world_wxyz, self._base_quat_world, quat_base_wxyz)

        return self.set_ee_target(pos=pos_world, quat=quat_world_wxyz)

    def set_joint_target(self, q: np.ndarray) -> None:
        """Directly set the joint-space target (skip IK).

        Accepts either a full nq vector or a shorter arm-only vector of
        length ARM_NV=7. In the latter case, non-arm entries (gripper,
        freejoint, etc.) of the internal target are left untouched.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size == self.model.nq:
            self._q_target = q.copy()
        elif q.size == self.ARM_NV:
            self._q_target = self._q_target.copy()
            self._q_target[: self.ARM_NV] = q
        else:
            raise ValueError(
                f"set_joint_target expects length {self.ARM_NV} or {self.model.nq}, got {q.size}"
            )

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Compute and write control signals for one sim step.

        After this call, invoke `mj_step(model, data)` for dynamic modes,
        or `mj_forward(model, data)` for `mode='kinematic'`.
        """
        # Roll the target history window
        self._q_hist[0] = self._q_hist[1]
        self._q_hist[1] = self._q_hist[2]
        self._q_hist[2] = self._q_target

        if self.mode == "kinematic":
            # Drive the arm only; leave the gripper (and any other extra
            # DoFs beyond the first 7) untouched so their own actuators /
            # freejoint dynamics keep owning those indices.
            data.qpos[: self.ARM_NV] = self._q_target[: self.ARM_NV]
            data.qvel[: self.ARM_NV] = 0.0
            return

        if self.mode == "pd_only":
            data.ctrl[: self.ARM_NU] = self._q_target[: self.ARM_NU]
            # Do NOT touch data.ctrl[ARM_NU:] — that's the gripper channel.
            data.qfrc_applied[: self.ARM_NV] = 0.0
            return

        # All other modes are PD + some form of inverse-dynamics FF.
        dt = max(model.opt.timestep, 1e-6)
        qd_target = (self._q_hist[2] - self._q_hist[1]) / dt
        qdd_target = (self._q_hist[2] - 2 * self._q_hist[1] + self._q_hist[0]) / (dt * dt)

        if self.mode == "gravity_ff":
            # qfrc_bias at (q_d, 0, 0) reduces to pure gravity torque
            self._id_data.qpos[:] = self._q_target
            self._id_data.qvel[:] = 0.0
            mujoco.mj_forward(model, self._id_data)
            tau_ff = self._id_data.qfrc_bias.copy()

        elif self.mode == "full_id_ff_ref":
            # mj_inverse at reference state (q_d, qd_d, qdd_d)
            self._id_data.qpos[:] = self._q_target
            # qd_target / qdd_target may only cover the first ARM_NV entries
            # when the caller set a joint target shorter than model.nv; pad.
            self._id_data.qvel[:] = 0.0
            self._id_data.qvel[: self.ARM_NV] = qd_target[: self.ARM_NV]
            self._id_data.qacc[:] = 0.0
            self._id_data.qacc[: self.ARM_NV] = qdd_target[: self.ARM_NV]
            mujoco.mj_inverse(model, self._id_data)
            tau_ff = self._id_data.qfrc_inverse.copy()

        elif self.mode == "full_id_ff_current":
            # Task-space PD folded into commanded acceleration, then
            # mj_inverse at CURRENT state (q, qd, qdd_cmd). Restricted to
            # the arm DoFs so extras (e.g. gripper fingers, freejoint cube)
            # are NOT servoed by this loop.
            n = self.ARM_NV
            qdd_cmd_full = np.zeros(model.nv)
            qdd_cmd_full[:n] = (
                qdd_target[:n]
                + self.kp_tsk * (self._q_target[:n] - data.qpos[:n])
                + self.kd_tsk * (qd_target[:n] - data.qvel[:n])
            )
            self._id_data.qpos[:] = data.qpos
            self._id_data.qvel[:] = data.qvel
            self._id_data.qacc[:] = qdd_cmd_full
            mujoco.mj_inverse(model, self._id_data)
            tau_ff = self._id_data.qfrc_inverse.copy()
        else:
            raise RuntimeError(f"unreachable mode {self.mode!r}")

        # Arm-only PD actuator targets; leave ctrl[ARM_NU:] for the gripper.
        data.ctrl[: self.ARM_NU] = self._q_target[: self.ARM_NU]
        # Arm-only feedforward torques; gripper / freejoint DoFs get zero so
        # their native dynamics (or the user's gripper setpoint) dominate.
        data.qfrc_applied[: model.nv] = 0.0
        data.qfrc_applied[: self.ARM_NV] = tau_ff[: self.ARM_NV]

    # -------------------------------------------------------------- IK

    def _current_ee_pose(self, mjdata: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        xpos = mjdata.xpos[self.ee_body_id] + mjdata.xmat[self.ee_body_id].reshape(3, 3) @ self.tool_offset
        xquat = mjdata.xquat[self.ee_body_id].copy()
        return xpos, xquat

    def _clamp_to_joint_limits(self, qpos: np.ndarray) -> None:
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j]:
                lo, hi = self.model.jnt_range[j]
                qpos[j] = np.clip(qpos[j], lo, hi)

    def _ik_pos_only(self, target_pos: np.ndarray) -> float:
        d = self._ik_data
        d.qpos[: self.model.nq] = self._q_target
        nv = self.model.nv
        n_arm = self.ARM_NV
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        step = 0.5
        for _ in range(self.ik_max_iter):
            mujoco.mj_forward(self.model, d)
            xpos, _ = self._current_ee_pose(d)
            err = target_pos - xpos
            norm = float(np.linalg.norm(err))
            if norm < self.ik_pos_tol:
                self._q_target = d.qpos[: self.model.nq].copy()
                return norm
            mujoco.mj_jacBody(self.model, d, jp, jr, self.ee_body_id)
            # Only move the 7 arm DoFs; the gripper / freejoint columns are
            # zeroed so they don't contribute to the solution.
            jp_a = jp[:, :n_arm]
            JJT = jp_a @ jp_a.T + self.ik_damping ** 2 * np.eye(3)
            dq_arm = jp_a.T @ np.linalg.solve(JJT, err)
            dq = np.zeros(nv)
            dq[:n_arm] = dq_arm
            mujoco.mj_integratePos(self.model, d.qpos, dq, step)
            self._clamp_to_joint_limits(d.qpos)
        self._q_target = d.qpos[: self.model.nq].copy()
        return float(np.linalg.norm(err))

    def _ik_6dof(self, target_pos: np.ndarray, target_quat: np.ndarray) -> float:
        d = self._ik_data
        d.qpos[: self.model.nq] = self._q_target
        nv = self.model.nv
        n_arm = self.ARM_NV
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        step = 0.5
        damping_pos = self.ik_damping
        damping_ori = 0.1
        lam2 = np.diag([damping_pos ** 2] * 3 + [damping_ori ** 2] * 3)
        for _ in range(self.ik_max_iter):
            mujoco.mj_forward(self.model, d)
            xpos, xquat = self._current_ee_pose(d)
            pos_err = target_pos - xpos

            q_inv = np.zeros(4)
            mujoco.mju_negQuat(q_inv, xquat)
            q_diff = np.zeros(4)
            mujoco.mju_mulQuat(q_diff, target_quat, q_inv)
            sgn = 1.0 if q_diff[0] >= 0 else -1.0
            ori_err = 2.0 * sgn * q_diff[1:4]

            if (
                np.linalg.norm(pos_err) < self.ik_pos_tol
                and np.linalg.norm(ori_err) < self.ik_ori_tol
            ):
                self._q_target = d.qpos[: self.model.nq].copy()
                return float(max(np.linalg.norm(pos_err), np.linalg.norm(ori_err)))

            mujoco.mj_jacBody(self.model, d, jp, jr, self.ee_body_id)
            J = np.vstack([jp, jr])[:, :n_arm]  # arm-only Jacobian
            err = np.concatenate([pos_err, ori_err])
            dq_arm = J.T @ np.linalg.solve(J @ J.T + lam2, err)
            dq = np.zeros(nv)
            dq[:n_arm] = dq_arm
            mujoco.mj_integratePos(self.model, d.qpos, dq, step)
            self._clamp_to_joint_limits(d.qpos)
        self._q_target = d.qpos[: self.model.nq].copy()
        return float(max(np.linalg.norm(pos_err), np.linalg.norm(ori_err)))
