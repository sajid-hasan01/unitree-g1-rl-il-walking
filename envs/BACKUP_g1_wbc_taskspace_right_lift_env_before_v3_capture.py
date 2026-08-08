from __future__ import annotations

from typing import Dict, List

import mujoco
import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


class G1WBCTaskspaceRightLiftEnv(G1TaskspaceRightLiftEnv):
    """
    WBC-lite multi-task right-foot lift.

    This is not a full QP solver yet. It is a prioritized whole-body-style
    controller built on top of the existing 15-DOF G1 position-control env.

    Task priority:
    1. torso/base upright + pelvis height protection
    2. left support-foot lock
    3. short right-foot lift and early landing
    4. soft root x/y velocity damping

    The main change from the dual-task env is the short swing window plus
    torso angular stabilization. The dual-task env proved that right-foot
    air-time and left support lock are possible, but it failed at up_z_low.
    """

    def __init__(
        self,
        support_lock_weight: float = 0.70,
        support_xy_weight: float = 0.22,
        support_z_weight: float = 1.20,
        support_ik_gain: float = 0.50,
        support_ik_damping: float = 0.065,
        support_ik_max_delta: float = 0.090,
        torso_pitch_gain: float = 0.20,
        torso_roll_gain: float = 0.08,
        angvel_pitch_gain: float = 0.115,
        angvel_roll_gain: float = 0.045,
        height_gain: float = 0.18,
        height_target: float = 0.790,
        **kwargs,
    ):
        # Attributes must exist before parent __init__, because parent __init__
        # calls self._get_obs() -> self._get_info().
        self.support_lock_weight = float(support_lock_weight)
        self.support_xy_weight = float(support_xy_weight)
        self.support_z_weight = float(support_z_weight)
        self.support_ik_gain = float(support_ik_gain)
        self.support_ik_damping = float(support_ik_damping)
        self.support_ik_max_delta = float(support_ik_max_delta)
        self.torso_pitch_gain = float(torso_pitch_gain)
        self.torso_roll_gain = float(torso_roll_gain)
        self.angvel_pitch_gain = float(angvel_pitch_gain)
        self.angvel_roll_gain = float(angvel_roll_gain)
        self.height_gain = float(height_gain)
        self.height_target = float(height_target)

        # Conservative WBC-lite defaults. Short swing is intentional: previous
        # controllers collapsed when swing stayed high around phi ~= 0.58.
        kwargs.setdefault("max_steps", 520)
        kwargs.setdefault("cycle_duration", 5.2)
        kwargs.setdefault("shift_start", 0.08)
        kwargs.setdefault("swing_start", 0.42)
        kwargs.setdefault("swing_end", 0.52)
        kwargs.setdefault("land_end", 0.60)
        kwargs.setdefault("target_clearance", 0.014)
        kwargs.setdefault("target_lateral_shift", 0.012)
        kwargs.setdefault("ik_gain", 0.70)
        kwargs.setdefault("ik_damping", 0.060)
        kwargs.setdefault("ik_max_delta", 0.085)
        kwargs.setdefault("xy_hold_weight", 0.180)
        kwargs.setdefault("z_lift_weight", 0.72)
        kwargs.setdefault("x_hard_limit", 0.36)
        kwargs.setdefault("y_hard_limit", 0.30)
        kwargs.setdefault("x_velocity_hard_limit", 1.35)
        kwargs.setdefault("y_velocity_hard_limit", 1.35)
        kwargs.setdefault("min_up_z", 0.70)
        super().__init__(**kwargs)

    def _root_orientation_proxies(self):
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        # MuJoCo gives row-major 3x3. Local z-axis in world coords is column 2.
        up_x = float(mat[2])
        up_y = float(mat[5])
        up_z = float(mat[8])
        return up_x, up_y, up_z

    def _support_ik_delta(self, site_id: int, dof_indices: List[int], task_error: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = jacp[:, dof_indices]
        damping = self.support_ik_damping
        A = J @ J.T + (damping * damping) * np.eye(3)
        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error
        delta = self.support_ik_gain * delta
        return np.clip(delta, -self.support_ik_max_delta, self.support_ik_max_delta)

    def _target_foot_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        """
        WBC v3 capture-step foot placement.

        Negative x is the observed backward-fall direction in our logs.
        When root x/xv becomes unsafe during swing, the right foot is placed
        slightly backward/under the falling body and forced down for touchdown.
        """
        target = super()._target_foot_position(action, info)

        phase = float(info["phase"])
        sw = float(info["swing_env"])
        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        up_z = float(info["up_z"])

        if phase > 0.46:
            capture = float(np.clip(
                max(0.0, -xv - 0.22) * 1.25
                + max(0.0, -x - 0.055) * 2.20
                + max(0.0, 0.94 - up_z) * 2.50,
                0.0,
                1.0,
            ))

            # Capture placement: move right-foot target in the falling direction.
            # Observed fall direction is negative x.
            target[0] += -0.070 * capture

            # Force touchdown by reducing vertical target once capture activates.
            target[2] -= 0.025 * capture
            target[2] = max(float(self.right_foot_p0[2]), float(target[2]))

        return target

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # WBC v3: bake in the best tested residual direction:
        # fixed_action=-1,0,-1,1 gave the best upright/lateral result.
        action_for_parent = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        phase_for_bias = float(info["phase"])
        if phase_for_bias >= 0.38:
            action_for_parent[0] = np.clip(action_for_parent[0] - 0.85, -1.0, 1.0)  # reduce lift
            action_for_parent[2] = np.clip(action_for_parent[2] - 0.85, -1.0, 1.0)  # cancel bad sagittal bias
            action_for_parent[3] = np.clip(action_for_parent[3] + 0.75, -1.0, 1.0)  # lateral correction

        # Parent creates the right-foot task-space lift and basic root feedback.
        target = super()._target_joint_position(action_for_parent, info)

        sw = float(info["swing_env"])
        sh = float(info["shift_env"])
        x = float(info["x_position"])
        y = float(info["y_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        h = float(info["base_height"])

        up_x, up_y, up_z = self._root_orientation_proxies()
        # qvel[3:6] are root angular velocities in MuJoCo free-joint coordinates.
        av = np.asarray(self.data.qvel[3:6], dtype=np.float64)

        # WBC task 1: torso/base upright and pelvis height protection.
        # Keep this conservative. Prior over-aggressive sagittal correction made
        # x collapse worse. This mainly reduces angular velocity/tilt.
        pitch_task = np.clip(
            -self.torso_pitch_gain * up_y - self.angvel_pitch_gain * av[1] - 0.10 * xv - 0.035 * x,
            -0.105,
            0.105,
        )
        roll_task = np.clip(
            -self.torso_roll_gain * up_x - self.angvel_roll_gain * av[0] - 0.045 * yv - 0.035 * (y - float(info["target_y_offset"])),
            -0.070,
            0.070,
        )
        height_task = np.clip(self.height_gain * (self.height_target - h), -0.055, 0.075)

        # Apply posture tasks through waist and stance leg. Use small gains; this
        # is a stabilizer, not a full torque-level controller.
        target[14] += -0.68 * pitch_task             # waist pitch
        target[0] += +0.58 * pitch_task              # left hip pitch
        target[4] += -0.46 * pitch_task              # left ankle pitch

        target[13] += -0.55 * roll_task              # waist roll
        target[1] += +0.35 * roll_task               # left hip roll
        target[5] += -0.25 * roll_task               # left ankle roll

        # Pelvis height support. More knee bend can lower; ankle/hip support helps
        # keep the stance leg from collapsing while the right foot lifts.
        support_env = min(1.0, max(0.0, 0.30 * sh + 0.70 * sw))
        target[3] += +0.025 * support_env + 0.38 * height_task    # left knee
        target[0] += +0.16 * height_task                          # left hip pitch
        target[4] += -0.14 * height_task                          # left ankle pitch

        # V2 sagittal capture: after the swing starts, resist backward base
        # velocity before x reaches the hard limit. This is intentionally phase-
        # gated; applying it during quiet standing can create unnecessary motion.
        phase = float(info["phase"])
        if phase > 0.44:
            capture = float(np.clip(max(0.0, -xv - 0.18) * 0.75 + max(0.0, -x - 0.030) * 0.55, 0.0, 0.16))
            target[0] += +0.42 * capture      # left hip pitch
            target[4] += -0.34 * capture      # left ankle pitch
            target[14] += -0.24 * capture     # waist pitch
            target[9] += -0.08 * capture      # begin extending swing knee for landing
            target[10] += -0.06 * capture     # swing ankle down

        # WBC task 2: lock left support foot in task space.
        if support_env > 0.02:
            left_leg_idx = [0, 1, 2, 3, 4, 5]
            left_dofs = [self.qvel_adrs[i] for i in left_leg_idx]
            current_left = self.data.site_xpos[self.left_foot_site].copy()
            desired_left = self.left_foot_p0.copy()
            err = desired_left - current_left
            err[0] *= self.support_xy_weight
            err[1] *= self.support_xy_weight
            err[2] *= self.support_z_weight
            dq = self._support_ik_delta(self.left_foot_site, left_dofs, err)
            current_q = np.array([self.data.qpos[self.qpos_adrs[i]] for i in left_leg_idx], dtype=np.float64)
            ik_target = current_q + dq
            blend = self.support_lock_weight * support_env
            for local_i, joint_i in enumerate(left_leg_idx):
                target[joint_i] = (1.0 - blend) * target[joint_i] + blend * ik_target[local_i]

        # WBC task 3: capture touchdown.
        # Once backward root momentum starts, do not keep holding the foot up.
        # Extend right knee/ankle and place the foot down quickly.
        phase = float(info["phase"])
        if phase > 0.48:
            capture = float(np.clip(
                max(0.0, -xv - 0.22) * 1.35
                + max(0.0, -x - 0.055) * 2.40
                + max(0.0, 0.94 - up_z) * 2.60,
                0.0,
                1.0,
            ))

            if capture > 0.02:
                target[9] += -0.42 * capture       # right knee extends for touchdown
                target[10] += -0.30 * capture      # right ankle down
                target[6] += -0.075 * capture      # right hip moves foot toward capture placement

                # Support-leg push while the capture step comes down.
                target[0] += +0.26 * capture       # left hip pitch
                target[4] += -0.22 * capture       # left ankle pitch
                target[14] += -0.16 * capture      # waist pitch

        # Re-clip modified targets.
        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])
        return target

    def _get_info(self) -> Dict[str, float]:
        info = super()._get_info()
        up_x, up_y, up_z = self._root_orientation_proxies()
        info["torso_up_x"] = float(up_x)
        info["torso_up_y"] = float(up_y)
        info["wbc_support_lock_weight"] = float(self.support_lock_weight)
        info["wbc_height_target"] = float(self.height_target)
        return info

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]):
        total, rinfo = super()._compute_reward(action, info)
        upright_bonus = 6.0 * max(float(info["up_z"]), 0.0)
        angular_penalty = -1.8 * float(info["root_ang_vel"])
        height_penalty = -10.0 * abs(float(info["base_height"]) - self.height_target)
        wbc_total = float(total + upright_bonus + angular_penalty + height_penalty)
        rinfo["reward_wbc_upright"] = float(upright_bonus)
        rinfo["reward_wbc_angvel"] = float(angular_penalty)
        rinfo["reward_wbc_height"] = float(height_penalty)
        rinfo["reward_total"] = wbc_total
        rinfo["reward_version"] = "wbc_taskspace_right_lift_v3_capture_step"
        return wbc_total, rinfo
