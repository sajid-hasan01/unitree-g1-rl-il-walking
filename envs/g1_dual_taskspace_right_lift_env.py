from __future__ import annotations

from typing import Dict, List

import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


class G1DualTaskspaceRightLiftEnv(G1TaskspaceRightLiftEnv):
    """
    Dual-task-space right-foot lift.

    Previous task-space controllers moved only the right swing foot. Recent logs
    show the right foot can reach visible clearance, but the root falls backward
    and the support side is not controlled enough. This class keeps the right
    foot lift task and adds a left support-foot lock with Jacobian IK.
    """

    def __init__(
        self,
        support_lock_weight: float = 0.95,
        support_xy_weight: float = 0.35,
        support_z_weight: float = 1.80,
        support_ik_gain: float = 0.85,
        support_ik_damping: float = 0.055,
        support_ik_max_delta: float = 0.16,
        **kwargs,
    ):
        # The parent constructor calls self._get_obs(), which calls this
        # subclass _get_info(). These fields must exist before super().__init__.
        self.support_lock_weight = float(support_lock_weight)
        self.support_xy_weight = float(support_xy_weight)
        self.support_z_weight = float(support_z_weight)
        self.support_ik_gain = float(support_ik_gain)
        self.support_ik_damping = float(support_ik_damping)
        self.support_ik_max_delta = float(support_ik_max_delta)

        kwargs.setdefault("max_steps", 520)
        kwargs.setdefault("cycle_duration", 5.2)
        kwargs.setdefault("shift_start", 0.08)
        kwargs.setdefault("swing_start", 0.42)
        kwargs.setdefault("swing_end", 0.72)
        kwargs.setdefault("land_end", 0.88)
        kwargs.setdefault("target_clearance", 0.030)
        kwargs.setdefault("target_lateral_shift", 0.018)
        kwargs.setdefault("ik_gain", 1.05)
        kwargs.setdefault("ik_damping", 0.050)
        kwargs.setdefault("ik_max_delta", 0.18)
        kwargs.setdefault("xy_hold_weight", 0.06)
        kwargs.setdefault("z_lift_weight", 1.15)
        kwargs.setdefault("x_hard_limit", 0.36)
        kwargs.setdefault("y_hard_limit", 0.30)
        kwargs.setdefault("x_velocity_hard_limit", 1.35)
        kwargs.setdefault("y_velocity_hard_limit", 1.35)

        super().__init__(**kwargs)

    def _support_ik_delta(self, site_id: int, dof_indices: List[int], task_error: np.ndarray) -> np.ndarray:
        import mujoco

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        j = jacp[:, dof_indices]

        damping = self.support_ik_damping
        a = j @ j.T + (damping * damping) * np.eye(3)
        try:
            delta = j.T @ np.linalg.solve(a, task_error)
        except np.linalg.LinAlgError:
            delta = j.T @ np.linalg.pinv(a) @ task_error

        delta = self.support_ik_gain * delta
        return np.clip(delta, -self.support_ik_max_delta, self.support_ik_max_delta)

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        swing = float(info["swing_env"])
        shift = float(info["shift_env"])
        lock_env = min(1.0, max(0.0, 0.35 * shift + 0.65 * max(swing, 0.0)))
        if lock_env <= 0.01:
            return target

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

        blend = self.support_lock_weight * lock_env
        for local_i, joint_i in enumerate(left_leg_idx):
            target[joint_i] = (1.0 - blend) * target[joint_i] + blend * ik_target[local_i]

        # Slight stance compliance so the support leg does not behave like a rigid stick.
        target[3] += 0.035 * lock_env

        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])

        return target

    def _get_info(self) -> Dict[str, float]:
        info = super()._get_info()
        info["support_lock_weight"] = float(self.support_lock_weight)
        info["support_xy_weight"] = float(self.support_xy_weight)
        info["support_z_weight"] = float(self.support_z_weight)
        info["reward_version"] = "dual_taskspace_right_lift_v1_support_lock"
        return info
