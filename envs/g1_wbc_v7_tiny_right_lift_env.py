from __future__ import annotations

from typing import Dict, List

import mujoco
import numpy as np

from envs.g1_wbc_neutral_only_env import G1WBCNeutralOnlyEnv


class G1WBCV7TinyRightLiftEnv(G1WBCNeutralOnlyEnv):
    """
    V7 tiny right-foot lift.

    Clean design:
    - start from neutral-only stable standing
    - no parent swing target
    - no capture touchdown
    - no support push
    - no support IK lock
    - only a tiny right-foot Z task-space lift
    """

    def __init__(
        self,
        tiny_lift_height: float = 0.005,
        lift_start: float = 0.46,
        lift_peak: float = 0.62,
        land_end: float = 0.78,
        foot_ik_gain: float = 0.18,
        foot_ik_damping: float = 0.080,
        foot_ik_max_delta: float = 0.025,
        **kwargs,
    ):
        self.tiny_lift_height = float(tiny_lift_height)
        self.lift_start = float(lift_start)
        self.lift_peak = float(lift_peak)
        self.landing_end = float(land_end)
        self.foot_ik_gain = float(foot_ik_gain)
        self.foot_ik_damping = float(foot_ik_damping)
        self.foot_ik_max_delta = float(foot_ik_max_delta)

        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _tiny_lift_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.lift_start:
            return 0.0

        if phi <= self.lift_peak:
            return self._smoothstep((phi - self.lift_start) / max(self.lift_peak - self.lift_start, 1e-6))

        if phi <= self.landing_end:
            return 1.0 - self._smoothstep((phi - self.lift_peak) / max(self.landing_end - self.lift_peak, 1e-6))

        return 0.0

    def _right_foot_ik_delta(self, task_error: np.ndarray) -> np.ndarray:
        right_leg_idx: List[int] = [6, 7, 8, 9, 10, 11]
        right_dofs = [self.qvel_adrs[i] for i in right_leg_idx]

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.right_foot_site)

        J = jacp[:, right_dofs]
        A = J @ J.T + (self.foot_ik_damping * self.foot_ik_damping) * np.eye(3)

        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error

        delta = self.foot_ik_gain * delta
        return np.clip(delta, -self.foot_ik_max_delta, self.foot_ik_max_delta)

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = self.stand_joint_pos.copy().astype(np.float64)

        phi = float(info["phase"])
        lift_env = self._tiny_lift_env(phi)

        current_right = self.data.site_xpos[self.right_foot_site].copy()
        desired_right = self.right_foot_p0.copy()

        desired_right[2] += self.tiny_lift_height * lift_env

        err = desired_right - current_right

        # Tiny diagnostic: only vertical lift first.
        # Keep XY authority almost zero to avoid dragging the body backward.
        err[0] *= 0.00
        err[1] *= 0.00
        err[2] *= 1.00

        dq = self._right_foot_ik_delta(err)

        right_leg_idx: List[int] = [6, 7, 8, 9, 10, 11]
        current_q = np.array([self.data.qpos[self.qpos_adrs[i]] for i in right_leg_idx], dtype=np.float64)
        ik_target = current_q + dq

        for local_i, joint_i in enumerate(right_leg_idx):
            target[joint_i] = ik_target[local_i]

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()
        phi = float(info["phase"])
        lift_env = self._tiny_lift_env(phi)

        info["wbc_state"] = "V7_TINY_RIGHT_LIFT"
        info["v7_lift_env"] = float(lift_env)
        info["v7_target_clearance"] = float(self.tiny_lift_height * lift_env)
        info["wbc_capture_active"] = False
        info["wbc_abort_lift"] = False
        info["wbc_touchdown_force"] = 0.0

        return info
