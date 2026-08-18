from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


class G1WBCV76ResidualRightLiftEnv(G1WBCV72PhaseRightLiftEnv):
    """
    V7.6 residual-RL right-lift environment.

    Base controller:
    FINAL_v72_phase_right_lift_shaped

    Zero action should reproduce:
    - right-foot lift
    - stable landing
    - max_steps completion

    RL action adds tiny residual corrections during swing only.
    """

    def __init__(
        self,
        residual_ankle_scale: float = 0.035,
        residual_knee_scale: float = 0.012,
        residual_hip_pitch_scale: float = 0.010,
        residual_roll_scale: float = 0.010,
        **kwargs,
    ):
        self.residual_ankle_scale = float(residual_ankle_scale)
        self.residual_knee_scale = float(residual_knee_scale)
        self.residual_hip_pitch_scale = float(residual_hip_pitch_scale)
        self.residual_roll_scale = float(residual_roll_scale)

        self._initial_gap = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)

        super().__init__(**kwargs)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def _foot_gap(self) -> float:
        right_y = float(self.data.site_xpos[self.right_foot_site][1])
        left_y = float(self.data.site_xpos[self.left_foot_site][1])
        return abs(right_y - left_y)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._initial_gap = self._foot_gap()
        self._last_action = np.zeros(4, dtype=np.float32)

        info["v76_gap"] = float(self._initial_gap)
        info["v76_gap_delta"] = 0.0

        return obs, info

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # Important:
        # base controller receives zero action, so zero residual exactly means V7.2 shaped.
        zero_action = np.zeros(4, dtype=np.float32)
        target = super()._target_joint_position(zero_action, info)

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < 4:
            padded = np.zeros(4, dtype=np.float32)
            padded[: a.shape[0]] = a
            a = padded

        a = np.clip(a[:4], -1.0, 1.0)
        self._last_action = a.copy()

        phi = float(info["phase"])
        lift = float(self._tiny_lift_env(phi))

        if lift <= 1e-6:
            return target

        # Controlled 15-DOF indices:
        # right_hip_pitch  = 6
        # right_hip_roll   = 7
        # right_knee       = 9
        # right_ankle_pitch= 10
        # right_ankle_roll = 11
        # waist_roll       = 13

        target[10] += self.residual_ankle_scale * float(a[0]) * lift
        target[9] += self.residual_knee_scale * float(a[1]) * lift
        target[6] += self.residual_hip_pitch_scale * float(a[2]) * lift

        roll = self.residual_roll_scale * float(a[3]) * lift
        target[7] += roll
        target[11] += -0.50 * roll
        target[13] += -0.20 * roll

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)

        phi = float(info["phase"])
        up_z = float(info["up_z"])
        right_clearance = float(info["right_foot_clearance"])
        right_contact = bool(info["right_contact"])
        support_slip = float(info["support_slip"])
        root_ang_vel = float(info["root_ang_vel"])
        x_velocity = float(info["x_velocity"])

        gap = self._foot_gap()
        gap_delta = gap - self._initial_gap

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] < 4:
            padded = np.zeros(4, dtype=np.float32)
            padded[: action_arr.shape[0]] = action_arr
            action_arr = padded
        action_arr = np.clip(action_arr[:4], -1.0, 1.0)

        swing_phase = 0.46 <= phi <= 0.82
        air_bonus = 1.0 if not right_contact else 0.0

        clearance_score = min(max(right_clearance / 0.035, 0.0), 1.0)

        reward = 0.25
        reward += 3.0 * max(up_z - 0.90, 0.0)

        if swing_phase:
            reward += 1.50 * clearance_score
            reward += 1.25 * air_bonus

        if phi >= 0.82:
            reward -= 3.00 * max(gap_delta, 0.0)
            reward -= 0.50 * abs(x_velocity)

        reward -= 1.00 * support_slip
        reward -= 0.20 * root_ang_vel
        reward -= 0.03 * float(np.sum(action_arr * action_arr))

        if terminated:
            reward -= 10.0

        info["v76_gap"] = float(gap)
        info["v76_gap_delta"] = float(gap_delta)
        info["v76_reward"] = float(reward)

        return obs, float(reward), terminated, truncated, info
