from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from envs.g1_phase_bc_residual_env import G1PhaseBCResidualEnv


class G1PhaseBCResidualEnvV2(G1PhaseBCResidualEnv):
    """
    V2 reward correction.

    Main changes:
    1. Positive target velocity.
    2. Stronger backward velocity penalty.
    3. Stronger backward position penalty.
    4. Early fall-warning penalty before full termination.
    5. Slightly stronger residual allowed during training.
    """

    def __init__(
        self,
        *args: Any,
        forward_reward_weight: float = 0.60,
        backward_velocity_penalty_weight: float = 2.50,
        backward_position_penalty_weight: float = 1.20,
        fall_warning_weight: float = 4.00,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.forward_reward_weight = float(forward_reward_weight)
        self.backward_velocity_penalty_weight = float(backward_velocity_penalty_weight)
        self.backward_position_penalty_weight = float(backward_position_penalty_weight)
        self.fall_warning_weight = float(fall_warning_weight)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        previous_x = float(self.data.qpos[0])

        target = self._compute_target(action)
        self._apply_target(target)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

        base_x = float(self.data.qpos[0])
        base_y = float(self.data.qpos[1])
        base_z = float(self.data.qpos[2])
        up_z = self._get_up_z()

        vx = (base_x - previous_x) / max(self.frame_skip * self.sim_dt, 1e-6)

        height_target = float(self.stand_qpos[2])
        height_error = base_z - height_target

        height_reward = math.exp(-(height_error * height_error) / 0.02)
        velocity_tracking_reward = math.exp(-((vx - self.target_velocity) ** 2) / 0.05)

        # Positive x is the desired forward direction for our aligned AMASS B3 replay.
        forward_reward = self.forward_reward_weight * float(np.clip(vx, -0.50, 0.50))

        # Penalize moving backward.
        backward_velocity = max(0.0, -vx)
        backward_velocity_penalty = self.backward_velocity_penalty_weight * backward_velocity * backward_velocity

        # Penalize drifting too far backward in position.
        backward_position = max(0.0, -base_x - 0.05)
        backward_position_penalty = self.backward_position_penalty_weight * backward_position

        lateral_penalty = 0.20 * abs(base_y)

        residual_penalty = 0.015 * float(np.mean(action * action))

        tracking_error = float(
            np.mean(np.abs(self._get_joint_pos().astype(np.float64) - target))
        )
        ctrl_penalty = 0.04 * tracking_error

        # Penalize before the robot fully falls.
        up_warning = max(0.0, 0.88 - up_z)
        height_warning = max(0.0, 0.68 - base_z)
        fall_warning_penalty = self.fall_warning_weight * (
            up_warning * up_warning + height_warning * height_warning
        )

        reward = (
            0.40
            + 1.60 * up_z
            + 0.50 * height_reward
            + 0.60 * velocity_tracking_reward
            + forward_reward
            - backward_velocity_penalty
            - backward_position_penalty
            - lateral_penalty
            - residual_penalty
            - ctrl_penalty
            - fall_warning_penalty
        )

        terminated = bool(base_z < self.fall_height or up_z < self.fall_up_z)

        if terminated:
            reward -= 25.0

        self.step_count += 1
        self.phase_index = (self.phase_index + 1) % len(self.phase_bc_targets)
        self.prev_action = action.copy()

        truncated = bool(self.step_count >= self.max_episode_steps)

        obs = self._get_obs()
        info = self._get_info(action=action, reward=reward)

        info["vx"] = float(vx)
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["forward_reward"] = float(forward_reward)
        info["backward_velocity_penalty"] = float(backward_velocity_penalty)
        info["backward_position_penalty"] = float(backward_position_penalty)
        info["fall_warning_penalty"] = float(fall_warning_penalty)

        return obs, float(reward), terminated, truncated, info
