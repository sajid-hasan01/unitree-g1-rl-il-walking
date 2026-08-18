from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from envs.g1_phase_bc_residual_env_v2 import G1PhaseBCResidualEnvV2


class G1PhaseBCResidualEnvV3(G1PhaseBCResidualEnvV2):
    """
    PPO V3-A environment.

    Goal:
    - Start from the stable PPO V2 behavior.
    - Keep upright balance.
    - Add stronger forward-progress pressure.
    - Penalize standing still near the starting root position.

    Important:
    Observation and action spaces are unchanged from V2.
    This allows policy-weight transfer from the PPO V2 final model.
    """

    def __init__(
        self,
        *args: Any,
        forward_velocity_weight: float = 1.20,
        velocity_tracking_weight: float = 0.70,
        displacement_reward_weight: float = 1.00,
        progress_deficit_penalty_weight: float = 2.00,
        backward_velocity_penalty_weight: float = 3.00,
        backward_position_penalty_weight: float = 2.00,
        standstill_penalty_weight: float = 1.20,
        fall_warning_weight: float = 5.00,
        min_progress_rate: float = 0.020,
        progress_grace_time: float = 4.00,
        standstill_velocity_threshold: float = 0.025,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.forward_velocity_weight = float(forward_velocity_weight)
        self.velocity_tracking_weight = float(velocity_tracking_weight)
        self.displacement_reward_weight = float(displacement_reward_weight)
        self.progress_deficit_penalty_weight = float(progress_deficit_penalty_weight)
        self.backward_velocity_penalty_weight = float(backward_velocity_penalty_weight)
        self.backward_position_penalty_weight = float(backward_position_penalty_weight)
        self.standstill_penalty_weight = float(standstill_penalty_weight)
        self.fall_warning_weight = float(fall_warning_weight)
        self.min_progress_rate = float(min_progress_rate)
        self.progress_grace_time = float(progress_grace_time)
        self.standstill_velocity_threshold = float(standstill_velocity_threshold)

        self.episode_start_x = 0.0

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        self.episode_start_x = float(self.data.qpos[0])
        return obs, info

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

        dt = max(self.frame_skip * self.sim_dt, 1e-6)
        vx = (base_x - previous_x) / dt

        elapsed_time = self.step_count * self.control_dt
        relative_x = base_x - self.episode_start_x

        height_target = float(self.stand_qpos[2])
        height_error = base_z - height_target

        height_reward = math.exp(-(height_error * height_error) / 0.020)

        velocity_tracking_reward = self.velocity_tracking_weight * math.exp(
            -((vx - self.target_velocity) ** 2) / 0.035
        )

        forward_velocity_reward = self.forward_velocity_weight * float(
            np.clip(vx, 0.0, 0.35)
        )

        displacement_reward = self.displacement_reward_weight * float(
            np.clip(relative_x, 0.0, 1.00)
        )

        desired_progress = self.min_progress_rate * max(
            0.0,
            elapsed_time - self.progress_grace_time,
        )

        progress_deficit = max(0.0, desired_progress - relative_x)
        progress_deficit_penalty = self.progress_deficit_penalty_weight * progress_deficit

        backward_velocity = max(0.0, -vx)
        backward_velocity_penalty = (
            self.backward_velocity_penalty_weight
            * backward_velocity
            * backward_velocity
        )

        backward_position = max(0.0, -relative_x - 0.03)
        backward_position_penalty = (
            self.backward_position_penalty_weight * backward_position
        )

        if self.step_count > self.transition_steps:
            standstill_gap = max(
                0.0,
                self.standstill_velocity_threshold - max(0.0, vx),
            )
            standstill_penalty = self.standstill_penalty_weight * standstill_gap
        else:
            standstill_penalty = 0.0

        lateral_penalty = 0.20 * abs(base_y)

        residual_penalty = 0.015 * float(np.mean(action * action))

        tracking_error = float(
            np.mean(np.abs(self._get_joint_pos().astype(np.float64) - target))
        )
        ctrl_penalty = 0.04 * tracking_error

        up_warning = max(0.0, 0.88 - up_z)
        height_warning = max(0.0, 0.68 - base_z)
        fall_warning_penalty = self.fall_warning_weight * (
            up_warning * up_warning + height_warning * height_warning
        )

        reward = (
            0.25
            + 1.25 * up_z
            + 0.45 * height_reward
            + velocity_tracking_reward
            + forward_velocity_reward
            + displacement_reward
            - progress_deficit_penalty
            - backward_velocity_penalty
            - backward_position_penalty
            - standstill_penalty
            - lateral_penalty
            - residual_penalty
            - ctrl_penalty
            - fall_warning_penalty
        )

        terminated = bool(base_z < self.fall_height or up_z < self.fall_up_z)

        if terminated:
            reward -= 30.0

        self.step_count += 1
        self.phase_index = (self.phase_index + 1) % len(self.phase_bc_targets)
        self.prev_action = action.copy()

        truncated = bool(self.step_count >= self.max_episode_steps)

        obs = self._get_obs()
        info = self._get_info(action=action, reward=reward)

        info["vx"] = float(vx)
        info["relative_x"] = float(relative_x)
        info["desired_progress"] = float(desired_progress)
        info["progress_deficit"] = float(progress_deficit)
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["forward_velocity_reward"] = float(forward_velocity_reward)
        info["velocity_tracking_reward"] = float(velocity_tracking_reward)
        info["displacement_reward"] = float(displacement_reward)
        info["progress_deficit_penalty"] = float(progress_deficit_penalty)
        info["backward_velocity_penalty"] = float(backward_velocity_penalty)
        info["backward_position_penalty"] = float(backward_position_penalty)
        info["standstill_penalty"] = float(standstill_penalty)
        info["fall_warning_penalty"] = float(fall_warning_penalty)

        return obs, float(reward), terminated, truncated, info

