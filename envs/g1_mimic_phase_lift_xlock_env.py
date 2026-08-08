from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.g1_mimic_phase_lift_env import G1MimicPhaseLiftEnv


class G1MimicPhaseLiftXLockEnv(G1MimicPhaseLiftEnv):
    """
    X-locked DeepMimic-lite phase lift environment.

    This version targets the observed failure:
    the policy gets foot clearance by translating the root forward.

    Main changes:
    - tighter root-x termination;
    - strong root-x and root-x-velocity penalties;
    - swing/lift reward is not allowed to dominate if root-x is drifting;
    - reference remains reward-only, never a direct control teacher.
    """

    def __init__(
        self,
        *args,
        x_soft_limit: float = 0.10,
        x_hard_limit: float = 0.22,
        x_velocity_soft_limit: float = 0.25,
        x_velocity_hard_limit: float = 1.05,
        no_lift_terminate_target: float = 0.012,
        no_lift_terminate_clearance: float = 0.004,
        no_lift_penalty_weight: float = 55.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x_soft_limit = float(x_soft_limit)
        self.x_hard_limit = float(x_hard_limit)
        self.x_velocity_soft_limit = float(x_velocity_soft_limit)
        self.x_velocity_hard_limit = float(x_velocity_hard_limit)
        self.no_lift_terminate_target = float(no_lift_terminate_target)
        self.no_lift_terminate_clearance = float(no_lift_terminate_clearance)
        self.no_lift_penalty_weight = float(no_lift_penalty_weight)

    def _main_clearance_and_target(self, info: Dict[str, float]) -> Tuple[float, float, bool]:
        if self.stage == "right_lift":
            return (
                float(info["right_foot_clearance"]),
                float(info["right_target_clearance"]),
                bool(info["right_swing"]),
            )
        if self.stage == "left_lift":
            return (
                float(info["left_foot_clearance"]),
                float(info["left_target_clearance"]),
                bool(info["left_swing"]),
            )
        clearance = min(float(info["left_foot_clearance"]), float(info["right_foot_clearance"]))
        target = max(float(info["left_target_clearance"]), float(info["right_target_clearance"]))
        swing = bool(info["left_swing"]) or bool(info["right_swing"])
        return clearance, target, swing

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]):
        reward, reward_info = super()._compute_reward(action, info)

        x_pos = float(info["x_position"])
        x_vel = float(info["x_velocity"])
        x_abs = abs(x_pos)
        xv_abs = abs(x_vel)

        clearance, target_clearance, is_swing = self._main_clearance_and_target(info)
        lift_progress = min(clearance / max(target_clearance, 1e-6), 1.0) if target_clearance > 1e-5 else 0.0

        x_soft_excess = max(0.0, x_abs - self.x_soft_limit)
        xvel_soft_excess = max(0.0, xv_abs - self.x_velocity_soft_limit)

        # Harsh but targeted. The previous policy sat at x≈+0.35 and still scored.
        # This makes that behavior strongly negative before termination.
        xlock_penalty = (
            -260.0 * x_soft_excess
            -95.0 * xvel_soft_excess
        )

        # Do not allow clearance reward to be purchased through root drift.
        # This term activates mainly during swing.
        drifted_lift_penalty = 0.0
        if is_swing and target_clearance > 0.004:
            drifted_lift_penalty = -30.0 * lift_progress * (
                max(0.0, x_abs - 0.08) / 0.14
                + max(0.0, xv_abs - 0.20) / 0.85
            )

        # v2: eliminate the standing-still optimum. In xlock-v1 the policy could
        # survive 700 steps with zero clearance, so survival must not dominate.
        no_lift_penalty = 0.0
        if is_swing and target_clearance > 0.004:
            missing_fraction = max(0.0, (target_clearance - clearance) / max(target_clearance, 1e-6))
            no_lift_penalty = -self.no_lift_penalty_weight * missing_fraction

        total = float(reward + xlock_penalty + drifted_lift_penalty + no_lift_penalty)

        reward_info.update(
            {
                "reward_xlock_penalty": float(xlock_penalty),
                "reward_drifted_lift_penalty": float(drifted_lift_penalty),
                "reward_no_lift_penalty": float(no_lift_penalty),
                "x_soft_limit": float(self.x_soft_limit),
                "x_hard_limit": float(self.x_hard_limit),
                "x_velocity_soft_limit": float(self.x_velocity_soft_limit),
                "x_velocity_hard_limit": float(self.x_velocity_hard_limit),
                "no_lift_terminate_target": float(self.no_lift_terminate_target),
                "no_lift_terminate_clearance": float(self.no_lift_terminate_clearance),
                "main_clearance": float(clearance),
                "main_target_clearance": float(target_clearance),
                "lift_progress": float(lift_progress),
                "reward_total": float(total),
                "reward_version": "mimic_phase_lift_xlock_v2_no_lift_gate",
            }
        )
        return total, reward_info

    def _no_lift_failure(self, info: Dict[str, float]) -> bool:
        clearance, target_clearance, is_swing = self._main_clearance_and_target(info)
        return (
            bool(is_swing)
            and target_clearance >= self.no_lift_terminate_target
            and clearance <= self.no_lift_terminate_clearance
        )

    def _terminated(self, info: Dict[str, float]) -> bool:
        if super()._terminated(info):
            return True
        if abs(float(info["x_position"])) > self.x_hard_limit:
            return True
        if abs(float(info["x_velocity"])) > self.x_velocity_hard_limit:
            return True
        if self._no_lift_failure(info):
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        base_reason = super().termination_reason(info)
        if base_reason != "not_terminated":
            return base_reason
        if abs(float(info["x_position"])) > self.x_hard_limit:
            return "xlock_position_limit"
        if abs(float(info["x_velocity"])) > self.x_velocity_hard_limit:
            return "xlock_velocity_limit"
        if self._no_lift_failure(info):
            return "no_lift_mid_swing"
        return "not_terminated"
