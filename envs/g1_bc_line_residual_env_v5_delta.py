import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env_v3 import (
    G1BCLineResidualEnvV3,
)


class G1BCLineResidualEnvV5Delta(
    G1BCLineResidualEnvV3
):

    """
    V5 architecture:

        BC nominal gait
            +
        frozen V2 residual policy
            +
        small trainable delta action

    The external Gym action is ONLY the new delta.

    The frozen V2 policy sees the exact same 62-D observation
    and produces the trusted base residual action.

    combined_action =
        frozen_v2_action
        + delta_action_scale * delta_action

    The combined action is then sent through the V3 physical
    control envelope.
    """

    def __init__(
        self,
        *args,
        frozen_v2_model=None,
        delta_action_scale=0.10,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        if frozen_v2_model is None:

            frozen_v2_model = (
                ROOT
                / "models"
                / "g1_ppo_bc_line_residual_v2_50k.zip"
            )

        self.frozen_v2_model_path = Path(
            frozen_v2_model
        )

        if not self.frozen_v2_model_path.exists():

            raise FileNotFoundError(
                self.frozen_v2_model_path
            )

        self.delta_action_scale = float(
            delta_action_scale
        )

        if not (
            0.0
            <= self.delta_action_scale
            <= 0.25
        ):

            raise ValueError(
                "delta_action_scale must be between 0 and 0.25"
            )

        # -------------------------------------------------------------
        # Load frozen V2 WITHOUT a training environment.
        # -------------------------------------------------------------

        self.frozen_v2 = PPO.load(
            str(
                self.frozen_v2_model_path
            ),
            device="cpu",
        )

        self.frozen_v2.policy.set_training_mode(
            False
        )

        # Validate compatibility immediately.
        if tuple(
            self.frozen_v2.observation_space.shape
        ) != tuple(
            self.observation_space.shape
        ):

            raise RuntimeError(
                "Frozen V2 observation space does not match V5."
            )

        if tuple(
            self.frozen_v2.action_space.shape
        ) != tuple(
            self.action_space.shape
        ):

            raise RuntimeError(
                "Frozen V2 action space does not match V5."
            )

        self._v5_obs = None

        self._previous_delta_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )


    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self._v5_obs = np.asarray(
            obs,
            dtype=np.float32,
        ).copy()

        self._previous_delta_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )

        info = dict(info)

        info.update(
            {
                "v5_delta_scale":
                    float(
                        self.delta_action_scale
                    ),

                "v5_base_action_max":
                    0.0,

                "v5_delta_action_max":
                    0.0,

                "v5_combined_action_max":
                    0.0,
            }
        )

        return obs, info


    def step(
        self,
        delta_action,
    ):

        if self._v5_obs is None:

            raise RuntimeError(
                "reset() must be called before step()."
            )

        delta_action = np.asarray(
            delta_action,
            dtype=np.float32,
        ).reshape(
            self.action_space.shape
        )

        delta_action = np.clip(
            delta_action,
            -1.0,
            1.0,
        )


        # =============================================================
        # FROZEN V2 BASE ACTION
        # =============================================================

        base_action, _ = self.frozen_v2.predict(
            self._v5_obs,
            deterministic=True,
        )

        base_action = np.asarray(
            base_action,
            dtype=np.float32,
        ).reshape(
            self.action_space.shape
        )


        # =============================================================
        # RESIDUAL-ON-RESIDUAL
        #
        # At scale 0.10:
        #
        # maximum action-space correction = +/-0.10
        #
        # maximum hip/knee target correction at full residual ramp:
        #
        #   0.14 rad * 0.10
        #   = 0.014 rad
        #   about 0.80 degrees
        #
        # Ankle roll is still further limited by V3's
        # per-joint 0.4225 envelope.
        # =============================================================

        scaled_delta = (
            self.delta_action_scale
            * delta_action
        )

        combined_action = np.clip(
            base_action
            + scaled_delta,
            -1.0,
            1.0,
        )


        # =============================================================
        # PHYSICAL STEP
        #
        # V3 reward/physics receives the combined action.
        # =============================================================

        (
            obs,
            base_reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            combined_action
        )


        # =============================================================
        # V5 EXTRA REWARD TERMS
        #
        # Goal:
        #
        # - preserve frozen V2 behavior
        # - prevent large unnecessary corrections
        # - explicitly preserve useful forward progress
        #
        # The V3 reward still supplies:
        # upright, height, line, yaw, angular-rate and fall shaping.
        # =============================================================

        vx = float(
            self.data.qvel[0]
        )

        y = float(
            info["y"]
        )

        yaw = np.deg2rad(
            float(
                info["yaw_deg"]
            )
        )

        up = float(
            info["up_z"]
        )


        # Stable forward progress only.
        #
        # Negative X velocity is desired.
        forward_ratio = np.clip(
            (-vx) / 0.18,
            0.0,
            1.5,
        )

        upright_gate = np.clip(
            (up - 0.65) / 0.25,
            0.0,
            1.0,
        )

        line_gate = np.exp(
            -(y * y) / 0.025
        )

        heading_gate = np.exp(
            -(yaw * yaw) / 0.15
        )

        stable_forward_bonus = (
            0.75
            * forward_ratio
            * upright_gate
            * line_gate
            * heading_gate
        )


        # Penalize only the NEW correction,
        # not the frozen V2 base policy.
        delta_effort_penalty = (
            0.06
            * float(
                np.mean(
                    delta_action ** 2
                )
            )
        )

        delta_smoothness_penalty = (
            0.04
            * float(
                np.mean(
                    (
                        delta_action
                        - self._previous_delta_action
                    ) ** 2
                )
            )
        )


        # Explicit anti-stall term.
        #
        # Only applies while sufficiently upright.
        stall_penalty = 0.0

        if (
            self.episode_step
            > self.stand_frames
            and up > 0.80
            and vx > -0.05
        ):

            stall_error = (
                vx + 0.05
            )

            stall_penalty = (
                0.50 * stall_error
                + 2.00 * stall_error ** 2
            )


        reward = (
            float(base_reward)
            + float(
                stable_forward_bonus
            )
            - float(
                delta_effort_penalty
            )
            - float(
                delta_smoothness_penalty
            )
            - float(
                stall_penalty
            )
        )


        info = dict(info)

        info.update(
            {
                "v5_base_action_max":
                    float(
                        np.max(
                            np.abs(
                                base_action
                            )
                        )
                    ),

                "v5_delta_action_max":
                    float(
                        np.max(
                            np.abs(
                                delta_action
                            )
                        )
                    ),

                "v5_scaled_delta_max":
                    float(
                        np.max(
                            np.abs(
                                scaled_delta
                            )
                        )
                    ),

                "v5_combined_action_max":
                    float(
                        np.max(
                            np.abs(
                                combined_action
                            )
                        )
                    ),

                "v5_stable_forward_bonus":
                    float(
                        stable_forward_bonus
                    ),

                "v5_delta_effort_penalty":
                    float(
                        delta_effort_penalty
                    ),

                "v5_delta_smoothness_penalty":
                    float(
                        delta_smoothness_penalty
                    ),

                "v5_stall_penalty":
                    float(
                        stall_penalty
                    ),
            }
        )


        self._previous_delta_action = (
            delta_action.copy()
        )

        self._v5_obs = np.asarray(
            obs,
            dtype=np.float32,
        ).copy()


        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
