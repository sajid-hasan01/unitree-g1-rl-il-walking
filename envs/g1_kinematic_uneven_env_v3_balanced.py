from __future__ import annotations

import numpy as np


# ============================================================
# BALANCED-START REWARD-V2 BASE
#
# This already provides:
#
# - frozen 12-DOF BC
# - 12-D PPO residual action
# - 62-D observation
# - uneven terrain
# - scripted root placement
# - residual smoothing
# - full-trajectory balanced random starts
# - mj_forward() only
#
# V3 changes ONLY the reward.
# ============================================================

from envs.g1_kinematic_uneven_env_v2_balanced import (
    G1KinematicUnevenEnv as G1KinematicUnevenEnvV2Balanced,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
)


# ============================================================
# EXACT V2 CONSTANT USED AS NORMALIZATION SCALE
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    PENETRATION_SCALE,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# REWARD-V3 ADDITION
#
# Offline sweep selection:
#
# absolute severe-penetration threshold = 30 mm
# severe penalty weight                 = 4.0
#
# These are reward hyperparameters.
# They are NOT measured robot constants.
#
# V2's existing phase-aware penetration term remains unchanged.
#
# V3 adds an independent safeguard against large ABSOLUTE
# terrain penetration:
#
#   actual_pen_i =
#       max(-signed_distance_i, 0)
#
#   max_actual_pen =
#       max(actual_pen_i)
#
#   severe_excess =
#       max(
#           max_actual_pen
#           -
#           SEVERE_PENETRATION_THRESHOLD,
#           0
#       )
#
#   severe_cost =
#       (
#           severe_excess
#           /
#           PENETRATION_SCALE
#       ) ** 2
#
#   reward_v3 =
#       reward_v2
#       -
#       SEVERE_PENETRATION_WEIGHT
#       *
#       severe_cost
# ============================================================

SEVERE_PENETRATION_THRESHOLD = 0.030

SEVERE_PENETRATION_WEIGHT = 4.0


class G1KinematicUnevenEnv(
    G1KinematicUnevenEnvV2Balanced
):

    """
    Reward-V3 balanced-start kinematic G1 environment.

    Reward V3 preserves Reward V2 completely and adds one
    penetration-only safeguard:

        severe absolute penetration penalty

    No changes are made to:

    - observation space
    - action space
    - residual scales
    - residual smoothing
    - frozen BC
    - terrain
    - root trajectory
    - joint limits
    - action bounds
    - balanced-start sampling
    - MuJoCo stepping behavior

    The environment remains kinematic and uses mj_forward()
    only through the inherited implementation.
    """

    def _reward(
        self,
        action,
        previous_residual_before_action,
    ):

        # ----------------------------------------------------
        # EXACT REWARD V2
        #
        # The parent ultimately calls the original V2
        # implementation.
        # ----------------------------------------------------

        (
            reward_v2,
            info,
        ) = super()._reward(
            action,
            previous_residual_before_action,
        )


        # ----------------------------------------------------
        # CURRENT SIGNED SOLE/TERRAIN DISTANCES
        #
        # Shape = (8,)
        #
        # Negative signed distance means terrain penetration.
        # ----------------------------------------------------

        current_distances = np.asarray(
            self.current_distances,
            dtype=np.float64,
        )

        if (
            current_distances.shape
            !=
            (8,)
        ):

            raise RuntimeError(
                "Reward V3 expected exactly 8 "
                "sole signed distances, got "
                f"{current_distances.shape}."
            )


        # ----------------------------------------------------
        # ABSOLUTE RAW PENETRATION
        #
        # Example:
        #
        # distance = +0.010 m
        # penetration = 0
        #
        # distance = -0.050 m
        # penetration = 0.050 m
        # ----------------------------------------------------

        actual_penetration = np.maximum(
            -current_distances,
            0.0,
        )

        max_actual_penetration = float(
            np.max(
                actual_penetration
            )
        )


        # ----------------------------------------------------
        # SEVERE EXCESS
        #
        # No V3 penalty at or below 30 mm.
        # ----------------------------------------------------

        severe_penetration_excess = float(
            max(
                max_actual_penetration
                -
                SEVERE_PENETRATION_THRESHOLD,
                0.0,
            )
        )


        # ----------------------------------------------------
        # NORMALIZED SQUARED COST
        #
        # Same 50 mm scale already used by V2 penetration.
        # ----------------------------------------------------

        severe_penetration_cost = float(
            (
                severe_penetration_excess
                /
                PENETRATION_SCALE
            )
            **
            2
        )


        # ----------------------------------------------------
        # WEIGHTED V3 ADDITION
        # ----------------------------------------------------

        severe_penetration_penalty = float(
            SEVERE_PENETRATION_WEIGHT
            *
            severe_penetration_cost
        )


        # ----------------------------------------------------
        # FINAL REWARD V3
        # ----------------------------------------------------

        reward_v3 = float(
            reward_v2
            -
            severe_penetration_penalty
        )


        # ----------------------------------------------------
        # DIAGNOSTICS
        #
        # Preserve every original Reward-V2 info field.
        # Add V3-specific fields.
        #
        # Important:
        # overwrite info["reward"] so it matches the reward
        # actually returned to PPO.
        # ----------------------------------------------------

        info = dict(
            info
        )

        info[
            "reward_v2_before_severe_penalty"
        ] = float(
            reward_v2
        )

        info[
            "max_actual_penetration_m"
        ] = float(
            max_actual_penetration
        )

        info[
            "severe_penetration_threshold_m"
        ] = float(
            SEVERE_PENETRATION_THRESHOLD
        )

        info[
            "severe_penetration_excess_m"
        ] = float(
            severe_penetration_excess
        )

        info[
            "severe_penetration_cost"
        ] = float(
            severe_penetration_cost
        )

        info[
            "severe_penetration_weight"
        ] = float(
            SEVERE_PENETRATION_WEIGHT
        )

        info[
            "severe_penetration_penalty"
        ] = float(
            severe_penetration_penalty
        )

        info[
            "reward"
        ] = float(
            reward_v3
        )


        return (
            reward_v3,
            info,
        )