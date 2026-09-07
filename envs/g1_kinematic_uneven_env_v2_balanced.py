from __future__ import annotations

import numpy as np


# ============================================================
# BASE REWARD-V2 ENVIRONMENT
#
# IMPORTANT:
#
# This file does NOT change:
#
# - Reward V2
# - observations
# - actions
# - residual scales
# - residual smoothing
# - terrain
# - BC policy
# - root trajectory
# - MuJoCo behavior
#
# It changes ONLY random training-start sampling.
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv as G1KinematicUnevenEnvV2,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# BALANCED RANDOM-START RANGE
#
# We exclude TRAIN_END_STEP itself because starting exactly
# at the final state would provide zero useful transitions.
#
# Therefore:
#
# start range = TRAIN_START_STEP .. TRAIN_END_STEP - 1
#
# Example:
#
# 150 .. 749
#
# Episodes that begin near the end are intentionally shorter.
#
# The original V2 environment already truncates when:
#
# current_step >= TRAIN_END_STEP
#
# so this does not require any change to step().
# ============================================================

BALANCED_START_MIN = (
    TRAIN_START_STEP
)

BALANCED_START_MAX = (
    TRAIN_END_STEP
    -
    1
)


class G1KinematicUnevenEnv(
    G1KinematicUnevenEnvV2
):

    """
    Reward-V2 environment with balanced trajectory-start
    sampling.

    The original Reward-V2 environment restricts random
    starts so every episode can run for the full configured
    episode length.

    That causes the late portion of the trajectory to receive
    substantially less training exposure.

    This subclass keeps the entire Reward-V2 environment
    unchanged except for reset-time start selection.

    Random starts are sampled uniformly from:

        TRAIN_START_STEP
        through
        TRAIN_END_STEP - 1

    Episodes starting near TRAIN_END_STEP are therefore shorter
    and naturally truncate when the reference reaches the end.

    Explicit reset options={"start_step": ...} are passed
    directly to the original environment unchanged.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        # ----------------------------------------------------
        # The base environment's own random_start logic must
        # not choose the start because it uses the restricted
        # latest-start rule.
        #
        # We control random starts here instead.
        # ----------------------------------------------------

        kwargs[
            "random_start"
        ] = False

        super().__init__(
            *args,
            **kwargs,
        )

        self.random_start = True

        self._balanced_rng = (
            np.random.default_rng()
        )


    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        # ----------------------------------------------------
        # EXPLICIT START REQUEST
        #
        # Keep the original environment behavior exactly.
        #
        # This is required for:
        #
        # - deterministic evaluation
        # - debugging
        # - boundary tests
        # - viewers
        # ----------------------------------------------------

        if (
            options is not None
            and
            "start_step"
            in options
        ):

            return super().reset(
                seed=seed,
                options=options,
            )


        # ----------------------------------------------------
        # RNG SEEDING
        #
        # Seed this start sampler whenever Gym/SB3 provides
        # an explicit seed.
        # ----------------------------------------------------

        if seed is not None:

            self._balanced_rng = (
                np.random.default_rng(
                    int(
                        seed
                    )
                )
            )


        # ----------------------------------------------------
        # BALANCED RANDOM START
        #
        # numpy.integers() upper bound is exclusive,
        # therefore TRAIN_END_STEP gives:
        #
        # TRAIN_START_STEP .. TRAIN_END_STEP - 1
        # ----------------------------------------------------

        start_step = int(
            self._balanced_rng.integers(
                BALANCED_START_MIN,
                TRAIN_END_STEP,
            )
        )


        # ----------------------------------------------------
        # Let the original Reward-V2 reset implementation do
        # all actual environment initialization.
        #
        # We only provide the selected start step.
        # ----------------------------------------------------

        observation, info = super().reset(
            seed=seed,
            options={
                "start_step":
                    start_step,
            },
        )


        # ----------------------------------------------------
        # EXTRA DIAGNOSTICS
        # ----------------------------------------------------

        info = dict(
            info
        )

        info[
            "balanced_random_start"
        ] = True

        info[
            "balanced_start_min"
        ] = int(
            BALANCED_START_MIN
        )

        info[
            "balanced_start_max"
        ] = int(
            BALANCED_START_MAX
        )


        return (
            observation,
            info,
        )