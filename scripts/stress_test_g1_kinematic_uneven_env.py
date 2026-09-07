import numpy as np

from envs.g1_kinematic_uneven_env import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


NUM_EPISODES = 20

BASE_SEED = 425


def assert_finite(
    name,
    value,
):
    array = np.asarray(
        value
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise RuntimeError(
            f"{name} contains NaN/Inf:\n"
            f"{value}"
        )


def main():

    print(
        "=" * 90
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "FINAL ENVIRONMENT STRESS TEST"
    )

    print(
        "=" * 90
    )

    env = (
        G1KinematicUnevenEnv(
            random_start=True
        )
    )

    try:

        # ====================================================
        # 1. DETERMINISTIC SEEDED RESET
        # ====================================================

        print()

        print(
            "1. SEEDED RESET TEST"
        )

        print(
            "-" * 90
        )

        obs_a, info_a = (
            env.reset(
                seed=12345
            )
        )

        start_a = int(
            info_a[
                "start_step"
            ]
        )

        obs_b, info_b = (
            env.reset(
                seed=12345
            )
        )

        start_b = int(
            info_b[
                "start_step"
            ]
        )

        reset_difference = float(
            np.max(
                np.abs(
                    obs_a
                    -
                    obs_b
                )
            )
        )

        print(
            "start A:",
            start_a,
        )

        print(
            "start B:",
            start_b,
        )

        print(
            "max observation difference:",
            reset_difference,
        )

        if (
            start_a
            !=
            start_b
        ):
            raise RuntimeError(
                "Seeded reset produced "
                "different start steps."
            )

        if (
            reset_difference
            !=
            0.0
        ):
            raise RuntimeError(
                "Seeded reset produced "
                "different observations."
            )

        print(
            "SEEDED RESET: PASS"
        )

        # ====================================================
        # 2. RANDOM EPISODES
        # ====================================================

        print()

        print(
            "2. RANDOM-ACTION EPISODES"
        )

        print(
            "-" * 90
        )

        all_starts = []

        all_rewards = []

        all_min_distances = []

        maximum_abs_residual_ratio = 0.0

        maximum_joint_clip_cost = 0.0

        maximum_joint_limit_violation = 0.0

        total_steps = 0

        total_terminated = 0

        total_truncated = 0

        for episode in range(
            NUM_EPISODES
        ):

            episode_seed = (
                BASE_SEED
                +
                episode
            )

            obs, reset_info = (
                env.reset(
                    seed=
                        episode_seed
                )
            )

            start_step = int(
                reset_info[
                    "start_step"
                ]
            )

            all_starts.append(
                start_step
            )

            if not (
                TRAIN_START_STEP
                <=
                start_step
                <=
                TRAIN_END_STEP
            ):
                raise RuntimeError(
                    "Invalid random start: "
                    f"{start_step}"
                )

            assert_finite(
                "reset observation",
                obs,
            )

            if not (
                env.observation_space
                .contains(
                    obs
                )
            ):
                raise RuntimeError(
                    "Reset observation is "
                    "outside observation_space."
                )

            episode_rewards = []

            episode_min_distances = []

            terminated = False

            truncated = False

            episode_step_count = 0

            while (
                not terminated
                and
                not truncated
            ):

                action = (
                    env.action_space
                    .sample()
                    .astype(
                        np.float32
                    )
                )

                if not (
                    env.action_space
                    .contains(
                        action
                    )
                ):
                    raise RuntimeError(
                        "Sampled action is "
                        "outside action_space."
                    )

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = (
                    env.step(
                        action
                    )
                )

                episode_step_count += 1
                total_steps += 1

                # --------------------------------------------
                # OBSERVATION CHECKS
                # --------------------------------------------

                assert_finite(
                    "observation",
                    next_obs,
                )

                if not (
                    env.observation_space
                    .contains(
                        next_obs
                    )
                ):
                    raise RuntimeError(
                        "Observation is outside "
                        "observation_space."
                    )

                # --------------------------------------------
                # REWARD CHECKS
                # --------------------------------------------

                assert_finite(
                    "reward",
                    reward,
                )

                assert_finite(
                    "signed_distances",
                    info[
                        "signed_distances"
                    ],
                )

                assert_finite(
                    "residual",
                    info[
                        "residual"
                    ],
                )

                episode_rewards.append(
                    float(
                        reward
                    )
                )

                all_rewards.append(
                    float(
                        reward
                    )
                )

                minimum_distance = float(
                    np.min(
                        info[
                            "signed_distances"
                        ]
                    )
                )

                episode_min_distances.append(
                    minimum_distance
                )

                all_min_distances.append(
                    minimum_distance
                )

                # --------------------------------------------
                # RESIDUAL BOUNDS
                # --------------------------------------------

                residual = np.asarray(
                    info[
                        "residual"
                    ],
                    dtype=np.float64,
                )

                residual_ratio = (
                    np.abs(
                        residual
                    )
                    /
                    env.residual_scales
                )

                max_ratio = float(
                    np.max(
                        residual_ratio
                    )
                )

                maximum_abs_residual_ratio = max(
                    maximum_abs_residual_ratio,
                    max_ratio,
                )

                if max_ratio > (
                    1.0
                    +
                    1e-6
                ):
                    raise RuntimeError(
                        "Smoothed residual exceeded "
                        "configured residual scales."
                    )

                # --------------------------------------------
                # JOINT LIMITS
                # --------------------------------------------

                applied_joints = (
                    np.asarray(
                        env.current_metadata[
                            "applied_joints"
                        ][
                            :12
                        ],
                        dtype=np.float64,
                    )
                )

                lower_violation = np.maximum(
                    env.joint_lower
                    -
                    applied_joints,
                    0.0,
                )

                upper_violation = np.maximum(
                    applied_joints
                    -
                    env.joint_upper,
                    0.0,
                )

                violation = float(
                    max(
                        np.max(
                            lower_violation
                        ),
                        np.max(
                            upper_violation
                        ),
                    )
                )

                maximum_joint_limit_violation = max(
                    maximum_joint_limit_violation,
                    violation,
                )

                if violation > 1e-7:
                    raise RuntimeError(
                        "Applied joint pose exceeded "
                        "mechanical limits."
                    )

                # --------------------------------------------
                # CLIPPING
                # --------------------------------------------

                clip_cost = float(
                    info[
                        "joint_clip_cost"
                    ]
                )

                maximum_joint_clip_cost = max(
                    maximum_joint_clip_cost,
                    clip_cost,
                )

                if clip_cost > 1e-7:
                    raise RuntimeError(
                        "Unexpected joint clipping "
                        f"during bounded random action: "
                        f"{clip_cost}"
                    )

                # --------------------------------------------
                # CURRENT STEP RANGE
                # --------------------------------------------

                current_step = int(
                    info[
                        "step"
                    ]
                )

                if not (
                    TRAIN_START_STEP
                    <=
                    current_step
                    <=
                    TRAIN_END_STEP
                ):
                    raise RuntimeError(
                        "Environment step outside "
                        "approved training range: "
                        f"{current_step}"
                    )

                obs = next_obs

            if terminated:
                total_terminated += 1

            if truncated:
                total_truncated += 1

            print(
                f"episode={episode + 1:02d} "
                f"seed={episode_seed} "
                f"start={start_step:03d} "
                f"steps={episode_step_count:03d} "
                f"reward_mean="
                f"{np.mean(episode_rewards):+.5f} "
                f"reward_min="
                f"{np.min(episode_rewards):+.5f} "
                f"reward_max="
                f"{np.max(episode_rewards):+.5f} "
                f"worst_distance="
                f"{np.min(episode_min_distances) * 1000:+.2f}mm "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

        # ====================================================
        # 3. RANDOM-START COVERAGE
        # ====================================================

        print()

        print(
            "3. RANDOM-START COVERAGE"
        )

        print(
            "-" * 90
        )

        starts = np.asarray(
            all_starts,
            dtype=np.int32,
        )

        print(
            "starts:",
            starts.tolist(),
        )

        print(
            "minimum start:",
            int(
                np.min(
                    starts
                )
            ),
        )

        print(
            "maximum start:",
            int(
                np.max(
                    starts
                )
            ),
        )

        print(
            "unique starts:",
            int(
                len(
                    np.unique(
                        starts
                    )
                )
            ),
        )

        if (
            len(
                np.unique(
                    starts
                )
            )
            <=
            1
        ):
            raise RuntimeError(
                "Random-start logic did not "
                "produce varied start states."
            )

        # ====================================================
        # 4. GLOBAL SUMMARY
        # ====================================================

        rewards = np.asarray(
            all_rewards,
            dtype=np.float64,
        )

        distances = np.asarray(
            all_min_distances,
            dtype=np.float64,
        )

        print()

        print(
            "=" * 90
        )

        print(
            "STRESS TEST SUMMARY"
        )

        print(
            "=" * 90
        )

        print(
            "Episodes:",
            NUM_EPISODES,
        )

        print(
            "Total environment steps:",
            total_steps,
        )

        print(
            "Terminated episodes:",
            total_terminated,
        )

        print(
            "Truncated episodes:",
            total_truncated,
        )

        print()

        print(
            "Reward:"
        )

        print(
            "  mean:",
            float(
                np.mean(
                    rewards
                )
            ),
        )

        print(
            "  std:",
            float(
                np.std(
                    rewards
                )
            ),
        )

        print(
            "  min:",
            float(
                np.min(
                    rewards
                )
            ),
        )

        print(
            "  max:",
            float(
                np.max(
                    rewards
                )
            ),
        )

        print()

        print(
            "Minimum signed terrain distance:"
        )

        print(
            "  mean:",
            f"{np.mean(distances) * 1000:+.3f} mm",
        )

        print(
            "  minimum:",
            f"{np.min(distances) * 1000:+.3f} mm",
        )

        print()

        print(
            "Maximum |residual / scale|:",
            maximum_abs_residual_ratio,
        )

        print(
            "Maximum joint clip cost:",
            maximum_joint_clip_cost,
        )

        print(
            "Maximum mechanical "
            "joint-limit violation:",
            maximum_joint_limit_violation,
        )

        print()

        print(
            "Observation checks: PASS"
        )

        print(
            "Finite-value checks: PASS"
        )

        print(
            "Residual-bound checks: PASS"
        )

        print(
            "Joint-limit checks: PASS"
        )

        print(
            "Episode truncation checks: PASS"
        )

        print(
            "Seeded reset check: PASS"
        )

        print(
            "Random-start check: PASS"
        )

        print()

        print(
            "=" * 90
        )

        print(
            "FINAL ENVIRONMENT STRESS TEST: PASSED"
        )

        print(
            "=" * 90
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No mujoco.mj_step() was called."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()