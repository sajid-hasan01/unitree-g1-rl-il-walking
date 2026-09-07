from __future__ import annotations

import numpy as np

from stable_baselines3.common.env_checker import check_env

from envs.g1_kinematic_uneven_env_v2_balanced import (
    G1KinematicUnevenEnv,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
)

from envs.g1_kinematic_uneven_env_v2 import (
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 425

EPISODE_LENGTH = 120

RESIDUAL_SMOOTHING = 0.35

RANDOM_START_SAMPLES = 600

RANDOM_ACTION_EPISODES = 30


# ============================================================
# HELPERS
# ============================================================

def require(
    condition,
    message,
):

    if not condition:

        raise RuntimeError(
            message
        )


def require_finite(
    name,
    value,
):

    array = np.asarray(
        value
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        f"Non-finite value detected in {name}.",
    )


def make_env():

    return G1KinematicUnevenEnv(
        episode_length=
            EPISODE_LENGTH,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


# ============================================================
# TEST 1
#
# SEEDED RESET DETERMINISM
# ============================================================

def test_seeded_reset_determinism():

    print(
        "[1/6] Seeded reset determinism..."
    )

    env_a = make_env()
    env_b = make_env()

    try:

        obs_a, info_a = env_a.reset(
            seed=SEED
        )

        obs_b, info_b = env_b.reset(
            seed=SEED
        )

        start_a = int(
            info_a[
                "start_step"
            ]
        )

        start_b = int(
            info_b[
                "start_step"
            ]
        )

        obs_difference = float(
            np.max(
                np.abs(
                    obs_a
                    -
                    obs_b
                )
            )
        )

        require(
            start_a
            ==
            start_b,
            (
                "Same seed produced different "
                "balanced start steps."
            ),
        )

        require(
            obs_difference
            <=
            1e-12,
            (
                "Same seed produced different "
                "initial observations."
            ),
        )

        require(
            info_a[
                "balanced_random_start"
            ]
            is True,
            (
                "Balanced-start diagnostic flag "
                "missing."
            ),
        )

        print(
            "  start A/B:",
            f"{start_a}/{start_b}",
        )

        print(
            "  max observation difference:",
            f"{obs_difference:.12g}",
        )

        print(
            "  PASS"
        )

    finally:

        env_a.close()
        env_b.close()


# ============================================================
# TEST 2
#
# RANDOM-START COVERAGE
# ============================================================

def test_random_start_coverage():

    print()
    print(
        "[2/6] Balanced random-start coverage..."
    )

    env = make_env()

    starts = []

    try:

        # ----------------------------------------------------
        # One explicit initial seed, then allow the environment
        # RNG to continue naturally exactly as it would during
        # PPO training.
        # ----------------------------------------------------

        _, info = env.reset(
            seed=SEED
        )

        starts.append(
            int(
                info[
                    "start_step"
                ]
            )
        )

        for _ in range(
            RANDOM_START_SAMPLES
            -
            1
        ):

            _, info = env.reset()

            starts.append(
                int(
                    info[
                        "start_step"
                    ]
                )
            )

    finally:

        env.close()

    starts = np.asarray(
        starts,
        dtype=np.int64,
    )

    require(
        np.all(
            starts
            >=
            BALANCED_START_MIN
        ),
        (
            "Balanced sampler produced a start "
            "below its configured minimum."
        ),
    )

    require(
        np.all(
            starts
            <=
            BALANCED_START_MAX
        ),
        (
            "Balanced sampler produced a start "
            "above its configured maximum."
        ),
    )

    after_old_limit = int(
        np.count_nonzero(
            starts
            >
            630
        )
    )

    after_700 = int(
        np.count_nonzero(
            starts
            >
            700
        )
    )

    after_730 = int(
        np.count_nonzero(
            starts
            >
            730
        )
    )

    unique_starts = int(
        np.unique(
            starts
        ).size
    )

    require(
        after_old_limit
        >
        0,
        (
            "No starts above the original "
            "step-630 limit were observed."
        ),
    )

    require(
        after_700
        >
        0,
        (
            "No starts above step 700 were "
            "observed."
        ),
    )

    require(
        after_730
        >
        0,
        (
            "No starts above step 730 were "
            "observed."
        ),
    )

    require(
        unique_starts
        >
        100,
        (
            "Unexpectedly low start-step diversity."
        ),
    )

    print(
        "  samples:",
        len(
            starts
        ),
    )

    print(
        "  min:",
        int(
            np.min(
                starts
            )
        ),
    )

    print(
        "  max:",
        int(
            np.max(
                starts
            )
        ),
    )

    print(
        "  unique starts:",
        unique_starts,
    )

    print(
        "  starts > 630:",
        after_old_limit,
    )

    print(
        "  starts > 700:",
        after_700,
    )

    print(
        "  starts > 730:",
        after_730,
    )

    print(
        "  PASS"
    )


# ============================================================
# TEST 3
#
# EXPLICIT STARTS
# ============================================================

def test_explicit_starts():

    print()
    print(
        "[3/6] Explicit start-step behavior..."
    )

    requested_steps = [
        TRAIN_START_STEP,
        395,
        630,
        631,
        682,
        700,
        741,
        BALANCED_START_MAX,
    ]

    env = make_env()

    try:

        for requested_step in (
            requested_steps
        ):

            observation, info = env.reset(
                seed=SEED,
                options={
                    "start_step":
                        requested_step,
                },
            )

            actual_step = int(
                info[
                    "start_step"
                ]
            )

            require(
                actual_step
                ==
                requested_step,
                (
                    "Explicit reset mismatch: "
                    f"requested={requested_step}, "
                    f"actual={actual_step}"
                ),
            )

            require_finite(
                "explicit-start observation",
                observation,
            )

            require(
                env.observation_space.contains(
                    observation
                ),
                (
                    "Explicit-start observation "
                    "outside observation space."
                ),
            )

            print(
                f"  requested {requested_step:3d} "
                f"-> actual {actual_step:3d}"
            )

        print(
            "  PASS"
        )

    finally:

        env.close()


# ============================================================
# TEST 4
#
# LATE-START TRUNCATION
# ============================================================

def test_late_start_truncation():

    print()
    print(
        "[4/6] Late-start episode truncation..."
    )

    cases = [
        (
            630,
            120,
        ),

        (
            631,
            119,
        ),

        (
            682,
            68,
        ),

        (
            700,
            50,
        ),

        (
            741,
            9,
        ),

        (
            749,
            1,
        ),
    ]

    for (
        start_step,
        expected_transitions,
    ) in cases:

        env = make_env()

        try:

            observation, info = env.reset(
                seed=SEED,
                options={
                    "start_step":
                        start_step,
                },
            )

            transition_count = 0

            terminated = False
            truncated = False

            while not (
                terminated
                or
                truncated
            ):

                action = np.zeros(
                    12,
                    dtype=np.float32,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                transition_count += 1

                require_finite(
                    "late-start observation",
                    observation,
                )

                require_finite(
                    "late-start reward",
                    reward,
                )

                require(
                    terminated
                    is False,
                    (
                        "Kinematic environment "
                        "unexpectedly terminated."
                    ),
                )

                require(
                    transition_count
                    <=
                    EPISODE_LENGTH,
                    (
                        "Episode exceeded configured "
                        "episode length."
                    ),
                )

            final_step = int(
                info[
                    "step"
                ]
            )

            require(
                truncated
                is True,
                (
                    "Expected truncation did not occur."
                ),
            )

            require(
                final_step
                ==
                TRAIN_END_STEP,
                (
                    "Late episode did not truncate at "
                    f"TRAIN_END_STEP={TRAIN_END_STEP}. "
                    f"Actual={final_step}"
                ),
            )

            require(
                transition_count
                ==
                expected_transitions,
                (
                    "Unexpected transition count for "
                    f"start={start_step}: "
                    f"expected={expected_transitions}, "
                    f"actual={transition_count}"
                ),
            )

            print(
                f"  start={start_step:3d} | "
                f"transitions={transition_count:3d} | "
                f"final={final_step:3d} | "
                "PASS"
            )

        finally:

            env.close()

    print(
        "  PASS"
    )


# ============================================================
# TEST 5
#
# RANDOM ACTION STRESS
# ============================================================

def test_random_action_stress():

    print()
    print(
        "[5/6] Random-action stress..."
    )

    env = make_env()

    rng = np.random.default_rng(
        SEED
    )

    total_steps = 0

    late_episode_count = 0

    truncated_count = 0

    terminated_count = 0

    min_reward = np.inf
    max_reward = -np.inf

    max_abs_observation = 0.0
    max_residual = 0.0
    max_clip_cost = 0.0

    try:

        for episode_index in range(
            RANDOM_ACTION_EPISODES
        ):

            if episode_index == 0:

                observation, info = env.reset(
                    seed=SEED
                )

            else:

                observation, info = env.reset()

            start_step = int(
                info[
                    "start_step"
                ]
            )

            if start_step > 630:

                late_episode_count += 1

            require_finite(
                "random-reset observation",
                observation,
            )

            require(
                env.observation_space.contains(
                    observation
                ),
                (
                    "Random-reset observation outside "
                    "observation space."
                ),
            )

            done = False

            episode_steps = 0

            while not done:

                action = rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=12,
                ).astype(
                    np.float32
                )

                require(
                    env.action_space.contains(
                        action
                    ),
                    (
                        "Generated stress-test action "
                        "outside action space."
                    ),
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                require_finite(
                    "random-step observation",
                    observation,
                )

                require_finite(
                    "random-step reward",
                    reward,
                )

                require(
                    env.observation_space.contains(
                        observation
                    ),
                    (
                        "Random-step observation outside "
                        "observation space."
                    ),
                )

                require_finite(
                    "signed distances",
                    info[
                        "signed_distances"
                    ],
                )

                require_finite(
                    "flat-reference distances",
                    info[
                        "flat_reference_distances"
                    ],
                )

                require_finite(
                    "residual",
                    info[
                        "residual"
                    ],
                )

                reward_value = float(
                    reward
                )

                min_reward = min(
                    min_reward,
                    reward_value,
                )

                max_reward = max(
                    max_reward,
                    reward_value,
                )

                max_abs_observation = max(
                    max_abs_observation,
                    float(
                        np.max(
                            np.abs(
                                observation
                            )
                        )
                    ),
                )

                max_residual = max(
                    max_residual,
                    float(
                        np.max(
                            np.abs(
                                info[
                                    "residual"
                                ]
                            )
                        )
                    ),
                )

                max_clip_cost = max(
                    max_clip_cost,
                    float(
                        info[
                            "joint_clip_cost"
                        ]
                    ),
                )

                total_steps += 1
                episode_steps += 1

                require(
                    episode_steps
                    <=
                    EPISODE_LENGTH,
                    (
                        "Random episode exceeded "
                        "episode-length limit."
                    ),
                )

                if terminated:

                    terminated_count += 1

                if truncated:

                    truncated_count += 1

                done = bool(
                    terminated
                    or
                    truncated
                )

        require(
            terminated_count
            ==
            0,
            (
                "Kinematic environment produced "
                "unexpected physical termination."
            ),
        )

        require(
            truncated_count
            ==
            RANDOM_ACTION_EPISODES,
            (
                "Not every random episode ended "
                "by truncation."
            ),
        )

        require(
            late_episode_count
            >
            0,
            (
                "Random-action stress did not include "
                "any start above step 630."
            ),
        )

        print(
            "  episodes:",
            RANDOM_ACTION_EPISODES,
        )

        print(
            "  total transitions:",
            total_steps,
        )

        print(
            "  starts > 630:",
            late_episode_count,
        )

        print(
            "  terminated:",
            terminated_count,
        )

        print(
            "  truncated:",
            truncated_count,
        )

        print(
            "  reward min/max:",
            (
                f"{min_reward:+.6f} / "
                f"{max_reward:+.6f}"
            ),
        )

        print(
            "  max |observation|:",
            f"{max_abs_observation:.6f}",
        )

        print(
            "  max |residual|:",
            f"{max_residual:.6f} rad",
        )

        print(
            "  max joint clip cost:",
            f"{max_clip_cost:.9f}",
        )

        print(
            "  PASS"
        )

    finally:

        env.close()


# ============================================================
# TEST 6
#
# STABLE-BASELINES3 CHECK_ENV
# ============================================================

def test_sb3_check_env():

    print()
    print(
        "[6/6] Stable-Baselines3 check_env..."
    )

    env = make_env()

    try:

        check_env(
            env,
            warn=True,
            skip_render_check=True,
        )

        print(
            "  PASS"
        )

    finally:

        env.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 105
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "REWARD-V2 BALANCED-START ENVIRONMENT STRESS TEST"
    )

    print(
        "=" * 105
    )

    print(
        "Training range:",
        (
            f"{TRAIN_START_STEP} "
            f"-> {TRAIN_END_STEP}"
        ),
    )

    print(
        "Balanced random-start range:",
        (
            f"{BALANCED_START_MIN} "
            f"-> {BALANCED_START_MAX}"
        ),
    )

    print(
        "Episode length:",
        EPISODE_LENGTH,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print(
        "Random-start coverage samples:",
        RANDOM_START_SAMPLES,
    )

    print(
        "Random-action episodes:",
        RANDOM_ACTION_EPISODES,
    )

    print()

    print(
        "Reward changes: NONE"
    )

    print(
        "Observation/action changes: NONE"
    )

    print(
        "Residual-bound changes: NONE"
    )

    print(
        "Physics: NONE -- inherited Reward-V2 "
        "mj_forward-only environment"
    )

    print(
        "=" * 105
    )

    test_seeded_reset_determinism()

    test_random_start_coverage()

    test_explicit_starts()

    test_late_start_truncation()

    test_random_action_stress()

    test_sb3_check_env()

    print()
    print(
        "=" * 105
    )

    print(
        "BALANCED-START V2 STRESS TEST PASSED"
    )

    print(
        "=" * 105
    )

    print(
        "No PPO training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "If all six tests passed, the balanced-start "
        "environment is ready for a controlled PPO "
        "smoke-training experiment."
    )

    print(
        "=" * 105
    )


if __name__ == "__main__":
    main()