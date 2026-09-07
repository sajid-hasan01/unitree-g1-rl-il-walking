from __future__ import annotations

import numpy as np

from stable_baselines3.common.env_checker import check_env


# ============================================================
# REWARD-V3 BALANCED ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v3_balanced import (
    G1KinematicUnevenEnv,
    SEVERE_PENETRATION_THRESHOLD,
    SEVERE_PENETRATION_WEIGHT,
    PENETRATION_SCALE,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
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

    value = np.asarray(
        value
    )

    require(
        np.all(
            np.isfinite(
                value
            )
        ),
        f"Non-finite value in {name}.",
    )


def make_env():

    return G1KinematicUnevenEnv(
        episode_length=
            EPISODE_LENGTH,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


# ============================================================
# EXACT V3 REWARD CHECK
# ============================================================

def validate_v3_info(
    reward,
    info,
):

    signed = np.asarray(
        info[
            "signed_distances"
        ],
        dtype=np.float64,
    )

    require(
        signed.shape
        ==
        (8,),
        (
            "Expected 8 signed sole distances, "
            f"got {signed.shape}."
        ),
    )


    # --------------------------------------------------------
    # ABSOLUTE PENETRATION
    # --------------------------------------------------------

    actual_penetration = np.maximum(
        -signed,
        0.0,
    )

    expected_max_penetration = float(
        np.max(
            actual_penetration
        )
    )


    # --------------------------------------------------------
    # SEVERE EXCESS
    # --------------------------------------------------------

    expected_excess = float(
        max(
            expected_max_penetration
            -
            SEVERE_PENETRATION_THRESHOLD,
            0.0,
        )
    )


    # --------------------------------------------------------
    # SEVERE COST
    # --------------------------------------------------------

    expected_cost = float(
        (
            expected_excess
            /
            PENETRATION_SCALE
        )
        **
        2
    )


    # --------------------------------------------------------
    # SEVERE PENALTY
    # --------------------------------------------------------

    expected_penalty = float(
        SEVERE_PENETRATION_WEIGHT
        *
        expected_cost
    )


    # --------------------------------------------------------
    # FINAL REWARD
    # --------------------------------------------------------

    expected_reward = float(
        info[
            "reward_v2_before_severe_penalty"
        ]
        -
        expected_penalty
    )


    require(
        np.isclose(
            info[
                "max_actual_penetration_m"
            ],
            expected_max_penetration,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "max_actual_penetration_m "
            "does not match signed distances."
        ),
    )

    require(
        np.isclose(
            info[
                "severe_penetration_excess_m"
            ],
            expected_excess,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "V3 severe penetration excess "
            "formula mismatch."
        ),
    )

    require(
        np.isclose(
            info[
                "severe_penetration_cost"
            ],
            expected_cost,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "V3 severe penetration cost "
            "formula mismatch."
        ),
    )

    require(
        np.isclose(
            info[
                "severe_penetration_penalty"
            ],
            expected_penalty,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "V3 weighted severe penetration "
            "penalty mismatch."
        ),
    )

    require(
        np.isclose(
            reward,
            expected_reward,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "Returned V3 reward does not match "
            "V2 reward minus severe penalty."
        ),
    )

    require(
        np.isclose(
            info[
                "reward"
            ],
            reward,
            rtol=1e-7,
            atol=1e-10,
        ),
        (
            "info['reward'] does not match "
            "returned V3 reward."
        ),
    )

    require(
        np.isclose(
            info[
                "severe_penetration_threshold_m"
            ],
            SEVERE_PENETRATION_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        ),
        "V3 threshold diagnostic mismatch.",
    )

    require(
        np.isclose(
            info[
                "severe_penetration_weight"
            ],
            SEVERE_PENETRATION_WEIGHT,
            rtol=0.0,
            atol=1e-12,
        ),
        "V3 weight diagnostic mismatch.",
    )


# ============================================================
# TEST 1
#
# KNOWN PENALIZED / UNPENALIZED FRAMES
# ============================================================

def test_known_frames():

    print(
        "[1/6] Known-frame V3 reward formula..."
    )

    cases = [
        (
            394,
            395,
            True,
        ),

        (
            623,
            624,
            False,
        ),
    ]

    for (
        reset_step,
        expected_step,
        expect_penalty,
    ) in cases:

        env = make_env()

        try:

            observation, info = env.reset(
                seed=SEED,
                options={
                    "start_step":
                        reset_step,
                },
            )

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

            require(
                int(
                    info[
                        "step"
                    ]
                )
                ==
                expected_step,
                (
                    "Unexpected test step: "
                    f"{info['step']}"
                ),
            )

            validate_v3_info(
                reward,
                info,
            )

            penalty = float(
                info[
                    "severe_penetration_penalty"
                ]
            )

            if expect_penalty:

                require(
                    penalty
                    >
                    0.0,
                    (
                        "Expected severe penalty "
                        "but received zero."
                    ),
                )

            else:

                require(
                    np.isclose(
                        penalty,
                        0.0,
                        atol=1e-12,
                    ),
                    (
                        "Expected zero severe penalty."
                    ),
                )

            print(
                f"  step={expected_step:3d} | "
                f"penetration="
                f"{info['max_actual_penetration_m'] * 1000:7.3f} mm | "
                f"V2={info['reward_v2_before_severe_penalty']:+.6f} | "
                f"severe={penalty:.6f} | "
                f"V3={reward:+.6f} | "
                "PASS"
            )

        finally:

            env.close()

    print(
        "  PASS"
    )


# ============================================================
# TEST 2
#
# BALANCED RANDOM START COVERAGE
# ============================================================

def test_random_start_coverage():

    print()
    print(
        "[2/6] Balanced random-start coverage..."
    )

    env = make_env()

    starts = []

    try:

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
        "Random start below configured minimum.",
    )

    require(
        np.all(
            starts
            <=
            BALANCED_START_MAX
        ),
        "Random start above configured maximum.",
    )

    above_630 = int(
        np.count_nonzero(
            starts
            >
            630
        )
    )

    above_700 = int(
        np.count_nonzero(
            starts
            >
            700
        )
    )

    above_730 = int(
        np.count_nonzero(
            starts
            >
            730
        )
    )

    unique_count = int(
        np.unique(
            starts
        ).size
    )

    require(
        above_630
        >
        0,
        "No starts above 630.",
    )

    require(
        above_700
        >
        0,
        "No starts above 700.",
    )

    require(
        above_730
        >
        0,
        "No starts above 730.",
    )

    require(
        unique_count
        >
        100,
        (
            "Unexpectedly low random-start "
            "diversity."
        ),
    )

    print(
        "  samples:",
        len(
            starts
        ),
    )

    print(
        "  min/max:",
        (
            f"{np.min(starts)} / "
            f"{np.max(starts)}"
        ),
    )

    print(
        "  unique:",
        unique_count,
    )

    print(
        "  starts >630:",
        above_630,
    )

    print(
        "  starts >700:",
        above_700,
    )

    print(
        "  starts >730:",
        above_730,
    )

    print(
        "  PASS"
    )


# ============================================================
# TEST 3
#
# LATE EPISODE TRUNCATION
# ============================================================

def test_late_truncation():

    print()
    print(
        "[3/6] Late-start truncation..."
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

                validate_v3_info(
                    reward,
                    info,
                )

                transition_count += 1

                require(
                    terminated
                    is False,
                    (
                        "Unexpected physical "
                        "termination."
                    ),
                )

                require(
                    transition_count
                    <=
                    EPISODE_LENGTH,
                    (
                        "Episode exceeded configured "
                        "maximum length."
                    ),
                )

            require(
                truncated
                is True,
                "Expected truncation.",
            )

            require(
                int(
                    info[
                        "step"
                    ]
                )
                ==
                TRAIN_END_STEP,
                (
                    "Late episode did not finish "
                    "at TRAIN_END_STEP."
                ),
            )

            require(
                transition_count
                ==
                expected_transitions,
                (
                    f"start={start_step}: expected "
                    f"{expected_transitions} transitions, "
                    f"got {transition_count}."
                ),
            )

            print(
                f"  start={start_step:3d} | "
                f"transitions={transition_count:3d} | "
                "PASS"
            )

        finally:

            env.close()

    print(
        "  PASS"
    )


# ============================================================
# TEST 4
#
# RANDOM ACTION REWARD STRESS
# ============================================================

def test_random_action_stress():

    print()
    print(
        "[4/6] Random-action V3 reward stress..."
    )

    env = make_env()

    rng = np.random.default_rng(
        SEED
    )

    total_transitions = 0

    terminated_count = 0

    truncated_count = 0

    severe_active_count = 0

    min_reward = np.inf

    max_reward = -np.inf

    max_penetration = 0.0

    max_severe_penalty = 0.0

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
                        "Generated random action "
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

                validate_v3_info(
                    reward,
                    info,
                )

                require_finite(
                    "observation",
                    observation,
                )

                require_finite(
                    "reward",
                    reward,
                )

                require_finite(
                    "residual",
                    info[
                        "residual"
                    ],
                )

                require_finite(
                    "signed_distances",
                    info[
                        "signed_distances"
                    ],
                )

                require(
                    env.observation_space.contains(
                        observation
                    ),
                    (
                        "Observation outside "
                        "observation space."
                    ),
                )

                if (
                    info[
                        "severe_penetration_penalty"
                    ]
                    >
                    0.0
                ):

                    severe_active_count += 1

                min_reward = min(
                    min_reward,
                    float(
                        reward
                    ),
                )

                max_reward = max(
                    max_reward,
                    float(
                        reward
                    ),
                )

                max_penetration = max(
                    max_penetration,
                    float(
                        info[
                            "max_actual_penetration_m"
                        ]
                    ),
                )

                max_severe_penalty = max(
                    max_severe_penalty,
                    float(
                        info[
                            "severe_penetration_penalty"
                        ]
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

                total_transitions += 1

                episode_steps += 1

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
                "Unexpected kinematic "
                "termination occurred."
            ),
        )

        require(
            truncated_count
            ==
            RANDOM_ACTION_EPISODES,
            (
                "Not every episode ended by "
                "truncation."
            ),
        )

        require(
            severe_active_count
            >
            0,
            (
                "Random stress never activated "
                "the V3 severe penalty."
            ),
        )

        print(
            "  episodes:",
            RANDOM_ACTION_EPISODES,
        )

        print(
            "  total transitions:",
            total_transitions,
        )

        print(
            "  severe-penalty active transitions:",
            severe_active_count,
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
            "  maximum penetration:",
            (
                f"{max_penetration * 1000:.3f} mm"
            ),
        )

        print(
            "  maximum severe penalty:",
            f"{max_severe_penalty:.6f}",
        )

        print(
            "  maximum joint clip cost:",
            f"{max_clip_cost:.9f}",
        )

        print(
            "  PASS"
        )

    finally:

        env.close()


# ============================================================
# TEST 5
#
# V3 NEVER INCREASES V2 REWARD
# ============================================================

def test_v3_monotonic_penalty():

    print()
    print(
        "[5/6] V3 reward monotonicity..."
    )

    env = make_env()

    rng = np.random.default_rng(
        SEED
        +
        1000
    )

    checked = 0

    active = 0

    try:

        observation, info = env.reset(
            seed=SEED
        )

        while checked < 1000:

            action = rng.uniform(
                low=-1.0,
                high=1.0,
                size=12,
            ).astype(
                np.float32
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

            validate_v3_info(
                reward,
                info,
            )

            reward_v2 = float(
                info[
                    "reward_v2_before_severe_penalty"
                ]
            )

            penalty = float(
                info[
                    "severe_penetration_penalty"
                ]
            )

            require(
                reward
                <=
                reward_v2
                +
                1e-12,
                (
                    "V3 reward became greater than "
                    "V2 reward."
                ),
            )

            if penalty == 0.0:

                require(
                    np.isclose(
                        reward,
                        reward_v2,
                        atol=1e-12,
                    ),
                    (
                        "Zero severe penalty did not "
                        "preserve exact V2 reward."
                    ),
                )

            else:

                active += 1

                require(
                    reward
                    <
                    reward_v2,
                    (
                        "Positive severe penalty did "
                        "not lower reward."
                    ),
                )

            checked += 1

            if (
                terminated
                or
                truncated
            ):

                observation, info = env.reset()

        require(
            active
            >
            0,
            (
                "No active severe-penalty samples "
                "encountered."
            ),
        )

        print(
            "  transitions checked:",
            checked,
        )

        print(
            "  severe penalty active:",
            active,
        )

        print(
            "  PASS"
        )

    finally:

        env.close()


# ============================================================
# TEST 6
#
# SB3 ENVIRONMENT CHECKER
# ============================================================

def test_check_env():

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
        "=" * 110
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "REWARD-V3 BALANCED ENVIRONMENT STRESS TEST"
    )

    print(
        "=" * 110
    )

    print(
        "Train range:",
        (
            f"{TRAIN_START_STEP} "
            f"-> {TRAIN_END_STEP}"
        ),
    )

    print(
        "Balanced start range:",
        (
            f"{BALANCED_START_MIN} "
            f"-> {BALANCED_START_MAX}"
        ),
    )

    print(
        "Episode maximum:",
        EPISODE_LENGTH,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print()

    print(
        "Reward V3 addition:"
    )

    print(
        "  severe threshold:",
        (
            f"{SEVERE_PENETRATION_THRESHOLD * 1000:.1f} mm"
        ),
    )

    print(
        "  severe weight:",
        SEVERE_PENETRATION_WEIGHT,
    )

    print(
        "  normalization scale:",
        (
            f"{PENETRATION_SCALE * 1000:.1f} mm"
        ),
    )

    print()

    print(
        "Reward V2 otherwise unchanged."
    )

    print(
        "Observation/action spaces unchanged."
    )

    print(
        "Residual bounds unchanged."
    )

    print(
        "Balanced sampling unchanged."
    )

    print(
        "No physics -- inherited mj_forward-only "
        "kinematic environment."
    )

    print(
        "=" * 110
    )

    test_known_frames()

    test_random_start_coverage()

    test_late_truncation()

    test_random_action_stress()

    test_v3_monotonic_penalty()

    test_check_env()

    print()
    print(
        "=" * 110
    )

    print(
        "REWARD-V3 BALANCED STRESS TEST PASSED"
    )

    print(
        "=" * 110
    )

    print(
        "No PPO training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "If all six tests pass, Reward V3 is ready "
        "for one controlled 8192-step PPO "
        "verification run."
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()