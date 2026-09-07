from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# EXISTING REWARD-V1 PPO CHECKPOINT
#
# We are NOT training this model.
#
# We use its deterministic actions inside Reward V2 so we can
# measure how a candidate foot-shape penalty would score the
# already-observed bad behavior.
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_smoke_seed425.zip"
)


SEED = 425

RESIDUAL_SMOOTHING = 0.35


# ============================================================
# VERIFIED FOOT-SPHERE ORDER
#
# 0,1 = left heel
# 2,3 = left toe
# 4,5 = right heel
# 6,7 = right toe
# ============================================================

LEFT_HEEL = np.array(
    [0, 1],
    dtype=np.int64,
)

LEFT_TOE = np.array(
    [2, 3],
    dtype=np.int64,
)

RIGHT_HEEL = np.array(
    [4, 5],
    dtype=np.int64,
)

RIGHT_TOE = np.array(
    [6, 7],
    dtype=np.int64,
)


# ============================================================
# CANDIDATE FOOT-SHAPE SETTINGS
#
# These are deliberately being SWEPT.
#
# They are NOT measured physical constants.
# ============================================================

TOLERANCE_MM_VALUES = [
    0.0,
    5.0,
    10.0,
    20.0,
]

FOOT_SHAPE_SCALE = 0.050

WEIGHT_VALUES = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
    6.0,
    8.0,
    10.0,
]


# Frames already known to contain useful diagnostic behavior.

SELECTED_STEPS = [
    435,
    436,
    437,
    624,
    708,
    716,
    720,
    723,
    728,
]


# ============================================================
# MODEL / ENV VALIDATION
# ============================================================

def validate_model_spaces(
    model,
    env,
):

    if (
        model.observation_space.shape
        !=
        env.observation_space.shape
    ):
        raise RuntimeError(
            "Saved PPO observation-space shape "
            "does not match Reward-V2 environment."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):
        raise RuntimeError(
            "Saved PPO action-space shape "
            "does not match Reward-V2 environment."
        )


# ============================================================
# FOOT GEOMETRY
# ============================================================

def heel_to_toe_gap(
    distances,
    heel_indices,
    toe_indices,
):

    heel_mean = np.mean(
        distances[
            heel_indices
        ]
    )

    toe_mean = np.mean(
        distances[
            toe_indices
        ]
    )

    return float(
        heel_mean
        -
        toe_mean
    )


def foot_shape_cost(
    current,
    flat_reference,
    tolerance_m,
):

    # --------------------------------------------------------
    # LEFT FOOT
    # --------------------------------------------------------

    left_gap = heel_to_toe_gap(
        current,
        LEFT_HEEL,
        LEFT_TOE,
    )

    left_reference_gap = heel_to_toe_gap(
        flat_reference,
        LEFT_HEEL,
        LEFT_TOE,
    )

    left_excess = max(
        left_gap
        -
        left_reference_gap
        -
        tolerance_m,
        0.0,
    )

    # --------------------------------------------------------
    # RIGHT FOOT
    # --------------------------------------------------------

    right_gap = heel_to_toe_gap(
        current,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    right_reference_gap = heel_to_toe_gap(
        flat_reference,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    right_excess = max(
        right_gap
        -
        right_reference_gap
        -
        tolerance_m,
        0.0,
    )

    # --------------------------------------------------------
    # NORMALIZED COST
    # --------------------------------------------------------

    cost = float(
        np.mean(
            np.square(
                np.array(
                    [
                        left_excess,
                        right_excess,
                    ],
                    dtype=np.float64,
                )
                /
                FOOT_SHAPE_SCALE
            )
        )
    )

    return {
        "cost":
            cost,

        "left_gap":
            left_gap,

        "left_reference_gap":
            left_reference_gap,

        "left_excess":
            left_excess,

        "right_gap":
            right_gap,

        "right_reference_gap":
            right_reference_gap,

        "right_excess":
            right_excess,
    }


# ============================================================
# ROLLOUT
# ============================================================

def run_rollout(
    env,
    model,
    use_ppo,
):

    observation, _ = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP,
        },
    )

    records = []

    while (
        env.current_step
        <
        TRAIN_END_STEP
    ):

        if use_ppo:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

        else:

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

        records.append(
            {
                "step":
                    int(
                        info[
                            "step"
                        ]
                    ),

                "world_x":
                    float(
                        info[
                            "world_x"
                        ]
                    ),

                "base_reward":
                    float(
                        reward
                    ),

                "distances":
                    np.asarray(
                        info[
                            "signed_distances"
                        ],
                        dtype=np.float64,
                    ).copy(),

                "flat_reference":
                    np.asarray(
                        info[
                            "flat_reference_distances"
                        ],
                        dtype=np.float64,
                    ).copy(),

                "penetration_cost":
                    float(
                        info[
                            "penetration_cost"
                        ]
                    ),
            }
        )

        if terminated:
            raise RuntimeError(
                "Unexpected termination during "
                "kinematic deterministic rollout."
            )

        if (
            truncated
            and
            env.current_step
            <
            TRAIN_END_STEP
        ):
            raise RuntimeError(
                "Environment truncated before "
                "TRAIN_END_STEP."
            )

    return records


# ============================================================
# ARRAY CONVERSION
# ============================================================

def records_to_arrays(
    records,
):

    return {
        "step":
            np.asarray(
                [
                    record[
                        "step"
                    ]
                    for record
                    in records
                ],
                dtype=np.int64,
            ),

        "world_x":
            np.asarray(
                [
                    record[
                        "world_x"
                    ]
                    for record
                    in records
                ],
                dtype=np.float64,
            ),

        "base_reward":
            np.asarray(
                [
                    record[
                        "base_reward"
                    ]
                    for record
                    in records
                ],
                dtype=np.float64,
            ),

        "distances":
            np.asarray(
                [
                    record[
                        "distances"
                    ]
                    for record
                    in records
                ],
                dtype=np.float64,
            ),

        "flat_reference":
            np.asarray(
                [
                    record[
                        "flat_reference"
                    ]
                    for record
                    in records
                ],
                dtype=np.float64,
            ),

        "penetration_cost":
            np.asarray(
                [
                    record[
                        "penetration_cost"
                    ]
                    for record
                    in records
                ],
                dtype=np.float64,
            ),
    }


# ============================================================
# FAIR-COMPARISON CHECK
# ============================================================

def validate_fair_comparison(
    bc,
    ppo,
):

    if not np.array_equal(
        bc[
            "step"
        ],
        ppo[
            "step"
        ],
    ):
        raise RuntimeError(
            "BC/PPO step arrays differ."
        )

    max_x_difference = float(
        np.max(
            np.abs(
                bc[
                    "world_x"
                ]
                -
                ppo[
                    "world_x"
                ]
            )
        )
    )

    max_flat_difference = float(
        np.max(
            np.abs(
                bc[
                    "flat_reference"
                ]
                -
                ppo[
                    "flat_reference"
                ]
            )
        )
    )

    print()
    print(
        "Fair-comparison invariants:"
    )

    print(
        "  transitions:",
        len(
            bc[
                "step"
            ]
        ),
    )

    print(
        "  max world-X difference:",
        f"{max_x_difference:.12f}",
    )

    print(
        "  max flat-reference difference:",
        f"{max_flat_difference:.12f}",
    )

    if max_x_difference > 1e-12:
        raise RuntimeError(
            "BC/PPO world-X mismatch."
        )

    if max_flat_difference > 1e-12:
        raise RuntimeError(
            "BC/PPO flat-reference mismatch."
        )


# ============================================================
# COMPUTE COST ARRAYS
# ============================================================

def compute_shape_arrays(
    rollout,
    tolerance_m,
):

    count = len(
        rollout[
            "step"
        ]
    )

    cost = np.zeros(
        count,
        dtype=np.float64,
    )

    left_gap = np.zeros(
        count,
        dtype=np.float64,
    )

    left_reference_gap = np.zeros(
        count,
        dtype=np.float64,
    )

    left_excess = np.zeros(
        count,
        dtype=np.float64,
    )

    right_gap = np.zeros(
        count,
        dtype=np.float64,
    )

    right_reference_gap = np.zeros(
        count,
        dtype=np.float64,
    )

    right_excess = np.zeros(
        count,
        dtype=np.float64,
    )

    for index in range(
        count
    ):

        result = foot_shape_cost(
            rollout[
                "distances"
            ][
                index
            ],

            rollout[
                "flat_reference"
            ][
                index
            ],

            tolerance_m,
        )

        cost[
            index
        ] = result[
            "cost"
        ]

        left_gap[
            index
        ] = result[
            "left_gap"
        ]

        left_reference_gap[
            index
        ] = result[
            "left_reference_gap"
        ]

        left_excess[
            index
        ] = result[
            "left_excess"
        ]

        right_gap[
            index
        ] = result[
            "right_gap"
        ]

        right_reference_gap[
            index
        ] = result[
            "right_reference_gap"
        ]

        right_excess[
            index
        ] = result[
            "right_excess"
        ]

    return {
        "cost":
            cost,

        "left_gap":
            left_gap,

        "left_reference_gap":
            left_reference_gap,

        "left_excess":
            left_excess,

        "right_gap":
            right_gap,

        "right_reference_gap":
            right_reference_gap,

        "right_excess":
            right_excess,
    }


# ============================================================
# REWARD COMPARISON
# ============================================================

def compare_rewards(
    bc_base_reward,
    ppo_base_reward,
    bc_shape_cost,
    ppo_shape_cost,
    weight,
):

    bc_adjusted = (
        bc_base_reward
        -
        weight
        *
        bc_shape_cost
    )

    ppo_adjusted = (
        ppo_base_reward
        -
        weight
        *
        ppo_shape_cost
    )

    difference = (
        ppo_adjusted
        -
        bc_adjusted
    )

    epsilon = 1e-9

    better = int(
        np.count_nonzero(
            difference
            >
            epsilon
        )
    )

    same = int(
        np.count_nonzero(
            np.abs(
                difference
            )
            <=
            epsilon
        )
    )

    worse = int(
        np.count_nonzero(
            difference
            <
            -
            epsilon
        )
    )

    return {
        "bc_mean":
            float(
                np.mean(
                    bc_adjusted
                )
            ),

        "ppo_mean":
            float(
                np.mean(
                    ppo_adjusted
                )
            ),

        "delta":
            float(
                np.mean(
                    difference
                )
            ),

        "better":
            better,

        "same":
            same,

        "worse":
            worse,

        "bc_adjusted":
            bc_adjusted,

        "ppo_adjusted":
            ppo_adjusted,
    }


# ============================================================
# SELECTED FRAME REPORT
# ============================================================

def print_selected_frames(
    bc,
    ppo,
    bc_shape,
    ppo_shape,
):

    print()
    print(
        "Selected diagnostic frames:"
    )

    print(
        "step | x       | "
        "BC baseR | PPO baseR | "
        "L PPO gap | L ref gap | "
        "L excess | "
        "BC shape | PPO shape"
    )

    print(
        "-" * 115
    )

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            ppo[
                "step"
            ]
            ==
            requested_step
        )[
            0
        ]

        if len(
            matches
        ) == 0:

            print(
                f"{requested_step:4d} | "
                "not present"
            )

            continue

        index = int(
            matches[
                0
            ]
        )

        print(
            f"{requested_step:4d} | "
            f"{ppo['world_x'][index]:+7.3f} | "
            f"{bc['base_reward'][index]:+8.4f} | "
            f"{ppo['base_reward'][index]:+9.4f} | "
            f"{ppo_shape['left_gap'][index] * 1000:+9.2f} | "
            f"{ppo_shape['left_reference_gap'][index] * 1000:+9.2f} | "
            f"{ppo_shape['left_excess'][index] * 1000:8.2f} | "
            f"{bc_shape['cost'][index]:8.4f} | "
            f"{ppo_shape['cost'][index]:9.4f}"
        )


# ============================================================
# THRESHOLD WEIGHT
# ============================================================

def threshold_weight(
    bc_reward,
    ppo_reward,
    bc_cost,
    ppo_cost,
):

    reward_advantage = (
        ppo_reward
        -
        bc_reward
    )

    cost_difference = (
        ppo_cost
        -
        bc_cost
    )

    if reward_advantage <= 0.0:

        return (
            "already BC-preferred"
        )

    if cost_difference <= 0.0:

        return (
            "cannot flip with positive weight"
        )

    threshold = (
        reward_advantage
        /
        cost_difference
    )

    return (
        f"{threshold:.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Reward-V1 PPO model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 110
    )

    print(
        "UNITREE G1 REWARD-V2 FOOT-SHAPE PENALTY OFFLINE SWEEP"
    )

    print(
        "=" * 110
    )

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Environment:"
    )

    print(
        "  Reward V2"
    )

    print(
        "  no PPO training"
    )

    print(
        "  no mj_step()"
    )

    print(
        "  deterministic V1 policy actions"
    )

    print()

    print(
        "Candidate foot-shape definition:"
    )

    print(
        "  gap = mean(heel distances) "
        "- mean(toe distances)"
    )

    print(
        "  excess = max("
        "current_gap - flat_BC_gap - tolerance, 0)"
    )

    print(
        "  positive excess means PPO is "
        "MORE toe-down than same-phase flat BC"
    )

    print()

    print(
        "Foot-shape normalization scale:"
    )

    print(
        f"  {FOOT_SHAPE_SCALE * 1000:.1f} mm"
    )

    print(
        "=" * 110
    )

    bc_env = (
        G1KinematicUnevenEnv(
            episode_length=
                (
                    TRAIN_END_STEP
                    -
                    TRAIN_START_STEP
                    +
                    1
                ),

            random_start=False,

            residual_smoothing=
                RESIDUAL_SMOOTHING,
        )
    )

    ppo_env = (
        G1KinematicUnevenEnv(
            episode_length=
                (
                    TRAIN_END_STEP
                    -
                    TRAIN_START_STEP
                    +
                    1
                ),

            random_start=False,

            residual_smoothing=
                RESIDUAL_SMOOTHING,
        )
    )

    try:

        print()
        print(
            "Loading Reward-V1 PPO..."
        )

        model = PPO.load(
            MODEL_PATH,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            model,
            ppo_env,
        )

        print(
            "Running BC-only trajectory "
            "inside Reward V2..."
        )

        bc_records = run_rollout(
            bc_env,
            model,
            use_ppo=False,
        )

        print(
            "Running deterministic V1 PPO "
            "inside Reward V2..."
        )

        ppo_records = run_rollout(
            ppo_env,
            model,
            use_ppo=True,
        )

        bc = records_to_arrays(
            bc_records
        )

        ppo = records_to_arrays(
            ppo_records
        )

        validate_fair_comparison(
            bc,
            ppo,
        )

        print()
        print(
            "Current Reward-V2 base reward:"
        )

        print(
            "  BC mean:",
            f"{np.mean(bc['base_reward']):+.6f}",
        )

        print(
            "  PPO mean:",
            f"{np.mean(ppo['base_reward']):+.6f}",
        )

        print(
            "  PPO - BC:",
            (
                f"{np.mean(ppo['base_reward']) - np.mean(bc['base_reward']):+.6f}"
            ),
        )

        print()
        print(
            "Current Reward-V2 penetration cost:"
        )

        print(
            "  BC mean:",
            f"{np.mean(bc['penetration_cost']):.6f}",
        )

        print(
            "  PPO mean:",
            f"{np.mean(ppo['penetration_cost']):.6f}",
        )

        # ====================================================
        # TOLERANCE SWEEP
        # ====================================================

        for tolerance_mm in (
            TOLERANCE_MM_VALUES
        ):

            tolerance_m = (
                tolerance_mm
                /
                1000.0
            )

            bc_shape = compute_shape_arrays(
                bc,
                tolerance_m,
            )

            ppo_shape = compute_shape_arrays(
                ppo,
                tolerance_m,
            )

            print()
            print(
                "=" * 110
            )

            print(
                "TOLERANCE:",
                f"{tolerance_mm:.1f} mm",
            )

            print(
                "=" * 110
            )

            print(
                "Mean foot-shape cost:"
            )

            print(
                "  BC :",
                f"{np.mean(bc_shape['cost']):.6f}",
            )

            print(
                "  PPO:",
                f"{np.mean(ppo_shape['cost']):.6f}",
            )

            print()

            print(
                "Mean LEFT extra toe-down:"
            )

            print(
                "  BC :",
                (
                    f"{np.mean(bc_shape['left_excess']) * 1000:.3f} mm"
                ),
            )

            print(
                "  PPO:",
                (
                    f"{np.mean(ppo_shape['left_excess']) * 1000:.3f} mm"
                ),
            )

            print()

            print(
                "Mean RIGHT extra toe-down:"
            )

            print(
                "  BC :",
                (
                    f"{np.mean(bc_shape['right_excess']) * 1000:.3f} mm"
                ),
            )

            print(
                "  PPO:",
                (
                    f"{np.mean(ppo_shape['right_excess']) * 1000:.3f} mm"
                ),
            )

            print()

            print(
                "weight | BC meanR | PPO meanR | "
                "deltaR   | PPO better | same | PPO worse"
            )

            print(
                "-" * 90
            )

            for weight in (
                WEIGHT_VALUES
            ):

                result = compare_rewards(
                    bc[
                        "base_reward"
                    ],

                    ppo[
                        "base_reward"
                    ],

                    bc_shape[
                        "cost"
                    ],

                    ppo_shape[
                        "cost"
                    ],

                    weight,
                )

                print(
                    f"{weight:6.2f} | "
                    f"{result['bc_mean']:+8.4f} | "
                    f"{result['ppo_mean']:+9.4f} | "
                    f"{result['delta']:+8.4f} | "
                    f"{result['better']:10d} | "
                    f"{result['same']:4d} | "
                    f"{result['worse']:9d}"
                )

            print_selected_frames(
                bc,
                ppo,
                bc_shape,
                ppo_shape,
            )

            print()
            print(
                "Minimum shape weight needed to make "
                "PPO no better than BC at selected frames:"
            )

            print(
                "step | threshold"
            )

            print(
                "-" * 50
            )

            for requested_step in (
                SELECTED_STEPS
            ):

                matches = np.where(
                    ppo[
                        "step"
                    ]
                    ==
                    requested_step
                )[
                    0
                ]

                if len(
                    matches
                ) == 0:

                    continue

                index = int(
                    matches[
                        0
                    ]
                )

                threshold = threshold_weight(
                    bc[
                        "base_reward"
                    ][
                        index
                    ],

                    ppo[
                        "base_reward"
                    ][
                        index
                    ],

                    bc_shape[
                        "cost"
                    ][
                        index
                    ],

                    ppo_shape[
                        "cost"
                    ][
                        index
                    ],
                )

                print(
                    f"{requested_step:4d} | "
                    f"{threshold}"
                )

        print()
        print(
            "=" * 110
        )

        print(
            "OFFLINE SWEEP COMPLETE"
        )

        print(
            "=" * 110
        )

        print(
            "No training was performed."
        )

        print(
            "No environment file was modified."
        )

        print(
            "No PPO model was modified."
        )

        print(
            "Review the tolerance/weight sweep "
            "before adding the foot-shape term "
            "to Reward V2."
        )

        print(
            "=" * 110
        )

    finally:

        bc_env.close()
        ppo_env.close()


if __name__ == "__main__":
    main()