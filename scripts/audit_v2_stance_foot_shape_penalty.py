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
# EXISTING REWARD-V1 PPO
#
# THIS SCRIPT DOES NOT TRAIN IT.
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_smoke_seed425.zip"
)


SEED = 425
RESIDUAL_SMOOTHING = 0.35

FOOT_SHAPE_SCALE = 0.050


# ============================================================
# VERIFIED FOOT SPHERE ORDER
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

LEFT_FOOT = np.array(
    [0, 1, 2, 3],
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

RIGHT_FOOT = np.array(
    [4, 5, 6, 7],
    dtype=np.int64,
)


# ============================================================
# CANDIDATE VALUES
#
# HYPERPARAMETERS ONLY.
# ============================================================

TOLERANCE_MM_VALUES = [
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
]

WEIGHT_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
]


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
# HELPERS
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
            "Observation-space mismatch."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):
        raise RuntimeError(
            "Action-space mismatch."
        )


def heel_to_toe_gap(
    distances,
    heel_indices,
    toe_indices,
):

    heel_mean = float(
        np.mean(
            distances[
                heel_indices
            ]
        )
    )

    toe_mean = float(
        np.mean(
            distances[
                toe_indices
            ]
        )
    )

    return (
        heel_mean
        -
        toe_mean
    )


def reference_lower_side(
    flat_reference,
):

    left_mean = float(
        np.mean(
            flat_reference[
                LEFT_FOOT
            ]
        )
    )

    right_mean = float(
        np.mean(
            flat_reference[
                RIGHT_FOOT
            ]
        )
    )

    if left_mean <= right_mean:
        return (
            "left",
            left_mean,
            right_mean,
        )

    return (
        "right",
        left_mean,
        right_mean,
    )


def stance_shape_cost(
    current,
    flat_reference,
    tolerance_m,
):

    (
        lower_side,
        flat_left_mean,
        flat_right_mean,
    ) = reference_lower_side(
        flat_reference
    )

    if lower_side == "left":

        current_gap = heel_to_toe_gap(
            current,
            LEFT_HEEL,
            LEFT_TOE,
        )

        reference_gap = heel_to_toe_gap(
            flat_reference,
            LEFT_HEEL,
            LEFT_TOE,
        )

    else:

        current_gap = heel_to_toe_gap(
            current,
            RIGHT_HEEL,
            RIGHT_TOE,
        )

        reference_gap = heel_to_toe_gap(
            flat_reference,
            RIGHT_HEEL,
            RIGHT_TOE,
        )

    excess = max(
        current_gap
        -
        reference_gap
        -
        tolerance_m,
        0.0,
    )

    cost = float(
        (
            excess
            /
            FOOT_SHAPE_SCALE
        )
        ** 2
    )

    return {
        "cost":
            cost,

        "side":
            lower_side,

        "current_gap":
            current_gap,

        "reference_gap":
            reference_gap,

        "excess":
            excess,

        "flat_left_mean":
            flat_left_mean,

        "flat_right_mean":
            flat_right_mean,
    }


# ============================================================
# ROLLOUT
# ============================================================

def rollout(
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

                "reward":
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
                "Unexpected termination."
            )

        if (
            truncated
            and
            env.current_step
            <
            TRAIN_END_STEP
        ):
            raise RuntimeError(
                "Unexpected early truncation."
            )

    return records


def to_arrays(
    records,
):

    return {
        "step":
            np.asarray(
                [
                    item[
                        "step"
                    ]
                    for item
                    in records
                ],
                dtype=np.int64,
            ),

        "world_x":
            np.asarray(
                [
                    item[
                        "world_x"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
            ),

        "reward":
            np.asarray(
                [
                    item[
                        "reward"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
            ),

        "distances":
            np.asarray(
                [
                    item[
                        "distances"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
            ),

        "flat_reference":
            np.asarray(
                [
                    item[
                        "flat_reference"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
            ),

        "penetration_cost":
            np.asarray(
                [
                    item[
                        "penetration_cost"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
            ),
    }


# ============================================================
# SHAPE ARRAYS
# ============================================================

def calculate_shape_arrays(
    rollout_data,
    tolerance_m,
):

    count = len(
        rollout_data[
            "step"
        ]
    )

    cost = np.zeros(
        count,
        dtype=np.float64,
    )

    gap = np.zeros(
        count,
        dtype=np.float64,
    )

    reference_gap = np.zeros(
        count,
        dtype=np.float64,
    )

    excess = np.zeros(
        count,
        dtype=np.float64,
    )

    side = []

    for index in range(
        count
    ):

        result = stance_shape_cost(
            rollout_data[
                "distances"
            ][
                index
            ],

            rollout_data[
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

        gap[
            index
        ] = result[
            "current_gap"
        ]

        reference_gap[
            index
        ] = result[
            "reference_gap"
        ]

        excess[
            index
        ] = result[
            "excess"
        ]

        side.append(
            result[
                "side"
            ]
        )

    return {
        "cost":
            cost,

        "gap":
            gap,

        "reference_gap":
            reference_gap,

        "excess":
            excess,

        "side":
            np.asarray(
                side,
                dtype=object,
            ),
    }


# ============================================================
# REWARD RESCORING
# ============================================================

def rescore(
    bc,
    ppo,
    bc_cost,
    ppo_cost,
    weight,
):

    bc_reward = (
        bc[
            "reward"
        ]
        -
        weight
        *
        bc_cost
    )

    ppo_reward = (
        ppo[
            "reward"
        ]
        -
        weight
        *
        ppo_cost
    )

    difference = (
        ppo_reward
        -
        bc_reward
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
                    bc_reward
                )
            ),

        "ppo_mean":
            float(
                np.mean(
                    ppo_reward
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
    }


def minimum_weight(
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

    if reward_advantage <= 0.0:
        return "already BC-preferred"

    cost_difference = (
        ppo_cost
        -
        bc_cost
    )

    if cost_difference <= 0.0:
        return "cannot flip"

    return (
        f"{reward_advantage / cost_difference:.6f}"
    )


# ============================================================
# FAIR COMPARISON
# ============================================================

def check_invariants(
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
            "Step mismatch."
        )

    world_x_difference = float(
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

    flat_difference = float(
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
        f"{world_x_difference:.12f}",
    )

    print(
        "  max flat-reference difference:",
        f"{flat_difference:.12f}",
    )

    if world_x_difference > 1e-12:
        raise RuntimeError(
            "World-X invariant failed."
        )

    if flat_difference > 1e-12:
        raise RuntimeError(
            "Flat-reference invariant failed."
        )


# ============================================================
# SELECTED FRAMES
# ============================================================

def print_selected_frames(
    bc,
    ppo,
    bc_shape,
    ppo_shape,
):

    print()
    print(
        "Selected frames:"
    )

    print(
        "step | side  | x       | "
        "BC R     | PPO R    | "
        "PPO gap | ref gap | excess | "
        "BC cost | PPO cost"
    )

    print(
        "-" * 110
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
        ) != 1:
            continue

        index = int(
            matches[
                0
            ]
        )

        print(
            f"{requested_step:4d} | "
            f"{str(ppo_shape['side'][index]):5s} | "
            f"{ppo['world_x'][index]:+7.3f} | "
            f"{bc['reward'][index]:+8.4f} | "
            f"{ppo['reward'][index]:+8.4f} | "
            f"{ppo_shape['gap'][index] * 1000:+7.2f} | "
            f"{ppo_shape['reference_gap'][index] * 1000:+7.2f} | "
            f"{ppo_shape['excess'][index] * 1000:6.2f} | "
            f"{bc_shape['cost'][index]:7.4f} | "
            f"{ppo_shape['cost'][index]:8.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Reward-V1 PPO checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 110
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "STANCE-GATED FOOT-SHAPE SWEEP"
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
        "No training."
    )

    print(
        "No mj_step()."
    )

    print(
        "No environment modification."
    )

    print()

    print(
        "Only the SAME-PHASE FLAT-BC "
        "LOWER FOOT receives shape cost."
    )

    print()

    print(
        "shape gap = heel mean - toe mean"
    )

    print(
        "excess = max("
        "current gap - flat-reference gap "
        "- tolerance, 0)"
    )

    print(
        "scale =",
        f"{FOOT_SHAPE_SCALE * 1000:.1f} mm",
    )

    print(
        "=" * 110
    )

    episode_length = (
        TRAIN_END_STEP
        -
        TRAIN_START_STEP
        +
        1
    )

    bc_env = G1KinematicUnevenEnv(
        episode_length=
            episode_length,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    ppo_env = G1KinematicUnevenEnv(
        episode_length=
            episode_length,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    try:

        print()
        print(
            "Loading PPO..."
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
            "Running BC-only in Reward V2..."
        )

        bc = to_arrays(
            rollout(
                bc_env,
                model,
                use_ppo=False,
            )
        )

        print(
            "Running deterministic V1 PPO "
            "in Reward V2..."
        )

        ppo = to_arrays(
            rollout(
                ppo_env,
                model,
                use_ppo=True,
            )
        )

        check_invariants(
            bc,
            ppo,
        )

        print()
        print(
            "Current V2 reward:"
        )

        print(
            "  BC mean:",
            f"{np.mean(bc['reward']):+.6f}",
        )

        print(
            "  PPO mean:",
            f"{np.mean(ppo['reward']):+.6f}",
        )

        print(
            "  PPO - BC:",
            (
                f"{np.mean(ppo['reward']) - np.mean(bc['reward']):+.6f}"
            ),
        )

        print()
        print(
            "Current penetration cost:"
        )

        print(
            "  BC mean:",
            f"{np.mean(bc['penetration_cost']):.6f}",
        )

        print(
            "  PPO mean:",
            f"{np.mean(ppo['penetration_cost']):.6f}",
        )

        for tolerance_mm in (
            TOLERANCE_MM_VALUES
        ):

            tolerance_m = (
                tolerance_mm
                /
                1000.0
            )

            bc_shape = calculate_shape_arrays(
                bc,
                tolerance_m,
            )

            ppo_shape = calculate_shape_arrays(
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
                "Mean stance-gated shape cost:"
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
                "Mean stance-foot extra toe-down:"
            )

            print(
                "  BC :",
                (
                    f"{np.mean(bc_shape['excess']) * 1000:.3f} mm"
                ),
            )

            print(
                "  PPO:",
                (
                    f"{np.mean(ppo_shape['excess']) * 1000:.3f} mm"
                ),
            )

            left_count = int(
                np.count_nonzero(
                    ppo_shape[
                        "side"
                    ]
                    ==
                    "left"
                )
            )

            right_count = int(
                np.count_nonzero(
                    ppo_shape[
                        "side"
                    ]
                    ==
                    "right"
                )
            )

            print()

            print(
                "Reference lower-foot counts:"
            )

            print(
                "  left :",
                left_count,
            )

            print(
                "  right:",
                right_count,
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

                result = rescore(
                    bc,
                    ppo,
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
                "Minimum stance-shape weight "
                "needed to make PPO no better "
                "than BC:"
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
                ) != 1:
                    continue

                index = int(
                    matches[
                        0
                    ]
                )

                result = minimum_weight(
                    bc[
                        "reward"
                    ][
                        index
                    ],

                    ppo[
                        "reward"
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
                    f"{result}"
                )

        print()
        print(
            "=" * 110
        )

        print(
            "STANCE-GATED SWEEP COMPLETE"
        )

        print(
            "=" * 110
        )

        print(
            "No training was performed."
        )

        print(
            "Do not modify Reward V2 until "
            "this output is reviewed."
        )

        print(
            "=" * 110
        )

    finally:

        bc_env.close()
        ppo_env.close()


if __name__ == "__main__":
    main()