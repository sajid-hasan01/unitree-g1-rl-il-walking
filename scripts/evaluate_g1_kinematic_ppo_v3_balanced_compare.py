from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# ORIGINAL REWARD-V2 ENVIRONMENT
#
# IMPORTANT:
#
# We deliberately evaluate ALL policies in the original
# Reward-V2 environment.
#
# V2 and V3 differ only in reward during training.
#
# Robot geometry, action application, observations, terrain,
# BC reference, root trajectory, residual smoothing and
# mj_forward-only execution are identical.
#
# Therefore using one environment gives a fair geometry
# comparison without training-start sampling affecting eval.
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
)


# ============================================================
# V3 REWARD HYPERPARAMETERS
#
# Used OFFLINE only to calculate the severe-penetration
# component for every evaluated policy.
# ============================================================

from envs.g1_kinematic_uneven_env_v3_balanced import (
    SEVERE_PENETRATION_THRESHOLD,
    SEVERE_PENETRATION_WEIGHT,
)


# ============================================================
# CHECKPOINTS
# ============================================================

ORIGINAL_V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)

BALANCED_V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_smoke_seed425.zip"
)

BALANCED_V3_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v3_balanced_verify_seed425.zip"
)


# ============================================================
# EVALUATION SETTINGS
# ============================================================

SEED = 425

RESIDUAL_SMOOTHING = 0.35

EVAL_STATES = (
    TRAIN_END_STEP
    -
    TRAIN_START_STEP
    +
    1
)

EVAL_TRANSITIONS = (
    TRAIN_END_STEP
    -
    TRAIN_START_STEP
)


# ============================================================
# SOLE GROUPS
# ============================================================

LEFT_HEEL = np.asarray(
    [0, 1],
    dtype=np.int64,
)

LEFT_TOE = np.asarray(
    [2, 3],
    dtype=np.int64,
)

RIGHT_HEEL = np.asarray(
    [4, 5],
    dtype=np.int64,
)

RIGHT_TOE = np.asarray(
    [6, 7],
    dtype=np.int64,
)


# ============================================================
# SEVERITY THRESHOLDS
# ============================================================

SEVERITY_THRESHOLDS_MM = [
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]


# ============================================================
# REGIONS
#
# Especially important:
#
# 486-495:
#   balanced V2 middle hotspot
#
# 529-533:
#   balanced V2 middle hotspot
#
# 675-700:
#   old V2 severe late failure
#
# 708-735:
#   late orientation / penetration region
#
# 730-733:
#   newest balanced-V2 severe hotspot
# ============================================================

REGIONS = [
    (
        "full",
        150,
        750,
    ),

    (
        "early",
        150,
        300,
    ),

    (
        "middle",
        301,
        649,
    ),

    (
        "late",
        650,
        750,
    ),

    (
        "middle_486_495",
        486,
        495,
    ),

    (
        "middle_529_533",
        529,
        533,
    ),

    (
        "old_failure_675_700",
        675,
        700,
    ),

    (
        "late_708_735",
        708,
        735,
    ),

    (
        "late_hotspot_730_733",
        730,
        733,
    ),
]


# ============================================================
# SELECTED STEPS
# ============================================================

SELECTED_STEPS = [
    395,
    435,

    486,
    487,
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,

    529,
    530,
    531,
    532,
    533,

    624,

    679,
    680,
    681,
    682,
    683,
    689,
    690,
    698,
    699,
    700,

    708,
    710,
    711,
    713,
    715,
    716,
    720,
    723,
    728,

    730,
    731,
    732,
    733,

    741,
    750,
]


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


def validate_model_spaces(
    model,
    env,
    name,
):

    require(
        model.observation_space.shape
        ==
        env.observation_space.shape,
        f"{name}: observation shape mismatch.",
    )

    require(
        model.action_space.shape
        ==
        env.action_space.shape,
        f"{name}: action shape mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.low,
            env.observation_space.low,
        ),
        f"{name}: observation lower bounds mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.high,
            env.observation_space.high,
        ),
        f"{name}: observation upper bounds mismatch.",
    )

    require(
        np.allclose(
            model.action_space.low,
            env.action_space.low,
        ),
        f"{name}: action lower bounds mismatch.",
    )

    require(
        np.allclose(
            model.action_space.high,
            env.action_space.high,
        ),
        f"{name}: action upper bounds mismatch.",
    )


# ============================================================
# V3 SEVERE COST
# ============================================================

def severe_penetration_metrics(
    distances,
):

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    actual_penetration = np.maximum(
        -distances,
        0.0,
    )

    max_actual_penetration = np.max(
        actual_penetration,
        axis=1,
    )

    severe_excess = np.maximum(
        max_actual_penetration
        -
        SEVERE_PENETRATION_THRESHOLD,
        0.0,
    )

    severe_cost = (
        severe_excess
        /
        PENETRATION_SCALE
    ) ** 2

    severe_penalty = (
        SEVERE_PENETRATION_WEIGHT
        *
        severe_cost
    )

    return {
        "max_actual_penetration":
            max_actual_penetration,

        "severe_excess":
            severe_excess,

        "severe_cost":
            severe_cost,

        "severe_penalty":
            severe_penalty,
    }


# ============================================================
# ROLLOUT
# ============================================================

def run_rollout(
    env,
    model=None,
):

    observation, reset_info = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP,
        },
    )

    require(
        int(
            reset_info[
                "start_step"
            ]
        )
        ==
        TRAIN_START_STEP,
        "Unexpected evaluation start.",
    )


    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    steps = [
        int(
            env.current_step
        )
    ]

    world_x = [
        float(
            env.current_metadata[
                "world_x"
            ]
        )
    ]

    distances = [
        env.current_distances
        .astype(
            np.float64
        )
        .copy()
    ]

    flat_reference = [
        env.flat_reference_distances[
            env.current_step
        ]
        .astype(
            np.float64
        )
        .copy()
    ]

    residuals = [
        env.previous_residual
        .astype(
            np.float64
        )
        .copy()
    ]

    actions = []

    reward_steps = []

    v2_rewards = []

    clip_costs = []


    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    for transition_index in range(
        EVAL_TRANSITIONS
    ):

        if model is None:

            action = np.zeros(
                12,
                dtype=np.float32,
            )

        else:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

        require(
            action.shape
            ==
            (12,),
            (
                "Unexpected action shape: "
                f"{action.shape}"
            ),
        )

        require(
            env.action_space.contains(
                action
            ),
            "Action outside action space.",
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
            np.all(
                np.isfinite(
                    observation
                )
            ),
            "Non-finite observation.",
        )

        require(
            np.isfinite(
                reward
            ),
            "Non-finite reward.",
        )

        require(
            terminated
            is False,
            (
                "Unexpected kinematic "
                f"termination at {info['step']}."
            ),
        )

        if (
            transition_index
            <
            EVAL_TRANSITIONS
            -
            1
        ):

            require(
                truncated
                is False,
                (
                    "Evaluation truncated too early "
                    f"at {info['step']}."
                ),
            )


        # ----------------------------------------------------
        # TRANSITION DATA
        # ----------------------------------------------------

        actions.append(
            action
            .astype(
                np.float64
            )
            .copy()
        )

        reward_steps.append(
            int(
                info[
                    "step"
                ]
            )
        )

        v2_rewards.append(
            float(
                reward
            )
        )

        clip_costs.append(
            float(
                info[
                    "joint_clip_cost"
                ]
            )
        )


        # ----------------------------------------------------
        # NEW STATE DATA
        # ----------------------------------------------------

        steps.append(
            int(
                info[
                    "step"
                ]
            )
        )

        world_x.append(
            float(
                info[
                    "world_x"
                ]
            )
        )

        distances.append(
            np.asarray(
                info[
                    "signed_distances"
                ],
                dtype=np.float64,
            ).copy()
        )

        flat_reference.append(
            np.asarray(
                info[
                    "flat_reference_distances"
                ],
                dtype=np.float64,
            ).copy()
        )

        residuals.append(
            np.asarray(
                info[
                    "residual"
                ],
                dtype=np.float64,
            ).copy()
        )


    result = {
        "steps":
            np.asarray(
                steps,
                dtype=np.int64,
            ),

        "world_x":
            np.asarray(
                world_x,
                dtype=np.float64,
            ),

        "distances":
            np.asarray(
                distances,
                dtype=np.float64,
            ),

        "flat":
            np.asarray(
                flat_reference,
                dtype=np.float64,
            ),

        "residuals":
            np.asarray(
                residuals,
                dtype=np.float64,
            ),

        "actions":
            np.asarray(
                actions,
                dtype=np.float64,
            ),

        "reward_steps":
            np.asarray(
                reward_steps,
                dtype=np.int64,
            ),

        "v2_rewards":
            np.asarray(
                v2_rewards,
                dtype=np.float64,
            ),

        "clip_costs":
            np.asarray(
                clip_costs,
                dtype=np.float64,
            ),
    }


    # --------------------------------------------------------
    # STRUCTURE VALIDATION
    # --------------------------------------------------------

    require(
        result[
            "steps"
        ].shape
        ==
        (
            EVAL_STATES,
        ),
        "Unexpected state count.",
    )

    require(
        result[
            "distances"
        ].shape
        ==
        (
            EVAL_STATES,
            8,
        ),
        "Unexpected distance shape.",
    )

    require(
        result[
            "residuals"
        ].shape
        ==
        (
            EVAL_STATES,
            12,
        ),
        "Unexpected residual shape.",
    )

    require(
        result[
            "actions"
        ].shape
        ==
        (
            EVAL_TRANSITIONS,
            12,
        ),
        "Unexpected action array shape.",
    )

    require(
        result[
            "v2_rewards"
        ].shape
        ==
        (
            EVAL_TRANSITIONS,
        ),
        "Unexpected reward count.",
    )


    # --------------------------------------------------------
    # OFFLINE V3 REWARD
    #
    # Rewards correspond to states 151..750, therefore use
    # distances[1:] for V3 severe penalty.
    # --------------------------------------------------------

    severe_transition = (
        severe_penetration_metrics(
            result[
                "distances"
            ][
                1:
            ]
        )
    )

    result[
        "v3_severe_penalties"
    ] = (
        severe_transition[
            "severe_penalty"
        ]
    )

    result[
        "offline_v3_rewards"
    ] = (
        result[
            "v2_rewards"
        ]
        -
        result[
            "v3_severe_penalties"
        ]
    )

    return result


# ============================================================
# GEOMETRY
# ============================================================

def geometry(
    rollout,
):

    current = rollout[
        "distances"
    ]

    flat = rollout[
        "flat"
    ]


    # --------------------------------------------------------
    # RAW ABSOLUTE PENETRATION
    # --------------------------------------------------------

    penetration = np.maximum(
        -current,
        0.0,
    )

    frame_penetration = np.max(
        penetration,
        axis=1,
    )

    positive_penetration = penetration[
        penetration
        >
        0.0
    ]


    # --------------------------------------------------------
    # SAME-PHASE DEFICIT
    # --------------------------------------------------------

    deficit = np.maximum(
        flat
        -
        current
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    frame_deficit = np.max(
        deficit,
        axis=1,
    )


    # --------------------------------------------------------
    # REWARD-V2 PHASE-AWARE PENETRATION
    # --------------------------------------------------------

    allowed_floor = np.minimum(
        flat,
        0.0,
    )

    phase_excess = np.maximum(
        allowed_floor
        -
        current
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    frame_phase_excess = np.max(
        phase_excess,
        axis=1,
    )

    phase_cost = np.mean(
        (
            phase_excess
            /
            PENETRATION_SCALE
        )
        **
        2,
        axis=1,
    )


    # --------------------------------------------------------
    # REWARD-V3 ABSOLUTE SEVERE PENETRATION
    # --------------------------------------------------------

    severe = severe_penetration_metrics(
        current
    )


    # --------------------------------------------------------
    # FOOT-SHAPE SECONDARY DIAGNOSTICS
    # --------------------------------------------------------

    left_gap = (
        np.mean(
            current[
                :,
                LEFT_HEEL
            ],
            axis=1,
        )
        -
        np.mean(
            current[
                :,
                LEFT_TOE
            ],
            axis=1,
        )
    )

    right_gap = (
        np.mean(
            current[
                :,
                RIGHT_HEEL
            ],
            axis=1,
        )
        -
        np.mean(
            current[
                :,
                RIGHT_TOE
            ],
            axis=1,
        )
    )

    flat_left_gap = (
        np.mean(
            flat[
                :,
                LEFT_HEEL
            ],
            axis=1,
        )
        -
        np.mean(
            flat[
                :,
                LEFT_TOE
            ],
            axis=1,
        )
    )

    flat_right_gap = (
        np.mean(
            flat[
                :,
                RIGHT_HEEL
            ],
            axis=1,
        )
        -
        np.mean(
            flat[
                :,
                RIGHT_TOE
            ],
            axis=1,
        )
    )


    # --------------------------------------------------------
    # SEVERITY COUNTS
    # --------------------------------------------------------

    severity_counts = {}

    for threshold_mm in (
        SEVERITY_THRESHOLDS_MM
    ):

        threshold_m = (
            threshold_mm
            /
            1000.0
        )

        severity_counts[
            threshold_mm
        ] = int(
            np.count_nonzero(
                frame_penetration
                >=
                threshold_m
            )
        )


    return {
        "penetration":
            penetration,

        "frame_penetration":
            frame_penetration,

        "deficit":
            deficit,

        "frame_deficit":
            frame_deficit,

        "phase_excess":
            phase_excess,

        "frame_phase_excess":
            frame_phase_excess,

        "phase_cost":
            phase_cost,

        "severe_excess":
            severe[
                "severe_excess"
            ],

        "severe_cost":
            severe[
                "severe_cost"
            ],

        "severe_penalty":
            severe[
                "severe_penalty"
            ],

        "left_gap":
            left_gap,

        "right_gap":
            right_gap,

        "left_extra":
            left_gap
            -
            flat_left_gap,

        "right_extra":
            right_gap
            -
            flat_right_gap,

        "penetrating_frames":
            int(
                np.count_nonzero(
                    frame_penetration
                    >
                    0.0
                )
            ),

        "penetrating_samples":
            int(
                np.count_nonzero(
                    penetration
                    >
                    0.0
                )
            ),

        "mean_penetration":
            (
                float(
                    np.mean(
                        positive_penetration
                    )
                )
                if
                positive_penetration.size
                else
                0.0
            ),

        "mean_frame_penetration":
            float(
                np.mean(
                    frame_penetration
                )
            ),

        "max_penetration":
            float(
                np.max(
                    penetration
                )
            ),

        "mean_deficit":
            float(
                np.mean(
                    deficit
                )
            ),

        "max_deficit":
            float(
                np.max(
                    deficit
                )
            ),

        "squared_deficit":
            float(
                np.sum(
                    deficit
                    **
                    2
                )
            ),

        "mean_phase_cost":
            float(
                np.mean(
                    phase_cost
                )
            ),

        "max_phase_excess":
            float(
                np.max(
                    phase_excess
                )
            ),

        "severe_active_frames":
            int(
                np.count_nonzero(
                    severe[
                        "severe_excess"
                    ]
                    >
                    0.0
                )
            ),

        "mean_severe_cost":
            float(
                np.mean(
                    severe[
                        "severe_cost"
                    ]
                )
            ),

        "max_severe_cost":
            float(
                np.max(
                    severe[
                        "severe_cost"
                    ]
                )
            ),

        "severity_counts":
            severity_counts,
    }


# ============================================================
# FAIR COMPARISON
# ============================================================

def validate_invariants(
    reference,
    candidate,
    name,
):

    step_diff = float(
        np.max(
            np.abs(
                reference[
                    "steps"
                ]
                -
                candidate[
                    "steps"
                ]
            )
        )
    )

    x_diff = float(
        np.max(
            np.abs(
                reference[
                    "world_x"
                ]
                -
                candidate[
                    "world_x"
                ]
            )
        )
    )

    flat_diff = float(
        np.max(
            np.abs(
                reference[
                    "flat"
                ]
                -
                candidate[
                    "flat"
                ]
            )
        )
    )

    print(
        f"{name:18s} | "
        f"stepDiff={step_diff:.1f} | "
        f"xDiff={x_diff:.12f} | "
        f"flatDiff={flat_diff:.12f}"
    )

    require(
        step_diff
        ==
        0.0,
        f"{name}: step invariant failed.",
    )

    require(
        x_diff
        <=
        1e-12,
        f"{name}: world-X invariant failed.",
    )

    require(
        flat_diff
        <=
        1e-12,
        f"{name}: flat-reference invariant failed.",
    )


# ============================================================
# GLOBAL TABLE
# ============================================================

def print_global_table(
    results,
):

    print()
    print(
        "=" * 190
    )

    print(
        "GLOBAL DETERMINISTIC GEOMETRY"
    )

    print(
        "=" * 190
    )

    print(
        "policy       | penFrm | penSamp | "
        "meanPen | maxPen | meanDef | maxDef | "
        "sqDef   | phaseCost | maxPhase | "
        "V3active | severeCost | V2R    | offlineV3R"
    )

    print(
        "-" * 190
    )

    for name in [
        "BC",
        "V2",
        "BALANCED_V2",
        "V3",
    ]:

        rollout = results[
            name
        ][
            "rollout"
        ]

        g = results[
            name
        ][
            "geometry"
        ]

        print(
            f"{name:12s} | "
            f"{g['penetrating_frames']:6d} | "
            f"{g['penetrating_samples']:7d} | "
            f"{g['mean_penetration'] * 1000:7.2f} | "
            f"{g['max_penetration'] * 1000:6.2f} | "
            f"{g['mean_deficit'] * 1000:7.2f} | "
            f"{g['max_deficit'] * 1000:6.2f} | "
            f"{g['squared_deficit']:7.4f} | "
            f"{g['mean_phase_cost']:9.5f} | "
            f"{g['max_phase_excess'] * 1000:8.2f} | "
            f"{g['severe_active_frames']:8d} | "
            f"{g['mean_severe_cost']:10.5f} | "
            f"{np.mean(rollout['v2_rewards']):+7.4f} | "
            f"{np.mean(rollout['offline_v3_rewards']):+10.4f}"
        )


# ============================================================
# V2 BALANCED -> V3 CHANGE
# ============================================================

def percent_change_reduction(
    old,
    new,
):

    old = float(
        old
    )

    new = float(
        new
    )

    if abs(
        old
    ) < 1e-12:

        return 0.0

    return (
        100.0
        *
        (
            old
            -
            new
        )
        /
        old
    )


def print_v3_change(
    results,
):

    old = results[
        "BALANCED_V2"
    ][
        "geometry"
    ]

    new = results[
        "V3"
    ][
        "geometry"
    ]

    print()
    print(
        "=" * 115
    )

    print(
        "BALANCED V2 -> BALANCED V3"
    )

    print(
        "=" * 115
    )

    metrics = [
        (
            "penetrating frames",
            old[
                "penetrating_frames"
            ],
            new[
                "penetrating_frames"
            ],
        ),

        (
            "penetrating samples",
            old[
                "penetrating_samples"
            ],
            new[
                "penetrating_samples"
            ],
        ),

        (
            "mean penetration",
            old[
                "mean_penetration"
            ],
            new[
                "mean_penetration"
            ],
        ),

        (
            "maximum penetration",
            old[
                "max_penetration"
            ],
            new[
                "max_penetration"
            ],
        ),

        (
            "mean deficit",
            old[
                "mean_deficit"
            ],
            new[
                "mean_deficit"
            ],
        ),

        (
            "maximum deficit",
            old[
                "max_deficit"
            ],
            new[
                "max_deficit"
            ],
        ),

        (
            "squared deficit",
            old[
                "squared_deficit"
            ],
            new[
                "squared_deficit"
            ],
        ),

        (
            "V2 phase cost",
            old[
                "mean_phase_cost"
            ],
            new[
                "mean_phase_cost"
            ],
        ),

        (
            "maximum phase excess",
            old[
                "max_phase_excess"
            ],
            new[
                "max_phase_excess"
            ],
        ),

        (
            "V3 severe active frames",
            old[
                "severe_active_frames"
            ],
            new[
                "severe_active_frames"
            ],
        ),

        (
            "mean V3 severe cost",
            old[
                "mean_severe_cost"
            ],
            new[
                "mean_severe_cost"
            ],
        ),
    ]

    for (
        label,
        original,
        v3,
    ) in metrics:

        print(
            f"{label:31s}: "
            f"{percent_change_reduction(original, v3):+8.2f}%"
        )


# ============================================================
# SEVERITY
# ============================================================

def print_severity(
    results,
):

    print()
    print(
        "=" * 100
    )

    print(
        "MAX RAW-PENETRATION SEVERITY COUNTS"
    )

    print(
        "=" * 100
    )

    print(
        "threshold | BC | Original V2 | Balanced V2 | V3"
    )

    print(
        "-" * 75
    )

    for threshold in (
        SEVERITY_THRESHOLDS_MM
    ):

        print(
            f"{threshold:8.1f} | "
            f"{results['BC']['geometry']['severity_counts'][threshold]:3d} | "
            f"{results['V2']['geometry']['severity_counts'][threshold]:11d} | "
            f"{results['BALANCED_V2']['geometry']['severity_counts'][threshold]:11d} | "
            f"{results['V3']['geometry']['severity_counts'][threshold]:3d}"
        )


# ============================================================
# REGIONS
# ============================================================

def print_regions(
    results,
):

    print()
    print(
        "=" * 170
    )

    print(
        "REGION-SPECIFIC PENETRATION"
    )

    print(
        "=" * 170
    )

    print(
        "region                 | policy       | "
        "penFrm | maxPen | meanMax | phaseCost | "
        "maxPhase | severeAct | severeCost | meanDef"
    )

    print(
        "-" * 170
    )

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        for name in [
            "BC",
            "V2",
            "BALANCED_V2",
            "V3",
        ]:

            rollout = results[
                name
            ][
                "rollout"
            ]

            g = results[
                name
            ][
                "geometry"
            ]

            mask = (
                (
                    rollout[
                        "steps"
                    ]
                    >=
                    start
                )
                &
                (
                    rollout[
                        "steps"
                    ]
                    <=
                    end
                )
            )

            frame_pen = (
                g[
                    "frame_penetration"
                ][
                    mask
                ]
            )

            phase_cost = (
                g[
                    "phase_cost"
                ][
                    mask
                ]
            )

            frame_phase = (
                g[
                    "frame_phase_excess"
                ][
                    mask
                ]
            )

            severe_excess = (
                g[
                    "severe_excess"
                ][
                    mask
                ]
            )

            severe_cost = (
                g[
                    "severe_cost"
                ][
                    mask
                ]
            )

            deficit = (
                g[
                    "deficit"
                ][
                    mask
                ]
            )

            print(
                f"{region_name:22s} | "
                f"{name:12s} | "
                f"{np.count_nonzero(frame_pen > 0.0):6d} | "
                f"{np.max(frame_pen) * 1000:6.2f} | "
                f"{np.mean(frame_pen) * 1000:7.2f} | "
                f"{np.mean(phase_cost):9.5f} | "
                f"{np.max(frame_phase) * 1000:8.2f} | "
                f"{np.count_nonzero(severe_excess > 0.0):9d} | "
                f"{np.mean(severe_cost):10.5f} | "
                f"{np.mean(deficit) * 1000:7.2f}"
            )

        print(
            "-" * 170
        )


# ============================================================
# SELECTED FRAMES
# ============================================================

def print_selected_steps(
    results,
):

    print()
    print(
        "=" * 175
    )

    print(
        "SELECTED-FRAME PENETRATION"
    )

    print(
        "=" * 175
    )

    print(
        "step | BC pen | V2 pen | B-V2 pen | V3 pen | "
        "B-V2 phase | V3 phase | "
        "B-V2 sev | V3 sev | "
        "B-V2 Lgap | V3 Lgap"
    )

    print(
        "-" * 175
    )

    bc_steps = results[
        "BC"
    ][
        "rollout"
    ][
        "steps"
    ]

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            bc_steps
            ==
            requested_step
        )[
            0
        ]

        if len(
            matches
        ) != 1:

            continue

        i = int(
            matches[
                0
            ]
        )

        bc = results[
            "BC"
        ][
            "geometry"
        ]

        old = results[
            "V2"
        ][
            "geometry"
        ]

        balanced = results[
            "BALANCED_V2"
        ][
            "geometry"
        ]

        v3 = results[
            "V3"
        ][
            "geometry"
        ]

        print(
            f"{requested_step:4d} | "
            f"{bc['frame_penetration'][i] * 1000:6.1f} | "
            f"{old['frame_penetration'][i] * 1000:6.1f} | "
            f"{balanced['frame_penetration'][i] * 1000:8.1f} | "
            f"{v3['frame_penetration'][i] * 1000:6.1f} | "
            f"{balanced['frame_phase_excess'][i] * 1000:10.1f} | "
            f"{v3['frame_phase_excess'][i] * 1000:8.1f} | "
            f"{balanced['severe_penalty'][i]:8.3f} | "
            f"{v3['severe_penalty'][i]:6.3f} | "
            f"{balanced['left_gap'][i] * 1000:+10.1f} | "
            f"{v3['left_gap'][i] * 1000:+8.1f}"
        )


# ============================================================
# WORST V3 FRAMES
# ============================================================

def print_worst_v3(
    results,
    count=25,
):

    v3 = results[
        "V3"
    ][
        "geometry"
    ]

    balanced = results[
        "BALANCED_V2"
    ][
        "geometry"
    ]

    rollout = results[
        "V3"
    ][
        "rollout"
    ]

    order = np.argsort(
        v3[
            "frame_penetration"
        ]
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 135
    )

    print(
        "WORST REMAINING V3 PENETRATION FRAMES"
    )

    print(
        "=" * 135
    )

    print(
        "rank | step | x       | "
        "Balanced V2 | V3 pen | change | "
        "V3 phase | severe penalty"
    )

    print(
        "-" * 135
    )

    for (
        rank,
        i,
    ) in enumerate(
        order,
        start=1,
    ):

        difference = (
            balanced[
                "frame_penetration"
            ][
                i
            ]
            -
            v3[
                "frame_penetration"
            ][
                i
            ]
        )

        print(
            f"{rank:4d} | "
            f"{rollout['steps'][i]:4d} | "
            f"{rollout['world_x'][i]:+7.3f} | "
            f"{balanced['frame_penetration'][i] * 1000:11.2f} | "
            f"{v3['frame_penetration'][i] * 1000:6.2f} | "
            f"{difference * 1000:+7.2f} | "
            f"{v3['frame_phase_excess'][i] * 1000:8.2f} | "
            f"{v3['severe_penalty'][i]:14.4f}"
        )


# ============================================================
# PER-FRAME V3 VS BALANCED V2
# ============================================================

def print_per_frame_outcome(
    results,
):

    old = results[
        "BALANCED_V2"
    ][
        "geometry"
    ][
        "frame_penetration"
    ]

    new = results[
        "V3"
    ][
        "geometry"
    ][
        "frame_penetration"
    ]

    delta = (
        old
        -
        new
    )

    tolerance = 1e-9

    better = int(
        np.count_nonzero(
            delta
            >
            tolerance
        )
    )

    same = int(
        np.count_nonzero(
            np.abs(
                delta
            )
            <=
            tolerance
        )
    )

    worse = int(
        np.count_nonzero(
            delta
            <
            -
            tolerance
        )
    )

    print()
    print(
        "=" * 105
    )

    print(
        "PER-FRAME MAX-PENETRATION OUTCOME: V3 VS BALANCED V2"
    )

    print(
        "=" * 105
    )

    print(
        "V3 better:",
        better,
    )

    print(
        "Same:",
        same,
    )

    print(
        "V3 worse:",
        worse,
    )


# ============================================================
# POLICY DIAGNOSTICS
# ============================================================

def print_policy_diagnostics(
    results,
):

    print()
    print(
        "=" * 140
    )

    print(
        "POLICY RESIDUAL DIAGNOSTICS"
    )

    print(
        "=" * 140
    )

    print(
        "policy       | mean|act| | max|act| | "
        "mean|res| | max|res| | "
        "L hip | L knee | L ankleP | "
        "R hip | R knee | R ankleP | maxClip"
    )

    print(
        "-" * 140
    )

    for name in [
        "V2",
        "BALANCED_V2",
        "V3",
    ]:

        rollout = results[
            name
        ][
            "rollout"
        ]

        residual = rollout[
            "residuals"
        ][
            1:
        ]

        action = rollout[
            "actions"
        ]

        print(
            f"{name:12s} | "
            f"{np.mean(np.abs(action)):9.5f} | "
            f"{np.max(np.abs(action)):8.5f} | "
            f"{np.mean(np.abs(residual)):9.5f} | "
            f"{np.max(np.abs(residual)):8.5f} | "
            f"{np.mean(residual[:, 0]):+6.3f} | "
            f"{np.mean(residual[:, 3]):+6.3f} | "
            f"{np.mean(residual[:, 4]):+8.3f} | "
            f"{np.mean(residual[:, 6]):+6.3f} | "
            f"{np.mean(residual[:, 9]):+6.3f} | "
            f"{np.mean(residual[:, 10]):+8.3f} | "
            f"{np.max(rollout['clip_costs']):7.5f}"
        )


# ============================================================
# DECISION SUMMARY
# ============================================================

def print_decision_summary(
    results,
):

    b = results[
        "BALANCED_V2"
    ][
        "geometry"
    ]

    v = results[
        "V3"
    ][
        "geometry"
    ]

    print()
    print(
        "=" * 120
    )

    print(
        "V3 VERIFICATION DECISION DATA"
    )

    print(
        "=" * 120
    )

    print(
        "Balanced-V2 maximum penetration:",
        f"{b['max_penetration'] * 1000:.3f} mm",
    )

    print(
        "V3 maximum penetration:",
        f"{v['max_penetration'] * 1000:.3f} mm",
    )

    print()

    print(
        "Balanced-V2 >=40 mm frames:",
        b[
            "severity_counts"
        ][
            40.0
        ],
    )

    print(
        "V3 >=40 mm frames:",
        v[
            "severity_counts"
        ][
            40.0
        ],
    )

    print()

    print(
        "Balanced-V2 >=50 mm frames:",
        b[
            "severity_counts"
        ][
            50.0
        ],
    )

    print(
        "V3 >=50 mm frames:",
        v[
            "severity_counts"
        ][
            50.0
        ],
    )

    print()

    print(
        "Balanced-V2 >=60 mm frames:",
        b[
            "severity_counts"
        ][
            60.0
        ],
    )

    print(
        "V3 >=60 mm frames:",
        v[
            "severity_counts"
        ][
            60.0
        ],
    )

    print()

    print(
        "Balanced-V2 V3-severe active frames:",
        b[
            "severe_active_frames"
        ],
    )

    print(
        "V3 V3-severe active frames:",
        v[
            "severe_active_frames"
        ],
    )

    print()

    print(
        "Balanced-V2 mean severe cost:",
        f"{b['mean_severe_cost']:.6f}",
    )

    print(
        "V3 mean severe cost:",
        f"{v['mean_severe_cost']:.6f}",
    )

    print()

    print(
        "Balanced-V2 phase cost:",
        f"{b['mean_phase_cost']:.6f}",
    )

    print(
        "V3 phase cost:",
        f"{v['mean_phase_cost']:.6f}",
    )

    print()

    print(
        "Balanced-V2 penetrating frames:",
        b[
            "penetrating_frames"
        ],
    )

    print(
        "V3 penetrating frames:",
        v[
            "penetrating_frames"
        ],
    )

    print()
    print(
        "Do not make the long-training decision "
        "from one number alone."
    )

    print(
        "Review maximum severity, >=40/50/60 counts, "
        "problem regions, and global penetration together."
    )

    print(
        "=" * 120
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [
        ORIGINAL_V2_MODEL,
        BALANCED_V2_MODEL,
        BALANCED_V3_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required checkpoint not found:\n"
                f"{path}"
            )


    print(
        "=" * 125
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "BC / V2 / BALANCED-V2 / V3 DETERMINISTIC COMPARISON"
    )

    print(
        "=" * 125
    )

    print()

    print(
        "Original V2:"
    )

    print(
        ORIGINAL_V2_MODEL
    )

    print()

    print(
        "Balanced V2:"
    )

    print(
        BALANCED_V2_MODEL
    )

    print()

    print(
        "Balanced V3:"
    )

    print(
        BALANCED_V3_MODEL
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        "  environment = ORIGINAL Reward V2"
    )

    print(
        "  explicit start =",
        TRAIN_START_STEP,
    )

    print(
        "  final step =",
        TRAIN_END_STEP,
    )

    print(
        "  residual smoothing =",
        RESIDUAL_SMOOTHING,
    )

    print(
        "  deterministic actions"
    )

    print(
        "  random starts disabled"
    )

    print(
        "  same terrain/root/reference for all"
    )

    print(
        "  no mj_step()"
    )

    print()

    print(
        "V3 offline severe reward:"
    )

    print(
        "  threshold:",
        (
            f"{SEVERE_PENETRATION_THRESHOLD * 1000:.1f} mm"
        ),
    )

    print(
        "  weight:",
        SEVERE_PENETRATION_WEIGHT,
    )

    print(
        "=" * 125
    )


    # ========================================================
    # ENVIRONMENTS
    # ========================================================

    bc_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    v2_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    balanced_v2_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    v3_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        # ====================================================
        # LOAD MODELS
        # ====================================================

        print()

        print(
            "Loading original V2..."
        )

        original_v2_model = PPO.load(
            ORIGINAL_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced V2..."
        )

        balanced_v2_model = PPO.load(
            BALANCED_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced V3..."
        )

        balanced_v3_model = PPO.load(
            BALANCED_V3_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )


        # ====================================================
        # SPACE VALIDATION
        # ====================================================

        validate_model_spaces(
            original_v2_model,
            v2_env,
            "Original V2",
        )

        validate_model_spaces(
            balanced_v2_model,
            balanced_v2_env,
            "Balanced V2",
        )

        validate_model_spaces(
            balanced_v3_model,
            v3_env,
            "Balanced V3",
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # ROLLOUTS
        # ====================================================

        print()

        print(
            "Running BC-only..."
        )

        bc = run_rollout(
            bc_env,
            model=None,
        )

        print(
            "Running original V2..."
        )

        v2 = run_rollout(
            v2_env,
            model=original_v2_model,
        )

        print(
            "Running balanced V2..."
        )

        balanced_v2 = run_rollout(
            balanced_v2_env,
            model=balanced_v2_model,
        )

        print(
            "Running V3..."
        )

        v3 = run_rollout(
            v3_env,
            model=balanced_v3_model,
        )


        # ====================================================
        # FAIRNESS
        # ====================================================

        print()

        print(
            "=" * 125
        )

        print(
            "FAIR-COMPARISON INVARIANTS"
        )

        print(
            "=" * 125
        )

        validate_invariants(
            bc,
            v2,
            "Original V2",
        )

        validate_invariants(
            bc,
            balanced_v2,
            "Balanced V2",
        )

        validate_invariants(
            bc,
            v3,
            "Balanced V3",
        )

        print(
            "PASS"
        )


        # ====================================================
        # RESULTS
        # ====================================================

        results = {
            "BC": {
                "rollout":
                    bc,

                "geometry":
                    geometry(
                        bc
                    ),
            },

            "V2": {
                "rollout":
                    v2,

                "geometry":
                    geometry(
                        v2
                    ),
            },

            "BALANCED_V2": {
                "rollout":
                    balanced_v2,

                "geometry":
                    geometry(
                        balanced_v2
                    ),
            },

            "V3": {
                "rollout":
                    v3,

                "geometry":
                    geometry(
                        v3
                    ),
            },
        }


        # ====================================================
        # REPORTS
        # ====================================================

        print_global_table(
            results
        )

        print_v3_change(
            results
        )

        print_severity(
            results
        )

        print_regions(
            results
        )

        print_selected_steps(
            results
        )

        print_per_frame_outcome(
            results
        )

        print_worst_v3(
            results,
            count=25,
        )

        print_policy_diagnostics(
            results
        )

        print_decision_summary(
            results
        )


        print()

        print(
            "=" * 125
        )

        print(
            "V3 DETERMINISTIC VERIFICATION COMPLETE"
        )

        print(
            "=" * 125
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No checkpoint was modified."
        )

        print()

        print(
            "Primary metric:"
        )

        print(
            "  absolute terrain penetration"
        )

        print()

        print(
            "Secondary diagnostics:"
        )

        print(
            "  same-phase deficit"
        )

        print(
            "  Reward-V2 phase cost"
        )

        print(
            "  Reward-V3 severe cost"
        )

        print(
            "  residual/posture behavior"
        )

        print()

        print(
            "Do NOT launch longer PPO training "
            "until these numbers are reviewed."
        )

        print(
            "=" * 125
        )


    finally:

        bc_env.close()

        v2_env.close()

        balanced_v2_env.close()

        v3_env.close()


if __name__ == "__main__":
    main()