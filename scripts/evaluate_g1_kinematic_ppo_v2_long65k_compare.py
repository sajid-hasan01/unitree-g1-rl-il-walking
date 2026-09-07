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
# ALL policies are evaluated here on exactly the same
# deterministic 150 -> 750 trajectory.
#
# Training random-start sampling is NOT used during evaluation.
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
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

BALANCED_V2_8K_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_smoke_seed425.zip"
)

BALANCED_V2_65K_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_long65k_seed425.zip"
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
# FOOT GROUPS
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
# PENETRATION SEVERITY THRESHOLDS
# ============================================================

SEVERITY_THRESHOLDS_MM = [
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]


# ============================================================
# IMPORTANT REGIONS
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
# IMPORTANT INDIVIDUAL FRAMES
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


    # ========================================================
    # INITIAL STATE -- STEP 150
    # ========================================================

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

    rewards = []

    clip_costs = []


    # ========================================================
    # 600 TRANSITIONS -- 150 -> 750
    # ========================================================

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
            "Action outside environment action space.",
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
            terminated
            is False,
            (
                "Unexpected kinematic termination "
                f"at step {info['step']}."
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
                    "Unexpected early truncation "
                    f"at step {info['step']}."
                ),
            )

        require(
            np.all(
                np.isfinite(
                    observation
                )
            ),
            (
                "Non-finite observation "
                f"at step {info['step']}."
            ),
        )

        require(
            np.isfinite(
                reward
            ),
            (
                "Non-finite reward "
                f"at step {info['step']}."
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

        rewards.append(
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
        # RESULTING STATE DATA
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


    # ========================================================
    # ARRAYS
    # ========================================================

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

        "rewards":
            np.asarray(
                rewards,
                dtype=np.float64,
            ),

        "clip_costs":
            np.asarray(
                clip_costs,
                dtype=np.float64,
            ),
    }


    # ========================================================
    # STRUCTURE CHECKS
    # ========================================================

    require(
        result[
            "steps"
        ].shape
        ==
        (
            EVAL_STATES,
        ),
        (
            "Unexpected number of evaluated states."
        ),
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
        (
            "Unexpected signed-distance shape."
        ),
    )

    require(
        result[
            "flat"
        ].shape
        ==
        (
            EVAL_STATES,
            8,
        ),
        (
            "Unexpected flat-reference shape."
        ),
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
        (
            "Unexpected residual shape."
        ),
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
        (
            "Unexpected action shape."
        ),
    )

    require(
        result[
            "rewards"
        ].shape
        ==
        (
            EVAL_TRANSITIONS,
        ),
        (
            "Unexpected reward shape."
        ),
    )

    require(
        int(
            result[
                "steps"
            ][
                0
            ]
        )
        ==
        TRAIN_START_STEP,
        "Wrong first state.",
    )

    require(
        int(
            result[
                "steps"
            ][
                -1
            ]
        )
        ==
        TRAIN_END_STEP,
        "Wrong final state.",
    )

    return result


# ============================================================
# GEOMETRY METRICS
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


    # ========================================================
    # ABSOLUTE RAW TERRAIN PENETRATION
    # ========================================================

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


    # ========================================================
    # SAME-PHASE DEFICIT
    # ========================================================

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


    # ========================================================
    # REWARD-V2 PHASE-AWARE PENETRATION
    # ========================================================

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


    # ========================================================
    # LEFT / RIGHT FOOT SHAPE
    #
    # Secondary diagnostic only.
    # ========================================================

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


    # ========================================================
    # SEVERITY COUNTS
    # ========================================================

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

        "left_gap":
            left_gap,

        "right_gap":
            right_gap,

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

        "severity_counts":
            severity_counts,
    }


# ============================================================
# FAIR-COMPARISON CHECK
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
        f"{name:20s} | "
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
# GLOBAL SUMMARY
# ============================================================

def print_global_summary(
    results,
):

    print()
    print(
        "=" * 175
    )

    print(
        "GLOBAL DETERMINISTIC GEOMETRY"
    )

    print(
        "=" * 175
    )

    print(
        "policy       | penFrm | penSamp | "
        "meanPen | meanMax | maxPen | "
        "meanDef | maxDef | sqDef   | "
        "phaseCost | maxPhase | reward"
    )

    print(
        "-" * 175
    )

    for name in [
        "BC",
        "V2",
        "B-V2-8K",
        "B-V2-65K",
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
            f"{g['mean_frame_penetration'] * 1000:7.2f} | "
            f"{g['max_penetration'] * 1000:6.2f} | "
            f"{g['mean_deficit'] * 1000:7.2f} | "
            f"{g['max_deficit'] * 1000:6.2f} | "
            f"{g['squared_deficit']:7.4f} | "
            f"{g['mean_phase_cost']:9.5f} | "
            f"{g['max_phase_excess'] * 1000:8.2f} | "
            f"{np.mean(rollout['rewards']):+7.4f}"
        )


# ============================================================
# 8K -> 65K CHANGE
# ============================================================

def reduction_percent(
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


def print_training_budget_change(
    results,
):

    old = results[
        "B-V2-8K"
    ][
        "geometry"
    ]

    new = results[
        "B-V2-65K"
    ][
        "geometry"
    ]

    print()
    print(
        "=" * 115
    )

    print(
        "BALANCED V2: 8K -> 65K TRAINING BUDGET"
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
            "mean frame max penetration",
            old[
                "mean_frame_penetration"
            ],
            new[
                "mean_frame_penetration"
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
    ]

    for (
        label,
        old_value,
        new_value,
    ) in metrics:

        print(
            f"{label:31s}: "
            f"{reduction_percent(old_value, new_value):+8.2f}%"
        )


# ============================================================
# SEVERITY COUNTS
# ============================================================

def print_severity_counts(
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
        "threshold | BC | V2-8K | Balanced-8K | Balanced-65K"
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
            f"{results['V2']['geometry']['severity_counts'][threshold]:6d} | "
            f"{results['B-V2-8K']['geometry']['severity_counts'][threshold]:11d} | "
            f"{results['B-V2-65K']['geometry']['severity_counts'][threshold]:12d}"
        )


# ============================================================
# REGION SUMMARY
# ============================================================

def print_regions(
    results,
):

    print()
    print(
        "=" * 165
    )

    print(
        "REGION-SPECIFIC PENETRATION"
    )

    print(
        "=" * 165
    )

    print(
        "region                 | policy       | "
        "penFrm | maxPen | meanMax | "
        "phaseCost | maxPhase | meanDef"
    )

    print(
        "-" * 165
    )

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        for name in [
            "BC",
            "V2",
            "B-V2-8K",
            "B-V2-65K",
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
                f"{np.mean(deficit) * 1000:7.2f}"
            )

        print(
            "-" * 165
        )


# ============================================================
# SELECTED STEPS
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
        "step | BC pen | V2 pen | B8K pen | B65K pen | "
        "B8K phase | B65K phase | "
        "B8K Lgap | B65K Lgap"
    )

    print(
        "-" * 175
    )

    steps = results[
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
            steps
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

        v2 = results[
            "V2"
        ][
            "geometry"
        ]

        b8 = results[
            "B-V2-8K"
        ][
            "geometry"
        ]

        b65 = results[
            "B-V2-65K"
        ][
            "geometry"
        ]

        print(
            f"{requested_step:4d} | "
            f"{bc['frame_penetration'][i] * 1000:6.1f} | "
            f"{v2['frame_penetration'][i] * 1000:6.1f} | "
            f"{b8['frame_penetration'][i] * 1000:7.1f} | "
            f"{b65['frame_penetration'][i] * 1000:8.1f} | "
            f"{b8['frame_phase_excess'][i] * 1000:9.1f} | "
            f"{b65['frame_phase_excess'][i] * 1000:10.1f} | "
            f"{b8['left_gap'][i] * 1000:+9.1f} | "
            f"{b65['left_gap'][i] * 1000:+10.1f}"
        )


# ============================================================
# PER-FRAME 65K OUTCOME
# ============================================================

def print_per_frame_outcome(
    results,
):

    old = results[
        "B-V2-8K"
    ][
        "geometry"
    ][
        "frame_penetration"
    ]

    new = results[
        "B-V2-65K"
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
        "=" * 110
    )

    print(
        "PER-FRAME MAX-PENETRATION OUTCOME: "
        "BALANCED V2 65K VS 8K"
    )

    print(
        "=" * 110
    )

    print(
        "65K better:",
        better,
    )

    print(
        "Same:",
        same,
    )

    print(
        "65K worse:",
        worse,
    )


# ============================================================
# WORST 65K FRAMES
# ============================================================

def print_worst_65k(
    results,
    count=25,
):

    old = results[
        "B-V2-8K"
    ][
        "geometry"
    ]

    new = results[
        "B-V2-65K"
    ][
        "geometry"
    ]

    rollout = results[
        "B-V2-65K"
    ][
        "rollout"
    ]

    order = np.argsort(
        new[
            "frame_penetration"
        ]
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 140
    )

    print(
        "WORST REMAINING BALANCED-V2 65K PENETRATION FRAMES"
    )

    print(
        "=" * 140
    )

    print(
        "rank | step | x       | "
        "8K pen | 65K pen | improvement | "
        "65K phase | 65K Lgap"
    )

    print(
        "-" * 140
    )

    for (
        rank,
        i,
    ) in enumerate(
        order,
        start=1,
    ):

        improvement = (
            old[
                "frame_penetration"
            ][
                i
            ]
            -
            new[
                "frame_penetration"
            ][
                i
            ]
        )

        print(
            f"{rank:4d} | "
            f"{rollout['steps'][i]:4d} | "
            f"{rollout['world_x'][i]:+7.3f} | "
            f"{old['frame_penetration'][i] * 1000:6.2f} | "
            f"{new['frame_penetration'][i] * 1000:7.2f} | "
            f"{improvement * 1000:+11.2f} | "
            f"{new['frame_phase_excess'][i] * 1000:9.2f} | "
            f"{new['left_gap'][i] * 1000:+9.2f}"
        )


# ============================================================
# POLICY DIAGNOSTICS
# ============================================================

def print_policy_diagnostics(
    results,
):

    print()
    print(
        "=" * 145
    )

    print(
        "POLICY RESIDUAL DIAGNOSTICS"
    )

    print(
        "=" * 145
    )

    print(
        "policy       | mean|act| | max|act| | "
        "mean|res| | max|res| | "
        "L hip | L knee | L ankleP | "
        "R hip | R knee | R ankleP | maxClip"
    )

    print(
        "-" * 145
    )

    for name in [
        "V2",
        "B-V2-8K",
        "B-V2-65K",
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
# FINAL DECISION DATA
# ============================================================

def print_decision_data(
    results,
):

    b8 = results[
        "B-V2-8K"
    ][
        "geometry"
    ]

    b65 = results[
        "B-V2-65K"
    ][
        "geometry"
    ]

    r8 = float(
        np.mean(
            results[
                "B-V2-8K"
            ][
                "rollout"
            ][
                "rewards"
            ]
        )
    )

    r65 = float(
        np.mean(
            results[
                "B-V2-65K"
            ][
                "rollout"
            ][
                "rewards"
            ]
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BALANCED V2 65K DECISION DATA"
    )

    print(
        "=" * 120
    )

    print(
        "8K maximum penetration:",
        f"{b8['max_penetration'] * 1000:.3f} mm",
    )

    print(
        "65K maximum penetration:",
        f"{b65['max_penetration'] * 1000:.3f} mm",
    )

    print()

    for threshold in [
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]:

        print(
            f">={threshold:.0f} mm frames: "
            f"8K={b8['severity_counts'][threshold]} | "
            f"65K={b65['severity_counts'][threshold]}"
        )

    print()

    print(
        "8K penetrating frames:",
        b8[
            "penetrating_frames"
        ],
    )

    print(
        "65K penetrating frames:",
        b65[
            "penetrating_frames"
        ],
    )

    print()

    print(
        "8K phase cost:",
        f"{b8['mean_phase_cost']:.6f}",
    )

    print(
        "65K phase cost:",
        f"{b65['mean_phase_cost']:.6f}",
    )

    print()

    print(
        "8K squared deficit:",
        f"{b8['squared_deficit']:.6f}",
    )

    print(
        "65K squared deficit:",
        f"{b65['squared_deficit']:.6f}",
    )

    print()

    print(
        "8K same-path Reward-V2:",
        f"{r8:+.6f}",
    )

    print(
        "65K same-path Reward-V2:",
        f"{r65:+.6f}",
    )

    print()

    print(
        "Important:"
    )

    print(
        "  Higher training reward alone is NOT enough."
    )

    print(
        "  Penetration geometry decides whether "
        "the 65K policy is actually better."
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
        BALANCED_V2_8K_MODEL,
        BALANCED_V2_65K_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required PPO checkpoint not found:\n"
                f"{path}"
            )


    print(
        "=" * 125
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "BALANCED REWARD-V2 8K VS 65K DETERMINISTIC COMPARISON"
    )

    print(
        "=" * 125
    )

    print()

    print(
        "Original V2 8K:"
    )

    print(
        ORIGINAL_V2_MODEL
    )

    print()

    print(
        "Balanced V2 8K:"
    )

    print(
        BALANCED_V2_8K_MODEL
    )

    print()

    print(
        "Balanced V2 65K:"
    )

    print(
        BALANCED_V2_65K_MODEL
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        "  Reward-V2 environment"
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
        "  deterministic policy actions"
    )

    print(
        "  random starts disabled"
    )

    print(
        "  same terrain"
    )

    print(
        "  same root trajectory"
    )

    print(
        "  same flat BC reference"
    )

    print(
        "  same action/residual system"
    )

    print(
        "  no mj_step()"
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

    b8_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    b65_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        # ====================================================
        # LOAD CHECKPOINTS
        # ====================================================

        print()

        print(
            "Loading original V2 8K..."
        )

        original_v2_model = PPO.load(
            ORIGINAL_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced V2 8K..."
        )

        balanced_v2_8k_model = PPO.load(
            BALANCED_V2_8K_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced V2 65K..."
        )

        balanced_v2_65k_model = PPO.load(
            BALANCED_V2_65K_MODEL,
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
            balanced_v2_8k_model,
            b8_env,
            "Balanced V2 8K",
        )

        validate_model_spaces(
            balanced_v2_65k_model,
            b65_env,
            "Balanced V2 65K",
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
            "Running original V2 8K..."
        )

        v2 = run_rollout(
            v2_env,
            model=original_v2_model,
        )

        print(
            "Running balanced V2 8K..."
        )

        b8 = run_rollout(
            b8_env,
            model=balanced_v2_8k_model,
        )

        print(
            "Running balanced V2 65K..."
        )

        b65 = run_rollout(
            b65_env,
            model=balanced_v2_65k_model,
        )


        # ====================================================
        # FAIR-COMPARISON INVARIANTS
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
            "Original V2 8K",
        )

        validate_invariants(
            bc,
            b8,
            "Balanced V2 8K",
        )

        validate_invariants(
            bc,
            b65,
            "Balanced V2 65K",
        )

        print(
            "PASS"
        )


        # ====================================================
        # BUILD RESULTS
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

            "B-V2-8K": {
                "rollout":
                    b8,

                "geometry":
                    geometry(
                        b8
                    ),
            },

            "B-V2-65K": {
                "rollout":
                    b65,

                "geometry":
                    geometry(
                        b65
                    ),
            },
        }


        # ====================================================
        # REPORTS
        # ====================================================

        print_global_summary(
            results
        )

        print_training_budget_change(
            results
        )

        print_severity_counts(
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

        print_worst_65k(
            results,
            count=25,
        )

        print_policy_diagnostics(
            results
        )

        print_decision_data(
            results
        )


        print()

        print(
            "=" * 125
        )

        print(
            "BALANCED REWARD-V2 65K "
            "DETERMINISTIC EVALUATION COMPLETE"
        )

        print(
            "=" * 125
        )

        print(
            "No training was performed."
        )

        print(
            "No checkpoint was modified."
        )

        print()

        print(
            "Primary decision:"
        )

        print(
            "  Did additional V2 training actually "
            "reduce terrain penetration?"
        )

        print()

        print(
            "Do NOT start V3 65K or extend V2 further "
            "until these geometry results are reviewed."
        )

        print(
            "=" * 125
        )


    finally:

        bc_env.close()

        v2_env.close()

        b8_env.close()

        b65_env.close()


if __name__ == "__main__":
    main()