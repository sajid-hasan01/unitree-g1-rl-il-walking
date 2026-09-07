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
# IMPORTANT:
#
# Evaluation deliberately uses the ORIGINAL Reward-V2
# environment.
#
# Balanced sampling exists only during training.
#
# Therefore all three trajectories below use exactly the same:
#
# - reference path
# - terrain
# - root placement
# - Reward V2
# - observations
# - residual smoothing
# - explicit start step
#
# No random starts are used here.
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

BALANCED_V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_smoke_seed425.zip"
)


# ============================================================
# EVALUATION
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


LEFT_HEEL = np.array(
    [0, 1],
    dtype=np.int64,
)

LEFT_TOE = np.array(
    [2, 3],
    dtype=np.int64,
)


SEVERITY_THRESHOLDS_MM = [
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
]


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
        "worst_pen",
        675,
        700,
    ),

    (
        "late_orient",
        708,
        735,
    ),
]


SELECTED_STEPS = [
    395,
    435,
    487,
    488,
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
    713,
    714,
    715,
    716,
    717,
    718,
    719,
    720,
    723,
    728,
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
        (
            f"{name}: observation-space "
            "shape mismatch."
        ),
    )

    require(
        model.action_space.shape
        ==
        env.action_space.shape,
        (
            f"{name}: action-space "
            "shape mismatch."
        ),
    )

    require(
        np.allclose(
            model.observation_space.low,
            env.observation_space.low,
        ),
        (
            f"{name}: observation lower "
            "bounds mismatch."
        ),
    )

    require(
        np.allclose(
            model.observation_space.high,
            env.observation_space.high,
        ),
        (
            f"{name}: observation upper "
            "bounds mismatch."
        ),
    )

    require(
        np.allclose(
            model.action_space.low,
            env.action_space.low,
        ),
        (
            f"{name}: action lower "
            "bounds mismatch."
        ),
    )

    require(
        np.allclose(
            model.action_space.high,
            env.action_space.high,
        ),
        (
            f"{name}: action upper "
            "bounds mismatch."
        ),
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
        "Unexpected evaluation start step.",
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
    rewards = []
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

        require(
            terminated
            is False,
            (
                "Unexpected kinematic "
                "termination."
            ),
        )

        if (
            transition_index
            <
            EVAL_TRANSITIONS - 1
        ):

            require(
                truncated
                is False,
                (
                    "Evaluation truncated "
                    "too early at step "
                    f"{info['step']}."
                ),
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
        "Unexpected action shape.",
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
    # RAW TERRAIN PENETRATION
    # --------------------------------------------------------

    penetration = np.maximum(
        -current,
        0.0,
    )

    frame_penetration = np.max(
        penetration,
        axis=1,
    )


    # --------------------------------------------------------
    # SAME-PHASE BC DEFICIT
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
    # LEFT FOOT HEEL-TO-TOE GAP
    #
    # This is SECONDARY diagnostics only.
    #
    # Positive = toe closer/lower than heel.
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


    positive_penetration = penetration[
        penetration
        >
        0.0
    ]


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

        "flat_left_gap":
            flat_left_gap,

        "left_extra":
            (
                left_gap
                -
                flat_left_gap
            ),

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
        f"{name}: X invariant failed.",
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
        "=" * 165
    )

    print(
        "GLOBAL DETERMINISTIC GEOMETRY"
    )

    print(
        "=" * 165
    )

    print(
        "policy       | penFrm | penSamp | "
        "meanPen | maxPen | "
        "meanDef | maxDef | sqDef    | "
        "phaseCost | maxPhase | reward"
    )

    print(
        "-" * 165
    )

    for name in [
        "BC",
        "V2",
        "BALANCED",
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
            f"{g['squared_deficit']:8.4f} | "
            f"{g['mean_phase_cost']:9.5f} | "
            f"{g['max_phase_excess'] * 1000:8.2f} | "
            f"{np.mean(rollout['rewards']):+7.4f}"
        )


# ============================================================
# V2 -> BALANCED CHANGE
# ============================================================

def percent_reduction(
    original,
    new,
):

    if abs(
        float(
            original
        )
    ) < 1e-12:

        return 0.0

    return (
        100.0
        *
        (
            float(
                original
            )
            -
            float(
                new
            )
        )
        /
        float(
            original
        )
    )


def print_balanced_change(
    results,
):

    old = results[
        "V2"
    ][
        "geometry"
    ]

    new = results[
        "BALANCED"
    ][
        "geometry"
    ]

    old_r = results[
        "V2"
    ][
        "rollout"
    ]

    new_r = results[
        "BALANCED"
    ][
        "rollout"
    ]

    print()
    print(
        "=" * 110
    )

    print(
        "ORIGINAL V2 -> BALANCED-START V2"
    )

    print(
        "=" * 110
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
            "V2 phase penetration cost",
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
        original,
        balanced,
    ) in metrics:

        print(
            f"{label:30s}: "
            f"{percent_reduction(original, balanced):+8.2f}%"
        )

    print()

    print(
        "Same-path Reward-V2 mean:"
    )

    print(
        "  original:",
        f"{np.mean(old_r['rewards']):+.6f}",
    )

    print(
        "  balanced:",
        f"{np.mean(new_r['rewards']):+.6f}",
    )

    print(
        "  change:",
        (
            f"{np.mean(new_r['rewards']) - np.mean(old_r['rewards']):+.6f}"
        ),
    )


# ============================================================
# SEVERITY COUNTS
# ============================================================

def print_severity(
    results,
):

    print()
    print(
        "=" * 95
    )

    print(
        "MAX-PENETRATION SEVERITY COUNTS"
    )

    print(
        "=" * 95
    )

    print(
        "threshold | BC | Original V2 | Balanced V2"
    )

    print(
        "-" * 60
    )

    for threshold in (
        SEVERITY_THRESHOLDS_MM
    ):

        print(
            f"{threshold:8.1f} | "
            f"{results['BC']['geometry']['severity_counts'][threshold]:2d} | "
            f"{results['V2']['geometry']['severity_counts'][threshold]:11d} | "
            f"{results['BALANCED']['geometry']['severity_counts'][threshold]:11d}"
        )


# ============================================================
# REGIONS
# ============================================================

def print_regions(
    results,
):

    print()
    print(
        "=" * 145
    )

    print(
        "REGION-SPECIFIC PENETRATION"
    )

    print(
        "=" * 145
    )

    print(
        "region       | policy       | "
        "penFrames | maxPen | meanMaxPen | "
        "phaseCost | maxPhase | meanDef"
    )

    print(
        "-" * 145
    )

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        for name in [
            "BC",
            "V2",
            "BALANCED",
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
                f"{region_name:12s} | "
                f"{name:12s} | "
                f"{np.count_nonzero(frame_pen > 0.0):9d} | "
                f"{np.max(frame_pen) * 1000:6.2f} | "
                f"{np.mean(frame_pen) * 1000:10.2f} | "
                f"{np.mean(phase_cost):9.5f} | "
                f"{np.max(frame_phase) * 1000:8.2f} | "
                f"{np.mean(deficit) * 1000:7.2f}"
            )

        print(
            "-" * 145
        )


# ============================================================
# SELECTED STEPS
# ============================================================

def print_selected_steps(
    results,
):

    print()
    print(
        "=" * 160
    )

    print(
        "SELECTED FRAME COMPARISON"
    )

    print(
        "=" * 160
    )

    print(
        "step | BC pen | V2 pen | BAL pen | "
        "V2 phase | BAL phase | "
        "V2 Lgap | BAL Lgap | "
        "V2 dHip | BAL dHip | "
        "V2 dKnee | BAL dKnee"
    )

    print(
        "-" * 160
    )

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            results[
                "BC"
            ][
                "rollout"
            ][
                "steps"
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

        i = int(
            matches[
                0
            ]
        )

        bc_g = results[
            "BC"
        ][
            "geometry"
        ]

        v2_g = results[
            "V2"
        ][
            "geometry"
        ]

        bal_g = results[
            "BALANCED"
        ][
            "geometry"
        ]

        v2_res = results[
            "V2"
        ][
            "rollout"
        ][
            "residuals"
        ][
            i
        ]

        bal_res = results[
            "BALANCED"
        ][
            "rollout"
        ][
            "residuals"
        ][
            i
        ]

        print(
            f"{requested_step:4d} | "
            f"{bc_g['frame_penetration'][i] * 1000:6.1f} | "
            f"{v2_g['frame_penetration'][i] * 1000:6.1f} | "
            f"{bal_g['frame_penetration'][i] * 1000:7.1f} | "
            f"{v2_g['frame_phase_excess'][i] * 1000:8.1f} | "
            f"{bal_g['frame_phase_excess'][i] * 1000:9.1f} | "
            f"{v2_g['left_gap'][i] * 1000:+7.1f} | "
            f"{bal_g['left_gap'][i] * 1000:+8.1f} | "
            f"{v2_res[0]:+7.3f} | "
            f"{bal_res[0]:+8.3f} | "
            f"{v2_res[3]:+8.3f} | "
            f"{bal_res[3]:+9.3f}"
        )


# ============================================================
# PER-FRAME ORIGINAL-vs-BALANCED
# ============================================================

def print_frame_outcome(
    results,
):

    old = results[
        "V2"
    ][
        "geometry"
    ][
        "frame_penetration"
    ]

    new = results[
        "BALANCED"
    ][
        "geometry"
    ][
        "frame_penetration"
    ]

    difference = (
        old
        -
        new
    )

    tolerance = 1e-9

    better = int(
        np.count_nonzero(
            difference
            >
            tolerance
        )
    )

    same = int(
        np.count_nonzero(
            np.abs(
                difference
            )
            <=
            tolerance
        )
    )

    worse = int(
        np.count_nonzero(
            difference
            <
            -tolerance
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PER-FRAME MAX-PENETRATION OUTCOME"
    )

    print(
        "=" * 100
    )

    print(
        "Balanced better:",
        better,
    )

    print(
        "Same:",
        same,
    )

    print(
        "Balanced worse:",
        worse,
    )


# ============================================================
# WORST BALANCED FRAMES
# ============================================================

def print_worst_balanced(
    results,
    count=20,
):

    bal = results[
        "BALANCED"
    ][
        "geometry"
    ]

    old = results[
        "V2"
    ][
        "geometry"
    ]

    rollout = results[
        "BALANCED"
    ][
        "rollout"
    ]

    order = np.argsort(
        bal[
            "frame_penetration"
        ]
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 120
    )

    print(
        "WORST REMAINING BALANCED-V2 PENETRATION FRAMES"
    )

    print(
        "=" * 120
    )

    print(
        "rank | step | x       | "
        "Original pen | Balanced pen | "
        "change | Balanced phase"
    )

    print(
        "-" * 120
    )

    for (
        rank,
        i,
    ) in enumerate(
        order,
        start=1,
    ):

        change = (
            old[
                "frame_penetration"
            ][
                i
            ]
            -
            bal[
                "frame_penetration"
            ][
                i
            ]
        )

        print(
            f"{rank:4d} | "
            f"{rollout['steps'][i]:4d} | "
            f"{rollout['world_x'][i]:+7.3f} | "
            f"{old['frame_penetration'][i] * 1000:12.2f} | "
            f"{bal['frame_penetration'][i] * 1000:12.2f} | "
            f"{change * 1000:+7.2f} | "
            f"{bal['frame_phase_excess'][i] * 1000:14.2f}"
        )


# ============================================================
# POLICY DIAGNOSTICS
# ============================================================

def print_policy_diagnostics(
    results,
):

    print()
    print(
        "=" * 120
    )

    print(
        "POLICY RESIDUAL DIAGNOSTICS"
    )

    print(
        "=" * 120
    )

    print(
        "policy       | mean|action| | max|action| | "
        "mean|residual| | max|residual| | "
        "L hip mean | L knee mean | L ankle mean | maxClip"
    )

    print(
        "-" * 120
    )

    for name in [
        "V2",
        "BALANCED",
    ]:

        rollout = results[
            name
        ][
            "rollout"
        ]

        residual = (
            rollout[
                "residuals"
            ][
                1:
            ]
        )

        action = rollout[
            "actions"
        ]

        print(
            f"{name:12s} | "
            f"{np.mean(np.abs(action)):12.5f} | "
            f"{np.max(np.abs(action)):11.5f} | "
            f"{np.mean(np.abs(residual)):14.5f} | "
            f"{np.max(np.abs(residual)):13.5f} | "
            f"{np.mean(residual[:, 0]):+10.4f} | "
            f"{np.mean(residual[:, 3]):+11.4f} | "
            f"{np.mean(residual[:, 4]):+12.4f} | "
            f"{np.max(rollout['clip_costs']):7.5f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [
        ORIGINAL_V2_MODEL,
        BALANCED_V2_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required checkpoint not found:\n"
                f"{path}"
            )


    print(
        "=" * 115
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "ORIGINAL-vs-BALANCED DETERMINISTIC COMPARISON"
    )

    print(
        "=" * 115
    )

    print(
        "Original model:",
        ORIGINAL_V2_MODEL,
    )

    print(
        "Balanced model:",
        BALANCED_V2_MODEL,
    )

    print()

    print(
        "Evaluation environment:"
    )

    print(
        "  ORIGINAL Reward V2"
    )

    print(
        "  explicit start =",
        TRAIN_START_STEP,
    )

    print(
        "  end =",
        TRAIN_END_STEP,
    )

    print(
        "  residual smoothing =",
        RESIDUAL_SMOOTHING,
    )

    print(
        "  deterministic policies"
    )

    print(
        "  random training sampler NOT used"
    )

    print(
        "  physics: NONE -- mj_forward only"
    )

    print(
        "=" * 115
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

    old_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    balanced_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        print()
        print(
            "Loading original V2 PPO..."
        )

        old_model = PPO.load(
            ORIGINAL_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced-start V2 PPO..."
        )

        balanced_model = PPO.load(
            BALANCED_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            old_model,
            old_env,
            "Original V2",
        )

        validate_model_spaces(
            balanced_model,
            balanced_env,
            "Balanced V2",
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
            "Running original Reward-V2 PPO..."
        )

        old = run_rollout(
            old_env,
            model=old_model,
        )

        print(
            "Running balanced-start Reward-V2 PPO..."
        )

        balanced = run_rollout(
            balanced_env,
            model=balanced_model,
        )


        # ====================================================
        # INVARIANTS
        # ====================================================

        print()
        print(
            "=" * 115
        )

        print(
            "FAIR-COMPARISON INVARIANTS"
        )

        print(
            "=" * 115
        )

        validate_invariants(
            bc,
            old,
            "Original V2",
        )

        validate_invariants(
            bc,
            balanced,
            "Balanced V2",
        )

        print(
            "PASS"
        )


        # ====================================================
        # METRICS
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
                    old,

                "geometry":
                    geometry(
                        old
                    ),
            },

            "BALANCED": {
                "rollout":
                    balanced,

                "geometry":
                    geometry(
                        balanced
                    ),
            },
        }


        print_global_table(
            results
        )

        print_balanced_change(
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

        print_frame_outcome(
            results
        )

        print_worst_balanced(
            results,
            count=20,
        )

        print_policy_diagnostics(
            results
        )


        print()
        print(
            "=" * 115
        )

        print(
            "DETERMINISTIC COMPARISON COMPLETE"
        )

        print(
            "=" * 115
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No environment/model file was modified."
        )

        print(
            "Primary decision metric: penetration."
        )

        print(
            "Left-leg residuals are reported only as "
            "secondary diagnostics."
        )

        print(
            "Do NOT increase the training budget until "
            "this comparison is reviewed."
        )

        print(
            "=" * 115
        )


    finally:

        bc_env.close()
        old_env.close()
        balanced_env.close()


if __name__ == "__main__":
    main()