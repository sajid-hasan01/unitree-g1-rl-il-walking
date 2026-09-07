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
# EXACT CURRENT REWARD-V1 ENVIRONMENT + CHECKPOINT
# ============================================================

from envs.g1_kinematic_uneven_env import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_smoke_seed425.zip"
)


RESIDUAL_SMOOTHING = 0.35
SEED = 425

EVAL_STATES = (
    TRAIN_END_STEP
    -
    TRAIN_START_STEP
    +
    1
)

SELECTED_STEPS = [
    708,
    716,
    720,
    723,
    728,
]

# These are diagnostic proximity sweeps only.
# They are NOT being declared as physical contact thresholds.
PROXIMITY_THRESHOLDS_MM = [
    5.0,
    10.0,
    20.0,
]


# ============================================================
# VERIFIED SPHERE ORDER
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
            "Saved PPO observation space does not "
            "match the current Reward-V1 environment."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):
        raise RuntimeError(
            "Saved PPO action space does not "
            "match the current Reward-V1 environment."
        )


def current_state_record(
    env,
):

    step = int(
        env.current_step
    )

    return {
        "step":
            step,

        "motion_frame":
            float(
                env.current_metadata[
                    "motion_frame"
                ]
            ),

        "world_x":
            float(
                env.current_metadata[
                    "world_x"
                ]
            ),

        "distances":
            env.current_distances
            .astype(
                np.float64
            )
            .copy(),

        "flat_reference":
            env.flat_reference_distances[
                step
            ]
            .astype(
                np.float64
            )
            .copy(),
    }


def rollout(
    env,
    model=None,
    use_ppo=False,
):

    observation, _ = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP,
        },
    )

    records = [
        current_state_record(
            env
        )
    ]

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
            _reward,
            _terminated,
            _truncated,
            _info,
        ) = env.step(
            action
        )

        records.append(
            current_state_record(
                env
            )
        )

    if len(records) != EVAL_STATES:
        raise RuntimeError(
            "Unexpected rollout length: "
            f"{len(records)} != {EVAL_STATES}"
        )

    return {
        "steps":
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

        "motion_frames":
            np.asarray(
                [
                    item[
                        "motion_frame"
                    ]
                    for item
                    in records
                ],
                dtype=np.float64,
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
    }


def foot_metrics(
    distances,
    heel_indices,
    toe_indices,
):

    heel_mean = np.mean(
        distances[
            :,
            heel_indices
        ],
        axis=1,
    )

    toe_mean = np.mean(
        distances[
            :,
            toe_indices
        ],
        axis=1,
    )

    foot_mean = np.mean(
        distances[
            :,
            np.concatenate(
                [
                    heel_indices,
                    toe_indices,
                ]
            )
        ],
        axis=1,
    )

    # Positive:
    # heels are farther from terrain than toes,
    # therefore the foot is more toe-down / toe-dominant.
    heel_minus_toe = (
        heel_mean
        -
        toe_mean
    )

    return {
        "heel_mean":
            heel_mean,

        "toe_mean":
            toe_mean,

        "foot_mean":
            foot_mean,

        "heel_minus_toe":
            heel_minus_toe,
    }


def percentile_text(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        f"mean={np.mean(values) * 1000:+8.3f} mm  "
        f"p50={np.percentile(values, 50) * 1000:+8.3f} mm  "
        f"p75={np.percentile(values, 75) * 1000:+8.3f} mm  "
        f"p90={np.percentile(values, 90) * 1000:+8.3f} mm  "
        f"p95={np.percentile(values, 95) * 1000:+8.3f} mm  "
        f"max={np.max(values) * 1000:+8.3f} mm"
    )


def toe_only_like_mask(
    metrics,
    threshold_mm,
):

    threshold_m = (
        float(
            threshold_mm
        )
        /
        1000.0
    )

    # Diagnostic definition:
    #
    # toe mean is at/inside the chosen terrain proximity band
    # while heel mean remains outside that same band.
    #
    # We sweep several thresholds rather than claiming one is
    # the physically correct contact threshold.
    return (
        (
            metrics[
                "toe_mean"
            ]
            <=
            threshold_m
        )
        &
        (
            metrics[
                "heel_mean"
            ]
            >
            threshold_m
        )
    )


def print_mode_summary(
    name,
    distances,
    flat_reference,
):

    left = foot_metrics(
        distances,
        LEFT_HEEL,
        LEFT_TOE,
    )

    right = foot_metrics(
        distances,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    flat_left = foot_metrics(
        flat_reference,
        LEFT_HEEL,
        LEFT_TOE,
    )

    flat_right = foot_metrics(
        flat_reference,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    # Reference-lower-foot labeling:
    # whichever foot has the smaller mean flat-reference
    # sole distance at that phase is treated as the lower /
    # more stance-like reference foot.
    flat_left_is_lower = (
        flat_left[
            "foot_mean"
        ]
        <=
        flat_right[
            "foot_mean"
        ]
    )

    flat_right_is_lower = (
        ~flat_left_is_lower
    )

    print()
    print(
        "=" * 100
    )
    print(
        name
    )
    print(
        "=" * 100
    )

    for (
        side_name,
        metrics,
        flat_metrics,
        lower_mask,
    ) in [
        (
            "LEFT",
            left,
            flat_left,
            flat_left_is_lower,
        ),
        (
            "RIGHT",
            right,
            flat_right,
            flat_right_is_lower,
        ),
    ]:

        print()
        print(
            side_name
        )
        print(
            "-" * 100
        )

        print(
            "heel - toe signed-distance gap "
            "(positive = toe is lower / closer):"
        )

        print(
            "  current:",
            percentile_text(
                metrics[
                    "heel_minus_toe"
                ]
            ),
        )

        print(
            "  flat ref:",
            percentile_text(
                flat_metrics[
                    "heel_minus_toe"
                ]
            ),
        )

        shape_error = (
            metrics[
                "heel_minus_toe"
            ]
            -
            flat_metrics[
                "heel_minus_toe"
            ]
        )

        print(
            "  current - flat gap:",
            percentile_text(
                shape_error
            ),
        )

        lower_count = int(
            np.count_nonzero(
                lower_mask
            )
        )

        print(
            "  reference-lower-foot frames:",
            f"{lower_count}/{EVAL_STATES}",
        )

        for threshold_mm in (
            PROXIMITY_THRESHOLDS_MM
        ):

            mask = toe_only_like_mask(
                metrics,
                threshold_mm,
            )

            all_count = int(
                np.count_nonzero(
                    mask
                )
            )

            lower_toe_only = int(
                np.count_nonzero(
                    mask
                    &
                    lower_mask
                )
            )

            lower_pct = (
                100.0
                *
                lower_toe_only
                /
                max(
                    lower_count,
                    1,
                )
            )

            print(
                f"  toe-only-like @ {threshold_mm:4.0f} mm band: "
                f"all={all_count:3d}/{EVAL_STATES}  "
                f"reference-lower-foot="
                f"{lower_toe_only:3d}/{lower_count} "
                f"({lower_pct:6.2f}%)"
            )

    return {
        "left":
            left,

        "right":
            right,

        "flat_left":
            flat_left,

        "flat_right":
            flat_right,

        "flat_left_is_lower":
            flat_left_is_lower,

        "flat_right_is_lower":
            flat_right_is_lower,
    }


def print_selected_steps(
    rollout_data,
    mode_metrics,
):

    steps = rollout_data[
        "steps"
    ]

    distances = rollout_data[
        "distances"
    ]

    flat_reference = rollout_data[
        "flat_reference"
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "SELECTED REWARD-V1 PPO FRAMES"
    )
    print(
        "=" * 100
    )

    print(
        "step | x       | L heel | L toe  | L h-t  | "
        "flat L h-t | R heel | R toe  | R h-t  | minDist"
    )
    print(
        "-" * 100
    )

    for step in SELECTED_STEPS:

        matches = np.where(
            steps
            ==
            int(
                step
            )
        )[
            0
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely locate step {step}."
            )

        index = int(
            matches[
                0
            ]
        )

        left = mode_metrics[
            "left"
        ]

        right = mode_metrics[
            "right"
        ]

        flat_left = mode_metrics[
            "flat_left"
        ]

        print(
            f"{step:4d} | "
            f"{rollout_data['world_x'][index]:+7.3f} | "
            f"{left['heel_mean'][index] * 1000:+6.1f} | "
            f"{left['toe_mean'][index] * 1000:+6.1f} | "
            f"{left['heel_minus_toe'][index] * 1000:+6.1f} | "
            f"{flat_left['heel_minus_toe'][index] * 1000:+10.1f} | "
            f"{right['heel_mean'][index] * 1000:+6.1f} | "
            f"{right['toe_mean'][index] * 1000:+6.1f} | "
            f"{right['heel_minus_toe'][index] * 1000:+6.1f} | "
            f"{np.min(distances[index]) * 1000:+7.1f}"
        )


def print_worst_left_toe_down(
    rollout_data,
    mode_metrics,
    count=20,
):

    left = mode_metrics[
        "left"
    ]

    flat_left = mode_metrics[
        "flat_left"
    ]

    gap_error = (
        left[
            "heel_minus_toe"
        ]
        -
        flat_left[
            "heel_minus_toe"
        ]
    )

    order = np.argsort(
        gap_error
    )[
        ::-1
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "TOP LEFT-FOOT TOE-DOWN DEVIATIONS FROM SAME-PHASE FLAT BC"
    )
    print(
        "=" * 100
    )

    print(
        "step | x       | L heel | L toe  | "
        "PPO h-t | flat h-t | gap error | ref lower?"
    )
    print(
        "-" * 100
    )

    for index in order[
        :count
    ]:

        print(
            f"{int(rollout_data['steps'][index]):4d} | "
            f"{rollout_data['world_x'][index]:+7.3f} | "
            f"{left['heel_mean'][index] * 1000:+6.1f} | "
            f"{left['toe_mean'][index] * 1000:+6.1f} | "
            f"{left['heel_minus_toe'][index] * 1000:+7.1f} | "
            f"{flat_left['heel_minus_toe'][index] * 1000:+8.1f} | "
            f"{gap_error[index] * 1000:+9.1f} | "
            f"{bool(mode_metrics['flat_left_is_lower'][index])}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Reward-V1 smoke PPO checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 100
    )
    print(
        "UNITREE G1 KINEMATIC PPO FOOT-SUPPORT AUDIT"
    )
    print(
        "=" * 100
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Range:",
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
    )

    print(
        "States:",
        EVAL_STATES,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print(
        "Physics:",
        "NONE -- mj_forward only",
    )

    print(
        "Purpose:"
    )

    print(
        "  Determine whether the observed left-foot "
        "toe-dominant placement is:"
    )

    print(
        "  (a) inherited from the same-phase flat BC reference,"
    )

    print(
        "  (b) already present in BC on uneven terrain, or"
    )

    print(
        "  (c) amplified/created by the Reward-V1 PPO residual."
    )

    print(
        "=" * 100
    )

    bc_env = G1KinematicUnevenEnv(
        episode_length=EVAL_STATES,
        random_start=False,
        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    ppo_env = G1KinematicUnevenEnv(
        episode_length=EVAL_STATES,
        random_start=False,
        residual_smoothing=
            RESIDUAL_SMOOTHING,
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
            "Running BC-only zero-residual rollout..."
        )

        bc = rollout(
            bc_env,
            model=None,
            use_ppo=False,
        )

        print(
            "Running deterministic Reward-V1 PPO rollout..."
        )

        ppo = rollout(
            ppo_env,
            model=model,
            use_ppo=True,
        )

        step_diff = float(
            np.max(
                np.abs(
                    bc[
                        "steps"
                    ]
                    -
                    ppo[
                        "steps"
                    ]
                )
            )
        )

        frame_diff = float(
            np.max(
                np.abs(
                    bc[
                        "motion_frames"
                    ]
                    -
                    ppo[
                        "motion_frames"
                    ]
                )
            )
        )

        x_diff = float(
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

        flat_diff = float(
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
            "  max step difference:",
            step_diff,
        )

        print(
            "  max motion-frame difference:",
            frame_diff,
        )

        print(
            "  max world-X difference:",
            x_diff,
        )

        print(
            "  max flat-reference difference:",
            flat_diff,
        )

        if (
            step_diff
            >
            0.0
            or
            frame_diff
            >
            1e-12
            or
            x_diff
            >
            1e-12
            or
            flat_diff
            >
            1e-12
        ):
            raise RuntimeError(
                "BC and PPO comparison invariants failed."
            )

        # Flat-reference summary:
        # use the same flat reference array as a "current" array
        # so current-flat gap is exactly zero by construction.
        flat_metrics = print_mode_summary(
            "A. SAME-PHASE FLAT BC REFERENCE",
            bc[
                "flat_reference"
            ],
            bc[
                "flat_reference"
            ],
        )

        bc_metrics = print_mode_summary(
            "B. BC-ONLY ON UNEVEN TERRAIN",
            bc[
                "distances"
            ],
            bc[
                "flat_reference"
            ],
        )

        ppo_metrics = print_mode_summary(
            "C. REWARD-V1 PPO ON UNEVEN TERRAIN",
            ppo[
                "distances"
            ],
            ppo[
                "flat_reference"
            ],
        )

        print_selected_steps(
            ppo,
            ppo_metrics,
        )

        print_worst_left_toe_down(
            ppo,
            ppo_metrics,
            count=20,
        )

        print()
        print(
            "=" * 100
        )
        print(
            "AUDIT COMPLETE"
        )
        print(
            "=" * 100
        )

        print(
            "Interpretation rule:"
        )

        print(
            "  If flat BC already has a large positive "
            "left heel-to-toe gap in lower-foot phases,"
        )

        print(
            "  the toe-down posture is inherited from "
            "the reference and Reward V2 alone cannot "
            "be expected to remove it."
        )

        print(
            "  If PPO makes the left gap much larger "
            "than flat BC / BC-only, the residual policy "
            "is creating or amplifying the problem."
        )

        print(
            "  Do NOT modify Reward V2 until these "
            "measured results are reviewed."
        )

        print(
            "=" * 100
        )

    finally:

        bc_env.close()
        ppo_env.close()


if __name__ == "__main__":
    main()