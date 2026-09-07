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


from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)


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
# LEFT JOINT INDICES
#
# 0 = left hip pitch
# 3 = left knee
# ============================================================

LEFT_HIP_PITCH_INDEX = 0
LEFT_KNEE_INDEX = 3


# ============================================================
# SCALE GRID
#
# These are diagnostic action multipliers only.
# Nothing here changes the environment or checkpoint.
# ============================================================

SCALE_VALUES = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.00,
]


# ============================================================
# FOOT SPHERES
# ============================================================

LEFT_HEEL = np.array(
    [0, 1],
    dtype=np.int64,
)

LEFT_TOE = np.array(
    [2, 3],
    dtype=np.int64,
)


# ============================================================
# REGIONS
# ============================================================

REGIONS = [
    (
        "early",
        151,
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
    175,
    250,
    395,
    500,
    594,
    624,
    682,
    689,
    698,
    708,
    716,
    723,
    728,
    741,
    750,
]


# ============================================================
# VALIDATION
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
            "Observation-space shape mismatch."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):
        raise RuntimeError(
            "Action-space shape mismatch."
        )

    if not np.allclose(
        model.observation_space.low,
        env.observation_space.low,
    ):
        raise RuntimeError(
            "Observation lower-bound mismatch."
        )

    if not np.allclose(
        model.observation_space.high,
        env.observation_space.high,
    ):
        raise RuntimeError(
            "Observation upper-bound mismatch."
        )

    if not np.allclose(
        model.action_space.low,
        env.action_space.low,
    ):
        raise RuntimeError(
            "Action lower-bound mismatch."
        )

    if not np.allclose(
        model.action_space.high,
        env.action_space.high,
    ):
        raise RuntimeError(
            "Action upper-bound mismatch."
        )


# ============================================================
# ROLLOUT
# ============================================================

def run_rollout(
    model,
    hip_scale,
    knee_scale,
):

    env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    try:

        obs, reset_info = env.reset(
            seed=SEED,
            options={
                "start_step":
                    TRAIN_START_STEP,
            },
        )

        if (
            int(
                reset_info[
                    "start_step"
                ]
            )
            !=
            TRAIN_START_STEP
        ):
            raise RuntimeError(
                "Unexpected evaluation start step."
            )

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

        flat = [
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

        executed_actions = []
        raw_actions = []
        rewards = []
        penetration_costs = []

        for transition_index in range(
            EVAL_TRANSITIONS
        ):

            raw_action, _ = model.predict(
                obs,
                deterministic=True,
            )

            raw_action = np.asarray(
                raw_action,
                dtype=np.float32,
            )

            if (
                raw_action.shape
                !=
                (12,)
            ):
                raise RuntimeError(
                    "Unexpected PPO action shape."
                )

            action = (
                raw_action.copy()
            )

            action[
                LEFT_HIP_PITCH_INDEX
            ] *= float(
                hip_scale
            )

            action[
                LEFT_KNEE_INDEX
            ] *= float(
                knee_scale
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(
                np.float32
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            raw_actions.append(
                raw_action
                .astype(
                    np.float64
                )
                .copy()
            )

            executed_actions.append(
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

            penetration_costs.append(
                float(
                    info[
                        "penetration_cost"
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

            flat.append(
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

            if (
                transition_index
                <
                EVAL_TRANSITIONS
                -
                1
                and
                (
                    terminated
                    or
                    truncated
                )
            ):
                raise RuntimeError(
                    "Rollout ended early at step "
                    f"{info['step']}."
                )

        data = {
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
                    flat,
                    dtype=np.float64,
                ),

            "residuals":
                np.asarray(
                    residuals,
                    dtype=np.float64,
                ),

            "raw_actions":
                np.asarray(
                    raw_actions,
                    dtype=np.float64,
                ),

            "actions":
                np.asarray(
                    executed_actions,
                    dtype=np.float64,
                ),

            "rewards":
                np.asarray(
                    rewards,
                    dtype=np.float64,
                ),

            "penetration_costs":
                np.asarray(
                    penetration_costs,
                    dtype=np.float64,
                ),
        }

        if (
            data[
                "steps"
            ].shape
            !=
            (
                EVAL_STATES,
            )
        ):
            raise RuntimeError(
                "Unexpected state count."
            )

        if (
            data[
                "distances"
            ].shape
            !=
            (
                EVAL_STATES,
                8,
            )
        ):
            raise RuntimeError(
                "Unexpected distance matrix."
            )

        if (
            data[
                "residuals"
            ].shape
            !=
            (
                EVAL_STATES,
                12,
            )
        ):
            raise RuntimeError(
                "Unexpected residual matrix."
            )

        return data

    finally:

        env.close()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    data,
):

    distances = (
        data[
            "distances"
        ]
    )

    flat = (
        data[
            "flat"
        ]
    )

    penetration = np.maximum(
        -distances,
        0.0,
    )

    penetration_mask = (
        penetration
        >
        0.0
    )

    positive_penetration = (
        penetration[
            penetration_mask
        ]
    )

    deficit = np.maximum(
        flat
        -
        distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    allowed_floor = np.minimum(
        flat,
        0.0,
    )

    phase_penetration = np.maximum(
        allowed_floor
        -
        distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    phase_cost = np.mean(
        (
            phase_penetration
            /
            PENETRATION_SCALE
        )
        **
        2,
        axis=1,
    )

    left_gap = (
        np.mean(
            distances[
                :,
                LEFT_HEEL
            ],
            axis=1,
        )
        -
        np.mean(
            distances[
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

    left_extra = (
        left_gap
        -
        flat_left_gap
    )

    return {
        "penetration":
            penetration,

        "deficit":
            deficit,

        "phase_penetration":
            phase_penetration,

        "phase_cost":
            phase_cost,

        "left_gap":
            left_gap,

        "flat_left_gap":
            flat_left_gap,

        "left_extra":
            left_extra,

        "penetrating_frames":
            int(
                np.count_nonzero(
                    np.any(
                        penetration_mask,
                        axis=1,
                    )
                )
            ),

        "penetrating_samples":
            int(
                np.count_nonzero(
                    penetration_mask
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

        "mean_phase_cost":
            float(
                np.mean(
                    phase_cost
                )
            ),

        "max_phase_penetration":
            float(
                np.max(
                    phase_penetration
                )
            ),

        "mean_left_gap":
            float(
                np.mean(
                    left_gap
                )
            ),

        "mean_left_extra":
            float(
                np.mean(
                    left_extra
                )
            ),
    }


# ============================================================
# REGION METRICS
# ============================================================

def region_metrics(
    data,
    metrics,
    start,
    end,
):

    state_mask = (
        (
            data[
                "steps"
            ]
            >=
            start
        )
        &
        (
            data[
                "steps"
            ]
            <=
            end
        )
    )

    transition_steps = (
        data[
            "steps"
        ][
            1:
        ]
    )

    transition_mask = (
        (
            transition_steps
            >=
            start
        )
        &
        (
            transition_steps
            <=
            end
        )
    )

    region_residual = (
        data[
            "residuals"
        ][
            1:
        ][
            transition_mask
        ]
    )

    return {
        "mean_left_gap":
            float(
                np.mean(
                    metrics[
                        "left_gap"
                    ][
                        state_mask
                    ]
                )
            ),

        "mean_left_extra":
            float(
                np.mean(
                    metrics[
                        "left_extra"
                    ][
                        state_mask
                    ]
                )
            ),

        "max_penetration":
            float(
                np.max(
                    metrics[
                        "penetration"
                    ][
                        state_mask
                    ]
                )
            ),

        "mean_phase_cost":
            float(
                np.mean(
                    metrics[
                        "phase_cost"
                    ][
                        state_mask
                    ]
                )
            ),

        "mean_hip_residual":
            float(
                np.mean(
                    region_residual[
                        :,
                        LEFT_HIP_PITCH_INDEX
                    ]
                )
            ),

        "mean_knee_residual":
            float(
                np.mean(
                    region_residual[
                        :,
                        LEFT_KNEE_INDEX
                    ]
                )
            ),
    }


# ============================================================
# PRINT GLOBAL GRID
# ============================================================

def print_global_grid(
    results,
):

    print()

    print(
        "=" * 190
    )

    print(
        "GLOBAL HIP/KNEE SCALE GRID"
    )

    print(
        "=" * 190
    )

    print(
        "hip | knee | reward  | penFrm | penSamp | "
        "meanPen | maxPen | phaseCost | "
        "meanDef | maxDef | "
        "LgapMean | LextraMean | "
        "dHipP | dKnee"
    )

    print(
        "-" * 190
    )

    for (
        hip_scale,
        knee_scale,
    ) in [
        (
            h,
            k,
        )
        for h in SCALE_VALUES
        for k in SCALE_VALUES
    ]:

        result = (
            results[
                (
                    hip_scale,
                    knee_scale,
                )
            ]
        )

        data = (
            result[
                "data"
            ]
        )

        metrics = (
            result[
                "metrics"
            ]
        )

        residual = (
            data[
                "residuals"
            ][
                1:
            ]
        )

        print(
            f"{hip_scale:3.2f} | "
            f"{knee_scale:4.2f} | "
            f"{np.mean(data['rewards']):+7.4f} | "
            f"{metrics['penetrating_frames']:6d} | "
            f"{metrics['penetrating_samples']:7d} | "
            f"{metrics['mean_penetration'] * 1000:7.2f} | "
            f"{metrics['max_penetration'] * 1000:6.2f} | "
            f"{metrics['mean_phase_cost']:9.5f} | "
            f"{metrics['mean_deficit'] * 1000:7.2f} | "
            f"{metrics['max_deficit'] * 1000:6.2f} | "
            f"{metrics['mean_left_gap'] * 1000:+8.2f} | "
            f"{metrics['mean_left_extra'] * 1000:+10.2f} | "
            f"{np.mean(residual[:, 0]):+6.3f} | "
            f"{np.mean(residual[:, 3]):+6.3f}"
        )


# ============================================================
# REGION TABLE
# ============================================================

def print_region_grid(
    results,
):

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        print()

        print(
            "=" * 155
        )

        print(
            f"REGION: {region_name} "
            f"({start} -> {end})"
        )

        print(
            "=" * 155
        )

        print(
            "hip | knee | "
            "LgapMean | LextraMean | "
            "maxPen | phaseCost | "
            "dHipP | dKnee"
        )

        print(
            "-" * 155
        )

        for hip_scale in (
            SCALE_VALUES
        ):

            for knee_scale in (
                SCALE_VALUES
            ):

                result = (
                    results[
                        (
                            hip_scale,
                            knee_scale,
                        )
                    ]
                )

                region = region_metrics(
                    result[
                        "data"
                    ],

                    result[
                        "metrics"
                    ],

                    start,
                    end,
                )

                print(
                    f"{hip_scale:3.2f} | "
                    f"{knee_scale:4.2f} | "
                    f"{region['mean_left_gap'] * 1000:+8.2f} | "
                    f"{region['mean_left_extra'] * 1000:+10.2f} | "
                    f"{region['max_penetration'] * 1000:6.2f} | "
                    f"{region['mean_phase_cost']:9.5f} | "
                    f"{region['mean_hip_residual']:+6.3f} | "
                    f"{region['mean_knee_residual']:+6.3f}"
                )


# ============================================================
# SELECTED FRAMES
# ============================================================

def print_selected_frames(
    results,
):

    print()

    print(
        "=" * 175
    )

    print(
        "SELECTED FRAME SCALE COMPARISON"
    )

    print(
        "=" * 175
    )

    print(
        "Only several representative scale pairs "
        "are shown here."
    )

    representative = [
        (
            1.00,
            1.00,
        ),

        (
            1.00,
            0.75,
        ),

        (
            1.00,
            0.50,
        ),

        (
            0.75,
            1.00,
        ),

        (
            0.75,
            0.75,
        ),

        (
            0.75,
            0.50,
        ),

        (
            0.50,
            1.00,
        ),

        (
            0.50,
            0.75,
        ),

        (
            0.50,
            0.50,
        ),

        (
            0.00,
            0.00,
        ),
    ]

    print()

    print(
        "step | hip | knee | "
        "pen mm | Lgap | flatL | Lextra | "
        "dHipP | dKnee"
    )

    print(
        "-" * 175
    )

    for step in (
        SELECTED_STEPS
    ):

        for (
            hip_scale,
            knee_scale,
        ) in representative:

            result = (
                results[
                    (
                        hip_scale,
                        knee_scale,
                    )
                ]
            )

            data = (
                result[
                    "data"
                ]
            )

            metrics = (
                result[
                    "metrics"
                ]
            )

            matches = np.where(
                data[
                    "steps"
                ]
                ==
                step
            )[
                0
            ]

            if (
                len(
                    matches
                )
                !=
                1
            ):
                continue

            index = int(
                matches[
                    0
                ]
            )

            residual = (
                data[
                    "residuals"
                ][
                    index
                ]
            )

            penetration = float(
                np.max(
                    metrics[
                        "penetration"
                    ][
                        index
                    ]
                )
            )

            print(
                f"{step:4d} | "
                f"{hip_scale:3.2f} | "
                f"{knee_scale:4.2f} | "
                f"{penetration * 1000:6.1f} | "
                f"{metrics['left_gap'][index] * 1000:+6.1f} | "
                f"{metrics['flat_left_gap'][index] * 1000:+6.1f} | "
                f"{metrics['left_extra'][index] * 1000:+7.1f} | "
                f"{residual[0]:+6.3f} | "
                f"{residual[3]:+6.3f}"
            )

        print(
            "-" * 175
        )


# ============================================================
# FAIR COMPARISON
# ============================================================

def check_invariants(
    results,
):

    full = (
        results[
            (
                1.00,
                1.00,
            )
        ][
            "data"
        ]
    )

    print()

    print(
        "=" * 120
    )

    print(
        "FAIR-COMPARISON INVARIANTS"
    )

    print(
        "=" * 120
    )

    max_step_difference = 0.0
    max_x_difference = 0.0
    max_flat_difference = 0.0

    for result in (
        results.values()
    ):

        data = (
            result[
                "data"
            ]
        )

        max_step_difference = max(
            max_step_difference,

            float(
                np.max(
                    np.abs(
                        data[
                            "steps"
                        ]
                        -
                        full[
                            "steps"
                        ]
                    )
                )
            ),
        )

        max_x_difference = max(
            max_x_difference,

            float(
                np.max(
                    np.abs(
                        data[
                            "world_x"
                        ]
                        -
                        full[
                            "world_x"
                        ]
                    )
                )
            ),
        )

        max_flat_difference = max(
            max_flat_difference,

            float(
                np.max(
                    np.abs(
                        data[
                            "flat"
                        ]
                        -
                        full[
                            "flat"
                        ]
                    )
                )
            ),
        )

    print(
        "Maximum step difference:",
        max_step_difference,
    )

    print(
        "Maximum world-X difference:",
        f"{max_x_difference:.12f}",
    )

    print(
        "Maximum flat-reference difference:",
        f"{max_flat_difference:.12f}",
    )

    if (
        max_step_difference
        !=
        0.0
        or
        max_x_difference
        >
        1e-12
        or
        max_flat_difference
        >
        1e-12
    ):
        raise RuntimeError(
            "Fair-comparison invariant FAILED."
        )

    print(
        "PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Reward-V2 PPO checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 120
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "LEFT HIP/KNEE ACTION-SCALE GRID"
    )

    print(
        "=" * 120
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Scales:",
        SCALE_VALUES,
    )

    print(
        "Total deterministic rollouts:",
        (
            len(
                SCALE_VALUES
            )
            **
            2
        ),
    )

    print()

    print(
        "No training."
    )

    print(
        "No reward modification."
    )

    print(
        "No environment modification."
    )

    print(
        "No mujoco.mj_step()."
    )

    print(
        "=" * 120
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    validation_env = (
        G1KinematicUnevenEnv(
            episode_length=
                EVAL_STATES,

            random_start=False,

            residual_smoothing=
                RESIDUAL_SMOOTHING,
        )
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
            validation_env,
        )

        print(
            "Model-space validation: PASS"
        )

    finally:

        validation_env.close()


    # ========================================================
    # RUN GRID
    # ========================================================

    results = {}

    total = (
        len(
            SCALE_VALUES
        )
        **
        2
    )

    run_number = 0

    for hip_scale in (
        SCALE_VALUES
    ):

        for knee_scale in (
            SCALE_VALUES
        ):

            run_number += 1

            print(
                f"[{run_number:02d}/{total:02d}] "
                f"hip={hip_scale:.2f}, "
                f"knee={knee_scale:.2f}"
            )

            data = run_rollout(
                model,
                hip_scale,
                knee_scale,
            )

            metrics = compute_metrics(
                data
            )

            results[
                (
                    hip_scale,
                    knee_scale,
                )
            ] = {
                "data":
                    data,

                "metrics":
                    metrics,
            }


    # ========================================================
    # CHECK + REPORT
    # ========================================================

    check_invariants(
        results
    )

    print_global_grid(
        results
    )

    print_region_grid(
        results
    )

    print_selected_frames(
        results
    )

    print()

    print(
        "=" * 120
    )

    print(
        "HIP/KNEE SCALE GRID COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "No PPO training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "Do not change Reward V2 or the residual "
        "bounds until this grid is reviewed."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()