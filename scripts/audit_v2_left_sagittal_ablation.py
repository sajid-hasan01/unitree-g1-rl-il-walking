from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)

SEED = 425
RESIDUAL_SMOOTHING = 0.35

EVAL_STATES = TRAIN_END_STEP - TRAIN_START_STEP + 1
EVAL_TRANSITIONS = TRAIN_END_STEP - TRAIN_START_STEP

LEFT_HEEL = np.array(
    [0, 1],
    dtype=np.int64,
)

LEFT_TOE = np.array(
    [2, 3],
    dtype=np.int64,
)


ABLATIONS = [
    (
        "FULL_V2",
        [],
    ),

    (
        "NO_LEFT_HIP_PITCH",
        [0],
    ),

    (
        "NO_LEFT_KNEE",
        [3],
    ),

    (
        "NO_LEFT_HIP_AND_KNEE",
        [0, 3],
    ),

    (
        "NO_LEFT_SAGITTAL",
        [0, 3, 4],
    ),
]


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


def run_mode(
    model,
    blocked_indices,
):

    env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    try:

        obs, info = env.reset(
            seed=SEED,
            options={
                "start_step":
                    TRAIN_START_STEP,
            },
        )

        if (
            int(
                info[
                    "start_step"
                ]
            )
            !=
            TRAIN_START_STEP
        ):
            raise RuntimeError(
                "Unexpected reset start step."
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

        rewards = []
        penetration_costs = []
        executed_actions = []
        raw_actions = []

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

            action = (
                raw_action.copy()
            )

            if blocked_indices:

                action[
                    np.asarray(
                        blocked_indices,
                        dtype=np.int64,
                    )
                ] = 0.0

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
                EVAL_TRANSITIONS - 1
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

            "executed_actions":
                np.asarray(
                    executed_actions,
                    dtype=np.float64,
                ),

            "raw_actions":
                np.asarray(
                    raw_actions,
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
                "Unexpected distance shape."
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
                "Unexpected residual shape."
            )

        if (
            data[
                "rewards"
            ].shape
            !=
            (
                EVAL_TRANSITIONS,
            )
        ):
            raise RuntimeError(
                "Unexpected reward shape."
            )

        if blocked_indices:

            blocked_residual = (
                data[
                    "residuals"
                ][
                    :,
                    blocked_indices
                ]
            )

            max_blocked_residual = float(
                np.max(
                    np.abs(
                        blocked_residual
                    )
                )
            )

            if (
                max_blocked_residual
                >
                1e-7
            ):
                raise RuntimeError(
                    "Blocked residual did not remain zero: "
                    f"{max_blocked_residual}"
                )

        return data

    finally:

        env.close()


def geometry_metrics(
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

    penetration_values = (
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

    gap = (
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

    flat_gap = (
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

    return {
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
                        penetration_values
                    )
                )
                if
                penetration_values.size
                else
                0.0
            ),

        "max_penetration":
            float(
                np.max(
                    penetration
                )
            ),

        "mean_all_deficit":
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

        "left_gap":
            gap,

        "flat_left_gap":
            flat_gap,

        "left_extra":
            (
                gap
                -
                flat_gap
            ),
    }


def policy_metrics(
    data,
):

    residuals = (
        data[
            "residuals"
        ][
            1:
        ]
    )

    actions = (
        data[
            "executed_actions"
        ]
    )

    return {
        "mean_reward":
            float(
                np.mean(
                    data[
                        "rewards"
                    ]
                )
            ),

        "mean_abs_action":
            float(
                np.mean(
                    np.abs(
                        actions
                    )
                )
            ),

        "max_abs_action":
            float(
                np.max(
                    np.abs(
                        actions
                    )
                )
            ),

        "near_boundary_actions":
            int(
                np.count_nonzero(
                    np.abs(
                        actions
                    )
                    >=
                    0.95
                )
            ),

        "mean_left_hip":
            float(
                np.mean(
                    residuals[
                        :,
                        0
                    ]
                )
            ),

        "mean_left_knee":
            float(
                np.mean(
                    residuals[
                        :,
                        3
                    ]
                )
            ),

        "mean_left_ankle":
            float(
                np.mean(
                    residuals[
                        :,
                        4
                    ]
                )
            ),

        "left_sagittal_rms":
            float(
                np.sqrt(
                    np.mean(
                        residuals[
                            :,
                            [
                                0,
                                3,
                                4,
                            ]
                        ]
                        **
                        2
                    )
                )
            ),
    }


def print_summary(
    results,
):

    print()

    print(
        "=" * 190
    )

    print(
        "COUNTERFACTUAL LEFT-SAGITTAL ABLATION SUMMARY"
    )

    print(
        "=" * 190
    )

    print(
        "mode                   | reward   | penFrm | penSamp | "
        "meanPen | maxPen | phaseCost | meanDef | maxDef | "
        "LgapMean | LextraMean | dHipP | dKnee | dAnkP | LsagRMS"
    )

    print(
        "-" * 190
    )

    for (
        name,
        result,
    ) in results.items():

        g = result[
            "geometry"
        ]

        p = result[
            "policy"
        ]

        print(
            f"{name:22s} | "
            f"{p['mean_reward']:+8.4f} | "
            f"{g['penetrating_frames']:6d} | "
            f"{g['penetrating_samples']:7d} | "
            f"{g['mean_penetration'] * 1000:7.2f} | "
            f"{g['max_penetration'] * 1000:6.2f} | "
            f"{g['mean_phase_cost']:9.5f} | "
            f"{g['mean_all_deficit'] * 1000:7.2f} | "
            f"{g['max_deficit'] * 1000:6.2f} | "
            f"{np.mean(g['left_gap']) * 1000:+8.2f} | "
            f"{np.mean(g['left_extra']) * 1000:+10.2f} | "
            f"{p['mean_left_hip']:+6.3f} | "
            f"{p['mean_left_knee']:+6.3f} | "
            f"{p['mean_left_ankle']:+6.3f} | "
            f"{p['left_sagittal_rms']:7.4f}"
        )


def print_regions(
    results,
):

    print()

    print(
        "=" * 155
    )

    print(
        "REGION COMPARISON"
    )

    print(
        "=" * 155
    )

    print(
        "mode                   | region       | "
        "LgapMean | LextraMean | maxPen | phaseCost | "
        "dHipP | dKnee | dAnkP"
    )

    print(
        "-" * 155
    )

    for (
        name,
        result,
    ) in results.items():

        data = result[
            "data"
        ]

        g = result[
            "geometry"
        ]

        transition_steps = (
            data[
                "steps"
            ][
                1:
            ]
        )

        for (
            region_name,
            start,
            end,
        ) in REGIONS:

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

            distances = (
                data[
                    "distances"
                ][
                    state_mask
                ]
            )

            flat = (
                data[
                    "flat"
                ][
                    state_mask
                ]
            )

            penetration = np.maximum(
                -distances,
                0.0,
            )

            allowed_floor = np.minimum(
                flat,
                0.0,
            )

            phase_pen = np.maximum(
                allowed_floor
                -
                distances
                -
                FLAT_DISTANCE_TOLERANCE,
                0.0,
            )

            phase_cost = np.mean(
                (
                    phase_pen
                    /
                    PENETRATION_SCALE
                )
                **
                2,
                axis=1,
            )

            residual = (
                data[
                    "residuals"
                ][
                    1:
                ][
                    transition_mask
                ]
            )

            print(
                f"{name:22s} | "
                f"{region_name:12s} | "
                f"{np.mean(g['left_gap'][state_mask]) * 1000:+8.2f} | "
                f"{np.mean(g['left_extra'][state_mask]) * 1000:+10.2f} | "
                f"{np.max(penetration) * 1000:6.2f} | "
                f"{np.mean(phase_cost):9.5f} | "
                f"{np.mean(residual[:, 0]):+6.3f} | "
                f"{np.mean(residual[:, 3]):+6.3f} | "
                f"{np.mean(residual[:, 4]):+6.3f}"
            )


def print_selected(
    results,
):

    print()

    print(
        "=" * 165
    )

    print(
        "SELECTED FRAME COMPARISON"
    )

    print(
        "=" * 165
    )

    print(
        "step | mode                   | "
        "pen mm | Lgap | flatL | Lextra | "
        "dHipP | dKnee | dAnkP"
    )

    print(
        "-" * 165
    )

    for step in (
        SELECTED_STEPS
    ):

        for (
            name,
            result,
        ) in results.items():

            data = result[
                "data"
            ]

            g = result[
                "geometry"
            ]

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

            i = int(
                matches[
                    0
                ]
            )

            residual = (
                data[
                    "residuals"
                ][
                    i
                ]
            )

            penetration = float(
                np.max(
                    np.maximum(
                        -data[
                            "distances"
                        ][
                            i
                        ],
                        0.0,
                    )
                )
            )

            print(
                f"{step:4d} | "
                f"{name:22s} | "
                f"{penetration * 1000:6.1f} | "
                f"{g['left_gap'][i] * 1000:+6.1f} | "
                f"{g['flat_left_gap'][i] * 1000:+6.1f} | "
                f"{g['left_extra'][i] * 1000:+7.1f} | "
                f"{residual[0]:+6.3f} | "
                f"{residual[3]:+6.3f} | "
                f"{residual[4]:+6.3f}"
            )

        print(
            "-" * 165
        )


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
        "LEFT-SAGITTAL COUNTERFACTUAL ABLATION"
    )

    print(
        "=" * 120
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "No PPO training."
    )

    print(
        "No reward modification."
    )

    print(
        "No mujoco.mj_step()."
    )

    print()

    print(
        "Diagnostic method: run the SAME deterministic PPO, "
        "but force selected normalized action dimensions to zero."
    )

    print(
        "Because residual state starts at zero, "
        "a blocked action dimension remains zero "
        "through the rollout."
    )

    print()

    print(
        "Ablations:"
    )

    for (
        name,
        blocked,
    ) in ABLATIONS:

        print(
            f"  {name:22s} -> "
            f"blocked action indices {blocked}"
        )

    print(
        "=" * 120
    )

    validation_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

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
            validation_env,
        )

        print(
            "Model-space validation: PASS"
        )

    finally:

        validation_env.close()

    results = {}

    for (
        name,
        blocked,
    ) in ABLATIONS:

        print()

        print(
            f"Running {name}..."
        )

        data = run_mode(
            model,
            blocked,
        )

        results[
            name
        ] = {
            "data":
                data,

            "geometry":
                geometry_metrics(
                    data
                ),

            "policy":
                policy_metrics(
                    data
                ),
        }

        print(
            f"  {name}: PASS"
        )

    full_steps = (
        results[
            "FULL_V2"
        ][
            "data"
        ][
            "steps"
        ]
    )

    full_x = (
        results[
            "FULL_V2"
        ][
            "data"
        ][
            "world_x"
        ]
    )

    full_flat = (
        results[
            "FULL_V2"
        ][
            "data"
        ][
            "flat"
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

    for (
        name,
        result,
    ) in results.items():

        data = result[
            "data"
        ]

        step_diff = float(
            np.max(
                np.abs(
                    data[
                        "steps"
                    ]
                    -
                    full_steps
                )
            )
        )

        x_diff = float(
            np.max(
                np.abs(
                    data[
                        "world_x"
                    ]
                    -
                    full_x
                )
            )
        )

        flat_diff = float(
            np.max(
                np.abs(
                    data[
                        "flat"
                    ]
                    -
                    full_flat
                )
            )
        )

        print(
            f"{name:22s} | "
            f"stepDiff={step_diff:.1f} | "
            f"xDiff={x_diff:.12f} | "
            f"flatDiff={flat_diff:.12f}"
        )

        if (
            step_diff
            !=
            0.0
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
                "Fair-comparison invariant failed "
                f"for {name}."
            )

    print_summary(
        results
    )

    print_regions(
        results
    )

    print_selected(
        results
    )

    print()

    print(
        "=" * 120
    )

    print(
        "ABLATION COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "Interpretation:"
    )

    print(
        "  If removing left hip pitch or left knee "
        "strongly reduces the distortion without "
        "destroying terrain geometry, that joint is "
        "a strong candidate for stronger BC-pose "
        "regularization."
    )

    print(
        "  If penetration becomes much worse, "
        "that residual is currently doing useful "
        "terrain work and should not simply be removed."
    )

    print(
        "No environment/model file was modified."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()