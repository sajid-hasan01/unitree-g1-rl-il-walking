import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from envs.g1_kinematic_uneven_env import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_smoke_seed425.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "logs"
    / "ppo_kinematic"
)

DEFAULT_SEED = 425

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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate a trained "
            "G1 kinematic residual PPO policy "
            "against the zero-residual BC-only baseline."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="smoke_seed425",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def require_finite(
    name,
    value,
):

    array = np.asarray(
        value
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):

        raise RuntimeError(
            f"{name} contains NaN/Inf."
        )


def safe_mean(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return 0.0

    return float(
        np.mean(
            values
        )
    )


def safe_median(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return 0.0

    return float(
        np.median(
            values
        )
    )


def safe_max(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return 0.0

    return float(
        np.max(
            values
        )
    )


def percent_reduction(
    baseline,
    ppo,
):

    baseline = float(
        baseline
    )

    ppo = float(
        ppo
    )

    if abs(
        baseline
    ) < 1e-12:
        return 0.0

    return float(
        100.0
        *
        (
            baseline
            -
            ppo
        )
        /
        baseline
    )


# ============================================================
# GEOMETRY METRICS
# ============================================================

def compute_geometry_metrics(
    distances,
    flat_reference,
):

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    flat_reference = np.asarray(
        flat_reference,
        dtype=np.float64,
    )

    if (
        distances.shape
        !=
        flat_reference.shape
    ):

        raise RuntimeError(
            "Distance/reference shape mismatch: "
            f"{distances.shape} vs "
            f"{flat_reference.shape}"
        )

    if (
        distances.ndim
        !=
        2
        or
        distances.shape[
            1
        ]
        !=
        8
    ):

        raise RuntimeError(
            "Expected distance matrix "
            f"(N,8), got "
            f"{distances.shape}"
        )

    # --------------------------------------------------------
    # ACTUAL TERRAIN PENETRATION
    # --------------------------------------------------------

    penetration_mask = (
        distances
        <
        0.0
    )

    penetration_depths = (
        -distances[
            penetration_mask
        ]
    )

    penetrating_frames_mask = np.any(
        penetration_mask,
        axis=1,
    )

    frame_penetration_depth = np.max(
        np.maximum(
            -distances,
            0.0,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # SAME-PHASE FLAT BC DEFICIT
    # --------------------------------------------------------

    deficit = np.maximum(
        flat_reference
        -
        distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    deficit_mask = (
        deficit
        >
        0.0
    )

    deficit_values = (
        deficit[
            deficit_mask
        ]
    )

    deficit_frames_mask = np.any(
        deficit_mask,
        axis=1,
    )

    frame_max_deficit = np.max(
        deficit,
        axis=1,
    )

    frame_mean_deficit = np.mean(
        deficit,
        axis=1,
    )

    # --------------------------------------------------------
    # VERIFIED SPHERE ORDER
    #
    # 0,1 = left heel
    # 2,3 = left toe
    # 4,5 = right heel
    # 6,7 = right toe
    # --------------------------------------------------------

    groups = {
        "left":
            [
                0,
                1,
                2,
                3,
            ],

        "right":
            [
                4,
                5,
                6,
                7,
            ],

        "heel":
            [
                0,
                1,
                4,
                5,
            ],

        "toe":
            [
                2,
                3,
                6,
                7,
            ],
    }

    group_metrics = {}

    for (
        group_name,
        indices,
    ) in groups.items():

        group_distances = (
            distances[
                :,
                indices
            ]
        )

        group_penetration = np.maximum(
            -group_distances,
            0.0,
        )

        group_deficit = (
            deficit[
                :,
                indices
            ]
        )

        positive_penetration = (
            group_penetration[
                group_penetration
                >
                0.0
            ]
        )

        positive_deficit = (
            group_deficit[
                group_deficit
                >
                0.0
            ]
        )

        group_metrics[
            group_name
        ] = {
            "penetrating_samples":
                int(
                    np.sum(
                        group_penetration
                        >
                        0.0
                    )
                ),

            "mean_penetration_depth_m":
                safe_mean(
                    positive_penetration
                ),

            "max_penetration_depth_m":
                safe_max(
                    positive_penetration
                ),

            "deficit_samples":
                int(
                    np.sum(
                        group_deficit
                        >
                        0.0
                    )
                ),

            "mean_deficit_m":
                safe_mean(
                    positive_deficit
                ),

            "max_deficit_m":
                safe_max(
                    positive_deficit
                ),
        }

    return {
        "num_frames":
            int(
                distances.shape[
                    0
                ]
            ),

        "num_sphere_samples":
            int(
                distances.size
            ),

        "penetrating_frames":
            int(
                np.sum(
                    penetrating_frames_mask
                )
            ),

        "penetrating_frame_rate":
            float(
                np.mean(
                    penetrating_frames_mask
                )
            ),

        "penetrating_sphere_samples":
            int(
                np.sum(
                    penetration_mask
                )
            ),

        "penetrating_sphere_rate":
            float(
                np.mean(
                    penetration_mask
                )
            ),

        "mean_penetration_depth_m":
            safe_mean(
                penetration_depths
            ),

        "median_penetration_depth_m":
            safe_median(
                penetration_depths
            ),

        "max_penetration_depth_m":
            safe_max(
                penetration_depths
            ),

        "mean_frame_worst_penetration_m":
            safe_mean(
                frame_penetration_depth
            ),

        "deficit_frames":
            int(
                np.sum(
                    deficit_frames_mask
                )
            ),

        "deficit_frame_rate":
            float(
                np.mean(
                    deficit_frames_mask
                )
            ),

        "deficit_sphere_samples":
            int(
                np.sum(
                    deficit_mask
                )
            ),

        "mean_positive_deficit_m":
            safe_mean(
                deficit_values
            ),

        "median_positive_deficit_m":
            safe_median(
                deficit_values
            ),

        "max_deficit_m":
            safe_max(
                deficit_values
            ),

        "mean_deficit_all_samples_m":
            float(
                np.mean(
                    deficit
                )
            ),

        "mean_frame_max_deficit_m":
            float(
                np.mean(
                    frame_max_deficit
                )
            ),

        "mean_frame_mean_deficit_m":
            float(
                np.mean(
                    frame_mean_deficit
                )
            ),

        "total_squared_deficit_m2":
            float(
                np.sum(
                    deficit
                    **
                    2
                )
            ),

        "groups":
            group_metrics,

        "_frame_penetration_depth":
            frame_penetration_depth,

        "_frame_max_deficit":
            frame_max_deficit,

        "_deficit_matrix":
            deficit,
    }


# ============================================================
# EVALUATION ROLLOUT
# ============================================================

def evaluate_policy_rollout(
    env,
    model,
    seed,
    use_ppo,
):

    observation, reset_info = env.reset(
        seed=seed,
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
            "Evaluation did not start "
            f"at step {TRAIN_START_STEP}."
        )

    require_finite(
        "initial observation",
        observation,
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    steps = [
        int(
            env.current_step
        )
    ]

    motion_frames = [
        float(
            env.current_metadata[
                "motion_frame"
            ]
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
        .copy()
    ]

    flat_reference = [
        env.flat_reference_distances[
            env.current_step
        ]
        .copy()
    ]

    residuals = [
        env.previous_residual
        .copy()
    ]

    actions = []
    rewards = []

    deficit_costs = []
    baseline_deficit_costs = []
    improvements = []
    excess_costs = []
    residual_costs = []
    smoothness_costs = []
    action_costs = []
    clip_costs = []

    terminated_count = 0
    truncated_count = 0

    # --------------------------------------------------------
    # STEP 150 -> 750
    # --------------------------------------------------------

    for transition_index in range(
        EVAL_TRANSITIONS
    ):

        if use_ppo:

            action, _ = model.predict(
                observation,
                state=None,
                episode_start=None,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if (
                action.shape
                !=
                (12,)
            ):

                raise RuntimeError(
                    "PPO predicted unexpected "
                    "action shape: "
                    f"{action.shape}"
                )

        else:

            action = np.zeros(
                12,
                dtype=np.float32,
            )

        require_finite(
            "action",
            action,
        )

        if not (
            env.action_space
            .contains(
                action
            )
        ):

            raise RuntimeError(
                "Evaluation action lies "
                "outside environment "
                "action space."
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        require_finite(
            "next observation",
            next_observation,
        )

        require_finite(
            "reward",
            reward,
        )

        require_finite(
            "signed distances",
            info[
                "signed_distances"
            ],
        )

        require_finite(
            "residual",
            info[
                "residual"
            ],
        )

        if not (
            env.observation_space
            .contains(
                next_observation
            )
        ):

            raise RuntimeError(
                "Evaluation observation "
                "lies outside observation space."
            )

        actions.append(
            action.copy()
        )

        rewards.append(
            float(
                reward
            )
        )

        deficit_costs.append(
            float(
                info[
                    "deficit_cost"
                ]
            )
        )

        baseline_deficit_costs.append(
            float(
                info[
                    "baseline_deficit_cost"
                ]
            )
        )

        improvements.append(
            float(
                info[
                    "improvement"
                ]
            )
        )

        excess_costs.append(
            float(
                info[
                    "excess_cost"
                ]
            )
        )

        residual_costs.append(
            float(
                info[
                    "residual_cost"
                ]
            )
        )

        smoothness_costs.append(
            float(
                info[
                    "smoothness_cost"
                ]
            )
        )

        action_costs.append(
            float(
                info[
                    "action_cost"
                ]
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

        motion_frames.append(
            float(
                info[
                    "motion_frame"
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
                dtype=np.float32,
            )
            .copy()
        )

        flat_reference.append(
            np.asarray(
                info[
                    "flat_reference_distances"
                ],
                dtype=np.float32,
            )
            .copy()
        )

        residuals.append(
            np.asarray(
                info[
                    "residual"
                ],
                dtype=np.float32,
            )
            .copy()
        )

        if terminated:
            terminated_count += 1

        if truncated:
            truncated_count += 1

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
                "Evaluation episode ended "
                "early at step "
                f"{info['step']}."
            )

        observation = (
            next_observation
        )

    # --------------------------------------------------------
    # CONVERT ARRAYS
    # --------------------------------------------------------

    steps = np.asarray(
        steps,
        dtype=np.int32,
    )

    motion_frames = np.asarray(
        motion_frames,
        dtype=np.float64,
    )

    world_x = np.asarray(
        world_x,
        dtype=np.float64,
    )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    flat_reference = np.asarray(
        flat_reference,
        dtype=np.float64,
    )

    residuals = np.asarray(
        residuals,
        dtype=np.float64,
    )

    actions = np.asarray(
        actions,
        dtype=np.float64,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if (
        len(
            steps
        )
        !=
        EVAL_STATES
    ):

        raise RuntimeError(
            "Unexpected number of "
            "evaluated states: "
            f"{len(steps)}"
        )

    if (
        steps[
            0
        ]
        !=
        TRAIN_START_STEP
        or
        steps[
            -1
        ]
        !=
        TRAIN_END_STEP
    ):

        raise RuntimeError(
            "Evaluation step range mismatch: "
            f"{steps[0]} -> "
            f"{steps[-1]}"
        )

    if (
        distances.shape
        !=
        (
            EVAL_STATES,
            8,
        )
    ):

        raise RuntimeError(
            "Unexpected distance shape: "
            f"{distances.shape}"
        )

    if (
        residuals.shape
        !=
        (
            EVAL_STATES,
            12,
        )
    ):

        raise RuntimeError(
            "Unexpected residual shape: "
            f"{residuals.shape}"
        )

    if (
        actions.shape
        !=
        (
            EVAL_TRANSITIONS,
            12,
        )
    ):

        raise RuntimeError(
            "Unexpected action shape: "
            f"{actions.shape}"
        )

    return {
        "steps":
            steps,

        "motion_frames":
            motion_frames,

        "world_x":
            world_x,

        "distances":
            distances,

        "flat_reference":
            flat_reference,

        "residuals":
            residuals,

        "actions":
            actions,

        "rewards":
            rewards,

        "deficit_costs":
            np.asarray(
                deficit_costs,
                dtype=np.float64,
            ),

        "baseline_deficit_costs":
            np.asarray(
                baseline_deficit_costs,
                dtype=np.float64,
            ),

        "improvements":
            np.asarray(
                improvements,
                dtype=np.float64,
            ),

        "excess_costs":
            np.asarray(
                excess_costs,
                dtype=np.float64,
            ),

        "residual_costs":
            np.asarray(
                residual_costs,
                dtype=np.float64,
            ),

        "smoothness_costs":
            np.asarray(
                smoothness_costs,
                dtype=np.float64,
            ),

        "action_costs":
            np.asarray(
                action_costs,
                dtype=np.float64,
            ),

        "clip_costs":
            np.asarray(
                clip_costs,
                dtype=np.float64,
            ),

        "terminated_count":
            int(
                terminated_count
            ),

        "truncated_count":
            int(
                truncated_count
            ),
    }


# ============================================================
# POLICY METRICS
# ============================================================

def compute_policy_metrics(
    rollout,
    residual_scales,
):

    residuals = (
        rollout[
            "residuals"
        ]
    )

    actions = (
        rollout[
            "actions"
        ]
    )

    transition_residuals = (
        residuals[
            1:
        ]
    )

    residual_delta = np.diff(
        residuals,
        axis=0,
    )

    normalized_residual = (
        transition_residuals
        /
        residual_scales[
            None,
            :
        ]
    )

    return {
        "mean_reward_per_transition":
            float(
                np.mean(
                    rollout[
                        "rewards"
                    ]
                )
            ),

        "total_reward":
            float(
                np.sum(
                    rollout[
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

        "rms_action":
            float(
                np.sqrt(
                    np.mean(
                        actions
                        **
                        2
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

        "mean_abs_residual_rad":
            float(
                np.mean(
                    np.abs(
                        transition_residuals
                    )
                )
            ),

        "rms_residual_rad":
            float(
                np.sqrt(
                    np.mean(
                        transition_residuals
                        **
                        2
                    )
                )
            ),

        "max_abs_residual_rad":
            float(
                np.max(
                    np.abs(
                        transition_residuals
                    )
                )
            ),

        "max_abs_normalized_residual":
            float(
                np.max(
                    np.abs(
                        normalized_residual
                    )
                )
            ),

        "mean_abs_residual_delta_rad":
            float(
                np.mean(
                    np.abs(
                        residual_delta
                    )
                )
            ),

        "rms_residual_delta_rad":
            float(
                np.sqrt(
                    np.mean(
                        residual_delta
                        **
                        2
                    )
                )
            ),

        "max_abs_residual_delta_rad":
            float(
                np.max(
                    np.abs(
                        residual_delta
                    )
                )
            ),

        "mean_deficit_cost":
            float(
                np.mean(
                    rollout[
                        "deficit_costs"
                    ]
                )
            ),

        "mean_baseline_deficit_cost":
            float(
                np.mean(
                    rollout[
                        "baseline_deficit_costs"
                    ]
                )
            ),

        "mean_improvement":
            float(
                np.mean(
                    rollout[
                        "improvements"
                    ]
                )
            ),

        "mean_excess_cost":
            float(
                np.mean(
                    rollout[
                        "excess_costs"
                    ]
                )
            ),

        "mean_residual_cost":
            float(
                np.mean(
                    rollout[
                        "residual_costs"
                    ]
                )
            ),

        "mean_smoothness_cost":
            float(
                np.mean(
                    rollout[
                        "smoothness_costs"
                    ]
                )
            ),

        "mean_action_cost":
            float(
                np.mean(
                    rollout[
                        "action_costs"
                    ]
                )
            ),

        "max_joint_clip_cost":
            float(
                np.max(
                    rollout[
                        "clip_costs"
                    ]
                )
            ),
    }


# ============================================================
# JSON HELPERS
# ============================================================

def strip_private_metrics(
    metrics,
):

    return {
        key:
            value

        for (
            key,
            value,
        ) in metrics.items()

        if not str(
            key
        ).startswith(
            "_"
        )
    }


# ============================================================
# PRINT GEOMETRY
# ============================================================

def print_geometry_summary(
    title,
    metrics,
):

    print()

    print(
        title
    )

    print(
        "-" * 100
    )

    print(
        "Frames:",
        metrics[
            "num_frames"
        ],
    )

    print(
        "Penetrating frames:",
        (
            f"{metrics['penetrating_frames']} "
            f"({100.0 * metrics['penetrating_frame_rate']:.2f}%)"
        ),
    )

    print(
        "Penetrating sphere samples:",
        (
            f"{metrics['penetrating_sphere_samples']} "
            f"({100.0 * metrics['penetrating_sphere_rate']:.2f}%)"
        ),
    )

    print(
        "Mean penetration depth:",
        (
            f"{metrics['mean_penetration_depth_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Median penetration depth:",
        (
            f"{metrics['median_penetration_depth_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Maximum penetration depth:",
        (
            f"{metrics['max_penetration_depth_m'] * 1000:.3f} mm"
        ),
    )

    print()

    print(
        "Flat-reference deficit frames:",
        (
            f"{metrics['deficit_frames']} "
            f"({100.0 * metrics['deficit_frame_rate']:.2f}%)"
        ),
    )

    print(
        "Mean positive deficit:",
        (
            f"{metrics['mean_positive_deficit_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Maximum deficit:",
        (
            f"{metrics['max_deficit_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Mean deficit across all sphere samples:",
        (
            f"{metrics['mean_deficit_all_samples_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Mean frame maximum deficit:",
        (
            f"{metrics['mean_frame_max_deficit_m'] * 1000:.3f} mm"
        ),
    )

    print(
        "Total squared deficit:",
        (
            f"{metrics['total_squared_deficit_m2']:.8f} m^2"
        ),
    )

    print()

    print(
        "Group metrics:"
    )

    for group_name in [
        "left",
        "right",
        "heel",
        "toe",
    ]:

        group = (
            metrics[
                "groups"
            ][
                group_name
            ]
        )

        print(
            f"  {group_name:5s}: "
            f"penetrating={group['penetrating_samples']:4d}, "
            f"mean_depth="
            f"{group['mean_penetration_depth_m'] * 1000:7.3f}mm, "
            f"max_depth="
            f"{group['max_penetration_depth_m'] * 1000:7.3f}mm, "
            f"deficit_samples={group['deficit_samples']:4d}, "
            f"mean_deficit="
            f"{group['mean_deficit_m'] * 1000:7.3f}mm, "
            f"max_deficit="
            f"{group['max_deficit_m'] * 1000:7.3f}mm"
        )


# ============================================================
# SAVE PER-STEP CSV
# ============================================================

def save_per_step_csv(
    path,
    baseline_rollout,
    ppo_rollout,
    baseline_metrics,
    ppo_metrics,
):

    baseline_deficit = (
        baseline_metrics[
            "_deficit_matrix"
        ]
    )

    ppo_deficit = (
        ppo_metrics[
            "_deficit_matrix"
        ]
    )

    baseline_penetration = (
        baseline_metrics[
            "_frame_penetration_depth"
        ]
    )

    ppo_penetration = (
        ppo_metrics[
            "_frame_penetration_depth"
        ]
    )

    baseline_max_deficit = (
        baseline_metrics[
            "_frame_max_deficit"
        ]
    )

    ppo_max_deficit = (
        ppo_metrics[
            "_frame_max_deficit"
        ]
    )

    header = [
        "step",
        "motion_frame",
        "world_x",

        "bc_frame_penetration_m",
        "ppo_frame_penetration_m",

        "bc_frame_max_deficit_m",
        "ppo_frame_max_deficit_m",

        "frame_max_deficit_improvement_m",
    ]

    header += [
        f"bc_distance_sphere_{index}_m"

        for index in range(
            8
        )
    ]

    header += [
        f"ppo_distance_sphere_{index}_m"

        for index in range(
            8
        )
    ]

    header += [
        f"flat_reference_sphere_{index}_m"

        for index in range(
            8
        )
    ]

    header += [
        f"bc_deficit_sphere_{index}_m"

        for index in range(
            8
        )
    ]

    header += [
        f"ppo_deficit_sphere_{index}_m"

        for index in range(
            8
        )
    ]

    header += [
        f"ppo_residual_joint_{index}_rad"

        for index in range(
            12
        )
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            header
        )

        for row_index in range(
            EVAL_STATES
        ):

            row = [
                int(
                    baseline_rollout[
                        "steps"
                    ][
                        row_index
                    ]
                ),

                float(
                    baseline_rollout[
                        "motion_frames"
                    ][
                        row_index
                    ]
                ),

                float(
                    baseline_rollout[
                        "world_x"
                    ][
                        row_index
                    ]
                ),

                float(
                    baseline_penetration[
                        row_index
                    ]
                ),

                float(
                    ppo_penetration[
                        row_index
                    ]
                ),

                float(
                    baseline_max_deficit[
                        row_index
                    ]
                ),

                float(
                    ppo_max_deficit[
                        row_index
                    ]
                ),

                float(
                    baseline_max_deficit[
                        row_index
                    ]
                    -
                    ppo_max_deficit[
                        row_index
                    ]
                ),
            ]

            row.extend(
                baseline_rollout[
                    "distances"
                ][
                    row_index
                ]
                .tolist()
            )

            row.extend(
                ppo_rollout[
                    "distances"
                ][
                    row_index
                ]
                .tolist()
            )

            row.extend(
                baseline_rollout[
                    "flat_reference"
                ][
                    row_index
                ]
                .tolist()
            )

            row.extend(
                baseline_deficit[
                    row_index
                ]
                .tolist()
            )

            row.extend(
                ppo_deficit[
                    row_index
                ]
                .tolist()
            )

            row.extend(
                ppo_rollout[
                    "residuals"
                ][
                    row_index
                ]
                .tolist()
            )

            writer.writerow(
                row
            )


# ============================================================
# SPACE VALIDATION
# ============================================================

def compare_space(
    saved_space,
    env_space,
    name,
):

    if (
        saved_space.shape
        !=
        env_space.shape
    ):

        raise RuntimeError(
            f"Saved PPO {name} shape "
            f"{saved_space.shape} "
            f"does not match current env "
            f"{env_space.shape}."
        )

    if (
        saved_space.dtype
        !=
        env_space.dtype
    ):

        raise RuntimeError(
            f"Saved PPO {name} dtype "
            f"{saved_space.dtype} "
            f"does not match current env "
            f"{env_space.dtype}."
        )

    if not np.allclose(
        saved_space.low,
        env_space.low,
    ):

        raise RuntimeError(
            f"Saved PPO {name} lower "
            "bounds do not match current "
            "environment."
        )

    if not np.allclose(
        saved_space.high,
        env_space.high,
    ):

        raise RuntimeError(
            f"Saved PPO {name} upper "
            "bounds do not match current "
            "environment."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    model_path = (
        args.model
        .expanduser()
        .resolve()
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "PPO model not found:\n"
            f"{model_path}"
        )

    eval_name = (
        args.name
        .strip()
        .replace(
            " ",
            "_",
        )
    )

    if not eval_name:

        raise ValueError(
            "--name cannot be empty."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        /
        f"g1_kinematic_ppo_{eval_name}_evaluation.json"
    )

    csv_path = (
        OUTPUT_DIR
        /
        f"g1_kinematic_ppo_{eval_name}_per_step.csv"
    )

    print(
        "=" * 100
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "DETERMINISTIC BC-vs-PPO EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        "Model:",
        model_path,
    )

    print(
        "Start step:",
        TRAIN_START_STEP,
    )

    print(
        "End step:",
        TRAIN_END_STEP,
    )

    print(
        "Evaluated states:",
        EVAL_STATES,
    )

    print(
        "Transitions:",
        EVAL_TRANSITIONS,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print(
        "Flat-reference tolerance:",
        (
            f"{FLAT_DISTANCE_TOLERANCE * 1000:.1f} mm"
        ),
    )

    print(
        "Physics:",
        "NONE -- mj_forward only",
    )

    # --------------------------------------------------------
    # TWO IDENTICAL ENVIRONMENTS
    # --------------------------------------------------------

    baseline_env = (
        G1KinematicUnevenEnv(
            episode_length=
                EVAL_STATES,

            random_start=False,

            residual_smoothing=
                RESIDUAL_SMOOTHING,
        )
    )

    ppo_env = (
        G1KinematicUnevenEnv(
            episode_length=
                EVAL_STATES,

            random_start=False,

            residual_smoothing=
                RESIDUAL_SMOOTHING,
        )
    )

    try:

        # ----------------------------------------------------
        # LOAD PPO
        # ----------------------------------------------------

        print()

        print(
            "Loading PPO model..."
        )

        model = PPO.load(
            model_path,
            env=None,
            device=
                args.device,
            force_reset=True,
        )

        print(
            "PPO model loaded."
        )

        print(
            "Stored observation space:",
            model.observation_space,
        )

        print(
            "Stored action space:",
            model.action_space,
        )

        compare_space(
            model.observation_space,
            ppo_env.observation_space,
            "observation space",
        )

        compare_space(
            model.action_space,
            ppo_env.action_space,
            "action space",
        )

        # ----------------------------------------------------
        # BC-ONLY
        # ----------------------------------------------------

        print()

        print(
            "Running BC-only "
            "zero-residual trajectory..."
        )

        baseline_rollout = (
            evaluate_policy_rollout(
                baseline_env,
                model=None,
                seed=
                    args.seed,
                use_ppo=False,
            )
        )

        print(
            "BC-only evaluation complete."
        )

        # ----------------------------------------------------
        # DETERMINISTIC PPO
        # ----------------------------------------------------

        print(
            "Running deterministic PPO "
            "trajectory..."
        )

        ppo_rollout = (
            evaluate_policy_rollout(
                ppo_env,
                model=model,
                seed=
                    args.seed,
                use_ppo=True,
            )
        )

        print(
            "PPO evaluation complete."
        )

        # ----------------------------------------------------
        # FAIR COMPARISON INVARIANTS
        # ----------------------------------------------------

        step_difference = float(
            np.max(
                np.abs(
                    baseline_rollout[
                        "steps"
                    ]
                    -
                    ppo_rollout[
                        "steps"
                    ]
                )
            )
        )

        frame_difference = float(
            np.max(
                np.abs(
                    baseline_rollout[
                        "motion_frames"
                    ]
                    -
                    ppo_rollout[
                        "motion_frames"
                    ]
                )
            )
        )

        x_difference = float(
            np.max(
                np.abs(
                    baseline_rollout[
                        "world_x"
                    ]
                    -
                    ppo_rollout[
                        "world_x"
                    ]
                )
            )
        )

        flat_reference_difference = float(
            np.max(
                np.abs(
                    baseline_rollout[
                        "flat_reference"
                    ]
                    -
                    ppo_rollout[
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
            step_difference,
        )

        print(
            "  max motion-frame difference:",
            frame_difference,
        )

        print(
            "  max world-X difference:",
            x_difference,
        )

        print(
            "  max flat-reference difference:",
            flat_reference_difference,
        )

        if (
            step_difference
            !=
            0.0
            or
            frame_difference
            >
            1e-12
            or
            x_difference
            >
            1e-12
            or
            flat_reference_difference
            >
            1e-12
        ):

            raise RuntimeError(
                "BC/PPO evaluation trajectories "
                "are not identical in "
                "reference timing."
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        baseline_geometry = (
            compute_geometry_metrics(
                baseline_rollout[
                    "distances"
                ],

                baseline_rollout[
                    "flat_reference"
                ],
            )
        )

        ppo_geometry = (
            compute_geometry_metrics(
                ppo_rollout[
                    "distances"
                ],

                ppo_rollout[
                    "flat_reference"
                ],
            )
        )

        baseline_policy = (
            compute_policy_metrics(
                baseline_rollout,

                baseline_env
                .residual_scales
                .astype(
                    np.float64
                ),
            )
        )

        ppo_policy = (
            compute_policy_metrics(
                ppo_rollout,

                ppo_env
                .residual_scales
                .astype(
                    np.float64
                ),
            )
        )

        # ----------------------------------------------------
        # GEOMETRIC SUMMARY
        # ----------------------------------------------------

        print()

        print(
            "=" * 100
        )

        print(
            "GEOMETRIC RESULTS"
        )

        print(
            "=" * 100
        )

        print_geometry_summary(
            "BC-ONLY",
            baseline_geometry,
        )

        print_geometry_summary(
            "DETERMINISTIC PPO",
            ppo_geometry,
        )

        # ----------------------------------------------------
        # PRECOMPUTE ALL PERCENTAGES
        #
        # This avoids multiline expressions inside f-strings.
        # ----------------------------------------------------

        penetrating_frame_reduction = (
            percent_reduction(
                baseline_geometry[
                    "penetrating_frames"
                ],

                ppo_geometry[
                    "penetrating_frames"
                ],
            )
        )

        penetrating_sphere_reduction = (
            percent_reduction(
                baseline_geometry[
                    "penetrating_sphere_samples"
                ],

                ppo_geometry[
                    "penetrating_sphere_samples"
                ],
            )
        )

        mean_penetration_reduction = (
            percent_reduction(
                baseline_geometry[
                    "mean_penetration_depth_m"
                ],

                ppo_geometry[
                    "mean_penetration_depth_m"
                ],
            )
        )

        max_penetration_reduction = (
            percent_reduction(
                baseline_geometry[
                    "max_penetration_depth_m"
                ],

                ppo_geometry[
                    "max_penetration_depth_m"
                ],
            )
        )

        deficit_frame_reduction = (
            percent_reduction(
                baseline_geometry[
                    "deficit_frames"
                ],

                ppo_geometry[
                    "deficit_frames"
                ],
            )
        )

        mean_deficit_reduction = (
            percent_reduction(
                baseline_geometry[
                    "mean_deficit_all_samples_m"
                ],

                ppo_geometry[
                    "mean_deficit_all_samples_m"
                ],
            )
        )

        max_deficit_reduction = (
            percent_reduction(
                baseline_geometry[
                    "max_deficit_m"
                ],

                ppo_geometry[
                    "max_deficit_m"
                ],
            )
        )

        squared_deficit_reduction = (
            percent_reduction(
                baseline_geometry[
                    "total_squared_deficit_m2"
                ],

                ppo_geometry[
                    "total_squared_deficit_m2"
                ],
            )
        )

        # ----------------------------------------------------
        # CHANGE SUMMARY
        # ----------------------------------------------------

        print()

        print(
            "=" * 100
        )

        print(
            "BC -> PPO CHANGE"
        )

        print(
            "=" * 100
        )

        print(
            "Penetrating-frame reduction:",
            (
                f"{penetrating_frame_reduction:+.2f}%"
            ),
        )

        print(
            "Penetrating-sphere-sample reduction:",
            (
                f"{penetrating_sphere_reduction:+.2f}%"
            ),
        )

        print(
            "Mean penetration-depth reduction:",
            (
                f"{mean_penetration_reduction:+.2f}%"
            ),
        )

        print(
            "Maximum penetration-depth reduction:",
            (
                f"{max_penetration_reduction:+.2f}%"
            ),
        )

        print(
            "Deficit-frame reduction:",
            (
                f"{deficit_frame_reduction:+.2f}%"
            ),
        )

        print(
            "Mean all-sample deficit reduction:",
            (
                f"{mean_deficit_reduction:+.2f}%"
            ),
        )

        print(
            "Maximum deficit reduction:",
            (
                f"{max_deficit_reduction:+.2f}%"
            ),
        )

        print(
            "Total squared-deficit reduction:",
            (
                f"{squared_deficit_reduction:+.2f}%"
            ),
        )

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward_change = (
            ppo_policy[
                "mean_reward_per_transition"
            ]
            -
            baseline_policy[
                "mean_reward_per_transition"
            ]
        )

        print()

        print(
            "Mean reward/transition:"
        )

        print(
            "  BC-only:",
            (
                f"{baseline_policy['mean_reward_per_transition']:+.6f}"
            ),
        )

        print(
            "  PPO:",
            (
                f"{ppo_policy['mean_reward_per_transition']:+.6f}"
            ),
        )

        print(
            "  change:",
            (
                f"{reward_change:+.6f}"
            ),
        )

        # ----------------------------------------------------
        # POLICY BEHAVIOR
        # ----------------------------------------------------

        print()

        print(
            "PPO policy behavior:"
        )

        print(
            "  mean |action|:",
            (
                f"{ppo_policy['mean_abs_action']:.6f}"
            ),
        )

        print(
            "  max |action|:",
            (
                f"{ppo_policy['max_abs_action']:.6f}"
            ),
        )

        print(
            "  mean |residual|:",
            (
                f"{ppo_policy['mean_abs_residual_rad']:.6f} rad"
            ),
        )

        print(
            "  max |residual|:",
            (
                f"{ppo_policy['max_abs_residual_rad']:.6f} rad"
            ),
        )

        print(
            "  max normalized residual:",
            (
                f"{ppo_policy['max_abs_normalized_residual']:.6f}"
            ),
        )

        print(
            "  mean |residual delta|:",
            (
                f"{ppo_policy['mean_abs_residual_delta_rad']:.6f} rad"
            ),
        )

        print(
            "  max joint clip cost:",
            (
                f"{ppo_policy['max_joint_clip_cost']:.9f}"
            ),
        )

        # ----------------------------------------------------
        # PER-FRAME COMPARISON
        # ----------------------------------------------------

        bc_frame_deficit = (
            baseline_geometry[
                "_frame_max_deficit"
            ]
        )

        ppo_frame_deficit = (
            ppo_geometry[
                "_frame_max_deficit"
            ]
        )

        frame_improvement = (
            bc_frame_deficit
            -
            ppo_frame_deficit
        )

        better_frames = int(
            np.sum(
                frame_improvement
                >
                1e-9
            )
        )

        same_frames = int(
            np.sum(
                np.abs(
                    frame_improvement
                )
                <=
                1e-9
            )
        )

        worse_frames = int(
            np.sum(
                frame_improvement
                <
                -1e-9
            )
        )

        print()

        print(
            "Per-frame max-deficit outcome:"
        )

        print(
            "  PPO better:",
            better_frames,
        )

        print(
            "  same:",
            same_frames,
        )

        print(
            "  PPO worse:",
            worse_frames,
        )

        # ----------------------------------------------------
        # 10 WORST BC FRAMES
        # ----------------------------------------------------

        worst_bc_indices = (
            np.argsort(
                bc_frame_deficit
            )[
                ::-1
            ][
                :10
            ]
        )

        print()

        print(
            "10 worst BC baseline frames:"
        )

        print(
            "step | frame   | x       | "
            "BC deficit mm | "
            "PPO deficit mm | "
            "improvement mm"
        )

        print(
            "-" * 82
        )

        for index in (
            worst_bc_indices
        ):

            print(
                f"{baseline_rollout['steps'][index]:4d} | "
                f"{baseline_rollout['motion_frames'][index]:7.2f} | "
                f"{baseline_rollout['world_x'][index]:+7.3f} | "
                f"{bc_frame_deficit[index] * 1000:13.3f} | "
                f"{ppo_frame_deficit[index] * 1000:14.3f} | "
                f"{frame_improvement[index] * 1000:+14.3f}"
            )

        # ----------------------------------------------------
        # 10 WORST REMAINING PPO FRAMES
        # ----------------------------------------------------

        worst_ppo_indices = (
            np.argsort(
                ppo_frame_deficit
            )[
                ::-1
            ][
                :10
            ]
        )

        print()

        print(
            "10 worst remaining PPO frames:"
        )

        print(
            "step | frame   | x       | "
            "BC deficit mm | "
            "PPO deficit mm | "
            "improvement mm"
        )

        print(
            "-" * 82
        )

        for index in (
            worst_ppo_indices
        ):

            print(
                f"{baseline_rollout['steps'][index]:4d} | "
                f"{baseline_rollout['motion_frames'][index]:7.2f} | "
                f"{baseline_rollout['world_x'][index]:+7.3f} | "
                f"{bc_frame_deficit[index] * 1000:13.3f} | "
                f"{ppo_frame_deficit[index] * 1000:14.3f} | "
                f"{frame_improvement[index] * 1000:+14.3f}"
            )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        output = {
            "model_path":
                str(
                    model_path
                ),

            "seed":
                int(
                    args.seed
                ),

            "deterministic":
                True,

            "evaluation_range": {
                "start_step":
                    TRAIN_START_STEP,

                "end_step":
                    TRAIN_END_STEP,

                "states":
                    EVAL_STATES,

                "transitions":
                    EVAL_TRANSITIONS,

                "residual_smoothing":
                    RESIDUAL_SMOOTHING,

                "flat_distance_tolerance_m":
                    FLAT_DISTANCE_TOLERANCE,
            },

            "fair_comparison_invariants": {
                "max_step_difference":
                    step_difference,

                "max_motion_frame_difference":
                    frame_difference,

                "max_world_x_difference":
                    x_difference,

                "max_flat_reference_difference":
                    flat_reference_difference,
            },

            "baseline_geometry":
                strip_private_metrics(
                    baseline_geometry
                ),

            "ppo_geometry":
                strip_private_metrics(
                    ppo_geometry
                ),

            "baseline_policy_metrics":
                baseline_policy,

            "ppo_policy_metrics":
                ppo_policy,

            "comparison": {
                "penetrating_frame_reduction_percent":
                    penetrating_frame_reduction,

                "penetrating_sphere_sample_reduction_percent":
                    penetrating_sphere_reduction,

                "mean_penetration_depth_reduction_percent":
                    mean_penetration_reduction,

                "max_penetration_depth_reduction_percent":
                    max_penetration_reduction,

                "deficit_frame_reduction_percent":
                    deficit_frame_reduction,

                "mean_all_sample_deficit_reduction_percent":
                    mean_deficit_reduction,

                "max_deficit_reduction_percent":
                    max_deficit_reduction,

                "total_squared_deficit_reduction_percent":
                    squared_deficit_reduction,

                "ppo_better_frames":
                    better_frames,

                "same_frames":
                    same_frames,

                "ppo_worse_frames":
                    worse_frames,
            },
        }

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        save_per_step_csv(
            csv_path,
            baseline_rollout,
            ppo_rollout,
            baseline_geometry,
            ppo_geometry,
        )

        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        print()

        print(
            "=" * 100
        )

        print(
            "OUTPUT FILES"
        )

        print(
            "=" * 100
        )

        print(
            "Summary JSON:",
            json_path,
        )

        print(
            "Per-step CSV:",
            csv_path,
        )

        print()

        print(
            "=" * 100
        )

        print(
            "DETERMINISTIC EVALUATION COMPLETE"
        )

        print(
            "=" * 100
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No mujoco.mj_step() was called."
        )

        print(
            "Do not increase the PPO training "
            "budget until these metrics "
            "are reviewed."
        )

    finally:

        baseline_env.close()

        ppo_env.close()


if __name__ == "__main__":
    main()