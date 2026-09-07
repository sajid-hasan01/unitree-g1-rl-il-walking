from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# EXACT BALANCED-V2 TRAINING CONSTANTS
#
# Import these from the already-tested trainer rather than
# duplicating them manually.
# ============================================================

from scripts.train_g1_kinematic_ppo_v2_balanced import (
    LEARNING_RATE,
    N_STEPS,
    BATCH_SIZE,
    N_EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    ENT_COEF,
    VF_COEF,
    MAX_GRAD_NORM,
    TARGET_KL,
    POLICY_NET_ARCH,
    DEFAULT_SEED,
    DEFAULT_RESIDUAL_SMOOTHING,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
)


# ============================================================
# BALANCED TRAINING ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2_balanced import (
    G1KinematicUnevenEnv as BalancedTrainingEnv,
)


# ============================================================
# ORIGINAL V2 ENVIRONMENT FOR FAIR DETERMINISTIC EVALUATION
#
# Evaluation uses explicit step 150 -> 750, therefore the
# random-start sampler is irrelevant during evaluation.
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv as EvaluationEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    DEFAULT_EPISODE_LENGTH,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
    / "ppo_kinematic"
)


# ============================================================
# DEFAULT EXPERIMENT SETTINGS
# ============================================================

DEFAULT_TIMESTEPS = 65536

DEFAULT_CHECKPOINT_INTERVAL = 4096

EVAL_STATE_COUNT = (
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
# EXISTING CURRENT CHAMPION
#
# This is evaluated only AFTER the new training run finishes.
# It is never loaded during training, so it cannot disturb
# training RNG state.
# ============================================================

CURRENT_CHAMPION_MODEL = (
    MODEL_DIR
    / "g1_kinematic_ppo_v2_balanced_long65k_seed425.zip"
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train Balanced Reward-V2 PPO from scratch while "
            "saving and deterministically evaluating post-update "
            "checkpoints."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="v2_balanced_checkpointed65k_seed425",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
    )

    parser.add_argument(
        "--residual-smoothing",
        type=float,
        default=DEFAULT_RESIDUAL_SMOOTHING,
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_args(
    args,
):

    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps must be positive."
        )

    if args.checkpoint_interval <= 0:
        raise ValueError(
            "--checkpoint-interval must be positive."
        )

    if args.episode_length <= 0:
        raise ValueError(
            "--episode-length must be positive."
        )

    if not (
        0.0
        <
        args.residual_smoothing
        <=
        1.0
    ):
        raise ValueError(
            "--residual-smoothing must be in (0, 1]."
        )

    if (
        N_STEPS
        %
        BATCH_SIZE
        !=
        0
    ):
        raise RuntimeError(
            "N_STEPS must be divisible by BATCH_SIZE."
        )

    if (
        args.checkpoint_interval
        %
        N_STEPS
        !=
        0
    ):
        raise ValueError(
            "--checkpoint-interval must be divisible "
            f"by PPO n_steps={N_STEPS}. "
            "This guarantees checkpoints occur after "
            "complete PPO rollout/update blocks."
        )

    if (
        args.timesteps
        %
        args.checkpoint_interval
        !=
        0
    ):
        raise ValueError(
            "--timesteps must be divisible by "
            "--checkpoint-interval for this controlled run."
        )


# ============================================================
# EXACT ORIGINAL SEEDING
# ============================================================

def set_training_seeds(
    seed,
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# RNG PRESERVATION
#
# Deterministic evaluation must not change the random stream
# subsequently used by PPO training.
# ============================================================

@contextmanager
def preserve_global_rng():

    python_state = random.getstate()

    numpy_state = np.random.get_state()

    torch_cpu_state = torch.get_rng_state()

    if torch.cuda.is_available():

        torch_cuda_states = (
            torch.cuda.get_rng_state_all()
        )

    else:

        torch_cuda_states = None

    try:

        yield

    finally:

        random.setstate(
            python_state
        )

        np.random.set_state(
            numpy_state
        )

        torch.set_rng_state(
            torch_cpu_state
        )

        if (
            torch_cuda_states
            is not None
        ):

            torch.cuda.set_rng_state_all(
                torch_cuda_states
            )


# ============================================================
# MODEL-SPACE CHECK
# ============================================================

def validate_model_spaces(
    model,
    env,
    name,
):

    if (
        model.observation_space.shape
        !=
        env.observation_space.shape
    ):

        raise RuntimeError(
            f"{name}: observation shape mismatch."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):

        raise RuntimeError(
            f"{name}: action shape mismatch."
        )

    if not np.allclose(
        model.observation_space.low,
        env.observation_space.low,
    ):

        raise RuntimeError(
            f"{name}: observation lower bounds mismatch."
        )

    if not np.allclose(
        model.observation_space.high,
        env.observation_space.high,
    ):

        raise RuntimeError(
            f"{name}: observation upper bounds mismatch."
        )

    if not np.allclose(
        model.action_space.low,
        env.action_space.low,
    ):

        raise RuntimeError(
            f"{name}: action lower bounds mismatch."
        )

    if not np.allclose(
        model.action_space.high,
        env.action_space.high,
    ):

        raise RuntimeError(
            f"{name}: action upper bounds mismatch."
        )


# ============================================================
# DETERMINISTIC GEOMETRY EVALUATION
# ============================================================

def evaluate_geometry(
    model,
    eval_env,
    seed,
):

    policy_training_state = bool(
        model.policy.training
    )

    with preserve_global_rng():

        observation, reset_info = eval_env.reset(
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
                "Evaluation did not begin at "
                f"step {TRAIN_START_STEP}."
            )


        # ----------------------------------------------------
        # STATE ARRAYS INCLUDE INITIAL STEP 150
        # ----------------------------------------------------

        distances = [
            np.asarray(
                eval_env.current_distances,
                dtype=np.float64,
            ).copy()
        ]

        flat_reference = [
            np.asarray(
                eval_env.flat_reference_distances[
                    eval_env.current_step
                ],
                dtype=np.float64,
            ).copy()
        ]

        steps = [
            int(
                eval_env.current_step
            )
        ]

        rewards = []

        clip_costs = []


        # ----------------------------------------------------
        # 600 DETERMINISTIC TRANSITIONS: 150 -> 750
        # ----------------------------------------------------

        for transition_index in range(
            EVAL_TRANSITIONS
        ):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if action.shape != (12,):

                raise RuntimeError(
                    "Unexpected deterministic action shape: "
                    f"{action.shape}"
                )

            if not eval_env.action_space.contains(
                action
            ):

                raise RuntimeError(
                    "Deterministic action is outside "
                    "the environment action space."
                )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = eval_env.step(
                action
            )

            if terminated:

                raise RuntimeError(
                    "Unexpected kinematic termination "
                    f"at step {info['step']}."
                )

            if (
                transition_index
                <
                EVAL_TRANSITIONS
                -
                1
                and
                truncated
            ):

                raise RuntimeError(
                    "Evaluation truncated early at "
                    f"step {info['step']}."
                )

            if not np.all(
                np.isfinite(
                    observation
                )
            ):

                raise RuntimeError(
                    "Non-finite evaluation observation."
                )

            if not np.isfinite(
                reward
            ):

                raise RuntimeError(
                    "Non-finite evaluation reward."
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

            steps.append(
                int(
                    info[
                        "step"
                    ]
                )
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


        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        flat_reference = np.asarray(
            flat_reference,
            dtype=np.float64,
        )

        steps = np.asarray(
            steps,
            dtype=np.int64,
        )

        rewards = np.asarray(
            rewards,
            dtype=np.float64,
        )

        clip_costs = np.asarray(
            clip_costs,
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # STRUCTURE CHECKS
        # ----------------------------------------------------

        if distances.shape != (
            EVAL_STATE_COUNT,
            8,
        ):

            raise RuntimeError(
                "Evaluation distance shape mismatch: "
                f"{distances.shape}"
            )

        if flat_reference.shape != (
            EVAL_STATE_COUNT,
            8,
        ):

            raise RuntimeError(
                "Flat-reference shape mismatch."
            )

        if (
            int(
                steps[
                    0
                ]
            )
            !=
            TRAIN_START_STEP
        ):

            raise RuntimeError(
                "Wrong first evaluation state."
            )

        if (
            int(
                steps[
                    -1
                ]
            )
            !=
            TRAIN_END_STEP
        ):

            raise RuntimeError(
                "Wrong final evaluation state."
            )


        # ====================================================
        # RAW ABSOLUTE PENETRATION
        # ====================================================

        penetration = np.maximum(
            -distances,
            0.0,
        )

        frame_max_penetration = np.max(
            penetration,
            axis=1,
        )

        positive_penetration = penetration[
            penetration
            >
            0.0
        ]


        # ====================================================
        # SAME-PHASE DEFICIT
        # ====================================================

        deficit = np.maximum(
            flat_reference
            -
            distances
            -
            FLAT_DISTANCE_TOLERANCE,
            0.0,
        )


        # ====================================================
        # EXACT V2 PHASE-AWARE PENETRATION METRIC
        # ====================================================

        allowed_floor = np.minimum(
            flat_reference,
            0.0,
        )

        phase_excess = np.maximum(
            allowed_floor
            -
            distances
            -
            FLAT_DISTANCE_TOLERANCE,
            0.0,
        )

        phase_cost_per_frame = np.mean(
            (
                phase_excess
                /
                PENETRATION_SCALE
            )
            **
            2,
            axis=1,
        )


        # ====================================================
        # REMAINING 65K HOTSPOT
        # ====================================================

        hotspot_mask = (
            (
                steps
                >=
                685
            )
            &
            (
                steps
                <=
                700
            )
        )

        hotspot_frame_pen = (
            frame_max_penetration[
                hotspot_mask
            ]
        )


        # ====================================================
        # METRICS
        # ====================================================

        metrics = {
            "penetrating_frames":
                int(
                    np.count_nonzero(
                        frame_max_penetration
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

            "mean_positive_penetration_m":
                float(
                    np.mean(
                        positive_penetration
                    )
                    if
                    positive_penetration.size
                    else
                    0.0
                ),

            "mean_frame_max_penetration_m":
                float(
                    np.mean(
                        frame_max_penetration
                    )
                ),

            "max_penetration_m":
                float(
                    np.max(
                        frame_max_penetration
                    )
                ),

            "frames_ge_20mm":
                int(
                    np.count_nonzero(
                        frame_max_penetration
                        >=
                        0.020
                    )
                ),

            "frames_ge_30mm":
                int(
                    np.count_nonzero(
                        frame_max_penetration
                        >=
                        0.030
                    )
                ),

            "frames_ge_40mm":
                int(
                    np.count_nonzero(
                        frame_max_penetration
                        >=
                        0.040
                    )
                ),

            "frames_ge_50mm":
                int(
                    np.count_nonzero(
                        frame_max_penetration
                        >=
                        0.050
                    )
                ),

            "frames_ge_60mm":
                int(
                    np.count_nonzero(
                        frame_max_penetration
                        >=
                        0.060
                    )
                ),

            "mean_deficit_m":
                float(
                    np.mean(
                        deficit
                    )
                ),

            "max_deficit_m":
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
                        phase_cost_per_frame
                    )
                ),

            "max_phase_excess_m":
                float(
                    np.max(
                        phase_excess
                    )
                ),

            "mean_v2_reward":
                float(
                    np.mean(
                        rewards
                    )
                ),

            "max_joint_clip_cost":
                float(
                    np.max(
                        clip_costs
                    )
                    if
                    clip_costs.size
                    else
                    0.0
                ),

            "hotspot_685_700_max_penetration_m":
                float(
                    np.max(
                        hotspot_frame_pen
                    )
                ),

            "hotspot_685_700_mean_max_penetration_m":
                float(
                    np.mean(
                        hotspot_frame_pen
                    )
                ),

            "hotspot_685_700_frames_ge_40mm":
                int(
                    np.count_nonzero(
                        hotspot_frame_pen
                        >=
                        0.040
                    )
                ),
        }


    # --------------------------------------------------------
    # Restore original policy train/eval mode as well.
    # --------------------------------------------------------

    model.policy.set_training_mode(
        policy_training_state
    )

    return metrics


# ============================================================
# CHECKPOINT SELECTION RULE
#
# Penetration is primary.
#
# Lower tuple is better:
#
# 1. eliminate >=50 mm failures
# 2. minimize absolute worst penetration
# 3. minimize >=40 mm failures
# 4. minimize >=30 mm failures
# 5. minimize >=20 mm failures
# 6. minimize all penetrating frames
# 7. minimize mean frame maximum
# 8. minimize V2 phase cost
# 9. minimize squared deficit
# ============================================================

def selection_key(
    metrics,
):

    return (
        int(
            metrics[
                "frames_ge_50mm"
            ]
        ),

        float(
            metrics[
                "max_penetration_m"
            ]
        ),

        int(
            metrics[
                "frames_ge_40mm"
            ]
        ),

        int(
            metrics[
                "frames_ge_30mm"
            ]
        ),

        int(
            metrics[
                "frames_ge_20mm"
            ]
        ),

        int(
            metrics[
                "penetrating_frames"
            ]
        ),

        float(
            metrics[
                "mean_frame_max_penetration_m"
            ]
        ),

        float(
            metrics[
                "mean_phase_cost"
            ]
        ),

        float(
            metrics[
                "squared_deficit"
            ]
        ),
    )


# ============================================================
# CSV ROW
# ============================================================

CSV_FIELDS = [
    "label",
    "timesteps",
    "checkpoint",
    "penetrating_frames",
    "penetrating_samples",
    "mean_positive_penetration_mm",
    "mean_frame_max_penetration_mm",
    "max_penetration_mm",
    "frames_ge_20mm",
    "frames_ge_30mm",
    "frames_ge_40mm",
    "frames_ge_50mm",
    "frames_ge_60mm",
    "mean_deficit_mm",
    "max_deficit_mm",
    "squared_deficit",
    "mean_phase_cost",
    "max_phase_excess_mm",
    "mean_v2_reward",
    "max_joint_clip_cost",
    "hotspot_685_700_max_penetration_mm",
    "hotspot_685_700_mean_max_penetration_mm",
    "hotspot_685_700_frames_ge_40mm",
]


def metrics_to_row(
    label,
    timesteps,
    checkpoint,
    metrics,
):

    return {
        "label":
            label,

        "timesteps":
            int(
                timesteps
            ),

        "checkpoint":
            str(
                checkpoint
            ),

        "penetrating_frames":
            metrics[
                "penetrating_frames"
            ],

        "penetrating_samples":
            metrics[
                "penetrating_samples"
            ],

        "mean_positive_penetration_mm":
            (
                metrics[
                    "mean_positive_penetration_m"
                ]
                *
                1000.0
            ),

        "mean_frame_max_penetration_mm":
            (
                metrics[
                    "mean_frame_max_penetration_m"
                ]
                *
                1000.0
            ),

        "max_penetration_mm":
            (
                metrics[
                    "max_penetration_m"
                ]
                *
                1000.0
            ),

        "frames_ge_20mm":
            metrics[
                "frames_ge_20mm"
            ],

        "frames_ge_30mm":
            metrics[
                "frames_ge_30mm"
            ],

        "frames_ge_40mm":
            metrics[
                "frames_ge_40mm"
            ],

        "frames_ge_50mm":
            metrics[
                "frames_ge_50mm"
            ],

        "frames_ge_60mm":
            metrics[
                "frames_ge_60mm"
            ],

        "mean_deficit_mm":
            (
                metrics[
                    "mean_deficit_m"
                ]
                *
                1000.0
            ),

        "max_deficit_mm":
            (
                metrics[
                    "max_deficit_m"
                ]
                *
                1000.0
            ),

        "squared_deficit":
            metrics[
                "squared_deficit"
            ],

        "mean_phase_cost":
            metrics[
                "mean_phase_cost"
            ],

        "max_phase_excess_mm":
            (
                metrics[
                    "max_phase_excess_m"
                ]
                *
                1000.0
            ),

        "mean_v2_reward":
            metrics[
                "mean_v2_reward"
            ],

        "max_joint_clip_cost":
            metrics[
                "max_joint_clip_cost"
            ],

        "hotspot_685_700_max_penetration_mm":
            (
                metrics[
                    "hotspot_685_700_max_penetration_m"
                ]
                *
                1000.0
            ),

        "hotspot_685_700_mean_max_penetration_mm":
            (
                metrics[
                    "hotspot_685_700_mean_max_penetration_m"
                ]
                *
                1000.0
            ),

        "hotspot_685_700_frames_ge_40mm":
            metrics[
                "hotspot_685_700_frames_ge_40mm"
            ],
    }


# ============================================================
# PRINT ONE CHECKPOINT
# ============================================================

def print_metrics(
    label,
    timesteps,
    metrics,
):

    print(
        f"{label:18s} | "
        f"steps={timesteps:6d} | "
        f"penFrm={metrics['penetrating_frames']:3d} | "
        f"max={metrics['max_penetration_m'] * 1000:7.3f} mm | "
        f">=20={metrics['frames_ge_20mm']:3d} | "
        f">=30={metrics['frames_ge_30mm']:3d} | "
        f">=40={metrics['frames_ge_40mm']:3d} | "
        f">=50={metrics['frames_ge_50mm']:3d} | "
        f"phase={metrics['mean_phase_cost']:.6f} | "
        f"def2={metrics['squared_deficit']:.6f} | "
        f"R={metrics['mean_v2_reward']:+.6f} | "
        f"hot685-700="
        f"{metrics['hotspot_685_700_max_penetration_m'] * 1000:7.3f} mm"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    run_name = (
        args.run_name
        .strip()
        .replace(
            " ",
            "_",
        )
    )

    if not run_name:
        raise ValueError(
            "--run-name cannot be empty."
        )


    # ========================================================
    # PATHS
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        MODEL_DIR
        /
        f"{run_name}_checkpoints"
    )

    best_model_path = (
        MODEL_DIR
        /
        f"g1_kinematic_ppo_{run_name}_best_geometry.zip"
    )

    final_model_path = (
        MODEL_DIR
        /
        f"g1_kinematic_ppo_{run_name}_final.zip"
    )

    csv_path = (
        LOG_DIR
        /
        f"g1_kinematic_ppo_{run_name}_geometry.csv"
    )

    json_path = (
        LOG_DIR
        /
        f"g1_kinematic_ppo_{run_name}_summary.json"
    )


    # ========================================================
    # PROTECT EXISTING WORK
    # ========================================================

    protected_paths = [
        checkpoint_dir,
        best_model_path,
        final_model_path,
        csv_path,
        json_path,
    ]

    for path in protected_paths:

        if path.exists():

            raise FileExistsError(
                "Refusing to overwrite existing output:\n"
                f"{path}\n"
                "Use a different --run-name."
            )


    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print(
        "=" * 125
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "BALANCED REWARD-V2 CHECKPOINTED TRAINING"
    )

    print(
        "=" * 125
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "  Find the best deterministic geometry checkpoint "
        "during a controlled Balanced-V2 training run."
    )

    print()

    print(
        "Training from scratch:",
        True,
    )

    print(
        "Reward:",
        "V2 unchanged",
    )

    print(
        "Balanced start range:",
        f"{BALANCED_START_MIN} -> {BALANCED_START_MAX}",
    )

    print(
        "Train range:",
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
    )

    print(
        "Timesteps:",
        args.timesteps,
    )

    print(
        "Checkpoint interval:",
        args.checkpoint_interval,
    )

    print(
        "Number of evaluated checkpoints:",
        (
            args.timesteps
            //
            args.checkpoint_interval
        ),
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Residual smoothing:",
        args.residual_smoothing,
    )

    print()

    print(
        "PPO:"
    )

    print(
        "  learning_rate:",
        LEARNING_RATE,
    )

    print(
        "  n_steps:",
        N_STEPS,
    )

    print(
        "  batch_size:",
        BATCH_SIZE,
    )

    print(
        "  n_epochs:",
        N_EPOCHS,
    )

    print(
        "  gamma:",
        GAMMA,
    )

    print(
        "  gae_lambda:",
        GAE_LAMBDA,
    )

    print(
        "  clip_range:",
        CLIP_RANGE,
    )

    print(
        "  ent_coef:",
        ENT_COEF,
    )

    print(
        "  vf_coef:",
        VF_COEF,
    )

    print(
        "  max_grad_norm:",
        MAX_GRAD_NORM,
    )

    print(
        "  target_kl:",
        TARGET_KL,
    )

    print(
        "  network:",
        POLICY_NET_ARCH,
    )

    print()

    print(
        "Selection priority:"
    )

    print(
        "  >=50 count -> max penetration -> >=40 -> "
        ">=30 -> >=20 -> penetrating frames -> "
        "mean max -> phase cost -> squared deficit"
    )

    print()

    print(
        "Current 65K champion is NOT loaded during training."
    )

    print(
        "It will be evaluated only after the new run finishes."
    )

    print()

    print(
        "No V3."
    )

    print(
        "No reward modification."
    )

    print(
        "No action-bound modification."
    )

    print(
        "No residual-scale modification."
    )

    print(
        "No mj_step()."
    )

    print(
        "=" * 125
    )


    # ========================================================
    # SEED BEFORE CREATING TRAINING ENV / PPO
    # ========================================================

    set_training_seeds(
        args.seed
    )


    training_env = None

    evaluation_env = None

    model = None

    training_start_time = (
        time.perf_counter()
    )

    rows = []

    checkpoint_results = []

    best_key = None

    best_checkpoint_path = None

    best_checkpoint_metrics = None

    best_checkpoint_timesteps = None


    try:

        # ====================================================
        # TRAINING ENVIRONMENT
        # ====================================================

        print()

        print(
            "Creating balanced V2 training environment..."
        )

        training_env = BalancedTrainingEnv(
            episode_length=
                args.episode_length,

            random_start=
                True,

            residual_smoothing=
                args.residual_smoothing,
        )

        print(
            "Training environment created."
        )


        # ====================================================
        # FRESH PPO MODEL
        # ====================================================

        print(
            "Creating fresh PPO model..."
        )

        policy_kwargs = {
            "net_arch":
                list(
                    POLICY_NET_ARCH
                ),
        }

        model = PPO(
            policy=
                "MlpPolicy",

            env=
                training_env,

            learning_rate=
                LEARNING_RATE,

            n_steps=
                N_STEPS,

            batch_size=
                BATCH_SIZE,

            n_epochs=
                N_EPOCHS,

            gamma=
                GAMMA,

            gae_lambda=
                GAE_LAMBDA,

            clip_range=
                CLIP_RANGE,

            normalize_advantage=
                True,

            ent_coef=
                ENT_COEF,

            vf_coef=
                VF_COEF,

            max_grad_norm=
                MAX_GRAD_NORM,

            use_sde=
                False,

            target_kl=
                TARGET_KL,

            policy_kwargs=
                policy_kwargs,

            verbose=
                1,

            seed=
                args.seed,

            device=
                args.device,
        )

        print(
            "Fresh PPO model created."
        )


        # ====================================================
        # EVALUATION ENVIRONMENT
        #
        # Creation is RNG-guarded so loading its frozen BC
        # components cannot change PPO's training RNG stream.
        # ====================================================

        with preserve_global_rng():

            evaluation_env = EvaluationEnv(
                episode_length=
                    EVAL_STATE_COUNT,

                random_start=
                    False,

                residual_smoothing=
                    args.residual_smoothing,
            )

        validate_model_spaces(
            model,
            evaluation_env,
            "Fresh PPO",
        )

        print(
            "Evaluation environment created."
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # CSV INITIALIZATION
        # ====================================================

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=
                    CSV_FIELDS,
            )

            writer.writeheader()


        # ====================================================
        # SEGMENTED TRAINING
        #
        # First segment uses reset_num_timesteps=True,
        # exactly like the original trainer.
        #
        # Later segments use False so SB3 keeps the current
        # training environment state and timestep counter.
        # ====================================================

        print()

        print(
            "=" * 125
        )

        print(
            "CHECKPOINTED BALANCED-V2 TRAINING START"
        )

        print(
            "=" * 125
        )

        first_segment = True

        while (
            model.num_timesteps
            <
            args.timesteps
        ):

            previous_timesteps = int(
                model.num_timesteps
            )

            expected_after = (
                previous_timesteps
                +
                args.checkpoint_interval
            )

            print()

            print(
                "-" * 125
            )

            print(
                f"TRAIN SEGMENT: "
                f"{previous_timesteps} -> {expected_after}"
            )

            print(
                "-" * 125
            )

            model.learn(
                total_timesteps=
                    args.checkpoint_interval,

                log_interval=
                    1,

                tb_log_name=
                    run_name,

                reset_num_timesteps=
                    first_segment,

                progress_bar=
                    False,
            )

            first_segment = False


            # ------------------------------------------------
            # EXACT TIMESTEP CHECK
            # ------------------------------------------------

            actual_timesteps = int(
                model.num_timesteps
            )

            if (
                actual_timesteps
                !=
                expected_after
            ):

                raise RuntimeError(
                    "Unexpected PPO timestep count after "
                    "segment. "
                    f"Expected {expected_after}, "
                    f"got {actual_timesteps}."
                )


            # ------------------------------------------------
            # SAVE POST-UPDATE CHECKPOINT
            # ------------------------------------------------

            checkpoint_path = (
                checkpoint_dir
                /
                (
                    f"{run_name}_"
                    f"{actual_timesteps:06d}_steps.zip"
                )
            )

            model.save(
                checkpoint_path
            )

            if not checkpoint_path.exists():

                raise RuntimeError(
                    "Checkpoint save failed:\n"
                    f"{checkpoint_path}"
                )


            # ------------------------------------------------
            # DETERMINISTIC GEOMETRY EVALUATION
            # ------------------------------------------------

            metrics = evaluate_geometry(
                model=
                    model,

                eval_env=
                    evaluation_env,

                seed=
                    args.seed,
            )


            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            print_metrics(
                label=
                    "CHECKPOINT",

                timesteps=
                    actual_timesteps,

                metrics=
                    metrics,
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            row = metrics_to_row(
                label=
                    "checkpoint",

                timesteps=
                    actual_timesteps,

                checkpoint=
                    checkpoint_path,

                metrics=
                    metrics,
            )

            rows.append(
                row
            )

            with csv_path.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=
                        CSV_FIELDS,
                )

                writer.writerow(
                    row
                )


            # ------------------------------------------------
            # BEST-GEOMETRY SELECTION
            # ------------------------------------------------

            key = selection_key(
                metrics
            )

            checkpoint_results.append(
                {
                    "timesteps":
                        actual_timesteps,

                    "checkpoint":
                        str(
                            checkpoint_path
                        ),

                    "metrics":
                        metrics,

                    "selection_key":
                        list(
                            key
                        ),
                }
            )

            if (
                best_key
                is None
                or
                key
                <
                best_key
            ):

                best_key = key

                best_checkpoint_path = (
                    checkpoint_path
                )

                best_checkpoint_metrics = (
                    metrics
                )

                best_checkpoint_timesteps = (
                    actual_timesteps
                )

                shutil.copy2(
                    checkpoint_path,
                    best_model_path,
                )

                print(
                    "  NEW BEST GEOMETRY CHECKPOINT"
                )


        # ====================================================
        # SAVE FINAL MODEL SEPARATELY
        # ====================================================

        model.save(
            final_model_path
        )

        if not final_model_path.exists():

            raise RuntimeError(
                "Final model save failed."
            )


        # ====================================================
        # TRAINING COMPLETE
        # ====================================================

        training_elapsed = (
            time.perf_counter()
            -
            training_start_time
        )

        print()

        print(
            "=" * 125
        )

        print(
            "CHECKPOINTED TRAINING FINISHED"
        )

        print(
            "=" * 125
        )

        print(
            "Actual PPO timesteps:",
            model.num_timesteps,
        )

        print(
            "Elapsed seconds:",
            f"{training_elapsed:.2f}",
        )

        print(
            "Best new checkpoint:",
            best_checkpoint_path,
        )

        print(
            "Best new checkpoint timesteps:",
            best_checkpoint_timesteps,
        )

        print()

        print(
            "Best-new geometry:"
        )

        print_metrics(
            label=
                "BEST NEW",

            timesteps=
                best_checkpoint_timesteps,

            metrics=
                best_checkpoint_metrics,
        )


        # ====================================================
        # NOW EVALUATE EXISTING 65K CHAMPION
        #
        # This happens only after training is completely over,
        # so loading another PPO model cannot affect training.
        # ====================================================

        champion_metrics = None

        champion_key = None

        if CURRENT_CHAMPION_MODEL.exists():

            print()

            print(
                "=" * 125
            )

            print(
                "EVALUATING EXISTING 65K CHAMPION"
            )

            print(
                "=" * 125
            )

            champion = PPO.load(
                CURRENT_CHAMPION_MODEL,
                env=None,
                device="cpu",
                force_reset=True,
            )

            validate_model_spaces(
                champion,
                evaluation_env,
                "Existing 65K champion",
            )

            champion_metrics = evaluate_geometry(
                model=
                    champion,

                eval_env=
                    evaluation_env,

                seed=
                    args.seed,
            )

            champion_key = selection_key(
                champion_metrics
            )

            print_metrics(
                label=
                    "CURRENT 65K",

                timesteps=
                    65536,

                metrics=
                    champion_metrics,
            )

            champion_row = metrics_to_row(
                label=
                    "existing_65k_champion",

                timesteps=
                    65536,

                checkpoint=
                    CURRENT_CHAMPION_MODEL,

                metrics=
                    champion_metrics,
            )

            with csv_path.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=
                        CSV_FIELDS,
                )

                writer.writerow(
                    champion_row
                )


        # ====================================================
        # FINAL COMPARISON
        # ====================================================

        print()

        print(
            "=" * 125
        )

        print(
            "FINAL GEOMETRY SELECTION"
        )

        print(
            "=" * 125
        )

        if (
            champion_key
            is None
        ):

            winner = (
                "BEST_NEW_CHECKPOINT"
            )

            print(
                "Existing 65K champion was not found."
            )

            print(
                "Best new checkpoint is the available winner."
            )

        elif (
            best_key
            <
            champion_key
        ):

            winner = (
                "BEST_NEW_CHECKPOINT"
            )

            print(
                "NEW CHECKPOINTED RUN BEATS "
                "THE EXISTING 65K CHAMPION."
            )

        elif (
            champion_key
            <
            best_key
        ):

            winner = (
                "EXISTING_65K_CHAMPION"
            )

            print(
                "EXISTING 65K CHAMPION REMAINS BETTER."
            )

        else:

            winner = (
                "TIE"
            )

            print(
                "Best new checkpoint and existing "
                "65K champion tie under the selection rule."
            )

        print()

        print(
            "Winner:",
            winner,
        )

        print()

        print(
            "Best new model copy:"
        )

        print(
            best_model_path
        )

        print()

        print(
            "Final new-run model:"
        )

        print(
            final_model_path
        )

        print()

        print(
            "All checkpoints:"
        )

        print(
            checkpoint_dir
        )

        print()

        print(
            "Geometry CSV:"
        )

        print(
            csv_path
        )


        # ====================================================
        # JSON SUMMARY
        # ====================================================

        summary = {
            "run_name":
                run_name,

            "training": {
                "from_scratch":
                    True,

                "reward":
                    "V2 unchanged",

                "timesteps":
                    int(
                        args.timesteps
                    ),

                "checkpoint_interval":
                    int(
                        args.checkpoint_interval
                    ),

                "seed":
                    int(
                        args.seed
                    ),

                "residual_smoothing":
                    float(
                        args.residual_smoothing
                    ),

                "balanced_start_min":
                    int(
                        BALANCED_START_MIN
                    ),

                "balanced_start_max":
                    int(
                        BALANCED_START_MAX
                    ),

                "learning_rate":
                    float(
                        LEARNING_RATE
                    ),

                "n_steps":
                    int(
                        N_STEPS
                    ),

                "batch_size":
                    int(
                        BATCH_SIZE
                    ),

                "n_epochs":
                    int(
                        N_EPOCHS
                    ),

                "gamma":
                    float(
                        GAMMA
                    ),

                "gae_lambda":
                    float(
                        GAE_LAMBDA
                    ),

                "clip_range":
                    float(
                        CLIP_RANGE
                    ),

                "ent_coef":
                    float(
                        ENT_COEF
                    ),

                "vf_coef":
                    float(
                        VF_COEF
                    ),

                "max_grad_norm":
                    float(
                        MAX_GRAD_NORM
                    ),

                "target_kl":
                    float(
                        TARGET_KL
                    ),

                "policy_net_arch":
                    [
                        int(
                            x
                        )
                        for x in POLICY_NET_ARCH
                    ],
            },

            "selection_rule": [
                "frames_ge_50mm",
                "max_penetration_m",
                "frames_ge_40mm",
                "frames_ge_30mm",
                "frames_ge_20mm",
                "penetrating_frames",
                "mean_frame_max_penetration_m",
                "mean_phase_cost",
                "squared_deficit",
            ],

            "best_new_checkpoint": {
                "timesteps":
                    int(
                        best_checkpoint_timesteps
                    ),

                "path":
                    str(
                        best_checkpoint_path
                    ),

                "best_copy_path":
                    str(
                        best_model_path
                    ),

                "metrics":
                    best_checkpoint_metrics,
            },

            "existing_65k_champion": {
                "path":
                    str(
                        CURRENT_CHAMPION_MODEL
                    ),

                "metrics":
                    champion_metrics,
            },

            "winner":
                winner,

            "final_model":
                str(
                    final_model_path
                ),

            "checkpoint_results":
                checkpoint_results,

            "training_elapsed_seconds":
                float(
                    training_elapsed
                ),
        }

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as json_file:

            json.dump(
                summary,
                json_file,
                indent=2,
            )

        print()

        print(
            "Summary JSON:"
        )

        print(
            json_path
        )

        print()

        print(
            "=" * 125
        )

        print(
            "CHECKPOINTED BALANCED-V2 EXPERIMENT COMPLETE"
        )

        print(
            "=" * 125
        )

        print(
            "Do NOT train V3 yet."
        )

        print(
            "Review the checkpoint geometry table first."
        )

        print(
            "=" * 125
        )


    finally:

        if training_env is not None:
            training_env.close()

        if evaluation_env is not None:
            evaluation_env.close()


if __name__ == "__main__":
    main()
