from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
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
# REWARD-V3 BALANCED ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v3_balanced import (
    G1KinematicUnevenEnv,
    SEVERE_PENETRATION_THRESHOLD,
    SEVERE_PENETRATION_WEIGHT,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
)


# ============================================================
# ORIGINAL REWARD-V2 CONSTANTS
#
# V3 preserves all of these.
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    RESIDUAL_SCALES,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    DEFAULT_EPISODE_LENGTH,
    FLAT_DISTANCE_TOLERANCE,
    DEFICIT_SCALE,
    PENETRATION_SCALE,
    DEFICIT_WEIGHT,
    IMPROVEMENT_WEIGHT,
    PENETRATION_WEIGHT,
    EXCESS_WEIGHT,
    RESIDUAL_WEIGHT,
    SMOOTHNESS_WEIGHT,
    ACTION_WEIGHT,
    JOINT_CLIP_WEIGHT,
)


# ============================================================
# EXACT PPO SETTINGS FROM THE ALREADY-TESTED
# REWARD-V2 BALANCED TRAINER
#
# This avoids manually duplicating the hyperparameters and
# accidentally changing the controlled experiment.
# ============================================================

from scripts.train_g1_kinematic_ppo_v2_balanced import (
    DEFAULT_TIMESTEPS,
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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh Unitree G1 residual PPO policy "
            "using Reward V3 with balanced full-trajectory "
            "random-start sampling."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="v3_balanced_verify_seed425",
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
# ARGUMENT VALIDATION
# ============================================================

def validate_args(
    args,
):

    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps must be positive."
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds(
    seed,
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# CONFIG RECORD
# ============================================================

def build_config(
    args,
):

    return {
        "training_stage":
            (
                "kinematic_residual_ppo_"
                "reward_v3_balanced_verification"
            ),

        "controlled_experiment": {
            "reference_training_run":
                "v2_balanced_smoke_seed425",

            "changed_variable":
                (
                    "Reward V3 severe absolute "
                    "penetration safeguard only"
                ),

            "training_from_scratch":
                True,

            "balanced_sampling_changed":
                False,

            "observations_changed":
                False,

            "action_space_changed":
                False,

            "residual_scales_changed":
                False,

            "residual_smoothing_changed":
                False,

            "ppo_hyperparameters_changed":
                False,

            "terrain_changed":
                False,

            "bc_policy_changed":
                False,

            "root_motion_changed":
                False,

            "physics_changed":
                False,
        },

        "timesteps_requested":
            int(
                args.timesteps
            ),

        "seed":
            int(
                args.seed
            ),

        "device":
            str(
                args.device
            ),

        "environment": {
            "source":
                "g1_kinematic_uneven_env_v3_balanced",

            "base_reward":
                "Reward V2",

            "reward_version":
                "V3",

            "observation_dim":
                62,

            "action_dim":
                12,

            "episode_length_max":
                int(
                    args.episode_length
                ),

            "train_start_step":
                int(
                    TRAIN_START_STEP
                ),

            "train_end_step":
                int(
                    TRAIN_END_STEP
                ),

            "balanced_start_min":
                int(
                    BALANCED_START_MIN
                ),

            "balanced_start_max":
                int(
                    BALANCED_START_MAX
                ),

            "residual_smoothing":
                float(
                    args.residual_smoothing
                ),

            "residual_scales_rad":
                [
                    float(
                        x
                    )
                    for x in RESIDUAL_SCALES
                ],

            "kinematic_only":
                True,

            "mujoco_update":
                "mj_forward only",
        },

        "reward_v2": {
            "flat_distance_tolerance_m":
                float(
                    FLAT_DISTANCE_TOLERANCE
                ),

            "deficit_scale_m":
                float(
                    DEFICIT_SCALE
                ),

            "penetration_scale_m":
                float(
                    PENETRATION_SCALE
                ),

            "deficit_weight":
                float(
                    DEFICIT_WEIGHT
                ),

            "improvement_weight":
                float(
                    IMPROVEMENT_WEIGHT
                ),

            "penetration_weight":
                float(
                    PENETRATION_WEIGHT
                ),

            "excess_weight":
                float(
                    EXCESS_WEIGHT
                ),

            "residual_weight":
                float(
                    RESIDUAL_WEIGHT
                ),

            "smoothness_weight":
                float(
                    SMOOTHNESS_WEIGHT
                ),

            "action_weight":
                float(
                    ACTION_WEIGHT
                ),

            "joint_clip_weight":
                float(
                    JOINT_CLIP_WEIGHT
                ),
        },

        "reward_v3_addition": {
            "severe_penetration_threshold_m":
                float(
                    SEVERE_PENETRATION_THRESHOLD
                ),

            "severe_penetration_threshold_mm":
                float(
                    SEVERE_PENETRATION_THRESHOLD
                    *
                    1000.0
                ),

            "severe_penetration_weight":
                float(
                    SEVERE_PENETRATION_WEIGHT
                ),

            "normalization_scale_m":
                float(
                    PENETRATION_SCALE
                ),

            "formula":
                (
                    "weight * "
                    "(max(max_actual_penetration "
                    "- threshold, 0) / scale)^2"
                ),
        },

        "ppo_hyperparameters": {
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
                list(
                    POLICY_NET_ARCH
                ),
        },

        "software": {
            "python":
                platform.python_version(),

            "numpy":
                np.__version__,

            "torch":
                torch.__version__,

            "mujoco":
                mujoco.__version__,

            "gymnasium":
                gymnasium.__version__,

            "stable_baselines3":
                stable_baselines3.__version__,
        },
    }


# ============================================================
# PRINT CONFIG
# ============================================================

def print_config(
    args,
):

    print(
        "=" * 110
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "REWARD-V3 BALANCED VERIFICATION TRAINER"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "  Fresh 8192-step verification training."
    )

    print(
        "  Test whether the new severe absolute "
        "penetration safeguard improves the policy "
        "before longer PPO training."
    )

    print()

    print(
        "Changed from balanced V2:"
    )

    print(
        "  Reward only:"
    )

    print(
        "    severe penetration threshold:",
        (
            f"{SEVERE_PENETRATION_THRESHOLD * 1000:.1f} mm"
        ),
    )

    print(
        "    severe penetration weight:",
        SEVERE_PENETRATION_WEIGHT,
    )

    print()

    print(
        "Unchanged:"
    )

    print(
        "  balanced random-start sampling"
    )

    print(
        "  62-D observation"
    )

    print(
        "  12-D normalized PPO action"
    )

    print(
        "  residual scales"
    )

    print(
        "  residual smoothing"
    )

    print(
        "  frozen BC"
    )

    print(
        "  uneven terrain"
    )

    print(
        "  root trajectory"
    )

    print(
        "  PPO hyperparameters"
    )

    print(
        "  seed"
    )

    print(
        "  kinematic mj_forward-only execution"
    )

    print()

    print(
        "Training:"
    )

    print(
        "  run name:",
        args.run_name,
    )

    print(
        "  from scratch:",
        True,
    )

    print(
        "  timesteps:",
        args.timesteps,
    )

    print(
        "  seed:",
        args.seed,
    )

    print(
        "  device:",
        args.device,
    )

    print()

    print(
        "Balanced start range:",
        (
            f"{BALANCED_START_MIN} "
            f"-> {BALANCED_START_MAX}"
        ),
    )

    print(
        "Maximum episode length:",
        args.episode_length,
    )

    print(
        "Residual smoothing:",
        args.residual_smoothing,
    )

    print()

    print(
        "PPO hyperparameters:"
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
        "  target_kl:",
        TARGET_KL,
    )

    print(
        "  network:",
        POLICY_NET_ARCH,
    )

    print()

    print(
        "Existing V1/V2/V2-balanced checkpoints "
        "will NOT be loaded or overwritten."
    )

    print(
        "=" * 110
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    set_seeds(
        args.seed
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
    # OUTPUT PATHS
    # ========================================================

    model_path = (
        MODEL_DIR
        /
        f"g1_kinematic_ppo_{run_name}.zip"
    )

    config_path = (
        LOG_DIR
        /
        f"g1_kinematic_ppo_{run_name}_config.json"
    )


    # ========================================================
    # PROTECT EXISTING FILES
    # ========================================================

    if model_path.exists():

        raise FileExistsError(
            "Refusing to overwrite existing model:\n"
            f"{model_path}\n"
            "Use a different --run-name."
        )

    if config_path.exists():

        raise FileExistsError(
            "Refusing to overwrite existing config:\n"
            f"{config_path}\n"
            "Use a different --run-name."
        )


    # ========================================================
    # CONFIG
    # ========================================================

    print_config(
        args
    )

    config = build_config(
        args
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    print()

    print(
        "Saved configuration:"
    )

    print(
        config_path
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = None

    model = None

    start_time = time.perf_counter()

    try:

        print()

        print(
            "Creating Reward-V3 balanced environment..."
        )

        env = G1KinematicUnevenEnv(
            episode_length=
                args.episode_length,

            random_start=
                True,

            residual_smoothing=
                args.residual_smoothing,
        )

        print(
            "Environment created."
        )

        print(
            "Observation space:",
            env.observation_space,
        )

        print(
            "Action space:",
            env.action_space,
        )


        # ====================================================
        # PPO
        # ====================================================

        print()

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
                env,

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

        print()

        print(
            "=" * 110
        )

        print(
            "REWARD-V3 VERIFICATION TRAINING START"
        )

        print(
            "=" * 110
        )


        # ====================================================
        # TRAIN
        # ====================================================

        model.learn(
            total_timesteps=
                int(
                    args.timesteps
                ),

            log_interval=
                1,

            reset_num_timesteps=
                True,

            progress_bar=
                False,
        )


        # ====================================================
        # FINISHED
        # ====================================================

        elapsed_seconds = (
            time.perf_counter()
            -
            start_time
        )

        print()

        print(
            "=" * 110
        )

        print(
            "REWARD-V3 VERIFICATION TRAINING FINISHED"
        )

        print(
            "=" * 110
        )

        print(
            "Actual SB3 timesteps:",
            int(
                model.num_timesteps
            ),
        )

        print(
            "Elapsed seconds:",
            f"{elapsed_seconds:.2f}",
        )

        if elapsed_seconds > 0.0:

            print(
                "Timesteps/sec:",
                (
                    f"{model.num_timesteps / elapsed_seconds:.2f}"
                ),
            )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        print()

        print(
            "Saving Reward-V3 verification model..."
        )

        model.save(
            model_path
        )

        if not model_path.exists():

            alternative_path = Path(
                str(
                    model_path
                )
                +
                ".zip"
            )

            if alternative_path.exists():

                model_path = (
                    alternative_path
                )

            else:

                raise RuntimeError(
                    "PPO save returned, but the "
                    "model file was not found."
                )

        print(
            "Model saved:"
        )

        print(
            model_path
        )


        # ====================================================
        # UPDATE CONFIG
        # ====================================================

        config[
            "training_result"
        ] = {
            "actual_timesteps":
                int(
                    model.num_timesteps
                ),

            "elapsed_seconds":
                float(
                    elapsed_seconds
                ),

            "timesteps_per_second":
                float(
                    (
                        model.num_timesteps
                        /
                        elapsed_seconds
                    )
                    if
                    elapsed_seconds
                    >
                    0.0
                    else
                    0.0
                ),

            "model_path":
                str(
                    model_path
                ),
        }

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=2,
            )

        print()

        print(
            "Updated configuration:"
        )

        print(
            config_path
        )

        print()

        print(
            "=" * 110
        )

        print(
            "REWARD-V3 VERIFICATION RUN COMPLETE"
        )

        print(
            "=" * 110
        )

        print(
            "This remains a short verification policy."
        )

        print()

        print(
            "Do NOT start the long PPO run yet."
        )

        print()

        print(
            "Next step:"
        )

        print(
            "  deterministic comparison of:"
        )

        print(
            "    BC"
        )

        print(
            "    original Reward V2"
        )

        print(
            "    balanced Reward V2"
        )

        print(
            "    balanced Reward V3"
        )

        print()

        print(
            "Primary decision:"
        )

        print(
            "  Did V3 reduce severe penetration "
            "without creating another new failure region?"
        )

        print(
            "=" * 110
        )

    finally:

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()