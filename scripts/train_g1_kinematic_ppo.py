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
# PROJECT IMPORT PATH
#
# This makes the script work both as:
#
#   python -m scripts.train_g1_kinematic_ppo
#
# and:
#
#   python .\scripts\train_g1_kinematic_ppo.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from envs.g1_kinematic_uneven_env import (
    G1KinematicUnevenEnv,
    RESIDUAL_SCALES,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    DEFAULT_EPISODE_LENGTH,
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
# INITIAL PPO HYPERPARAMETERS
#
# IMPORTANT:
#
# These are TRAINING HYPERPARAMETERS.
#
# They are not measured physical facts about the G1.
#
# We will judge them using training/evaluation results.
# ============================================================

DEFAULT_TIMESTEPS = 8192

LEARNING_RATE = 3e-4

N_STEPS = 512

BATCH_SIZE = 128

N_EPOCHS = 10

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.20

ENT_COEF = 0.0

VF_COEF = 0.50

MAX_GRAD_NORM = 0.50

TARGET_KL = 0.03

POLICY_NET_ARCH = [
    128,
    128,
]

DEFAULT_SEED = 425

DEFAULT_RESIDUAL_SMOOTHING = 0.35


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train residual PPO on top of the frozen "
            "12-DOF G1 BC walking policy using the "
            "kinematic uneven-terrain environment."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=(
            "Total PPO training timesteps. "
            "Default: 8192 for smoke training."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Training/random seed."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="smoke_seed425",
        help=(
            "Name used for saved model/config files."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "PyTorch/SB3 device. "
            "Current verified environment is CPU."
        ),
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help=(
            "Maximum environment episode length."
        ),
    )

    parser.add_argument(
        "--residual-smoothing",
        type=float,
        default=DEFAULT_RESIDUAL_SMOOTHING,
        help=(
            "Residual transition smoothing alpha."
        ),
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
            "--residual-smoothing must be "
            "in the interval (0, 1]."
        )

    # One environment is used.
    #
    # Rollout buffer size therefore equals N_STEPS.
    if (
        N_STEPS
        %
        BATCH_SIZE
        !=
        0
    ):
        raise RuntimeError(
            "For this one-environment trainer, "
            "N_STEPS must be divisible by "
            "BATCH_SIZE. "
            f"N_STEPS={N_STEPS}, "
            f"BATCH_SIZE={BATCH_SIZE}"
        )

    if (
        args.timesteps
        <
        N_STEPS
    ):
        print(
            "WARNING: requested timesteps are "
            "smaller than one PPO rollout."
        )

        print(
            "SB3 will still need to collect "
            "a complete rollout before updating."
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
# CONFIGURATION RECORD
# ============================================================

def build_config(
    args,
):

    return {
        "project":
            (
                "RL and IL Based Walking, Balance, "
                "and Adaptive Terrain Control of "
                "Unitree G1 Humanoid Robot in MuJoCo"
            ),

        "training_stage":
            "kinematic_residual_ppo",

        "policy":
            "MlpPolicy",

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
            "class":
                "G1KinematicUnevenEnv",

            "observation_dim":
                62,

            "action_dim":
                12,

            "action_space":
                "normalized Box(-1, +1)",

            "episode_length":
                int(
                    args.episode_length
                ),

            "random_start":
                True,

            "train_start_step":
                int(
                    TRAIN_START_STEP
                ),

            "train_end_step":
                int(
                    TRAIN_END_STEP
                ),

            "residual_smoothing":
                float(
                    args.residual_smoothing
                ),

            "residual_scales_rad":
                [
                    float(
                        value
                    )

                    for value
                    in RESIDUAL_SCALES
                ],

            "physics_step":
                False,

            "mujoco_update":
                "mj_forward only",
        },

        "ppo_hyperparameters": {
            "learning_rate":
                LEARNING_RATE,

            "n_steps":
                N_STEPS,

            "batch_size":
                BATCH_SIZE,

            "n_epochs":
                N_EPOCHS,

            "gamma":
                GAMMA,

            "gae_lambda":
                GAE_LAMBDA,

            "clip_range":
                CLIP_RANGE,

            "ent_coef":
                ENT_COEF,

            "vf_coef":
                VF_COEF,

            "max_grad_norm":
                MAX_GRAD_NORM,

            "target_kl":
                TARGET_KL,

            "policy_net_arch":
                POLICY_NET_ARCH,
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

def print_training_configuration(
    args,
):

    print(
        "=" * 100
    )

    print(
        "UNITREE G1 KINEMATIC RESIDUAL PPO TRAINER"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Training type:"
    )

    print(
        "  Frozen BC + PPO residual corrections"
    )

    print(
        "  Kinematic only"
    )

    print(
        "  mujoco.mj_step() is NOT used "
        "by the environment"
    )

    print()

    print(
        "Run:"
    )

    print(
        "  name:",
        args.run_name,
    )

    print(
        "  seed:",
        args.seed,
    )

    print(
        "  device:",
        args.device,
    )

    print(
        "  requested timesteps:",
        args.timesteps,
    )

    print()

    print(
        "Environment:"
    )

    print(
        "  observation:",
        "(62,)",
    )

    print(
        "  action:",
        "(12,) normalized [-1,+1]",
    )

    print(
        "  episode length:",
        args.episode_length,
    )

    print(
        "  train range:",
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
    )

    print(
        "  random start:",
        True,
    )

    print(
        "  residual smoothing:",
        args.residual_smoothing,
    )

    print()

    print(
        "Residual scales [rad]:"
    )

    print(
        " ",
        np.asarray(
            RESIDUAL_SCALES
        ),
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
        "  policy network:",
        POLICY_NET_ARCH,
    )

    print()

    print(
        "This is the initial PPO configuration."
    )

    print(
        "Learning quality will be evaluated "
        "before any long training run."
    )

    print(
        "=" * 100
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

    print_training_configuration(
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
        "Saved run configuration:"
    )

    print(
        config_path
    )

    print()

    print(
        "Creating training environment..."
    )

    env = None

    training_start_time = (
        time.perf_counter()
    )

    try:

        env = (
            G1KinematicUnevenEnv(
                episode_length=
                    args.episode_length,

                random_start=True,

                residual_smoothing=
                    args.residual_smoothing,
            )
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

        print()

        print(
            "Creating PPO model..."
        )

        policy_kwargs = {
            "net_arch":
                POLICY_NET_ARCH,
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
            "PPO model created."
        )

        print()

        print(
            "=" * 100
        )

        print(
            "PPO TRAINING START"
        )

        print(
            "=" * 100
        )

        model.learn(
            total_timesteps=
                int(
                    args.timesteps
                ),

            log_interval=
                1,

            tb_log_name=
                run_name,

            reset_num_timesteps=
                True,

            progress_bar=
                False,
        )

        print()

        print(
            "=" * 100
        )

        print(
            "PPO TRAINING FINISHED"
        )

        print(
            "=" * 100
        )

        elapsed_seconds = (
            time.perf_counter()
            -
            training_start_time
        )

        print(
            "Actual SB3 timesteps:",
            int(
                model.num_timesteps
            ),
        )

        print(
            "Elapsed time:",
            f"{elapsed_seconds:.2f} seconds",
        )

        if (
            elapsed_seconds
            >
            0.0
        ):
            print(
                "Average environment/training rate:",
                (
                    f"{model.num_timesteps / elapsed_seconds:.2f} "
                    "timesteps/second"
                ),
            )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        print()

        print(
            "Saving PPO model..."
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
                    "PPO save returned but the "
                    "model file could not be found."
                )

        print(
            "Model saved:"
        )

        print(
            model_path
        )

        # ----------------------------------------------------
        # UPDATE CONFIG WITH TRAINING RESULT
        # ----------------------------------------------------

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
            "Updated run configuration:"
        )

        print(
            config_path
        )

        print()

        print(
            "=" * 100
        )

        print(
            "TRAINING RUN COMPLETE"
        )

        print(
            "=" * 100
        )

        print(
            "The PPO model has been saved."
        )

        print(
            "Do NOT treat this smoke model as "
            "the final policy yet."
        )

        print(
            "Next step: quantitatively compare "
            "this policy against the BC-only "
            "uneven-terrain baseline."
        )

        print(
            "=" * 100
        )

    finally:

        if env is not None:

            env.close()


if __name__ == "__main__":
    main()