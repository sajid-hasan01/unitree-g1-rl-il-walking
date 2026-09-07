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
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# BALANCED-START REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2_balanced import (
    G1KinematicUnevenEnv,
    BALANCED_START_MIN,
    BALANCED_START_MAX,
)


# ============================================================
# EXACT REWARD-V2 CONSTANTS
#
# Reward is NOT changed in this experiment.
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
# PPO HYPERPARAMETERS
#
# EXACTLY THE SAME AS THE EXISTING REWARD-V2 SMOKE TRAINER.
#
# These are training hyperparameters, not physical constants.
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
            "12-DOF G1 BC walking policy using Reward V2 "
            "with balanced full-trajectory random starts."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=(
            "Total PPO training timesteps. "
            "Default: 8192 for the controlled smoke run."
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
        default="v2_balanced_smoke_seed425",
        help=(
            "Name used for saved model/config files."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "PyTorch/SB3 device."
        ),
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help=(
            "Maximum episode length. "
            "Late-start episodes may truncate earlier "
            "when TRAIN_END_STEP is reached."
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

    if (
        N_STEPS
        %
        BATCH_SIZE
        !=
        0
    ):
        raise RuntimeError(
            "N_STEPS must be divisible by BATCH_SIZE. "
            f"N_STEPS={N_STEPS}, "
            f"BATCH_SIZE={BATCH_SIZE}"
        )

    if args.timesteps < N_STEPS:

        print(
            "WARNING: requested timesteps are "
            "smaller than one PPO rollout."
        )

        print(
            "SB3 will still collect a complete "
            "rollout before updating."
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
            (
                "kinematic_residual_ppo_reward_v2_"
                "balanced_start"
            ),

        "controlled_experiment": {
            "reference_run":
                "v2_smoke_seed425",

            "changed_variable":
                (
                    "random training start sampling only"
                ),

            "reward_changed":
                False,

            "observation_changed":
                False,

            "action_space_changed":
                False,

            "residual_scales_changed":
                False,

            "residual_smoothing_changed":
                False,

            "ppo_hyperparameters_changed":
                False,

            "physics_changed":
                False,
        },

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

            "source":
                (
                    "g1_kinematic_uneven_env_v2_balanced"
                ),

            "base_reward_environment":
                (
                    "g1_kinematic_uneven_env_v2"
                ),

            "observation_dim":
                62,

            "action_dim":
                12,

            "action_space":
                "normalized Box(-1, +1)",

            "episode_length_max":
                int(
                    args.episode_length
                ),

            "late_episodes_may_be_shorter":
                True,

            "random_start":
                True,

            "random_start_strategy":
                (
                    "uniform full-trajectory transition starts"
                ),

            "balanced_start_min":
                int(
                    BALANCED_START_MIN
                ),

            "balanced_start_max":
                int(
                    BALANCED_START_MAX
                ),

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

            "reward_version":
                "v2_unchanged",

            "reward_hyperparameters": {
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
# PRINT CONFIGURATION
# ============================================================

def print_training_configuration(
    args,
):

    print(
        "=" * 105
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "REWARD-V2 BALANCED-START TRAINER"
    )

    print(
        "=" * 105
    )

    print()

    print(
        "Controlled experiment:"
    )

    print(
        "  baseline training setup:"
    )

    print(
        "    Reward-V2 smoke"
    )

    print(
        "  changed variable:"
    )

    print(
        "    random start sampling ONLY"
    )

    print()

    print(
        "Unchanged:"
    )

    print(
        "  Reward V2"
    )

    print(
        "  62-D observation"
    )

    print(
        "  12-D normalized action"
    )

    print(
        "  residual scales"
    )

    print(
        "  residual smoothing"
    )

    print(
        "  terrain"
    )

    print(
        "  frozen BC"
    )

    print(
        "  root motion"
    )

    print(
        "  PPO architecture/hyperparameters"
    )

    print(
        "  seed"
    )

    print(
        "  requested smoke-training budget"
    )

    print(
        "  kinematic mj_forward-only simulation"
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
        "  train range:",
        (
            f"{TRAIN_START_STEP} "
            f"-> {TRAIN_END_STEP}"
        ),
    )

    print(
        "  balanced random-start range:",
        (
            f"{BALANCED_START_MIN} "
            f"-> {BALANCED_START_MAX}"
        ),
    )

    print(
        "  maximum episode length:",
        args.episode_length,
    )

    print(
        "  late-start episodes:",
        (
            "naturally truncate at TRAIN_END_STEP"
        ),
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
        "Reward V2 -- UNCHANGED:"
    )

    print(
        "  flat distance tolerance:",
        (
            f"{FLAT_DISTANCE_TOLERANCE * 1000.0:.1f} mm"
        ),
    )

    print(
        "  deficit scale:",
        (
            f"{DEFICIT_SCALE * 1000.0:.1f} mm"
        ),
    )

    print(
        "  penetration scale:",
        (
            f"{PENETRATION_SCALE * 1000.0:.1f} mm"
        ),
    )

    print(
        "  deficit weight:",
        DEFICIT_WEIGHT,
    )

    print(
        "  improvement weight:",
        IMPROVEMENT_WEIGHT,
    )

    print(
        "  penetration weight:",
        PENETRATION_WEIGHT,
    )

    print(
        "  excess weight:",
        EXCESS_WEIGHT,
    )

    print(
        "  residual weight:",
        RESIDUAL_WEIGHT,
    )

    print(
        "  smoothness weight:",
        SMOOTHNESS_WEIGHT,
    )

    print(
        "  action weight:",
        ACTION_WEIGHT,
    )

    print(
        "  joint clip weight:",
        JOINT_CLIP_WEIGHT,
    )

    print()

    print(
        "PPO hyperparameters -- UNCHANGED:"
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
        "IMPORTANT:"
    )

    print(
        "  This trains a NEW PPO model from scratch."
    )

    print(
        "  Existing V1 and V2 checkpoints are not loaded."
    )

    print(
        "  Existing checkpoints are not overwritten "
        "when a new run name is used."
    )

    print(
        "=" * 105
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


    # --------------------------------------------------------
    # PROTECT EXISTING OUTPUTS
    # --------------------------------------------------------

    if model_path.exists():

        raise FileExistsError(
            "Refusing to overwrite existing PPO model:\n"
            f"{model_path}\n"
            "Choose a different --run-name."
        )

    if config_path.exists():

        raise FileExistsError(
            "Refusing to overwrite existing config:\n"
            f"{config_path}\n"
            "Choose a different --run-name."
        )


    # --------------------------------------------------------
    # PRINT + SAVE CONFIG
    # --------------------------------------------------------

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
        "Creating balanced-start training environment..."
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

                random_start=
                    True,

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

        print(
            "Balanced start range:",
            (
                f"{BALANCED_START_MIN} "
                f"-> {BALANCED_START_MAX}"
            ),
        )

        print()


        # ====================================================
        # PPO MODEL
        # ====================================================

        print(
            "Creating NEW PPO model from scratch..."
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
            "=" * 105
        )

        print(
            "BALANCED-START PPO TRAINING START"
        )

        print(
            "=" * 105
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

            tb_log_name=
                run_name,

            reset_num_timesteps=
                True,

            progress_bar=
                False,
        )


        # ====================================================
        # FINISHED
        # ====================================================

        print()

        print(
            "=" * 105
        )

        print(
            "BALANCED-START PPO TRAINING FINISHED"
        )

        print(
            "=" * 105
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

        if elapsed_seconds > 0.0:

            print(
                "Average environment/training rate:",
                (
                    f"{model.num_timesteps / elapsed_seconds:.2f} "
                    "timesteps/second"
                ),
            )


        # ====================================================
        # SAVE
        # ====================================================

        print()

        print(
            "Saving balanced-start PPO model..."
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
                    "PPO save returned but the model "
                    "file could not be found."
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
            "Updated run configuration:"
        )

        print(
            config_path
        )

        print()

        print(
            "=" * 105
        )

        print(
            "BALANCED-START TRAINING RUN COMPLETE"
        )

        print(
            "=" * 105
        )

        print(
            "This is still a smoke-training policy."
        )

        print(
            "Do NOT increase the training budget "
            "until deterministic geometry evaluation."
        )

        print(
            "Next comparison:"
        )

        print(
            "  BC-only"
        )

        print(
            "  original Reward-V2 PPO"
        )

        print(
            "  balanced-start Reward-V2 PPO"
        )

        print(
            "=" * 105
        )


    finally:

        if env is not None:

            env.close()


if __name__ == "__main__":
    main()