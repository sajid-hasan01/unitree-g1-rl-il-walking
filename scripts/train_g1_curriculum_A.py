import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


TOTAL_STEPS = 12288
START_MIN = 0
START_MAX = 80

CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_curriculum_A_checkpoints"
)

FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_curriculum_A_final"
)


class CurriculumEnv(
    G1ClosedLoopTrackingEnvV3
):

    def __init__(self):

        super().__init__(
            fixed_start_frame=0,
            random_reference_start=False,

            reset_position_noise=0.0,
            reset_velocity_noise=0.0,

            control_hz=50.0,
            target_smoothing=0.20,

            max_episode_steps=400,
        )

        self.curriculum_rng = (
            np.random.default_rng(42)
        )


    def reset(
        self,
        seed=None,
        options=None,
    ):

        if seed is not None:
            self.curriculum_rng = (
                np.random.default_rng(seed)
            )

        self.fixed_start_frame = int(
            self.curriculum_rng.integers(
                START_MIN,
                START_MAX + 1,
            )
        )

        return super().reset(
            seed=seed,
            options=options,
        )


def make_env():

    return Monitor(
        CurriculumEnv()
    )


# ================================================================
# PREFLIGHT
# ================================================================

probe = CurriculumEnv()

obs, info = probe.reset(
    seed=1234
)

print("=" * 110)
print("CURRICULUM PILOT A")
print("=" * 110)

print(
    "observation:",
    obs.shape,
)

print(
    "action:",
    probe.action_space.shape,
)

print(
    "reference frames:",
    probe.num_frames,
)

print(
    "training start range:",
    f"{START_MIN}..{START_MAX}",
)

print(
    "first sampled start:",
    info[
        "reset_reference_frame"
    ],
)

probe.close()


# ================================================================
# ENV
# ================================================================

env = DummyVecEnv(
    [make_env]
)


# ================================================================
# CONSERVATIVE PPO
# ================================================================

model = PPO(
    "MlpPolicy",

    env,

    learning_rate=7.5e-5,

    n_steps=2048,

    batch_size=256,

    n_epochs=3,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.10,

    ent_coef=0.0,

    vf_coef=0.50,

    max_grad_norm=0.50,

    target_kl=0.015,

    policy_kwargs={
        "net_arch": [
            256,
            256,
            128,
        ],

        "activation_fn":
            nn.ELU,

        "log_std_init":
            -2.0,
    },

    verbose=1,

    seed=42,

    device="cpu",
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint = CheckpointCallback(
    save_freq=4096,

    save_path=str(
        CHECKPOINT_DIR
    ),

    name_prefix=
        "g1_curriculum_A",
)


print()
print("=" * 110)
print("TRAINING")
print("=" * 110)

print(
    "total steps:",
    TOTAL_STEPS,
)

print(
    "start range:",
    START_MIN,
    "to",
    START_MAX,
)

print(
    "learning rate:",
    7.5e-5,
)

print(
    "clip range:",
    0.10,
)

print(
    "epochs:",
    3,
)

print(
    "target KL:",
    0.015,
)


model.learn(
    total_timesteps=
        TOTAL_STEPS,

    callback=
        checkpoint,

    progress_bar=False,
)


model.save(
    str(
        FINAL_MODEL
    )
)


env.close()


print()
print("=" * 110)
print("CURRICULUM PILOT A COMPLETE")
print("=" * 110)

print(
    "model:",
    str(FINAL_MODEL) + ".zip",
)
