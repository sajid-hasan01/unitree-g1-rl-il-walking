from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.g1_bc_line_residual_env import (
    G1BCLineResidualEnv,
)


MODEL_DIR = ROOT / "models"
CHECKPOINT_DIR = (
    MODEL_DIR
    / "ppo_bc_line_residual_v1_checkpoints"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def make_raw_env():

    return G1BCLineResidualEnv(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        # Keep residual authority conservative.
        residual_scale=0.05,

        # Current BC/OpenHE natural walking direction.
        target_velocity=-0.18,

        max_episode_steps=600,
    )


print("=" * 100)
print("BC -> RESIDUAL PPO V1")
print("STRAIGHT-LINE STABILIZATION")
print("=" * 100)

print()
print("Configuration:")
print("  desired direction       = -X")
print("  desired line            = Y = 0")
print("  desired yaw             = 0 deg")
print("  start frame             = 25")
print("  stand frames            = 45")
print("  transition frames       = 90")
print("  full BC frame           = 135")
print("  residual scale          = 0.05 rad")
print("  target velocity         = -0.18 m/s")
print("  terrain                 = flat")
print()


# ================================================================
# ENVIRONMENT VALIDATION
# ================================================================

test_env = make_raw_env()

print(
    "Observation space:",
    test_env.observation_space,
)

print(
    "Action space:",
    test_env.action_space,
)

print()
print("Running SB3 environment checker...")

check_env(
    test_env,
    warn=True,
)

test_env.close()

print("Environment checker: PASS")
print()


# ================================================================
# TRAINING ENV
# ================================================================

def make_monitored_env():

    env = make_raw_env()

    return Monitor(env)


vec_env = DummyVecEnv(
    [make_monitored_env]
)


# ================================================================
# PPO
#
# Conservative settings:
#
# - small LR because BC already provides the gait
# - clip 0.15 prevents large policy updates
# - moderate entropy only
# - 128x128 policy sufficient for 62-D state
# ================================================================

model = PPO(
    policy="MlpPolicy",
    env=vec_env,

    learning_rate=1.0e-4,

    n_steps=1024,
    batch_size=64,

    n_epochs=10,

    gamma=0.995,
    gae_lambda=0.95,

    clip_range=0.15,

    ent_coef=0.002,
    vf_coef=0.5,

    max_grad_norm=0.5,

    policy_kwargs={
        "net_arch": [128, 128],
    },

    verbose=1,

    seed=42,

    device="cpu",
)


checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="g1_bc_line_residual",
)


print("=" * 100)
print("TRAINING 50,000 STEPS")
print("=" * 100)

model.learn(
    total_timesteps=50000,
    callback=checkpoint_callback,
    progress_bar=False,
)


output = (
    MODEL_DIR
    / "g1_ppo_bc_line_residual_v1_50k"
)

model.save(
    str(output)
)

vec_env.close()


print()
print("=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)

print(
    "Final model:",
    str(output) + ".zip",
)

print(
    "Checkpoints:",
    CHECKPOINT_DIR,
)
