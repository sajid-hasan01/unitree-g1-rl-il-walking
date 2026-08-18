import sys
from pathlib import Path

import torch.nn as nn


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT),
    )


from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)

from stable_baselines3.common.env_checker import (
    check_env,
)

from stable_baselines3.common.monitor import (
    Monitor,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)


from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


TOTAL_STEPS = 20000


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_tracking_v3_pilot_checkpoints"
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_tracking_v3_pilot_final"
)


def make_env():

    return Monitor(
        G1ClosedLoopTrackingEnvV3(

            random_reference_start=True,

            fixed_start_frame=None,

            reset_position_noise=0.0,

            reset_velocity_noise=0.0,

            control_hz=50.0,

            target_smoothing=0.20,

            max_episode_steps=400,
        )
    )


# =====================================================================
# PREFLIGHT
# =====================================================================

probe = G1ClosedLoopTrackingEnvV3(

    fixed_start_frame=50,

    random_reference_start=False,

    reset_position_noise=0.0,

    reset_velocity_noise=0.0,
)


obs, info = probe.reset(
    seed=1234
)


print("=" * 110)
print("TRACKING V3 PPO PILOT PREFLIGHT")
print("=" * 110)

print(
    "obs:",
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
    "reference FPS:",
    probe.reference_fps,
)

print(
    "control Hz:",
    probe.control_hz,
)

print(
    "frame skip:",
    probe.frame_skip,
)

print(
    "contact reference:",
    probe.has_reference_contact,
)

print(
    "reset frame:",
    info[
        "reset_reference_frame"
    ],
)


if obs.shape != (
    166,
):

    raise RuntimeError(
        f"Bad V3 obs shape: "
        f"{obs.shape}"
    )


print()
print(
    "Running SB3 environment checker..."
)

check_env(
    probe,
    warn=True,
)


probe.close()


print(
    "Environment checker: PASS"
)


# =====================================================================
# VECTOR ENV
# =====================================================================

env = DummyVecEnv(
    [
        make_env
    ]
)


# =====================================================================
# PPO
#
# Lower initial policy std because action=0 already means:
#
#     use the reference trajectory directly.
#
# PPO should learn controlled corrections rather than destroying
# the reference immediately with large random residual actions.
# =====================================================================

model = PPO(
    "MlpPolicy",

    env,

    learning_rate=2.5e-4,

    n_steps=1024,

    batch_size=128,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.20,

    ent_coef=0.0005,

    vf_coef=0.50,

    max_grad_norm=0.50,

    policy_kwargs={
        "net_arch":
            [
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

    save_freq=5000,

    save_path=str(
        CHECKPOINT_DIR
    ),

    name_prefix=
        "g1_tracking_v3_pilot",
)


print()
print("=" * 110)
print("STARTING PPO PILOT")
print("=" * 110)

print(
    "total steps:",
    TOTAL_STEPS,
)

print(
    "checkpoints:",
    "every 5,000 steps",
)

print(
    "policy residual std init:",
    "exp(-2.0) ~= 0.135",
)

print(
    "training device:",
    "CPU",
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
print("PILOT COMPLETE")
print("=" * 110)

print(
    "final model:",
    str(
        FINAL_MODEL
    )
    + ".zip"
)

print("=" * 110)
