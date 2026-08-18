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

from stable_baselines3.common.monitor import (
    Monitor,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from envs.g1_closed_loop_tracking_env import (
    G1ClosedLoopTrackingEnv,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_tracking_v1_checkpoints"
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_closed_loop_tracking_v1_50k.zip"
)


def make_env():

    return Monitor(
        G1ClosedLoopTrackingEnv(

            random_reference_start=True,

            fixed_start_frame=None,

            control_hz=50.0,

            target_velocity=-0.18,

            reset_position_noise=0.015,

            reset_velocity_noise=0.05,

            target_smoothing=0.20,

            max_episode_steps=400,
        )
    )


# ================================================================
# PRE-FLIGHT
# ================================================================

probe = G1ClosedLoopTrackingEnv(
    random_reference_start=False,
    fixed_start_frame=42,
    reset_position_noise=0.0,
    reset_velocity_noise=0.0,
)


obs, info = probe.reset(
    seed=1234
)


print("=" * 100)
print("CLOSED-LOOP TRACKER PRE-FLIGHT")
print("=" * 100)

print(
    "obs shape:",
    obs.shape,
)

print(
    "action shape:",
    probe.action_space.shape,
)

print(
    "control Hz:",
    probe.control_hz,
)

print(
    "MuJoCo dt:",
    probe.sim_dt,
)

print(
    "physics steps/control:",
    probe.frame_skip,
)

print(
    "reference frames:",
    probe.num_frames,
)

print(
    "reference FPS:",
    probe.reference_fps,
)


print()
print("Per-joint action offsets:")

for name, scale in zip(
    probe.joint_names,
    probe.action_scale,
):

    print(
        f"  {name:27s} "
        f"{scale:.5f} rad"
    )


probe.close()


# ================================================================
# PPO
# ================================================================

vec_env = DummyVecEnv(
    [
        make_env
    ]
)


model = PPO(
    "MlpPolicy",

    vec_env,

    learning_rate=3.0e-4,

    n_steps=2048,

    batch_size=128,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.20,

    ent_coef=0.001,

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
            -1.5,
    },

    verbose=1,

    seed=42,

    device="auto",
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint = (
    CheckpointCallback(
        save_freq=10000,

        save_path=str(
            CHECKPOINT_DIR
        ),

        name_prefix=
            "g1_tracking_v1",
    )
)


print()
print("=" * 100)
print("TRAINING CLOSED-LOOP MOTION TRACKER")
print("=" * 100)

print(
    "Training steps: 50,000"
)

print(
    "Random reference-state initialization: ON"
)

print(
    "BC network used for locomotion: NO"
)

print(
    "Reference joint position tracking: YES"
)

print(
    "Reference joint velocity tracking: YES"
)

print(
    "Reference contact tracking: YES"
)


model.learn(
    total_timesteps=50000,

    callback=checkpoint,

    progress_bar=False,
)


model.save(
    str(
        FINAL_MODEL
    )
)


vec_env.close()


print()
print(
    "Saved:",
    FINAL_MODEL
)
