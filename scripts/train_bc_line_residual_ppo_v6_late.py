import sys
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from envs.g1_bc_line_residual_env_v6_late import (
    G1BCLineResidualEnvV6Late,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v6_late_checkpoints"
)

FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v6_late_10k.zip"
)


CORRECTION_START = 95
DELTA_SCALE = 0.10


def make_env(
    warm,
):

    return G1BCLineResidualEnvV6Late(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        delta_action_scale=DELTA_SCALE,

        correction_start_step=CORRECTION_START,
        correction_ramp_frames=5,

        warm_start_training=warm,
    )


# =====================================================================
# FULL ZERO-DELTA BASELINE
# =====================================================================

def full_zero_delta_eval():

    env = make_env(
        False
    )

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0

        max_y = 0.0
        max_yaw = 0.0

        lsw = 0
        rsw = 0

        prev_l = None
        prev_r = None

        final = {}

        while True:

            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            steps += 1

            max_y = max(
                max_y,
                abs(
                    float(
                        info["y"]
                    )
                ),
            )

            max_yaw = max(
                max_yaw,
                abs(
                    float(
                        info["yaw_deg"]
                    )
                ),
            )

            left = bool(
                info["left_contact"]
            )

            right = bool(
                info["right_contact"]
            )

            if (
                prev_l is not None
                and left != prev_l
            ):
                lsw += 1

            if (
                prev_r is not None
                and right != prev_r
            ):
                rsw += 1

            prev_l = left
            prev_r = right

            final = info

            if terminated or truncated:
                break


        return {
            "steps":
                steps,

            "x":
                float(
                    final["x"]
                ),

            "max_y":
                max_y,

            "max_yaw":
                max_yaw,

            "lsw":
                lsw,

            "rsw":
                rsw,
        }

    finally:

        env.close()


print("=" * 100)
print("V6-LATE FULL-TRAJECTORY PRE-FLIGHT")
print("=" * 100)

baseline = full_zero_delta_eval()

print(
    f"steps={baseline['steps']} "
    f"X={baseline['x']:+.3f} "
    f"maxY={baseline['max_y']:.3f} "
    f"maxYaw={baseline['max_yaw']:.1f} "
    f"L/R={baseline['lsw']}/{baseline['rsw']}"
)


if not (
    115
    <= baseline["steps"]
    <= 123
):

    raise SystemExit(
        "ABORT: V6 zero-delta baseline does not "
        "reproduce frozen V2."
    )


if baseline["max_y"] > 0.15:

    raise SystemExit(
        "ABORT: V6 baseline maxY regression."
    )


if baseline["max_yaw"] > 40.0:

    raise SystemExit(
        "ABORT: V6 baseline yaw regression."
    )


print("FULL ZERO-DELTA PRE-FLIGHT: PASS")


# =====================================================================
# INSPECT WARM-START STATE
# =====================================================================

warm_probe = make_env(
    True
)

try:

    obs, info = warm_probe.reset(
        seed=1234
    )

    print()
    print("=" * 100)
    print("V6 TRAINING START STATE")
    print("=" * 100)

    print(
        f"physical_step={warm_probe.episode_step}"
    )

    print(
        f"X={float(info['x']):+.3f} "
        f"Y={float(info['y']):+.3f} "
        f"yaw={float(info['yaw_deg']):+.1f} "
        f"up={float(info['up_z']):.3f}"
    )

    print(
        "Next physical step will be the first "
        "step on which V6 correction can act."
    )

finally:

    warm_probe.close()


# =====================================================================
# FOCUSED TRAINING ENV
# =====================================================================

def make_train_env():

    return Monitor(
        make_env(
            True
        )
    )


vec_env = DummyVecEnv(
    [
        make_train_env
    ]
)


# =====================================================================
# NEW LATE-RESCUE PPO
#
# Completely separate from V5.
# Frozen V2 remains inside environment.
#
# Initial deterministic correction = exactly zero.
# =====================================================================

model = PPO(
    "MlpPolicy",
    vec_env,

    learning_rate=5.0e-5,

    # Short late-phase episodes:
    # 512 is more appropriate than 1024.
    n_steps=512,

    batch_size=64,
    n_epochs=5,

    gamma=0.995,
    gae_lambda=0.95,

    clip_range=0.10,

    ent_coef=0.0001,
    vf_coef=0.50,

    max_grad_norm=0.50,

    target_kl=0.01,

    policy_kwargs={
        "net_arch": [
            64,
            64,
        ],

        # Slightly less exploration than V5.
        "log_std_init":
            -2.3,
    },

    verbose=1,
    seed=42,
    device="cpu",
)


# =====================================================================
# EXACT ZERO INITIAL MEAN
# =====================================================================

with th.no_grad():

    model.policy.action_net.weight.zero_()
    model.policy.action_net.bias.zero_()


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint_callback = CheckpointCallback(
    save_freq=1024,
    save_path=str(
        CHECKPOINT_DIR
    ),
    name_prefix="g1_v6_late",
)


print()
print("=" * 100)
print("V6-LATE TRAINING")
print("=" * 100)

print(
    "Frozen base       : V2 50k"
)

print(
    "Physical envelope : V3"
)

print(
    "Correction start  : step 95"
)

print(
    "Correction ramp   : 5 frames"
)

print(
    "Delta scale       : 0.10"
)

print(
    "Training reset    : frozen V2 state at step 94"
)

print(
    "Learning rate     : 5e-5"
)

print(
    "Initial log std   : -2.3"
)

print(
    "Training steps    : 10,240"
)


model.learn(
    total_timesteps=10240,
    callback=checkpoint_callback,
    progress_bar=False,
)


model.save(
    str(
        FINAL_MODEL
    )
)


vec_env.close()


print()
print("=" * 100)
print("V6-LATE TRAINING COMPLETE")
print("=" * 100)

print(
    "Final:",
    FINAL_MODEL
)

print(
    "Checkpoints:",
    CHECKPOINT_DIR
)
