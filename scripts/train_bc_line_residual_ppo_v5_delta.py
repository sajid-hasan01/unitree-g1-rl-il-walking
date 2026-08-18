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

from stable_baselines3.common.env_checker import (
    check_env,
)

from stable_baselines3.common.monitor import (
    Monitor,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from envs.g1_bc_line_residual_env_v5_delta import (
    G1BCLineResidualEnvV5Delta,
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v5_delta_10k.zip"
)

CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v5_delta_checkpoints"
)


def make_raw_env():

    return G1BCLineResidualEnvV5Delta(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        delta_action_scale=0.10,
    )


def evaluate_zero_delta():

    env = make_raw_env()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0

        max_y = 0.0
        max_yaw = 0.0
        min_up = 1.0

        lsw = 0
        rsw = 0

        prev_l = None
        prev_r = None

        max_combined = 0.0

        final = {}

        while True:

            delta = np.zeros(
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
                delta
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

            min_up = min(
                min_up,
                float(
                    info["up_z"]
                ),
            )

            max_combined = max(
                max_combined,
                float(
                    info[
                        "v5_combined_action_max"
                    ]
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
            "steps": steps,
            "x": float(
                final["x"]
            ),
            "max_y": max_y,
            "max_yaw": max_yaw,
            "min_up": min_up,
            "lsw": lsw,
            "rsw": rsw,
            "combined_max": max_combined,
        }

    finally:

        env.close()


# =====================================================================
# ENVIRONMENT CHECK
# =====================================================================

print("=" * 100)
print("V5 ENVIRONMENT CHECK")
print("=" * 100)

check = make_raw_env()

check_env(
    check,
    warn=True,
)

check.close()

print("Gym environment check: PASS")


# =====================================================================
# ZERO-DELTA PRE-FLIGHT
#
# This MUST reproduce:
#
# Frozen V2 + V3 physical envelope
#
# ~119 steps
# maxY ~0.130
# maxYaw ~31.9
# L/R ~3/4
# =====================================================================

print()
print("=" * 100)
print("V5 ZERO-DELTA PRE-FLIGHT")
print("=" * 100)

baseline = evaluate_zero_delta()

print(
    f"steps={baseline['steps']} "
    f"X={baseline['x']:+.3f} "
    f"maxY={baseline['max_y']:.3f} "
    f"maxYaw={baseline['max_yaw']:.1f} "
    f"L/R={baseline['lsw']}/{baseline['rsw']} "
    f"combinedAmax={baseline['combined_max']:.3f}"
)


if not (
    115
    <= baseline["steps"]
    <= 123
):

    raise SystemExit(
        "ABORT: zero-delta V5 does not reproduce "
        "the frozen-V2 benchmark."
    )


if baseline["max_y"] > 0.15:

    raise SystemExit(
        "ABORT: zero-delta maxY is unexpectedly high."
    )


if baseline["max_yaw"] > 40.0:

    raise SystemExit(
        "ABORT: zero-delta yaw is unexpectedly high."
    )


if (
    baseline["lsw"] < 2
    or baseline["rsw"] < 2
):

    raise SystemExit(
        "ABORT: zero-delta bilateral gait was not preserved."
    )


print("ZERO-DELTA PRE-FLIGHT: PASS")


# =====================================================================
# TRAINING ENV
# =====================================================================

def make_train_env():

    return Monitor(
        make_raw_env()
    )


vec_env = DummyVecEnv(
    [
        make_train_env
    ]
)


# =====================================================================
# NEW DELTA PPO
#
# IMPORTANT:
# This is NOT V2.
# This policy controls only the small correction.
#
# log_std_init = -2:
# small initial exploration.
# =====================================================================

model = PPO(
    "MlpPolicy",
    vec_env,

    learning_rate=5.0e-5,

    n_steps=1024,
    batch_size=64,
    n_epochs=5,

    gamma=0.995,
    gae_lambda=0.95,

    clip_range=0.10,

    ent_coef=0.0002,
    vf_coef=0.50,

    max_grad_norm=0.50,

    target_kl=0.01,

    policy_kwargs={
        "net_arch": [64, 64],
        "log_std_init": -2.0,
    },

    verbose=1,
    seed=42,
    device="cpu",
)


# =====================================================================
# FORCE INITIAL DETERMINISTIC DELTA TO EXACT ZERO
#
# This guarantees that before learning:
#
# deterministic V5
# =
# frozen V2 exactly.
#
# The layer remains trainable afterward.
# =====================================================================

with th.no_grad():

    model.policy.action_net.weight.zero_()
    model.policy.action_net.bias.zero_()


print()
print("=" * 100)
print("V5 DELTA POLICY INITIALIZATION")
print("=" * 100)

print(
    "Frozen base       : V2 50k"
)

print(
    "Physical envelope : V3"
)

print(
    "Delta scale       : 0.10"
)

print(
    "Initial mean      : EXACT ZERO"
)

print(
    "Initial log std   : -2.0"
)

print(
    "Learning rate     : 5e-5"
)

print(
    "Clip range        : 0.10"
)

print(
    "Training steps    : 10,240"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint_callback = CheckpointCallback(
    save_freq=2048,
    save_path=str(
        CHECKPOINT_DIR
    ),
    name_prefix="g1_v5_delta",
)


# =====================================================================
# TRAIN ONLY 10,240 STEPS
# =====================================================================

print()
print("=" * 100)
print("TRAINING V5 DELTA PPO")
print("=" * 100)


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


print()
print("=" * 100)
print("V5 TRAINING COMPLETE")
print("=" * 100)

print(
    "Final:",
    FINAL_MODEL
)

print(
    "Checkpoints:",
    CHECKPOINT_DIR
)


vec_env.close()
