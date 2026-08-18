import sys
from pathlib import Path

import numpy as np

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

from stable_baselines3.common.utils import (
    FloatSchedule,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from envs.g1_bc_line_residual_env_v4_ft import (
    G1BCLineResidualEnvV4FT,
)


SOURCE_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v2_50k.zip"
)

FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v4_ft_10k.zip"
)

CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v4_ft_checkpoints"
)


def raw_env():

    return G1BCLineResidualEnvV4FT(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,
    )


def evaluate(policy):

    env = raw_env()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0

        max_y = 0.0
        max_yaw = 0.0
        min_up = 1.0

        left_switches = 0
        right_switches = 0

        previous_left = None
        previous_right = None

        max_action = 0.0

        final = {}

        while True:

            action, _ = policy.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

            max_action = max(
                max_action,
                float(
                    np.max(
                        np.abs(action)
                    )
                ),
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1

            max_y = max(
                max_y,
                abs(float(info["y"])),
            )

            max_yaw = max(
                max_yaw,
                abs(float(info["yaw_deg"])),
            )

            min_up = min(
                min_up,
                float(info["up_z"]),
            )

            left = bool(
                info["left_contact"]
            )

            right = bool(
                info["right_contact"]
            )

            if (
                previous_left is not None
                and left != previous_left
            ):
                left_switches += 1

            if (
                previous_right is not None
                and right != previous_right
            ):
                right_switches += 1

            previous_left = left
            previous_right = right

            final = info

            if terminated or truncated:
                break


        return {
            "steps": steps,
            "x": float(final["x"]),
            "max_y": max_y,
            "max_yaw": max_yaw,
            "min_up": min_up,
            "lsw": left_switches,
            "rsw": right_switches,
            "amax": max_action,
        }

    finally:

        env.close()


if not SOURCE_MODEL.exists():

    raise FileNotFoundError(
        SOURCE_MODEL
    )


print("=" * 100)
print("V4-FT PRE-FLIGHT")
print("=" * 100)

source_policy = PPO.load(
    str(SOURCE_MODEL),
    device="cpu",
)

baseline = evaluate(
    source_policy
)

print(
    "V2 policy in V4-FT env:"
)

print(
    f"steps={baseline['steps']} "
    f"X={baseline['x']:+.3f} "
    f"maxY={baseline['max_y']:.3f} "
    f"maxYaw={baseline['max_yaw']:.1f} "
    f"L/R={baseline['lsw']}/{baseline['rsw']} "
    f"Amax={baseline['amax']:.3f}"
)


# This should reproduce the proven V2-policy/V3-envelope
# result around 119 steps.

if not (
    115 <= baseline["steps"] <= 123
):

    raise SystemExit(
        "ABORT: V4-FT pre-flight does not reproduce "
        "the expected V2 + restricted-ankle behavior."
    )


if baseline["max_y"] > 0.20:

    raise SystemExit(
        "ABORT: unexpected lateral regression."
    )


if baseline["max_yaw"] > 45.0:

    raise SystemExit(
        "ABORT: unexpected yaw regression."
    )


print()
print("PRE-FLIGHT: PASS")
print()


# =====================================================================
# TRAINING ENVIRONMENT
# =====================================================================

def make_train_env():

    return Monitor(
        raw_env()
    )


vec_env = DummyVecEnv(
    [make_train_env]
)


# =====================================================================
# CRITICAL:
# LOAD V2 WEIGHTS.
#
# We are NOT initializing another PPO policy from scratch.
# =====================================================================

model = PPO.load(
    str(SOURCE_MODEL),
    env=vec_env,
    device="cpu",
)


# =====================================================================
# VERY CONSERVATIVE FINE-TUNING
# =====================================================================

model.learning_rate = 2.0e-5

# Rebuild LR schedule after changing learning rate.
model._setup_lr_schedule()


# Small PPO clipping:
# discourage moving far from the successful V2 controller.
model.clip_range = FloatSchedule(
    0.08
)


# Less exploration than V2/V3.
model.ent_coef = 0.0005


# Fewer optimization passes per rollout.
model.n_epochs = 5


# Stop an update if policy changes too aggressively.
model.target_kl = 0.01


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint_callback = CheckpointCallback(
    save_freq=2048,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="g1_v4_ft",
)


print("=" * 100)
print("V4-FT TRAINING")
print("=" * 100)

print("Starting policy : V2 50k")
print("Environment     : V2 reward + V3 ankle envelope")
print("Learning rate   : 2e-5")
print("Clip range      : 0.08")
print("Epochs          : 5")
print("Target KL       : 0.01")
print("Additional steps: 10,240")
print()


model.learn(
    total_timesteps=10240,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
    progress_bar=False,
)


model.save(
    str(FINAL_MODEL)
)


print()
print("=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)

print(
    "Final:",
    FINAL_MODEL,
)

print(
    "Checkpoints:",
    CHECKPOINT_DIR,
)

vec_env.close()
