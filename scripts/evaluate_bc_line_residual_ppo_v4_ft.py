import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env_v4_ft import (
    G1BCLineResidualEnvV4FT,
)


SOURCE = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v2_50k.zip"
)

FINAL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v4_ft_10k.zip"
)

CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v4_ft_checkpoints"
)


def make_env():

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


def evaluate(path):

    policy = PPO.load(
        str(path),
        device="cpu",
    )

    env = make_env()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0
        max_y = 0.0
        max_yaw = 0.0
        min_up = 1.0
        max_action = 0.0

        lsw = 0
        rsw = 0

        prev_l = None
        prev_r = None

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

            l = bool(
                info["left_contact"]
            )

            r = bool(
                info["right_contact"]
            )

            if (
                prev_l is not None
                and l != prev_l
            ):
                lsw += 1

            if (
                prev_r is not None
                and r != prev_r
            ):
                rsw += 1

            prev_l = l
            prev_r = r

            final = info

            if terminated or truncated:
                break


        return {
            "steps": steps,
            "post": max(0, steps - 135),

            "x":
                float(final["x"]),

            "max_y":
                max_y,

            "max_yaw":
                max_yaw,

            "min_up":
                min_up,

            "lsw":
                lsw,

            "rsw":
                rsw,

            "amax":
                max_action,
        }

    finally:

        env.close()


candidates = [
    ("V2_SOURCE", SOURCE),
]


for path in sorted(
    CHECKPOINT_DIR.glob("*.zip")
):

    candidates.append(
        (
            path.stem,
            path,
        )
    )


if FINAL.exists():

    candidates.append(
        (
            "V4_FINAL",
            FINAL,
        )
    )


print("=" * 130)
print("V4-FT CHECKPOINT EVALUATION")
print("=" * 130)

print(
    f"{'MODEL':32s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'AMAX':>6s} "
    f"{'OK':>4s}"
)

print("-" * 130)


results = []


for label, path in candidates:

    if not path.exists():
        continue

    r = evaluate(path)

    good = (
        r["max_y"] < 0.15
        and r["max_yaw"] < 40.0
        and r["lsw"] >= 2
        and r["rsw"] >= 2
    )

    print(
        f"{label:32s} "
        f"{r['steps']:5d} "
        f"{r['post']:5d} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['lsw']:4d} "
        f"{r['rsw']:4d} "
        f"{r['amax']:6.3f} "
        f"{str(good):>4s}"
    )

    r["label"] = label
    r["path"] = path
    r["good"] = good

    results.append(r)


print()
print("=" * 130)
print("BEST CONTROLLED CHECKPOINT")
print("=" * 130)


controlled = [
    r
    for r in results
    if r["good"]
]


if controlled:

    best = max(
        controlled,
        key=lambda r: r["steps"],
    )

    print(
        f"model   = {best['label']}"
    )

    print(
        f"steps   = {best['steps']}"
    )

    print(
        f"post    = {best['post']}"
    )

    print(
        f"maxY    = {best['max_y']:.3f}"
    )

    print(
        f"maxYaw  = {best['max_yaw']:.1f}"
    )

    print(
        f"L/R     = {best['lsw']}/{best['rsw']}"
    )

    print(
        f"path    = {best['path']}"
    )

else:

    print(
        "No fine-tuned checkpoint beat the "
        "trajectory-quality gates."
    )


print()
print("REFERENCE TO BEAT:")
print(
    "119 steps | maxY≈0.130 | maxYaw≈31.9 | L/R≈3/4"
)

print()
print(
    "Major milestone: >135 steps with "
    "maxY<0.15 and maxYaw<40."
)

print("=" * 130)
