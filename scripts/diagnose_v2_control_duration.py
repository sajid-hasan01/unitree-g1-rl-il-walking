import sys
import faulthandler
from pathlib import Path

import numpy as np

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from envs.g1_bc_line_residual_env_v3 import (
    G1BCLineResidualEnvV3,
)


V2_MODEL = ROOT / "models" / "g1_ppo_bc_line_residual_v2_50k.zip"
V3_MODEL = ROOT / "models" / "g1_ppo_bc_line_residual_v3_50k.zip"


print("Loading policies...", flush=True)

v2 = PPO.load(
    str(V2_MODEL),
    device="cpu",
)

v3 = PPO.load(
    str(V3_MODEL),
    device="cpu",
)

print("Policies loaded.", flush=True)


def make_env():

    return G1BCLineResidualEnvV3(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,
        residual_scale=0.14,
        residual_ramp_frames=30,
        target_velocity=-0.18,
        max_episode_steps=600,
    )


cases = [
    ("V3_ALL", None),

    ("V2_60_77", 77),
    ("V2_60_85", 85),
    ("V2_60_90", 90),
    ("V2_60_95", 95),
    ("V2_60_100", 100),
    ("V2_60_105", 105),
    ("V2_60_110", 110),
    ("V2_60_115", 115),

    ("V2_ALL", "ALL"),
]


print()
print("=" * 150)
print("V2 EARLY-CONTROL DURATION SWEEP")
print("V3 ENVIRONMENT — READ ONLY")
print("=" * 150)

print(
    f"{'CASE':14s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'MAXWY':>7s} "
    f"{'UP95':>5s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'RCLR':>7s} "
    f"{'WY75':>7s} "
    f"{'WY90':>7s} "
    f"{'WY100':>7s}"
)

print("-" * 150)


for label, end_step in cases:

    env = make_env()

    try:

        obs, _ = env.reset(seed=1234)

        step = 0

        max_y = 0.0
        max_yaw = 0.0
        min_up = 1.0
        max_abs_wy = 0.0

        up95 = None
        up90 = None
        up80 = None

        wy75 = float("nan")
        wy90 = float("nan")
        wy100 = float("nan")

        lsw = 0
        rsw = 0

        prev_l = None
        prev_r = None

        max_rclr = float("-inf")

        final = {}

        while True:

            next_step = step + 1

            # ---------------------------------------------------------
            # Controller choice
            # ---------------------------------------------------------

            if end_step == "ALL":

                policy = v2

            elif end_step is None:

                policy = v3

            elif (
                60 <= next_step <= end_step
            ):

                policy = v2

            else:

                policy = v3


            action, _ = policy.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)


            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            step += 1


            y = float(info["y"])
            yaw = float(info["yaw_deg"])
            up = float(info["up_z"])

            wy = float(
                env.data.qvel[4]
            )


            max_y = max(
                max_y,
                abs(y),
            )

            max_yaw = max(
                max_yaw,
                abs(yaw),
            )

            min_up = min(
                min_up,
                up,
            )

            max_abs_wy = max(
                max_abs_wy,
                abs(wy),
            )


            if step == 75:
                wy75 = wy

            if step == 90:
                wy90 = wy

            if step == 100:
                wy100 = wy


            if (
                up95 is None
                and up <= 0.95
            ):
                up95 = step

            if (
                up90 is None
                and up <= 0.90
            ):
                up90 = step

            if (
                up80 is None
                and up <= 0.80
            ):
                up80 = step


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


            if (
                not right
                and up >= 0.80
            ):

                max_rclr = max(
                    max_rclr,
                    1000.0
                    * float(
                        info[
                            "right_true_clearance"
                        ]
                    ),
                )


            final = info


            if terminated or truncated:
                break


        if not np.isfinite(max_rclr):
            max_rclr = float("nan")


        post = max(
            0,
            step - 135,
        )


        def ev(value):
            return "-" if value is None else str(value)


        print(
            f"{label:14s} "
            f"{step:5d} "
            f"{post:5d} "
            f"{max_y:7.3f} "
            f"{max_yaw:8.1f} "
            f"{min_up:7.3f} "
            f"{max_abs_wy:7.2f} "
            f"{ev(up95):>5s} "
            f"{ev(up90):>5s} "
            f"{ev(up80):>5s} "
            f"{lsw:4d} "
            f"{rsw:4d} "
            f"{max_rclr:7.1f} "
            f"{wy75:+7.3f} "
            f"{wy90:+7.3f} "
            f"{wy100:+7.3f}",
            flush=True,
        )

    finally:

        env.close()


print()
print("=" * 150)
print("HOW TO INTERPRET")
print("=" * 150)

print("""
The key question:

How long must V2 remain in control before V3 can take over
without immediately losing the stability benefit?

Examples:

If V2_60_90 ~= 115-119 steps:
    the important V2 behavior ends around the first right swing.

If V2_60_100 ~= 115-119 but V2_60_90 does not:
    V2 control is needed through the first right swing.

If V2_60_110 or V2_60_115 is necessary:
    V3 is failing through essentially the entire first gait cycle.

If even V2_60_115 drops strongly after switching to V3:
    V3 is fundamentally unable to maintain the stabilized gait,
    and the correct next strategy is V2 fine-tuning rather than
    trying to repair V3.

Also compare:

WY75
WY90
WY100

to see exactly when pitch angular velocity begins diverging.
""")

print("NO TRAINING WAS PERFORMED.")
print("=" * 150)
