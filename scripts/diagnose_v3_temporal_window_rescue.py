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


V2_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v2_50k.zip"
)

V3_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v3_50k.zip"
)


print("Loading V2 and V3...", flush=True)

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


# V3 is used everywhere except inside the selected window,
# where ALL 15 actions come from V2.
cases = [
    ("V3_BASE", None),

    ("V2_WIN_45_60", (45, 60)),

    ("V2_WIN_60_70", (60, 70)),
    ("V2_WIN_70_77", (70, 77)),
    ("V2_WIN_77_85", (77, 85)),

    ("V2_WIN_60_77", (60, 77)),
    ("V2_WIN_70_85", (70, 85)),

    ("V2_WIN_60_85", (60, 85)),
]


print()
print("=" * 145)
print("V3 TEMPORAL WINDOW RESCUE TEST")
print("V3 ENVIRONMENT - NO TRAINING")
print("=" * 145)

print(
    f"{'CASE':18s} "
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
    f"{'UP75':>7s}"
)

print("-" * 145)


for label, window in cases:

    print(
        f"Running {label}...",
        flush=True,
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
        max_abs_wy = 0.0

        first_up95 = None
        first_up90 = None
        first_up80 = None

        left_switches = 0
        right_switches = 0

        previous_left = None
        previous_right = None

        max_right_clearance = float("-inf")

        wy75 = float("nan")
        up75 = float("nan")

        final = {}

        while True:

            upcoming_step = steps + 1

            # Both policies observe the same physical state.
            a_v3, _ = v3.predict(
                obs,
                deterministic=True,
            )

            a_v2, _ = v2.predict(
                obs,
                deterministic=True,
            )

            a_v3 = np.asarray(
                a_v3,
                dtype=np.float32,
            ).reshape(-1)

            a_v2 = np.asarray(
                a_v2,
                dtype=np.float32,
            ).reshape(-1)


            # Default controller is V3.
            action = a_v3.copy()


            # Replace ALL action channels with V2 only
            # inside the selected temporal window.
            if window is not None:

                start, end = window

                if (
                    start
                    <= upcoming_step
                    <= end
                ):
                    action = a_v2.copy()


            action = np.clip(
                action,
                -1.0,
                1.0,
            )


            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1


            y = float(
                info["y"]
            )

            yaw = float(
                info["yaw_deg"]
            )

            up = float(
                info["up_z"]
            )

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


            if steps == 75:

                wy75 = wy
                up75 = up


            if (
                first_up95 is None
                and up <= 0.95
            ):
                first_up95 = steps


            if (
                first_up90 is None
                and up <= 0.90
            ):
                first_up90 = steps


            if (
                first_up80 is None
                and up <= 0.80
            ):
                first_up80 = steps


            left_contact = bool(
                info["left_contact"]
            )

            right_contact = bool(
                info["right_contact"]
            )


            if (
                previous_left is not None
                and left_contact != previous_left
            ):
                left_switches += 1


            if (
                previous_right is not None
                and right_contact != previous_right
            ):
                right_switches += 1


            previous_left = left_contact
            previous_right = right_contact


            if (
                not right_contact
                and up >= 0.80
            ):

                max_right_clearance = max(
                    max_right_clearance,
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


        post = max(
            0,
            steps - 135,
        )


        if not np.isfinite(
            max_right_clearance
        ):
            max_right_clearance = float("nan")


        def evt(value):

            if value is None:
                return "-"

            return str(value)


        print(
            f"{label:18s} "
            f"{steps:5d} "
            f"{post:5d} "
            f"{max_y:7.3f} "
            f"{max_yaw:8.1f} "
            f"{min_up:7.3f} "
            f"{max_abs_wy:7.2f} "
            f"{evt(first_up95):>5s} "
            f"{evt(first_up90):>5s} "
            f"{evt(first_up80):>5s} "
            f"{left_switches:4d} "
            f"{right_switches:4d} "
            f"{max_right_clearance:7.1f} "
            f"{wy75:+7.3f} "
            f"{up75:7.3f}",
            flush=True,
        )

    finally:

        env.close()


print()
print("=" * 145)
print("INTERPRETATION")
print("=" * 145)

print("""
Main test:

V2_WIN_60_85

If this approaches the V2-policy/V3-env benchmark (~119 steps),
then V3's major failure has been localized to steps 60-85.

Compare:

45-60
60-70
70-77
77-85

to determine where the harmful behavior begins.

Important cumulative cases:

60-77
70-85
60-85

If 77-85 alone gives a large improvement:
    first-swing control is the dominant issue.

If 60-70 alone gives a large improvement:
    pre-swing momentum preparation is the dominant issue.

If no short window helps but 60-85 does:
    V2 requires coordinated state preparation across the whole
    transition interval.

WY75 is especially important.

Baseline V3 WY75 is approximately -0.250 rad/s.
V2 is approximately -0.024 rad/s.

If a successful window pushes WY75 toward zero and also
increases survival, that strongly links the early pitch momentum
to the later fall.
""")

print("NO TRAINING WAS PERFORMED.")
print("=" * 145)
