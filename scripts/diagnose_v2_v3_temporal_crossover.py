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


print("Loading V2/V3 policies...", flush=True)

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


# -------------------------------------------------------------
# Cases:
#
# mode = fixed:
#   one policy for complete episode
#
# mode = v2_to_v3:
#   V2 through cutoff, then V3
#
# mode = v3_to_v2:
#   V3 through cutoff, then V2
# -------------------------------------------------------------

cases = [
    ("V3_ALL", "fixed_v3", None),
    ("V2_ALL", "fixed_v2", None),

    ("V2_TO_V3_60", "v2_to_v3", 60),
    ("V2_TO_V3_70", "v2_to_v3", 70),
    ("V2_TO_V3_77", "v2_to_v3", 77),
    ("V2_TO_V3_85", "v2_to_v3", 85),
    ("V2_TO_V3_95", "v2_to_v3", 95),

    ("V3_TO_V2_60", "v3_to_v2", 60),
    ("V3_TO_V2_70", "v3_to_v2", 70),
    ("V3_TO_V2_77", "v3_to_v2", 77),
    ("V3_TO_V2_85", "v3_to_v2", 85),
    ("V3_TO_V2_95", "v3_to_v2", 95),
]


print()
print("=" * 150)
print("V2 / V3 TEMPORAL CROSSOVER DIAGNOSTIC")
print("V3 PHYSICAL ENVIRONMENT — NO TRAINING")
print("=" * 150)

print(
    f"{'CASE':17s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'MAXWY':>7s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'RCLR':>7s} "
    f"{'S75_WY':>8s} "
    f"{'S75_UP':>8s}"
)

print("-" * 150)


for label, mode, cutoff in cases:

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

        first_up90 = None
        first_up80 = None

        left_switches = 0
        right_switches = 0

        prev_left = None
        prev_right = None

        max_right_clear = float("-inf")

        step75_wy = float("nan")
        step75_up = float("nan")

        final = {}

        while True:

            upcoming_step = steps + 1

            # ---------------------------------------------------------
            # Select controller for this step.
            # ---------------------------------------------------------

            if mode == "fixed_v2":

                selected_policy = v2

            elif mode == "fixed_v3":

                selected_policy = v3

            elif mode == "v2_to_v3":

                if upcoming_step <= cutoff:
                    selected_policy = v2
                else:
                    selected_policy = v3

            elif mode == "v3_to_v2":

                if upcoming_step <= cutoff:
                    selected_policy = v3
                else:
                    selected_policy = v2

            else:

                raise RuntimeError(
                    f"Unknown mode: {mode}"
                )


            action, _ = selected_policy.predict(
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

                step75_wy = wy
                step75_up = up


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


            left = bool(
                info["left_contact"]
            )

            right = bool(
                info["right_contact"]
            )


            if (
                prev_left is not None
                and left != prev_left
            ):
                left_switches += 1


            if (
                prev_right is not None
                and right != prev_right
            ):
                right_switches += 1


            prev_left = left
            prev_right = right


            if (
                not right
                and up >= 0.80
            ):

                max_right_clear = max(
                    max_right_clear,
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


        if not np.isfinite(
            max_right_clear
        ):
            max_right_clear = float("nan")


        post = max(
            0,
            steps - 135,
        )


        def event(value):

            if value is None:
                return "-"

            return str(value)


        print(
            f"{label:17s} "
            f"{steps:5d} "
            f"{post:5d} "
            f"{max_y:7.3f} "
            f"{max_yaw:8.1f} "
            f"{min_up:7.3f} "
            f"{max_abs_wy:7.2f} "
            f"{event(first_up90):>5s} "
            f"{event(first_up80):>5s} "
            f"{left_switches:4d} "
            f"{right_switches:4d} "
            f"{max_right_clear:7.1f} "
            f"{step75_wy:+8.3f} "
            f"{step75_up:8.3f}",
            flush=True,
        )

    finally:

        env.close()


print()
print("=" * 150)
print("INTERPRETATION")
print("=" * 150)

print("""
A) V2_TO_V3 tests whether V2 must prepare the robot's physical
   state before V3 takes control.

Example:

V2_TO_V3_77 >> V3_ALL

means the important difference develops BEFORE the first swing
and V2 places the robot into a better initial walking state.


B) V3_TO_V2 tests whether V2 can rescue a trajectory after V3
   has already controlled it.

Example:

V3_TO_V2_77 ≈ V2_ALL

means the early V3 state is still recoverable and V2's continuous
feedback strategy is what matters.


C) If V2_TO_V3 cases quickly collapse back toward ~99 but
   V3_TO_V2 cases recover strongly:

the problem is NOT initial-state preparation.

It means V3's ongoing closed-loop policy is wrong.


D) If V3_TO_V2_77 or V3_TO_V2_85 cannot recover:

V3 has already pushed the robot into an unrecoverable dynamic
state before/right around first swing.


E) V2_ALL remains the benchmark:

~119 steps
maxY ~0.130
maxYaw ~31.9
bilateral gait.
""")

print("NO TRAINING WAS PERFORMED.")
print("=" * 150)
