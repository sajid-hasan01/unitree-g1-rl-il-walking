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


print("Loading V2 and V3 policies...", flush=True)

v2_policy = PPO.load(
    str(V2_MODEL),
    device="cpu",
)

v3_policy = PPO.load(
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


probe = make_env()

names = list(
    probe.joint_names
)

probe.close()


def idx(name):
    return names.index(name)


LEFT_SUPPORT_SAG = [
    idx("left_hip_pitch_joint"),
    idx("left_knee_joint"),
    idx("left_ankle_pitch_joint"),
]

RIGHT_SWING_SAG = [
    idx("right_hip_pitch_joint"),
    idx("right_knee_joint"),
    idx("right_ankle_pitch_joint"),
]

WAIST_PITCH = [
    idx("waist_pitch_joint"),
]

BOTH_LEGS_SAG = (
    LEFT_SUPPORT_SAG
    + RIGHT_SWING_SAG
)

BOTH_SAG_WAIST = (
    BOTH_LEGS_SAG
    + WAIST_PITCH
)


CASES = {
    "V3_BASELINE":
        [],

    "FULL_V2_WINDOW":
        list(range(len(names))),

    "LEFT_SUPPORT_SAG":
        LEFT_SUPPORT_SAG,

    "RIGHT_SWING_SAG":
        RIGHT_SWING_SAG,

    "WAIST_PITCH":
        WAIST_PITCH,

    "BOTH_LEGS_SAG":
        BOTH_LEGS_SAG,

    "BOTH_SAG_WAIST":
        BOTH_SAG_WAIST,
}


WINDOW_START = 78
WINDOW_END = 105


print()
print("=" * 155)
print("V2 -> V3 CRITICAL-WINDOW ACTION TRANSPLANT")
print("V3 ENVIRONMENT — READ ONLY — NO TRAINING")
print(
    f"V2 action replacement window: "
    f"{WINDOW_START}..{WINDOW_END}"
)
print("=" * 155)

print(
    f"{'CASE':20s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'MAX|WY|':>8s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'RAIR':>5s} "
    f"{'RCLR':>7s}"
)

print("-" * 155)


for label, transplant_indices in CASES.items():

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

        right_air_frames = 0
        max_right_clear = float("-inf")

        final = {}

        while True:

            # Both policies see the EXACT SAME current observation.
            v3_action, _ = v3_policy.predict(
                obs,
                deterministic=True,
            )

            v2_action, _ = v2_policy.predict(
                obs,
                deterministic=True,
            )

            v3_action = np.asarray(
                v3_action,
                dtype=np.float32,
            ).reshape(-1)

            v2_action = np.asarray(
                v2_action,
                dtype=np.float32,
            ).reshape(-1)

            action = v3_action.copy()

            # The next env.step() produces step steps+1.
            upcoming_step = steps + 1

            if (
                WINDOW_START
                <= upcoming_step
                <= WINDOW_END
                and len(transplant_indices) > 0
            ):

                action[
                    transplant_indices
                ] = v2_action[
                    transplant_indices
                ]

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


            if not right:

                right_air_frames += 1

                if up >= 0.80:

                    max_right_clear = max(
                        max_right_clear,
                        1000.0
                        * float(
                            info[
                                "right_true_clearance"
                            ]
                        ),
                    )


            if (
                78 <= steps <= 105
                and steps % 5 == 0
            ):

                print(
                    f"  {label:20s} "
                    f"step={steps:03d} "
                    f"Rcontact={int(right)} "
                    f"Rclr="
                    f"{1000.0 * float(info['right_true_clearance']):+6.1f} "
                    f"up={up:.3f} "
                    f"wy={wy:+.2f} "
                    f"yaw={yaw:+.1f}",
                    flush=True,
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
            f"{label:20s} "
            f"{steps:5d} "
            f"{post:5d} "
            f"{float(final['x']):+8.3f} "
            f"{max_y:7.3f} "
            f"{max_yaw:8.1f} "
            f"{min_up:7.3f} "
            f"{max_abs_wy:8.2f} "
            f"{event(first_up90):>5s} "
            f"{event(first_up80):>5s} "
            f"{left_switches:4d} "
            f"{right_switches:4d} "
            f"{right_air_frames:5d} "
            f"{max_right_clear:7.1f}",
            flush=True,
        )

    finally:

        env.close()


print()
print("=" * 155)
print("INTERPRETATION")
print("=" * 155)

print("""
1. FULL_V2_WINDOW is the positive control.

If it substantially extends V3 beyond 99 steps:
    the critical information is indeed in V2's actions
    during steps 78-105.

2. LEFT_SUPPORT_SAG improvement means:
    V3 is failing to use the LEFT stance leg correctly
    while the right leg swings.

3. RIGHT_SWING_SAG improvement means:
    V3's right swing trajectory itself is creating the
    destabilizing pitch momentum.

4. WAIST_PITCH improvement means:
    torso counter-rotation is the missing stabilizer.

5. BOTH_LEGS_SAG or BOTH_SAG_WAIST rescuing the rollout
   when individual groups do not means:
    the solution depends on coordinated multi-joint
    sagittal behavior rather than one joint.

6. If FULL_V2_WINDOW does NOT rescue V3:
    the causal difference begins before step 78 or relies
    strongly on roll/yaw/ankle-roll channels.
""")

print("NO TRAINING WAS PERFORMED.")
print("=" * 155)
