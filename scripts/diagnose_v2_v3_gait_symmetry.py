import sys
import csv
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


print("Loading policies...", flush=True)

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
joint_names = list(probe.joint_names)
probe.close()


pairs = [
    (
        "hip_pitch",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
    ),
    (
        "hip_roll",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
    ),
    (
        "hip_yaw",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
    ),
    (
        "knee",
        "left_knee_joint",
        "right_knee_joint",
    ),
    (
        "ankle_pitch",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    ),
    (
        "ankle_roll",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ),
]


pair_indices = {}

for label, left, right in pairs:

    pair_indices[label] = (
        joint_names.index(left),
        joint_names.index(right),
    )


policies = [
    ("V2", v2_policy),
    ("V3", v3_policy),
]


all_summaries = {}


for policy_name, policy in policies:

    print()
    print("=" * 110)
    print(
        f"{policy_name} POLICY IN V3 ENVIRONMENT"
    )
    print("=" * 110)

    env = make_env()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        rows = []

        step = 0

        previous_left = None
        previous_right = None

        contact_events = []

        first_up90 = None
        first_up80 = None

        max_pair_asym = {
            name: {
                "value": 0.0,
                "step": None,
                "left": 0.0,
                "right": 0.0,
            }
            for name in pair_indices
        }

        while True:

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

            left_contact = bool(
                info["left_contact"]
            )

            right_contact = bool(
                info["right_contact"]
            )

            up = float(
                info["up_z"]
            )

            wx = float(
                env.data.qvel[3]
            )

            wy = float(
                env.data.qvel[4]
            )

            wz = float(
                env.data.qvel[5]
            )


            if (
                first_up90 is None
                and up <= 0.90
            ):
                first_up90 = step

            if (
                first_up80 is None
                and up <= 0.80
            ):
                first_up80 = step


            if (
                previous_left is not None
                and left_contact != previous_left
            ):

                contact_events.append(
                    (
                        step,
                        "LEFT",
                        "CONTACT"
                        if left_contact
                        else "AIR",
                    )
                )

            if (
                previous_right is not None
                and right_contact != previous_right
            ):

                contact_events.append(
                    (
                        step,
                        "RIGHT",
                        "CONTACT"
                        if right_contact
                        else "AIR",
                    )
                )


            previous_left = left_contact
            previous_right = right_contact


            pair_values = {}

            for label, (li, ri) in pair_indices.items():

                left_action = float(
                    action[li]
                )

                right_action = float(
                    action[ri]
                )

                # Absolute difference is used only as
                # a diagnostic indicator.
                asymmetry = abs(
                    left_action
                    - right_action
                )

                pair_values[label] = (
                    left_action,
                    right_action,
                    asymmetry,
                )

                if (
                    asymmetry
                    > max_pair_asym[label]["value"]
                ):

                    max_pair_asym[label] = {
                        "value": asymmetry,
                        "step": step,
                        "left": left_action,
                        "right": right_action,
                    }


            row = {
                "step": step,

                "x":
                    float(info["x"]),

                "y":
                    float(info["y"]),

                "yaw_deg":
                    float(info["yaw_deg"]),

                "up_z":
                    up,

                "wx":
                    wx,

                "wy":
                    wy,

                "wz":
                    wz,

                "left_contact":
                    int(left_contact),

                "right_contact":
                    int(right_contact),

                "left_clear_mm":
                    1000.0
                    * float(
                        info[
                            "left_true_clearance"
                        ]
                    ),

                "right_clear_mm":
                    1000.0
                    * float(
                        info[
                            "right_true_clearance"
                        ]
                    ),
            }


            for label, (
                left_action,
                right_action,
                asymmetry,
            ) in pair_values.items():

                row[
                    f"{label}_L"
                ] = left_action

                row[
                    f"{label}_R"
                ] = right_action

                row[
                    f"{label}_diff"
                ] = asymmetry


            rows.append(row)


            # Print the critical transition region.
            if (
                65 <= step <= 120
                and step % 5 == 0
            ):

                print(
                    f"step={step:03d} "
                    f"L/R={int(left_contact)}/{int(right_contact)} "
                    f"Lclr={row['left_clear_mm']:+6.1f} "
                    f"Rclr={row['right_clear_mm']:+6.1f} "
                    f"Y={row['y']:+.3f} "
                    f"yaw={row['yaw_deg']:+6.1f} "
                    f"up={up:.3f} "
                    f"w=({wx:+.2f},{wy:+.2f},{wz:+.2f})",
                    flush=True,
                )


            if terminated or truncated:
                break


        csv_path = (
            ROOT
            / "results"
            / f"{policy_name.lower()}_policy_v3env_symmetry_trace.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)


        print()
        print(
            f"{policy_name} survived {step} steps"
        )

        print(
            f"UP90 = {first_up90}"
        )

        print(
            f"UP80 = {first_up80}"
        )


        print()
        print("CONTACT EVENTS")

        for event_step, foot, state in contact_events:

            if event_step >= 45:

                print(
                    f"  step={event_step:03d} "
                    f"{foot:5s} -> {state}"
                )


        print()
        print("MAXIMUM LEFT/RIGHT ACTION DIFFERENCE")

        print(
            f"{'PAIR':16s} "
            f"{'STEP':>5s} "
            f"{'LEFT':>8s} "
            f"{'RIGHT':>8s} "
            f"{'DIFF':>8s}"
        )

        print("-" * 52)

        for label in pair_indices:

            result = max_pair_asym[
                label
            ]

            print(
                f"{label:16s} "
                f"{result['step']:5d} "
                f"{result['left']:+8.3f} "
                f"{result['right']:+8.3f} "
                f"{result['value']:8.3f}"
            )


        all_summaries[
            policy_name
        ] = {
            "steps": step,
            "up90": first_up90,
            "up80": first_up80,
            "events": contact_events,
            "asym": max_pair_asym,
        }


        print()
        print(
            "CSV:",
            csv_path,
        )

    finally:

        env.close()


print()
print("=" * 110)
print("DIRECT V2 vs V3 SUMMARY")
print("=" * 110)

print(
    "V2 steps:",
    all_summaries["V2"]["steps"],
)

print(
    "V3 steps:",
    all_summaries["V3"]["steps"],
)

print()

print(
    "V2 UP90 / UP80:",
    all_summaries["V2"]["up90"],
    "/",
    all_summaries["V2"]["up80"],
)

print(
    "V3 UP90 / UP80:",
    all_summaries["V3"]["up90"],
    "/",
    all_summaries["V3"]["up80"],
)

print()
print(
    "Look especially for the FIRST point where "
    "V3 loses alternating right-foot contact or develops "
    "a much larger left/right action imbalance than V2."
)

print()
print("NO TRAINING WAS PERFORMED.")
print("=" * 110)
