import sys
import faulthandler
from pathlib import Path

import numpy as np

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env_v2 import (
    G1BCLineResidualEnvV2,
)

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


print("Loading V2 policy...", flush=True)

v2_policy = PPO.load(
    str(V2_MODEL),
    device="cpu",
)

print("Loading V3 policy...", flush=True)

v3_policy = PPO.load(
    str(V3_MODEL),
    device="cpu",
)

print("Both policies loaded.", flush=True)


def make_v2_env():

    return G1BCLineResidualEnvV2(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,
    )


def make_v3_env():

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
    (
        "V2POL_V2ENV",
        v2_policy,
        make_v2_env,
    ),
    (
        "V2POL_V3ENV",
        v2_policy,
        make_v3_env,
    ),
    (
        "V3POL_V2ENV",
        v3_policy,
        make_v2_env,
    ),
    (
        "V3POL_V3ENV",
        v3_policy,
        make_v3_env,
    ),
]


print()
print("=" * 145)
print("V2/V3 POLICY x ENVIRONMENT CROSS TEST")
print("READ ONLY — NO TRAINING")
print("=" * 145)

print(
    f"{'CASE':16s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'Y':>8s} "
    f"{'MAXY':>7s} "
    f"{'YAW':>8s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LCLR':>7s} "
    f"{'RCLR':>7s} "
    f"{'AMAX':>6s}"
)

print("-" * 145)


for (
    label,
    policy,
    env_factory,
) in cases:

    print(
        f"Running {label}...",
        flush=True,
    )

    env = env_factory()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0

        max_y = 0.0
        max_yaw = 0.0
        min_up = 1.0
        max_action = 0.0

        left_switches = 0
        right_switches = 0

        prev_left = None
        prev_right = None

        first_up90 = None
        first_up80 = None

        max_left_clear = float("-inf")
        max_right_clear = float("-inf")

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

            y = float(
                info["y"]
            )

            yaw = float(
                info["yaw_deg"]
            )

            up = float(
                info["up_z"]
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


            # Only count clearance while the body
            # is still reasonably upright.
            if up >= 0.80:

                if not left:

                    max_left_clear = max(
                        max_left_clear,
                        1000.0
                        * float(
                            info[
                                "left_true_clearance"
                            ]
                        ),
                    )

                if not right:

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


        post = max(
            0,
            steps - 135,
        )


        def event(value):

            if value is None:
                return "-"

            return str(value)


        if not np.isfinite(
            max_left_clear
        ):
            max_left_clear = float("nan")

        if not np.isfinite(
            max_right_clear
        ):
            max_right_clear = float("nan")


        print(
            f"{label:16s} "
            f"{steps:5d} "
            f"{post:5d} "
            f"{float(final['x']):+8.3f} "
            f"{float(final['y']):+8.3f} "
            f"{max_y:7.3f} "
            f"{float(final['yaw_deg']):+8.1f} "
            f"{max_yaw:8.1f} "
            f"{min_up:7.3f} "
            f"{left_switches:4d} "
            f"{right_switches:4d} "
            f"{event(first_up90):>5s} "
            f"{event(first_up80):>5s} "
            f"{max_left_clear:7.1f} "
            f"{max_right_clear:7.1f} "
            f"{max_action:6.3f}",
            flush=True,
        )

    finally:

        env.close()


print()
print("=" * 145)
print("HOW TO INTERPRET")
print("=" * 145)

print("""
Key comparison #1:

V2POL_V2ENV
vs
V2POL_V3ENV

This measures how much survival is lost purely because of
the V3 environment/control-envelope changes.

Key comparison #2:

V3POL_V2ENV
vs
V3POL_V3ENV

This tells us whether restoring the V2 control envelope
rescues the V3 learned policy.

Most important:

V2POL_V3ENV
vs
V3POL_V3ENV

Same V3 physical controller envelope, different learned policy.

If V2 policy remains much better:
    V3 PPO LEARNING / REWARD is the main regression.

If V3 policy becomes much better inside V2 environment:
    V3 CONTROL ENVELOPE is the main regression.

If both cross-tests behave unexpectedly:
    inspect observation/control compatibility and contact timing next.
""")

print("NO TRAINING WAS PERFORMED.")
print("=" * 145)
