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


MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v3_50k.zip"
)

if not MODEL.exists():
    raise FileNotFoundError(MODEL)


print("Loading V3 PPO...", flush=True)

policy = PPO.load(
    str(MODEL),
    device="cpu",
)

print("Model loaded.", flush=True)


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


sagittal_names = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
]

sagittal_idx = [
    names.index(name)
    for name in sagittal_names
]


print()
print("Sagittal joints:")

for name in sagittal_names:
    print("  ", name)


gains = [
    1.00,
    1.15,
    1.30,
    1.45,
    1.60,
]


results = []


print()
print("=" * 145)
print("V3 SAGITTAL RESCUE SWEEP — READ ONLY")
print("=" * 145)

print(
    f"{'GAIN':>6} "
    f"{'STEP':>5} "
    f"{'POST':>5} "
    f"{'X':>8} "
    f"{'Y':>8} "
    f"{'MAXY':>7} "
    f"{'YAW':>8} "
    f"{'MAXYAW':>8} "
    f"{'MINUP':>7} "
    f"{'LSW':>4} "
    f"{'RSW':>4} "
    f"{'UP90':>5} "
    f"{'UP80':>5} "
    f"{'UP70':>5} "
    f"{'AMAX':>6}"
)

print("-" * 145)


for gain in gains:

    print(
        f"\nStarting gain={gain:.2f}",
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
        max_action = 0.0

        left_switches = 0
        right_switches = 0

        previous_left = None
        previous_right = None

        first_up90 = None
        first_up80 = None
        first_up70 = None

        final = {}

        while True:

            raw_action, _ = policy.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                raw_action,
                dtype=np.float32,
            ).reshape(-1)

            # ---------------------------------------------------------
            # Diagnostic intervention:
            # ONLY amplify hip-pitch + knee PPO outputs.
            # ---------------------------------------------------------

            action[sagittal_idx] *= gain

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

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

            if (
                first_up70 is None
                and up <= 0.70
            ):
                first_up70 = steps

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

            if steps % 25 == 0:

                print(
                    f"  gain={gain:.2f} "
                    f"step={steps:03d} "
                    f"Y={y:+.3f} "
                    f"yaw={yaw:+.1f} "
                    f"up={up:.3f}",
                    flush=True,
                )

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


        print(
            f"{gain:6.2f} "
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
            f"{event(first_up90):>5} "
            f"{event(first_up80):>5} "
            f"{event(first_up70):>5} "
            f"{max_action:6.3f}",
            flush=True,
        )


        results.append(
            {
                "gain": gain,
                "steps": steps,
                "post": post,
                "max_y": max_y,
                "max_yaw": max_yaw,
                "lsw": left_switches,
                "rsw": right_switches,
                "amax": max_action,
            }
        )

    finally:

        env.close()


print()
print("=" * 145)
print("SUMMARY")
print("=" * 145)

for result in results:

    print(
        f"gain={result['gain']:.2f} "
        f"steps={result['steps']} "
        f"post={result['post']} "
        f"maxY={result['max_y']:.3f} "
        f"maxYaw={result['max_yaw']:.1f} "
        f"L/R={result['lsw']}/{result['rsw']} "
        f"Amax={result['amax']:.3f}"
    )


print()
print("=" * 145)
print("INTERPRETATION")
print("=" * 145)

print(
    "If gain 1.15-1.45 gives substantially more than 99 steps "
    "while maxY stays below ~0.15:"
)

print(
    "  -> V3 has useful stabilization directions but insufficient "
    "sagittal magnitude."
)

print()

print(
    "If all gains remain around ~99:"
)

print(
    "  -> the problem is policy timing/direction, not simply amplitude."
)

print()

print(
    "If larger gains reduce survival:"
)

print(
    "  -> sagittal under-authority is not the main bottleneck."
)

print()

print(
    "If survival improves but yaw or Y explodes:"
)

print(
    "  -> sagittal correction is useful but couples into asymmetric "
    "contact/yaw dynamics."
)

print()
print("NO TRAINING WAS PERFORMED.")
print("=" * 145)
