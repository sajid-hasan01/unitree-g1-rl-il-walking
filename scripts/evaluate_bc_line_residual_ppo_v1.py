from pathlib import Path
import sys
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env import (
    G1BCLineResidualEnv,
)


MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v1_50k.zip"
)


if not MODEL.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL}"
    )


def make_env():

    return G1BCLineResidualEnv(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,
        residual_scale=0.05,
        target_velocity=-0.18,
        max_episode_steps=600,
    )


model = PPO.load(
    str(MODEL),
    device="cpu",
)


episodes = 5

results = []


print("=" * 112)
print("BC -> RESIDUAL PPO V1 — 50K PHYSICAL EVALUATION")
print("=" * 112)

print(
    "Model:",
    MODEL,
)

print(
    "Goal: stabilize BC gait while following Y=0, yaw=0, direction=-X"
)

print()


for ep in range(episodes):

    env = make_env()

    obs, info = env.reset(
        seed=1000 + ep
    )

    steps = 0
    reward_sum = 0.0

    max_abs_y = 0.0
    rms_y_sum = 0.0

    max_abs_yaw = 0.0

    min_up = 1.0
    min_z = 999.0

    max_left_clear = -999.0
    max_right_clear = -999.0

    left_air = 0
    right_air = 0

    left_switches = 0
    right_switches = 0

    previous_left = None
    previous_right = None

    max_action = 0.0
    action_sq_sum = 0.0

    final_info = {}

    terminated = False
    truncated = False


    while True:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        max_action = max(
            max_action,
            float(
                np.max(
                    np.abs(action)
                )
            ),
        )

        action_sq_sum += float(
            np.mean(
                action ** 2
            )
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        steps += 1
        reward_sum += float(reward)

        y = float(
            info["y"]
        )

        yaw = float(
            info["yaw_deg"]
        )

        max_abs_y = max(
            max_abs_y,
            abs(y),
        )

        rms_y_sum += (
            y * y
        )

        max_abs_yaw = max(
            max_abs_yaw,
            abs(yaw),
        )

        min_up = min(
            min_up,
            float(
                info["up_z"]
            ),
        )

        min_z = min(
            min_z,
            float(
                info["z"]
            ),
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


        if not left:

            left_air += 1

            if (
                info["up_z"] >= 0.80
                and info["z"] >= 0.65
            ):

                max_left_clear = max(
                    max_left_clear,
                    float(
                        info[
                            "left_true_clearance"
                        ]
                    ),
                )


        if not right:

            right_air += 1

            if (
                info["up_z"] >= 0.80
                and info["z"] >= 0.65
            ):

                max_right_clear = max(
                    max_right_clear,
                    float(
                        info[
                            "right_true_clearance"
                        ]
                    ),
                )


        final_info = info


        if terminated or truncated:
            break


    rms_y = math.sqrt(
        rms_y_sum
        / max(
            steps,
            1,
        )
    )

    mean_action_rms = math.sqrt(
        action_sq_sum
        / max(
            steps,
            1,
        )
    )


    if max_left_clear > -900:

        left_clear_mm = (
            1000.0
            * max_left_clear
        )

    else:
        left_clear_mm = float("nan")


    if max_right_clear > -900:

        right_clear_mm = (
            1000.0
            * max_right_clear
        )

    else:
        right_clear_mm = float("nan")


    post_full = max(
        0,
        steps - 135,
    )


    result = {
        "steps": steps,
        "post": post_full,

        "x": float(
            final_info["x"]
        ),

        "y": float(
            final_info["y"]
        ),

        "max_y": max_abs_y,
        "rms_y": rms_y,

        "yaw": float(
            final_info["yaw_deg"]
        ),

        "max_yaw": max_abs_yaw,

        "up": min_up,
        "z": min_z,

        "lsw": left_switches,
        "rsw": right_switches,

        "lair": left_air,
        "rair": right_air,

        "lclr": left_clear_mm,
        "rclr": right_clear_mm,

        "reward": reward_sum,

        "max_action": max_action,

        "action_rms": mean_action_rms,

        "terminated": int(
            terminated
        ),

        "truncated": int(
            truncated
        ),
    }

    results.append(result)


    print(
        f"ep={ep:02d} "
        f"steps={steps:03d} "
        f"post={post_full:03d} "
        f"X={result['x']:+.3f} "
        f"Y={result['y']:+.3f} "
        f"maxY={result['max_y']:.3f} "
        f"yaw={result['yaw']:+.1f} "
        f"maxYaw={result['max_yaw']:.1f} "
        f"up={result['up']:.3f} "
        f"L/Rsw={left_switches}/{right_switches} "
        f"L/Rair={left_air}/{right_air} "
        f"Lclr={left_clear_mm:.1f} "
        f"Rclr={right_clear_mm:.1f} "
        f"Amax={max_action:.2f}"
    )


    env.close()


print()
print("=" * 112)
print("SUMMARY")
print("=" * 112)


def arr(key):

    return np.array(
        [
            x[key]
            for x in results
        ],
        dtype=float,
    )


print(
    f"steps mean/std          = "
    f"{arr('steps').mean():.1f} / "
    f"{arr('steps').std():.1f}"
)

print(
    f"post-full-BC mean       = "
    f"{arr('post').mean():.1f}"
)

print(
    f"final X mean            = "
    f"{arr('x').mean():+.4f} m"
)

print(
    f"max |Y| mean            = "
    f"{arr('max_y').mean():.4f} m"
)

print(
    f"RMS Y mean              = "
    f"{arr('rms_y').mean():.4f} m"
)

print(
    f"max |yaw| mean          = "
    f"{arr('max_yaw').mean():.2f} deg"
)

print(
    f"minimum up_z mean       = "
    f"{arr('up').mean():.4f}"
)

print(
    f"left switches mean      = "
    f"{arr('lsw').mean():.2f}"
)

print(
    f"right switches mean     = "
    f"{arr('rsw').mean():.2f}"
)

print(
    f"max action mean         = "
    f"{arr('max_action').mean():.3f}"
)

print()


# ================================================================
# MILESTONE LOGIC
# ================================================================

mean_steps = arr(
    "steps"
).mean()

mean_max_y = arr(
    "max_y"
).mean()

mean_max_yaw = arr(
    "max_yaw"
).mean()

mean_lsw = arr(
    "lsw"
).mean()

mean_rsw = arr(
    "rsw"
).mean()


milestone_1 = (
    mean_steps > 135
)

line_reasonable = (
    mean_max_y < 0.15
)

yaw_reasonable = (
    mean_max_yaw < 45.0
)

gait_preserved = (
    mean_lsw >= 2.0
    and mean_rsw >= 2.0
)


print("=" * 112)
print("VERDICT")
print("=" * 112)

print(
    "Survives full BC transition :",
    milestone_1,
)

print(
    "Lateral line reasonable     :",
    line_reasonable,
)

print(
    "Yaw reasonable              :",
    yaw_reasonable,
)

print(
    "Bilateral gait preserved    :",
    gait_preserved,
)


if (
    milestone_1
    and line_reasonable
    and yaw_reasonable
):

    print()
    print(
        "V1 RESULT: STABILIZATION PROGRESS CONFIRMED"
    )

else:

    print()
    print(
        "V1 RESULT: NOT YET STABLE — ANALYZE BEFORE MORE TRAINING"
    )

print("=" * 112)
