import math
import numpy as np

from envs.g1_bc_line_residual_env import (
    G1BCLineResidualEnv,
)


env = G1BCLineResidualEnv(
    start_frame=25,
    stand_frames=45,
    transition_frames=90,
    target_smoothing=0.35,
    residual_scale=0.05,
    target_velocity=-0.18,
    max_episode_steps=600,
)

obs, info = env.reset()

print("=" * 100)
print("BC LINE RESIDUAL ENV ? ZERO-ACTION VALIDATION")
print("=" * 100)

print("Observation shape =", obs.shape)
print("Action shape      =", env.action_space.shape)
print("Start frame       =", env.start_frame)
print("Transition        =", env.transition_frames)
print("Residual action   = EXACT ZERO")
print()

steps = 0

max_abs_y = 0.0
max_abs_yaw = 0.0

min_up = 1.0
min_z = 999.0

left_switches = 0
right_switches = 0

prev_l = None
prev_r = None

max_left_clear = -999.0
max_right_clear = -999.0

final_info = {}

while True:

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    steps += 1

    max_abs_y = max(
        max_abs_y,
        abs(info["y"]),
    )

    max_abs_yaw = max(
        max_abs_yaw,
        abs(info["yaw_deg"]),
    )

    min_up = min(
        min_up,
        info["up_z"],
    )

    min_z = min(
        min_z,
        info["z"],
    )

    l = info["left_contact"]
    r = info["right_contact"]

    if prev_l is not None and l != prev_l:
        left_switches += 1

    if prev_r is not None and r != prev_r:
        right_switches += 1

    prev_l = l
    prev_r = r

    if (
        not l
        and info["up_z"] >= 0.80
        and info["z"] >= 0.65
    ):
        max_left_clear = max(
            max_left_clear,
            info["left_true_clearance"],
        )

    if (
        not r
        and info["up_z"] >= 0.80
        and info["z"] >= 0.65
    ):
        max_right_clear = max(
            max_right_clear,
            info["right_true_clearance"],
        )

    final_info = info

    if terminated or truncated:
        break


print("steps                     =", steps)
print(
    "full BC authority frame   =",
    env.stand_frames
    + env.transition_frames,
)

print(
    "frames after full BC      =",
    max(
        0,
        steps
        - (
            env.stand_frames
            + env.transition_frames
        ),
    ),
)

print()
print("STRAIGHT LINE")

print(
    f"final X                   = "
    f"{final_info['x']:+.4f} m"
)

print(
    f"final Y                   = "
    f"{final_info['y']:+.4f} m"
)

print(
    f"max |Y|                   = "
    f"{max_abs_y:.4f} m"
)

print(
    f"final yaw                 = "
    f"{final_info['yaw_deg']:+.2f} deg"
)

print(
    f"max |yaw|                 = "
    f"{max_abs_yaw:.2f} deg"
)

print()
print("BALANCE")

print(
    f"min up_z                  = "
    f"{min_up:.4f}"
)

print(
    f"min base height           = "
    f"{min_z:.4f} m"
)

print()
print("GAIT")

print(
    "left contact switches     =",
    left_switches,
)

print(
    "right contact switches    =",
    right_switches,
)

if max_left_clear > -900:

    print(
        f"max LEFT true clearance   = "
        f"{1000*max_left_clear:.3f} mm"
    )

else:
    print(
        "max LEFT true clearance   = none"
    )

if max_right_clear > -900:

    print(
        f"max RIGHT true clearance  = "
        f"{1000*max_right_clear:.3f} mm"
    )

else:
    print(
        "max RIGHT true clearance  = none"
    )

print()
print("=" * 100)

# Reference from the independent transition=90 evaluator:
print("REFERENCE transition-90:")
print("  frames      ~94")
print("  final X     ~-0.574 m")
print("  final Y     ~+0.059 m")
print("  final yaw   ~+13 deg")
print()
print(
    "Small differences are acceptable because "
    "the Gym termination threshold is slightly different."
)

env.close()
