import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from envs.g1_closed_loop_tracking_env_v2 import (
    G1ClosedLoopTrackingEnvV2,
)


# Correspond approximately to original 30-Hz frames:
#
#   0, 42, 83, 125, 167
#
# after conversion to 50 Hz.
STARTS = [
    0,
    70,
    138,
    208,
    278,
]


def run(start):

    env = G1ClosedLoopTrackingEnvV2(

        fixed_start_frame=start,

        random_reference_start=False,

        reset_position_noise=0.0,

        reset_velocity_noise=0.0,

        target_smoothing=0.20,

        max_episode_steps=400,
    )


    obs, info = env.reset(
        seed=1234
    )


    print()
    print(
        f"START={start:03d} "
        f"initial_vx="
        f"{env.data.qvel[0]:+.3f} "
        f"ref_vx="
        f"{info['reference_vx']:+.3f} "
        f"ref_yaw="
        f"{info['reference_yaw_deg']:+.1f}"
    )


    step = 0

    max_y = 0.0
    max_yaw = 0.0
    min_up = 1.0

    qerr2 = 0.0
    rooterr2 = 0.0
    velerr2 = 0.0
    orient2 = 0.0

    final = {}


    while True:

        action = np.zeros(
            15,
            dtype=np.float32,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        step += 1


        max_y = max(
            max_y,
            abs(
                float(
                    info["y"]
                )
            ),
        )


        max_yaw = max(
            max_yaw,
            abs(
                float(
                    info["yaw_deg"]
                )
            ),
        )


        min_up = min(
            min_up,
            float(
                info["up_z"]
            ),
        )


        qerr2 += (
            float(
                info[
                    "q_error_rms"
                ]
            ) ** 2
        )


        rooterr2 += (
            float(
                info[
                    "root_position_error"
                ]
            ) ** 2
        )


        velerr2 += (
            float(
                info[
                    "root_velocity_error"
                ]
            ) ** 2
        )


        orient2 += (
            float(
                info[
                    "orientation_error_deg"
                ]
            ) ** 2
        )


        final = info


        if (
            step == 1
            or step % 25 == 0
            or terminated
            or truncated
        ):

            print(
                f"  step={step:03d} "
                f"ref={info['reference_index']:03d} "
                f"x={info['x']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"y={info['y']:+.3f} "
                f"yaw={info['yaw_deg']:+.1f} "
                f"up={info['up_z']:.3f} "
                f"qerr="
                f"{info['q_error_rms']:.3f} "
                f"rooterr="
                f"{info['root_position_error']:.3f} "
                f"verr="
                f"{info['root_velocity_error']:.3f} "
                f"orierr="
                f"{info['orientation_error_deg']:.1f}"
            )


        if terminated or truncated:
            break


    result = {
        "start":
            start,

        "steps":
            step,

        "done":
            bool(
                final.get(
                    "completed",
                    False,
                )
            ),

        "x":
            float(
                final["x"]
            ),

        "max_y":
            max_y,

        "max_yaw":
            max_yaw,

        "min_up":
            min_up,

        "qerr":
            float(
                np.sqrt(
                    qerr2
                    / max(
                        step,
                        1,
                    )
                )
            ),

        "rooterr":
            float(
                np.sqrt(
                    rooterr2
                    / max(
                        step,
                        1,
                    )
                )
            ),

        "velerr":
            float(
                np.sqrt(
                    velerr2
                    / max(
                        step,
                        1,
                    )
                )
            ),

        "orierr":
            float(
                np.sqrt(
                    orient2
                    / max(
                        step,
                        1,
                    )
                )
            ),
    }


    env.close()

    return result


print("=" * 150)
print("TRACKING V2 ROOT-ALIGNED REFERENCE BASELINE")
print("ZERO PPO ACTION")
print("NO TRAINING")
print("=" * 150)


results = [
    run(s)
    for s in STARTS
]


print()
print("=" * 150)
print("FINAL V2 RESULTS")
print("=" * 150)

print(
    f"{'START':>5s} "
    f"{'STEP':>5s} "
    f"{'DONE':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'YAW':>7s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'ROOTERR':>8s} "
    f"{'VELERR':>8s} "
    f"{'ORIERR':>8s}"
)

print("-" * 150)


for r in results:

    print(
        f"{r['start']:5d} "
        f"{r['steps']:5d} "
        f"{str(r['done']):>5s} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:7.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['qerr']:7.3f} "
        f"{r['rooterr']:8.3f} "
        f"{r['velerr']:8.3f} "
        f"{r['orierr']:8.2f}"
    )


completed = sum(
    int(
        r["done"]
    )
    for r in results
)


print()
print("=" * 150)
print("DECISION")
print("=" * 150)

print(
    f"completed = "
    f"{completed}/{len(results)}"
)

print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 150)
