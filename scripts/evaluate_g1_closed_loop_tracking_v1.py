import sys
from pathlib import Path

import numpy as np

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from stable_baselines3 import PPO

from envs.g1_closed_loop_tracking_env import (
    G1ClosedLoopTrackingEnv,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_tracking_v1_checkpoints"
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_closed_loop_tracking_v1_50k.zip"
)


# Source-frame 25 at 30 Hz corresponds to about frame 42 at 50 Hz.
START_FRAMES = [
    0,
    42,
    83,
    125,
    167,
]


def rollout(
    model,
    start_frame,
):

    env = G1ClosedLoopTrackingEnv(

        fixed_start_frame=
            start_frame,

        random_reference_start=
            False,

        reset_position_noise=
            0.0,

        reset_velocity_noise=
            0.0,

        control_hz=
            50.0,

        target_velocity=
            -0.18,

        max_episode_steps=
            400,
    )


    try:

        obs, _ = env.reset(
            seed=1234
        )


        steps = 0

        max_y = 0.0
        max_yaw = 0.0

        min_up = 1.0

        qerr_sum = 0.0

        contact_sum = 0.0

        action_max = 0.0

        left_switches = 0
        right_switches = 0

        previous_left = None
        previous_right = None

        final = {}


        while True:

            action, _ = (
                model.predict(
                    obs,
                    deterministic=True,
                )
            )


            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(
                -1
            )


            action_max = max(
                action_max,

                float(
                    np.max(
                        np.abs(
                            action
                        )
                    )
                ),
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


            steps += 1


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
                        info[
                            "yaw_deg"
                        ]
                    )
                ),
            )


            min_up = min(
                min_up,
                float(
                    info[
                        "up_z"
                    ]
                ),
            )


            qerr_sum += float(
                info[
                    "q_error_rms"
                ]
            ) ** 2


            contact_sum += float(
                info[
                    "contact_match"
                ]
            )


            left = bool(
                info[
                    "left_contact"
                ]
            )

            right = bool(
                info[
                    "right_contact"
                ]
            )


            if (
                previous_left
                is not None
                and left
                != previous_left
            ):

                left_switches += 1


            if (
                previous_right
                is not None
                and right
                != previous_right
            ):

                right_switches += 1


            previous_left = left
            previous_right = right

            final = info


            if terminated or truncated:
                break


        return {
            "start":
                start_frame,

            "steps":
                steps,

            "seconds":
                steps / 50.0,

            "completed":
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
                        qerr_sum
                        / max(
                            steps,
                            1,
                        )
                    )
                ),

            "contact":
                float(
                    contact_sum
                    / max(
                        steps,
                        1,
                    )
                ),

            "lsw":
                left_switches,

            "rsw":
                right_switches,

            "amax":
                action_max,
        }


    finally:

        env.close()


models = []


for path in sorted(
    CHECKPOINT_DIR.glob(
        "*.zip"
    )
):

    models.append(
        path
    )


if FINAL_MODEL.exists():

    models.append(
        FINAL_MODEL
    )


if not models:

    raise SystemExit(
        "No tracking checkpoints found."
    )


print("=" * 150)
print("CLOSED-LOOP TRACKING EVALUATION")
print("=" * 150)


summaries = []


for path in models:

    policy = PPO.load(
        str(path),
        device="cpu",
    )


    rows = [
        rollout(
            policy,
            start,
        )
        for start in START_FRAMES
    ]


    completed = sum(
        int(
            r["completed"]
        )
        for r in rows
    )


    mean_time = float(
        np.mean(
            [
                r["seconds"]
                for r in rows
            ]
        )
    )


    worst_y = max(
        r["max_y"]
        for r in rows
    )


    worst_yaw = max(
        r["max_yaw"]
        for r in rows
    )


    mean_qerr = float(
        np.mean(
            [
                r["qerr"]
                for r in rows
            ]
        )
    )


    mean_contact = float(
        np.mean(
            [
                r["contact"]
                for r in rows
            ]
        )
    )


    print()
    print(
        path.name
    )


    print(
        f"{'START':>5s} "
        f"{'STEP':>5s} "
        f"{'SEC':>6s} "
        f"{'DONE':>5s} "
        f"{'X':>8s} "
        f"{'MAXY':>7s} "
        f"{'YAW':>7s} "
        f"{'MINUP':>7s} "
        f"{'QERR':>7s} "
        f"{'CONTACT':>8s} "
        f"{'L/R':>7s} "
        f"{'AMAX':>6s}"
    )


    for r in rows:

        print(
            f"{r['start']:5d} "
            f"{r['steps']:5d} "
            f"{r['seconds']:6.2f} "
            f"{str(r['completed']):>5s} "
            f"{r['x']:+8.3f} "
            f"{r['max_y']:7.3f} "
            f"{r['max_yaw']:7.1f} "
            f"{r['min_up']:7.3f} "
            f"{r['qerr']:7.3f} "
            f"{r['contact']:8.3f} "
            f"{r['lsw']}/{r['rsw']:>5d} "
            f"{r['amax']:6.3f}"
        )


    summaries.append(
        {
            "path":
                path,

            "completed":
                completed,

            "mean_time":
                mean_time,

            "worst_y":
                worst_y,

            "worst_yaw":
                worst_yaw,

            "mean_qerr":
                mean_qerr,

            "mean_contact":
                mean_contact,
        }
    )


best = max(
    summaries,

    key=lambda r: (
        r["completed"],
        -r["mean_qerr"],
        r["mean_contact"],
        r["mean_time"],
    ),
)


print()
print("=" * 150)
print("BEST TRACKER")
print("=" * 150)

print(
    "model       =",
    best["path"],
)

print(
    "completed   =",
    f"{best['completed']}/{len(START_FRAMES)}",
)

print(
    "mean time   =",
    f"{best['mean_time']:.2f}s",
)

print(
    "worst maxY  =",
    f"{best['worst_y']:.3f}",
)

print(
    "worst yaw   =",
    f"{best['worst_yaw']:.1f}",
)

print(
    "mean q error=",
    f"{best['mean_qerr']:.3f} rad",
)

print(
    "contact acc =",
    f"{best['mean_contact']:.3f}",
)


print()
print(
    "Stage-1 success target:"
)

print(
    "  >= 4/5 reference starts completed"
)

print(
    "  mean joint tracking error < 0.20 rad"
)

print(
    "  no catastrophic lateral/yaw exploit"
)
