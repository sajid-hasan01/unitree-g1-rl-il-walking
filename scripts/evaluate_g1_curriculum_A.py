import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


STARTS = [
    0,
    20,
    40,
    60,
    80,
]


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_curriculum_A_checkpoints"
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_curriculum_A_final.zip"
)


def checkpoint_step(path):

    m = re.search(
        r"_(\d+)_steps",
        path.name,
    )

    if not m:
        return 10**9

    return int(
        m.group(1)
    )


def rollout(
    model,
    start,
):

    env = G1ClosedLoopTrackingEnvV3(

        fixed_start_frame=start,

        random_reference_start=False,

        reset_position_noise=0.0,

        reset_velocity_noise=0.0,

        target_smoothing=0.20,

        max_episode_steps=400,
    )


    obs, _ = env.reset(
        seed=1234
    )


    remaining = max(
        env.num_frames
        - 1
        - start,
        1,
    )


    steps = 0

    q2 = 0.0
    root2 = 0.0
    vel2 = 0.0
    ori2 = 0.0
    foot2 = 0.0

    contact_sum = 0.0

    min_up = 1.0

    max_action = 0.0

    final = {}


    while True:

        if model is None:

            action = np.zeros(
                15,
                dtype=np.float32,
            )

        else:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(15)


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

        min_up = min(
            min_up,
            float(
                info["up_z"]
            ),
        )


        q2 += (
            float(
                info["q_error_rms"]
            ) ** 2
        )


        root2 += (
            float(
                info[
                    "root_position_error"
                ]
            ) ** 2
        )


        vel2 += (
            float(
                info[
                    "root_velocity_error"
                ]
            ) ** 2
        )


        ori2 += (
            float(
                info[
                    "orientation_error_deg"
                ]
            ) ** 2
        )


        foot2 += (
            float(
                info[
                    "foot_position_error"
                ]
            ) ** 2
        )


        contact_sum += float(
            info["contact_match"]
        )


        final = info


        if terminated or truncated:
            break


    result = {
        "start":
            start,

        "steps":
            steps,

        "done":
            bool(
                final.get(
                    "completed",
                    False,
                )
            ),

        "progress":
            min(
                1.0,
                steps / remaining,
            ),

        "minup":
            min_up,

        "qerr":
            np.sqrt(
                q2 / steps
            ),

        "root":
            np.sqrt(
                root2 / steps
            ),

        "vel":
            np.sqrt(
                vel2 / steps
            ),

        "ori":
            np.sqrt(
                ori2 / steps
            ),

        "foot":
            np.sqrt(
                foot2 / steps
            ),

        "contact":
            contact_sum / steps,

        "amax":
            max_action,
    }


    env.close()

    return result


def evaluate(
    model,
):

    return [
        rollout(
            model,
            s,
        )
        for s in STARTS
    ]


def summary(
    label,
    rows,
):

    return {
        "label":
            label,

        "done":
            sum(
                int(r["done"])
                for r in rows
            ),

        "prog":
            np.mean(
                [
                    r["progress"]
                    for r in rows
                ]
            ),

        "qerr":
            np.mean(
                [
                    r["qerr"]
                    for r in rows
                ]
            ),

        "root":
            np.mean(
                [
                    r["root"]
                    for r in rows
                ]
            ),

        "vel":
            np.mean(
                [
                    r["vel"]
                    for r in rows
                ]
            ),

        "ori":
            np.mean(
                [
                    r["ori"]
                    for r in rows
                ]
            ),

        "foot":
            np.mean(
                [
                    r["foot"]
                    for r in rows
                ]
            ),

        "contact":
            np.mean(
                [
                    r["contact"]
                    for r in rows
                ]
            ),
    }


def print_detail(
    label,
    rows,
):

    print()
    print("=" * 140)
    print(label)
    print("=" * 140)

    print(
        f"{'START':>5s} "
        f"{'STEP':>5s} "
        f"{'DONE':>5s} "
        f"{'PROG':>6s} "
        f"{'MINUP':>7s} "
        f"{'QERR':>7s} "
        f"{'ROOT':>7s} "
        f"{'VEL':>7s} "
        f"{'ORI':>8s} "
        f"{'FOOT':>7s} "
        f"{'CONTACT':>8s} "
        f"{'AMAX':>6s}"
    )

    print("-" * 140)

    for r in rows:

        print(
            f"{r['start']:5d} "
            f"{r['steps']:5d} "
            f"{str(r['done']):>5s} "
            f"{r['progress']:6.3f} "
            f"{r['minup']:7.3f} "
            f"{r['qerr']:7.3f} "
            f"{r['root']:7.3f} "
            f"{r['vel']:7.3f} "
            f"{r['ori']:8.2f} "
            f"{r['foot']:7.3f} "
            f"{r['contact']:8.3f} "
            f"{r['amax']:6.3f}"
        )


models = [
    (
        "ZERO_ACTION",
        None,
    )
]


for path in sorted(
    CHECKPOINT_DIR.glob(
        "*.zip"
    ),
    key=checkpoint_step,
):

    models.append(
        (
            path.name,
            PPO.load(
                str(path),
                device="cpu",
            ),
        )
    )


if FINAL_MODEL.exists():

    models.append(
        (
            FINAL_MODEL.name,
            PPO.load(
                str(FINAL_MODEL),
                device="cpu",
            ),
        )
    )


summaries = []


for label, model in models:

    rows = evaluate(
        model
    )

    print_detail(
        label,
        rows,
    )

    summaries.append(
        summary(
            label,
            rows,
        )
    )


print()
print("=" * 125)
print("CURRICULUM A SUMMARY")
print("=" * 125)

print(
    f"{'MODEL':42s} "
    f"{'DONE':>5s} "
    f"{'PROG':>7s} "
    f"{'QERR':>7s} "
    f"{'ROOT':>7s} "
    f"{'VEL':>7s} "
    f"{'ORI':>8s} "
    f"{'FOOT':>7s} "
    f"{'CONTACT':>8s}"
)

print("-" * 125)


for s in summaries:

    print(
        f"{s['label'][:42]:42s} "
        f"{s['done']:5d} "
        f"{s['prog']:7.3f} "
        f"{s['qerr']:7.3f} "
        f"{s['root']:7.3f} "
        f"{s['vel']:7.3f} "
        f"{s['ori']:8.2f} "
        f"{s['foot']:7.3f} "
        f"{s['contact']:8.3f}"
    )


base = summaries[0]


if len(summaries) > 1:

    best = max(
        summaries[1:],

        key=lambda x: (
            x["done"],
            x["prog"],
            -x["ori"],
            -x["root"],
            x["contact"],
        ),
    )


    print()
    print("=" * 125)
    print("BEST CURRICULUM A")
    print("=" * 125)

    print(
        "model:",
        best["label"],
    )

    print(
        "completion:",
        f"{best['done']}/"
        f"{len(STARTS)}",
    )

    print(
        "progress:",
        f"{base['prog']:.3f}",
        "->",
        f"{best['prog']:.3f}",
    )

    print(
        "orientation:",
        f"{base['ori']:.2f}",
        "->",
        f"{best['ori']:.2f}",
    )

    print(
        "root:",
        f"{base['root']:.3f}",
        "->",
        f"{best['root']:.3f}",
    )

    print(
        "velocity:",
        f"{base['vel']:.3f}",
        "->",
        f"{best['vel']:.3f}",
    )

    print(
        "foot:",
        f"{base['foot']:.3f}",
        "->",
        f"{best['foot']:.3f}",
    )

    print(
        "contact:",
        f"{base['contact']:.3f}",
        "->",
        f"{best['contact']:.3f}",
    )


print("=" * 125)
