import argparse
import re
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


from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "g1_tracking_v3_pilot_checkpoints"
)


FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_tracking_v3_pilot_final.zip"
)


# Avoid an artificially easy near-end start.
START_FRAMES = [
    0,
    50,
    100,
    150,
    200,
]


def checkpoint_step(
    path,
):

    m = re.search(
        r"_(\d+)_steps",
        path.name,
    )

    if not m:
        return 10**12

    return int(
        m.group(1)
    )


def rollout(
    model,
    start_frame,
    label,
):

    env = G1ClosedLoopTrackingEnvV3(

        fixed_start_frame=
            start_frame,

        random_reference_start=
            False,

        reset_position_noise=
            0.0,

        reset_velocity_noise=
            0.0,

        target_smoothing=
            0.20,

        max_episode_steps=
            400,
    )


    obs, reset_info = env.reset(
        seed=1234
    )


    initial_x = float(
        env.data.qpos[0]
    )


    remaining_reference = max(
        env.num_frames
        - 1
        - start_frame,

        1,
    )


    steps = 0

    min_up = 1.0

    qerr2 = 0.0
    rooterr2 = 0.0
    velerr2 = 0.0
    orierr2 = 0.0
    footerr2 = 0.0

    contact_sum = 0.0

    max_action = 0.0

    final = {}


    while True:

        if model is None:

            action = np.zeros(
                15,
                dtype=np.float32,
            )

        else:

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
                15,
            )


        max_action = max(
            max_action,

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


        min_up = min(
            min_up,

            float(
                info[
                    "up_z"
                ]
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


        orierr2 += (
            float(
                info[
                    "orientation_error_deg"
                ]
            ) ** 2
        )


        footerr2 += (
            float(
                info[
                    "foot_position_error"
                ]
            ) ** 2
        )


        contact_sum += float(
            info[
                "contact_match"
            ]
        )


        final = info


        if (
            terminated
            or truncated
        ):
            break


    completed = bool(
        final.get(
            "completed",
            False,
        )
    )


    progress_ratio = min(
        1.0,

        steps
        / remaining_reference,
    )


    final_dx = (
        float(
            final["x"]
        )
        - initial_x
    )


    final_idx = int(
        final[
            "reference_index"
        ]
    )


    expected_dx = float(
        env.ref_root_pos[
            final_idx,
            0,
        ]
        - env.ref_root_pos[
            start_frame,
            0,
        ]
    )


    result = {

        "label":
            label,

        "start":
            start_frame,

        "steps":
            steps,

        "done":
            completed,

        "progress":
            progress_ratio,

        "dx":
            final_dx,

        "refdx":
            expected_dx,

        "xerr":
            final_dx
            - expected_dx,

        "minup":
            min_up,

        "qerr":
            float(
                np.sqrt(
                    qerr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "rooterr":
            float(
                np.sqrt(
                    rooterr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "velerr":
            float(
                np.sqrt(
                    velerr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "orierr":
            float(
                np.sqrt(
                    orierr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "footerr":
            float(
                np.sqrt(
                    footerr2
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

        "amax":
            max_action,
    }


    env.close()

    return result


def evaluate_model(
    model,
    label,
):

    return [
        rollout(
            model,
            start,
            label,
        )

        for start
        in START_FRAMES
    ]


def summarize(
    label,
    rows,
):

    completed = sum(
        int(
            r["done"]
        )
        for r in rows
    )


    return {

        "label":
            label,

        "completed":
            completed,

        "progress":
            float(
                np.mean(
                    [
                        r["progress"]
                        for r in rows
                    ]
                )
            ),

        "qerr":
            float(
                np.mean(
                    [
                        r["qerr"]
                        for r in rows
                    ]
                )
            ),

        "rooterr":
            float(
                np.mean(
                    [
                        r["rooterr"]
                        for r in rows
                    ]
                )
            ),

        "velerr":
            float(
                np.mean(
                    [
                        r["velerr"]
                        for r in rows
                    ]
                )
            ),

        "orierr":
            float(
                np.mean(
                    [
                        r["orierr"]
                        for r in rows
                    ]
                )
            ),

        "footerr":
            float(
                np.mean(
                    [
                        r["footerr"]
                        for r in rows
                    ]
                )
            ),

        "contact":
            float(
                np.mean(
                    [
                        r["contact"]
                        for r in rows
                    ]
                )
            ),
    }


def print_rows(
    label,
    rows,
):

    print()
    print("=" * 170)
    print(label)
    print("=" * 170)


    print(
        f"{'START':>5s} "
        f"{'STEP':>5s} "
        f"{'DONE':>5s} "
        f"{'PROG':>6s} "
        f"{'DX':>8s} "
        f"{'REFDX':>8s} "
        f"{'XERR':>8s} "
        f"{'MINUP':>7s} "
        f"{'QERR':>7s} "
        f"{'ROOTERR':>8s} "
        f"{'VELERR':>8s} "
        f"{'ORIERR':>8s} "
        f"{'FOOTERR':>8s} "
        f"{'CONTACT':>8s} "
        f"{'AMAX':>6s}"
    )


    print("-" * 170)


    for r in rows:

        print(
            f"{r['start']:5d} "
            f"{r['steps']:5d} "
            f"{str(r['done']):>5s} "
            f"{r['progress']:6.2f} "
            f"{r['dx']:+8.3f} "
            f"{r['refdx']:+8.3f} "
            f"{r['xerr']:+8.3f} "
            f"{r['minup']:7.3f} "
            f"{r['qerr']:7.3f} "
            f"{r['rooterr']:8.3f} "
            f"{r['velerr']:8.3f} "
            f"{r['orierr']:8.2f} "
            f"{r['footerr']:8.3f} "
            f"{r['contact']:8.3f} "
            f"{r['amax']:6.3f}"
        )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--baseline-only",
    action="store_true",
)


args = parser.parse_args()


print("=" * 170)
print("G1 CLOSED-LOOP TRACKING V3 EVALUATION")
print("=" * 170)


# =====================================================================
# ZERO-ACTION BASELINE
# =====================================================================

baseline_rows = evaluate_model(
    None,
    "ZERO_ACTION_V4_REFERENCE",
)


print_rows(
    "ZERO-ACTION V4 REFERENCE BASELINE",
    baseline_rows,
)


all_summaries = [
    summarize(
        "ZERO_ACTION",
        baseline_rows,
    )
]


if not args.baseline_only:

    models = []


    for path in sorted(
        CHECKPOINT_DIR.glob(
            "*.zip"
        ),

        key=checkpoint_step,
    ):

        models.append(
            path
        )


    if FINAL_MODEL.exists():

        models.append(
            FINAL_MODEL
        )


    for path in models:

        print()
        print(
            "Loading:",
            path.name,
        )


        model = PPO.load(
            str(path),
            device="cpu",
        )


        rows = evaluate_model(
            model,
            path.name,
        )


        print_rows(
            path.name,
            rows,
        )


        all_summaries.append(
            summarize(
                path.name,
                rows,
            )
        )


print()
print("=" * 150)
print("SUMMARY")
print("=" * 150)


print(
    f"{'MODEL':45s} "
    f"{'DONE':>5s} "
    f"{'PROG':>6s} "
    f"{'QERR':>7s} "
    f"{'ROOT':>7s} "
    f"{'VEL':>7s} "
    f"{'ORI':>7s} "
    f"{'FOOT':>7s} "
    f"{'CONTACT':>8s}"
)


print("-" * 150)


for s in all_summaries:

    print(
        f"{s['label'][:45]:45s} "
        f"{s['completed']:5d} "
        f"{s['progress']:6.3f} "
        f"{s['qerr']:7.3f} "
        f"{s['rooterr']:7.3f} "
        f"{s['velerr']:7.3f} "
        f"{s['orierr']:7.2f} "
        f"{s['footerr']:7.3f} "
        f"{s['contact']:8.3f}"
    )


if len(
    all_summaries
) > 1:

    trained = (
        all_summaries[1:]
    )


    best = max(
        trained,

        key=lambda s: (
            s["completed"],
            s["progress"],
            s["contact"],
            -s["orierr"],
            -s["rooterr"],
            -s["footerr"],
        ),
    )


    base = (
        all_summaries[0]
    )


    print()
    print("=" * 150)
    print("BEST PILOT CHECKPOINT")
    print("=" * 150)

    print(
        "model:",
        best["label"],
    )

    print(
        "completion:",
        f"{best['completed']}/"
        f"{len(START_FRAMES)}",
    )

    print(
        "mean progress:",
        f"{best['progress']:.3f}",
    )

    print(
        "orientation error:",
        f"{best['orierr']:.2f} deg",
    )

    print(
        "root error:",
        f"{best['rooterr']:.3f} m",
    )

    print(
        "foot error:",
        f"{best['footerr']:.3f} m",
    )

    print(
        "contact agreement:",
        f"{best['contact']:.3f}",
    )


    print()
    print("=" * 150)
    print("CHANGE VS ZERO-ACTION BASELINE")
    print("=" * 150)

    print(
        "progress:",
        f"{base['progress']:.3f}",
        "->",
        f"{best['progress']:.3f}",
    )

    print(
        "orientation:",
        f"{base['orierr']:.2f}",
        "->",
        f"{best['orierr']:.2f}",
        "deg",
    )

    print(
        "root error:",
        f"{base['rooterr']:.3f}",
        "->",
        f"{best['rooterr']:.3f}",
        "m",
    )

    print(
        "foot error:",
        f"{base['footerr']:.3f}",
        "->",
        f"{best['footerr']:.3f}",
        "m",
    )

    print(
        "contact:",
        f"{base['contact']:.3f}",
        "->",
        f"{best['contact']:.3f}",
    )


print()
print("=" * 150)
