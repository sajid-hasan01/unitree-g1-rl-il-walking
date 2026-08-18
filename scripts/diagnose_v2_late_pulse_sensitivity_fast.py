import sys
import csv
import faulthandler
from pathlib import Path

import numpy as np

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.g1_bc_line_residual_env_v5_delta import (
    G1BCLineResidualEnvV5Delta,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DELTA_SCALE = 0.25

WINDOWS = [
    (95, 99),
    (100, 104),
    (105, 109),
    (110, 114),
    (115, 119),
]

AMPLITUDES = [
    -1.00,
    -0.50,
    -0.25,
    +0.25,
    +0.50,
    +1.00,
]


# =====================================================================
# CREATE ONE ENVIRONMENT ONLY
#
# IMPORTANT:
# Frozen V2 PPO is loaded here ONCE.
# It will NOT be reloaded 450 times.
# =====================================================================

print("=" * 110, flush=True)
print("CREATING ONE REUSABLE V5 ENVIRONMENT", flush=True)
print("=" * 110, flush=True)


env = G1BCLineResidualEnvV5Delta(
    start_frame=25,
    stand_frames=45,
    transition_frames=90,
    target_smoothing=0.35,

    residual_scale=0.14,
    residual_ramp_frames=30,

    target_velocity=-0.18,
    max_episode_steps=600,

    delta_action_scale=DELTA_SCALE,
)


JOINT_NAMES = list(
    env.joint_names
)

ACTION_DIM = int(
    env.action_space.shape[0]
)


print("Environment loaded.", flush=True)
print("Frozen V2 loaded ONCE.", flush=True)

print(
    f"Action dim   : {ACTION_DIM}",
    flush=True,
)

print(
    f"Delta scale  : {DELTA_SCALE}",
    flush=True,
)

print()


for i, name in enumerate(JOINT_NAMES):

    print(
        f"{i:02d}  {name}",
        flush=True,
    )


# =====================================================================
# REUSED-ENV ROLLOUT
# =====================================================================

def rollout(
    joint_index=None,
    amplitude=0.0,
    window=None,
):

    obs, _ = env.reset(
        seed=1234
    )

    step = 0

    max_y = 0.0
    max_yaw = 0.0

    min_up = 1.0
    max_wy = 0.0

    up95 = None
    up90 = None
    up80 = None

    lsw = 0
    rsw = 0

    prev_l = None
    prev_r = None

    max_combined = 0.0
    max_scaled_delta = 0.0

    max_left_clear = float("-inf")
    max_right_clear = float("-inf")

    final = {}


    while True:

        upcoming_step = (
            step + 1
        )

        delta = np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )


        if (
            joint_index is not None
            and window is not None
        ):

            start, end = window

            if (
                start
                <= upcoming_step
                <= end
            ):

                delta[
                    joint_index
                ] = amplitude


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            delta
        )

        step += 1


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

        max_wy = max(
            max_wy,
            abs(wy),
        )


        if (
            up95 is None
            and up <= 0.95
        ):
            up95 = step


        if (
            up90 is None
            and up <= 0.90
        ):
            up90 = step


        if (
            up80 is None
            and up <= 0.80
        ):
            up80 = step


        left = bool(
            info["left_contact"]
        )

        right = bool(
            info["right_contact"]
        )


        if (
            prev_l is not None
            and left != prev_l
        ):
            lsw += 1


        if (
            prev_r is not None
            and right != prev_r
        ):
            rsw += 1


        prev_l = left
        prev_r = right


        if (
            not left
            and up >= 0.80
        ):

            max_left_clear = max(
                max_left_clear,

                1000.0
                * float(
                    info[
                        "left_true_clearance"
                    ]
                ),
            )


        if (
            not right
            and up >= 0.80
        ):

            max_right_clear = max(
                max_right_clear,

                1000.0
                * float(
                    info[
                        "right_true_clearance"
                    ]
                ),
            )


        max_combined = max(
            max_combined,

            float(
                info[
                    "v5_combined_action_max"
                ]
            ),
        )


        max_scaled_delta = max(
            max_scaled_delta,

            float(
                info[
                    "v5_scaled_delta_max"
                ]
            ),
        )


        final = info


        if terminated or truncated:
            break


    if not np.isfinite(
        max_left_clear
    ):

        max_left_clear = float(
            "nan"
        )


    if not np.isfinite(
        max_right_clear
    ):

        max_right_clear = float(
            "nan"
        )


    return {
        "steps":
            step,

        "post":
            max(
                0,
                step - 135,
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

        "max_wy":
            max_wy,

        "up95":
            up95,

        "up90":
            up90,

        "up80":
            up80,

        "lsw":
            lsw,

        "rsw":
            rsw,

        "left_clear":
            max_left_clear,

        "right_clear":
            max_right_clear,

        "scaled_delta_max":
            max_scaled_delta,

        "combined_max":
            max_combined,
    }


# =====================================================================
# BASELINE
# =====================================================================

print()
print("=" * 110, flush=True)
print("BASELINE CHECK", flush=True)
print("=" * 110, flush=True)


baseline = rollout()


print(
    f"steps={baseline['steps']} "
    f"X={baseline['x']:+.3f} "
    f"maxY={baseline['max_y']:.3f} "
    f"maxYaw={baseline['max_yaw']:.1f} "
    f"UP95/90/80="
    f"{baseline['up95']}/"
    f"{baseline['up90']}/"
    f"{baseline['up80']} "
    f"L/R={baseline['lsw']}/{baseline['rsw']} "
    f"CAMAX={baseline['combined_max']:.3f}",
    flush=True,
)


if baseline["steps"] != 119:

    raise SystemExit(
        "ABORT: baseline no longer equals "
        "the trusted 119-step reference."
    )


print()
print("BASELINE: PASS", flush=True)


# =====================================================================
# CSV
#
# Results are written immediately.
# If interrupted later, completed tests remain saved.
# =====================================================================

csv_path = (
    ROOT
    / "results"
    / "v2_late_pulse_sensitivity_fast.csv"
)


fieldnames = [
    "joint_index",
    "joint_name",
    "window_start",
    "window_end",
    "amplitude",
    "steps",
    "post",
    "x",
    "max_y",
    "max_yaw",
    "min_up",
    "max_wy",
    "up95",
    "up90",
    "up80",
    "lsw",
    "rsw",
    "left_clear",
    "right_clear",
    "scaled_delta_max",
    "combined_max",
    "controlled",
]


results = []


total = (
    ACTION_DIM
    * len(WINDOWS)
    * len(AMPLITUDES)
)


print()
print("=" * 110, flush=True)
print(
    f"STARTING {total} PULSE TESTS",
    flush=True,
)
print(
    "Progress will print every 10 tests.",
    flush=True,
)
print("=" * 110, flush=True)


best_controlled_steps = (
    baseline["steps"]
)

best_raw_steps = (
    baseline["steps"]
)

best_controlled = None
best_raw = None


with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    csv_file.flush()


    count = 0


    for joint_index, joint_name in enumerate(
        JOINT_NAMES
    ):

        for window in WINDOWS:

            for amplitude in AMPLITUDES:

                count += 1


                r = rollout(
                    joint_index=joint_index,
                    amplitude=amplitude,
                    window=window,
                )


                controlled = (
                    r["x"] <= -0.55
                    and r["max_y"] < 0.15
                    and r["max_yaw"] < 40.0
                    and r["lsw"] >= 2
                    and r["rsw"] >= 2
                    and r["combined_max"] < 0.98
                )


                row = {
                    "joint_index":
                        joint_index,

                    "joint_name":
                        joint_name,

                    "window_start":
                        window[0],

                    "window_end":
                        window[1],

                    "amplitude":
                        amplitude,

                    **r,

                    "controlled":
                        controlled,
                }


                results.append(
                    row
                )


                writer.writerow(
                    row
                )

                csv_file.flush()


                # -------------------------------------------------
                # Track best raw result.
                # -------------------------------------------------

                if r["steps"] > best_raw_steps:

                    best_raw_steps = (
                        r["steps"]
                    )

                    best_raw = row


                    print()
                    print(
                        "*** NEW RAW SURVIVAL BEST ***",
                        flush=True,
                    )

                    print(
                        f"{joint_name} "
                        f"W={window[0]}-{window[1]} "
                        f"A={amplitude:+.2f} "
                        f"steps={r['steps']} "
                        f"X={r['x']:+.3f} "
                        f"maxY={r['max_y']:.3f} "
                        f"maxYaw={r['max_yaw']:.1f} "
                        f"OK={controlled}",
                        flush=True,
                    )


                # -------------------------------------------------
                # Track best valid controlled result.
                # -------------------------------------------------

                if (
                    controlled
                    and r["steps"]
                    > best_controlled_steps
                ):

                    best_controlled_steps = (
                        r["steps"]
                    )

                    best_controlled = row


                    print()
                    print(
                        "**************************************",
                        flush=True,
                    )

                    print(
                        "*** NEW CONTROLLED SURVIVAL BEST ***",
                        flush=True,
                    )

                    print(
                        "**************************************",
                        flush=True,
                    )

                    print(
                        f"joint={joint_name}",
                        flush=True,
                    )

                    print(
                        f"window="
                        f"{window[0]}-{window[1]}",
                        flush=True,
                    )

                    print(
                        f"amp={amplitude:+.2f}",
                        flush=True,
                    )

                    print(
                        f"steps={r['steps']}",
                        flush=True,
                    )

                    print(
                        f"X={r['x']:+.3f}",
                        flush=True,
                    )

                    print(
                        f"maxY={r['max_y']:.3f}",
                        flush=True,
                    )

                    print(
                        f"maxYaw={r['max_yaw']:.1f}",
                        flush=True,
                    )

                    print(
                        f"L/R="
                        f"{r['lsw']}/{r['rsw']}",
                        flush=True,
                    )


                # -------------------------------------------------
                # ALWAYS show periodic progress.
                # -------------------------------------------------

                if (
                    count == 1
                    or count % 10 == 0
                    or count == total
                ):

                    print(
                        f"[{count:03d}/{total}] "
                        f"joint={joint_index:02d} "
                        f"{joint_name:25s} "
                        f"W={window[0]}-{window[1]} "
                        f"A={amplitude:+.2f} "
                        f"last={r['steps']:3d} "
                        f"bestRaw={best_raw_steps:3d} "
                        f"bestControlled="
                        f"{best_controlled_steps:3d}",
                        flush=True,
                    )


# =====================================================================
# RANK RESULTS
# =====================================================================

controlled_results = [
    r
    for r in results
    if r["controlled"]
]


controlled_results.sort(
    key=lambda r: (
        -r["steps"],
        r["max_yaw"],
        r["max_y"],
    )
)


all_ranked = sorted(
    results,
    key=lambda r: (
        -r["steps"],
        r["max_yaw"],
        r["max_y"],
    )
)


print()
print("=" * 145, flush=True)
print(
    "TOP 20 CONTROLLED RESULTS",
    flush=True,
)
print("=" * 145, flush=True)

print(
    f"{'JOINT':26s} "
    f"{'WINDOW':>9s} "
    f"{'AMP':>6s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'L/R':>7s} "
    f"{'CAMAX':>6s}",
    flush=True,
)

print("-" * 145, flush=True)


for r in controlled_results[:20]:

    window_text = (
        f"{r['window_start']}-"
        f"{r['window_end']}"
    )

    contacts = (
        f"{r['lsw']}/"
        f"{r['rsw']}"
    )

    print(
        f"{r['joint_name']:26s} "
        f"{window_text:>9s} "
        f"{r['amplitude']:+6.2f} "
        f"{r['steps']:5d} "
        f"{r['post']:5d} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{str(r['up90']):>5s} "
        f"{str(r['up80']):>5s} "
        f"{contacts:>7s} "
        f"{r['combined_max']:6.3f}",
        flush=True,
    )


print()
print("=" * 145, flush=True)
print(
    "TOP 15 RAW SURVIVAL RESULTS",
    flush=True,
)
print("=" * 145, flush=True)


for r in all_ranked[:15]:

    print(
        f"{r['joint_name']:26s} "
        f"W={r['window_start']}-{r['window_end']} "
        f"A={r['amplitude']:+.2f} "
        f"STEP={r['steps']:3d} "
        f"X={r['x']:+.3f} "
        f"Y={r['max_y']:.3f} "
        f"YAW={r['max_yaw']:.1f} "
        f"L/R={r['lsw']}/{r['rsw']} "
        f"OK={r['controlled']}",
        flush=True,
    )


print()
print("=" * 145, flush=True)
print(
    "FINAL DIAGNOSTIC CONCLUSION",
    flush=True,
)
print("=" * 145, flush=True)


if best_controlled is not None:

    r = best_controlled

    print(
        "CONTROLLED LATE RESCUE FOUND.",
        flush=True,
    )

    print(
        f"joint  = {r['joint_name']}",
        flush=True,
    )

    print(
        f"window = "
        f"{r['window_start']}-"
        f"{r['window_end']}",
        flush=True,
    )

    print(
        f"amp    = {r['amplitude']:+.2f}",
        flush=True,
    )

    print(
        f"steps  = {r['steps']}",
        flush=True,
    )

    print(
        f"gain   = "
        f"{r['steps'] - baseline['steps']:+d}",
        flush=True,
    )

    print(
        f"X      = {r['x']:+.3f}",
        flush=True,
    )

    print(
        f"maxY   = {r['max_y']:.3f}",
        flush=True,
    )

    print(
        f"maxYaw = {r['max_yaw']:.1f}",
        flush=True,
    )


else:

    print(
        "No single-joint pulse beat the "
        "119-step baseline while remaining controlled.",
        flush=True,
    )


print()
print(
    "CSV:",
    csv_path,
    flush=True,
)

print(
    "NO TRAINING WAS PERFORMED.",
    flush=True,
)


env.close()
