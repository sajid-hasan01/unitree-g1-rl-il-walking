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


# ================================================================
# TEST CONFIGURATION
# ================================================================

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


def make_env():

    return G1BCLineResidualEnvV5Delta(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        # Diagnostic authority only.
        delta_action_scale=DELTA_SCALE,
    )


probe = make_env()

JOINT_NAMES = list(
    probe.joint_names
)

ACTION_DIM = int(
    probe.action_space.shape[0]
)

probe.close()


print("=" * 110)
print("V2 LATE-FAILURE PULSE SENSITIVITY MAP")
print("=" * 110)

print(
    f"Controlled joints : {ACTION_DIM}"
)

print(
    f"Delta scale       : {DELTA_SCALE:.2f}"
)

print(
    f"Windows           : {WINDOWS}"
)

print(
    f"Amplitudes        : {AMPLITUDES}"
)

print()

for i, name in enumerate(JOINT_NAMES):

    print(
        f"{i:02d}  {name}"
    )


# ================================================================
# ONE PHYSICAL ROLLOUT
# ================================================================

def rollout(
    joint_index=None,
    amplitude=0.0,
    window=None,
):

    env = make_env()

    try:

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


            # ----------------------------------------------------
            # Apply exactly one joint pulse in exactly one window.
            # ----------------------------------------------------

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

            "y_final":
                float(
                    final["y"]
                ),

            "yaw_final":
                float(
                    final["yaw_deg"]
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

    finally:

        env.close()


# ================================================================
# BASELINE
# ================================================================

print()
print("=" * 110)
print("BASELINE")
print("=" * 110)

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
    f"CAMAX={baseline['combined_max']:.3f}"
)


if baseline["steps"] != 119:

    print(
        "WARNING: baseline differs from expected "
        "119-step reference."
    )


# ================================================================
# FULL PULSE MAP
# ================================================================

results = []

total = (
    ACTION_DIM
    * len(WINDOWS)
    * len(AMPLITUDES)
)

count = 0


print()
print("=" * 110)
print(
    f"RUNNING {total} READ-ONLY PULSE ROLLOUTS"
)
print("=" * 110)


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


            # Print immediately only if something
            # actually improves survival.
            if r["steps"] > baseline["steps"]:

                print(
                    f"[{count:03d}/{total}] "
                    f"{joint_name:25s} "
                    f"W={window[0]:03d}-{window[1]:03d} "
                    f"A={amplitude:+.2f} "
                    f"STEP={r['steps']:3d} "
                    f"X={r['x']:+.3f} "
                    f"maxY={r['max_y']:.3f} "
                    f"maxYaw={r['max_yaw']:.1f} "
                    f"L/R={r['lsw']}/{r['rsw']} "
                    f"CA={r['combined_max']:.3f} "
                    f"OK={controlled}",
                    flush=True,
                )


# ================================================================
# SAVE ALL RESULTS
# ================================================================

csv_path = (
    ROOT
    / "results"
    / "v2_late_pulse_sensitivity.csv"
)


with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ================================================================
# RANK CONTROLLED RESULTS
# ================================================================

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
        -abs(r["x"]),
    )
)


print()
print("=" * 145)
print("TOP CONTROLLED PULSE RESULTS")
print("=" * 145)

print(
    f"{'JOINT':26s} "
    f"{'WIN':>9s} "
    f"{'AMP':>6s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'LCLR':>7s} "
    f"{'RCLR':>7s} "
    f"{'CAMAX':>6s}"
)

print("-" * 145)


for r in controlled_results[:30]:

    window_text = (
        f"{r['window_start']}-"
        f"{r['window_end']}"
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
        f"{r['min_up']:7.3f} "
        f"{str(r['up90']):>5s} "
        f"{str(r['up80']):>5s} "
        f"{r['lsw']:4d} "
        f"{r['rsw']:4d} "
        f"{r['left_clear']:7.1f} "
        f"{r['right_clear']:7.1f} "
        f"{r['combined_max']:6.3f}"
    )


# ================================================================
# ALSO SHOW BEST SURVIVAL EVEN IF UNCONTROLLED
# ================================================================

all_ranked = sorted(
    results,
    key=lambda r: (
        -r["steps"],
        r["max_yaw"],
        r["max_y"],
    ),
)


print()
print("=" * 145)
print("TOP RAW SURVIVAL RESULTS")
print("=" * 145)

print(
    f"{'JOINT':26s} "
    f"{'WIN':>9s} "
    f"{'AMP':>6s} "
    f"{'STEP':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'L/R':>7s} "
    f"{'CAMAX':>6s} "
    f"{'OK':>5s}"
)

print("-" * 110)


for r in all_ranked[:20]:

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
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{contacts:>7s} "
        f"{r['combined_max']:6.3f} "
        f"{str(r['controlled']):>5s}"
    )


# ================================================================
# BEST RESULT
# ================================================================

print()
print("=" * 145)
print("DIAGNOSTIC CONCLUSION")
print("=" * 145)


improved_controlled = [
    r
    for r in controlled_results
    if r["steps"] > baseline["steps"]
]


if improved_controlled:

    best = improved_controlled[0]

    print(
        "A controlled late rescue direction EXISTS."
    )

    print()

    print(
        f"joint   = {best['joint_name']}"
    )

    print(
        f"window  = "
        f"{best['window_start']}-"
        f"{best['window_end']}"
    )

    print(
        f"amp     = {best['amplitude']:+.2f}"
    )

    print(
        f"steps   = {best['steps']}"
    )

    print(
        f"gain    = "
        f"{best['steps'] - baseline['steps']:+d}"
    )

    print(
        f"X       = {best['x']:+.3f}"
    )

    print(
        f"maxY    = {best['max_y']:.3f}"
    )

    print(
        f"maxYaw  = {best['max_yaw']:.1f}"
    )

    print(
        f"L/R     = "
        f"{best['lsw']}/{best['rsw']}"
    )

    print()

    print(
        "NEXT: refine this exact joint/window/sign "
        "and test paired corrections."
    )

else:

    print(
        "No single-joint late pulse improved "
        "controlled survival beyond baseline."
    )

    print()

    print(
        "That would reject the hypothesis that "
        "a simple local residual-position correction "
        "can rescue V2 near failure."
    )

    print()

    print(
        "NEXT would be contact/nominal-gait modification "
        "or coordinated multi-joint control, not more PPO."
    )


print()
print(
    "CSV:",
    csv_path
)

print()
print("NO TRAINING WAS PERFORMED.")
print("=" * 145)
