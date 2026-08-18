import sys
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


DELTA_SCALE = 0.25


# ================================================================
# ONE REUSED ENV
# ================================================================

print("=" * 105)
print("V2 COORDINATED LATE-RESCUE DIAGNOSTIC")
print("READ ONLY - NO TRAINING")
print("=" * 105)


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


names = list(
    env.joint_names
)


def j(name):
    return names.index(name)


LHP = j("left_hip_pitch_joint")
LK  = j("left_knee_joint")

RHP = j("right_hip_pitch_joint")
RK  = j("right_knee_joint")

LAP = j("left_ankle_pitch_joint")
RAP = j("right_ankle_pitch_joint")

WP = j("waist_pitch_joint")


print("Joint indices:")

for idx in [
    LHP,
    LK,
    RHP,
    RK,
    LAP,
    RAP,
    WP,
]:

    print(
        f"  {idx:02d} {names[idx]}"
    )


# ================================================================
# Pulse format:
#
# (start_step, end_step, joint_index, amplitude)
#
# Multiple pulses may overlap.
# ================================================================

CASES = [
    (
        "BASELINE",
        [],
    ),

    # ------------------------------------------------------------
    # Direct support-leg pair:
    # strongest evidence from single-joint map.
    # ------------------------------------------------------------

    (
        "LPAIR_050",
        [
            (100, 104, LHP, -0.50),
            (100, 104, LK,  -0.50),
        ],
    ),

    (
        "LPAIR_H100_K050",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -0.50),
        ],
    ),

    (
        "LPAIR_H050_K100",
        [
            (100, 104, LHP, -0.50),
            (100, 104, LK,  -1.00),
        ],
    ),

    (
        "LPAIR_100",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),
        ],
    ),


    # ------------------------------------------------------------
    # Right swing hip preconditioning + left support pair.
    # ------------------------------------------------------------

    (
        "RHP050_LPAIR050",
        [
            (95, 99, RHP, -0.50),

            (100, 104, LHP, -0.50),
            (100, 104, LK,  -0.50),
        ],
    ),

    (
        "RHP100_LPAIR050",
        [
            (95, 99, RHP, -1.00),

            (100, 104, LHP, -0.50),
            (100, 104, LK,  -0.50),
        ],
    ),

    (
        "RHP050_LPAIR100",
        [
            (95, 99, RHP, -0.50),

            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),
        ],
    ),

    (
        "RHP100_LPAIR100",
        [
            (95, 99, RHP, -1.00),

            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),
        ],
    ),


    # ------------------------------------------------------------
    # Separate which part matters.
    # ------------------------------------------------------------

    (
        "RHP100_LHP100",
        [
            (95, 99, RHP, -1.00),
            (100, 104, LHP, -1.00),
        ],
    ),

    (
        "RHP100_LK100",
        [
            (95, 99, RHP, -1.00),
            (100, 104, LK, -1.00),
        ],
    ),


    # ------------------------------------------------------------
    # Longer coordinated support pulse.
    # ------------------------------------------------------------

    (
        "LPAIR_050_LONG",
        [
            (95, 104, LHP, -0.50),
            (95, 104, LK,  -0.50),
        ],
    ),

    (
        "LPAIR_100_LONG",
        [
            (95, 104, LHP, -1.00),
            (95, 104, LK,  -1.00),
        ],
    ),


    # ------------------------------------------------------------
    # Add next-phase right knee correction.
    # Single pulse map showed RK +1 at 105-109 remained controlled.
    # ------------------------------------------------------------

    (
        "LPAIR100_RK050",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),

            (105, 109, RK, +0.50),
        ],
    ),

    (
        "LPAIR100_RK100",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),

            (105, 109, RK, +1.00),
        ],
    ),

    (
        "FULL_SEQ_050",
        [
            (95, 99, RHP, -0.50),

            (100, 104, LHP, -0.50),
            (100, 104, LK,  -0.50),

            (105, 109, RK, +0.50),
        ],
    ),

    (
        "FULL_SEQ_100",
        [
            (95, 99, RHP, -1.00),

            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),

            (105, 109, RK, +1.00),
        ],
    ),


    # ------------------------------------------------------------
    # Small torso counter-pitch tests.
    # Do NOT give waist large authority.
    # ------------------------------------------------------------

    (
        "LPAIR100_WP_POS",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),

            (100, 104, WP, +0.50),
        ],
    ),

    (
        "LPAIR100_WP_NEG",
        [
            (100, 104, LHP, -1.00),
            (100, 104, LK,  -1.00),

            (100, 104, WP, -0.50),
        ],
    ),
]


# ================================================================
# ROLLOUT
# ================================================================

def rollout(pulses):

    obs, _ = env.reset(
        seed=1234
    )

    step = 0

    max_y = 0.0
    max_yaw = 0.0
    min_up = 1.0
    max_abs_wy = 0.0

    up95 = None
    up90 = None
    up80 = None
    up70 = None

    lsw = 0
    rsw = 0

    prev_l = None
    prev_r = None

    max_combined = 0.0

    max_left_clear = float("-inf")
    max_right_clear = float("-inf")

    # Capture important states.
    snapshots = {}

    final = {}

    while True:

        next_step = (
            step + 1
        )

        delta = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )


        # ========================================================
        # APPLY ALL ACTIVE PULSES
        # ========================================================

        for (
            start,
            end,
            joint_idx,
            amp,
        ) in pulses:

            if (
                start
                <= next_step
                <= end
            ):

                delta[
                    joint_idx
                ] += amp


        delta = np.clip(
            delta,
            -1.0,
            1.0,
        )


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

        max_abs_wy = max(
            max_abs_wy,
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

        if (
            up70 is None
            and up <= 0.70
        ):
            up70 = step


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


        if step in [
            95,
            100,
            105,
            110,
            115,
            119,
            120,
            125,
            130,
            135,
        ]:

            snapshots[
                step
            ] = {
                "up": up,
                "wy": wy,
                "y": y,
                "yaw": yaw,
                "left": left,
                "right": right,
            }


        final = info


        if terminated or truncated:
            break


    if not np.isfinite(
        max_left_clear
    ):
        max_left_clear = float("nan")


    if not np.isfinite(
        max_right_clear
    ):
        max_right_clear = float("nan")


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
            max_abs_wy,

        "up95":
            up95,

        "up90":
            up90,

        "up80":
            up80,

        "up70":
            up70,

        "lsw":
            lsw,

        "rsw":
            rsw,

        "lclear":
            max_left_clear,

        "rclear":
            max_right_clear,

        "combined":
            max_combined,

        "snapshots":
            snapshots,
    }


# ================================================================
# RUN
# ================================================================

results = []


print()
print("=" * 165)

print(
    f"{'CASE':22s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'MAXWY':>7s} "
    f"{'UP95':>5s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'UP70':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'LCLR':>7s} "
    f"{'RCLR':>7s} "
    f"{'CAMAX':>6s} "
    f"{'OK':>5s}"
)

print("=" * 165)


def ev(x):

    return (
        "-"
        if x is None
        else str(x)
    )


for index, (
    label,
    pulses,
) in enumerate(
    CASES,
    start=1,
):

    r = rollout(
        pulses
    )


    controlled = (
        r["x"] <= -0.55
        and r["max_y"] < 0.15
        and r["max_yaw"] < 40.0
        and r["lsw"] >= 2
        and r["rsw"] >= 2
        and r["combined"] < 0.98
    )


    print(
        f"{label:22s} "
        f"{r['steps']:5d} "
        f"{r['post']:5d} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['max_wy']:7.2f} "
        f"{ev(r['up95']):>5s} "
        f"{ev(r['up90']):>5s} "
        f"{ev(r['up80']):>5s} "
        f"{ev(r['up70']):>5s} "
        f"{r['lsw']:4d} "
        f"{r['rsw']:4d} "
        f"{r['lclear']:7.1f} "
        f"{r['rclear']:7.1f} "
        f"{r['combined']:6.3f} "
        f"{str(controlled):>5s}",
        flush=True,
    )


    r["label"] = label
    r["controlled"] = controlled
    r["pulses"] = pulses

    results.append(
        r
    )


# ================================================================
# RANK
# ================================================================

baseline = results[0]

ranked = sorted(
    results[1:],
    key=lambda r: (
        -r["steps"],
        r["max_yaw"],
        r["max_y"],
    ),
)


print()
print("=" * 165)
print("TOP COMBINATIONS")
print("=" * 165)


for r in ranked[:10]:

    print(
        f"{r['label']:22s} "
        f"steps={r['steps']:3d} "
        f"gain={r['steps'] - baseline['steps']:+d} "
        f"X={r['x']:+.3f} "
        f"maxY={r['max_y']:.3f} "
        f"maxYaw={r['max_yaw']:.1f} "
        f"UP90={ev(r['up90'])} "
        f"UP80={ev(r['up80'])} "
        f"L/R={r['lsw']}/{r['rsw']} "
        f"OK={r['controlled']}",
        flush=True,
    )


# ================================================================
# BEST CONTROLLED
# ================================================================

controlled_improvements = [
    r
    for r in results[1:]
    if (
        r["controlled"]
        and r["steps"] > baseline["steps"]
    )
]


print()
print("=" * 165)
print("CONCLUSION")
print("=" * 165)


if controlled_improvements:

    best = max(
        controlled_improvements,
        key=lambda r: (
            r["steps"],
            -r["max_yaw"],
            -r["max_y"],
        ),
    )

    print(
        "CONTROLLED COORDINATED RESCUE FOUND."
    )

    print(
        f"case     = {best['label']}"
    )

    print(
        f"steps    = {best['steps']}"
    )

    print(
        f"gain     = "
        f"{best['steps'] - baseline['steps']:+d}"
    )

    print(
        f"X        = {best['x']:+.3f}"
    )

    print(
        f"maxY     = {best['max_y']:.3f}"
    )

    print(
        f"maxYaw   = {best['max_yaw']:.1f}"
    )

    print(
        f"UP95/90/80 = "
        f"{ev(best['up95'])}/"
        f"{ev(best['up90'])}/"
        f"{ev(best['up80'])}"
    )

    print(
        f"L/R      = "
        f"{best['lsw']}/{best['rsw']}"
    )

    print()
    print("Pulse definition:")

    for (
        start,
        end,
        idx,
        amp,
    ) in best["pulses"]:

        print(
            f"  {start}-{end}: "
            f"{names[idx]} "
            f"{amp:+.2f}"
        )

else:

    print(
        "No tested coordinated sagittal sequence "
        "beat 119 steps while remaining controlled."
    )

    print()

    print(
        "If this happens, stop residual-position "
        "rescue experiments."
    )

    print(
        "The next target should be the nominal BC/contact "
        "trajectory itself rather than another PPO version."
    )


print()
print("NO TRAINING WAS PERFORMED.")
print("=" * 165)


env.close()
