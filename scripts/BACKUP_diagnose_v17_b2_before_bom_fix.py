from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.evaluate_deterministic_left_lift import (
    add_args,
    make_env,
)


# ======================================================================
# USE EXACTLY THE SAME DEFAULT ARGUMENTS AS THE VALIDATED EVALUATOR
# ======================================================================

parser = argparse.ArgumentParser(add_help=False)
add_args(parser)
args = parser.parse_args([])

# Explicitly preserve the B1/B2 experiment inputs.
args.target_clearance = 0.026
args.swing_z_weight = 1.00
args.swing_ik_gain = 0.90
args.swing_ik_max_delta = 0.14
args.max_steps = 650


env = make_env(args)

obs, info = env.reset()

rows = []

first_lift_step = None
first_air_step = None
confirm_step = None
touchdown_step = None
shared_step = None

prev_confirm = False
prev_td = False
prev_shared = False


while True:

    obs, reward, terminated, truncated, info = env.step()

    step = int(info["episode_step"])

    raw_sw = float(
        info["raw_swing_env"]
    )

    sw = float(
        info["swing_env"]
    )

    guard = float(
        info["swing_guard_ratio"]
    )

    desired_clear = float(
        info["main_target_clearance"]
    )

    actual_clear = float(
        info["left_foot_clearance"]
    )

    z_error = (
        desired_clear
        - actual_clear
    )

    # Current B1 normal-swing authority.
    normal_blend = float(
        np.clip(
            0.40 + 0.60 * sw,
            0.0,
            1.0,
        )
    )

    lift = bool(
        info["lift_enabled"]
    )

    contact = bool(
        info["left_contact"]
    )

    swing_confirmed = bool(
        info.get(
            "swing_confirmed",
            False,
        )
    )

    touchdown = bool(
        info.get(
            "touchdown_latched",
            False,
        )
    )

    shared = bool(
        info.get(
            "recovery_load_ready",
            False,
        )
    )

    if lift and first_lift_step is None:
        first_lift_step = step

    if (
        not contact
        and float(
            info.get(
                "left_normal_force",
                0.0,
            )
        ) <= 1.0
        and first_air_step is None
    ):
        first_air_step = step

    if swing_confirmed and not prev_confirm:
        confirm_step = step

    if touchdown and not prev_td:
        touchdown_step = step

    if shared and not prev_shared:
        shared_step = step

    prev_confirm = swing_confirmed
    prev_td = touchdown
    prev_shared = shared


    if (
        lift
        or raw_sw > 0.0
        or swing_confirmed
        or touchdown
    ):

        rows.append(
            {
                "step": step,
                "phi": float(
                    info["phase"]
                ),
                "raw_sw": raw_sw,
                "sw": sw,
                "guard": guard,
                "blend": normal_blend,
                "desired_clear": desired_clear,
                "actual_clear": actual_clear,
                "z_error": z_error,
                "left_contact": int(contact),
                "left_force": float(
                    info.get(
                        "left_normal_force",
                        0.0,
                    )
                ),
                "left_load": float(
                    info.get(
                        "left_load_ratio",
                        0.0,
                    )
                ),
                "airOK": int(
                    swing_confirmed
                ),
                "td": int(
                    touchdown
                ),
                "loadOK": int(
                    shared
                ),
                "com_error": float(
                    info["com_error_y"]
                ),
                "up_z": float(
                    info["up_z"]
                ),
                "x": float(
                    info["x_position"]
                ),
                "xv": float(
                    info["x_velocity"]
                ),
            }
        )

    if terminated or truncated:
        break


# ======================================================================
# SAVE FULL TRACE
# ======================================================================

out = Path(
    "results/v17_b2_swing_tracking_trace.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with out.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# ======================================================================
# CRITICAL STATISTICS
# ======================================================================

max_desired_row = max(
    rows,
    key=lambda r: r[
        "desired_clear"
    ],
)

max_actual_row = max(
    rows,
    key=lambda r: r[
        "actual_clear"
    ],
)

active_rows = [
    r
    for r in rows
    if (
        r["sw"] > 0.05
        and not r["td"]
    )
]

max_error_row = max(
    active_rows,
    key=lambda r: r[
        "z_error"
    ],
)


def print_row(label, r):

    print(
        f"{label:<18s} "
        f"step={r['step']:04d} "
        f"phi={r['phi']:.3f} "
        f"raw={r['raw_sw']:.3f} "
        f"sw={r['sw']:.3f} "
        f"guard={r['guard']:.3f} "
        f"blend={r['blend']:.3f} "
        f"Zdes={1000*r['desired_clear']:6.2f}mm "
        f"Zact={1000*r['actual_clear']:6.2f}mm "
        f"Zerr={1000*r['z_error']:+6.2f}mm "
        f"C={r['left_contact']} "
        f"LF={r['left_force']:6.1f} "
        f"Lload={r['left_load']:.2f} "
        f"COMe={r['com_error']:+.4f} "
        f"xv={r['xv']:+.4f}"
    )


print()
print("=" * 148)
print("V17-B2 SWING TRACKING DIAGNOSTIC")
print("=" * 148)

print()
print("EVENT STEPS")
print("-" * 148)

print(
    "lift enabled     :",
    first_lift_step,
)

print(
    "first force-free :",
    first_air_step,
)

print(
    "swing confirmed  :",
    confirm_step,
)

print(
    "touchdown latched:",
    touchdown_step,
)

print(
    "shared support   :",
    shared_step,
)


print()
print("CRITICAL ROWS")
print("-" * 148)

print_row(
    "MAX DESIRED",
    max_desired_row,
)

print_row(
    "MAX ACTUAL",
    max_actual_row,
)

print_row(
    "MAX Z ERROR",
    max_error_row,
)


# ======================================================================
# PRINT SWING TRACE
#
# Print every 3rd step plus important transitions.
# ======================================================================

print()
print("SWING TRACE")
print("-" * 148)

interesting_steps = set()

for r in rows:

    s = int(
        r["step"]
    )

    if s % 3 == 0:
        interesting_steps.add(
            s
        )

for center in [
    first_air_step,
    confirm_step,
    touchdown_step,
]:

    if center is None:
        continue

    for d in range(
        -3,
        4,
    ):
        interesting_steps.add(
            center + d
        )


for r in rows:

    if int(
        r["step"]
    ) in interesting_steps:

        print_row(
            "",
            r,
        )


print()
print("=" * 148)
print("FINAL INTERPRETATION NUMBERS")
print("=" * 148)

max_desired = float(
    max_desired_row[
        "desired_clear"
    ]
)

max_actual = float(
    max_actual_row[
        "actual_clear"
    ]
)

print(
    f"Maximum requested clearance : "
    f"{1000*max_desired:.3f} mm"
)

print(
    f"Maximum achieved clearance  : "
    f"{1000*max_actual:.3f} mm"
)

print(
    f"Peak request-achievement gap: "
    f"{1000*(max_desired-max_actual):.3f} mm"
)

print(
    f"Strict threshold            : "
    f"{1000*args.strict_clearance:.3f} mm"
)

print(
    f"Remaining physical gap      : "
    f"{1000*(args.strict_clearance-max_actual):.3f} mm"
)

print()
print(
    "CSV:",
    out,
)

print()
print(
    "NO CONTROLLER PARAMETERS WERE MODIFIED."
)

env.close()
