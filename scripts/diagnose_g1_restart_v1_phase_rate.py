from pathlib import Path
import math
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
    quat_angle,
)


# ================================================================
# CONFIG
# ================================================================

STARTS = [
    0,
    15,
    30,
    45,
    57,
    75,
    85,
    105,
    114,
    135,
    140,
]


PHASE_RATES = [
    0.60,
    0.75,
    0.90,
    1.00,
    1.10,
    1.25,
    1.40,
]


REFERENCE_HORIZON = 120

MAX_EXTRA_CONTROL_STEPS = 25


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


print("=" * 180)
print("RESTART V1 REFERENCE PHASE-RATE DIAGNOSTIC")
print("ZERO RESIDUAL ACTION")
print("NO CEM")
print("NO PPO")
print("=" * 180)

print(
    "reference:",
    env.reference_path,
)

print(
    "frames:",
    env.num_frames,
)

print(
    "control Hz:",
    env.reference_fps,
)

print(
    "starts:",
    STARTS,
)

print(
    "phase rates:",
    PHASE_RATES,
)


# ================================================================
# CUSTOM PHASE STEP
#
# Unlike env.step(), this does NOT blindly increment
# reference_frame by exactly +1.
#
# Physics control frequency always remains exactly 50 Hz.
# Only reference-motion progression changes.
# ================================================================

def step_at_reference_frame(
    target_frame,
):

    target_frame = int(
        np.clip(
            target_frame,
            0,
            env.num_frames - 1,
        )
    )


    zero_action = np.zeros(
        29,
        dtype=np.float64,
    )


    target = env._action_target(
        zero_action,
        target_frame,
    )


    env.last_target = (
        target.copy()
    )


    env._apply_target(
        target
    )


    for _ in range(
        env.frame_skip
    ):

        mujoco.mj_step(
            env.model,
            env.data,
        )


    env._current_frame = (
        target_frame
    )


    (
        failed,
        reason,
    ) = env._physical_failure()


    return (
        failed,
        reason,
    )


# ================================================================
# ONE ROLLOUT
# ================================================================

def rollout(
    start,
    phase_rate,
):

    env.reset(
        options={
            "start_frame":
                int(start)
        }
    )


    start = int(
        start
    )


    segment_end = min(
        env.num_frames - 1,
        start
        + REFERENCE_HORIZON,
    )


    reference_span = (
        segment_end
        - start
    )


    if reference_span <= 0:

        raise RuntimeError(
            "Invalid reference span."
        )


    # Floating phase.
    phase = float(
        start
    )


    # At a slow rate we deliberately permit more wall-clock
    # control steps so it has a fair opportunity to traverse the
    # SAME number of reference frames.
    expected_steps = int(
        math.ceil(
            reference_span
            / phase_rate
        )
    )


    control_limit = (
        expected_steps
        + MAX_EXTRA_CONTROL_STEPS
    )


    wall_steps = 0

    reason = ""

    completed_segment = False


    min_up = env._up_z()


    ori_sum = 0.0

    q_sum = 0.0

    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0


    final_x_error = 0.0
    final_y_error = 0.0
    final_z_error = 0.0


    max_x_error = 0.0
    max_y_error = 0.0
    max_z_error = 0.0


    actual_start_x = float(
        env.data.qpos[
            0
        ]
    )


    last_target_frame = (
        start
    )


    previous_target_frame = (
        start
    )


    holds = 0
    skips = 0


    while (
        wall_steps
        < control_limit
    ):

        phase = min(
            float(
                segment_end
            ),
            phase
            + phase_rate,
        )


        # Monotonic discrete time warp.
        target_frame = int(
            math.floor(
                phase
                + 1e-9
            )
        )


        target_frame = max(
            target_frame,
            last_target_frame,
        )


        target_frame = min(
            target_frame,
            segment_end,
        )


        delta_frame = (
            target_frame
            - previous_target_frame
        )


        if delta_frame == 0:
            holds += 1

        elif delta_frame > 1:
            skips += (
                delta_frame
                - 1
            )


        previous_target_frame = (
            target_frame
        )

        last_target_frame = (
            target_frame
        )


        (
            failed,
            failure_reason,
        ) = step_at_reference_frame(
            target_frame
        )


        wall_steps += 1


        ref_q = env.ref_q[
            target_frame
        ]


        actual_q = env._joint_q()


        q_error = float(
            np.sqrt(
                np.mean(
                    (
                        actual_q
                        - ref_q
                    ) ** 2
                )
            )
        )


        orientation_error = math.degrees(
            quat_angle(
                env.ref_root_quat[
                    target_frame
                ],
                env.data.qpos[
                    3:7
                ],
            )
        )


        root_error = (
            env.data.qpos[
                0:3
            ]
            -
            env.ref_root_pos[
                target_frame
            ]
        )


        ex = float(
            root_error[0]
        )

        ey = float(
            root_error[1]
        )

        ez = float(
            root_error[2]
        )


        x_sum += abs(
            ex
        )

        y_sum += abs(
            ey
        )

        z_sum += abs(
            ez
        )


        final_x_error = ex
        final_y_error = ey
        final_z_error = ez


        max_x_error = max(
            max_x_error,
            abs(
                ex
            ),
        )

        max_y_error = max(
            max_y_error,
            abs(
                ey
            ),
        )

        max_z_error = max(
            max_z_error,
            abs(
                ez
            ),
        )


        ori_sum += (
            orientation_error
        )

        q_sum += (
            q_error
        )


        min_up = min(
            min_up,
            env._up_z(),
        )


        if failed:

            reason = (
                failure_reason
            )

            break


        if target_frame >= segment_end:

            completed_segment = True

            reason = (
                "segment_end"
            )

            break


    if not reason:

        reason = (
            "control_limit"
        )


    ref_advance = (
        last_target_frame
        - start
    )


    reference_progress = (
        ref_advance
        / reference_span
    )


    divisor = max(
        wall_steps,
        1,
    )


    actual_dx = float(
        env.data.qpos[
            0
        ]
        - actual_start_x
    )


    desired_dx = float(
        env.ref_root_pos[
            last_target_frame,
            0
        ]
        -
        env.ref_root_pos[
            start,
            0
        ]
    )


    # Fraction of desired forward translation physically obtained.
    if abs(
        desired_dx
    ) > 1e-6:

        forward_ratio = (
            actual_dx
            / desired_dx
        )

    else:

        forward_ratio = (
            float("nan")
        )


    return {

        "start":
            start,

        "rate":
            phase_rate,

        "wall_steps":
            wall_steps,

        "wall_time":
            wall_steps
            / env.reference_fps,

        "target_frame":
            last_target_frame,

        "ref_advance":
            ref_advance,

        "ref_span":
            reference_span,

        "progress":
            reference_progress,

        "completed":
            completed_segment,

        "reason":
            reason,

        "min_up":
            min_up,

        "orientation":
            ori_sum
            / divisor,

        "q_error":
            q_sum
            / divisor,

        "x":
            x_sum
            / divisor,

        "y":
            y_sum
            / divisor,

        "z":
            z_sum
            / divisor,

        "max_x":
            max_x_error,

        "max_y":
            max_y_error,

        "max_z":
            max_z_error,

        "final_x":
            final_x_error,

        "final_y":
            final_y_error,

        "final_z":
            final_z_error,

        "actual_dx":
            actual_dx,

        "desired_dx":
            desired_dx,

        "forward_ratio":
            forward_ratio,

        "holds":
            holds,

        "skips":
            skips,
    }


# ================================================================
# RUN ALL RATES / STARTS
# ================================================================

results = []


for phase_rate in PHASE_RATES:

    for start in STARTS:

        r = rollout(
            start,
            phase_rate,
        )

        results.append(
            r
        )


# ================================================================
# PER-START DETAIL
# ================================================================

print()
print("=" * 210)
print("PER-START PHASE-RATE RESULTS")
print("=" * 210)

print(
    f"{'RATE':>5s} "
    f"{'START':>5s} "
    f"{'REF':>7s} "
    f"{'PROG':>6s} "
    f"{'CTRL':>5s} "
    f"{'SEC':>6s} "
    f"{'ORI':>7s} "
    f"{'MINUP':>6s} "
    f"{'QERR':>6s} "
    f"{'X':>6s} "
    f"{'Y':>6s} "
    f"{'Z':>6s} "
    f"{'DX':>7s} "
    f"{'REFDX':>7s} "
    f"{'DX/R':>6s} "
    f"{'HOLD':>5s} "
    f"{'SKIP':>5s} "
    f"{'REASON':>14s}"
)

print("-" * 210)


for r in results:

    ratio = (
        r["forward_ratio"]
    )


    ratio_text = (
        f"{ratio:6.2f}"
        if np.isfinite(
            ratio
        )
        else "   nan"
    )


    print(
        f"{r['rate']:5.2f} "
        f"{r['start']:5d} "
        f"{r['ref_advance']:3d}/"
        f"{r['ref_span']:<3d} "
        f"{r['progress']:6.3f} "
        f"{r['wall_steps']:5d} "
        f"{r['wall_time']:6.2f} "
        f"{r['orientation']:7.2f} "
        f"{r['min_up']:6.3f} "
        f"{r['q_error']:6.3f} "
        f"{r['x']:6.3f} "
        f"{r['y']:6.3f} "
        f"{r['z']:6.3f} "
        f"{r['actual_dx']:+7.3f} "
        f"{r['desired_dx']:+7.3f} "
        f"{ratio_text} "
        f"{r['holds']:5d} "
        f"{r['skips']:5d} "
        f"{r['reason']:>14s}"
    )


# ================================================================
# AGGREGATE BY RATE
# ================================================================

def rows_for_rate(
    rate,
):

    return [
        r
        for r in results
        if abs(
            r["rate"]
            - rate
        ) < 1e-9
    ]


def aggregate(
    rate,
):

    rows = rows_for_rate(
        rate
    )


    return {

        "rate":
            rate,

        "progress":
            float(
                np.mean(
                    [
                        r["progress"]
                        for r in rows
                    ]
                )
            ),

        "min_progress":
            float(
                np.min(
                    [
                        r["progress"]
                        for r in rows
                    ]
                )
            ),

        "orientation":
            float(
                np.mean(
                    [
                        r["orientation"]
                        for r in rows
                    ]
                )
            ),

        "min_up":
            float(
                np.mean(
                    [
                        r["min_up"]
                        for r in rows
                    ]
                )
            ),

        "q":
            float(
                np.mean(
                    [
                        r["q_error"]
                        for r in rows
                    ]
                )
            ),

        "x":
            float(
                np.mean(
                    [
                        r["x"]
                        for r in rows
                    ]
                )
            ),

        "y":
            float(
                np.mean(
                    [
                        r["y"]
                        for r in rows
                    ]
                )
            ),

        "z":
            float(
                np.mean(
                    [
                        r["z"]
                        for r in rows
                    ]
                )
            ),

        "wall_time":
            float(
                np.mean(
                    [
                        r["wall_time"]
                        for r in rows
                    ]
                )
            ),

        "completed":
            int(
                sum(
                    int(
                        r["completed"]
                    )
                    for r in rows
                )
            ),
    }


summary = {
    rate:
        aggregate(
            rate
        )
    for rate in PHASE_RATES
}


print()
print("=" * 165)
print("PHASE-RATE AGGREGATE")
print("=" * 165)

print(
    f"{'RATE':>5s} "
    f"{'PROG':>7s} "
    f"{'MIN-P':>7s} "
    f"{'ORI':>8s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'XERR':>7s} "
    f"{'YERR':>7s} "
    f"{'ZERR':>7s} "
    f"{'TIME':>7s} "
    f"{'DONE':>5s}"
)

print("-" * 165)


for rate in PHASE_RATES:

    s = summary[
        rate
    ]


    print(
        f"{rate:5.2f} "
        f"{s['progress']:7.3f} "
        f"{s['min_progress']:7.3f} "
        f"{s['orientation']:8.2f} "
        f"{s['min_up']:7.3f} "
        f"{s['q']:7.3f} "
        f"{s['x']:7.3f} "
        f"{s['y']:7.3f} "
        f"{s['z']:7.3f} "
        f"{s['wall_time']:7.2f} "
        f"{s['completed']:5d}"
    )


# ================================================================
# GLOBAL BEST CONSTANT RATE
# ================================================================

baseline = summary[
    1.00
]


best_constant_rate = max(
    PHASE_RATES,

    key=lambda rate: (
        summary[
            rate
        ][
            "progress"
        ],

        summary[
            rate
        ][
            "min_progress"
        ],

        -summary[
            rate
        ][
            "orientation"
        ],

        -abs(
            rate
            - 1.0
        ),
    ),
)


best_constant = summary[
    best_constant_rate
]


constant_gain = (
    best_constant[
        "progress"
    ]
    -
    baseline[
        "progress"
    ]
)


# ================================================================
# PER-START ORACLE RATE
#
# This does NOT become a controller.
#
# It simply asks:
# "If each start could choose its own constant reference rate,
#  how much better could it do?"
#
# Large oracle gain + different best rates across starts
# => strong evidence for adaptive timing.
# ================================================================

oracle_rows = []


print()
print("=" * 175)
print("PER-START BEST PHASE RATE")
print("=" * 175)


for start in STARTS:

    candidates = [
        r
        for r in results
        if r[
            "start"
        ] == start
    ]


    best = max(
        candidates,

        key=lambda r: (
            r[
                "progress"
            ],

            -r[
                "orientation"
            ],

            -r[
                "q_error"
            ],

            -abs(
                r[
                    "rate"
                ]
                - 1.0
            ),
        ),
    )


    fixed = next(
        r
        for r in candidates
        if abs(
            r["rate"]
            - 1.0
        ) < 1e-9
    )


    oracle_rows.append(
        best
    )


    print(
        f"start={start:03d} "
        f"fixed={fixed['progress']:.3f} "
        f"best={best['progress']:.3f} "
        f"gain="
        f"{best['progress']-fixed['progress']:+.3f} "
        f"bestRate={best['rate']:.2f}x "
        f"ori={fixed['orientation']:.2f}"
        f"->{best['orientation']:.2f}"
    )


oracle_progress = float(
    np.mean(
        [
            r["progress"]
            for r in oracle_rows
        ]
    )
)


oracle_gain = (
    oracle_progress
    -
    baseline[
        "progress"
    ]
)


preferred_rates = np.asarray(
    [
        r["rate"]
        for r in oracle_rows
    ],
    dtype=np.float64,
)


nonunit_preference_count = int(
    np.sum(
        np.abs(
            preferred_rates
            - 1.0
        )
        >= 0.09
    )
)


unique_preferred_rates = int(
    len(
        np.unique(
            preferred_rates
        )
    )
)


# ================================================================
# DECISION
# ================================================================

print()
print("=" * 175)
print("REFERENCE-TIMING DECISION")
print("=" * 175)

print(
    "fixed 1.00x progress:",
    f"{baseline['progress']:.3f}",
)

print(
    "best global rate:",
    f"{best_constant_rate:.2f}x",
)

print(
    "best global progress:",
    f"{best_constant['progress']:.3f}",
)

print(
    "global rate gain:",
    f"{constant_gain:+.3f}",
)

print(
    "per-start oracle progress:",
    f"{oracle_progress:.3f}",
)

print(
    "per-start oracle gain:",
    f"{oracle_gain:+.3f}",
)

print(
    "starts preferring !=1.00x:",
    f"{nonunit_preference_count}/"
    f"{len(STARTS)}",
)

print(
    "unique preferred rates:",
    unique_preferred_rates,
)


# ================================================================
# INTERPRETATION
# ================================================================

print()


if (
    best_constant_rate
    != 1.00

    and constant_gain
    >= 0.08
):

    print(
        "RESULT: GLOBAL REFERENCE SPEED MISMATCH"
    )

    print(
        f"A constant {best_constant_rate:.2f}x "
        "reference rate materially outperforms "
        "the current 1.00x clock."
    )

    print(
        "NEXT: validate that constant rate with "
        "the closed-loop controller before PPO."
    )


elif (
    oracle_gain
    >= 0.10

    and nonunit_preference_count
    >= 6

    and unique_preferred_rates
    >= 3
):

    print(
        "RESULT: ADAPTIVE PHASE IS JUSTIFIED"
    )

    print(
        "Different motion phases prefer materially "
        "different reference speeds."
    )

    print(
        "A single fixed 50-Hz reference clock is "
        "therefore too restrictive."
    )

    print(
        "NEXT: implement bounded state-dependent "
        "phase-rate adaptation."
    )


elif (
    oracle_gain
    >= 0.05
):

    print(
        "RESULT: PHASE TIMING HAS MARGINAL EFFECT"
    )

    print(
        "Timing matters somewhat, but it is not yet "
        "large enough to call it the primary bottleneck."
    )

    print(
        "Do NOT start PPO yet."
    )


else:

    print(
        "RESULT: FIXED PHASE IS NOT THE PRIMARY BOTTLENECK"
    )

    print(
        "Even allowing each start to choose its own "
        "reference rate gives little improvement."
    )

    print(
        "The next investigation should return to "
        "dynamic/contact feasibility rather than "
        "adding adaptive phase."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_phase_rate_diagnostic.npz"
)


np.savez(
    output,

    phase_rates=
        np.asarray(
            PHASE_RATES,
            dtype=np.float32,
        ),

    starts=
        np.asarray(
            STARTS,
            dtype=np.int32,
        ),

    fixed_progress=
        np.asarray(
            [
                baseline[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    best_constant_rate=
        np.asarray(
            [
                best_constant_rate
            ],
            dtype=np.float32,
        ),

    constant_gain=
        np.asarray(
            [
                constant_gain
            ],
            dtype=np.float32,
        ),

    oracle_progress=
        np.asarray(
            [
                oracle_progress
            ],
            dtype=np.float32,
        ),

    oracle_gain=
        np.asarray(
            [
                oracle_gain
            ],
            dtype=np.float32,
        ),

    preferred_rates=
        preferred_rates.astype(
            np.float32
        ),
)


print()
print(
    "diagnostic artifact:",
    output,
)

print()
print(
    "ZERO RESIDUAL ACTION."
)

print(
    "NO CEM."
)

print(
    "NO PPO TRAINING."
)

print("=" * 175)


env.close()
