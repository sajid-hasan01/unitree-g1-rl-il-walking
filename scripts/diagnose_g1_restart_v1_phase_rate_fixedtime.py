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


RATES = [
    0.60,
    0.75,
    0.90,
    1.00,
    1.10,
    1.25,
    1.40,
]


MAX_RATE = max(
    RATES
)


FIXED_HORIZON = 80


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


print("=" * 185)
print("FIXED-PHYSICAL-TIME PHASE-RATE DIAGNOSTIC")
print("CONTINUOUS REFERENCE INTERPOLATION")
print("ZERO RESIDUAL ACTION")
print("NO CEM / NO PPO")
print("=" * 185)

print(
    "reference:",
    env.reference_path,
)

print(
    "frames:",
    env.num_frames,
)

print(
    "physics/control Hz:",
    env.reference_fps,
)

print(
    "rates:",
    RATES,
)

print(
    "nominal fixed horizon:",
    FIXED_HORIZON,
)


# ================================================================
# QUATERNION SLERP
# wxyz
# ================================================================

def normalize_quat(
    q,
):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return (
        q
        /
        max(
            np.linalg.norm(q),
            1e-12,
        )
    )


def slerp(
    q0,
    q1,
    alpha,
):

    q0 = normalize_quat(
        q0
    )

    q1 = normalize_quat(
        q1
    )


    dot = float(
        np.dot(
            q0,
            q1,
        )
    )


    if dot < 0.0:

        q1 = -q1
        dot = -dot


    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )


    if dot > 0.9995:

        q = (
            (1.0 - alpha)
            * q0
            +
            alpha
            * q1
        )

        return normalize_quat(
            q
        )


    theta = math.acos(
        dot
    )


    sin_theta = math.sin(
        theta
    )


    a = (
        math.sin(
            (1.0 - alpha)
            * theta
        )
        /
        sin_theta
    )


    b = (
        math.sin(
            alpha
            * theta
        )
        /
        sin_theta
    )


    return normalize_quat(
        a * q0
        +
        b * q1
    )


def quat_angle_deg(
    a,
    b,
):

    a = normalize_quat(
        a
    )

    b = normalize_quat(
        b
    )


    dot = abs(
        float(
            np.dot(
                a,
                b,
            )
        )
    )


    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )


    return math.degrees(
        2.0
        * math.acos(
            dot
        )
    )


# ================================================================
# CONTINUOUS REFERENCE
# ================================================================

def interpolate_reference(
    phase,
):

    phase = float(
        np.clip(
            phase,
            0.0,
            env.num_frames - 1.0,
        )
    )


    lo = int(
        math.floor(
            phase
        )
    )


    hi = min(
        lo + 1,
        env.num_frames - 1,
    )


    alpha = (
        phase
        - lo
    )


    q = (
        (1.0 - alpha)
        * env.ref_q[
            lo
        ]
        +
        alpha
        * env.ref_q[
            hi
        ]
    )


    root_pos = (
        (1.0 - alpha)
        * env.ref_root_pos[
            lo
        ]
        +
        alpha
        * env.ref_root_pos[
            hi
        ]
    )


    root_quat = slerp(
        env.ref_root_quat[
            lo
        ],
        env.ref_root_quat[
            hi
        ],
        alpha,
    )


    return (
        q,
        root_pos,
        root_quat,
    )


# ================================================================
# APPLY INTERPOLATED NATIVE POSITION TARGET
# ================================================================

def apply_joint_target(
    target,
):

    target = np.asarray(
        target,
        dtype=np.float64,
    ).copy()


    for j, jid in enumerate(
        env.joint_ids
    ):

        if env.model.jnt_limited[
            jid
        ]:

            lo, hi = (
                env.model.jnt_range[
                    jid
                ]
            )

            target[j] = np.clip(
                target[j],
                lo,
                hi,
            )


        aid = env.actuator_ids[
            j
        ]


        if env.model.actuator_ctrllimited[
            aid
        ]:

            lo, hi = (
                env.model.actuator_ctrlrange[
                    aid
                ]
            )

            target[j] = np.clip(
                target[j],
                lo,
                hi,
            )


    env._apply_target(
        target
    )

    env.last_target = (
        target.copy()
    )


# ================================================================
# ROLLOUT
#
# CRITICAL:
#
# For each start, every phase rate receives the SAME maximum
# number of actual MuJoCo control steps.
# ================================================================

def rollout(
    start,
    rate,
):

    env.reset(
        options={
            "start_frame":
                int(start),
        }
    )


    remaining_frames = (
        env.num_frames
        - 1
        - start
    )


    # Ensure even 1.40x cannot run beyond the available reference.
    common_steps = min(
        FIXED_HORIZON,
        int(
            math.floor(
                remaining_frames
                / MAX_RATE
            )
        ),
    )


    if common_steps < 10:

        raise RuntimeError(
            f"Too few common steps at start {start}"
        )


    phase = float(
        start
    )


    actual_start_x = float(
        env.data.qpos[
            0
        ]
    )


    steps = 0

    reason = "horizon"


    ori_sum = 0.0
    q_sum = 0.0

    root_sum = 0.0

    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0


    min_up = env._up_z()


    final_phase = (
        phase
    )


    final_ref_x = float(
        env.ref_root_pos[
            start,
            0
        ]
    )


    for _ in range(
        common_steps
    ):

        phase += rate


        (
            q_ref,
            root_ref,
            quat_ref,
        ) = interpolate_reference(
            phase
        )


        apply_joint_target(
            q_ref
        )


        for _ in range(
            env.frame_skip
        ):

            mujoco.mj_step(
                env.model,
                env.data,
            )


        # Integer field only for compatibility with helper methods.
        env._current_frame = int(
            np.clip(
                round(
                    phase
                ),
                0,
                env.num_frames - 1,
            )
        )


        steps += 1


        q_actual = env._joint_q()


        q_error = float(
            np.sqrt(
                np.mean(
                    (
                        q_actual
                        - q_ref
                    ) ** 2
                )
            )
        )


        root_error = (
            env.data.qpos[
                0:3
            ]
            -
            root_ref
        )


        orientation_error = (
            quat_angle_deg(
                quat_ref,
                env.data.qpos[
                    3:7
                ],
            )
        )


        q_sum += (
            q_error
        )


        ori_sum += (
            orientation_error
        )


        root_sum += float(
            np.linalg.norm(
                root_error
            )
        )


        x_sum += abs(
            float(
                root_error[0]
            )
        )


        y_sum += abs(
            float(
                root_error[1]
            )
        )


        z_sum += abs(
            float(
                root_error[2]
            )
        )


        min_up = min(
            min_up,
            env._up_z(),
        )


        final_phase = (
            phase
        )

        final_ref_x = float(
            root_ref[0]
        )


        failed, failure_reason = (
            env._physical_failure()
        )


        if failed:

            reason = (
                failure_reason
            )

            break


    divisor = max(
        steps,
        1,
    )


    survival = (
        steps
        / common_steps
    )


    actual_dx = float(
        env.data.qpos[
            0
        ]
        -
        actual_start_x
    )


    desired_dx = float(
        final_ref_x
        -
        env.ref_root_pos[
            start,
            0
        ]
    )


    if abs(
        desired_dx
    ) > 1e-6:

        forward_ratio = (
            actual_dx
            / desired_dx
        )

    else:

        forward_ratio = np.nan


    return {

        "start":
            start,

        "rate":
            rate,

        "steps":
            steps,

        "max_steps":
            common_steps,

        "survival":
            survival,

        "time":
            steps
            / env.reference_fps,

        "phase":
            final_phase,

        "orientation":
            ori_sum
            / divisor,

        "q":
            q_sum
            / divisor,

        "root":
            root_sum
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

        "min_up":
            min_up,

        "actual_dx":
            actual_dx,

        "desired_dx":
            desired_dx,

        "forward_ratio":
            forward_ratio,

        "reason":
            reason,
    }


# ================================================================
# RUN
# ================================================================

results = []


for rate in RATES:

    for start in STARTS:

        results.append(
            rollout(
                start,
                rate,
            )
        )


# ================================================================
# DETAIL
# ================================================================

print()
print("=" * 205)
print("FIXED-TIME PER-START RESULTS")
print("=" * 205)

print(
    f"{'RATE':>5s} "
    f"{'START':>5s} "
    f"{'STEP':>8s} "
    f"{'SURV':>6s} "
    f"{'SEC':>6s} "
    f"{'ORI':>7s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'ROOT':>7s} "
    f"{'XERR':>7s} "
    f"{'YERR':>7s} "
    f"{'ZERR':>7s} "
    f"{'DX':>7s} "
    f"{'REFDX':>7s} "
    f"{'DX/R':>7s} "
    f"{'REASON':>14s}"
)

print("-" * 205)


for r in results:

    ratio = (
        r[
            "forward_ratio"
        ]
    )


    ratio_text = (
        f"{ratio:7.2f}"
        if np.isfinite(
            ratio
        )
        else "    nan"
    )


    print(
        f"{r['rate']:5.2f} "
        f"{r['start']:5d} "
        f"{r['steps']:3d}/"
        f"{r['max_steps']:<3d} "
        f"{r['survival']:6.3f} "
        f"{r['time']:6.2f} "
        f"{r['orientation']:7.2f} "
        f"{r['min_up']:7.3f} "
        f"{r['q']:7.3f} "
        f"{r['root']:7.3f} "
        f"{r['x']:7.3f} "
        f"{r['y']:7.3f} "
        f"{r['z']:7.3f} "
        f"{r['actual_dx']:+7.3f} "
        f"{r['desired_dx']:+7.3f} "
        f"{ratio_text} "
        f"{r['reason']:>14s}"
    )


# ================================================================
# RATE AGGREGATE
# ================================================================

def aggregate(
    rate,
):

    rows = [
        r
        for r in results
        if abs(
            r["rate"]
            - rate
        ) < 1e-9
    ]


    def mean(key):

        return float(
            np.mean(
                [
                    x[key]
                    for x in rows
                ]
            )
        )


    return {
        "survival":
            mean(
                "survival"
            ),

        "min_survival":
            float(
                np.min(
                    [
                        x[
                            "survival"
                        ]
                        for x in rows
                    ]
                )
            ),

        "time":
            mean(
                "time"
            ),

        "orientation":
            mean(
                "orientation"
            ),

        "min_up":
            mean(
                "min_up"
            ),

        "q":
            mean(
                "q"
            ),

        "root":
            mean(
                "root"
            ),

        "x":
            mean(
                "x"
            ),

        "y":
            mean(
                "y"
            ),

        "z":
            mean(
                "z"
            ),
    }


summary = {
    rate:
        aggregate(
            rate
        )
    for rate in RATES
}


print()
print("=" * 170)
print("FIXED-PHYSICAL-TIME RATE SUMMARY")
print("=" * 170)

print(
    f"{'RATE':>5s} "
    f"{'SURV':>7s} "
    f"{'MIN-S':>7s} "
    f"{'TIME':>7s} "
    f"{'ORI':>8s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'ROOT':>7s} "
    f"{'XERR':>7s} "
    f"{'YERR':>7s} "
    f"{'ZERR':>7s}"
)

print("-" * 170)


for rate in RATES:

    s = summary[
        rate
    ]


    print(
        f"{rate:5.2f} "
        f"{s['survival']:7.3f} "
        f"{s['min_survival']:7.3f} "
        f"{s['time']:7.2f} "
        f"{s['orientation']:8.2f} "
        f"{s['min_up']:7.3f} "
        f"{s['q']:7.3f} "
        f"{s['root']:7.3f} "
        f"{s['x']:7.3f} "
        f"{s['y']:7.3f} "
        f"{s['z']:7.3f}"
    )


# ================================================================
# GLOBAL BEST
# ================================================================

baseline = summary[
    1.00
]


best_rate = max(
    RATES,

    key=lambda rate: (
        summary[
            rate
        ][
            "survival"
        ],

        summary[
            rate
        ][
            "min_survival"
        ],

        -summary[
            rate
        ][
            "orientation"
        ],

        -summary[
            rate
        ][
            "root"
        ],

        -abs(
            rate
            - 1.0
        ),
    ),
)


best_global = summary[
    best_rate
]


global_survival_gain = (
    best_global[
        "survival"
    ]
    -
    baseline[
        "survival"
    ]
)


# ================================================================
# PER-START ORACLE
#
# Now rate selection is based on equal physical-time survival.
# ================================================================

print()
print("=" * 175)
print("FIXED-TIME PER-START BEST RATE")
print("=" * 175)


oracle = []


for start in STARTS:

    candidates = [
        r
        for r in results
        if r["start"]
        == start
    ]


    fixed = next(
        r
        for r in candidates
        if abs(
            r["rate"]
            - 1.0
        ) < 1e-9
    )


    best = max(
        candidates,

        key=lambda r: (
            r[
                "survival"
            ],

            -r[
                "orientation"
            ],

            -r[
                "root"
            ],

            -r[
                "q"
            ],

            -abs(
                r[
                    "rate"
                ]
                - 1.0
            ),
        ),
    )


    oracle.append(
        best
    )


    print(
        f"start={start:03d} "
        f"fixedSurv="
        f"{fixed['survival']:.3f} "
        f"bestSurv="
        f"{best['survival']:.3f} "
        f"gain="
        f"{best['survival']-fixed['survival']:+.3f} "
        f"rate="
        f"{best['rate']:.2f}x "
        f"ori="
        f"{fixed['orientation']:.2f}"
        f"->{best['orientation']:.2f} "
        f"root="
        f"{fixed['root']:.3f}"
        f"->{best['root']:.3f}"
    )


fixed_mean_survival = float(
    np.mean(
        [
            r["survival"]
            for r in results
            if abs(
                r["rate"]
                - 1.0
            ) < 1e-9
        ]
    )
)


oracle_mean_survival = float(
    np.mean(
        [
            r["survival"]
            for r in oracle
        ]
    )
)


oracle_gain = (
    oracle_mean_survival
    -
    fixed_mean_survival
)


preferred_rates = np.asarray(
    [
        r["rate"]
        for r in oracle
    ],
    dtype=np.float64,
)


nonunit_count = int(
    np.sum(
        np.abs(
            preferred_rates
            - 1.0
        )
        >= 0.09
    )
)


unique_rates = len(
    np.unique(
        preferred_rates
    )
)


# ================================================================
# DECISION
# ================================================================

print()
print("=" * 175)
print("CORRECTED REFERENCE-TIMING DECISION")
print("=" * 175)

print(
    "1.00x mean survival:",
    f"{baseline['survival']:.3f}",
)

print(
    "best global rate:",
    f"{best_rate:.2f}x",
)

print(
    "best global survival:",
    f"{best_global['survival']:.3f}",
)

print(
    "global survival gain:",
    f"{global_survival_gain:+.3f}",
)

print(
    "oracle mean survival:",
    f"{oracle_mean_survival:.3f}",
)

print(
    "oracle survival gain:",
    f"{oracle_gain:+.3f}",
)

print(
    "starts preferring !=1.00x:",
    f"{nonunit_count}/"
    f"{len(STARTS)}",
)

print(
    "unique preferred rates:",
    unique_rates,
)


print()


if (
    best_rate != 1.00

    and global_survival_gain >= 0.08

    and (
        best_global[
            "orientation"
        ]
        -
        baseline[
            "orientation"
        ]
    ) <= 5.0

    and (
        best_global[
            "root"
        ]
        -
        baseline[
            "root"
        ]
    ) <= 0.05
):

    print(
        "RESULT: GLOBAL PHASE-RATE MISMATCH CONFIRMED"
    )

    print(
        f"{best_rate:.2f}x improves actual "
        "equal-time physical survival."
    )

    print(
        "NEXT: validate this rate with the "
        "closed-loop balance controller."
    )


elif (
    oracle_gain >= 0.10

    and nonunit_count >= 6

    and unique_rates >= 3
):

    print(
        "RESULT: ADAPTIVE PHASE IS JUSTIFIED"
    )

    print(
        "Different starts prefer different phase rates "
        "even under equal physical-time evaluation."
    )

    print(
        "NEXT: implement bounded state-dependent "
        "phase synchronization."
    )


elif oracle_gain >= 0.05:

    print(
        "RESULT: PHASE HAS A MARGINAL EFFECT"
    )

    print(
        "Timing contributes, but it is not yet "
        "the dominant bottleneck."
    )

    print(
        "NO PPO YET."
    )


else:

    print(
        "RESULT: PHASE RATE IS NOT THE PRIMARY BOTTLENECK"
    )

    print(
        "The previous 1.40x result was mainly a "
        "reference-progress scoring artifact."
    )

    print(
        "NEXT: investigate dynamic/contact feasibility "
        "and support-transition mechanics."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_phase_rate_fixedtime.npz"
)


np.savez(
    output,

    rates=
        np.asarray(
            RATES,
            dtype=np.float32,
        ),

    starts=
        np.asarray(
            STARTS,
            dtype=np.int32,
        ),

    fixed_survival=
        np.asarray(
            [
                baseline[
                    "survival"
                ]
            ],
            dtype=np.float32,
        ),

    best_rate=
        np.asarray(
            [
                best_rate
            ],
            dtype=np.float32,
        ),

    global_survival_gain=
        np.asarray(
            [
                global_survival_gain
            ],
            dtype=np.float32,
        ),

    oracle_survival_gain=
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
    "artifact:",
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
    "NO PPO."
)

print("=" * 175)


env.close()
