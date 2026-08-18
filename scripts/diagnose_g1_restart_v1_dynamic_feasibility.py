from pathlib import Path
import math
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


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


HORIZON = 120


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model


DT = (
    1.0
    / env.reference_fps
)


TOTAL_MASS = float(
    mujoco.mj_getTotalmass(
        model
    )
)


GRAVITY = abs(
    float(
        model.opt.gravity[
            2
        ]
    )
)


BODY_WEIGHT = (
    TOTAL_MASS
    * GRAVITY
)


print("=" * 185)
print("G1 29-DOF REFERENCE DYNAMIC FEASIBILITY AUDIT")
print("+ CORRECT INVERSE-DYNAMICS FEEDFORWARD TEST")
print("NO CEM")
print("NO PPO")
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
    "control Hz:",
    env.reference_fps,
)

print(
    "model mass:",
    f"{TOTAL_MASS:.3f} kg",
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)

print(
    "actuated DOF:",
    len(
        env.vaddrs
    ),
)


# ================================================================
# REFERENCE ACCELERATION
#
# qvel is already in MuJoCo nv coordinates.
# Central difference gives desired qacc.
# ================================================================

ref_vel = np.asarray(
    env.ref_full_qvel,
    dtype=np.float64,
)


ref_acc = np.zeros_like(
    ref_vel
)


ref_acc[
    1:-1
] = (
    ref_vel[
        2:
    ]
    -
    ref_vel[
        :-2
    ]
) / (
    2.0
    * DT
)


ref_acc[
    0
] = (
    ref_vel[
        1
    ]
    -
    ref_vel[
        0
    ]
) / DT


ref_acc[
    -1
] = (
    ref_vel[
        -1
    ]
    -
    ref_vel[
        -2
    ]
) / DT


# ================================================================
# INVERSE-DYNAMICS AUDIT
# ================================================================

inv_data = mujoco.MjData(
    model
)


N = env.num_frames


inverse_tau = np.zeros(
    (
        N,
        29,
    ),
    dtype=np.float64,
)


root_force = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


root_torque = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


left_normal = np.zeros(
    N,
    dtype=np.float64,
)


right_normal = np.zeros(
    N,
    dtype=np.float64,
)


contact_count = np.zeros(
    N,
    dtype=np.int32,
)


left_set = set(
    env.left_sole_geoms
)

right_set = set(
    env.right_sole_geoms
)


contact_force = np.zeros(
    6,
    dtype=np.float64,
)


for frame in range(
    N
):

    inv_data.qpos[:] = (
        env.ref_full_qpos[
            frame
        ]
    )

    inv_data.qvel[:] = (
        env.ref_full_qvel[
            frame
        ]
    )


    # Construct kinematics and active contacts.
    mujoco.mj_forward(
        model,
        inv_data,
    )


    # Desired reference acceleration.
    inv_data.qacc[:] = (
        ref_acc[
            frame
        ]
    )


    # Inverse dynamics with the active constraints/contact state.
    mujoco.mj_inverse(
        model,
        inv_data,
    )


    root_force[
        frame
    ] = (
        inv_data.qfrc_inverse[
            0:3
        ]
    )


    root_torque[
        frame
    ] = (
        inv_data.qfrc_inverse[
            3:6
        ]
    )


    inverse_tau[
        frame
    ] = np.asarray(
        [
            inv_data.qfrc_inverse[
                vadr
            ]
            for vadr
            in env.vaddrs
        ],
        dtype=np.float64,
    )


    # ------------------------------------------------------------
    # Floor/sole inverse contact forces.
    # ------------------------------------------------------------

    for cid in range(
        inv_data.ncon
    ):

        con = inv_data.contact[
            cid
        ]


        g1 = int(
            con.geom1
        )

        g2 = int(
            con.geom2
        )


        pair = {
            g1,
            g2,
        }


        if env.floor_geom not in pair:
            continue


        other = (
            g2
            if g1
            == env.floor_geom
            else g1
        )


        if (
            other not in left_set
            and
            other not in right_set
        ):
            continue


        contact_force[:] = 0.0


        mujoco.mj_contactForce(
            model,
            inv_data,
            cid,
            contact_force,
        )


        # First component is contact-normal force
        # in MuJoCo contact coordinates.
        fn = abs(
            float(
                contact_force[
                    0
                ]
            )
        )


        contact_count[
            frame
        ] += 1


        if other in left_set:

            left_normal[
                frame
            ] += fn


        if other in right_set:

            right_normal[
                frame
            ] += fn


# ================================================================
# ROOT RESIDUAL METRICS
# ================================================================

root_force_norm = np.linalg.norm(
    root_force,
    axis=1,
)


root_torque_norm = np.linalg.norm(
    root_torque,
    axis=1,
)


root_force_bw = (
    root_force_norm
    / BODY_WEIGHT
)


# ================================================================
# ACTUATOR REQUIREMENT METRICS
# ================================================================

effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


torque_ratio = (
    np.abs(
        inverse_tau
    )
    /
    effort[
        None,
        :
    ]
)


any_over_100 = np.any(
    torque_ratio
    > 1.0,
    axis=1,
)


any_over_80 = np.any(
    torque_ratio
    > 0.80,
    axis=1,
)


# ================================================================
# SUPPORT / CONTACT AUDIT
# ================================================================

normal_total = (
    left_normal
    +
    right_normal
)


normal_bw = (
    normal_total
    / BODY_WEIGHT
)


support_expected_left = (
    env.ref_support[
        :,
        0
    ]
    > 0.5
)


support_expected_right = (
    env.ref_support[
        :,
        1
    ]
    > 0.5
)


weak_left_support = (
    support_expected_left
    &
    (
        left_normal
        <
        0.05
        * BODY_WEIGHT
    )
)


weak_right_support = (
    support_expected_right
    &
    (
        right_normal
        <
        0.05
        * BODY_WEIGHT
    )
)


# ================================================================
# REPORT AUDIT
# ================================================================

print()
print("=" * 185)
print("INVERSE-DYNAMICS REFERENCE AUDIT")
print("=" * 185)


def percentile_text(
    x,
):

    return (
        f"p50={np.percentile(x, 50):.4f} "
        f"p90={np.percentile(x, 90):.4f} "
        f"p95={np.percentile(x, 95):.4f} "
        f"p99={np.percentile(x, 99):.4f} "
        f"max={np.max(x):.4f}"
    )


print()
print(
    "FLOATING-BASE UNEXPLAINED FORCE / BODY WEIGHT"
)

print(
    percentile_text(
        root_force_bw
    )
)


print()
print(
    "FLOATING-BASE UNEXPLAINED FORCE [N]"
)

print(
    percentile_text(
        root_force_norm
    )
)


print()
print(
    "FLOATING-BASE UNEXPLAINED TORQUE [Nm]"
)

print(
    percentile_text(
        root_torque_norm
    )
)


print()
print(
    "REQUIRED ACTUATOR TORQUE / EFFORT LIMIT"
)

print(
    percentile_text(
        torque_ratio.reshape(
            -1
        )
    )
)


print()
print(
    "frames any joint >80% effort:",
    int(
        np.sum(
            any_over_80
        )
    ),
    "/",
    N,
    f"({100*np.mean(any_over_80):.1f}%)",
)


print(
    "frames any joint >100% effort:",
    int(
        np.sum(
            any_over_100
        )
    ),
    "/",
    N,
    f"({100*np.mean(any_over_100):.1f}%)",
)


print()
print(
    "TOTAL INVERSE CONTACT NORMAL / BODY WEIGHT"
)

print(
    percentile_text(
        normal_bw
    )
)


print()
print(
    "frames with sole-floor contact:",
    int(
        np.sum(
            contact_count
            > 0
        )
    ),
    "/",
    N,
)


print(
    "expected-left-support frames with <5% BW left force:",
    int(
        np.sum(
            weak_left_support
        )
    ),
    "/",
    int(
        np.sum(
            support_expected_left
        )
    ),
)


print(
    "expected-right-support frames with <5% BW right force:",
    int(
        np.sum(
            weak_right_support
        )
    ),
    "/",
    int(
        np.sum(
            support_expected_right
        )
    ),
)


# ================================================================
# PER-JOINT TORQUE REQUIREMENT
# ================================================================

print()
print("=" * 185)
print("PER-JOINT INVERSE-DYNAMICS REQUIREMENT")
print("=" * 185)

print(
    f"{'#':>2s} "
    f"{'JOINT':29s} "
    f"{'LIMIT':>8s} "
    f"{'P50':>8s} "
    f"{'P95':>8s} "
    f"{'MAX':>8s} "
    f"{'P95/L':>8s} "
    f"{'MAX/L':>8s}"
)


for j, name in enumerate(
    env.ref_names
):

    abs_tau = np.abs(
        inverse_tau[
            :,
            j
        ]
    )


    p50 = float(
        np.percentile(
            abs_tau,
            50,
        )
    )


    p95 = float(
        np.percentile(
            abs_tau,
            95,
        )
    )


    maximum = float(
        np.max(
            abs_tau
        )
    )


    print(
        f"{j:2d} "
        f"{name:29s} "
        f"{effort[j]:8.2f} "
        f"{p50:8.2f} "
        f"{p95:8.2f} "
        f"{maximum:8.2f} "
        f"{p95/effort[j]:8.3f} "
        f"{maximum/effort[j]:8.3f}"
    )


# ================================================================
# NATIVE POSITION TARGET CLIPPING
# ================================================================

def clip_target(
    target,
):

    target = np.asarray(
        target,
        dtype=np.float64,
    ).copy()


    for j, jid in enumerate(
        env.joint_ids
    ):

        if model.jnt_limited[
            jid
        ]:

            lo, hi = (
                model.jnt_range[
                    jid
                ]
            )


            target[
                j
            ] = float(
                np.clip(
                    target[
                        j
                    ],
                    lo,
                    hi,
                )
            )


        aid = env.actuator_ids[
            j
        ]


        if model.actuator_ctrllimited[
            aid
        ]:

            lo, hi = (
                model.actuator_ctrlrange[
                    aid
                ]
            )


            target[
                j
            ] = float(
                np.clip(
                    target[
                        j
                    ],
                    lo,
                    hi,
                )
            )


    return target


# ================================================================
# CONTROLLER MODES
#
# BASE:
#   ctrl = q_ref
#
# VELFF:
#   ctrl = q_ref + Kd/Kp * qdot_ref
#
# IDFF:
#   ctrl = q_ref
#        + (tau_inverse + Kd*qdot_ref)/Kp
#
# tau_inverse is limited to the physical actuator effort limit.
# ================================================================

MODES = [
    "BASE",
    "VELFF",
    "IDFF",
]


def target_for_mode(
    mode,
    frame,
):

    qref = env.ref_q[
        frame
    ]


    qdref = env.ref_qd[
        frame
    ]


    if mode == "BASE":

        target = (
            qref.copy()
        )


    elif mode == "VELFF":

        target = (
            qref
            +
            (
                env.damping
                * qdref
            )
            /
            env.stiffness
        )


    elif mode == "IDFF":

        tau = np.clip(
            inverse_tau[
                frame
            ],
            -effort,
            effort,
        )


        target = (
            qref
            +
            (
                tau
                +
                env.damping
                * qdref
            )
            /
            env.stiffness
        )


    else:

        raise ValueError(
            mode
        )


    return clip_target(
        target
    )


# ================================================================
# DYNAMIC REPLAY
# ================================================================

def rollout(
    mode,
    start,
):

    env.reset(
        options={
            "start_frame":
                int(
                    start
                ),
        }
    )


    maximum_steps = min(
        HORIZON,
        env.num_frames
        - 1
        - start,
    )


    steps = 0

    reason = ""


    ori_sum = 0.0
    q_sum = 0.0
    root_sum = 0.0

    min_up = (
        env._up_z()
    )


    saturation_count = 0
    actuator_samples = 0


    shift_sum = 0.0
    shift_max = 0.0


    while steps < maximum_steps:

        target_frame = (
            env._current_frame
            + 1
        )


        target = target_for_mode(
            mode,
            target_frame,
        )


        shift = (
            target
            -
            env.ref_q[
                target_frame
            ]
        )


        shift_rms = float(
            np.sqrt(
                np.mean(
                    shift ** 2
                )
            )
        )


        shift_sum += (
            shift_rms
        )


        shift_max = max(
            shift_max,
            float(
                np.max(
                    np.abs(
                        shift
                    )
                )
            ),
        )


        env._apply_target(
            target
        )

        env.last_target = (
            target.copy()
        )


        for _ in range(
            env.frame_skip
        ):

            mujoco.mj_step(
                model,
                env.data,
            )


            actual_force = np.abs(
                np.asarray(
                    env.data.actuator_force,
                    dtype=np.float64,
                )
            )


            saturation_count += int(
                np.sum(
                    actual_force
                    >=
                    0.99
                    * effort
                )
            )


            actuator_samples += 29


        env._current_frame = (
            target_frame
        )


        steps += 1


        q_actual = (
            env._joint_q()
        )


        q_error = float(
            np.sqrt(
                np.mean(
                    (
                        q_actual
                        -
                        env.ref_q[
                            target_frame
                        ]
                    ) ** 2
                )
            )
        )


        root_error = float(
            np.linalg.norm(
                env.data.qpos[
                    0:3
                ]
                -
                env.ref_root_pos[
                    target_frame
                ]
            )
        )


        orientation = math.degrees(
            quat_angle(
                env.ref_root_quat[
                    target_frame
                ],
                env.data.qpos[
                    3:7
                ],
            )
        )


        q_sum += (
            q_error
        )

        root_sum += (
            root_error
        )

        ori_sum += (
            orientation
        )


        min_up = min(
            min_up,
            env._up_z(),
        )


        failed, failure_reason = (
            env._physical_failure()
        )


        if failed:

            reason = (
                failure_reason
            )

            break


    if not reason:

        if (
            env._current_frame
            >= env.num_frames - 1
        ):

            reason = (
                "reference_end"
            )

        else:

            reason = (
                "horizon"
            )


    divisor = max(
        steps,
        1,
    )


    return {

        "mode":
            mode,

        "start":
            start,

        "steps":
            steps,

        "max_steps":
            maximum_steps,

        "progress":
            steps
            / maximum_steps,

        "orientation":
            ori_sum
            / divisor,

        "root":
            root_sum
            / divisor,

        "q":
            q_sum
            / divisor,

        "min_up":
            min_up,

        "saturation":
            saturation_count
            / max(
                actuator_samples,
                1,
            ),

        "shift_rms":
            shift_sum
            / divisor,

        "shift_max":
            shift_max,

        "reason":
            reason,
    }


# ================================================================
# RUN REPLAYS
# ================================================================

print()
print("=" * 190)
print("NATIVE-SERVO FEEDFORWARD REPLAY")
print("=" * 190)


results = []


for mode in MODES:

    for start in STARTS:

        result = rollout(
            mode,
            start,
        )


        results.append(
            result
        )


# ================================================================
# PER-START REPLAY RESULTS
# ================================================================

print()
print(
    f"{'MODE':7s} "
    f"{'START':>5s} "
    f"{'STEPS':>8s} "
    f"{'PROG':>6s} "
    f"{'ORI':>7s} "
    f"{'ROOT':>7s} "
    f"{'QERR':>7s} "
    f"{'MINUP':>7s} "
    f"{'SAT':>7s} "
    f"{'SHIFT':>7s} "
    f"{'MAXSH':>7s} "
    f"{'REASON':>14s}"
)

print("-" * 155)


for r in results:

    print(
        f"{r['mode']:7s} "
        f"{r['start']:5d} "
        f"{r['steps']:3d}/"
        f"{r['max_steps']:<3d} "
        f"{r['progress']:6.3f} "
        f"{r['orientation']:7.2f} "
        f"{r['root']:7.3f} "
        f"{r['q']:7.3f} "
        f"{r['min_up']:7.3f} "
        f"{r['saturation']:7.3f} "
        f"{r['shift_rms']:7.4f} "
        f"{r['shift_max']:7.4f} "
        f"{r['reason']:>14s}"
    )


# ================================================================
# AGGREGATE REPLAY
# ================================================================

def aggregate(
    mode,
):

    rows = [
        r
        for r in results
        if r[
            "mode"
        ] == mode
    ]


    def mean(
        key,
    ):

        return float(
            np.mean(
                [
                    r[
                        key
                    ]
                    for r in rows
                ]
            )
        )


    return {

        "progress":
            mean(
                "progress"
            ),

        "steps":
            mean(
                "steps"
            ),

        "orientation":
            mean(
                "orientation"
            ),

        "root":
            mean(
                "root"
            ),

        "q":
            mean(
                "q"
            ),

        "min_up":
            mean(
                "min_up"
            ),

        "saturation":
            mean(
                "saturation"
            ),

        "shift":
            mean(
                "shift_rms"
            ),
    }


summary = {
    mode:
        aggregate(
            mode
        )
    for mode in MODES
}


print()
print("=" * 160)
print("FEEDFORWARD AGGREGATE")
print("=" * 160)

print(
    f"{'MODE':7s} "
    f"{'STEPS':>8s} "
    f"{'PROG':>7s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'QERR':>8s} "
    f"{'MINUP':>8s} "
    f"{'SAT':>8s} "
    f"{'SHIFT':>8s}"
)

print("-" * 160)


for mode in MODES:

    s = summary[
        mode
    ]


    print(
        f"{mode:7s} "
        f"{s['steps']:8.1f} "
        f"{s['progress']:7.3f} "
        f"{s['orientation']:8.2f} "
        f"{s['root']:8.3f} "
        f"{s['q']:8.3f} "
        f"{s['min_up']:8.3f} "
        f"{s['saturation']:8.3f} "
        f"{s['shift']:8.4f}"
    )


# ================================================================
# BASELINE CONSISTENCY
# ================================================================

expected = {
    0:
        102,

    57:
        51,

    114:
        58,
}


baseline_consistent = True


for start, expected_steps in expected.items():

    row = next(
        r
        for r in results
        if (
            r["mode"]
            == "BASE"

            and
            r["start"]
            == start
        )
    )


    if row[
        "steps"
    ] != expected_steps:

        baseline_consistent = False


print()
print(
    "baseline exact 102/51/58 reproduction:",
    baseline_consistent,
)


# ================================================================
# DECISION METRICS
# ================================================================

base = summary[
    "BASE"
]


velf = summary[
    "VELFF"
]


idff = summary[
    "IDFF"
]


id_progress_gain = (
    idff[
        "progress"
    ]
    -
    base[
        "progress"
    ]
)


id_orientation_change = (
    idff[
        "orientation"
    ]
    -
    base[
        "orientation"
    ]
)


id_root_change = (
    idff[
        "root"
    ]
    -
    base[
        "root"
    ]
)


root_force_p95_bw = float(
    np.percentile(
        root_force_bw,
        95,
    )
)


root_force_max_bw = float(
    np.max(
        root_force_bw
    )
)


torque_p95_ratio = float(
    np.percentile(
        torque_ratio,
        95,
    )
)


torque_max_ratio = float(
    np.max(
        torque_ratio
    )
)


overlimit_frame_fraction = float(
    np.mean(
        any_over_100
    )
)


print()
print("=" * 185)
print("DYNAMIC FEASIBILITY DECISION")
print("=" * 185)

print(
    "root residual p95:",
    f"{root_force_p95_bw:.3f} body-weight",
)

print(
    "root residual max:",
    f"{root_force_max_bw:.3f} body-weight",
)

print(
    "joint torque p95/limit:",
    f"{torque_p95_ratio:.3f}",
)

print(
    "joint torque max/limit:",
    f"{torque_max_ratio:.3f}",
)

print(
    "frames with required torque > limit:",
    f"{100*overlimit_frame_fraction:.1f}%",
)

print()
print(
    "IDFF progress gain:",
    f"{id_progress_gain:+.3f}",
)

print(
    "IDFF orientation change:",
    f"{id_orientation_change:+.2f} deg",
)

print(
    "IDFF root-error change:",
    f"{id_root_change:+.3f} m",
)

print(
    "IDFF actual force saturation:",
    f"{100*idff['saturation']:.2f}%",
)


print()


# ================================================================
# INTERPRETATION
# ================================================================

if (
    id_progress_gain
    >= 0.12

    and id_orientation_change
    <= 5.0

    and id_root_change
    <= 0.05
):

    print(
        "RESULT: INVERSE-DYNAMICS FEEDFORWARD IS IMPORTANT"
    )

    print(
        "The reference becomes substantially more "
        "trackable when its required dynamics are "
        "added to the native position servo."
    )

    print(
        "NEXT: integrate bounded inverse-dynamics "
        "feedforward into Restart V2, then validate "
        "before PPO."
    )


elif (
    root_force_p95_bw
    >= 0.15

    or root_force_max_bw
    >= 0.40
):

    print(
        "RESULT: REFERENCE HAS SIGNIFICANT "
        "FLOATING-BASE PHYSICS RESIDUAL"
    )

    print(
        "The grounded kinematic reference is not "
        "fully dynamically consistent with the "
        "MuJoCo G1/contact model."
    )

    print(
        "NEXT: dynamically project / optimize the "
        "reference instead of adding more feedback."
    )


elif (
    overlimit_frame_fraction
    >= 0.10

    or torque_p95_ratio
    >= 0.90
):

    print(
        "RESULT: ACTUATOR-LIMIT BOTTLENECK"
    )

    print(
        "A meaningful part of the reference requires "
        "joint effort close to or above the model's "
        "available actuator limits."
    )

    print(
        "NEXT: motion-speed/amplitude feasibility "
        "optimization before PPO."
    )


elif (
    id_progress_gain
    < 0.05
):

    print(
        "RESULT: FEEDFORWARD DOES NOT SOLVE THE FAILURE"
    )

    print(
        "Reference torque demand is not the missing "
        "controller ingredient."
    )

    print(
        "NEXT: inspect support-transition mechanics, "
        "impact timing, friction, and contact impulse "
        "consistency."
    )


else:

    print(
        "RESULT: DYNAMIC FEASIBILITY / FEEDFORWARD "
        "IS MARGINAL"
    )

    print(
        "Do NOT start PPO yet."
    )

    print(
        "NEXT: isolate the specific high-residual "
        "frames and support transitions."
    )


# ================================================================
# SAVE DIAGNOSTIC
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_dynamic_feasibility.npz"
)


np.savez(
    output,

    ref_acc=
        ref_acc.astype(
            np.float32
        ),

    inverse_tau=
        inverse_tau.astype(
            np.float32
        ),

    root_force=
        root_force.astype(
            np.float32
        ),

    root_torque=
        root_torque.astype(
            np.float32
        ),

    left_normal=
        left_normal.astype(
            np.float32
        ),

    right_normal=
        right_normal.astype(
            np.float32
        ),

    root_force_bw=
        root_force_bw.astype(
            np.float32
        ),

    torque_ratio=
        torque_ratio.astype(
            np.float32
        ),

    id_progress_gain=
        np.asarray(
            [
                id_progress_gain
            ],
            dtype=np.float32,
        ),

    baseline_consistent=
        np.asarray(
            [
                baseline_consistent
            ],
            dtype=np.bool_,
        ),
)


print()
print(
    "diagnostic artifact:",
    output,
)

print()
print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 185)


env.close()
