from pathlib import Path
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

DEPTHS = [
    1e-6,     # 0.001 mm = 1 micron
    1e-5,     # 0.010 mm
    1e-4,     # 0.100 mm
]

TRANSITION_WINDOW = 2


env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model

N = env.num_frames

DT = 1.0 / env.reference_fps


TOTAL_MASS = float(
    mujoco.mj_getTotalmass(model)
)

GRAVITY = abs(
    float(model.opt.gravity[2])
)

BODY_WEIGHT = (
    TOTAL_MASS
    * GRAVITY
)


LEFT = set(
    env.left_sole_geoms
)

RIGHT = set(
    env.right_sole_geoms
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


print("=" * 190)
print("KINEMATICALLY-CONSISTENT CONTACT-SNAPPED")
print("INVERSE-DYNAMICS AUDIT")
print("QVEL/QACC RECOMPUTED FROM SNAPPED QPOS")
print("NO CEM / NO PPO")
print("=" * 190)

print("reference:", env.reference_path)
print("frames:", N)
print("frequency:", env.reference_fps)
print("mass:", f"{TOTAL_MASS:.3f} kg")
print("body weight:", f"{BODY_WEIGHT:.2f} N")
print(
    "contact depths:",
    [
        f"{1000*x:.3f} mm"
        for x in DEPTHS
    ],
)


# ================================================================
# SUPPORT MASK
# ================================================================

def support_mask(frame):

    support = (
        np.asarray(
            env.ref_support[frame],
            dtype=np.float64,
        )
        > 0.5
    )


    if not np.any(support):

        support = (
            np.asarray(
                env.ref_contact[frame],
                dtype=np.float64,
            )
            > 0.5
        )


    return support


# ================================================================
# TRANSITION MASK
# ================================================================

transition = np.zeros(
    N,
    dtype=bool,
)


for frame in range(
    1,
    N,
):

    if not np.array_equal(
        env.ref_support[frame],
        env.ref_support[frame - 1],
    ):

        lo = max(
            0,
            frame - TRANSITION_WINDOW,
        )

        hi = min(
            N,
            frame + TRANSITION_WINDOW + 1,
        )

        transition[
            lo:hi
        ] = True


support_count = np.asarray(
    [
        np.sum(
            support_mask(frame)
        )
        for frame in range(N)
    ],
    dtype=np.int32,
)


single_support = (
    support_count == 1
)

double_support = (
    support_count >= 2
)


steady_single = (
    single_support
    &
    ~transition
)

steady_double = (
    double_support
    &
    ~transition
)


# ================================================================
# GEOMETRY
# ================================================================

probe = mujoco.MjData(
    model
)


def sole_clearance(
    data,
    geoms,
):

    floor_z = float(
        data.geom_xpos[
            env.floor_geom,
            2,
        ]
    )


    clearances = []


    for gid in geoms:

        radius = float(
            model.geom_size[
                gid,
                0,
            ]
        )


        bottom = float(
            data.geom_xpos[
                gid,
                2,
            ]
            - radius
        )


        clearances.append(
            bottom - floor_z
        )


    return min(
        clearances
    )


def choose_anchor(
    frame,
):

    probe.qpos[:] = (
        env.ref_full_qpos[
            frame
        ]
    )

    probe.qvel[:] = (
        env.ref_full_qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        probe,
    )


    left_clearance = (
        sole_clearance(
            probe,
            env.left_sole_geoms,
        )
    )


    right_clearance = (
        sole_clearance(
            probe,
            env.right_sole_geoms,
        )
    )


    support = support_mask(
        frame
    )


    candidates = []


    if support[0]:

        candidates.append(
            (
                0,
                left_clearance,
            )
        )


    if support[1]:

        candidates.append(
            (
                1,
                right_clearance,
            )
        )


    if not candidates:

        candidates = [
            (
                0,
                left_clearance,
            ),
            (
                1,
                right_clearance,
            ),
        ]


    return min(
        candidates,
        key=lambda x: x[1],
    )


# ================================================================
# BUILD A COMPLETE SNAPPED QPOS TRAJECTORY
# ================================================================

def build_snapped_qpos(
    depth,
):

    qpos = np.asarray(
        env.ref_full_qpos,
        dtype=np.float64,
    ).copy()


    shifts = np.zeros(
        N,
        dtype=np.float64,
    )


    anchors = np.zeros(
        N,
        dtype=np.int8,
    )


    for frame in range(N):

        (
            anchor,
            clearance,
        ) = choose_anchor(
            frame
        )


        anchors[
            frame
        ] = anchor


        # target clearance = -depth
        dz = (
            -depth
            - clearance
        )


        # Contact snapping only lowers root.
        dz = min(
            0.0,
            dz,
        )


        # Safety only.
        dz = max(
            -0.020,
            dz,
        )


        qpos[
            frame,
            2
        ] += dz


        shifts[
            frame
        ] = dz


    return (
        qpos,
        shifts,
        anchors,
    )


# ================================================================
# DIFFERENTIATE QPOS CORRECTLY
# ================================================================

def differentiate_qpos(
    qpos,
):

    qvel = np.zeros(
        (
            N,
            model.nv,
        ),
        dtype=np.float64,
    )


    # Forward endpoint.
    mujoco.mj_differentiatePos(
        model,
        qvel[0],
        DT,
        qpos[0],
        qpos[1],
    )


    # Central differences.
    for frame in range(
        1,
        N - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            qvel[frame],
            2.0 * DT,
            qpos[frame - 1],
            qpos[frame + 1],
        )


    # Backward endpoint.
    mujoco.mj_differentiatePos(
        model,
        qvel[-1],
        DT,
        qpos[-2],
        qpos[-1],
    )


    return qvel


def differentiate_velocity(
    qvel,
):

    qacc = np.zeros_like(
        qvel
    )


    qacc[1:-1] = (
        qvel[2:]
        -
        qvel[:-2]
    ) / (
        2.0 * DT
    )


    qacc[0] = (
        qvel[1]
        -
        qvel[0]
    ) / DT


    qacc[-1] = (
        qvel[-1]
        -
        qvel[-2]
    ) / DT


    return qacc


# ================================================================
# CONTACT
# ================================================================

def contact_state(
    data,
):

    left = False
    right = False


    for cid in range(
        data.ncon
    ):

        con = data.contact[
            cid
        ]


        if int(
            con.efc_address
        ) < 0:

            continue


        g1 = int(
            con.geom1
        )

        g2 = int(
            con.geom2
        )


        if env.floor_geom not in (
            g1,
            g2,
        ):

            continue


        other = (
            g2
            if g1
            == env.floor_geom
            else g1
        )


        if other in LEFT:
            left = True


        elif other in RIGHT:
            right = True


    return (
        left,
        right,
    )


# ================================================================
# INVERSE AUDIT
# ================================================================

def audit(
    depth,
):

    (
        qpos,
        shifts,
        anchors,
    ) = build_snapped_qpos(
        depth
    )


    # CRITICAL DIFFERENCE FROM PREVIOUS AUDIT.
    qvel = differentiate_qpos(
        qpos
    )


    qacc = differentiate_velocity(
        qvel
    )


    data = mujoco.MjData(
        model
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


    joint_tau = np.zeros(
        (
            N,
            29,
        ),
        dtype=np.float64,
    )


    normal_force = np.zeros(
        N,
        dtype=np.float64,
    )


    anchor_active = np.zeros(
        N,
        dtype=bool,
    )


    expected_support_active = np.zeros(
        N,
        dtype=bool,
    )


    force6 = np.zeros(
        6,
        dtype=np.float64,
    )


    for frame in range(N):

        data.qpos[:] = (
            qpos[
                frame
            ]
        )


        data.qvel[:] = (
            qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            data,
        )


        # Restore trajectory-consistent acceleration.
        data.qacc[:] = (
            qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            data,
        )


        (
            left_active,
            right_active,
        ) = contact_state(
            data
        )


        if anchors[
            frame
        ] == 0:

            anchor_active[
                frame
            ] = (
                left_active
            )

        else:

            anchor_active[
                frame
            ] = (
                right_active
            )


        expected = support_mask(
            frame
        )


        all_expected = True


        if expected[0]:
            all_expected &= (
                left_active
            )


        if expected[1]:
            all_expected &= (
                right_active
            )


        expected_support_active[
            frame
        ] = (
            all_expected
        )


        root_force[
            frame
        ] = (
            data.qfrc_inverse[
                0:3
            ]
        )


        root_torque[
            frame
        ] = (
            data.qfrc_inverse[
                3:6
            ]
        )


        for j, vadr in enumerate(
            env.vaddrs
        ):

            joint_tau[
                frame,
                j
            ] = (
                data.qfrc_inverse[
                    vadr
                ]
            )


        for cid in range(
            data.ncon
        ):

            con = data.contact[
                cid
            ]


            if int(
                con.efc_address
            ) < 0:

                continue


            g1 = int(
                con.geom1
            )

            g2 = int(
                con.geom2
            )


            if env.floor_geom not in (
                g1,
                g2,
            ):

                continue


            other = (
                g2
                if g1
                == env.floor_geom
                else g1
            )


            if (
                other not in LEFT
                and
                other not in RIGHT
            ):

                continue


            force6[:] = 0.0


            mujoco.mj_contactForce(
                model,
                data,
                cid,
                force6,
            )


            normal_force[
                frame
            ] += abs(
                float(
                    force6[
                        0
                    ]
                )
            )


    force_bw = (
        np.linalg.norm(
            root_force,
            axis=1,
        )
        /
        BODY_WEIGHT
    )


    torque_ratio = (
        np.abs(
            joint_tau
        )
        /
        effort[
            None,
            :
        ]
    )


    max_torque_ratio = (
        np.max(
            torque_ratio,
            axis=1,
        )
    )


    normal_bw = (
        normal_force
        /
        BODY_WEIGHT
    )


    velocity_change = (
        qvel
        -
        np.asarray(
            env.ref_full_qvel,
            dtype=np.float64,
        )
    )


    velocity_delta_rms = (
        np.sqrt(
            np.mean(
                velocity_change ** 2,
                axis=1,
            )
        )
    )


    shift_step = np.zeros(
        N,
        dtype=np.float64,
    )


    shift_step[1:] = (
        np.diff(
            shifts
        )
    )


    return {

        "depth":
            depth,

        "qpos":
            qpos,

        "qvel":
            qvel,

        "qacc":
            qacc,

        "shift":
            shifts,

        "shift_step":
            shift_step,

        "anchor_active":
            anchor_active,

        "expected_active":
            expected_support_active,

        "force_bw":
            force_bw,

        "normal_bw":
            normal_bw,

        "max_torque_ratio":
            max_torque_ratio,

        "velocity_delta_rms":
            velocity_delta_rms,

        "root_vertical_velocity":
            qvel[
                :,
                2
            ],

        "root_vertical_accel":
            qacc[
                :,
                2
            ],
    }


# ================================================================
# RUN ALL DEPTHS
# ================================================================

results = []


for depth in DEPTHS:

    print()
    print(
        "Running depth:",
        f"{1000*depth:.3f} mm",
    )


    result = audit(
        depth
    )


    results.append(
        result
    )


# ================================================================
# REPORT
# ================================================================

def p(
    x,
    value,
):

    return float(
        np.percentile(
            x,
            value,
        )
    )


print()
print("=" * 195)
print("CONSISTENT CONTACT-SNAPPED DEPTH SUMMARY")
print("=" * 195)

print(
    f"{'DEPTH':>9s} "
    f"{'ANCH':>7s} "
    f"{'ALLSUP':>7s} "
    f"{'ROOT50':>8s} "
    f"{'ROOT95':>8s} "
    f"{'ROOTMAX':>8s} "
    f"{'FN95':>8s} "
    f"{'TAU95':>8s} "
    f"{'>LIMIT':>8s} "
    f"{'SINGLE':>8s} "
    f"{'DOUBLE':>8s} "
    f"{'TRANS':>8s}"
)

print("-" * 195)


for r in results:

    force = r[
        "force_bw"
    ]


    tau = r[
        "max_torque_ratio"
    ]


    def safe_p95(mask):

        if not np.any(mask):
            return float("nan")

        return p(
            force[
                mask
            ],
            95,
        )


    print(
        f"{1000*r['depth']:8.3f}m "
        f"{100*np.mean(r['anchor_active']):6.1f}% "
        f"{100*np.mean(r['expected_active']):6.1f}% "
        f"{p(force,50):8.3f} "
        f"{p(force,95):8.3f} "
        f"{np.max(force):8.3f} "
        f"{p(r['normal_bw'],95):8.3f} "
        f"{p(tau,95):8.3f} "
        f"{100*np.mean(tau > 1.0):7.1f}% "
        f"{safe_p95(steady_single):8.3f} "
        f"{safe_p95(steady_double):8.3f} "
        f"{safe_p95(transition):8.3f}"
    )


# ================================================================
# KINEMATIC CONSISTENCY REPORT
# ================================================================

reference_result = results[
    0
]


print()
print("=" * 195)
print("SNAPPING-INDUCED KINEMATIC CHANGE — 1 MICRON")
print("=" * 195)


shift_mm = (
    -1000.0
    * reference_result[
        "shift"
    ]
)


step_mm = (
    1000.0
    * np.abs(
        reference_result[
            "shift_step"
        ]
    )
)


print(
    "root lowering:",
    f"p50={p(shift_mm,50):.2f} mm",
    f"p95={p(shift_mm,95):.2f} mm",
    f"max={np.max(shift_mm):.2f} mm",
)


print(
    "frame-to-frame shift change:",
    f"p50={p(step_mm,50):.2f} mm",
    f"p95={p(step_mm,95):.2f} mm",
    f"max={np.max(step_mm):.2f} mm",
)


print(
    "qvel change RMS:",
    f"p50={p(reference_result['velocity_delta_rms'],50):.4f}",
    f"p95={p(reference_result['velocity_delta_rms'],95):.4f}",
    f"max={np.max(reference_result['velocity_delta_rms']):.4f}",
)


print(
    "root |vz|:",
    f"p95={p(np.abs(reference_result['root_vertical_velocity']),95):.3f} m/s",
    f"max={np.max(np.abs(reference_result['root_vertical_velocity'])):.3f} m/s",
)


print(
    "root |az|:",
    f"p95={p(np.abs(reference_result['root_vertical_accel']),95):.2f} m/s^2",
    f"max={np.max(np.abs(reference_result['root_vertical_accel'])):.2f} m/s^2",
)


# ================================================================
# DEPTH SENSITIVITY
# ================================================================

small = results[
    0
]

large = results[
    -1
]


small_p95 = p(
    small[
        "force_bw"
    ],
    95,
)


large_p95 = p(
    large[
        "force_bw"
    ],
    95,
)


sensitivity_ratio = (
    large_p95
    /
    max(
        small_p95,
        1e-9,
    )
)


single_p95 = p(
    small[
        "force_bw"
    ][
        steady_single
    ],
    95,
) if np.any(
    steady_single
) else float("nan")


transition_p95 = p(
    small[
        "force_bw"
    ][
        transition
    ],
    95,
) if np.any(
    transition
) else float("nan")


tau_p95 = p(
    small[
        "max_torque_ratio"
    ],
    95,
)


overlimit = float(
    np.mean(
        small[
            "max_torque_ratio"
        ]
        > 1.0
    )
)


anchor_validity = float(
    np.mean(
        small[
            "anchor_active"
        ]
    )
)


print()
print("=" * 195)
print("CONSISTENT INVERSE-DYNAMICS DECISION")
print("=" * 195)

print(
    "1-micron anchor validity:",
    f"{100*anchor_validity:.1f}%",
)

print(
    "1-micron root residual p95:",
    f"{small_p95:.3f} BW",
)

print(
    "100um / 1um residual ratio:",
    f"{sensitivity_ratio:.2f}x",
)

print(
    "steady single-support p95:",
    f"{single_p95:.3f} BW",
)

print(
    "transition p95:",
    f"{transition_p95:.3f} BW",
)

print(
    "torque p95/limit:",
    f"{tau_p95:.3f}",
)

print(
    "over-limit frames:",
    f"{100*overlimit:.1f}%",
)


print()


if anchor_validity < 0.95:

    print(
        "RESULT: CONTACT VALIDITY FAILED"
    )

    print(
        "Do not interpret the residual."
    )


elif (
    sensitivity_ratio >= 2.0
):

    print(
        "RESULT: PREVIOUS HUGE RESIDUAL WAS "
        "STRONGLY PENETRATION-SENSITIVE"
    )

    print(
        "Contact stabilization materially contaminated "
        "the previous 0.1-mm inverse-dynamics result."
    )

    print(
        "Use the 1-micron result for diagnosis."
    )


elif (
    small_p95 >= 0.30
    and
    single_p95 >= 0.25
):

    print(
        "RESULT: DYNAMIC INCONSISTENCY IS NOW "
        "STRONGLY SUPPORTED"
    )

    print(
        "Large residual remains even with:"
    )

    print(
        "  - consistent qpos/qvel/qacc"
    )

    print(
        "  - 1-micron penetration"
    )

    print(
        "  - active support contact"
    )

    print(
        "  - steady single-support frames"
    )

    print(
        "NEXT: dynamically project / optimize "
        "the reference before PPO."
    )


elif (
    transition_p95
    >
    max(
        0.25,
        1.5 * single_p95,
    )
):

    print(
        "RESULT: SUPPORT TRANSITIONS ARE "
        "THE PRIMARY DYNAMIC BOTTLENECK"
    )

    print(
        "Steady single stance is much cleaner than "
        "contact-change phases."
    )

    print(
        "NEXT: repair transition timing / "
        "double-support blending."
    )


elif (
    small_p95 <= 0.15
    and
    tau_p95 <= 0.80
):

    print(
        "RESULT: REFERENCE FEASIBILITY PASSES "
        "THE CONSISTENT AUDIT"
    )

    print(
        "The earlier extreme residuals were "
        "diagnostic artifacts."
    )

    print(
        "NEXT: final controller validation, "
        "then small PPO pilot."
    )


else:

    print(
        "RESULT: FEASIBILITY REMAINS MARGINAL"
    )

    print(
        "Inspect support-type residuals before PPO."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_contact_snapped_consistent.npz"
)


np.savez(
    output,

    depths=
        np.asarray(
            DEPTHS,
            dtype=np.float64,
        ),

    one_micron_qpos=
        small[
            "qpos"
        ].astype(
            np.float32
        ),

    one_micron_qvel=
        small[
            "qvel"
        ].astype(
            np.float32
        ),

    one_micron_qacc=
        small[
            "qacc"
        ].astype(
            np.float32
        ),

    one_micron_force_bw=
        small[
            "force_bw"
        ].astype(
            np.float32
        ),

    transition=
        transition,

    steady_single=
        steady_single,

    steady_double=
        steady_double,
)


print()
print(
    "artifact:",
    output,
)

print()
print(
    "ORIGINAL REFERENCE FILE WAS NOT MODIFIED."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 195)


env.close()
