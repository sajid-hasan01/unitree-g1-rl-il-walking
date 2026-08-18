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


# =================================================================
# CONFIG
# =================================================================

RAMP_FRAMES = 4
MIN_SINGLE_SUPPORT_SEGMENT = 5

# Conservative bound for Pass 1.
MAX_XY_CORRECTION = 0.080       # 8 cm vector magnitude

# Inverse-dynamics audit contact depth.
CONTACT_DEPTH = 1e-6            # 0.001 mm = 1 micron

TRANSITION_WINDOW = 2


OUTPUT_REFERENCE = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_support_locked_candidate.npz"
)


OUTPUT_DIAGNOSTIC = (
    ROOT
    / "results"
    / "g1_dynamic_v2_support_lock_pass1.npz"
)


# =================================================================
# ENVIRONMENT
# =================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model

N = env.num_frames
DT = 1.0 / env.reference_fps


old_qpos = np.asarray(
    env.ref_full_qpos,
    dtype=np.float64,
).copy()


old_qvel = np.asarray(
    env.ref_full_qvel,
    dtype=np.float64,
).copy()


support = (
    np.asarray(
        env.ref_support,
        dtype=np.float64,
    )
    > 0.5
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


TOTAL_MASS = float(
    mujoco.mj_getTotalmass(
        model
    )
)


GRAVITY = abs(
    float(
        model.opt.gravity[2]
    )
)


BODY_WEIGHT = (
    TOTAL_MASS
    * GRAVITY
)


LEFT_GEOMS = list(
    env.left_sole_geoms
)


RIGHT_GEOMS = list(
    env.right_sole_geoms
)


LEFT_SET = set(
    LEFT_GEOMS
)


RIGHT_SET = set(
    RIGHT_GEOMS
)


print("=" * 195)
print("G1 DYNAMIC V2 — SUPPORT-LOCK PROJECTION PASS 1")
print("ROOT X/Y ONLY")
print("29 JOINT ANGLES PRESERVED")
print("ROOT Z + ROOT ORIENTATION PRESERVED")
print("FOLLOWED BY CONSISTENT CONTACT-SNAPPED INVERSE AUDIT")
print("NO CEM / NO PPO")
print("=" * 195)

print(
    "source reference:",
    env.reference_path,
)

print(
    "candidate output:",
    OUTPUT_REFERENCE,
)

print(
    "frames:",
    N,
)

print(
    "reference frequency:",
    env.reference_fps,
)

print(
    "mass:",
    f"{TOTAL_MASS:.3f} kg",
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)


# =================================================================
# SUPPORT SEGMENTS
# =================================================================

support_count = np.sum(
    support,
    axis=1,
)


single_support = (
    support_count == 1
)


segments = []


frame = 0


while frame < N:

    if not single_support[frame]:

        frame += 1
        continue


    side = int(
        np.argmax(
            support[frame]
        )
    )


    start = frame


    frame += 1


    while (
        frame < N
        and
        single_support[frame]
        and
        int(
            np.argmax(
                support[frame]
            )
        ) == side
    ):

        frame += 1


    end = (
        frame - 1
    )


    length = (
        end - start + 1
    )


    if length >= MIN_SINGLE_SUPPORT_SEGMENT:

        segments.append(
            (
                start,
                end,
                side,
            )
        )


print()
print("=" * 195)
print("SINGLE-SUPPORT SEGMENTS")
print("=" * 195)

print(
    "segments:",
    len(
        segments
    ),
)


for i, (
    start,
    end,
    side,
) in enumerate(
    segments
):

    print(
        f"{i:02d}: "
        f"{'LEFT ' if side == 0 else 'RIGHT'} "
        f"frames {start:03d}..{end:03d} "
        f"length={end-start+1}"
    )


# =================================================================
# FOOT REPRESENTATIVE POINT
#
# Use mean XYZ of the four sole sphere centers.
# Since only root translation changes, this point translates
# one-to-one with the root correction.
# =================================================================

geom_data = mujoco.MjData(
    model
)


def sole_center(
    data,
    geoms,
):

    return np.mean(
        data.geom_xpos[
            geoms
        ],
        axis=0,
    ).copy()


def compute_sole_trajectories(
    qpos_trajectory,
):

    left = np.zeros(
        (
            N,
            3,
        ),
        dtype=np.float64,
    )


    right = np.zeros(
        (
            N,
            3,
        ),
        dtype=np.float64,
    )


    for frame in range(N):

        geom_data.qpos[:] = (
            qpos_trajectory[
                frame
            ]
        )


        geom_data.qvel[:] = 0.0


        mujoco.mj_forward(
            model,
            geom_data,
        )


        left[
            frame
        ] = sole_center(
            geom_data,
            LEFT_GEOMS,
        )


        right[
            frame
        ] = sole_center(
            geom_data,
            RIGHT_GEOMS,
        )


    return (
        left,
        right,
    )


print()
print(
    "Computing original sole trajectories..."
)


(
    old_left,
    old_right,
) = compute_sole_trajectories(
    old_qpos
)


# =================================================================
# SMOOTH SUPPORT WINDOW
# =================================================================

def smoothstep(
    x,
):

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )


    return (
        x
        * x
        * (
            3.0
            -
            2.0
            * x
        )
    )


def segment_window(
    length,
):

    w = np.zeros(
        length,
        dtype=np.float64,
    )


    if length <= 1:

        return w


    for i in range(
        length
    ):

        if RAMP_FRAMES <= 0:

            alpha = 1.0

        else:

            distance_from_edge = min(
                i,
                length - 1 - i,
            )


            alpha = min(
                1.0,
                distance_from_edge
                / float(
                    RAMP_FRAMES
                ),
            )


        w[
            i
        ] = smoothstep(
            alpha
        )


    return w


# =================================================================
# BUILD ROOT X/Y CORRECTION
# =================================================================

xy_correction = np.zeros(
    (
        N,
        2,
    ),
    dtype=np.float64,
)


lock_weight = np.zeros(
    N,
    dtype=np.float64,
)


clipped_frames = 0


print()
print("=" * 195)
print("SUPPORT-LOCK RAW PROJECTION")
print("=" * 195)


for seg_index, (
    start,
    end,
    side,
) in enumerate(
    segments
):

    indices = np.arange(
        start,
        end + 1,
        dtype=np.int32,
    )


    foot = (
        old_left
        if side == 0
        else old_right
    )


    foot_xy = (
        foot[
            indices,
            0:2
        ]
    )


    # -------------------------------------------------------------
    # World-space target for the stance foot.
    #
    # Median minimizes sensitivity to the segment boundaries and
    # keeps root correction balanced around zero.
    # -------------------------------------------------------------

    target_xy = np.median(
        foot_xy,
        axis=0,
    )


    raw = (
        target_xy[
            None,
            :
        ]
        -
        foot_xy
    )


    # Conservative vector-magnitude clipping.
    norms = np.linalg.norm(
        raw,
        axis=1,
    )


    over = (
        norms
        >
        MAX_XY_CORRECTION
    )


    if np.any(
        over
    ):

        scales = (
            MAX_XY_CORRECTION
            /
            np.maximum(
                norms[
                    over
                ],
                1e-12,
            )
        )


        raw[
            over
        ] *= scales[
            :,
            None
        ]


        clipped_frames += int(
            np.sum(
                over
            )
        )


    window = segment_window(
        len(
            indices
        )
    )


    correction = (
        raw
        *
        window[
            :,
            None
        ]
    )


    xy_correction[
        indices
    ] = (
        correction
    )


    lock_weight[
        indices
    ] = (
        window
    )


    raw_magnitude = (
        1000.0
        * np.linalg.norm(
            raw,
            axis=1,
        )
    )


    applied_magnitude = (
        1000.0
        * np.linalg.norm(
            correction,
            axis=1,
        )
    )


    original_spread = (
        1000.0
        * np.sqrt(
            np.mean(
                np.sum(
                    (
                        foot_xy
                        -
                        target_xy[
                            None,
                            :
                        ]
                    ) ** 2,
                    axis=1,
                )
            )
        )
    )


    print(
        f"segment={seg_index:02d} "
        f"side={'L' if side == 0 else 'R'} "
        f"frames={start:03d}-{end:03d} "
        f"footSpreadRMS="
        f"{original_spread:6.2f}mm "
        f"rawP95="
        f"{np.percentile(raw_magnitude,95):6.2f}mm "
        f"appliedMax="
        f"{np.max(applied_magnitude):6.2f}mm"
    )


# =================================================================
# CREATE CANDIDATE QPOS
# =================================================================

new_qpos = old_qpos.copy()


new_qpos[
    :,
    0:2
] += (
    xy_correction
)


# Explicit invariants.
joint_difference = float(
    np.max(
        np.abs(
            new_qpos[
                :,
                7:
            ]
            -
            old_qpos[
                :,
                7:
            ]
        )
    )
)


root_z_difference = float(
    np.max(
        np.abs(
            new_qpos[
                :,
                2
            ]
            -
            old_qpos[
                :,
                2
            ]
        )
    )
)


root_quat_difference = float(
    np.max(
        np.abs(
            new_qpos[
                :,
                3:7
            ]
            -
            old_qpos[
                :,
                3:7
            ]
        )
    )
)


if joint_difference > 1e-12:

    raise RuntimeError(
        "Pass 1 unexpectedly changed joint angles."
    )


if root_z_difference > 1e-12:

    raise RuntimeError(
        "Pass 1 unexpectedly changed root Z."
    )


if root_quat_difference > 1e-12:

    raise RuntimeError(
        "Pass 1 unexpectedly changed root orientation."
    )


# =================================================================
# CORRECT QVEL FROM CANDIDATE QPOS
# =================================================================

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


    mujoco.mj_differentiatePos(
        model,
        qvel[0],
        DT,
        qpos[0],
        qpos[1],
    )


    for frame in range(
        1,
        N - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            qvel[
                frame
            ],
            2.0 * DT,
            qpos[
                frame - 1
            ],
            qpos[
                frame + 1
            ],
        )


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


    qacc[
        1:-1
    ] = (
        qvel[
            2:
        ]
        -
        qvel[
            :-2
        ]
    ) / (
        2.0
        * DT
    )


    qacc[
        0
    ] = (
        qvel[
            1
        ]
        -
        qvel[
            0
        ]
    ) / DT


    qacc[
        -1
    ] = (
        qvel[
            -1
        ]
        -
        qvel[
            -2
        ]
    ) / DT


    return qacc


new_qvel = differentiate_qpos(
    new_qpos
)


# =================================================================
# FOOT SLIP VALIDATION
# =================================================================

print()
print(
    "Computing candidate sole trajectories..."
)


(
    new_left,
    new_right,
) = compute_sole_trajectories(
    new_qpos
)


def collect_stance_speeds(
    left,
    right,
    core_only,
):

    speeds = []


    for (
        start,
        end,
        side,
    ) in segments:

        foot = (
            left
            if side == 0
            else right
        )


        if core_only:

            local_start = min(
                end,
                start + RAMP_FRAMES,
            )


            local_end = max(
                local_start,
                end - RAMP_FRAMES,
            )

        else:

            local_start = (
                start
            )

            local_end = (
                end
            )


        for frame in range(
            local_start,
            local_end,
        ):

            velocity = (
                foot[
                    frame + 1,
                    0:2
                ]
                -
                foot[
                    frame,
                    0:2
                ]
            ) / DT


            speeds.append(
                float(
                    np.linalg.norm(
                        velocity
                    )
                )
            )


    return np.asarray(
        speeds,
        dtype=np.float64,
    )


old_speed_all = collect_stance_speeds(
    old_left,
    old_right,
    core_only=False,
)


new_speed_all = collect_stance_speeds(
    new_left,
    new_right,
    core_only=False,
)


old_speed_core = collect_stance_speeds(
    old_left,
    old_right,
    core_only=True,
)


new_speed_core = collect_stance_speeds(
    new_left,
    new_right,
    core_only=True,
)


# =================================================================
# ROOT-CORRECTION METRICS
# =================================================================

corr_mag = np.linalg.norm(
    xy_correction,
    axis=1,
)


corr_step = np.zeros(
    N,
    dtype=np.float64,
)


corr_step[
    1:
] = np.linalg.norm(
    np.diff(
        xy_correction,
        axis=0,
    ),
    axis=1,
)


print()
print("=" * 195)
print("SUPPORT-LOCK GEOMETRIC RESULT")
print("=" * 195)

print(
    "joint-angle maximum change:",
    f"{joint_difference:.3e} rad",
)

print(
    "root-Z maximum change:",
    f"{1000*root_z_difference:.6f} mm",
)

print(
    "root-quaternion maximum change:",
    f"{root_quat_difference:.3e}",
)

print()
print(
    "XY root correction:",
    f"p50={1000*np.percentile(corr_mag,50):.2f} mm",
    f"p95={1000*np.percentile(corr_mag,95):.2f} mm",
    f"max={1000*np.max(corr_mag):.2f} mm",
)

print(
    "frame-to-frame XY correction change:",
    f"p95={1000*np.percentile(corr_step,95):.2f} mm",
    f"max={1000*np.max(corr_step):.2f} mm",
)

print(
    "correction-clipped frames:",
    clipped_frames,
    "/",
    N,
)

print()
print(
    "STANCE FOOT SPEED — ALL SINGLE SUPPORT"
)

print(
    "original:",
    f"p50={np.percentile(old_speed_all,50):.4f} m/s",
    f"p95={np.percentile(old_speed_all,95):.4f} m/s",
)

print(
    "candidate:",
    f"p50={np.percentile(new_speed_all,50):.4f} m/s",
    f"p95={np.percentile(new_speed_all,95):.4f} m/s",
)


print()
print(
    "STANCE FOOT SPEED — CORE SINGLE SUPPORT"
)

print(
    "original:",
    f"p50={np.percentile(old_speed_core,50):.4f} m/s",
    f"p95={np.percentile(old_speed_core,95):.4f} m/s",
)

print(
    "candidate:",
    f"p50={np.percentile(new_speed_core,50):.4f} m/s",
    f"p95={np.percentile(new_speed_core,95):.4f} m/s",
)


# =================================================================
# TRANSITION MASK
# =================================================================

transition = np.zeros(
    N,
    dtype=bool,
)


for frame in range(
    1,
    N,
):

    if not np.array_equal(
        support[
            frame
        ],
        support[
            frame - 1
        ],
    ):

        lo = max(
            0,
            frame - TRANSITION_WINDOW,
        )


        hi = min(
            N,
            frame
            + TRANSITION_WINDOW
            + 1,
        )


        transition[
            lo:hi
        ] = True


steady_single = (
    single_support
    &
    ~transition
)


double_support = (
    support_count >= 2
)


steady_double = (
    double_support
    &
    ~transition
)


# =================================================================
# CONTACT-SNAPPED CONSISTENT INVERSE DYNAMICS
# =================================================================

audit_probe = mujoco.MjData(
    model
)


def fallback_support_mask(
    frame,
):

    expected = (
        support[
            frame
        ].copy()
    )


    if not np.any(
        expected
    ):

        expected = (
            np.asarray(
                env.ref_contact[
                    frame
                ],
                dtype=np.float64,
            )
            > 0.5
        )


    return expected


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


    minimum = float(
        "inf"
    )


    for gid in geoms:

        radius = float(
            model.geom_size[
                gid,
                0
            ]
        )


        bottom = float(
            data.geom_xpos[
                gid,
                2
            ]
            -
            radius
        )


        minimum = min(
            minimum,
            bottom - floor_z,
        )


    return minimum


def choose_anchor(
    qpos_trajectory,
    frame,
):

    audit_probe.qpos[:] = (
        qpos_trajectory[
            frame
        ]
    )


    audit_probe.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        audit_probe,
    )


    left_clearance = sole_clearance(
        audit_probe,
        LEFT_GEOMS,
    )


    right_clearance = sole_clearance(
        audit_probe,
        RIGHT_GEOMS,
    )


    expected = fallback_support_mask(
        frame
    )


    candidates = []


    if expected[
        0
    ]:

        candidates.append(
            (
                0,
                left_clearance,
            )
        )


    if expected[
        1
    ]:

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


def build_contact_snapped(
    qpos_trajectory,
):

    snapped = (
        qpos_trajectory.copy()
    )


    z_shift = np.zeros(
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
            qpos_trajectory,
            frame,
        )


        anchors[
            frame
        ] = anchor


        dz = (
            -CONTACT_DEPTH
            - clearance
        )


        # Lower only.
        dz = min(
            0.0,
            dz,
        )


        # Defensive bound.
        dz = max(
            -0.020,
            dz,
        )


        snapped[
            frame,
            2
        ] += dz


        z_shift[
            frame
        ] = dz


    return (
        snapped,
        z_shift,
        anchors,
    )


def active_contacts(
    data,
):

    left = False
    right = False


    for cid in range(
        data.ncon
    ):

        contact = (
            data.contact[
                cid
            ]
        )


        if int(
            contact.efc_address
        ) < 0:

            continue


        g1 = int(
            contact.geom1
        )


        g2 = int(
            contact.geom2
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


        if other in LEFT_SET:

            left = True


        elif other in RIGHT_SET:

            right = True


    return (
        left,
        right,
    )


def inverse_audit(
    name,
    qpos_trajectory,
):

    print()
    print(
        "Running consistent inverse audit:",
        name,
    )


    (
        snapped_qpos,
        z_shift,
        anchors,
    ) = build_contact_snapped(
        qpos_trajectory
    )


    snapped_qvel = differentiate_qpos(
        snapped_qpos
    )


    snapped_qacc = differentiate_velocity(
        snapped_qvel
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


    joint_tau = np.zeros(
        (
            N,
            29,
        ),
        dtype=np.float64,
    )


    anchor_active = np.zeros(
        N,
        dtype=bool,
    )


    for frame in range(N):

        data.qpos[:] = (
            snapped_qpos[
                frame
            ]
        )


        data.qvel[:] = (
            snapped_qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            data,
        )


        data.qacc[:] = (
            snapped_qacc[
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
        ) = active_contacts(
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


        root_force[
            frame
        ] = (
            data.qfrc_inverse[
                0:3
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


    root_bw = (
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


    max_torque_ratio = np.max(
        torque_ratio,
        axis=1,
    )


    def safe_p95(
        mask,
    ):

        if not np.any(
            mask
        ):

            return float(
                "nan"
            )


        return float(
            np.percentile(
                root_bw[
                    mask
                ],
                95,
            )
        )


    result = {

        "name":
            name,

        "root_bw":
            root_bw,

        "torque_ratio":
            max_torque_ratio,

        "anchor_validity":
            float(
                np.mean(
                    anchor_active
                )
            ),

        "overall_p50":
            float(
                np.percentile(
                    root_bw,
                    50,
                )
            ),

        "overall_p95":
            float(
                np.percentile(
                    root_bw,
                    95,
                )
            ),

        "overall_max":
            float(
                np.max(
                    root_bw
                )
            ),

        "single_p95":
            safe_p95(
                steady_single
            ),

        "double_p95":
            safe_p95(
                steady_double
            ),

        "transition_p95":
            safe_p95(
                transition
            ),

        "tau_p95":
            float(
                np.percentile(
                    max_torque_ratio,
                    95,
                )
            ),

        "overlimit":
            float(
                np.mean(
                    max_torque_ratio
                    > 1.0
                )
            ),

        "z_shift":
            z_shift,

        "qvel":
            snapped_qvel,

        "qacc":
            snapped_qacc,
    }


    return result


baseline_audit = inverse_audit(
    "ORIGINAL",
    old_qpos,
)


candidate_audit = inverse_audit(
    "SUPPORT_LOCKED",
    new_qpos,
)


# =================================================================
# AUDIT REPORT
# =================================================================

print()
print("=" * 195)
print("ORIGINAL VS SUPPORT-LOCKED DYNAMIC FEASIBILITY")
print("=" * 195)

print(
    f"{'REFERENCE':16s} "
    f"{'ANCH':>7s} "
    f"{'ROOT50':>9s} "
    f"{'ROOT95':>9s} "
    f"{'ROOTMAX':>9s} "
    f"{'SINGLE':>9s} "
    f"{'DOUBLE':>9s} "
    f"{'TRANS':>9s} "
    f"{'TAU95':>9s} "
    f"{'>LIMIT':>9s}"
)

print("-" * 195)


for r in (
    baseline_audit,
    candidate_audit,
):

    print(
        f"{r['name']:16s} "
        f"{100*r['anchor_validity']:6.1f}% "
        f"{r['overall_p50']:9.3f} "
        f"{r['overall_p95']:9.3f} "
        f"{r['overall_max']:9.3f} "
        f"{r['single_p95']:9.3f} "
        f"{r['double_p95']:9.3f} "
        f"{r['transition_p95']:9.3f} "
        f"{r['tau_p95']:9.3f} "
        f"{100*r['overlimit']:8.1f}%"
    )


# =================================================================
# IMPROVEMENT
# =================================================================

def fractional_reduction(
    before,
    after,
):

    return (
        before
        -
        after
    ) / max(
        abs(
            before
        ),
        1e-12,
    )


overall_reduction = (
    fractional_reduction(
        baseline_audit[
            "overall_p95"
        ],
        candidate_audit[
            "overall_p95"
        ],
    )
)


single_reduction = (
    fractional_reduction(
        baseline_audit[
            "single_p95"
        ],
        candidate_audit[
            "single_p95"
        ],
    )
)


transition_reduction = (
    fractional_reduction(
        baseline_audit[
            "transition_p95"
        ],
        candidate_audit[
            "transition_p95"
        ],
    )
)


tau_reduction = (
    fractional_reduction(
        baseline_audit[
            "tau_p95"
        ],
        candidate_audit[
            "tau_p95"
        ],
    )
)


print()
print("=" * 195)
print("SUPPORT-LOCK PASS-1 DECISION")
print("=" * 195)

print(
    "baseline anchor validity:",
    f"{100*baseline_audit['anchor_validity']:.1f}%",
)

print(
    "candidate anchor validity:",
    f"{100*candidate_audit['anchor_validity']:.1f}%",
)


print()
print(
    "overall p95:",
    f"{baseline_audit['overall_p95']:.3f}",
    "->",
    f"{candidate_audit['overall_p95']:.3f}",
    "BW",
    f"({100*overall_reduction:+.1f}%)",
)


print(
    "steady single p95:",
    f"{baseline_audit['single_p95']:.3f}",
    "->",
    f"{candidate_audit['single_p95']:.3f}",
    "BW",
    f"({100*single_reduction:+.1f}%)",
)


print(
    "transition p95:",
    f"{baseline_audit['transition_p95']:.3f}",
    "->",
    f"{candidate_audit['transition_p95']:.3f}",
    "BW",
    f"({100*transition_reduction:+.1f}%)",
)


print(
    "torque p95/limit:",
    f"{baseline_audit['tau_p95']:.3f}",
    "->",
    f"{candidate_audit['tau_p95']:.3f}",
    f"({100*tau_reduction:+.1f}%)",
)


print(
    "over-limit frames:",
    f"{100*baseline_audit['overlimit']:.1f}%",
    "->",
    f"{100*candidate_audit['overlimit']:.1f}%",
)


print()
print(
    "root correction p95:",
    f"{1000*np.percentile(corr_mag,95):.2f} mm",
)

print(
    "root correction max:",
    f"{1000*np.max(corr_mag):.2f} mm",
)

print(
    "core stance speed p95:",
    f"{np.percentile(old_speed_core,95):.4f}",
    "->",
    f"{np.percentile(new_speed_core,95):.4f}",
    "m/s",
)


# =================================================================
# SAVE COMPATIBLE NPZ
#
# Preserve original schema and replace arrays by matching actual
# values/shapes from the environment.
# =================================================================

reference_path = Path(
    env.reference_path
)


with np.load(
    reference_path,
    allow_pickle=True,
) as source:

    payload = {
        key:
            source[
                key
            ].copy()

        for key in source.files
    }


def numeric_array(
    value,
):

    return (
        isinstance(
            value,
            np.ndarray,
        )
        and
        np.issubdtype(
            value.dtype,
            np.number,
        )
    )


def matches(
    value,
    target,
):

    if not numeric_array(
        value
    ):

        return False


    if value.shape != target.shape:

        return False


    return bool(
        np.allclose(
            value,
            target,
            rtol=1e-5,
            atol=5e-6,
            equal_nan=True,
        )
    )


patched_qpos = []
patched_qvel = []
patched_root_pos = []
patched_root_linvel = []


old_root_pos = (
    old_qpos[
        :,
        0:3
    ]
)


new_root_pos = (
    new_qpos[
        :,
        0:3
    ]
)


old_root_linvel = (
    old_qvel[
        :,
        0:3
    ]
)


new_root_linvel = (
    new_qvel[
        :,
        0:3
    ]
)


for key in list(
    payload.keys()
):

    value = payload[
        key
    ]


    if matches(
        value,
        old_qpos,
    ):

        payload[
            key
        ] = (
            new_qpos.astype(
                value.dtype
            )
        )

        patched_qpos.append(
            key
        )

        continue


    if matches(
        value,
        old_qvel,
    ):

        payload[
            key
        ] = (
            new_qvel.astype(
                value.dtype
            )
        )

        patched_qvel.append(
            key
        )

        continue


    if matches(
        value,
        old_root_pos,
    ):

        payload[
            key
        ] = (
            new_root_pos.astype(
                value.dtype
            )
        )

        patched_root_pos.append(
            key
        )

        continue


    if matches(
        value,
        old_root_linvel,
    ):

        payload[
            key
        ] = (
            new_root_linvel.astype(
                value.dtype
            )
        )

        patched_root_linvel.append(
            key
        )


# -------------------------------------------------------------
# Defensive fallback for a uniquely-shaped full qpos/qvel key.
# -------------------------------------------------------------

if not patched_qpos:

    candidates = [
        key
        for key, value in payload.items()

        if (
            numeric_array(
                value
            )
            and
            value.shape
            == old_qpos.shape
        )
    ]


    if len(
        candidates
    ) == 1:

        key = candidates[
            0
        ]


        payload[
            key
        ] = (
            new_qpos.astype(
                payload[
                    key
                ].dtype
            )
        )


        patched_qpos.append(
            key
        )


if not patched_qvel:

    candidates = [
        key
        for key, value in payload.items()

        if (
            numeric_array(
                value
            )
            and
            value.shape
            == old_qvel.shape
        )
    ]


    if len(
        candidates
    ) == 1:

        key = candidates[
            0
        ]


        payload[
            key
        ] = (
            new_qvel.astype(
                payload[
                    key
                ].dtype
            )
        )


        patched_qvel.append(
            key
        )


if not patched_qpos:

    print()
    print(
        "AVAILABLE NPZ KEYS / SHAPES:"
    )


    for key, value in payload.items():

        print(
            key,
            getattr(
                value,
                "shape",
                None,
            ),
            getattr(
                value,
                "dtype",
                None,
            ),
        )


    raise RuntimeError(
        "Could not safely identify the source "
        "full-qpos array in the NPZ. "
        "Candidate was NOT saved."
    )


# Add diagnostic metadata. Environment may simply ignore these.
payload[
    "support_lock_xy_correction_m"
] = (
    xy_correction.astype(
        np.float32
    )
)


payload[
    "support_lock_weight"
] = (
    lock_weight.astype(
        np.float32
    )
)


payload[
    "support_lock_version"
] = np.asarray(
    "dynamic_v2_pass1_root_xy",
)


np.savez_compressed(
    OUTPUT_REFERENCE,
    **payload,
)


print()
print("=" * 195)
print("REFERENCE FILE SCHEMA PATCH")
print("=" * 195)

print(
    "patched full-qpos keys:",
    patched_qpos,
)

print(
    "patched full-qvel keys:",
    patched_qvel,
)

print(
    "patched root-position keys:",
    patched_root_pos,
)

print(
    "patched root-linear-velocity keys:",
    patched_root_linvel,
)


# =================================================================
# SAVE FULL DIAGNOSTIC
# =================================================================

np.savez_compressed(
    OUTPUT_DIAGNOSTIC,

    old_qpos=
        old_qpos.astype(
            np.float32
        ),

    new_qpos=
        new_qpos.astype(
            np.float32
        ),

    new_qvel=
        new_qvel.astype(
            np.float32
        ),

    xy_correction=
        xy_correction.astype(
            np.float32
        ),

    lock_weight=
        lock_weight.astype(
            np.float32
        ),

    baseline_root_bw=
        baseline_audit[
            "root_bw"
        ].astype(
            np.float32
        ),

    candidate_root_bw=
        candidate_audit[
            "root_bw"
        ].astype(
            np.float32
        ),

    baseline_tau_ratio=
        baseline_audit[
            "torque_ratio"
        ].astype(
            np.float32
        ),

    candidate_tau_ratio=
        candidate_audit[
            "torque_ratio"
        ].astype(
            np.float32
        ),

    overall_reduction=
        np.asarray(
            [
                overall_reduction
            ],
            dtype=np.float32,
        ),

    single_reduction=
        np.asarray(
            [
                single_reduction
            ],
            dtype=np.float32,
        ),

    transition_reduction=
        np.asarray(
            [
                transition_reduction
            ],
            dtype=np.float32,
        ),
)


# =================================================================
# FINAL DECISION
# =================================================================

print()
print("=" * 195)
print("FINAL PASS-1 VERDICT")
print("=" * 195)


max_correction = float(
    np.max(
        corr_mag
    )
)


if (
    candidate_audit[
        "anchor_validity"
    ] < 0.95
):

    print(
        "RESULT: AUDIT CONTACT VALIDITY FAILED"
    )

    print(
        "Do not interpret the dynamic result."
    )


elif (
    max_correction
    >= MAX_XY_CORRECTION
    * 0.999

    and
    clipped_frames
    > 0
):

    print(
        "NOTE: ROOT-XY CORRECTION HIT THE 8-CM "
        "CONSERVATIVE LIMIT."
    )


if (
    candidate_audit[
        "anchor_validity"
    ] >= 0.95

    and
    single_reduction
    >= 0.50

    and
    candidate_audit[
        "single_p95"
    ] <= 6.5

    and
    candidate_audit[
        "overall_p95"
    ]
    <
    baseline_audit[
        "overall_p95"
    ]

    and
    candidate_audit[
        "overlimit"
    ]
    <
    baseline_audit[
        "overlimit"
    ]
):

    print(
        "RESULT: SUPPORT-LOCK PASS 1 IS "
        "DYNAMICALLY USEFUL"
    )

    print(
        "Stance-foot world drift was a major part "
        "of the single-support inconsistency."
    )

    print(
        "KEEP this candidate for Dynamic V2."
    )

    print(
        "NEXT: repair support-transition blending "
        "while preserving this stance lock."
    )


elif (
    candidate_audit[
        "anchor_validity"
    ] >= 0.95

    and
    single_reduction
    >= 0.25

    and
    candidate_audit[
        "overall_p95"
    ]
    <=
    baseline_audit[
        "overall_p95"
    ]
    * 1.10
):

    print(
        "RESULT: SUPPORT-LOCK PASS 1 HELPS, "
        "BUT IS NOT SUFFICIENT"
    )

    print(
        "Root X/Y stance locking removes a meaningful "
        "part of the dynamic inconsistency."
    )

    print(
        "NEXT: Dynamic V2 Pass 2 should also optimize "
        "root orientation and lower-body joints."
    )


else:

    print(
        "RESULT: ROOT X/Y SUPPORT LOCK IS "
        "NOT SUFFICIENT"
    )

    print(
        "The major single-support residual cannot be "
        "fixed by translating the pelvis alone."
    )

    print(
        "NEXT: Dynamic V2 Pass 2 — constrained "
        "dynamics-aware optimization of:"
    )

    print(
        "  root XYZ / roll / pitch"
    )

    print(
        "  hip pitch + roll"
    )

    print(
        "  knee"
    )

    print(
        "  ankle pitch + roll"
    )

    print(
        "while strongly preserving the original motion."
    )


print()
print(
    "candidate reference:",
    OUTPUT_REFERENCE,
)

print(
    "diagnostic artifact:",
    OUTPUT_DIAGNOSTIC,
)

print()
print(
    "ORIGINAL REFERENCE WAS NOT MODIFIED."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 195)


env.close()


