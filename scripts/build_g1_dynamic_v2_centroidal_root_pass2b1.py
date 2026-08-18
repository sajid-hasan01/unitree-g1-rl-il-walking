from pathlib import Path
import math
import sys

import mujoco
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


# ================================================================
# CONFIG
# ================================================================

KNOT_COUNT = 9

# First/last knot remain exactly zero.
INTERIOR_KNOTS = KNOT_COUNT - 2

# Smooth centroidal correction limits.
XY_LIMIT = 0.025                 # 25 mm
ROLL_LIMIT = math.radians(3.0)
PITCH_LIMIT = math.radians(3.0)

PARAM_LIMITS = np.asarray(
    [
        XY_LIMIT,
        XY_LIMIT,
        ROLL_LIMIT,
        PITCH_LIMIT,
    ],
    dtype=np.float64,
)


CONTACT_DEPTH = 1e-6             # 0.001 mm
TRANSITION_WINDOW = 2


# Objective weighting.
ROOT_FORCE_WEIGHT = 1.00
ROOT_TORQUE_WEIGHT = 0.30
TORQUE_EXCESS_WEIGHT = 0.15

MOTION_REG_WEIGHT = 0.30
SMOOTHNESS_WEIGHT = 0.45


# Scale root torque into dimensionless units.
ROOT_TORQUE_LENGTH_SCALE = 0.50


MAX_NFEV = 55


OUTPUT_REFERENCE = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_centroidal_root_candidate.npz"
)


OUTPUT_DIAGNOSTIC = (
    ROOT
    / "results"
    / "g1_dynamic_v2_centroidal_root_pass2b1.npz"
)


# ================================================================
# ENV
# ================================================================

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


support_count = np.sum(
    support,
    axis=1,
)


single_support = (
    support_count == 1
)


double_support = (
    support_count >= 2
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


ROOT_TORQUE_SCALE = (
    BODY_WEIGHT
    * ROOT_TORQUE_LENGTH_SCALE
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


print("=" * 205)
print("G1 DYNAMIC V2 — PASS 2B1")
print("DIRECT CENTROIDAL / FLOATING-BASE DYNAMICS OPTIMIZATION")
print("SMOOTH ROOT X/Y + ROLL/PITCH SPLINE")
print("STARTS FROM IMMUTABLE ORIGINAL REFERENCE")
print("NO FOOT-LOCK OBJECTIVE")
print("NO CEM / NO PPO")
print("=" * 205)

print(
    "reference:",
    env.reference_path,
)

print(
    "frames:",
    N,
)

print(
    "frequency:",
    env.reference_fps,
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)

print(
    "optimization variables:",
    INTERIOR_KNOTS * 4,
)


# ================================================================
# FROZEN MASK — SAME DEFINITION AS PASS 2A
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
        support[frame],
        support[frame - 1],
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


steady_double = (
    double_support
    &
    ~transition
)


print()
print("FROZEN AUDIT MASK")

print(
    "single support:",
    int(
        np.sum(
            single_support
        )
    ),
)

print(
    "steady single:",
    int(
        np.sum(
            steady_single
        )
    ),
)

print(
    "steady double:",
    int(
        np.sum(
            steady_double
        )
    ),
)

print(
    "transition:",
    int(
        np.sum(
            transition
        )
    ),
)


# ================================================================
# FRAME WEIGHTS
# ================================================================

frame_weight = np.full(
    N,
    0.20,
    dtype=np.float64,
)


frame_weight[
    steady_double
] = 0.30


frame_weight[
    transition
] = 0.70


frame_weight[
    steady_single
] = 1.00


# ================================================================
# DIFFERENTIATION
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
        2.0 * DT
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


# ================================================================
# SPLINE PARAMETERIZATION
#
# Optimizer works in normalized coordinates [-1,+1].
# ================================================================

knot_frames = np.linspace(
    0.0,
    float(
        N - 1
    ),
    KNOT_COUNT,
)


sample_frames = np.arange(
    N,
    dtype=np.float64,
)


def decode_parameters(
    flat,
):

    normalized = np.asarray(
        flat,
        dtype=np.float64,
    ).reshape(
        INTERIOR_KNOTS,
        4,
    )


    full_knots = np.zeros(
        (
            KNOT_COUNT,
            4,
        ),
        dtype=np.float64,
    )


    full_knots[
        1:-1
    ] = (
        normalized
        *
        PARAM_LIMITS[
            None,
            :
        ]
    )


    correction = np.zeros(
        (
            N,
            4,
        ),
        dtype=np.float64,
    )


    for dim in range(
        4
    ):

        spline = CubicSpline(
            knot_frames,
            full_knots[
                :,
                dim
            ],
            bc_type="natural",
        )


        correction[
            :,
            dim
        ] = spline(
            sample_frames
        )


    return (
        normalized,
        full_knots,
        correction,
    )


# ================================================================
# APPLY ROOT CORRECTION
#
# correction columns:
# 0 = world X
# 1 = world Y
# 2 = root roll tangent
# 3 = root pitch tangent
# ================================================================

def build_candidate(
    flat,
):

    (
        normalized,
        full_knots,
        correction,
    ) = decode_parameters(
        flat
    )


    qpos = old_qpos.copy()


    dq = np.zeros(
        model.nv,
        dtype=np.float64,
    )


    for frame in range(N):

        dq[:] = 0.0


        dq[
            0
        ] = correction[
            frame,
            0
        ]


        dq[
            1
        ] = correction[
            frame,
            1
        ]


        dq[
            3
        ] = correction[
            frame,
            2
        ]


        dq[
            4
        ] = correction[
            frame,
            3
        ]


        mujoco.mj_integratePos(
            model,
            qpos[
                frame
            ],
            dq,
            1.0,
        )


    return (
        qpos,
        normalized,
        full_knots,
        correction,
    )


# ================================================================
# CONTACT SNAP
# ================================================================

contact_probe = mujoco.MjData(
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


        bottom = (
            float(
                data.geom_xpos[
                    gid,
                    2
                ]
            )
            -
            radius
        )


        minimum = min(
            minimum,
            bottom - floor_z,
        )


    return minimum


def expected_support(
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


def choose_anchor(
    qpos,
    frame,
):

    contact_probe.qpos[:] = (
        qpos[
            frame
        ]
    )


    contact_probe.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        contact_probe,
    )


    left_clearance = (
        sole_clearance(
            contact_probe,
            LEFT_GEOMS,
        )
    )


    right_clearance = (
        sole_clearance(
            contact_probe,
            RIGHT_GEOMS,
        )
    )


    expected = expected_support(
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
        key=lambda item:
            item[
                1
            ],
    )


def contact_snap(
    qpos,
):

    snapped = (
        qpos.copy()
    )


    anchors = np.zeros(
        N,
        dtype=np.int8,
    )


    z_shift = np.zeros(
        N,
        dtype=np.float64,
    )


    for frame in range(N):

        (
            anchor,
            clearance,
        ) = choose_anchor(
            qpos,
            frame,
        )


        anchors[
            frame
        ] = anchor


        dz = (
            -CONTACT_DEPTH
            -
            clearance
        )


        dz = min(
            0.0,
            dz,
        )


        dz = max(
            -0.025,
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
        anchors,
        z_shift,
    )


# ================================================================
# ACTIVE CONTACTS
# ================================================================

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
            ==
            env.floor_geom
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


# ================================================================
# DYNAMICS EVALUATION
# ================================================================

dynamics_data = mujoco.MjData(
    model
)


def dynamics(
    qpos,
    detailed=False,
):

    (
        snapped_qpos,
        anchors,
        z_shift,
    ) = contact_snap(
        qpos
    )


    qvel = differentiate_qpos(
        snapped_qpos
    )


    qacc = differentiate_velocity(
        qvel
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


    anchor_active = np.zeros(
        N,
        dtype=bool,
    )


    for frame in range(N):

        dynamics_data.qpos[:] = (
            snapped_qpos[
                frame
            ]
        )


        dynamics_data.qvel[:] = (
            qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            dynamics_data,
        )


        dynamics_data.qacc[:] = (
            qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            dynamics_data,
        )


        (
            left_active,
            right_active,
        ) = active_contacts(
            dynamics_data
        )


        anchor_active[
            frame
        ] = (
            left_active
            if anchors[
                frame
            ] == 0
            else right_active
        )


        root_force[
            frame
        ] = (
            dynamics_data.qfrc_inverse[
                0:3
            ]
        )


        root_torque[
            frame
        ] = (
            dynamics_data.qfrc_inverse[
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
                dynamics_data.qfrc_inverse[
                    vadr
                ]
            )


    result = {

        "snapped_qpos":
            snapped_qpos,

        "qvel":
            qvel,

        "qacc":
            qacc,

        "anchors":
            anchors,

        "z_shift":
            z_shift,

        "root_force":
            root_force,

        "root_torque":
            root_torque,

        "joint_tau":
            joint_tau,

        "anchor_active":
            anchor_active,
    }


    return result


# ================================================================
# OBJECTIVE
# ================================================================

objective_calls = 0


def objective(
    flat,
):

    global objective_calls


    objective_calls += 1


    (
        qpos,
        normalized,
        full_knots,
        correction,
    ) = build_candidate(
        flat
    )


    dyn = dynamics(
        qpos
    )


    residual = []


    for frame in range(N):

        w = math.sqrt(
            frame_weight[
                frame
            ]
        )


        # --------------------------------------------------------
        # Unactuated translational dynamics.
        # --------------------------------------------------------

        force_residual = (
            ROOT_FORCE_WEIGHT
            *
            w
            *
            dyn[
                "root_force"
            ][
                frame
            ]
            /
            BODY_WEIGHT
        )


        residual.extend(
            force_residual.tolist()
        )


        # --------------------------------------------------------
        # Unactuated rotational dynamics.
        # --------------------------------------------------------

        torque_residual = (
            ROOT_TORQUE_WEIGHT
            *
            w
            *
            dyn[
                "root_torque"
            ][
                frame
            ]
            /
            ROOT_TORQUE_SCALE
        )


        residual.extend(
            torque_residual.tolist()
        )


        # --------------------------------------------------------
        # Penalize effort requirement above 80% authority.
        # --------------------------------------------------------

        torque_ratio = (
            np.abs(
                dyn[
                    "joint_tau"
                ][
                    frame
                ]
            )
            /
            effort
        )


        excess = np.maximum(
            torque_ratio
            -
            0.80,
            0.0,
        )


        residual.extend(
            (
                TORQUE_EXCESS_WEIGHT
                *
                w
                *
                excess
            ).tolist()
        )


    # ------------------------------------------------------------
    # Motion preservation in normalized knot coordinates.
    # ------------------------------------------------------------

    residual.extend(
        (
            MOTION_REG_WEIGHT
            *
            normalized.reshape(
                -1
            )
        ).tolist()
    )


    # ------------------------------------------------------------
    # Penalize second differences of spline knots.
    # ------------------------------------------------------------

    normalized_full = np.zeros(
        (
            KNOT_COUNT,
            4,
        ),
        dtype=np.float64,
    )


    normalized_full[
        1:-1
    ] = (
        normalized
    )


    second_difference = (
        normalized_full[
            2:
        ]
        -
        2.0
        *
        normalized_full[
            1:-1
        ]
        +
        normalized_full[
            :-2
        ]
    )


    residual.extend(
        (
            SMOOTHNESS_WEIGHT
            *
            second_difference.reshape(
                -1
            )
        ).tolist()
    )


    result = np.asarray(
        residual,
        dtype=np.float64,
    )


    if not np.all(
        np.isfinite(
            result
        )
    ):

        raise RuntimeError(
            "Objective produced non-finite values."
        )


    if (
        objective_calls == 1
        or
        objective_calls % 20 == 0
    ):

        force_bw = (
            np.linalg.norm(
                dyn[
                    "root_force"
                ],
                axis=1,
            )
            /
            BODY_WEIGHT
        )


        print(
            f"objectiveCall={objective_calls:04d} "
            f"rootP50="
            f"{np.percentile(force_bw,50):.3f} "
            f"rootP95="
            f"{np.percentile(force_bw,95):.3f} "
            f"singleP95="
            f"{np.percentile(force_bw[steady_single],95):.3f}"
        )


    return result


# ================================================================
# BASELINE
# ================================================================

print()
print("=" * 205)
print("BASELINE DYNAMIC AUDIT")
print("=" * 205)


baseline_dyn = dynamics(
    old_qpos
)


def summarize(
    dyn,
):

    root_bw = (
        np.linalg.norm(
            dyn[
                "root_force"
            ],
            axis=1,
        )
        /
        BODY_WEIGHT
    )


    root_torque_scaled = (
        np.linalg.norm(
            dyn[
                "root_torque"
            ],
            axis=1,
        )
        /
        ROOT_TORQUE_SCALE
    )


    torque_ratio = (
        np.abs(
            dyn[
                "joint_tau"
            ]
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


    def p95(
        mask,
    ):

        return float(
            np.percentile(
                root_bw[
                    mask
                ],
                95,
            )
        )


    return {

        "root_bw":
            root_bw,

        "root_torque_scaled":
            root_torque_scaled,

        "max_torque_ratio":
            max_torque_ratio,

        "anchor_validity":
            float(
                np.mean(
                    dyn[
                        "anchor_active"
                    ]
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
            p95(
                steady_single
            ),

        "double_p95":
            p95(
                steady_double
            ),

        "transition_p95":
            p95(
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
    }


baseline = summarize(
    baseline_dyn
)


print(
    "anchor validity:",
    f"{100*baseline['anchor_validity']:.1f}%",
)

print(
    "overall p95:",
    f"{baseline['overall_p95']:.3f} BW",
)

print(
    "steady single p95:",
    f"{baseline['single_p95']:.3f} BW",
)

print(
    "transition p95:",
    f"{baseline['transition_p95']:.3f} BW",
)

print(
    "torque p95:",
    f"{baseline['tau_p95']:.3f}",
)

print(
    "over-limit:",
    f"{100*baseline['overlimit']:.1f}%",
)


# ================================================================
# BASELINE CONSISTENCY GATE
# ================================================================

if abs(
    baseline[
        "overall_p95"
    ]
    -
    11.255
) > 0.75:

    raise RuntimeError(
        "Baseline overall p95 does not reproduce "
        "the frozen Pass-2A audit closely enough. "
        f"Observed={baseline['overall_p95']:.3f}"
    )


if baseline[
    "anchor_validity"
] < 0.95:

    raise RuntimeError(
        "Baseline contact validity failed."
    )


# ================================================================
# OPTIMIZE
# ================================================================

print()
print("=" * 205)
print("RUNNING DIRECT DYNAMICS OPTIMIZATION")
print("=" * 205)


x0 = np.zeros(
    INTERIOR_KNOTS
    *
    4,
    dtype=np.float64,
)


result = least_squares(
    objective,

    x0=x0,

    bounds=(
        -1.0,
        1.0,
    ),

    method="trf",

    jac="2-point",

    diff_step=1e-3,

    loss="soft_l1",

    f_scale=1.0,

    x_scale="jac",

    max_nfev=MAX_NFEV,

    ftol=5e-5,
    xtol=5e-5,
    gtol=5e-5,

    verbose=1,
)


print()
print(
    "optimizer success:",
    result.success,
)

print(
    "optimizer status:",
    result.status,
)

print(
    "optimizer message:",
    result.message,
)

print(
    "optimizer nfev:",
    result.nfev,
)

print(
    "optimizer cost:",
    f"{result.cost:.6f}",
)


# ================================================================
# BUILD FINAL CANDIDATE
# ================================================================

(
    candidate_qpos,
    candidate_normalized,
    candidate_knots,
    candidate_correction,
) = build_candidate(
    result.x
)


candidate_dyn = dynamics(
    candidate_qpos
)


candidate = summarize(
    candidate_dyn
)


# Candidate qvel WITHOUT audit-only contact snapping.
candidate_qvel = differentiate_qpos(
    candidate_qpos
)


# ================================================================
# CORRECTION METRICS
# ================================================================

xy_mag = np.linalg.norm(
    candidate_correction[
        :,
        0:2
    ],
    axis=1,
)


roll_abs = np.abs(
    candidate_correction[
        :,
        2
    ]
)


pitch_abs = np.abs(
    candidate_correction[
        :,
        3
    ]
)


correction_step = np.zeros(
    (
        N,
        4,
    ),
    dtype=np.float64,
)


correction_step[
    1:
] = np.diff(
    candidate_correction,
    axis=0,
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 205)
print("ORIGINAL VS PASS-2B1 DYNAMIC FEASIBILITY")
print("=" * 205)

print(
    f"{'REF':12s} "
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

print("-" * 205)


for name, summary in (
    (
        "ORIGINAL",
        baseline,
    ),
    (
        "PASS2B1",
        candidate,
    ),
):

    print(
        f"{name:12s} "
        f"{100*summary['anchor_validity']:6.1f}% "
        f"{summary['overall_p50']:9.3f} "
        f"{summary['overall_p95']:9.3f} "
        f"{summary['overall_max']:9.3f} "
        f"{summary['single_p95']:9.3f} "
        f"{summary['double_p95']:9.3f} "
        f"{summary['transition_p95']:9.3f} "
        f"{summary['tau_p95']:9.3f} "
        f"{100*summary['overlimit']:8.1f}%"
    )


def reduction(
    before,
    after,
):

    return (
        before - after
    ) / max(
        abs(before),
        1e-12,
    )


overall_reduction = reduction(
    baseline[
        "overall_p95"
    ],
    candidate[
        "overall_p95"
    ],
)


single_reduction = reduction(
    baseline[
        "single_p95"
    ],
    candidate[
        "single_p95"
    ],
)


transition_reduction = reduction(
    baseline[
        "transition_p95"
    ],
    candidate[
        "transition_p95"
    ],
)


tau_reduction = reduction(
    baseline[
        "tau_p95"
    ],
    candidate[
        "tau_p95"
    ],
)


print()
print("=" * 205)
print("PASS-2B1 CORRECTION SIZE")
print("=" * 205)

print(
    "root XY:",
    f"p50={1000*np.percentile(xy_mag,50):.2f}mm",
    f"p95={1000*np.percentile(xy_mag,95):.2f}mm",
    f"max={1000*np.max(xy_mag):.2f}mm",
)

print(
    "root roll:",
    f"p95={np.degrees(np.percentile(roll_abs,95)):.2f}deg",
    f"max={np.degrees(np.max(roll_abs)):.2f}deg",
)

print(
    "root pitch:",
    f"p95={np.degrees(np.percentile(pitch_abs,95)):.2f}deg",
    f"max={np.degrees(np.max(pitch_abs)):.2f}deg",
)

print(
    "frame-step XY:",
    f"p95={1000*np.percentile(np.linalg.norm(correction_step[:,0:2],axis=1),95):.2f}mm",
)

print(
    "frame-step rotation:",
    f"p95={np.degrees(np.percentile(np.linalg.norm(correction_step[:,2:4],axis=1),95)):.3f}deg",
)


print()
print("=" * 205)
print("PASS-2B1 DECISION")
print("=" * 205)

print(
    "overall p95:",
    f"{baseline['overall_p95']:.3f}",
    "->",
    f"{candidate['overall_p95']:.3f}",
    "BW",
    f"({100*overall_reduction:+.1f}%)",
)

print(
    "steady single p95:",
    f"{baseline['single_p95']:.3f}",
    "->",
    f"{candidate['single_p95']:.3f}",
    "BW",
    f"({100*single_reduction:+.1f}%)",
)

print(
    "transition p95:",
    f"{baseline['transition_p95']:.3f}",
    "->",
    f"{candidate['transition_p95']:.3f}",
    "BW",
    f"({100*transition_reduction:+.1f}%)",
)

print(
    "torque p95:",
    f"{baseline['tau_p95']:.3f}",
    "->",
    f"{candidate['tau_p95']:.3f}",
    f"({100*tau_reduction:+.1f}%)",
)

print(
    "over-limit:",
    f"{100*baseline['overlimit']:.1f}%",
    "->",
    f"{100*candidate['overlimit']:.1f}%",
)


# ================================================================
# PATCH SOURCE NPZ
# ================================================================

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

    return (
        numeric_array(
            value
        )
        and
        value.shape
        ==
        target.shape
        and
        np.allclose(
            value,
            target,
            rtol=1e-5,
            atol=5e-6,
            equal_nan=True,
        )
    )


patched = {
    "full_qpos": [],
    "full_qvel": [],
    "root_pos": [],
    "root_quat": [],
    "root_linvel": [],
    "root_angvel": [],
}


replacement_sets = [
    (
        "full_qpos",
        old_qpos,
        candidate_qpos,
    ),
    (
        "full_qvel",
        old_qvel,
        candidate_qvel,
    ),
    (
        "root_pos",
        old_qpos[:, 0:3],
        candidate_qpos[:, 0:3],
    ),
    (
        "root_quat",
        old_qpos[:, 3:7],
        candidate_qpos[:, 3:7],
    ),
    (
        "root_linvel",
        old_qvel[:, 0:3],
        candidate_qvel[:, 0:3],
    ),
    (
        "root_angvel",
        old_qvel[:, 3:6],
        candidate_qvel[:, 3:6],
    ),
]


for (
    label,
    old_array,
    new_array,
) in replacement_sets:

    for key in list(
        payload.keys()
    ):

        value = payload[
            key
        ]


        if matches(
            value,
            old_array,
        ):

            payload[
                key
            ] = (
                new_array.astype(
                    value.dtype
                )
            )


            patched[
                label
            ].append(
                key
            )


if not patched[
    "full_qpos"
]:

    raise RuntimeError(
        "Could not patch full_qpos safely. "
        "Candidate not saved."
    )


payload[
    "dynamic_v2_pass"
] = np.asarray(
    "pass2b1_direct_root_dynamics",
)


payload[
    "pass2b1_root_correction"
] = (
    candidate_correction.astype(
        np.float32
    )
)


payload[
    "pass2b1_knots"
] = (
    candidate_knots.astype(
        np.float32
    )
)


np.savez_compressed(
    OUTPUT_REFERENCE,
    **payload,
)


# ================================================================
# DIAGNOSTIC SAVE
# ================================================================

np.savez_compressed(
    OUTPUT_DIAGNOSTIC,

    candidate_qpos=
        candidate_qpos.astype(
            np.float32
        ),

    candidate_qvel=
        candidate_qvel.astype(
            np.float32
        ),

    correction=
        candidate_correction.astype(
            np.float32
        ),

    knots=
        candidate_knots.astype(
            np.float32
        ),

    optimizer_x=
        result.x.astype(
            np.float32
        ),

    original_root_bw=
        baseline[
            "root_bw"
        ].astype(
            np.float32
        ),

    candidate_root_bw=
        candidate[
            "root_bw"
        ].astype(
            np.float32
        ),

    original_tau_ratio=
        baseline[
            "max_torque_ratio"
        ].astype(
            np.float32
        ),

    candidate_tau_ratio=
        candidate[
            "max_torque_ratio"
        ].astype(
            np.float32
        ),

    overall_reduction=
        np.asarray(
            [overall_reduction],
            dtype=np.float32,
        ),

    single_reduction=
        np.asarray(
            [single_reduction],
            dtype=np.float32,
        ),

    transition_reduction=
        np.asarray(
            [transition_reduction],
            dtype=np.float32,
        ),
)


print()
print("=" * 205)
print("NPZ PATCH")
print("=" * 205)

for key, values in patched.items():

    print(
        f"{key:14s}:",
        values,
    )


# ================================================================
# FINAL VERDICT
# ================================================================

print()
print("=" * 205)
print("FINAL PASS-2B1 VERDICT")
print("=" * 205)


if candidate[
    "anchor_validity"
] < 0.95:

    print(
        "RESULT: CONTACT VALIDITY FAILED"
    )

    print(
        "Do not interpret optimizer result."
    )


elif (
    overall_reduction >= 0.30
    and
    single_reduction >= 0.35
    and
    candidate[
        "tau_p95"
    ]
    <
    baseline[
        "tau_p95"
    ]
    and
    candidate[
        "overlimit"
    ]
    <
    baseline[
        "overlimit"
    ]
):

    print(
        "RESULT: DIRECT ROOT-DYNAMICS OPTIMIZATION "
        "IS A MAJOR IMPROVEMENT"
    )

    print(
        "KEEP Pass-2B1 candidate."
    )

    print(
        "NEXT: extend the same objective to a small "
        "lower-body spline set and target transitions."
    )


elif (
    overall_reduction >= 0.15
    or
    single_reduction >= 0.20
):

    print(
        "RESULT: DIRECT ROOT-DYNAMICS OPTIMIZATION HELPS"
    )

    print(
        "The centroidal/root trajectory is part of "
        "the mismatch, but root correction alone "
        "is insufficient."
    )

    print(
        "KEEP as an intermediate diagnostic."
    )

    print(
        "NEXT: Pass-2B2 adds support-leg hip/knee/"
        "ankle spline corrections to the same "
        "direct dynamics objective."
    )


else:

    print(
        "RESULT: ROOT CENTROIDAL CORRECTION "
        "IS NOT SUFFICIENT"
    )

    print(
        "Do not keep this as the final reference."
    )

    print(
        "NEXT: Pass-2B2 must optimize lower-body "
        "joint motion directly against inverse "
        "dynamics, or reconsider medium_02 as "
        "the source motion."
    )


print()
print(
    "candidate:",
    OUTPUT_REFERENCE,
)

print(
    "diagnostic:",
    OUTPUT_DIAGNOSTIC,
)

print()
print(
    "ORIGINAL REFERENCE WAS NOT MODIFIED."
)

print(
    "PASS-1 AND PASS-2A CANDIDATES WERE NOT USED."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 205)


env.close()
