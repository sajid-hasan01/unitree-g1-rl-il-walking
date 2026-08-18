from pathlib import Path
import csv
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

REFERENCE = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
    / "medium_08_50hz_grounded.npz"
)

OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_medium08_support_snap_validity.npz"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_medium08_support_snap_validity.csv"
)

CONTACT_DEPTH = 1e-6

# A labelled stance foot is considered geometrically higher
# than the non-support foot if the difference exceeds 0.5 mm.
MISMATCH_TOL = 0.0005

TRANSITION_WINDOW = 2


JOINT_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",

    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",

    "waist_yaw",
    "waist_roll",
    "waist_pitch",

    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


# ================================================================
# ENVIRONMENT
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)

model = env.model


if not REFERENCE.exists():
    raise FileNotFoundError(
        REFERENCE
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

FLOOR_GEOM = int(
    env.floor_geom
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


print("=" * 205)
print("G1 MEDIUM_08 SUPPORT-SNAP VALIDITY AUDIT")
print("EXPECTED-STANCE SNAP vs GLOBAL-LOWEST-FOOT SNAP")
print("TESTING WHETHER THE PREVIOUS INVERSE AUDIT CREATES")
print("UNINTENDED SECOND-FOOT PENETRATION")
print("NO OPTIMIZATION / NO REFERENCE MODIFICATION / NO PPO")
print("=" * 205)

print(
    "reference:",
    REFERENCE,
)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "model:",
    model.nq,
    model.nv,
    model.nu,
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)


# ================================================================
# LOAD REFERENCE
# ================================================================

with np.load(
    REFERENCE,
    allow_pickle=True,
) as f:

    qpos = np.asarray(
        f[
            "full_qpos"
        ],
        dtype=np.float64,
    ).copy()


    support = (
        np.asarray(
            f[
                "support_mask"
            ],
            dtype=np.float64,
        )
        > 0.5
    )


    contact_mask = (
        np.asarray(
            f[
                "contact_mask"
            ],
            dtype=np.float64,
        )
        > 0.5
    )


    if "fps" in f.files:

        fps_array = np.asarray(
            f[
                "fps"
            ]
        ).reshape(
            -1
        )

        FPS = float(
            fps_array[
                0
            ]
        )

    else:

        FPS = 50.0


N = len(
    qpos
)

DT = 1.0 / FPS


print()
print(
    "frames:",
    N,
)

print(
    "fps:",
    FPS,
)


# ================================================================
# HELPERS
# ================================================================

def phase_name(
    mask,
):

    left = bool(
        mask[
            0
        ]
    )

    right = bool(
        mask[
            1
        ]
    )


    if left and right:
        return "B"

    if left:
        return "L"

    if right:
        return "R"

    return "N"


def differentiate_qpos(
    trajectory,
):

    n = len(
        trajectory
    )


    velocity = np.zeros(
        (
            n,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        velocity[
            0
        ],
        DT,
        trajectory[
            0
        ],
        trajectory[
            1
        ],
    )


    for frame in range(
        1,
        n - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            velocity[
                frame
            ],
            2.0 * DT,
            trajectory[
                frame - 1
            ],
            trajectory[
                frame + 1
            ],
        )


    mujoco.mj_differentiatePos(
        model,
        velocity[
            -1
        ],
        DT,
        trajectory[
            -2
        ],
        trajectory[
            -1
        ],
    )


    return velocity


def differentiate_array(
    values,
):

    derivative = np.zeros_like(
        values
    )


    derivative[
        1:-1
    ] = (
        values[
            2:
        ]
        -
        values[
            :-2
        ]
    ) / (
        2.0 * DT
    )


    derivative[
        0
    ] = (
        values[
            1
        ]
        -
        values[
            0
        ]
    ) / DT


    derivative[
        -1
    ] = (
        values[
            -1
        ]
        -
        values[
            -2
        ]
    ) / DT


    return derivative


# ================================================================
# RAW SOLE CLEARANCES
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
            FLOOR_GEOM,
            2
        ]
    )


    value = float(
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


        value = min(
            value,
            bottom
            -
            floor_z,
        )


    return value


left_clearance = np.zeros(
    N,
    dtype=np.float64,
)

right_clearance = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    probe.qpos[:] = (
        qpos[
            frame
        ]
    )

    probe.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        probe,
    )


    left_clearance[
        frame
    ] = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    right_clearance[
        frame
    ] = sole_clearance(
        probe,
        RIGHT_GEOMS,
    )


# ================================================================
# SUPPORT ORDER / MISMATCH
# ================================================================

support_count = np.sum(
    support,
    axis=1,
)

single_support = (
    support_count
    ==
    1
)

double_support = (
    support_count
    >=
    2
)


transition_mask = np.zeros(
    N,
    dtype=bool,
)


transition_frames = []


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

        transition_frames.append(
            frame
        )


        lo = max(
            0,
            frame
            -
            TRANSITION_WINDOW,
        )


        hi = min(
            N,
            frame
            +
            TRANSITION_WINDOW
            +
            1,
        )


        transition_mask[
            lo:hi
        ] = True


steady_single = (
    single_support
    &
    ~transition_mask
)

steady_double = (
    double_support
    &
    ~transition_mask
)


support_side = np.full(
    N,
    -1,
    dtype=np.int8,
)


support_order_gap = np.full(
    N,
    np.nan,
    dtype=np.float64,
)


support_order_mismatch = np.zeros(
    N,
    dtype=bool,
)


label_already_double = np.zeros(
    N,
    dtype=bool,
)


for frame in range(N):

    if not single_support[
        frame
    ]:

        continue


    side = int(
        np.argmax(
            support[
                frame
            ]
        )
    )


    support_side[
        frame
    ] = side


    if side == 0:

        stance_clearance = (
            left_clearance[
                frame
            ]
        )

        other_clearance = (
            right_clearance[
                frame
            ]
        )

        label_already_double[
            frame
        ] = bool(
            contact_mask[
                frame,
                1
            ]
        )


    else:

        stance_clearance = (
            right_clearance[
                frame
            ]
        )

        other_clearance = (
            left_clearance[
                frame
            ]
        )

        label_already_double[
            frame
        ] = bool(
            contact_mask[
                frame,
                0
            ]
        )


    gap = (
        stance_clearance
        -
        other_clearance
    )


    support_order_gap[
        frame
    ] = gap


    support_order_mismatch[
        frame
    ] = (
        gap
        >
        MISMATCH_TOL
    )


# ================================================================
# BUILD TWO SNAP TRAJECTORIES
# ================================================================

def expected_mask(
    frame,
):

    mask = (
        support[
            frame
        ].copy()
    )


    if not np.any(
        mask
    ):

        mask = (
            contact_mask[
                frame
            ].copy()
        )


    return mask


expected_qpos = (
    qpos.copy()
)

global_qpos = (
    qpos.copy()
)


expected_anchor = np.zeros(
    N,
    dtype=np.int8,
)

global_anchor = np.zeros(
    N,
    dtype=np.int8,
)


expected_shift = np.zeros(
    N,
    dtype=np.float64,
)

global_shift = np.zeros(
    N,
    dtype=np.float64,
)


forced_other_penetration = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    # ------------------------------------------------------------
    # CURRENT METHOD:
    # choose lowest foot only among the expected support set.
    # ------------------------------------------------------------

    mask = expected_mask(
        frame
    )


    choices = []


    if mask[
        0
    ]:

        choices.append(
            (
                0,
                left_clearance[
                    frame
                ],
            )
        )


    if mask[
        1
    ]:

        choices.append(
            (
                1,
                right_clearance[
                    frame
                ],
            )
        )


    if not choices:

        choices = [
            (
                0,
                left_clearance[
                    frame
                ],
            ),
            (
                1,
                right_clearance[
                    frame
                ],
            ),
        ]


    (
        exp_side,
        exp_clearance,
    ) = min(
        choices,
        key=lambda item:
            item[
                1
            ],
    )


    dz_expected = (
        -CONTACT_DEPTH
        -
        exp_clearance
    )


    # Same semantics as previous diagnostics:
    # lower only.
    dz_expected = min(
        0.0,
        dz_expected,
    )


    dz_expected = max(
        -0.025,
        dz_expected,
    )


    expected_anchor[
        frame
    ] = exp_side


    expected_shift[
        frame
    ] = dz_expected


    expected_qpos[
        frame,
        2
    ] += dz_expected


    # ------------------------------------------------------------
    # How far did this expected-support snap force the OTHER foot
    # below the floor?
    # ------------------------------------------------------------

    if single_support[
        frame
    ]:

        side = support_side[
            frame
        ]


        if side == 0:

            other_after = (
                right_clearance[
                    frame
                ]
                +
                dz_expected
            )

        else:

            other_after = (
                left_clearance[
                    frame
                ]
                +
                dz_expected
            )


        forced_other_penetration[
            frame
        ] = max(
            0.0,
            -other_after,
        )


    # ------------------------------------------------------------
    # CORRECTED COUNTERFACTUAL:
    # choose the physically lowest sole, regardless of label.
    #
    # This guarantees that lowering the root to first contact
    # does not intentionally drive another lower foot through
    # the floor.
    # ------------------------------------------------------------

    if (
        left_clearance[
            frame
        ]
        <=
        right_clearance[
            frame
        ]
    ):

        glo_side = 0

        glo_clearance = (
            left_clearance[
                frame
            ]
        )

    else:

        glo_side = 1

        glo_clearance = (
            right_clearance[
                frame
            ]
        )


    dz_global = (
        -CONTACT_DEPTH
        -
        glo_clearance
    )


    dz_global = min(
        0.0,
        dz_global,
    )


    dz_global = max(
        -0.025,
        dz_global,
    )


    global_anchor[
        frame
    ] = glo_side


    global_shift[
        frame
    ] = dz_global


    global_qpos[
        frame,
        2
    ] += dz_global


# ================================================================
# ACTIVE CONTACT / FORCE
# ================================================================

def contact_information(
    data,
):

    left = False
    right = False

    sole_count = 0
    total_normal = 0.0


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


        if FLOOR_GEOM not in (
            g1,
            g2,
        ):

            continue


        other = (
            g2
            if g1
            ==
            FLOOR_GEOM
            else g1
        )


        if (
            other not in LEFT_SET
            and
            other not in RIGHT_SET
        ):

            continue


        if other in LEFT_SET:

            left = True


        if other in RIGHT_SET:

            right = True


        sole_count += 1


        force = np.zeros(
            6,
            dtype=np.float64,
        )


        mujoco.mj_contactForce(
            model,
            data,
            cid,
            force,
        )


        total_normal += max(
            float(
                force[
                    0
                ]
            ),
            0.0,
        )


    return (
        left,
        right,
        sole_count,
        total_normal,
    )


# ================================================================
# INVERSE AUDIT
# ================================================================

def audit(
    label,
    trajectory,
    anchors,
    shift,
):

    print()
    print(
        "AUDIT:",
        label,
    )


    qvel = differentiate_qpos(
        trajectory
    )


    qacc = differentiate_array(
        qvel
    )


    shift_velocity = differentiate_array(
        shift
    )


    shift_acceleration = differentiate_array(
        shift_velocity
    )


    data = mujoco.MjData(
        model
    )


    root_bw = np.zeros(
        N,
        dtype=np.float64,
    )


    root_force_components = np.zeros(
        (
            N,
            3,
        ),
        dtype=np.float64,
    )


    max_tau = np.zeros(
        N,
        dtype=np.float64,
    )


    max_joint = np.zeros(
        N,
        dtype=np.int32,
    )


    contact_phase = np.empty(
        N,
        dtype="<U1",
    )


    contact_count = np.zeros(
        N,
        dtype=np.int32,
    )


    normal_bw = np.zeros(
        N,
        dtype=np.float64,
    )


    anchor_active = np.zeros(
        N,
        dtype=bool,
    )


    for frame in range(N):

        data.qpos[:] = (
            trajectory[
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


        data.qacc[:] = (
            qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            data,
        )


        root_force = (
            data.qfrc_inverse[
                0:3
            ]
        )


        root_force_components[
            frame
        ] = (
            root_force
            /
            BODY_WEIGHT
        )


        root_bw[
            frame
        ] = (
            np.linalg.norm(
                root_force
            )
            /
            BODY_WEIGHT
        )


        joint_tau = np.asarray(
            [
                data.qfrc_inverse[
                    vadr
                ]
                for vadr in env.vaddrs
            ],
            dtype=np.float64,
        )


        ratio = (
            np.abs(
                joint_tau
            )
            /
            effort
        )


        max_joint[
            frame
        ] = int(
            np.argmax(
                ratio
            )
        )


        max_tau[
            frame
        ] = float(
            np.max(
                ratio
            )
        )


        (
            left,
            right,
            count,
            normal,
        ) = contact_information(
            data
        )


        contact_phase[
            frame
        ] = phase_name(
            np.asarray(
                [
                    left,
                    right,
                ],
                dtype=bool,
            )
        )


        contact_count[
            frame
        ] = count


        normal_bw[
            frame
        ] = (
            normal
            /
            BODY_WEIGHT
        )


        anchor_active[
            frame
        ] = (
            left
            if anchors[
                frame
            ]
            ==
            0
            else right
        )


    return {

        "qvel":
            qvel,

        "qacc":
            qacc,

        "shift_velocity":
            shift_velocity,

        "shift_acceleration":
            shift_acceleration,

        "root_bw":
            root_bw,

        "root_force_components":
            root_force_components,

        "max_tau":
            max_tau,

        "max_joint":
            max_joint,

        "contact_phase":
            contact_phase,

        "contact_count":
            contact_count,

        "normal_bw":
            normal_bw,

        "anchor_validity":
            float(
                np.mean(
                    anchor_active
                )
            ),
    }


expected_result = audit(
    "EXPECTED_SUPPORT",
    expected_qpos,
    expected_anchor,
    expected_shift,
)


global_result = audit(
    "GLOBAL_LOWEST",
    global_qpos,
    global_anchor,
    global_shift,
)


# ================================================================
# METRIC HELPERS
# ================================================================

def p95_mask(
    values,
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
            values[
                mask
            ],
            95,
        )
    )


def summary(
    result,
):

    return {

        "anchor":
            result[
                "anchor_validity"
            ],

        "root50":
            float(
                np.percentile(
                    result[
                        "root_bw"
                    ],
                    50,
                )
            ),

        "root95":
            float(
                np.percentile(
                    result[
                        "root_bw"
                    ],
                    95,
                )
            ),

        "rootmax":
            float(
                np.max(
                    result[
                        "root_bw"
                    ]
                )
            ),

        "single":
            p95_mask(
                result[
                    "root_bw"
                ],
                steady_single,
            ),

        "double":
            p95_mask(
                result[
                    "root_bw"
                ],
                steady_double,
            ),

        "transition":
            p95_mask(
                result[
                    "root_bw"
                ],
                transition_mask,
            ),

        "tau95":
            float(
                np.percentile(
                    result[
                        "max_tau"
                    ],
                    95,
                )
            ),

        "over":
            float(
                np.mean(
                    result[
                        "max_tau"
                    ]
                    >
                    1.0
                )
            ),

        "normal95":
            float(
                np.percentile(
                    result[
                        "normal_bw"
                    ],
                    95,
                )
            ),
    }


expected_summary = summary(
    expected_result
)


global_summary = summary(
    global_result
)


# ================================================================
# BASELINE REPRODUCTION
# ================================================================

print()
print("=" * 205)
print("EXPECTED-SUPPORT BASELINE REPRODUCTION")
print("=" * 205)

print(
    "anchor:",
    f"{100*expected_summary['anchor']:.1f}%",
)

print(
    "root p95:",
    f"{expected_summary['root95']:.3f}",
)

print(
    "single:",
    f"{expected_summary['single']:.3f}",
)

print(
    "transition:",
    f"{expected_summary['transition']:.3f}",
)

print(
    "tau p95:",
    f"{expected_summary['tau95']:.3f}",
)


if abs(
    expected_summary[
        "root95"
    ]
    -
    13.403
) > 0.80:

    raise RuntimeError(
        "Expected-support baseline changed."
    )


if abs(
    expected_summary[
        "single"
    ]
    -
    0.869
) > 0.35:

    raise RuntimeError(
        "Expected-support single-support baseline changed."
    )


if abs(
    expected_summary[
        "transition"
    ]
    -
    31.033
) > 2.0:

    raise RuntimeError(
        "Expected-support transition baseline changed."
    )


# ================================================================
# SUPPORT ORDER ANALYSIS
# ================================================================

single_count = int(
    np.sum(
        single_support
    )
)


mismatch_count = int(
    np.sum(
        support_order_mismatch
    )
)


steady_mismatch_count = int(
    np.sum(
        support_order_mismatch
        &
        steady_single
    )
)


transition_mismatch_count = int(
    np.sum(
        support_order_mismatch
        &
        transition_mask
    )
)


forced_penetration_frames = (
    forced_other_penetration
    >
    1e-6
)


print()
print("=" * 205)
print("SUPPORT-ORDER VALIDITY")
print("=" * 205)

print(
    "single-support frames:",
    single_count,
)

print(
    "stance foot higher than swing foot >0.5mm:",
    mismatch_count,
    f"({100*mismatch_count/max(single_count,1):.1f}%)",
)

print(
    "steady-single mismatches:",
    steady_mismatch_count,
)

print(
    "transition-window mismatches:",
    transition_mismatch_count,
)

print(
    "frames where expected snap drives other foot below floor:",
    int(
        np.sum(
            forced_penetration_frames
        )
    ),
)

print(
    "forced other-foot penetration:",
    f"p50={1000*np.percentile(forced_other_penetration[forced_penetration_frames],50):.2f}mm"
    if np.any(
        forced_penetration_frames
    )
    else "p50=0",

    f"p95={1000*np.percentile(forced_other_penetration[forced_penetration_frames],95):.2f}mm"
    if np.any(
        forced_penetration_frames
    )
    else "p95=0",

    f"max={1000*np.max(forced_other_penetration):.2f}mm",
)


# ================================================================
# MAIN COMPARISON
# ================================================================

print()
print("=" * 205)
print("EXPECTED-SUPPORT vs GLOBAL-LOWEST SNAP")
print("=" * 205)

print(
    f"{'MODE':18s} "
    f"{'ANCH':>7s} "
    f"{'ROOT50':>9s} "
    f"{'ROOT95':>9s} "
    f"{'ROOTMAX':>9s} "
    f"{'SINGLE':>9s} "
    f"{'DOUBLE':>9s} "
    f"{'TRANS':>9s} "
    f"{'TAU95':>9s} "
    f"{'>LIMIT':>9s} "
    f"{'NORM95':>9s}"
)

print("-" * 205)


for name, s in (
    (
        "EXPECTED_SUPPORT",
        expected_summary,
    ),
    (
        "GLOBAL_LOWEST",
        global_summary,
    ),
):

    print(
        f"{name:18s} "
        f"{100*s['anchor']:6.1f}% "
        f"{s['root50']:9.3f} "
        f"{s['root95']:9.3f} "
        f"{s['rootmax']:9.3f} "
        f"{s['single']:9.3f} "
        f"{s['double']:9.3f} "
        f"{s['transition']:9.3f} "
        f"{s['tau95']:9.3f} "
        f"{100*s['over']:8.1f}% "
        f"{s['normal95']:9.3f}"
    )


# ================================================================
# MISMATCH-SPECIFIC RESIDUAL
# ================================================================

matched_single = (
    single_support
    &
    ~support_order_mismatch
)


mismatched_single = (
    single_support
    &
    support_order_mismatch
)


print()
print("=" * 205)
print("SINGLE-SUPPORT: GEOMETRICALLY MATCHED vs MISMATCHED")
print("=" * 205)


for label, mask in (
    (
        "MATCHED",
        matched_single,
    ),
    (
        "MISMATCHED",
        mismatched_single,
    ),
):

    if not np.any(
        mask
    ):

        continue


    print()
    print(
        label,
        "frames:",
        int(
            np.sum(
                mask
            )
        ),
    )


    print(
        " EXPECTED root p50/p95:",
        f"{np.percentile(expected_result['root_bw'][mask],50):.3f}",
        f"{np.percentile(expected_result['root_bw'][mask],95):.3f}",
    )


    print(
        " GLOBAL   root p50/p95:",
        f"{np.percentile(global_result['root_bw'][mask],50):.3f}",
        f"{np.percentile(global_result['root_bw'][mask],95):.3f}",
    )


    print(
        " EXPECTED contact-normal p95:",
        f"{np.percentile(expected_result['normal_bw'][mask],95):.3f}",
    )


    print(
        " GLOBAL   contact-normal p95:",
        f"{np.percentile(global_result['normal_bw'][mask],95):.3f}",
    )


# ================================================================
# TOP CURRENT SPIKES
# ================================================================

print()
print("=" * 205)
print("TOP 20 EXPECTED-SUPPORT SPIKES")
print("=" * 205)


top = np.argsort(
    expected_result[
        "root_bw"
    ]
)[
    ::-1
][
    :20
]


top_mismatch_count = 0


for rank, frame in enumerate(
    top,
    start=1,
):

    mismatch = bool(
        support_order_mismatch[
            frame
        ]
    )


    if mismatch:

        top_mismatch_count += 1


    side = support_side[
        frame
    ]


    if side == 0:

        stance_mm = (
            1000
            *
            left_clearance[
                frame
            ]
        )

        other_mm = (
            1000
            *
            right_clearance[
                frame
            ]
        )

    elif side == 1:

        stance_mm = (
            1000
            *
            right_clearance[
                frame
            ]
        )

        other_mm = (
            1000
            *
            left_clearance[
                frame
            ]
        )

    else:

        stance_mm = float(
            "nan"
        )

        other_mm = float(
            "nan"
        )


    print(
        f"{rank:02d}. "
        f"frame={frame:03d} "
        f"sup={phase_name(support[frame])} "
        f"label={phase_name(contact_mask[frame])} "
        f"expectedAct={expected_result['contact_phase'][frame]} "
        f"globalAct={global_result['contact_phase'][frame]} "
        f"root={expected_result['root_bw'][frame]:7.3f}"
        f"->{global_result['root_bw'][frame]:7.3f} "
        f"normal={expected_result['normal_bw'][frame]:7.2f}"
        f"->{global_result['normal_bw'][frame]:7.2f} "
        f"tau={expected_result['max_tau'][frame]:7.2f}"
        f"->{global_result['max_tau'][frame]:7.2f} "
        f"stanceClr={stance_mm:6.2f}mm "
        f"otherClr={other_mm:6.2f}mm "
        f"forcedPen={1000*forced_other_penetration[frame]:6.2f}mm "
        f"mismatch={mismatch} "
        f"joint={JOINT_NAMES[expected_result['max_joint'][frame]]}"
    )


print()
print(
    "top-20 spikes that are support-order mismatches:",
    top_mismatch_count,
    "/ 20",
)


# ================================================================
# SPECIFIC KNOWN FRAMES
# ================================================================

important_frames = [
    35,
    53,
    80,
    117,
    118,
    138,
    139,
    140,
    141,
    142,
]


print()
print("=" * 205)
print("KNOWN TRANSITION-SPIKE FRAMES")
print("=" * 205)

print(
    "frm Sup Lab "
    "Lclr Rclr expDz gloDz forcedPen "
    "ExpAct GloAct "
    "ExpRoot GloRoot "
    "ExpNorm GloNorm "
    "ExpTau GloTau"
)


for frame in important_frames:

    if frame >= N:
        continue


    print(
        f"{frame:3d} "
        f"{phase_name(support[frame]):>3s} "
        f"{phase_name(contact_mask[frame]):>3s} "
        f"{1000*left_clearance[frame]:5.1f} "
        f"{1000*right_clearance[frame]:5.1f} "
        f"{1000*expected_shift[frame]:6.1f} "
        f"{1000*global_shift[frame]:6.1f} "
        f"{1000*forced_other_penetration[frame]:8.1f} "
        f"{expected_result['contact_phase'][frame]:>6s} "
        f"{global_result['contact_phase'][frame]:>6s} "
        f"{expected_result['root_bw'][frame]:7.2f} "
        f"{global_result['root_bw'][frame]:7.2f} "
        f"{expected_result['normal_bw'][frame]:7.2f} "
        f"{global_result['normal_bw'][frame]:7.2f} "
        f"{expected_result['max_tau'][frame]:7.2f} "
        f"{global_result['max_tau'][frame]:7.2f}"
    )


# ================================================================
# SAVE CSV
# ================================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    fieldnames = [
        "frame",
        "time_s",

        "support",
        "contact_label",

        "left_clearance_mm",
        "right_clearance_mm",

        "support_order_gap_mm",
        "support_order_mismatch",

        "contact_label_has_other_foot",

        "expected_shift_mm",
        "global_shift_mm",

        "forced_other_penetration_mm",

        "expected_contact_phase",
        "global_contact_phase",

        "expected_root_bw",
        "global_root_bw",

        "expected_normal_bw",
        "global_normal_bw",

        "expected_tau_ratio",
        "global_tau_ratio",

        "expected_shift_acceleration",
        "global_shift_acceleration",

        "is_transition",
    ]


    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    for frame in range(N):

        writer.writerow(
            {
                "frame":
                    frame,

                "time_s":
                    frame
                    *
                    DT,

                "support":
                    phase_name(
                        support[
                            frame
                        ]
                    ),

                "contact_label":
                    phase_name(
                        contact_mask[
                            frame
                        ]
                    ),

                "left_clearance_mm":
                    1000
                    *
                    left_clearance[
                        frame
                    ],

                "right_clearance_mm":
                    1000
                    *
                    right_clearance[
                        frame
                    ],

                "support_order_gap_mm":
                    1000
                    *
                    support_order_gap[
                        frame
                    ]
                    if np.isfinite(
                        support_order_gap[
                            frame
                        ]
                    )
                    else np.nan,

                "support_order_mismatch":
                    int(
                        support_order_mismatch[
                            frame
                        ]
                    ),

                "contact_label_has_other_foot":
                    int(
                        label_already_double[
                            frame
                        ]
                    ),

                "expected_shift_mm":
                    1000
                    *
                    expected_shift[
                        frame
                    ],

                "global_shift_mm":
                    1000
                    *
                    global_shift[
                        frame
                    ],

                "forced_other_penetration_mm":
                    1000
                    *
                    forced_other_penetration[
                        frame
                    ],

                "expected_contact_phase":
                    expected_result[
                        "contact_phase"
                    ][
                        frame
                    ],

                "global_contact_phase":
                    global_result[
                        "contact_phase"
                    ][
                        frame
                    ],

                "expected_root_bw":
                    expected_result[
                        "root_bw"
                    ][
                        frame
                    ],

                "global_root_bw":
                    global_result[
                        "root_bw"
                    ][
                        frame
                    ],

                "expected_normal_bw":
                    expected_result[
                        "normal_bw"
                    ][
                        frame
                    ],

                "global_normal_bw":
                    global_result[
                        "normal_bw"
                    ][
                        frame
                    ],

                "expected_tau_ratio":
                    expected_result[
                        "max_tau"
                    ][
                        frame
                    ],

                "global_tau_ratio":
                    global_result[
                        "max_tau"
                    ][
                        frame
                    ],

                "expected_shift_acceleration":
                    expected_result[
                        "shift_acceleration"
                    ][
                        frame
                    ],

                "global_shift_acceleration":
                    global_result[
                        "shift_acceleration"
                    ][
                        frame
                    ],

                "is_transition":
                    int(
                        transition_mask[
                            frame
                        ]
                    ),
            }
        )


# ================================================================
# SAVE NPZ
# ================================================================

np.savez_compressed(
    OUTPUT_NPZ,

    left_clearance=
        left_clearance.astype(
            np.float32
        ),

    right_clearance=
        right_clearance.astype(
            np.float32
        ),

    support=
        support,

    contact_mask=
        contact_mask,

    transition_mask=
        transition_mask,

    support_order_gap=
        support_order_gap.astype(
            np.float32
        ),

    support_order_mismatch=
        support_order_mismatch,

    forced_other_penetration=
        forced_other_penetration.astype(
            np.float32
        ),

    expected_shift=
        expected_shift.astype(
            np.float32
        ),

    global_shift=
        global_shift.astype(
            np.float32
        ),

    expected_root_bw=
        expected_result[
            "root_bw"
        ].astype(
            np.float32
        ),

    global_root_bw=
        global_result[
            "root_bw"
        ].astype(
            np.float32
        ),

    expected_normal_bw=
        expected_result[
            "normal_bw"
        ].astype(
            np.float32
        ),

    global_normal_bw=
        global_result[
            "normal_bw"
        ].astype(
            np.float32
        ),

    expected_tau=
        expected_result[
            "max_tau"
        ].astype(
            np.float32
        ),

    global_tau=
        global_result[
            "max_tau"
        ].astype(
            np.float32
        ),
)


# ================================================================
# DECISION
# ================================================================

root_reduction = (
    expected_summary[
        "root95"
    ]
    -
    global_summary[
        "root95"
    ]
) / max(
    expected_summary[
        "root95"
    ],
    1e-9,
)


transition_reduction = (
    expected_summary[
        "transition"
    ]
    -
    global_summary[
        "transition"
    ]
) / max(
    expected_summary[
        "transition"
    ],
    1e-9,
)


tau_reduction = (
    expected_summary[
        "tau95"
    ]
    -
    global_summary[
        "tau95"
    ]
) / max(
    expected_summary[
        "tau95"
    ],
    1e-9,
)


print()
print("=" * 205)
print("FINAL SUPPORT-SNAP VALIDITY DECISION")
print("=" * 205)

print(
    "overall root p95 reduction:",
    f"{100*root_reduction:+.1f}%",
)

print(
    "transition p95 reduction:",
    f"{100*transition_reduction:+.1f}%",
)

print(
    "torque p95 reduction:",
    f"{100*tau_reduction:+.1f}%",
)

print(
    "top-20 mismatch fraction:",
    f"{top_mismatch_count}/20",
)

print()


if (
    transition_reduction
    >=
    0.50
    and
    root_reduction
    >=
    0.30
):

    print(
        "RESULT: EXPECTED-STANCE CONTACT SNAP "
        "WAS STRONGLY CONTAMINATING THE AUDIT."
    )

    print(
        "The old transition residuals cannot be interpreted "
        "as pure reference dynamic inconsistency."
    )

    print(
        "NEXT:"
    )

    print(
        "Rebuild support/contact labels from actual G1 sole "
        "geometry and contact ordering before any new "
        "root-dynamics repair."
    )


elif (
    transition_reduction
    >=
    0.25
    or
    top_mismatch_count
    >=
    8
):

    print(
        "RESULT: SUPPORT-ORDER / SNAP ARTIFACT "
        "IS A MAJOR CONTRIBUTOR."
    )

    print(
        "Some genuine dynamic inconsistency may remain, "
        "but the current audit exaggerates it."
    )

    print(
        "NEXT:"
    )

    print(
        "Re-derive G1 support/contact timing first, then "
        "rerun the consistent inverse audit."
    )


else:

    print(
        "RESULT: GLOBAL-LOWEST CONTACT PROJECTION "
        "DOES NOT REMOVE THE FAILURE."
    )

    print(
        "The transition problem remains genuinely dynamic."
    )

    print(
        "NEXT:"
    )

    print(
        "Proceed to a smooth transition-only root/contact "
        "trajectory repair."
    )


print()
print("=" * 205)
print("OUTPUTS")
print("=" * 205)

print(
    "CSV:",
    OUTPUT_CSV,
)

print(
    "NPZ:",
    OUTPUT_NPZ,
)

print()
print(
    "REFERENCE WAS NOT MODIFIED."
)

print(
    "NO OPTIMIZATION."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 205)


env.close()
