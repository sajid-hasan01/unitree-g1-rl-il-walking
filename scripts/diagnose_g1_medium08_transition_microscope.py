from pathlib import Path
import csv
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

REFERENCE = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
    / "medium_08_50hz_grounded.npz"
)

OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_medium08_transition_microscope.npz"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_medium08_transition_microscope.csv"
)

OUTPUT_PNG = (
    ROOT
    / "results"
    / "g1_medium08_transition_microscope.png"
)


CONTACT_DEPTH = 1e-6

# Print +/- this many frames around every support change.
WINDOW = 6

# Same transition definition as previous audits.
TRANSITION_WINDOW = 2

# Normalization for floating-base rotational residual.
ROOT_TORQUE_LENGTH_SCALE = 0.50


# ================================================================
# G1 JOINT NAMES
# ================================================================

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
# ENV / MODEL
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

FLOOR_GEOM = int(
    env.floor_geom
)


LEFT_BODY = int(
    model.geom_bodyid[
        LEFT_GEOMS[0]
    ]
)

RIGHT_BODY = int(
    model.geom_bodyid[
        RIGHT_GEOMS[0]
    ]
)


effort_limits = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


print("=" * 210)
print("G1 MEDIUM_08 TRANSITION MICROSCOPE")
print("PHASE / ROOT / COM / FOOT / CONTACT / INVERSE-DYNAMICS AUDIT")
print("NO OPTIMIZATION / NO CEM / NO PPO")
print("=" * 210)

print(
    "reference:",
    REFERENCE,
)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "model nq/nv/nu:",
    model.nq,
    model.nv,
    model.nu,
)

print(
    "mass:",
    f"{TOTAL_MASS:.3f} kg",
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)

print(
    "left sole geoms:",
    LEFT_GEOMS,
)

print(
    "right sole geoms:",
    RIGHT_GEOMS,
)


# ================================================================
# LOAD MEDIUM_08
# ================================================================

with np.load(
    REFERENCE,
    allow_pickle=True,
) as f:

    required = [
        "full_qpos",
        "full_qvel",
        "support_mask",
        "contact_mask",
    ]

    for key in required:

        if key not in f.files:

            raise RuntimeError(
                f"Required key missing: {key}"
            )


    qpos = np.asarray(
        f[
            "full_qpos"
        ],
        dtype=np.float64,
    ).copy()


    stored_qvel = np.asarray(
        f[
            "full_qvel"
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

        fps_value = np.asarray(
            f[
                "fps"
            ]
        ).reshape(
            -1
        )

        FPS = float(
            fps_value[
                0
            ]
        )

    else:

        FPS = 50.0


N = qpos.shape[
    0
]

DT = (
    1.0
    /
    FPS
)


if qpos.shape != (
    N,
    model.nq,
):

    raise RuntimeError(
        f"qpos shape invalid: {qpos.shape}"
    )


if stored_qvel.shape != (
    N,
    model.nv,
):

    raise RuntimeError(
        f"qvel shape invalid: {stored_qvel.shape}"
    )


if support.shape != (
    N,
    2,
):

    raise RuntimeError(
        f"support shape invalid: {support.shape}"
    )


if contact_mask.shape != (
    N,
    2,
):

    raise RuntimeError(
        f"contact shape invalid: {contact_mask.shape}"
    )


print()
print("REFERENCE")

print(
    "frames:",
    N,
)

print(
    "fps:",
    FPS,
)

print(
    "duration:",
    f"{(N-1)/FPS:.3f} s",
)

print(
    "qpos:",
    qpos.shape,
)

print(
    "stored qvel:",
    stored_qvel.shape,
)


# ================================================================
# BASIC HELPERS
# ================================================================

def differentiate_qpos(
    trajectory,
):

    count = len(
        trajectory
    )


    velocity = np.zeros(
        (
            count,
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
        count - 1,
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

    values = np.asarray(
        values,
        dtype=np.float64,
    )


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
        2.0
        *
        DT
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


def quat_to_rpy_deg(
    q,
):

    # q = w, x, y, z

    w, x, y, z = q


    sinr_cosp = (
        2.0
        *
        (
            w*x
            +
            y*z
        )
    )

    cosr_cosp = (
        1.0
        -
        2.0
        *
        (
            x*x
            +
            y*y
        )
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )


    sinp = (
        2.0
        *
        (
            w*y
            -
            z*x
        )
    )

    sinp = float(
        np.clip(
            sinp,
            -1.0,
            1.0,
        )
    )

    pitch = math.asin(
        sinp
    )


    siny_cosp = (
        2.0
        *
        (
            w*z
            +
            x*y
        )
    )

    cosy_cosp = (
        1.0
        -
        2.0
        *
        (
            y*y
            +
            z*z
        )
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )


    return np.degrees(
        np.asarray(
            [
                roll,
                pitch,
                yaw,
            ],
            dtype=np.float64,
        )
    )


def quat_angle(
    q0,
    q1,
):

    q0 = (
        q0
        /
        max(
            np.linalg.norm(
                q0
            ),
            1e-12,
        )
    )

    q1 = (
        q1
        /
        max(
            np.linalg.norm(
                q1
            ),
            1e-12,
        )
    )


    dot = abs(
        float(
            np.dot(
                q0,
                q1,
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


    return (
        2.0
        *
        math.acos(
            dot
        )
    )


def percentile_rank(
    array,
    value,
):

    array = np.asarray(
        array,
        dtype=np.float64,
    )


    finite = array[
        np.isfinite(
            array
        )
    ]


    if len(
        finite
    ) == 0:

        return float(
            "nan"
        )


    return float(
        100.0
        *
        np.mean(
            finite
            <=
            value
        )
    )


def correlation(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )


    mask = (
        np.isfinite(
            a
        )
        &
        np.isfinite(
            b
        )
    )


    a = a[
        mask
    ]

    b = b[
        mask
    ]


    if (
        len(
            a
        )
        <
        3
        or
        np.std(
            a
        )
        <
        1e-12
        or
        np.std(
            b
        )
        <
        1e-12
    ):

        return float(
            "nan"
        )


    return float(
        np.corrcoef(
            a,
            b,
        )[
            0,
            1
        ]
    )


# ================================================================
# RECOMPUTE RAW DERIVATIVES
# ================================================================

raw_qvel = differentiate_qpos(
    qpos
)

raw_qacc = differentiate_array(
    raw_qvel
)


qvel_difference = (
    raw_qvel
    -
    stored_qvel
)


print()
print("=" * 210)
print("REFERENCE DERIVATIVE CONSISTENCY")
print("=" * 210)

print(
    "stored-vs-recomputed qvel RMS:",
    f"{np.sqrt(np.mean(qvel_difference**2)):.6f}",
)

print(
    "stored-vs-recomputed qvel max:",
    f"{np.max(np.abs(qvel_difference)):.6f}",
)


# ================================================================
# RAW GEOMETRY / COM / FOOT TRAJECTORIES
# ================================================================

geometry_data = mujoco.MjData(
    model
)


left_foot_pos = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


right_foot_pos = np.zeros_like(
    left_foot_pos
)


left_foot_quat = np.zeros(
    (
        N,
        4,
    ),
    dtype=np.float64,
)


right_foot_quat = np.zeros_like(
    left_foot_quat
)


left_clearance = np.zeros(
    N,
    dtype=np.float64,
)


right_clearance = np.zeros(
    N,
    dtype=np.float64,
)


com_position = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


root_rpy = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


def sole_rep_position(
    data,
    geoms,
):

    return np.mean(
        data.geom_xpos[
            geoms
        ],
        axis=0,
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


for frame in range(N):

    geometry_data.qpos[:] = (
        qpos[
            frame
        ]
    )

    geometry_data.qvel[:] = (
        raw_qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        geometry_data,
    )


    left_foot_pos[
        frame
    ] = sole_rep_position(
        geometry_data,
        LEFT_GEOMS,
    )


    right_foot_pos[
        frame
    ] = sole_rep_position(
        geometry_data,
        RIGHT_GEOMS,
    )


    left_foot_quat[
        frame
    ] = (
        geometry_data.xquat[
            LEFT_BODY
        ]
    )


    right_foot_quat[
        frame
    ] = (
        geometry_data.xquat[
            RIGHT_BODY
        ]
    )


    left_clearance[
        frame
    ] = sole_clearance(
        geometry_data,
        LEFT_GEOMS,
    )


    right_clearance[
        frame
    ] = sole_clearance(
        geometry_data,
        RIGHT_GEOMS,
    )


    com_position[
        frame
    ] = (
        geometry_data.subtree_com[
            0
        ]
    )


    root_rpy[
        frame
    ] = quat_to_rpy_deg(
        qpos[
            frame,
            3:7
        ]
    )


left_foot_vel = differentiate_array(
    left_foot_pos
)

right_foot_vel = differentiate_array(
    right_foot_pos
)


left_foot_speed_xy = np.linalg.norm(
    left_foot_vel[
        :,
        0:2
    ],
    axis=1,
)


right_foot_speed_xy = np.linalg.norm(
    right_foot_vel[
        :,
        0:2
    ],
    axis=1,
)


left_foot_angular_speed = np.zeros(
    N,
    dtype=np.float64,
)

right_foot_angular_speed = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(
    N - 1
):

    left_foot_angular_speed[
        frame
    ] = (
        quat_angle(
            left_foot_quat[
                frame
            ],
            left_foot_quat[
                frame + 1
            ],
        )
        /
        DT
    )


    right_foot_angular_speed[
        frame
    ] = (
        quat_angle(
            right_foot_quat[
                frame
            ],
            right_foot_quat[
                frame + 1
            ],
        )
        /
        DT
    )


left_foot_angular_speed[
    -1
] = left_foot_angular_speed[
    -2
]

right_foot_angular_speed[
    -1
] = right_foot_angular_speed[
    -2
]


com_velocity = differentiate_array(
    com_position
)

com_acceleration = differentiate_array(
    com_velocity
)


# ================================================================
# SUPPORT CENTER / COM DISTANCE
# ================================================================

support_center_xy = np.full(
    (
        N,
        2,
    ),
    np.nan,
    dtype=np.float64,
)


com_support_distance = np.full(
    N,
    np.nan,
    dtype=np.float64,
)


for frame in range(N):

    points = []


    if support[
        frame,
        0
    ]:

        points.append(
            left_foot_pos[
                frame,
                0:2
            ]
        )


    if support[
        frame,
        1
    ]:

        points.append(
            right_foot_pos[
                frame,
                0:2
            ]
        )


    if points:

        support_center_xy[
            frame
        ] = np.mean(
            np.asarray(
                points
            ),
            axis=0,
        )


        com_support_distance[
            frame
        ] = float(
            np.linalg.norm(
                com_position[
                    frame,
                    0:2
                ]
                -
                support_center_xy[
                    frame
                ]
            )
        )


# ================================================================
# CONTACT-SNAP TRAJECTORY
# ================================================================

probe = mujoco.MjData(
    model
)


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
            contact_mask[
                frame
            ].copy()
        )


    return expected


def choose_anchor(
    frame,
):

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


    lc = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    rc = sole_clearance(
        probe,
        RIGHT_GEOMS,
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
                lc,
            )
        )


    if expected[
        1
    ]:

        candidates.append(
            (
                1,
                rc,
            )
        )


    if not candidates:

        candidates = [
            (
                0,
                lc,
            ),
            (
                1,
                rc,
            ),
        ]


    return min(
        candidates,
        key=lambda item:
            item[
                1
            ],
    )


snapped_qpos = qpos.copy()


anchor_side = np.zeros(
    N,
    dtype=np.int8,
)


snap_z = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    (
        side,
        clearance,
    ) = choose_anchor(
        frame
    )


    anchor_side[
        frame
    ] = side


    dz = (
        -CONTACT_DEPTH
        -
        clearance
    )


    # Only lower root. Never raise it.
    dz = min(
        0.0,
        dz,
    )


    # Safety cap used in prior diagnostic family.
    dz = max(
        -0.025,
        dz,
    )


    snapped_qpos[
        frame,
        2
    ] += dz


    snap_z[
        frame
    ] = dz


snapped_qvel = differentiate_qpos(
    snapped_qpos
)

snapped_qacc = differentiate_array(
    snapped_qvel
)


snap_z_velocity = differentiate_array(
    snap_z
)

snap_z_acceleration = differentiate_array(
    snap_z_velocity
)


snap_root_acc_delta = (
    snapped_qacc[
        :,
        0:3
    ]
    -
    raw_qacc[
        :,
        0:3
    ]
)


# ================================================================
# CONSISTENT INVERSE DYNAMICS
# ================================================================

inverse_data = mujoco.MjData(
    model
)


root_force = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


root_force_bw_components = np.zeros_like(
    root_force
)


root_bw = np.zeros(
    N,
    dtype=np.float64,
)


root_torque = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


root_torque_scaled = np.zeros(
    N,
    dtype=np.float64,
)


joint_tau = np.zeros(
    (
        N,
        29,
    ),
    dtype=np.float64,
)


max_tau_ratio = np.zeros(
    N,
    dtype=np.float64,
)


max_tau_joint = np.zeros(
    N,
    dtype=np.int32,
)


actual_contact = np.zeros(
    (
        N,
        2,
    ),
    dtype=bool,
)


active_sole_contact_count = np.zeros(
    N,
    dtype=np.int32,
)


contact_normal_bw = np.zeros(
    N,
    dtype=np.float64,
)


def contact_information(
    data,
):

    left = False
    right = False

    count = 0

    normal_force = 0.0


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


        count += 1


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


        # Contact-frame component 0 is the normal force.
        normal_force += max(
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
        count,
        normal_force,
    )


for frame in range(N):

    inverse_data.qpos[:] = (
        snapped_qpos[
            frame
        ]
    )


    inverse_data.qvel[:] = (
        snapped_qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        inverse_data,
    )


    inverse_data.qacc[:] = (
        snapped_qacc[
            frame
        ]
    )


    mujoco.mj_inverse(
        model,
        inverse_data,
    )


    root_force[
        frame
    ] = (
        inverse_data.qfrc_inverse[
            0:3
        ]
    )


    root_force_bw_components[
        frame
    ] = (
        root_force[
            frame
        ]
        /
        BODY_WEIGHT
    )


    root_bw[
        frame
    ] = (
        np.linalg.norm(
            root_force[
                frame
            ]
        )
        /
        BODY_WEIGHT
    )


    root_torque[
        frame
    ] = (
        inverse_data.qfrc_inverse[
            3:6
        ]
    )


    root_torque_scaled[
        frame
    ] = (
        np.linalg.norm(
            root_torque[
                frame
            ]
        )
        /
        ROOT_TORQUE_SCALE
    )


    for joint_index, vadr in enumerate(
        env.vaddrs
    ):

        joint_tau[
            frame,
            joint_index
        ] = (
            inverse_data.qfrc_inverse[
                vadr
            ]
        )


    ratios = (
        np.abs(
            joint_tau[
                frame
            ]
        )
        /
        effort_limits
    )


    max_tau_joint[
        frame
    ] = int(
        np.argmax(
            ratios
        )
    )


    max_tau_ratio[
        frame
    ] = float(
        np.max(
            ratios
        )
    )


    (
        left_active,
        right_active,
        contact_count,
        normal_force,
    ) = contact_information(
        inverse_data
    )


    actual_contact[
        frame,
        0
    ] = left_active


    actual_contact[
        frame,
        1
    ] = right_active


    active_sole_contact_count[
        frame
    ] = contact_count


    contact_normal_bw[
        frame
    ] = (
        normal_force
        /
        BODY_WEIGHT
    )


# ================================================================
# FROZEN TRANSITION MASK
# ================================================================

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


single_mask = (
    np.sum(
        support,
        axis=1,
    )
    ==
    1
)


double_mask = (
    np.sum(
        support,
        axis=1,
    )
    >=
    2
)


steady_single = (
    single_mask
    &
    ~transition_mask
)


steady_double = (
    double_mask
    &
    ~transition_mask
)


# ================================================================
# BASELINE REPRODUCTION
# ================================================================

def safe_p95(
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


overall95 = float(
    np.percentile(
        root_bw,
        95,
    )
)


single95 = safe_p95(
    root_bw,
    steady_single,
)


double95 = safe_p95(
    root_bw,
    steady_double,
)


transition95 = safe_p95(
    root_bw,
    transition_mask,
)


tau95 = float(
    np.percentile(
        max_tau_ratio,
        95,
    )
)


overlimit = float(
    np.mean(
        max_tau_ratio
        >
        1.0
    )
)


anchor_valid = np.mean(
    np.asarray(
        [
            actual_contact[
                i,
                anchor_side[
                    i
                ]
            ]
            for i in range(N)
        ],
        dtype=np.float64,
    )
)


print()
print("=" * 210)
print("BASELINE REPRODUCTION")
print("=" * 210)

print(
    "anchor validity:",
    f"{100*anchor_valid:.1f}%",
)

print(
    "root p95:",
    f"{overall95:.3f} BW",
)

print(
    "steady single p95:",
    f"{single95:.3f} BW",
)

print(
    "steady double p95:",
    f"{double95:.3f} BW",
)

print(
    "transition p95:",
    f"{transition95:.3f} BW",
)

print(
    "tau p95:",
    f"{tau95:.3f}",
)

print(
    "over-limit:",
    f"{100*overlimit:.1f}%",
)


# Gate against the previous medium_08 audit.
if anchor_valid < 0.95:

    raise RuntimeError(
        "Anchor validity failed; transition microscope invalid."
    )


if abs(
    overall95
    -
    13.403
) > 0.80:

    raise RuntimeError(
        "Medium_08 overall baseline changed: "
        f"{overall95:.3f}"
    )


if abs(
    single95
    -
    0.869
) > 0.35:

    raise RuntimeError(
        "Medium_08 steady-single baseline changed: "
        f"{single95:.3f}"
    )


if abs(
    transition95
    -
    31.033
) > 2.0:

    raise RuntimeError(
        "Medium_08 transition baseline changed: "
        f"{transition95:.3f}"
    )


# ================================================================
# FOOT / ROOT EVENT METRICS
# ================================================================

max_foot_speed = np.maximum(
    np.linalg.norm(
        left_foot_vel,
        axis=1,
    ),
    np.linalg.norm(
        right_foot_vel,
        axis=1,
    ),
)


max_foot_speed_xy = np.maximum(
    left_foot_speed_xy,
    right_foot_speed_xy,
)


max_foot_vz_abs = np.maximum(
    np.abs(
        left_foot_vel[
            :,
            2
        ]
    ),
    np.abs(
        right_foot_vel[
            :,
            2
        ]
    ),
)


max_foot_angspeed = np.maximum(
    left_foot_angular_speed,
    right_foot_angular_speed,
)


raw_root_acc_mag = np.linalg.norm(
    raw_qacc[
        :,
        0:3
    ],
    axis=1,
)


raw_root_angacc_mag = np.linalg.norm(
    raw_qacc[
        :,
        3:6
    ],
    axis=1,
)


snap_acc_delta_mag = np.linalg.norm(
    snap_root_acc_delta,
    axis=1,
)


com_acc_mag = np.linalg.norm(
    com_acceleration,
    axis=1,
)


# ================================================================
# GLOBAL DISTRIBUTION COMPARISON
# ================================================================

def distribution_line(
    name,
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    print(
        f"{name:27s} "
        f"steadySingle p50={np.percentile(values[steady_single],50):9.3f} "
        f"p95={np.percentile(values[steady_single],95):9.3f} | "
        f"transition p50={np.percentile(values[transition_mask],50):9.3f} "
        f"p95={np.percentile(values[transition_mask],95):9.3f}"
    )


print()
print("=" * 210)
print("STEADY-SUPPORT vs TRANSITION DISTRIBUTIONS")
print("=" * 210)

distribution_line(
    "root residual [BW]",
    root_bw,
)

distribution_line(
    "raw root |acc| [m/s2]",
    raw_root_acc_mag,
)

distribution_line(
    "raw root |ang acc|",
    raw_root_angacc_mag,
)

distribution_line(
    "snap-induced |acc|",
    snap_acc_delta_mag,
)

distribution_line(
    "|snap Z accel| [m/s2]",
    np.abs(
        snap_z_acceleration
    ),
)

distribution_line(
    "COM |acc| [m/s2]",
    com_acc_mag,
)

distribution_line(
    "max foot speed [m/s]",
    max_foot_speed,
)

distribution_line(
    "max foot XY speed [m/s]",
    max_foot_speed_xy,
)

distribution_line(
    "max |foot vz| [m/s]",
    max_foot_vz_abs,
)

distribution_line(
    "max foot ang vel [rad/s]",
    max_foot_angspeed,
)

distribution_line(
    "max torque / limit",
    max_tau_ratio,
)

distribution_line(
    "contact normal / BW",
    contact_normal_bw,
)


# ================================================================
# CORRELATIONS
# ================================================================

print()
print("=" * 210)
print("CORRELATION WITH ROOT RESIDUAL")
print("Correlation is diagnostic only; it does not prove causality.")
print("=" * 210)


correlations = {
    "raw_root_acc":
        correlation(
            root_bw,
            raw_root_acc_mag,
        ),

    "raw_root_abs_az":
        correlation(
            root_bw,
            np.abs(
                raw_qacc[
                    :,
                    2
                ]
            ),
        ),

    "snap_delta_acc":
        correlation(
            root_bw,
            snap_acc_delta_mag,
        ),

    "snap_abs_az":
        correlation(
            root_bw,
            np.abs(
                snap_z_acceleration
            ),
        ),

    "com_acc":
        correlation(
            root_bw,
            com_acc_mag,
        ),

    "max_foot_speed":
        correlation(
            root_bw,
            max_foot_speed,
        ),

    "max_foot_xy_speed":
        correlation(
            root_bw,
            max_foot_speed_xy,
        ),

    "max_foot_vz":
        correlation(
            root_bw,
            max_foot_vz_abs,
        ),

    "foot_angular_speed":
        correlation(
            root_bw,
            max_foot_angspeed,
        ),

    "torque_ratio":
        correlation(
            root_bw,
            max_tau_ratio,
        ),

    "contact_normal":
        correlation(
            root_bw,
            contact_normal_bw,
        ),

    "com_support_distance":
        correlation(
            root_bw,
            com_support_distance,
        ),
}


for name, value in correlations.items():

    print(
        f"{name:26s}:",
        f"{value:+.3f}",
    )


# ================================================================
# TRANSITION EVENTS
# ================================================================

print()
print("=" * 210)
print("SUPPORT TRANSITIONS")
print("=" * 210)

print(
    "events:",
    len(
        transition_frames
    ),
)


event_summary = []


for event_index, event_frame in enumerate(
    transition_frames
):

    old_phase = phase_name(
        support[
            event_frame - 1
        ]
    )


    new_phase = phase_name(
        support[
            event_frame
        ]
    )


    old_mask = support[
        event_frame - 1
    ]


    new_mask = support[
        event_frame
    ]


    added = (
        new_mask
        &
        ~old_mask
    )


    removed = (
        old_mask
        &
        ~new_mask
    )


    added_names = []


    if added[
        0
    ]:

        added_names.append(
            "L"
        )


    if added[
        1
    ]:

        added_names.append(
            "R"
        )


    removed_names = []


    if removed[
        0
    ]:

        removed_names.append(
            "L"
        )


    if removed[
        1
    ]:

        removed_names.append(
            "R"
        )


    lo = max(
        0,
        event_frame
        -
        WINDOW,
    )


    hi = min(
        N,
        event_frame
        +
        WINDOW
        +
        1,
    )


    window_indices = np.arange(
        lo,
        hi,
        dtype=np.int32,
    )


    local_root_bw = root_bw[
        window_indices
    ]


    peak_local_index = int(
        np.argmax(
            local_root_bw
        )
    )


    peak_frame = int(
        window_indices[
            peak_local_index
        ]
    )


    peak_offset = (
        peak_frame
        -
        event_frame
    )


    dominant_axis_index = int(
        np.argmax(
            np.abs(
                root_force_bw_components[
                    peak_frame
                ]
            )
        )
    )


    dominant_axis = [
        "X",
        "Y",
        "Z",
    ][
        dominant_axis_index
    ]


    entering_linear = float(
        "nan"
    )

    entering_xy = float(
        "nan"
    )

    entering_vz = float(
        "nan"
    )

    entering_ang = float(
        "nan"
    )


    if added[
        0
    ]:

        entering_linear = float(
            np.linalg.norm(
                left_foot_vel[
                    event_frame
                ]
            )
        )

        entering_xy = float(
            left_foot_speed_xy[
                event_frame
            ]
        )

        entering_vz = float(
            left_foot_vel[
                event_frame,
                2
            ]
        )

        entering_ang = float(
            left_foot_angular_speed[
                event_frame
            ]
        )


    elif added[
        1
    ]:

        entering_linear = float(
            np.linalg.norm(
                right_foot_vel[
                    event_frame
                ]
            )
        )

        entering_xy = float(
            right_foot_speed_xy[
                event_frame
            ]
        )

        entering_vz = float(
            right_foot_vel[
                event_frame,
                2
            ]
        )

        entering_ang = float(
            right_foot_angular_speed[
                event_frame
            ]
        )


    leaving_linear = float(
        "nan"
    )

    leaving_vz = float(
        "nan"
    )


    if removed[
        0
    ]:

        leaving_linear = float(
            np.linalg.norm(
                left_foot_vel[
                    event_frame
                ]
            )
        )

        leaving_vz = float(
            left_foot_vel[
                event_frame,
                2
            ]
        )


    elif removed[
        1
    ]:

        leaving_linear = float(
            np.linalg.norm(
                right_foot_vel[
                    event_frame
                ]
            )
        )

        leaving_vz = float(
            right_foot_vel[
                event_frame,
                2
            ]
        )


    event_entry = {

        "event_index":
            event_index,

        "frame":
            event_frame,

        "old_phase":
            old_phase,

        "new_phase":
            new_phase,

        "added":
            "".join(
                added_names
            )
            or "-",

        "removed":
            "".join(
                removed_names
            )
            or "-",

        "peak_frame":
            peak_frame,

        "peak_offset":
            peak_offset,

        "peak_root_bw":
            float(
                root_bw[
                    peak_frame
                ]
            ),

        "dominant_force_axis":
            dominant_axis,

        "peak_tau_ratio":
            float(
                max_tau_ratio[
                    peak_frame
                ]
            ),

        "peak_tau_joint":
            JOINT_NAMES[
                max_tau_joint[
                    peak_frame
                ]
            ],

        "peak_snap_az":
            float(
                snap_z_acceleration[
                    peak_frame
                ]
            ),

        "peak_raw_az":
            float(
                raw_qacc[
                    peak_frame,
                    2
                ]
            ),

        "peak_com_acc":
            float(
                com_acc_mag[
                    peak_frame
                ]
            ),

        "enter_speed":
            entering_linear,

        "enter_xy":
            entering_xy,

        "enter_vz":
            entering_vz,

        "enter_ang":
            entering_ang,

        "leave_speed":
            leaving_linear,

        "leave_vz":
            leaving_vz,
    }


    event_summary.append(
        event_entry
    )


    print()
    print("-" * 210)

    print(
        f"EVENT {event_index:02d} | "
        f"frame {event_frame:03d} | "
        f"{old_phase} -> {new_phase} | "
        f"added={event_entry['added']} "
        f"removed={event_entry['removed']}"
    )


    print(
        f"peak residual in +/-{WINDOW} frames: "
        f"frame={peak_frame:03d} "
        f"offset={peak_offset:+d} "
        f"root={root_bw[peak_frame]:.3f} BW "
        f"dominant={dominant_axis} "
        f"tau={max_tau_ratio[peak_frame]:.3f} "
        f"({JOINT_NAMES[max_tau_joint[peak_frame]]})"
    )


    print(
        "peak percentiles:"
    )

    print(
        f"  raw |root acc|   "
        f"{raw_root_acc_mag[peak_frame]:.3f} "
        f"({percentile_rank(raw_root_acc_mag, raw_root_acc_mag[peak_frame]):.1f} percentile)"
    )

    print(
        f"  snap-induced acc "
        f"{snap_acc_delta_mag[peak_frame]:.3f} "
        f"({percentile_rank(snap_acc_delta_mag, snap_acc_delta_mag[peak_frame]):.1f} percentile)"
    )

    print(
        f"  COM |acc|        "
        f"{com_acc_mag[peak_frame]:.3f} "
        f"({percentile_rank(com_acc_mag, com_acc_mag[peak_frame]):.1f} percentile)"
    )

    print(
        f"  max foot speed   "
        f"{max_foot_speed[peak_frame]:.3f} "
        f"({percentile_rank(max_foot_speed, max_foot_speed[peak_frame]):.1f} percentile)"
    )


    if np.isfinite(
        entering_linear
    ):

        print(
            "entering-foot state at support change:"
        )

        print(
            f"  speed={entering_linear:.3f} m/s "
            f"XY={entering_xy:.3f} m/s "
            f"Vz={entering_vz:+.3f} m/s "
            f"angular={math.degrees(entering_ang):.1f} deg/s"
        )


    if np.isfinite(
        leaving_linear
    ):

        print(
            "leaving-foot state at support change:"
        )

        print(
            f"  speed={leaving_linear:.3f} m/s "
            f"Vz={leaving_vz:+.3f} m/s"
        )


    # ------------------------------------------------------------
    # TABLE 1: PHASE / KINEMATICS
    # ------------------------------------------------------------

    print()
    print(
        "KINEMATICS"
    )

    print(
        "frm off Sup Cnt "
        " Lclr  Rclr    dz "
        " rawVz rawAz snapAz   dAz "
        " COMa  Cdist "
        " Lxy   Lvz   Rxy   Rvz"
    )


    for frame in window_indices:

        marker = (
            "*"
            if frame
            ==
            event_frame
            else " "
        )


        print(
            f"{frame:3d}{marker} "
            f"{frame-event_frame:+3d} "
            f"{phase_name(support[frame]):>3s} "
            f"{phase_name(contact_mask[frame]):>3s} "
            f"{1000*left_clearance[frame]:6.2f} "
            f"{1000*right_clearance[frame]:6.2f} "
            f"{1000*snap_z[frame]:6.2f} "
            f"{raw_qvel[frame,2]:+6.3f} "
            f"{raw_qacc[frame,2]:+6.2f} "
            f"{snapped_qacc[frame,2]:+6.2f} "
            f"{snap_root_acc_delta[frame,2]:+6.2f} "
            f"{com_acc_mag[frame]:6.2f} "
            f"{1000*com_support_distance[frame]:6.1f} "
            f"{left_foot_speed_xy[frame]:5.3f} "
            f"{left_foot_vel[frame,2]:+5.3f} "
            f"{right_foot_speed_xy[frame]:5.3f} "
            f"{right_foot_vel[frame,2]:+5.3f}"
        )


    # ------------------------------------------------------------
    # TABLE 2: DYNAMICS
    # ------------------------------------------------------------

    print()
    print(
        "DYNAMICS"
    )

    print(
        "frm off Sup Act "
        " rootBW     Fx     Fy     Fz "
        " rootT "
        " tauMax joint                       "
        " nBW  nCon "
        " roll pitch"
    )


    for frame in window_indices:

        marker = (
            "*"
            if frame
            ==
            event_frame
            else " "
        )


        active_phase = phase_name(
            actual_contact[
                frame
            ]
        )


        print(
            f"{frame:3d}{marker} "
            f"{frame-event_frame:+3d} "
            f"{phase_name(support[frame]):>3s} "
            f"{active_phase:>3s} "
            f"{root_bw[frame]:7.3f} "
            f"{root_force_bw_components[frame,0]:+6.2f} "
            f"{root_force_bw_components[frame,1]:+6.2f} "
            f"{root_force_bw_components[frame,2]:+6.2f} "
            f"{root_torque_scaled[frame]:6.3f} "
            f"{max_tau_ratio[frame]:7.3f} "
            f"{JOINT_NAMES[max_tau_joint[frame]]:27s} "
            f"{contact_normal_bw[frame]:5.2f} "
            f"{active_sole_contact_count[frame]:4d} "
            f"{root_rpy[frame,0]:+6.2f} "
            f"{root_rpy[frame,1]:+6.2f}"
        )


# ================================================================
# TOP RESIDUAL FRAMES
# ================================================================

print()
print("=" * 210)
print("TOP 15 ROOT-RESIDUAL FRAMES")
print("=" * 210)


top_indices = np.argsort(
    root_bw
)[
    ::-1
][
    :15
]


for rank, frame in enumerate(
    top_indices,
    start=1,
):

    nearest_transition = min(
        transition_frames,
        key=lambda t:
            abs(
                t
                -
                frame
            ),
    )


    offset = (
        frame
        -
        nearest_transition
    )


    print(
        f"{rank:02d}. "
        f"frame={frame:03d} "
        f"phase={phase_name(support[frame])} "
        f"root={root_bw[frame]:7.3f} BW "
        f"F=["
        f"{root_force_bw_components[frame,0]:+.2f},"
        f"{root_force_bw_components[frame,1]:+.2f},"
        f"{root_force_bw_components[frame,2]:+.2f}] "
        f"tau={max_tau_ratio[frame]:6.3f} "
        f"{JOINT_NAMES[max_tau_joint[frame]]:24s} "
        f"nearestTransition={nearest_transition:03d} "
        f"offset={offset:+d} "
        f"snapAz={snap_z_acceleration[frame]:+.2f} "
        f"rawAz={raw_qacc[frame,2]:+.2f} "
        f"foot={max_foot_speed[frame]:.3f}"
    )


# ================================================================
# TRANSITION EVENT SUMMARY
# ================================================================

print()
print("=" * 210)
print("TRANSITION EVENT SUMMARY")
print("=" * 210)

print(
    "evt frame phase   add rem peak off rootBW axis "
    "tauMax joint                       "
    "rawAz snapAz COMa enterSp enterVz"
)


for event in event_summary:

    print(
        f"{event['event_index']:3d} "
        f"{event['frame']:5d} "
        f"{event['old_phase']}->{event['new_phase']:<2s} "
        f"{event['added']:>3s} "
        f"{event['removed']:>3s} "
        f"{event['peak_frame']:4d} "
        f"{event['peak_offset']:+3d} "
        f"{event['peak_root_bw']:6.2f} "
        f"{event['dominant_force_axis']:>4s} "
        f"{event['peak_tau_ratio']:6.2f} "
        f"{event['peak_tau_joint']:27s} "
        f"{event['peak_raw_az']:+6.2f} "
        f"{event['peak_snap_az']:+6.2f} "
        f"{event['peak_com_acc']:6.2f} "
        f"{event['enter_speed']:7.3f} "
        f"{event['enter_vz']:+7.3f}"
    )


# ================================================================
# CSV SAVE
# ================================================================

csv_columns = [
    "frame",
    "time_s",

    "support_phase",
    "contact_label_phase",
    "actual_contact_phase",

    "is_transition_window",

    "left_clearance_mm",
    "right_clearance_mm",

    "snap_z_mm",
    "snap_z_velocity_mps",
    "snap_z_acceleration_mps2",

    "root_x",
    "root_y",
    "root_z",

    "root_vx",
    "root_vy",
    "root_vz",

    "root_ax",
    "root_ay",
    "root_az",

    "snapped_root_ax",
    "snapped_root_ay",
    "snapped_root_az",

    "snap_delta_ax",
    "snap_delta_ay",
    "snap_delta_az",

    "root_roll_deg",
    "root_pitch_deg",
    "root_yaw_deg",

    "root_wx",
    "root_wy",
    "root_wz",

    "root_alphax",
    "root_alphay",
    "root_alphaz",

    "com_x",
    "com_y",
    "com_z",

    "com_vx",
    "com_vy",
    "com_vz",

    "com_ax",
    "com_ay",
    "com_az",

    "com_support_distance_m",

    "left_foot_x",
    "left_foot_y",
    "left_foot_z",

    "left_foot_vx",
    "left_foot_vy",
    "left_foot_vz",

    "left_foot_xy_speed",
    "left_foot_angular_speed_rad_s",

    "right_foot_x",
    "right_foot_y",
    "right_foot_z",

    "right_foot_vx",
    "right_foot_vy",
    "right_foot_vz",

    "right_foot_xy_speed",
    "right_foot_angular_speed_rad_s",

    "root_force_x_bw",
    "root_force_y_bw",
    "root_force_z_bw",
    "root_force_norm_bw",

    "root_torque_scaled",

    "max_tau_ratio",
    "max_tau_joint",

    "contact_normal_bw",
    "active_sole_contact_count",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_columns,
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

                "support_phase":
                    phase_name(
                        support[
                            frame
                        ]
                    ),

                "contact_label_phase":
                    phase_name(
                        contact_mask[
                            frame
                        ]
                    ),

                "actual_contact_phase":
                    phase_name(
                        actual_contact[
                            frame
                        ]
                    ),

                "is_transition_window":
                    int(
                        transition_mask[
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

                "snap_z_mm":
                    1000
                    *
                    snap_z[
                        frame
                    ],

                "snap_z_velocity_mps":
                    snap_z_velocity[
                        frame
                    ],

                "snap_z_acceleration_mps2":
                    snap_z_acceleration[
                        frame
                    ],

                "root_x":
                    qpos[
                        frame,
                        0
                    ],

                "root_y":
                    qpos[
                        frame,
                        1
                    ],

                "root_z":
                    qpos[
                        frame,
                        2
                    ],

                "root_vx":
                    raw_qvel[
                        frame,
                        0
                    ],

                "root_vy":
                    raw_qvel[
                        frame,
                        1
                    ],

                "root_vz":
                    raw_qvel[
                        frame,
                        2
                    ],

                "root_ax":
                    raw_qacc[
                        frame,
                        0
                    ],

                "root_ay":
                    raw_qacc[
                        frame,
                        1
                    ],

                "root_az":
                    raw_qacc[
                        frame,
                        2
                    ],

                "snapped_root_ax":
                    snapped_qacc[
                        frame,
                        0
                    ],

                "snapped_root_ay":
                    snapped_qacc[
                        frame,
                        1
                    ],

                "snapped_root_az":
                    snapped_qacc[
                        frame,
                        2
                    ],

                "snap_delta_ax":
                    snap_root_acc_delta[
                        frame,
                        0
                    ],

                "snap_delta_ay":
                    snap_root_acc_delta[
                        frame,
                        1
                    ],

                "snap_delta_az":
                    snap_root_acc_delta[
                        frame,
                        2
                    ],

                "root_roll_deg":
                    root_rpy[
                        frame,
                        0
                    ],

                "root_pitch_deg":
                    root_rpy[
                        frame,
                        1
                    ],

                "root_yaw_deg":
                    root_rpy[
                        frame,
                        2
                    ],

                "root_wx":
                    raw_qvel[
                        frame,
                        3
                    ],

                "root_wy":
                    raw_qvel[
                        frame,
                        4
                    ],

                "root_wz":
                    raw_qvel[
                        frame,
                        5
                    ],

                "root_alphax":
                    raw_qacc[
                        frame,
                        3
                    ],

                "root_alphay":
                    raw_qacc[
                        frame,
                        4
                    ],

                "root_alphaz":
                    raw_qacc[
                        frame,
                        5
                    ],

                "com_x":
                    com_position[
                        frame,
                        0
                    ],

                "com_y":
                    com_position[
                        frame,
                        1
                    ],

                "com_z":
                    com_position[
                        frame,
                        2
                    ],

                "com_vx":
                    com_velocity[
                        frame,
                        0
                    ],

                "com_vy":
                    com_velocity[
                        frame,
                        1
                    ],

                "com_vz":
                    com_velocity[
                        frame,
                        2
                    ],

                "com_ax":
                    com_acceleration[
                        frame,
                        0
                    ],

                "com_ay":
                    com_acceleration[
                        frame,
                        1
                    ],

                "com_az":
                    com_acceleration[
                        frame,
                        2
                    ],

                "com_support_distance_m":
                    com_support_distance[
                        frame
                    ],

                "left_foot_x":
                    left_foot_pos[
                        frame,
                        0
                    ],

                "left_foot_y":
                    left_foot_pos[
                        frame,
                        1
                    ],

                "left_foot_z":
                    left_foot_pos[
                        frame,
                        2
                    ],

                "left_foot_vx":
                    left_foot_vel[
                        frame,
                        0
                    ],

                "left_foot_vy":
                    left_foot_vel[
                        frame,
                        1
                    ],

                "left_foot_vz":
                    left_foot_vel[
                        frame,
                        2
                    ],

                "left_foot_xy_speed":
                    left_foot_speed_xy[
                        frame
                    ],

                "left_foot_angular_speed_rad_s":
                    left_foot_angular_speed[
                        frame
                    ],

                "right_foot_x":
                    right_foot_pos[
                        frame,
                        0
                    ],

                "right_foot_y":
                    right_foot_pos[
                        frame,
                        1
                    ],

                "right_foot_z":
                    right_foot_pos[
                        frame,
                        2
                    ],

                "right_foot_vx":
                    right_foot_vel[
                        frame,
                        0
                    ],

                "right_foot_vy":
                    right_foot_vel[
                        frame,
                        1
                    ],

                "right_foot_vz":
                    right_foot_vel[
                        frame,
                        2
                    ],

                "right_foot_xy_speed":
                    right_foot_speed_xy[
                        frame
                    ],

                "right_foot_angular_speed_rad_s":
                    right_foot_angular_speed[
                        frame
                    ],

                "root_force_x_bw":
                    root_force_bw_components[
                        frame,
                        0
                    ],

                "root_force_y_bw":
                    root_force_bw_components[
                        frame,
                        1
                    ],

                "root_force_z_bw":
                    root_force_bw_components[
                        frame,
                        2
                    ],

                "root_force_norm_bw":
                    root_bw[
                        frame
                    ],

                "root_torque_scaled":
                    root_torque_scaled[
                        frame
                    ],

                "max_tau_ratio":
                    max_tau_ratio[
                        frame
                    ],

                "max_tau_joint":
                    JOINT_NAMES[
                        max_tau_joint[
                            frame
                        ]
                    ],

                "contact_normal_bw":
                    contact_normal_bw[
                        frame
                    ],

                "active_sole_contact_count":
                    active_sole_contact_count[
                        frame
                    ],
            }
        )


# ================================================================
# NPZ SAVE
# ================================================================

np.savez_compressed(
    OUTPUT_NPZ,

    qpos=
        qpos.astype(
            np.float32
        ),

    raw_qvel=
        raw_qvel.astype(
            np.float32
        ),

    raw_qacc=
        raw_qacc.astype(
            np.float32
        ),

    snapped_qpos=
        snapped_qpos.astype(
            np.float32
        ),

    snapped_qvel=
        snapped_qvel.astype(
            np.float32
        ),

    snapped_qacc=
        snapped_qacc.astype(
            np.float32
        ),

    support=
        support,

    contact_mask=
        contact_mask,

    actual_contact=
        actual_contact,

    transition_mask=
        transition_mask,

    transition_frames=
        np.asarray(
            transition_frames,
            dtype=np.int32,
        ),

    snap_z=
        snap_z.astype(
            np.float32
        ),

    snap_z_velocity=
        snap_z_velocity.astype(
            np.float32
        ),

    snap_z_acceleration=
        snap_z_acceleration.astype(
            np.float32
        ),

    snap_root_acc_delta=
        snap_root_acc_delta.astype(
            np.float32
        ),

    root_bw=
        root_bw.astype(
            np.float32
        ),

    root_force_bw_components=
        root_force_bw_components.astype(
            np.float32
        ),

    root_torque_scaled=
        root_torque_scaled.astype(
            np.float32
        ),

    max_tau_ratio=
        max_tau_ratio.astype(
            np.float32
        ),

    max_tau_joint=
        max_tau_joint,

    contact_normal_bw=
        contact_normal_bw.astype(
            np.float32
        ),

    active_sole_contact_count=
        active_sole_contact_count,

    left_foot_pos=
        left_foot_pos.astype(
            np.float32
        ),

    right_foot_pos=
        right_foot_pos.astype(
            np.float32
        ),

    left_foot_vel=
        left_foot_vel.astype(
            np.float32
        ),

    right_foot_vel=
        right_foot_vel.astype(
            np.float32
        ),

    left_clearance=
        left_clearance.astype(
            np.float32
        ),

    right_clearance=
        right_clearance.astype(
            np.float32
        ),

    com_position=
        com_position.astype(
            np.float32
        ),

    com_velocity=
        com_velocity.astype(
            np.float32
        ),

    com_acceleration=
        com_acceleration.astype(
            np.float32
        ),

    com_support_distance=
        com_support_distance.astype(
            np.float32
        ),
)


# ================================================================
# OPTIONAL VISUAL PLOT
# ================================================================

plot_written = False


try:

    import matplotlib.pyplot as plt


    time = (
        np.arange(
            N
        )
        *
        DT
    )


    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            15,
            13,
        ),
        sharex=True,
    )


    axes[
        0
    ].plot(
        time,
        root_bw,
        label="root residual / BW",
    )

    axes[
        0
    ].set_ylabel(
        "Root BW"
    )

    axes[
        0
    ].legend()


    axes[
        1
    ].plot(
        time,
        raw_qacc[
            :,
            2
        ],
        label="raw root az",
    )

    axes[
        1
    ].plot(
        time,
        snap_z_acceleration,
        label="snap-induced az",
    )

    axes[
        1
    ].set_ylabel(
        "m/s²"
    )

    axes[
        1
    ].legend()


    axes[
        2
    ].plot(
        time,
        left_foot_speed_xy,
        label="left foot XY",
    )

    axes[
        2
    ].plot(
        time,
        right_foot_speed_xy,
        label="right foot XY",
    )

    axes[
        2
    ].set_ylabel(
        "m/s"
    )

    axes[
        2
    ].legend()


    axes[
        3
    ].plot(
        time,
        contact_normal_bw,
        label="contact normal / BW",
    )

    axes[
        3
    ].plot(
        time,
        max_tau_ratio,
        label="max torque / limit",
    )

    axes[
        3
    ].set_ylabel(
        "ratio"
    )

    axes[
        3
    ].legend()


    phase_numeric = np.zeros(
        N,
        dtype=np.float64,
    )


    for i in range(N):

        phase = phase_name(
            support[
                i
            ]
        )


        phase_numeric[
            i
        ] = {
            "N":
                0.0,
            "L":
                1.0,
            "R":
                2.0,
            "B":
                3.0,
        }[
            phase
        ]


    axes[
        4
    ].step(
        time,
        phase_numeric,
        where="post",
        label="support phase",
    )

    axes[
        4
    ].set_yticks(
        [
            0,
            1,
            2,
            3,
        ],
        [
            "N",
            "L",
            "R",
            "B",
        ],
    )

    axes[
        4
    ].set_ylabel(
        "Support"
    )

    axes[
        4
    ].set_xlabel(
        "Time [s]"
    )


    for axis in axes:

        for frame in transition_frames:

            axis.axvline(
                frame
                *
                DT,
                alpha=0.25,
            )


    fig.suptitle(
        "G1 medium_08 transition microscope"
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_PNG,
        dpi=160,
    )


    plt.close(
        fig
    )


    plot_written = True


except Exception as exc:

    print()
    print(
        "Plot skipped:",
        repr(
            exc
        ),
    )


# ================================================================
# FINAL DIAGNOSTIC CLASSIFICATION
# ================================================================

transition_snap_p95 = float(
    np.percentile(
        np.abs(
            snap_z_acceleration[
                transition_mask
            ]
        ),
        95,
    )
)


steady_snap_p95 = float(
    np.percentile(
        np.abs(
            snap_z_acceleration[
                steady_single
            ]
        ),
        95,
    )
)


transition_foot_p95 = float(
    np.percentile(
        max_foot_speed[
            transition_mask
        ],
        95,
    )
)


steady_foot_p95 = float(
    np.percentile(
        max_foot_speed[
            steady_single
        ],
        95,
    )
)


transition_raw_acc_p95 = float(
    np.percentile(
        raw_root_acc_mag[
            transition_mask
        ],
        95,
    )
)


steady_raw_acc_p95 = float(
    np.percentile(
        raw_root_acc_mag[
            steady_single
        ],
        95,
    )
)


print()
print("=" * 210)
print("FINAL TRANSITION DIAGNOSIS")
print("=" * 210)

print(
    "root residual:",
    f"steady single={single95:.3f} BW",
    f"transition={transition95:.3f} BW",
)

print(
    "snap |az| p95:",
    f"steady={steady_snap_p95:.3f}",
    f"transition={transition_snap_p95:.3f}",
    "m/s²",
)

print(
    "max foot speed p95:",
    f"steady={steady_foot_p95:.3f}",
    f"transition={transition_foot_p95:.3f}",
    "m/s",
)

print(
    "raw root |acc| p95:",
    f"steady={steady_raw_acc_p95:.3f}",
    f"transition={transition_raw_acc_p95:.3f}",
    "m/s²",
)

print()
print(
    "Key correlations:"
)

print(
    " root residual vs snap-induced acceleration:",
    f"{correlations['snap_delta_acc']:+.3f}",
)

print(
    " root residual vs raw root acceleration:",
    f"{correlations['raw_root_acc']:+.3f}",
)

print(
    " root residual vs max foot speed:",
    f"{correlations['max_foot_speed']:+.3f}",
)

print(
    " root residual vs contact normal:",
    f"{correlations['contact_normal']:+.3f}",
)

print(
    " root residual vs torque ratio:",
    f"{correlations['torque_ratio']:+.3f}",
)


# This is deliberately conservative:
# no single correlation is treated as proof.

if (
    transition_snap_p95
    >
    2.0
    *
    max(
        steady_snap_p95,
        1e-9,
    )
    and
    correlations[
        "snap_delta_acc"
    ]
    >
    0.50
):

    print()
    print(
        "PRIMARY FLAG:"
    )

    print(
        "CONTACT-SNAP / ROOT-Z DERIVATIVE DISCONTINUITY "
        "IS STRONGLY ASSOCIATED WITH TRANSITION SPIKES."
    )

    print(
        "NEXT TEST:"
    )

    print(
        "Build a smooth support-aware root-Z/contact projection "
        "across transition windows, then rerun this exact audit."
    )


elif (
    transition_foot_p95
    >
    1.5
    *
    max(
        steady_foot_p95,
        1e-9,
    )
    and
    correlations[
        "max_foot_speed"
    ]
    >
    0.40
):

    print()
    print(
        "PRIMARY FLAG:"
    )

    print(
        "FOOT TOUCHDOWN/LIFTOFF KINEMATICS ARE STRONGLY "
        "ASSOCIATED WITH THE TRANSITION SPIKES."
    )

    print(
        "NEXT TEST:"
    )

    print(
        "Repair support/contact timing and touchdown velocity "
        "without changing steady-support motion."
    )


elif (
    transition_raw_acc_p95
    >
    1.5
    *
    max(
        steady_raw_acc_p95,
        1e-9,
    )
    and
    correlations[
        "raw_root_acc"
    ]
    >
    0.40
):

    print()
    print(
        "PRIMARY FLAG:"
    )

    print(
        "RAW ROOT / CENTROIDAL ACCELERATION IS STRONGLY "
        "ASSOCIATED WITH TRANSITION SPIKES."
    )

    print(
        "NEXT TEST:"
    )

    print(
        "Repair the root momentum trajectory only inside "
        "support-transfer windows."
    )


else:

    print()
    print(
        "PRIMARY FLAG:"
    )

    print(
        "NO SINGLE SCALAR EXPLAINS THE TRANSITION FAILURE."
    )

    print(
        "The evidence points to a coupled contact-timing / "
        "root-momentum / touchdown problem."
    )

    print(
        "NEXT TEST:"
    )

    print(
        "Use the event tables to identify the common transition "
        "pattern, then construct one constrained transition-only "
        "repair rather than another whole-clip optimizer."
    )


print()
print("=" * 210)
print("OUTPUTS")
print("=" * 210)

print(
    "CSV:",
    OUTPUT_CSV,
)

print(
    "NPZ:",
    OUTPUT_NPZ,
)

if plot_written:

    print(
        "PNG:",
        OUTPUT_PNG,
    )

else:

    print(
        "PNG: not written (matplotlib unavailable or plot error)"
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

print("=" * 210)


env.close()
