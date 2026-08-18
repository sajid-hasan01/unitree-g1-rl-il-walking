from pathlib import Path
import math
import sys

import mujoco
import numpy as np
import scipy
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

RAMP_FRAMES = 4
MIN_SEGMENT = 5

CONTACT_DEPTH = 1e-6
TRANSITION_WINDOW = 2

# Foot-pose residual scales.
POS_SCALE = 0.0020
ROT_SCALE = math.radians(2.0)

# Motion preservation.
REG_WEIGHT = 0.25

# Bounds are generalized-coordinate increments.
ROOT_XYZ_LIMIT = np.asarray(
    [0.035, 0.035, 0.020],
    dtype=np.float64,
)

# Root rotational velocity-vector increments.
ROOT_ROT_LIMIT = np.asarray(
    [
        math.radians(4.0),
        math.radians(4.0),
        math.radians(2.0),
    ],
    dtype=np.float64,
)

# hip pitch, hip roll, hip yaw, knee,
# ankle pitch, ankle roll
LEG_LIMIT = np.asarray(
    [
        0.12,
        0.10,
        0.06,
        0.14,
        0.12,
        0.10,
    ],
    dtype=np.float64,
)


OUTPUT_REFERENCE = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_support_pose_candidate.npz"
)

OUTPUT_DIAGNOSTIC = (
    ROOT
    / "results"
    / "g1_dynamic_v2_support_pose_pass2a.npz"
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
    mujoco.mj_getTotalmass(model)
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


if not all(
    int(model.geom_bodyid[g])
    == LEFT_BODY
    for g in LEFT_GEOMS
):

    raise RuntimeError(
        "Left sole geoms do not share one rigid body."
    )


if not all(
    int(model.geom_bodyid[g])
    == RIGHT_BODY
    for g in RIGHT_GEOMS
):

    raise RuntimeError(
        "Right sole geoms do not share one rigid body."
    )


print("=" * 200)
print("G1 DYNAMIC V2 — PASS 2A")
print("FULL STANCE-FOOT SE(3) CONSTRAINED PROJECTION")
print("BOUNDED ROOT + SUPPORT-LEG CORRECTIONS")
print("STARTING FROM ORIGINAL MEDIUM_02")
print("FOLLOWED BY CONSISTENT 1-MICRON INVERSE AUDIT")
print("NO CEM / NO PPO")
print("=" * 200)

print("source:", env.reference_path)
print("frames:", N)
print("frequency:", env.reference_fps)
print("SciPy:", scipy.__version__)
print("left support body:", LEFT_BODY)
print("right support body:", RIGHT_BODY)
print("body weight:", f"{BODY_WEIGHT:.2f} N")


# ================================================================
# QUATERNION HELPERS — wxyz
# ================================================================

def normalize_quat(q):

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


def quat_conj(q):

    q = normalize_quat(q)

    return np.asarray(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ],
        dtype=np.float64,
    )


def quat_mul(a, b):

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.asarray(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )


def quat_error_vector(
    target,
    current,
):

    q = quat_mul(
        normalize_quat(target),
        quat_conj(current),
    )

    if q[0] < 0.0:
        q = -q

    q = normalize_quat(q)

    v = q[1:4]

    nv = float(
        np.linalg.norm(v)
    )

    if nv < 1e-12:
        return np.zeros(
            3,
            dtype=np.float64,
        )

    angle = (
        2.0
        * math.atan2(
            nv,
            max(q[0], 1e-12),
        )
    )

    return (
        v / nv
        * angle
    )


def quat_angle(
    a,
    b,
):

    return float(
        np.linalg.norm(
            quat_error_vector(
                a,
                b,
            )
        )
    )


def slerp(
    q0,
    q1,
    alpha,
):

    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)

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

        return normalize_quat(
            (1.0-alpha)*q0
            +
            alpha*q1
        )

    theta = math.acos(dot)

    s = math.sin(theta)

    return normalize_quat(
        (
            math.sin(
                (1.0-alpha)*theta
            )
            / s
        ) * q0
        +
        (
            math.sin(
                alpha*theta
            )
            / s
        ) * q1
    )


# ================================================================
# SUPPORT SEGMENTS
# ================================================================

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

    end = frame - 1

    if (
        end - start + 1
        >= MIN_SEGMENT
    ):

        segments.append(
            (
                start,
                end,
                side,
            )
        )


print()
print("single-support segments:", len(segments))

for i, (
    start,
    end,
    side,
) in enumerate(segments):

    print(
        f"{i:02d}: "
        f"{'LEFT' if side == 0 else 'RIGHT'} "
        f"{start:03d}..{end:03d} "
        f"N={end-start+1}"
    )


# ================================================================
# TRANSITION MASK — FROZEN DEFINITION
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

        transition[lo:hi] = True


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
print(
    "FROZEN AUDIT MASK:"
)

print(
    "single-support frames:",
    int(np.sum(single_support)),
)

print(
    "steady-single frames:",
    int(np.sum(steady_single)),
)

print(
    "steady-double frames:",
    int(np.sum(steady_double)),
)

print(
    "transition frames:",
    int(np.sum(transition)),
)


# ================================================================
# WINDOW
# ================================================================

def smoothstep(x):

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x*x*(3.0 - 2.0*x)
    )


def segment_window(length):

    result = np.zeros(
        length,
        dtype=np.float64,
    )

    for i in range(length):

        distance = min(
            i,
            length - 1 - i,
        )

        alpha = min(
            1.0,
            distance
            / float(
                max(
                    RAMP_FRAMES,
                    1,
                )
            ),
        )

        result[i] = smoothstep(
            alpha
        )

    return result


# ================================================================
# FOOT BODY POSES
# ================================================================

pose_data = mujoco.MjData(
    model
)


def body_pose(
    qpos,
    body_id,
):

    pose_data.qpos[:] = qpos
    pose_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        pose_data,
    )

    return (
        pose_data.xpos[
            body_id
        ].copy(),
        normalize_quat(
            pose_data.xquat[
                body_id
            ].copy()
        ),
    )


old_left_pos = np.zeros(
    (N, 3),
    dtype=np.float64,
)

old_right_pos = np.zeros_like(
    old_left_pos
)

old_left_quat = np.zeros(
    (N, 4),
    dtype=np.float64,
)

old_right_quat = np.zeros_like(
    old_left_quat
)


for frame in range(N):

    (
        old_left_pos[frame],
        old_left_quat[frame],
    ) = body_pose(
        old_qpos[frame],
        LEFT_BODY,
    )

    (
        old_right_pos[frame],
        old_right_quat[frame],
    ) = body_pose(
        old_qpos[frame],
        RIGHT_BODY,
    )


# ================================================================
# GENERALIZED DOF SELECTION
# ================================================================

# Free-root qvel addresses are 0..5.
ROOT_DOFS = np.asarray(
    [
        0, 1, 2,
        3, 4, 5,
    ],
    dtype=np.int32,
)


LEFT_JOINT_INDICES = [
    0, 1, 2, 3, 4, 5
]

RIGHT_JOINT_INDICES = [
    6, 7, 8, 9, 10, 11
]


def selected_dofs(side):

    leg_indices = (
        LEFT_JOINT_INDICES
        if side == 0
        else RIGHT_JOINT_INDICES
    )

    leg_dofs = np.asarray(
        [
            env.vaddrs[i]
            for i in leg_indices
        ],
        dtype=np.int32,
    )

    selected = np.concatenate(
        [
            ROOT_DOFS,
            leg_dofs,
        ]
    )

    limits = np.concatenate(
        [
            ROOT_XYZ_LIMIT,
            ROOT_ROT_LIMIT,
            LEG_LIMIT,
        ]
    )

    return (
        selected,
        limits,
        leg_indices,
    )


# ================================================================
# PER-FRAME BOUNDED POSE PROJECTION
# ================================================================

ik_data = mujoco.MjData(
    model
)


def qpos_from_delta(
    original,
    selected,
    delta,
):

    qpos = np.asarray(
        original,
        dtype=np.float64,
    ).copy()

    dq = np.zeros(
        model.nv,
        dtype=np.float64,
    )

    dq[selected] = delta

    mujoco.mj_integratePos(
        model,
        qpos,
        dq,
        1.0,
    )

    return qpos


def pose_residual(
    delta,
    original_qpos,
    body_id,
    target_pos,
    target_quat,
    selected,
    limits,
):

    qpos = qpos_from_delta(
        original_qpos,
        selected,
        delta,
    )

    ik_data.qpos[:] = qpos
    ik_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        ik_data,
    )

    current_pos = (
        ik_data.xpos[
            body_id
        ]
    )

    current_quat = (
        ik_data.xquat[
            body_id
        ]
    )

    position_error = (
        current_pos
        -
        target_pos
    ) / POS_SCALE

    orientation_error = (
        quat_error_vector(
            target_quat,
            current_quat,
        )
        / ROT_SCALE
    )

    regularization = (
        REG_WEIGHT
        * delta
        / limits
    )

    return np.concatenate(
        [
            position_error,
            orientation_error,
            regularization,
        ]
    )


new_qpos = old_qpos.copy()

frame_delta = np.zeros(
    (
        N,
        12,
    ),
    dtype=np.float64,
)

optimizer_cost = np.zeros(
    N,
    dtype=np.float64,
)

optimizer_nfev = np.zeros(
    N,
    dtype=np.int32,
)


print()
print("=" * 200)
print("RUNNING STANCE-FOOT POSE PROJECTION")
print("=" * 200)


for segment_index, (
    start,
    end,
    side,
) in enumerate(segments):

    body = (
        LEFT_BODY
        if side == 0
        else RIGHT_BODY
    )

    original_positions = (
        old_left_pos
        if side == 0
        else old_right_pos
    )

    original_quaternions = (
        old_left_quat
        if side == 0
        else old_right_quat
    )

    length = (
        end - start + 1
    )

    window = segment_window(
        length
    )


    # Core region used to define locked world pose.
    core_start = min(
        end,
        start + RAMP_FRAMES,
    )

    core_end = max(
        core_start,
        end - RAMP_FRAMES,
    )

    core_indices = np.arange(
        core_start,
        core_end + 1,
        dtype=np.int32,
    )


    lock_position = np.median(
        original_positions[
            core_indices
        ],
        axis=0,
    )


    mid_frame = int(
        core_indices[
            len(core_indices) // 2
        ]
    )


    lock_quaternion = (
        original_quaternions[
            mid_frame
        ].copy()
    )


    (
        selected,
        limits,
        leg_indices,
    ) = selected_dofs(
        side
    )


    segment_position_before = []
    segment_rotation_before = []

    segment_position_after = []
    segment_rotation_after = []


    for local_index, frame in enumerate(
        range(
            start,
            end + 1,
        )
    ):

        weight = float(
            window[
                local_index
            ]
        )


        if weight <= 1e-8:
            continue


        original_position = (
            original_positions[
                frame
            ]
        )


        original_quaternion = (
            original_quaternions[
                frame
            ]
        )


        target_position = (
            (1.0 - weight)
            * original_position
            +
            weight
            * lock_position
        )


        target_quaternion = slerp(
            original_quaternion,
            lock_quaternion,
            weight,
        )


        segment_position_before.append(
            np.linalg.norm(
                original_position
                -
                target_position
            )
        )


        segment_rotation_before.append(
            quat_angle(
                target_quaternion,
                original_quaternion,
            )
        )


        result = least_squares(
            pose_residual,

            x0=np.zeros(
                len(selected),
                dtype=np.float64,
            ),

            bounds=(
                -limits,
                limits,
            ),

            method="trf",

            loss="soft_l1",

            f_scale=1.0,

            max_nfev=45,

            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,

            args=(
                old_qpos[
                    frame
                ],
                body,
                target_position,
                target_quaternion,
                selected,
                limits,
            ),
        )


        delta = (
            result.x
        )


        projected = qpos_from_delta(
            old_qpos[
                frame
            ],
            selected,
            delta,
        )


        # Clamp hinge joints to compiled physical range.
        for joint_index in leg_indices:

            jid = env.joint_ids[
                joint_index
            ]

            if model.jnt_limited[
                jid
            ]:

                qadr = int(
                    model.jnt_qposadr[
                        jid
                    ]
                )

                lo, hi = (
                    model.jnt_range[
                        jid
                    ]
                )

                projected[
                    qadr
                ] = np.clip(
                    projected[
                        qadr
                    ],
                    lo,
                    hi,
                )


        new_qpos[
            frame
        ] = projected


        frame_delta[
            frame
        ] = delta


        optimizer_cost[
            frame
        ] = float(
            result.cost
        )


        optimizer_nfev[
            frame
        ] = int(
            result.nfev
        )


        (
            new_position,
            new_quaternion,
        ) = body_pose(
            projected,
            body,
        )


        segment_position_after.append(
            np.linalg.norm(
                new_position
                -
                target_position
            )
        )


        segment_rotation_after.append(
            quat_angle(
                target_quaternion,
                new_quaternion,
            )
        )


    before_pos = (
        1000.0
        * np.asarray(
            segment_position_before
        )
    )

    after_pos = (
        1000.0
        * np.asarray(
            segment_position_after
        )
    )

    before_rot = (
        np.degrees(
            np.asarray(
                segment_rotation_before
            )
        )
    )

    after_rot = (
        np.degrees(
            np.asarray(
                segment_rotation_after
            )
        )
    )


    print(
        f"segment={segment_index:02d} "
        f"{'L' if side == 0 else 'R'} "
        f"{start:03d}-{end:03d} "
        f"posP95="
        f"{np.percentile(before_pos,95):.2f}"
        f"->{np.percentile(after_pos,95):.2f}mm "
        f"rotP95="
        f"{np.percentile(before_rot,95):.2f}"
        f"->{np.percentile(after_rot,95):.2f}deg"
    )


# ================================================================
# DIFFERENTIATE CANDIDATE QPOS
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


new_qvel = differentiate_qpos(
    new_qpos
)


# ================================================================
# CANDIDATE FOOT-POSE TRAJECTORIES
# ================================================================

new_left_pos = np.zeros_like(
    old_left_pos
)

new_right_pos = np.zeros_like(
    old_right_pos
)

new_left_quat = np.zeros_like(
    old_left_quat
)

new_right_quat = np.zeros_like(
    old_right_quat
)


for frame in range(N):

    (
        new_left_pos[frame],
        new_left_quat[frame],
    ) = body_pose(
        new_qpos[
            frame
        ],
        LEFT_BODY,
    )


    (
        new_right_pos[frame],
        new_right_quat[frame],
    ) = body_pose(
        new_qpos[
            frame
        ],
        RIGHT_BODY,
    )


# ================================================================
# CORE STANCE FOOT SPEED — LINEAR + ANGULAR
# ================================================================

def stance_speeds(
    left_pos,
    left_quat,
    right_pos,
    right_quat,
):

    linear = []
    angular = []


    for (
        start,
        end,
        side,
    ) in segments:

        local_start = min(
            end,
            start + RAMP_FRAMES,
        )

        local_end = max(
            local_start,
            end - RAMP_FRAMES,
        )


        position = (
            left_pos
            if side == 0
            else right_pos
        )

        quaternion = (
            left_quat
            if side == 0
            else right_quat
        )


        for frame in range(
            local_start,
            local_end,
        ):

            linear.append(
                np.linalg.norm(
                    position[
                        frame + 1
                    ]
                    -
                    position[
                        frame
                    ]
                ) / DT
            )


            angular.append(
                quat_angle(
                    quaternion[
                        frame + 1
                    ],
                    quaternion[
                        frame
                    ],
                ) / DT
            )


    return (
        np.asarray(
            linear,
            dtype=np.float64,
        ),
        np.asarray(
            angular,
            dtype=np.float64,
        ),
    )


(
    old_linear_speed,
    old_angular_speed,
) = stance_speeds(
    old_left_pos,
    old_left_quat,
    old_right_pos,
    old_right_quat,
)


(
    new_linear_speed,
    new_angular_speed,
) = stance_speeds(
    new_left_pos,
    new_left_quat,
    new_right_pos,
    new_right_quat,
)


# ================================================================
# CORRECTION SIZE
# ================================================================

root_translation_delta = (
    new_qpos[
        :,
        0:3
    ]
    -
    old_qpos[
        :,
        0:3
    ]
)


root_translation_mag = np.linalg.norm(
    root_translation_delta,
    axis=1,
)


root_rotation_delta = np.asarray(
    [
        quat_angle(
            new_qpos[
                i,
                3:7
            ],
            old_qpos[
                i,
                3:7
            ],
        )
        for i in range(N)
    ],
    dtype=np.float64,
)


joint_delta = (
    new_qpos[
        :,
        7:36
    ]
    -
    old_qpos[
        :,
        7:36
    ]
)


joint_delta_max_frame = np.max(
    np.abs(
        joint_delta
    ),
    axis=1,
)


print()
print("=" * 200)
print("PASS-2A GEOMETRIC RESULT")
print("=" * 200)

print(
    "root translation:",
    f"p50={1000*np.percentile(root_translation_mag,50):.2f}mm",
    f"p95={1000*np.percentile(root_translation_mag,95):.2f}mm",
    f"max={1000*np.max(root_translation_mag):.2f}mm",
)

print(
    "root orientation change:",
    f"p50={np.degrees(np.percentile(root_rotation_delta,50)):.2f}deg",
    f"p95={np.degrees(np.percentile(root_rotation_delta,95)):.2f}deg",
    f"max={np.degrees(np.max(root_rotation_delta)):.2f}deg",
)

print(
    "max lower-joint change/frame:",
    f"p95={np.degrees(np.percentile(joint_delta_max_frame,95)):.2f}deg",
    f"max={np.degrees(np.max(joint_delta_max_frame)):.2f}deg",
)

print()
print(
    "CORE STANCE LINEAR SPEED:"
)

print(
    "original:",
    f"p50={np.percentile(old_linear_speed,50):.4f}",
    f"p95={np.percentile(old_linear_speed,95):.4f}",
    "m/s",
)

print(
    "candidate:",
    f"p50={np.percentile(new_linear_speed,50):.4f}",
    f"p95={np.percentile(new_linear_speed,95):.4f}",
    "m/s",
)

print()
print(
    "CORE STANCE ANGULAR SPEED:"
)

print(
    "original:",
    f"p50={np.degrees(np.percentile(old_angular_speed,50)):.2f}",
    f"p95={np.degrees(np.percentile(old_angular_speed,95)):.2f}",
    "deg/s",
)

print(
    "candidate:",
    f"p50={np.degrees(np.percentile(new_angular_speed,50)):.2f}",
    f"p95={np.degrees(np.percentile(new_angular_speed,95)):.2f}",
    "deg/s",
)


# ================================================================
# CONTACT-SNAPPED CONSISTENT INVERSE AUDIT
# ================================================================

audit_probe = mujoco.MjData(
    model
)


def expected_support(frame):

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


    lc = sole_clearance(
        audit_probe,
        LEFT_GEOMS,
    )

    rc = sole_clearance(
        audit_probe,
        RIGHT_GEOMS,
    )


    expected = expected_support(
        frame
    )

    candidates = []


    if expected[0]:

        candidates.append(
            (
                0,
                lc,
            )
        )


    if expected[1]:

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
        key=lambda x: x[1],
    )


def build_contact_snapped(
    qpos_trajectory,
):

    snapped = (
        qpos_trajectory.copy()
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


        dz = min(
            0.0,
            dz,
        )


        dz = max(
            -0.020,
            dz,
        )


        snapped[
            frame,
            2
        ] += dz


    return (
        snapped,
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
    label,
    qpos_trajectory,
):

    print()
    print(
        "inverse audit:",
        label,
    )


    (
        snapped_qpos,
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


    def safe_p95(mask):

        if not np.any(mask):
            return float("nan")

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

        "max_torque_ratio":
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
    }


baseline = inverse_audit(
    "ORIGINAL",
    old_qpos,
)


candidate = inverse_audit(
    "PASS2A",
    new_qpos,
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 200)
print("ORIGINAL VS PASS-2A DYNAMIC FEASIBILITY")
print("=" * 200)

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

print("-" * 200)


for name, result in (
    (
        "ORIGINAL",
        baseline,
    ),
    (
        "PASS2A",
        candidate,
    ),
):

    print(
        f"{name:12s} "
        f"{100*result['anchor_validity']:6.1f}% "
        f"{result['overall_p50']:9.3f} "
        f"{result['overall_p95']:9.3f} "
        f"{result['overall_max']:9.3f} "
        f"{result['single_p95']:9.3f} "
        f"{result['double_p95']:9.3f} "
        f"{result['transition_p95']:9.3f} "
        f"{result['tau_p95']:9.3f} "
        f"{100*result['overlimit']:8.1f}%"
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
print("=" * 200)
print("PASS-2A DECISION")
print("=" * 200)

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
# SAVE COMPATIBLE CANDIDATE
# ================================================================

new_joint_q = (
    new_qpos[
        :,
        7:36
    ]
)

new_joint_qd = (
    new_qvel[
        :,
        6:35
    ]
)


old_joint_q = (
    old_qpos[
        :,
        7:36
    ]
)

old_joint_qd = (
    old_qvel[
        :,
        6:35
    ]
)


reference_path = Path(
    env.reference_path
)


with np.load(
    reference_path,
    allow_pickle=True,
) as source:

    payload = {
        key:
            source[key].copy()

        for key in source.files
    }


def numeric_array(value):

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
        numeric_array(value)
        and
        value.shape == target.shape
        and
        np.allclose(
            value,
            target,
            rtol=1e-5,
            atol=5e-6,
            equal_nan=True,
        )
    )


replacements = [
    (
        old_qpos,
        new_qpos,
        "full_qpos",
    ),
    (
        old_qvel,
        new_qvel,
        "full_qvel",
    ),
    (
        old_joint_q,
        new_joint_q,
        "joint_q",
    ),
    (
        old_joint_qd,
        new_joint_qd,
        "joint_qd",
    ),
    (
        old_qpos[:, 0:3],
        new_qpos[:, 0:3],
        "root_pos",
    ),
    (
        old_qpos[:, 3:7],
        new_qpos[:, 3:7],
        "root_quat",
    ),
    (
        old_qvel[:, 0:3],
        new_qvel[:, 0:3],
        "root_linvel",
    ),
    (
        old_qvel[:, 3:6],
        new_qvel[:, 3:6],
        "root_angvel",
    ),
]


patched = {}


for (
    old_array,
    new_array,
    label,
) in replacements:

    patched[
        label
    ] = []


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
        "Could not identify full_qpos in source NPZ. "
        "Candidate was not saved."
    )


payload[
    "dynamic_v2_pass"
] = np.asarray(
    "pass2a_support_foot_se3",
)


payload[
    "pass2a_generalized_delta"
] = (
    frame_delta.astype(
        np.float32
    )
)


np.savez_compressed(
    OUTPUT_REFERENCE,
    **payload,
)


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

    frame_delta=
        frame_delta.astype(
            np.float32
        ),

    optimizer_cost=
        optimizer_cost.astype(
            np.float32
        ),

    optimizer_nfev=
        optimizer_nfev,

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
print("=" * 200)
print("NPZ PATCH")
print("=" * 200)

for label, keys in patched.items():

    print(
        f"{label:14s}:",
        keys,
    )


print()
print("=" * 200)
print("FINAL PASS-2A VERDICT")
print("=" * 200)


if (
    candidate[
        "anchor_validity"
    ] < 0.95
):

    print(
        "RESULT: CONTACT VALIDITY FAILED."
    )

    print(
        "Do not interpret candidate dynamics."
    )


elif (
    single_reduction >= 0.50
    and
    overall_reduction >= 0.25
    and
    candidate[
        "tau_p95"
    ]
    <
    baseline[
        "tau_p95"
    ]
):

    print(
        "RESULT: FULL STANCE-FOOT POSE REPAIR "
        "IS A MAJOR IMPROVEMENT"
    )

    print(
        "KEEP Pass-2A candidate."
    )

    print(
        "NEXT: target support-transition dynamics "
        "and then validate in physics."
    )


elif (
    single_reduction >= 0.25
    and
    candidate[
        "overall_p95"
    ]
    <=
    baseline[
        "overall_p95"
    ]
    * 1.10
):

    print(
        "RESULT: FULL STANCE-FOOT POSE REPAIR HELPS "
        "BUT IS NOT SUFFICIENT"
    )

    print(
        "KEEP it as an intermediate candidate."
    )

    print(
        "NEXT: Pass 2B direct dynamics-aware "
        "COM / momentum optimization."
    )


else:

    print(
        "RESULT: FULL STANCE-FOOT POSE REPAIR "
        "IS NOT SUFFICIENT"
    )

    print(
        "Contact kinematics are no longer the "
        "main explanation."
    )

    print(
        "NEXT: Pass 2B should directly optimize "
        "centroidal/dynamic consistency rather than "
        "foot locking."
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
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 200)


env.close()
