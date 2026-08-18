from pathlib import Path
import math

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_il_15dof.npz"
)

DST = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_v2.npz"
)

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

TARGET_FPS = 50.0


# =====================================================================
# QUATERNION HELPERS
# Quaternion convention everywhere here:
#
#     [w, x, y, z]
# =====================================================================

def qnormalize(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    n = np.linalg.norm(q)

    if n < 1e-12:

        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    return q / n


def qmul(a, b):

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.array(
        [
            aw*bw - ax*bx - ay*by - az*bz,

            aw*bx + ax*bw + ay*bz - az*by,

            aw*by - ax*bz + ay*bw + az*bx,

            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )


def qslerp(q0, q1, alpha):

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)

    dot = float(
        np.dot(
            q0,
            q1,
        )
    )

    # q and -q represent the same orientation.
    # Keep the interpolation on the short arc.
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

        return qnormalize(
            q0
            + alpha
            * (
                q1 - q0
            )
        )


    theta0 = math.acos(
        dot
    )

    sin_theta0 = math.sin(
        theta0
    )

    theta = (
        theta0
        * alpha
    )

    s0 = (
        math.sin(
            theta0 - theta
        )
        / sin_theta0
    )

    s1 = (
        math.sin(
            theta
        )
        / sin_theta0
    )

    return qnormalize(
        s0 * q0
        + s1 * q1
    )


def quat_yaw(q):

    w, x, y, z = qnormalize(q)

    return math.atan2(
        2.0
        * (
            w*z
            + x*y
        ),

        1.0
        - 2.0
        * (
            y*y
            + z*z
        ),
    )


# =====================================================================
# LOAD
# =====================================================================

if not SRC.exists():

    raise FileNotFoundError(
        SRC
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        MODEL_PATH
    )


d = np.load(
    SRC,
    allow_pickle=True,
)


required = [
    "joint_pos_15",
    "root_positions",
    "root_rotations",
    "controlled_joint_names",
    "fps",
]


for key in required:

    if key not in d:

        raise RuntimeError(
            f"Missing required key: {key}"
        )


src_joint_q = np.asarray(
    d["joint_pos_15"],
    dtype=np.float64,
)

src_root_pos = np.asarray(
    d["root_positions"],
    dtype=np.float64,
)

src_root_quat = np.asarray(
    d["root_rotations"],
    dtype=np.float64,
)


src_fps = float(
    np.asarray(
        d["fps"]
    ).reshape(-1)[0]
)


joint_names = [
    str(x)
    for x in d[
        "controlled_joint_names"
    ]
]


n_src = int(
    src_joint_q.shape[0]
)

duration = (
    (n_src - 1)
    / src_fps
)


n_dst = (
    int(
        round(
            duration
            * TARGET_FPS
        )
    )
    + 1
)


src_t = np.linspace(
    0.0,
    duration,
    n_src,
)

dst_t = np.linspace(
    0.0,
    duration,
    n_dst,
)


print("=" * 110)
print("G1 TRACKING REFERENCE V2")
print("FULL ROOT-STATE ALIGNMENT")
print("=" * 110)

print(
    f"source frames = {n_src}"
)

print(
    f"source FPS    = {src_fps}"
)

print(
    f"target frames = {n_dst}"
)

print(
    f"target FPS    = {TARGET_FPS}"
)

print(
    f"duration      = {duration:.3f}s"
)


# =====================================================================
# INTERPOLATE JOINT POSITIONS
# =====================================================================

joint_q = np.empty(
    (
        n_dst,
        15,
    ),
    dtype=np.float64,
)


for j in range(15):

    joint_q[:, j] = np.interp(
        dst_t,
        src_t,
        src_joint_q[:, j],
    )


# =====================================================================
# INTERPOLATE ROOT POSITION
# =====================================================================

root_raw = np.empty(
    (
        n_dst,
        3,
    ),
    dtype=np.float64,
)


for j in range(3):

    root_raw[:, j] = np.interp(
        dst_t,
        src_t,
        src_root_pos[:, j],
    )


# =====================================================================
# RESAMPLE SOURCE ROOT QUATERNION WITH SLERP
# =====================================================================

root_quat_source = np.empty(
    (
        n_dst,
        4,
    ),
    dtype=np.float64,
)


for k, t in enumerate(dst_t):

    u = (
        t
        * src_fps
    )

    i0 = int(
        np.floor(u)
    )

    i0 = int(
        np.clip(
            i0,
            0,
            n_src - 1,
        )
    )


    i1 = min(
        i0 + 1,
        n_src - 1,
    )


    alpha = float(
        u - i0
    )


    root_quat_source[k] = qslerp(
        src_root_quat[i0],
        src_root_quat[i1],
        alpha,
    )


# =====================================================================
# WORLD-FRAME ALIGNMENT
#
# Source:
#     forward approximately +Y
#     root yaw approximately +90 degrees
#
# Tracking canonical:
#     forward +X
#     root yaw approximately 0 degrees
#
# Apply global -90 degree Z rotation:
#
#     x_new = +y_source
#     y_new = -x_source
#
# Quaternion:
#
#     q_new = q_align * q_source
# =====================================================================

root_pos = np.empty_like(
    root_raw
)


root_pos[:, 0] = (
    root_raw[:, 1]
    - root_raw[0, 1]
)

root_pos[:, 1] = -(
    root_raw[:, 0]
    - root_raw[0, 0]
)

root_pos[:, 2] = (
    root_raw[:, 2]
)


half = (
    -0.5
    * math.pi
    / 2.0
)


q_align = np.array(
    [
        math.cos(
            math.pi / 4.0
        ),

        0.0,
        0.0,

        -math.sin(
            math.pi / 4.0
        ),
    ],
    dtype=np.float64,
)


root_quat = np.empty_like(
    root_quat_source
)


for k in range(
    n_dst
):

    root_quat[k] = qnormalize(
        qmul(
            q_align,
            root_quat_source[k],
        )
    )


    # Quaternion sign continuity.
    if (
        k > 0
        and np.dot(
            root_quat[k - 1],
            root_quat[k],
        ) < 0.0
    ):

        root_quat[k] *= -1.0


# =====================================================================
# BUILD FULL MUJOCO QPOS REFERENCE
#
# Then use mj_differentiatePos.
#
# This gives correct:
#
# - global root linear velocity
# - body-local root angular velocity
# - hinge velocities
# =====================================================================

model = mujoco.MjModel.from_xml_path(
    str(
        MODEL_PATH
    )
)


if model.nkey > 0:

    stand_qpos = (
        model.key_qpos[
            0
        ].copy()
    )

else:

    stand_qpos = np.zeros(
        model.nq,
        dtype=np.float64,
    )

    stand_qpos[2] = 0.80
    stand_qpos[3] = 1.0


qpos_addresses = []
qvel_addresses = []


for name in joint_names:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )


    if jid < 0:

        raise RuntimeError(
            f"Joint not found: {name}"
        )


    qpos_addresses.append(
        int(
            model.jnt_qposadr[
                jid
            ]
        )
    )


    qvel_addresses.append(
        int(
            model.jnt_dofadr[
                jid
            ]
        )
    )


full_qpos = np.tile(
    stand_qpos[
        None,
        :
    ],
    (
        n_dst,
        1,
    ),
)


full_qpos[:, 0:3] = (
    root_pos
)

full_qpos[:, 3:7] = (
    root_quat
)


for j, qadr in enumerate(
    qpos_addresses
):

    full_qpos[
        :,
        qadr,
    ] = joint_q[
        :,
        j,
    ]


for k in range(
    n_dst
):

    mujoco.mj_normalizeQuat(
        model,
        full_qpos[k],
    )


full_qvel = np.zeros(
    (
        n_dst,
        model.nv,
    ),
    dtype=np.float64,
)


dt = (
    1.0
    / TARGET_FPS
)


# First sample:
mujoco.mj_differentiatePos(
    model,
    full_qvel[0],
    dt,
    full_qpos[0],
    full_qpos[1],
)


# Central difference internally.
for k in range(
    1,
    n_dst - 1,
):

    mujoco.mj_differentiatePos(
        model,
        full_qvel[k],
        2.0 * dt,
        full_qpos[k - 1],
        full_qpos[k + 1],
    )


# Final sample:
mujoco.mj_differentiatePos(
    model,
    full_qvel[-1],
    dt,
    full_qpos[-2],
    full_qpos[-1],
)


root_velocity = (
    full_qvel[
        :,
        0:3,
    ].astype(
        np.float32
    )
)


root_ang_vel_local = (
    full_qvel[
        :,
        3:6,
    ].astype(
        np.float32
    )
)


joint_vel = np.empty(
    (
        n_dst,
        15,
    ),
    dtype=np.float32,
)


for j, vadr in enumerate(
    qvel_addresses
):

    joint_vel[:, j] = (
        full_qvel[
            :,
            vadr,
        ]
    )


# =====================================================================
# CONTACTS
#
# Current source has no trustworthy contact labels.
# Leave them explicitly disabled.
# =====================================================================

contact_mask = np.zeros(
    (
        n_dst,
        2,
    ),
    dtype=np.float32,
)


# =====================================================================
# SAVE
# =====================================================================

DST.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.savez(
    DST,

    joint_pos_15=
        joint_q.astype(
            np.float32
        ),

    joint_vel_15=
        joint_vel.astype(
            np.float32
        ),

    root_positions=
        root_pos.astype(
            np.float32
        ),

    root_velocity=
        root_velocity,

    root_quat_wxyz=
        root_quat.astype(
            np.float32
        ),

    root_ang_vel_local=
        root_ang_vel_local,

    contact_mask=
        contact_mask,

    has_contact_mask=
        np.array(
            [False],
            dtype=np.bool_,
        ),

    controlled_joint_names=
        np.asarray(
            joint_names
        ),

    fps=
        np.array(
            [TARGET_FPS],
            dtype=np.float32,
        ),

    source_fps=
        np.array(
            [src_fps],
            dtype=np.float32,
        ),

    source_frames=
        np.array(
            [n_src],
            dtype=np.int32,
        ),

    coordinate_frame=
        np.array(
            [
                "source +Y/+90yaw -> "
                "MuJoCo +X/0yaw"
            ]
        ),

    source_dataset=
        np.array(
            [str(SRC)]
        ),
)


# =====================================================================
# VALIDATION OUTPUT
# =====================================================================

yaw_deg = np.array(
    [
        math.degrees(
            quat_yaw(q)
        )
        for q in root_quat
    ]
)


print()
print("=" * 110)
print("ALIGNED ROOT VALIDATION")
print("=" * 110)

print(
    f"final X = "
    f"{root_pos[-1,0]:+.4f} m"
)

print(
    f"final Y = "
    f"{root_pos[-1,1]:+.4f} m"
)

print(
    f"mean VX = "
    f"{np.mean(root_velocity[:,0]):+.4f} m/s"
)

print(
    f"VX min/max = "
    f"{np.min(root_velocity[:,0]):+.4f} / "
    f"{np.max(root_velocity[:,0]):+.4f}"
)

print(
    f"yaw mean = "
    f"{np.mean(yaw_deg):+.2f} deg"
)

print(
    f"yaw min/max = "
    f"{np.min(yaw_deg):+.2f} / "
    f"{np.max(yaw_deg):+.2f} deg"
)

print(
    "root angular velocity "
    "local max abs =",
    float(
        np.max(
            np.abs(
                root_ang_vel_local
            )
        )
    ),
)

print(
    "joint velocity max abs =",
    float(
        np.max(
            np.abs(
                joint_vel
            )
        )
    ),
)


print()
print("Selected frames:")


for k in [
    0,
    50,
    100,
    150,
    200,
    250,
    310,
]:

    if k >= n_dst:
        continue

    print(
        f"frame={k:03d} "
        f"x={root_pos[k,0]:+.3f} "
        f"y={root_pos[k,1]:+.3f} "
        f"z={root_pos[k,2]:.3f} "
        f"vx={root_velocity[k,0]:+.3f} "
        f"yaw={yaw_deg[k]:+.1f}"
    )


print()
print(
    "Saved:",
    DST
)

print("=" * 110)
