from pathlib import Path
import math

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RAW = (
    ROOT
    / "datasets"
    / "raw"
    / "amass_g1"
    / "g1"
    / "ACCAD"
    / "Female1Walking_c3d"
    / "B1-standtowalk_poses_120_jpos.npz"
)

OLD15 = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_il_15dof.npz"
)

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

OUT = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_29dof_v1_grounded.npz"
)


TARGET_FPS = 50.0

TARGET_MIN_CLEARANCE = 0.003
MAX_Z_CHANGE_PER_FRAME = 0.004
SMOOTH_HALF_WINDOW = 4

CONTACT_CLEARANCE = 0.015
SUPPORT_SPEED = 0.30


# =====================================================================
# QUATERNION
# Everything is wxyz.
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
        np.dot(q0, q1)
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

        return qnormalize(
            q0
            + alpha
            * (
                q1 - q0
            )
        )

    theta0 = math.acos(dot)

    sin0 = math.sin(theta0)

    theta = theta0 * alpha

    s0 = (
        math.sin(
            theta0 - theta
        )
        / sin0
    )

    s1 = (
        math.sin(theta)
        / sin0
    )

    return qnormalize(
        s0*q0 + s1*q1
    )


def yaw_deg(q):

    w, x, y, z = qnormalize(q)

    yaw = math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0*(y*y + z*z),
    )

    return math.degrees(yaw)


# =====================================================================
# LOAD
# =====================================================================

for path in [
    RAW,
    OLD15,
    MODEL_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


raw = np.load(
    RAW,
    allow_pickle=True,
)

old15 = np.load(
    OLD15,
    allow_pickle=True,
)


raw_names = [
    str(x)
    for x in raw["dof_names"]
]

raw_q_source = np.asarray(
    raw["dof_positions"],
    dtype=np.float64,
)

# ================================================================
# REPRODUCE ORIGINAL LOWER-BODY PREPROCESSING
#
# Difference audit proved that the original 15-DOF dataset differs
# from the raw retarget only in the two hip-roll channels:
#
#   left_hip_roll  = raw + 0.070 rad
#   right_hip_roll = raw - 0.070 rad
#
# Everything else, including all 14 upper-body joints, is preserved.
# ================================================================

raw_q = raw_q_source.copy()

raw_q[:, 1] += 0.070
raw_q[:, 7] -= 0.070

raw_body_pos = np.asarray(
    raw["body_positions"],
    dtype=np.float64,
)

raw_body_quat = np.asarray(
    raw["body_rotations"],
    dtype=np.float64,
)

raw_body_names = [
    str(x)
    for x in raw["body_names"]
]


src_fps = float(
    np.asarray(
        raw["fps"]
    ).reshape(-1)[0]
)


# =====================================================================
# BASIC VALIDATION
# =====================================================================

print("=" * 125)
print("FULL 29-DOF G1 REFERENCE BUILDER")
print("=" * 125)

print(
    "raw frames:",
    raw_q.shape[0],
)

print(
    "raw DOFs:",
    raw_q.shape[1],
)

print(
    "raw bodies:",
    raw_body_pos.shape[1],
)

print(
    "raw FPS:",
    src_fps,
)


if raw_q.shape[1] != 29:

    raise RuntimeError(
        "Expected exactly 29 raw G1 DOFs."
    )


if len(raw_names) != 29:

    raise RuntimeError(
        "Expected exactly 29 raw joint names."
    )


if raw_body_pos.shape[1] != 30:

    raise RuntimeError(
        "Expected 30 raw G1 bodies."
    )


if raw_body_names[0] != "pelvis":

    raise RuntimeError(
        "Expected body 0 to be pelvis."
    )


# =====================================================================
# VALIDATE ORIGINAL 15-DOF EXTRACTION
# =====================================================================

old_names = [
    str(x)
    for x in old15[
        "controlled_joint_names"
    ]
]

old_q = np.asarray(
    old15[
        "joint_pos_15"
    ],
    dtype=np.float64,
)


print()
print("=" * 125)
print("15-DOF EXTRACTION VALIDATION")
print("=" * 125)


name_match = (
    raw_names[:15]
    == old_names
)


print(
    "first 15 names match:",
    name_match,
)


if not name_match:

    raise RuntimeError(
        "Raw first-15 joint ordering does not "
        "match old 15-DOF dataset."
    )


q15_diff = np.max(
    np.abs(
        raw_q[:, :15]
        - old_q
    )
)


print(
    "max processed29-vs-old15 q difference:",
    f"{q15_diff:.9f} rad",
)


if q15_diff > 1e-6:

    raise RuntimeError(
        "Reconstructed 29-DOF preprocessing does not "
        "match the old 15-DOF dataset."
    )


print(
    "left hip-roll preprocessing:",
    "+0.070000 rad",
)

print(
    "right hip-roll preprocessing:",
    "-0.070000 rad",
)

print(
    "upper-body preprocessing:",
    "UNCHANGED RAW RETARGET",
)


# =====================================================================
# VALIDATE PELVIS SOURCE
# =====================================================================

old_root_pos = np.asarray(
    old15[
        "root_positions"
    ],
    dtype=np.float64,
)

old_root_quat = np.asarray(
    old15[
        "root_rotations"
    ],
    dtype=np.float64,
)


raw_root_pos = (
    raw_body_pos[:, 0, :]
)

raw_root_quat = (
    raw_body_quat[:, 0, :]
)


root_pos_diff = np.max(
    np.abs(
        raw_root_pos
        - old_root_pos
    )
)


# quaternion sign q and -q are equivalent
quat_direct = np.linalg.norm(
    raw_root_quat
    - old_root_quat,
    axis=1,
)

quat_flipped = np.linalg.norm(
    raw_root_quat
    + old_root_quat,
    axis=1,
)

quat_diff = float(
    np.max(
        np.minimum(
            quat_direct,
            quat_flipped,
        )
    )
)


print()
print("=" * 125)
print("PELVIS SOURCE VALIDATION")
print("=" * 125)

print(
    "root position max diff:",
    f"{root_pos_diff:.9f}",
)

print(
    "root quaternion equivalent diff:",
    f"{quat_diff:.9f}",
)


if root_pos_diff > 1e-5:

    raise RuntimeError(
        "Processed root positions do not match "
        "raw pelvis body positions."
    )


if quat_diff > 1e-5:

    raise RuntimeError(
        "Processed root quaternions do not match "
        "raw pelvis rotations."
    )


# =====================================================================
# MODEL + EXACT JOINT ORDER VALIDATION
# =====================================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


model_names = []

qaddrs = []
vaddrs = []
aids = []


for aid in range(
    model.nu
):

    jid = int(
        model.actuator_trnid[
            aid,
            0,
        ]
    )

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        jid,
    )

    model_names.append(
        str(name)
    )

    qaddrs.append(
        int(
            model.jnt_qposadr[
                jid
            ]
        )
    )

    vaddrs.append(
        int(
            model.jnt_dofadr[
                jid
            ]
        )
    )

    aids.append(aid)


print()
print("=" * 125)
print("29-DOF MODEL ORDER VALIDATION")
print("=" * 125)

print(
    "raw names == MuJoCo actuator order:",
    raw_names == model_names,
)


if raw_names != model_names:

    for i, (
        raw_name,
        model_name,
    ) in enumerate(
        zip(
            raw_names,
            model_names,
        )
    ):

        if raw_name != model_name:

            print(
                f"Mismatch {i}: "
                f"{raw_name} != {model_name}"
            )

    raise RuntimeError(
        "Raw 29-DOF order does not match "
        "MuJoCo actuator order."
    )


# =====================================================================
# RESAMPLE 30 -> 50 HZ
# =====================================================================

n_src = raw_q.shape[0]

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


joint_q = np.empty(
    (
        n_dst,
        29,
    ),
    dtype=np.float64,
)


for j in range(29):

    joint_q[:, j] = np.interp(
        dst_t,
        src_t,
        raw_q[:, j],
    )


root_raw = np.empty(
    (
        n_dst,
        3,
    ),
    dtype=np.float64,
)


for axis in range(3):

    root_raw[:, axis] = np.interp(
        dst_t,
        src_t,
        raw_root_pos[:, axis],
    )


root_quat_source = np.empty(
    (
        n_dst,
        4,
    ),
    dtype=np.float64,
)


for k, t in enumerate(dst_t):

    u = t * src_fps

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
        raw_root_quat[i0],
        raw_root_quat[i1],
        alpha,
    )


# =====================================================================
# SOURCE +Y -> MUJOCO +X
#
# Global -90-degree rotation around Z.
#
# x_new = +y_source
# y_new = -x_source
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


for k in range(n_dst):

    root_quat[k] = qnormalize(
        qmul(
            q_align,
            root_quat_source[k],
        )
    )


    if (
        k > 0
        and np.dot(
            root_quat[k - 1],
            root_quat[k],
        ) < 0.0
    ):

        root_quat[k] *= -1.0


# =====================================================================
# STANDING KEYFRAME
# =====================================================================

if model.nkey > 0:

    template_qpos = (
        model.key_qpos[
            0
        ].copy()
    )

else:

    template_qpos = np.zeros(
        model.nq,
        dtype=np.float64,
    )

    template_qpos[2] = 0.80
    template_qpos[3] = 1.0


# =====================================================================
# TRUE SOLE GEOMS
# =====================================================================

left_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_foot",
)

right_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_foot",
)

floor_gid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor",
)


if min(
    left_site,
    right_site,
    floor_gid,
) < 0:

    raise RuntimeError(
        "Could not locate foot/floor geometry."
    )


left_body = int(
    model.site_bodyid[
        left_site
    ]
)

right_body = int(
    model.site_bodyid[
        right_site
    ]
)


floor_z = float(
    model.geom_pos[
        floor_gid,
        2,
    ]
)


def find_sole_geoms(body_id):

    result = []

    for gid in range(
        model.ngeom
    ):

        if int(
            model.geom_bodyid[
                gid
            ]
        ) != body_id:

            continue


        if int(
            model.geom_type[
                gid
            ]
        ) != int(
            mujoco.mjtGeom.mjGEOM_SPHERE
        ):

            continue


        radius = float(
            model.geom_size[
                gid,
                0,
            ]
        )


        if radius <= 0.010:

            result.append(
                gid
            )


    return sorted(result)


left_sole = find_sole_geoms(
    left_body
)

right_sole = find_sole_geoms(
    right_body
)


print()
print(
    "left sole geoms:",
    left_sole,
)

print(
    "right sole geoms:",
    right_sole,
)


if not left_sole or not right_sole:

    raise RuntimeError(
        "True sole spheres not found."
    )


# =====================================================================
# QPOS HELPER
# =====================================================================

def make_qpos(
    frame,
    positions,
):

    qpos = (
        template_qpos.copy()
    )


    qpos[0:3] = positions[
        frame
    ]

    qpos[3:7] = root_quat[
        frame
    ]


    for j, qadr in enumerate(
        qaddrs
    ):

        qpos[
            qadr
        ] = joint_q[
            frame,
            j,
        ]


    mujoco.mj_normalizeQuat(
        model,
        qpos,
    )


    return qpos


def sole_clearance(
    geom_ids,
):

    values = []


    for gid in geom_ids:

        center_z = float(
            data.geom_xpos[
                gid,
                2,
            ]
        )

        radius = float(
            model.geom_size[
                gid,
                0,
            ]
        )

        values.append(
            center_z
            - radius
            - floor_z
        )


    return float(
        min(values)
    )


# =====================================================================
# RAW GROUNDING PASS
# =====================================================================

left_clear_raw = np.zeros(
    n_dst,
    dtype=np.float64,
)

right_clear_raw = np.zeros(
    n_dst,
    dtype=np.float64,
)


for frame in range(n_dst):

    data.qpos[:] = make_qpos(
        frame,
        root_pos,
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


    left_clear_raw[
        frame
    ] = sole_clearance(
        left_sole
    )

    right_clear_raw[
        frame
    ] = sole_clearance(
        right_sole
    )


lowest_raw = np.minimum(
    left_clear_raw,
    right_clear_raw,
)


required_dz = (
    TARGET_MIN_CLEARANCE
    - lowest_raw
)


# =====================================================================
# SMOOTH + HARD FLOOR ENVELOPE
# =====================================================================

half = SMOOTH_HALF_WINDOW


kernel = np.concatenate(
    [
        np.arange(
            1,
            half + 2,
            dtype=np.float64,
        ),

        np.arange(
            half,
            0,
            -1,
            dtype=np.float64,
        ),
    ]
)


kernel /= np.sum(
    kernel
)


padded = np.pad(
    required_dz,
    (
        half,
        half,
    ),
    mode="edge",
)


smooth_dz = np.convolve(
    padded,
    kernel,
    mode="valid",
)


dz = np.maximum(
    smooth_dz,
    required_dz,
)


for _ in range(4):

    for i in range(
        1,
        n_dst,
    ):

        minimum_allowed = (
            dz[i - 1]
            - MAX_Z_CHANGE_PER_FRAME
        )

        if dz[i] < minimum_allowed:

            dz[i] = (
                minimum_allowed
            )


    for i in range(
        n_dst - 2,
        -1,
        -1,
    ):

        minimum_allowed = (
            dz[i + 1]
            - MAX_Z_CHANGE_PER_FRAME
        )

        if dz[i] < minimum_allowed:

            dz[i] = (
                minimum_allowed
            )


    dz = np.maximum(
        dz,
        required_dz,
    )


if np.max(dz) > 0.12:

    raise RuntimeError(
        "Ground correction exceeded 120 mm."
    )


root_grounded = (
    root_pos.copy()
)


root_grounded[:, 2] += (
    dz
)


# =====================================================================
# FULL 36-D QPOS
# =====================================================================

full_qpos = np.zeros(
    (
        n_dst,
        model.nq,
    ),
    dtype=np.float64,
)


for frame in range(n_dst):

    full_qpos[
        frame
    ] = make_qpos(
        frame,
        root_grounded,
    )


# =====================================================================
# RECOMPUTE ALL VELOCITIES WITH MUJOCO
#
# Ignore raw dof_velocities completely.
# =====================================================================

dt = (
    1.0
    / TARGET_FPS
)


full_qvel = np.zeros(
    (
        n_dst,
        model.nv,
    ),
    dtype=np.float64,
)


mujoco.mj_differentiatePos(
    model,
    full_qvel[0],
    dt,
    full_qpos[0],
    full_qpos[1],
)


for frame in range(
    1,
    n_dst - 1,
):

    mujoco.mj_differentiatePos(
        model,
        full_qvel[
            frame
        ],

        2.0 * dt,

        full_qpos[
            frame - 1
        ],

        full_qpos[
            frame + 1
        ],
    )


mujoco.mj_differentiatePos(
    model,
    full_qvel[-1],
    dt,
    full_qpos[-2],
    full_qpos[-1],
)


joint_vel = np.empty(
    (
        n_dst,
        29,
    ),
    dtype=np.float32,
)


for j, vadr in enumerate(
    vaddrs
):

    joint_vel[
        :,
        j,
    ] = full_qvel[
        :,
        vadr,
    ]


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


# =====================================================================
# FINAL FK + BODY TARGETS
# =====================================================================

body_ids = []


for name in raw_body_names:

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:

        raise RuntimeError(
            f"Raw body missing in MuJoCo: {name}"
        )

    body_ids.append(
        bid
    )


body_pos = np.zeros(
    (
        n_dst,
        30,
        3,
    ),
    dtype=np.float32,
)

body_quat = np.zeros(
    (
        n_dst,
        30,
        4,
    ),
    dtype=np.float32,
)


left_clear = np.zeros(
    n_dst,
    dtype=np.float64,
)

right_clear = np.zeros(
    n_dst,
    dtype=np.float64,
)

left_foot_pos = np.zeros(
    (
        n_dst,
        3,
    ),
    dtype=np.float32,
)

right_foot_pos = np.zeros(
    (
        n_dst,
        3,
    ),
    dtype=np.float32,
)


for frame in range(n_dst):

    data.qpos[:] = full_qpos[
        frame
    ]

    data.qvel[:] = full_qvel[
        frame
    ]


    mujoco.mj_forward(
        model,
        data,
    )


    for bi, bid in enumerate(
        body_ids
    ):

        body_pos[
            frame,
            bi,
        ] = data.xpos[
            bid
        ]


        body_quat[
            frame,
            bi,
        ] = data.xquat[
            bid
        ]


    left_clear[
        frame
    ] = sole_clearance(
        left_sole
    )


    right_clear[
        frame
    ] = sole_clearance(
        right_sole
    )


    left_foot_pos[
        frame
    ] = data.site_xpos[
        left_site
    ]


    right_foot_pos[
        frame
    ] = data.site_xpos[
        right_site
    ]


left_foot_vel = np.gradient(
    left_foot_pos,
    dt,
    axis=0,
).astype(
    np.float32
)


right_foot_vel = np.gradient(
    right_foot_pos,
    dt,
    axis=0,
).astype(
    np.float32
)


left_speed = np.linalg.norm(
    left_foot_vel[:, 0:2],
    axis=1,
)

right_speed = np.linalg.norm(
    right_foot_vel[:, 0:2],
    axis=1,
)


left_contact = (
    left_clear <= CONTACT_CLEARANCE
)

right_contact = (
    right_clear <= CONTACT_CLEARANCE
)


contact_mask = np.stack(
    [
        left_contact,
        right_contact,
    ],
    axis=1,
).astype(
    np.float32
)


left_support = (
    left_contact
    & (
        left_speed
        <= SUPPORT_SPEED
    )
)

right_support = (
    right_contact
    & (
        right_speed
        <= SUPPORT_SPEED
    )
)


support_mask = np.stack(
    [
        left_support,
        right_support,
    ],
    axis=1,
).astype(
    np.float32
)


# =====================================================================
# SAVE
# =====================================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.savez(
    OUT,

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

    dof_names=
        np.asarray(
            raw_names
        ),

    body_names=
        np.asarray(
            raw_body_names
        ),

    joint_pos_29=
        joint_q.astype(
            np.float32
        ),

    joint_vel_29=
        joint_vel,

    full_qpos=
        full_qpos.astype(
            np.float32
        ),

    full_qvel=
        full_qvel.astype(
            np.float32
        ),

    root_positions=
        root_grounded.astype(
            np.float32
        ),

    root_quat_wxyz=
        root_quat.astype(
            np.float32
        ),

    root_velocity=
        root_velocity,

    root_ang_vel_local=
        root_ang_vel_local,

    body_positions=
        body_pos,

    body_rotations_wxyz=
        body_quat,

    left_foot_pos=
        left_foot_pos,

    right_foot_pos=
        right_foot_pos,

    left_foot_vel=
        left_foot_vel,

    right_foot_vel=
        right_foot_vel,

    left_sole_clearance=
        left_clear.astype(
            np.float32
        ),

    right_sole_clearance=
        right_clear.astype(
            np.float32
        ),

    contact_mask=
        contact_mask,

    support_mask=
        support_mask,

    has_contact_mask=
        np.array(
            [True],
            dtype=np.bool_,
        ),

    root_z_correction=
        dz.astype(
            np.float32
        ),

    coordinate_frame=
        np.array(
            [
                "source +Y -> MuJoCo +X; "
                "global -90deg Z rotation"
            ]
        ),

    source_dataset=
        np.array(
            [str(RAW)]
        ),
)


# =====================================================================
# REPORT
# =====================================================================

all_clearance = np.concatenate(
    [
        left_clear,
        right_clear,
    ]
)


print()
print("=" * 125)
print("29-DOF GROUNDED REFERENCE RESULT")
print("=" * 125)

print(
    "output:",
    OUT,
)

print(
    "frames:",
    n_dst,
)

print(
    "FPS:",
    TARGET_FPS,
)

print(
    "duration:",
    f"{duration:.3f}s",
)


print()
print(
    "final X:",
    f"{root_grounded[-1,0]:+.4f} m",
)

print(
    "final Y:",
    f"{root_grounded[-1,1]:+.4f} m",
)


yaw_values = np.array(
    [
        yaw_deg(q)
        for q in root_quat
    ]
)


print(
    "yaw mean/min/max:",
    f"{np.mean(yaw_values):+.2f} / "
    f"{np.min(yaw_values):+.2f} / "
    f"{np.max(yaw_values):+.2f} deg",
)


print()
print(
    "root Z correction "
    "min/median/max:",
    f"{np.min(dz)*1000:+.2f} / "
    f"{np.median(dz)*1000:+.2f} / "
    f"{np.max(dz)*1000:+.2f} mm",
)


print(
    "max dz/frame:",
    f"{np.max(np.abs(np.diff(dz)))*1000:.2f} mm",
)


print()
print(
    "LEFT clearance min/median/max:",
    f"{np.min(left_clear)*1000:+.2f} / "
    f"{np.median(left_clear)*1000:+.2f} / "
    f"{np.max(left_clear)*1000:+.2f} mm",
)

print(
    "RIGHT clearance min/median/max:",
    f"{np.min(right_clear)*1000:+.2f} / "
    f"{np.median(right_clear)*1000:+.2f} / "
    f"{np.max(right_clear)*1000:+.2f} mm",
)


print(
    "penetration < -1 mm:",
    int(
        np.sum(
            all_clearance
            < -0.001
        )
    ),
)


print()
print(
    "any contact coverage:",
    f"{100*np.mean(left_contact | right_contact):.1f}%",
)

print(
    "stable support coverage:",
    f"{100*np.mean(left_support | right_support):.1f}%",
)

print(
    "double contact frames:",
    int(
        np.sum(
            left_contact
            & right_contact
        )
    ),
)


print()
print("=" * 125)
print("UPPER-BODY MOTION")
print("=" * 125)


for j in range(
    15,
    29,
):

    span = (
        np.max(
            joint_q[:, j]
        )
        - np.min(
            joint_q[:, j]
        )
    )

    std = np.std(
        joint_q[:, j]
    )

    print(
        f"{j:02d} "
        f"{raw_names[j]:28s} "
        f"std={std:.4f}rad "
        f"span={span:.4f}rad "
        f"({math.degrees(span):.2f}deg)"
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 125)
