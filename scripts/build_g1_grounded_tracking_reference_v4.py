from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_v2.npz"
)

DST = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_v4_grounded.npz"
)

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


# ================================================================
# GROUNDING CONSTANTS
# ================================================================

TARGET_MIN_CLEARANCE = 0.003     # 3 mm hard floor margin

CONTACT_CLEARANCE = 0.015        # <= 15 mm -> expected contact

SUPPORT_SPEED = 0.30             # contact + slow foot -> stable support

SMOOTH_HALF_WINDOW = 4

MAX_Z_CHANGE_PER_FRAME = 0.004   # 4 mm / frame @ 50 Hz
                                 # = max 0.20 m/s grounding-envelope slope


# ================================================================
# LOAD REFERENCE V2
# ================================================================

if not SRC.exists():
    raise FileNotFoundError(SRC)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


d = np.load(
    SRC,
    allow_pickle=True,
)


joint_q = np.asarray(
    d["joint_pos_15"],
    dtype=np.float64,
)

root_pos = np.asarray(
    d["root_positions"],
    dtype=np.float64,
)

root_quat = np.asarray(
    d["root_quat_wxyz"],
    dtype=np.float64,
)

joint_names = [
    str(x)
    for x in d[
        "controlled_joint_names"
    ]
]

fps = float(
    np.asarray(
        d["fps"]
    ).reshape(-1)[0]
)


n = int(
    joint_q.shape[0]
)

dt = (
    1.0
    / fps
)


# ================================================================
# MUJOCO
# ================================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


if model.nkey > 0:

    stand_qpos = (
        model.key_qpos[0].copy()
    )

else:

    stand_qpos = np.zeros(
        model.nq,
        dtype=np.float64,
    )

    stand_qpos[2] = 0.80
    stand_qpos[3] = 1.0


floor_gid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor",
)

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


if min(
    floor_gid,
    left_site,
    right_site,
) < 0:

    raise RuntimeError(
        "Required floor/foot site was not found."
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


# ================================================================
# TRUE SOLE GEOMETRY
# ================================================================

def find_sole_geoms(body_id):

    result = []

    for gid in range(
        model.ngeom
    ):

        if int(
            model.geom_bodyid[
                gid
            ]
        ) != int(body_id):
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
            result.append(gid)


    return sorted(result)


left_sole = find_sole_geoms(
    left_body
)

right_sole = find_sole_geoms(
    right_body
)


if not left_sole or not right_sole:

    raise RuntimeError(
        "Could not find true sole geoms."
    )


print("=" * 125)
print("G1 GROUNDED TRACKING REFERENCE V4")
print("HARD NON-PENETRATION + CONTACT/SUPPORT SEPARATION")
print("=" * 125)

print(
    "left sole:",
    left_sole,
)

print(
    "right sole:",
    right_sole,
)

print(
    "frames:",
    n,
)

print(
    "fps:",
    fps,
)


# ================================================================
# JOINT ADDRESSES
# ================================================================

qaddrs = []
vaddrs = []


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


# ================================================================
# FK HELPERS
# ================================================================

def set_reference_pose(
    frame,
    positions,
):

    data.qpos[:] = (
        stand_qpos
    )

    data.qvel[:] = 0.0


    data.qpos[
        0:3
    ] = positions[
        frame
    ]


    data.qpos[
        3:7
    ] = root_quat[
        frame
    ]


    for j, qadr in enumerate(
        qaddrs
    ):

        data.qpos[
            qadr
        ] = joint_q[
            frame,
            j,
        ]


    mujoco.mj_normalizeQuat(
        model,
        data.qpos,
    )


    mujoco.mj_forward(
        model,
        data,
    )


def sole_clearance(
    geom_ids,
):

    result = []


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


        result.append(
            center_z
            - radius
            - floor_z
        )


    return float(
        min(result)
    )


# ================================================================
# RAW FK
# ================================================================

left_raw_clear = np.zeros(
    n,
    dtype=np.float64,
)

right_raw_clear = np.zeros(
    n,
    dtype=np.float64,
)

left_raw_pos = np.zeros(
    (n, 3),
    dtype=np.float64,
)

right_raw_pos = np.zeros(
    (n, 3),
    dtype=np.float64,
)


for frame in range(n):

    set_reference_pose(
        frame,
        root_pos,
    )


    left_raw_clear[
        frame
    ] = sole_clearance(
        left_sole
    )


    right_raw_clear[
        frame
    ] = sole_clearance(
        right_sole
    )


    left_raw_pos[
        frame
    ] = data.site_xpos[
        left_site
    ]


    right_raw_pos[
        frame
    ] = data.site_xpos[
        right_site
    ]


left_raw_vel = np.gradient(
    left_raw_pos,
    dt,
    axis=0,
)

right_raw_vel = np.gradient(
    right_raw_pos,
    dt,
    axis=0,
)


left_speed = np.linalg.norm(
    left_raw_vel[
        :,
        0:2,
    ],
    axis=1,
)

right_speed = np.linalg.norm(
    right_raw_vel[
        :,
        0:2,
    ],
    axis=1,
)


# ================================================================
# REQUIRED HARD GROUNDING
#
# This is the key V4 change.
#
# We do not ask which foot is "support" first.
#
# Whichever physical sole is lowest determines the minimum root
# height necessary to prevent ANY geometry from penetrating.
# ================================================================

lowest_raw_clear = np.minimum(
    left_raw_clear,
    right_raw_clear,
)


required_dz = (
    TARGET_MIN_CLEARANCE
    - lowest_raw_clear
)


# ================================================================
# SMOOTH DESIRED PROFILE
# ================================================================

half = (
    SMOOTH_HALF_WINDOW
)


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


# ================================================================
# HARD NON-PENETRATION PROJECTION
#
# Smoothing must NEVER lower the robot below the minimum
# collision-free root height.
# ================================================================

dz = np.maximum(
    smooth_dz,
    required_dz,
)


# ================================================================
# RATE-LIMIT ENVELOPE
#
# Important:
#
# We only RAISE neighboring frames.
#
# We never lower required collision-free frames.
#
# This spreads sudden vertical corrections smoothly while preserving
# the hard non-penetration guarantee.
# ================================================================

for _ in range(4):

    # Forward propagation.
    for i in range(
        1,
        n,
    ):

        minimum_allowed = (
            dz[i - 1]
            - MAX_Z_CHANGE_PER_FRAME
        )


        if dz[i] < minimum_allowed:

            dz[i] = (
                minimum_allowed
            )


    # Backward propagation.
    for i in range(
        n - 2,
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


    # Re-assert physical floor constraint.
    dz = np.maximum(
        dz,
        required_dz,
    )


# Safety bound only.
if np.max(
    dz
) > 0.12:

    raise RuntimeError(
        "Ground correction exceeded 120 mm. "
        "Reference likely contains a more serious problem."
    )


# ================================================================
# CORRECT ROOT Z
# ================================================================

root_corrected = (
    root_pos.copy()
)


root_corrected[
    :,
    2,
] += dz


# ================================================================
# FULL QPOS
# ================================================================

full_qpos = np.tile(
    stand_qpos[
        None,
        :
    ],
    (
        n,
        1,
    ),
)


full_qpos[
    :,
    0:3,
] = root_corrected


full_qpos[
    :,
    3:7,
] = root_quat


for j, qadr in enumerate(
    qaddrs
):

    full_qpos[
        :,
        qadr,
    ] = joint_q[
        :,
        j,
    ]


for frame in range(n):

    mujoco.mj_normalizeQuat(
        model,
        full_qpos[
            frame
        ],
    )


# ================================================================
# QUATERNION-AWARE VELOCITIES
# ================================================================

full_qvel = np.zeros(
    (
        n,
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
    n - 1,
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
        n,
        15,
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


# ================================================================
# FINAL FK
# ================================================================

left_clear = np.zeros(
    n,
    dtype=np.float64,
)

right_clear = np.zeros(
    n,
    dtype=np.float64,
)

left_pos = np.zeros(
    (n, 3),
    dtype=np.float64,
)

right_pos = np.zeros(
    (n, 3),
    dtype=np.float64,
)


for frame in range(n):

    set_reference_pose(
        frame,
        root_corrected,
    )


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


    left_pos[
        frame
    ] = data.site_xpos[
        left_site
    ]


    right_pos[
        frame
    ] = data.site_xpos[
        right_site
    ]


left_vel = np.gradient(
    left_pos,
    dt,
    axis=0,
).astype(
    np.float32
)


right_vel = np.gradient(
    right_pos,
    dt,
    axis=0,
).astype(
    np.float32
)


left_speed_final = np.linalg.norm(
    left_vel[
        :,
        0:2,
    ],
    axis=1,
)


right_speed_final = np.linalg.norm(
    right_vel[
        :,
        0:2,
    ],
    axis=1,
)


# ================================================================
# CONTACT VS SUPPORT
#
# CONTACT:
# geometry is close enough to floor to physically collide.
#
# SUPPORT:
# contact AND low horizontal speed.
#
# They are intentionally different.
# ================================================================

left_contact = (
    left_clear
    <= CONTACT_CLEARANCE
)

right_contact = (
    right_clear
    <= CONTACT_CLEARANCE
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
        left_speed_final
        <= SUPPORT_SPEED
    )
)

right_support = (
    right_contact
    & (
        right_speed_final
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


# ================================================================
# METRICS
# ================================================================

any_contact = (
    left_contact
    | right_contact
)

any_support = (
    left_support
    | right_support
)


both_contact = (
    left_contact
    & right_contact
)

both_support = (
    left_support
    & right_support
)


no_contact = (
    ~left_contact
    & ~right_contact
)


all_clearances = np.concatenate(
    [
        left_clear,
        right_clear,
    ]
)


penetration_1mm = int(
    np.sum(
        all_clearances
        < -0.001
    )
)


penetration_2mm = int(
    np.sum(
        all_clearances
        < -0.002
    )
)


penetration_5mm = int(
    np.sum(
        all_clearances
        < -0.005
    )
)


contact_clearances = []


for i in range(n):

    if left_contact[i]:

        contact_clearances.append(
            left_clear[i]
        )


    if right_contact[i]:

        contact_clearances.append(
            right_clear[i]
        )


contact_clearances = np.asarray(
    contact_clearances,
    dtype=np.float64,
)


support_clearances = []


for i in range(n):

    if left_support[i]:

        support_clearances.append(
            left_clear[i]
        )


    if right_support[i]:

        support_clearances.append(
            right_clear[i]
        )


support_clearances = np.asarray(
    support_clearances,
    dtype=np.float64,
)


def switches(mask):

    return (
        np.where(
            mask[1:]
            != mask[:-1]
        )[0]
        + 1
    ).tolist()


# ================================================================
# SAVE
# ================================================================

np.savez(
    DST,

    joint_pos_15=
        joint_q.astype(
            np.float32
        ),

    joint_vel_15=
        joint_vel,

    root_positions=
        root_corrected.astype(
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

    support_mask=
        support_mask,

    has_contact_mask=
        np.array(
            [True],
            dtype=np.bool_,
        ),

    left_foot_pos=
        left_pos.astype(
            np.float32
        ),

    right_foot_pos=
        right_pos.astype(
            np.float32
        ),

    left_foot_vel=
        left_vel,

    right_foot_vel=
        right_vel,

    left_sole_clearance=
        left_clear.astype(
            np.float32
        ),

    right_sole_clearance=
        right_clear.astype(
            np.float32
        ),

    root_z_correction=
        dz.astype(
            np.float32
        ),

    controlled_joint_names=
        np.asarray(
            joint_names
        ),

    fps=
        np.array(
            [fps],
            dtype=np.float32,
        ),

    coordinate_frame=
        np.array(
            [
                "+X forward, yaw approximately 0"
            ]
        ),

    grounding_method=
        np.array(
            [
                "hard lowest-sole nonpenetration envelope"
            ]
        ),

    source_dataset=
        np.array(
            [str(SRC)]
        ),
)


# ================================================================
# REPORT
# ================================================================

print()
print("=" * 125)
print("ROOT-Z CORRECTION")
print("=" * 125)

print(
    f"min    = "
    f"{np.min(dz)*1000:+.2f} mm"
)

print(
    f"median = "
    f"{np.median(dz)*1000:+.2f} mm"
)

print(
    f"max    = "
    f"{np.max(dz)*1000:+.2f} mm"
)


dz_diff = np.diff(
    dz
)


print(
    f"max frame-to-frame change = "
    f"{np.max(np.abs(dz_diff))*1000:.2f} mm"
)

print(
    f"max induced grounding slope = "
    f"{np.max(np.abs(dz_diff))*fps:.3f} m/s"
)


print()
print("=" * 125)
print("FINAL FOOT CLEARANCE")
print("=" * 125)

print(
    f"LEFT  min/median/max = "
    f"{np.min(left_clear)*1000:+.2f} / "
    f"{np.median(left_clear)*1000:+.2f} / "
    f"{np.max(left_clear)*1000:+.2f} mm"
)

print(
    f"RIGHT min/median/max = "
    f"{np.min(right_clear)*1000:+.2f} / "
    f"{np.median(right_clear)*1000:+.2f} / "
    f"{np.max(right_clear)*1000:+.2f} mm"
)


print()
print(
    "all-foot penetration samples:"
)

print(
    "  < -1 mm:",
    penetration_1mm,
)

print(
    "  < -2 mm:",
    penetration_2mm,
)

print(
    "  < -5 mm:",
    penetration_5mm,
)


print()
print("=" * 125)
print("CONTACT MASK")
print("=" * 125)

print(
    "left contact frames:",
    int(
        np.sum(
            left_contact
        )
    ),
)

print(
    "right contact frames:",
    int(
        np.sum(
            right_contact
        )
    ),
)

print(
    "double contact:",
    int(
        np.sum(
            both_contact
        )
    ),
)

print(
    "no-contact frames:",
    int(
        np.sum(
            no_contact
        )
    ),
)

print(
    "any-contact coverage:",
    f"{100.0*np.mean(any_contact):.1f}%",
)


print(
    "left contact switches:",
    switches(
        left_contact
    ),
)

print(
    "right contact switches:",
    switches(
        right_contact
    ),
)


print()
print("=" * 125)
print("STABLE SUPPORT MASK")
print("=" * 125)

print(
    "left support frames:",
    int(
        np.sum(
            left_support
        )
    ),
)

print(
    "right support frames:",
    int(
        np.sum(
            right_support
        )
    ),
)

print(
    "double support:",
    int(
        np.sum(
            both_support
        )
    ),
)

print(
    "any stable support coverage:",
    f"{100.0*np.mean(any_support):.1f}%",
)


print(
    "left support switches:",
    switches(
        left_support
    ),
)

print(
    "right support switches:",
    switches(
        right_support
    ),
)


if len(
    contact_clearances
) > 0:

    print()
    print(
        "contact clearance "
        "min/median/p95/max:"
    )

    print(
        f"  "
        f"{np.min(contact_clearances)*1000:+.2f} / "
        f"{np.median(contact_clearances)*1000:+.2f} / "
        f"{np.percentile(contact_clearances,95)*1000:+.2f} / "
        f"{np.max(contact_clearances)*1000:+.2f} mm"
    )


if len(
    support_clearances
) > 0:

    print()
    print(
        "stable-support clearance "
        "min/median/p95/max:"
    )

    print(
        f"  "
        f"{np.min(support_clearances)*1000:+.2f} / "
        f"{np.median(support_clearances)*1000:+.2f} / "
        f"{np.percentile(support_clearances,95)*1000:+.2f} / "
        f"{np.max(support_clearances)*1000:+.2f} mm"
    )


print()
print("=" * 125)
print("SELECTED FRAMES")
print("=" * 125)


for frame in [
    0,
    25,
    50,
    70,
    90,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    278,
    300,
    310,
]:

    if frame >= n:
        continue


    print(
        f"frame={frame:03d} "
        f"dz={dz[frame]*1000:+6.1f}mm "
        f"Lclr={left_clear[frame]*1000:+7.1f}mm "
        f"Rclr={right_clear[frame]*1000:+7.1f}mm "
        f"Lspd={left_speed_final[frame]:.3f} "
        f"Rspd={right_speed_final[frame]:.3f} "
        f"C="
        f"{int(left_contact[frame])}/"
        f"{int(right_contact[frame])} "
        f"S="
        f"{int(left_support[frame])}/"
        f"{int(right_support[frame])}"
    )


print()
print("=" * 125)
print("DECISION")
print("=" * 125)


contact_coverage = float(
    np.mean(
        any_contact
    )
)


if (
    penetration_1mm == 0
    and contact_coverage >= 0.85
    and np.max(
        np.abs(
            dz_diff
        )
    ) <= (
        MAX_Z_CHANGE_PER_FRAME
        + 1e-8
    )
):

    print(
        "GROUNDING STATUS: PASS"
    )

    print(
        "No meaningful sole penetration remains."
    )

    print(
        "Reference is ready for the first "
        "closed-loop PPO tracking pilot."
    )

else:

    print(
        "GROUNDING STATUS: REVIEW"
    )

    print(
        "Do not train yet."
    )


print()
print(
    "Saved:",
    DST
)

print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 125)
