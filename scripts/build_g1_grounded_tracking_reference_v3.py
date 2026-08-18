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
    / "g1_amass_walking_tracking_50hz_v3_grounded.npz"
)

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


TARGET_CLEARANCE = 0.004     # 4 mm
DOUBLE_SUPPORT_SPEED = 0.18
SINGLE_SUPPORT_SPEED = 0.32
FALLBACK_SUPPORT_SPEED = 0.22

MAX_DZ_CHANGE_PER_FRAME = 0.003
SMOOTH_HALF_WINDOW = 5


# =====================================================================
# LOAD
# =====================================================================

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


n = joint_q.shape[0]

dt = 1.0 / fps


# =====================================================================
# MODEL
# =====================================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


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


left_body = int(
    model.site_bodyid[left_site]
)

right_body = int(
    model.site_bodyid[right_site]
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
            model.geom_bodyid[gid]
        ) != body_id:
            continue

        if int(
            model.geom_type[gid]
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
            f"Joint missing: {name}"
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


def set_pose(frame, positions):

    data.qpos[:] = stand_qpos
    data.qvel[:] = 0.0

    data.qpos[0:3] = positions[
        frame
    ]

    data.qpos[3:7] = root_quat[
        frame
    ]

    for j, qadr in enumerate(
        qaddrs
    ):

        data.qpos[qadr] = joint_q[
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


def clearance(geoms):

    values = []

    for gid in geoms:

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
# FIRST FK PASS
# =====================================================================

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

    set_pose(
        frame,
        root_pos,
    )

    left_clear[frame] = (
        clearance(
            left_sole
        )
    )

    right_clear[frame] = (
        clearance(
            right_sole
        )
    )

    left_pos[frame] = (
        data.site_xpos[
            left_site
        ]
    )

    right_pos[frame] = (
        data.site_xpos[
            right_site
        ]
    )


left_vel = np.gradient(
    left_pos,
    dt,
    axis=0,
)

right_vel = np.gradient(
    right_pos,
    dt,
    axis=0,
)


left_speed = np.linalg.norm(
    left_vel[:, 0:2],
    axis=1,
)

right_speed = np.linalg.norm(
    right_vel[:, 0:2],
    axis=1,
)


# =====================================================================
# PRELIMINARY SUPPORT CLASSIFICATION
#
# This intentionally relies mainly on WORLD FOOT SPEED.
#
# A stance foot should move slowly relative to the ground.
# =====================================================================

support = np.zeros(
    (n, 2),
    dtype=bool,
)


for i in range(n):

    ls = float(
        left_speed[i]
    )

    rs = float(
        right_speed[i]
    )


    # -------------------------------------------------------------
    # DOUBLE SUPPORT
    # -------------------------------------------------------------

    if (
        ls < DOUBLE_SUPPORT_SPEED
        and rs < DOUBLE_SUPPORT_SPEED
    ):

        height_difference = abs(
            left_clear[i]
            - right_clear[i]
        )


        if height_difference <= 0.030:

            support[i] = [
                True,
                True,
            ]

        else:

            # If the feet differ by >3 cm,
            # treat only the lower one as support.
            if (
                left_clear[i]
                < right_clear[i]
            ):

                support[i, 0] = True

            else:

                support[i, 1] = True

        continue


    # -------------------------------------------------------------
    # LEFT SUPPORT
    # -------------------------------------------------------------

    if (
        ls < SINGLE_SUPPORT_SPEED
        and (
            rs > 0.50
            or ls < 0.60 * rs
        )
    ):

        support[i, 0] = True
        continue


    # -------------------------------------------------------------
    # RIGHT SUPPORT
    # -------------------------------------------------------------

    if (
        rs < SINGLE_SUPPORT_SPEED
        and (
            ls > 0.50
            or rs < 0.60 * ls
        )
    ):

        support[i, 1] = True
        continue


    # -------------------------------------------------------------
    # CONSERVATIVE FALLBACK
    # -------------------------------------------------------------

    if min(
        ls,
        rs,
    ) < FALLBACK_SUPPORT_SPEED:

        if ls <= rs:

            support[i, 0] = True

        else:

            support[i, 1] = True


# =====================================================================
# FILL VERY SHORT CONTACT GAPS
# =====================================================================

def fill_short_gaps(
    mask,
    max_gap=3,
):

    mask = mask.copy()

    i = 1

    while i < len(mask) - 1:

        if mask[i]:

            i += 1
            continue


        start = i


        while (
            i < len(mask)
            and not mask[i]
        ):

            i += 1


        end = i

        gap = (
            end
            - start
        )


        if (
            gap <= max_gap
            and start > 0
            and end < len(mask)
            and mask[start - 1]
            and mask[end]
        ):

            mask[
                start:end
            ] = True


    return mask


support[:, 0] = fill_short_gaps(
    support[:, 0]
)

support[:, 1] = fill_short_gaps(
    support[:, 1]
)


# =====================================================================
# PER-FRAME REQUIRED ROOT-Z CORRECTION
# =====================================================================

dz_raw = np.full(
    n,
    np.nan,
    dtype=np.float64,
)


for i in range(n):

    left_contact = bool(
        support[i, 0]
    )

    right_contact = bool(
        support[i, 1]
    )


    if (
        left_contact
        and right_contact
    ):

        # Ground the lower of the two support soles.
        selected_clearance = min(
            left_clear[i],
            right_clear[i],
        )


    elif left_contact:

        selected_clearance = (
            left_clear[i]
        )


    elif right_contact:

        selected_clearance = (
            right_clear[i]
        )


    else:

        continue


    dz_raw[i] = (
        TARGET_CLEARANCE
        - selected_clearance
    )


# =====================================================================
# INTERPOLATE NON-CONTACT FRAMES
# =====================================================================

valid = np.isfinite(
    dz_raw
)


if np.sum(valid) < 2:

    raise RuntimeError(
        "Not enough support frames "
        "to construct Z correction."
    )


indices = np.arange(n)


dz = np.interp(
    indices,
    indices[valid],
    dz_raw[valid],
)


# =====================================================================
# SMOOTH
#
# Triangular kernel.
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
    dz,
    (
        half,
        half,
    ),
    mode="edge",
)


dz = np.convolve(
    padded,
    kernel,
    mode="valid",
)


# =====================================================================
# RATE LIMIT THE CORRECTION
# =====================================================================

for i in range(
    1,
    n,
):

    dz[i] = np.clip(
        dz[i],

        dz[i - 1]
        - MAX_DZ_CHANGE_PER_FRAME,

        dz[i - 1]
        + MAX_DZ_CHANGE_PER_FRAME,
    )


for i in range(
    n - 2,
    -1,
    -1,
):

    dz[i] = np.clip(
        dz[i],

        dz[i + 1]
        - MAX_DZ_CHANGE_PER_FRAME,

        dz[i + 1]
        + MAX_DZ_CHANGE_PER_FRAME,
    )


# Limit only clearly unreasonable corrections.
dz = np.clip(
    dz,
    -0.020,
    +0.090,
)


# =====================================================================
# APPLY ROOT-Z CORRECTION
# =====================================================================

root_corrected = (
    root_pos.copy()
)


root_corrected[:, 2] += (
    dz
)


# =====================================================================
# BUILD FULL QPOS AND RECOMPUTE VELOCITY
# =====================================================================

full_qpos = np.tile(
    stand_qpos[None, :],
    (
        n,
        1,
    ),
)


full_qpos[:, 0:3] = (
    root_corrected
)

full_qpos[:, 3:7] = (
    root_quat
)


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


for i in range(n):

    mujoco.mj_normalizeQuat(
        model,
        full_qpos[i],
    )


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


for i in range(
    1,
    n - 1,
):

    mujoco.mj_differentiatePos(
        model,
        full_qvel[i],
        2.0 * dt,
        full_qpos[i - 1],
        full_qpos[i + 1],
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


root_ang_vel = (
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

    joint_vel[:, j] = (
        full_qvel[
            :,
            vadr,
        ]
    )


# =====================================================================
# SECOND FK PASS
# =====================================================================

left_clear2 = np.zeros(
    n,
    dtype=np.float64,
)

right_clear2 = np.zeros(
    n,
    dtype=np.float64,
)

left_pos2 = np.zeros(
    (n, 3),
    dtype=np.float64,
)

right_pos2 = np.zeros(
    (n, 3),
    dtype=np.float64,
)


for frame in range(n):

    set_pose(
        frame,
        root_corrected,
    )


    left_clear2[
        frame
    ] = clearance(
        left_sole
    )


    right_clear2[
        frame
    ] = clearance(
        right_sole
    )


    left_pos2[
        frame
    ] = data.site_xpos[
        left_site
    ]


    right_pos2[
        frame
    ] = data.site_xpos[
        right_site
    ]


left_vel2 = np.gradient(
    left_pos2,
    dt,
    axis=0,
).astype(
    np.float32
)


right_vel2 = np.gradient(
    right_pos2,
    dt,
    axis=0,
).astype(
    np.float32
)


# =====================================================================
# VALIDATE SUPPORT CONTACTS
#
# Do not discard a support phase just because smoothing places
# the sole a few millimetres above zero.
# =====================================================================

contact_mask = (
    support.astype(
        np.float32
    )
)


support_clearances = []


for i in range(n):

    if support[i, 0]:

        support_clearances.append(
            left_clear2[i]
        )


    if support[i, 1]:

        support_clearances.append(
            right_clear2[i]
        )


support_clearances = np.asarray(
    support_clearances,
    dtype=np.float64,
)


# =====================================================================
# SAVE
# =====================================================================

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
        root_ang_vel,

    contact_mask=
        contact_mask,

    has_contact_mask=
        np.array(
            [True],
            dtype=np.bool_,
        ),

    left_foot_pos=
        left_pos2.astype(
            np.float32
        ),

    right_foot_pos=
        right_pos2.astype(
            np.float32
        ),

    left_foot_vel=
        left_vel2,

    right_foot_vel=
        right_vel2,

    left_sole_clearance=
        left_clear2.astype(
            np.float32
        ),

    right_sole_clearance=
        right_clear2.astype(
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
                "+X forward, yaw ~0"
            ]
        ),

    grounding_method=
        np.array(
            [
                "support-speed phase-dependent root-Z correction"
            ]
        ),

    source_dataset=
        np.array(
            [str(SRC)]
        ),
)


# =====================================================================
# REPORT
# =====================================================================

left_count = int(
    np.sum(
        support[:, 0]
    )
)

right_count = int(
    np.sum(
        support[:, 1]
    )
)

both_count = int(
    np.sum(
        support[:, 0]
        & support[:, 1]
    )
)

neither_count = int(
    np.sum(
        ~support[:, 0]
        & ~support[:, 1]
    )
)


def switch_frames(mask):

    return (
        np.where(
            mask[1:]
            != mask[:-1]
        )[0]
        + 1
    ).tolist()


print("=" * 120)
print("G1 GROUNDED TRACKING REFERENCE V3")
print("=" * 120)

print(
    "frames:",
    n,
)

print(
    "fps:",
    fps,
)


print()
print("=" * 120)
print("ROOT-Z CORRECTION")
print("=" * 120)

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


print()
print("=" * 120)
print("FINAL CONTACT MASK")
print("=" * 120)

print(
    "left frames:",
    left_count,
)

print(
    "right frames:",
    right_count,
)

print(
    "double support:",
    both_count,
)

print(
    "neither:",
    neither_count,
)

print(
    "left switches:",
    switch_frames(
        support[:, 0]
    ),
)

print(
    "right switches:",
    switch_frames(
        support[:, 1]
    ),
)


print()
print("=" * 120)
print("FINAL SOLE CLEARANCE")
print("=" * 120)

print(
    f"LEFT  min/median/max = "
    f"{np.min(left_clear2)*1000:+.2f} / "
    f"{np.median(left_clear2)*1000:+.2f} / "
    f"{np.max(left_clear2)*1000:+.2f} mm"
)

print(
    f"RIGHT min/median/max = "
    f"{np.min(right_clear2)*1000:+.2f} / "
    f"{np.median(right_clear2)*1000:+.2f} / "
    f"{np.max(right_clear2)*1000:+.2f} mm"
)


print()
print(
    "SUPPORT FOOT CLEARANCE:"
)

print(
    f"min    = "
    f"{np.min(support_clearances)*1000:+.2f} mm"
)

print(
    f"median = "
    f"{np.median(support_clearances)*1000:+.2f} mm"
)

print(
    f"p95    = "
    f"{np.percentile(support_clearances,95)*1000:+.2f} mm"
)

print(
    f"max    = "
    f"{np.max(support_clearances)*1000:+.2f} mm"
)


penetrating_support = int(
    np.sum(
        support_clearances
        < -0.005
    )
)


print(
    "support samples below -5 mm:",
    penetrating_support,
)


print()
print("=" * 120)
print("SELECTED FRAMES")
print("=" * 120)


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
        f"Lclr={left_clear2[frame]*1000:+7.1f}mm "
        f"Rclr={right_clear2[frame]*1000:+7.1f}mm "
        f"Lspd={left_speed[frame]:.3f} "
        f"Rspd={right_speed[frame]:.3f} "
        f"L/R="
        f"{int(support[frame,0])}/"
        f"{int(support[frame,1])}"
    )


print()
print("=" * 120)
print("DECISION METRICS")
print("=" * 120)

coverage = (
    1.0
    - neither_count
    / n
)


print(
    f"at-least-one-foot support coverage = "
    f"{100.0*coverage:.1f}%"
)

print(
    f"support penetration count (< -5mm) = "
    f"{penetrating_support}"
)


if (
    coverage >= 0.60
    and penetrating_support
    <= max(
        5,
        int(
            0.03
            * len(
                support_clearances
            )
        ),
    )
):

    print()
    print(
        "GROUNDING STATUS: PASS"
    )

    print(
        "Reference is suitable for "
        "closed-loop physics-tracking PPO."
    )

else:

    print()
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

print("=" * 120)
