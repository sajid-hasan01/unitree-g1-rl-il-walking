import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
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


# =====================================================================
# LOAD
# =====================================================================

if not DATASET.exists():
    raise FileNotFoundError(DATASET)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


d = np.load(
    DATASET,
    allow_pickle=True,
)


q_ref = np.asarray(
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

fps = float(
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


num_frames = q_ref.shape[0]

dt = 1.0 / fps


# =====================================================================
# MUJOCO MODEL
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

if floor_gid < 0:
    raise RuntimeError(
        "Floor geom not found."
    )


floor_z = float(
    model.geom_pos[
        floor_gid,
        2,
    ]
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


if left_site < 0 or right_site < 0:

    raise RuntimeError(
        "Foot site not found."
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


# =====================================================================
# FIND TRUE SOLE SPHERES
# =====================================================================

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


print("=" * 120)
print("G1 REFERENCE FOOT / CONTACT GEOMETRY AUDIT")
print("KINEMATIC REFERENCE ONLY - NO TRAINING")
print("=" * 120)

print(
    "frames:",
    num_frames,
)

print(
    "fps:",
    fps,
)

print(
    "floor z:",
    floor_z,
)

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
        "Could not identify sole geoms."
    )


# =====================================================================
# CONTROLLED JOINT ADDRESSES
# =====================================================================

qaddrs = []


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


# =====================================================================
# HELPERS
# =====================================================================

def sole_clearance(geom_ids):

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


        bottom_z = (
            center_z
            - radius
        )


        values.append(
            bottom_z
            - floor_z
        )


    return float(
        min(values)
    )


# =====================================================================
# KINEMATIC REPLAY
# =====================================================================

left_clear = np.zeros(
    num_frames,
    dtype=np.float64,
)

right_clear = np.zeros(
    num_frames,
    dtype=np.float64,
)


left_pos = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)

right_pos = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)


for frame in range(
    num_frames
):

    data.qpos[:] = stand_qpos
    data.qvel[:] = 0.0


    data.qpos[
        0:3
    ] = root_pos[
        frame
    ]


    data.qpos[
        3:7
    ] = root_quat[
        frame
    ]


    for i, qadr in enumerate(
        qaddrs
    ):

        data.qpos[
            qadr
        ] = q_ref[
            frame,
            i,
        ]


    mujoco.mj_normalizeQuat(
        model,
        data.qpos,
    )


    mujoco.mj_forward(
        model,
        data,
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


# =====================================================================
# FOOT VELOCITIES
# =====================================================================

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


left_speed_xy = np.linalg.norm(
    left_vel[:, 0:2],
    axis=1,
)

right_speed_xy = np.linalg.norm(
    right_vel[:, 0:2],
    axis=1,
)


# =====================================================================
# SUMMARY
# =====================================================================

print()
print("=" * 120)
print("RAW SOLE CLEARANCE")
print("=" * 120)


def stats(name, x):

    print(
        f"{name:10s}: "
        f"min={np.min(x)*1000:+8.2f} mm "
        f"p05={np.percentile(x,5)*1000:+8.2f} mm "
        f"median={np.median(x)*1000:+8.2f} mm "
        f"p95={np.percentile(x,95)*1000:+8.2f} mm "
        f"max={np.max(x)*1000:+8.2f} mm"
    )


stats(
    "LEFT",
    left_clear,
)

stats(
    "RIGHT",
    right_clear,
)


minimum_support_clearance = np.minimum(
    left_clear,
    right_clear,
)


print()
print(
    "minimum-foot clearance median:",
    f"{np.median(minimum_support_clearance)*1000:+.2f} mm",
)

print(
    "minimum-foot clearance p10:",
    f"{np.percentile(minimum_support_clearance,10)*1000:+.2f} mm",
)


# =====================================================================
# GLOBAL ROOT-Z ALIGNMENT ESTIMATE
#
# We only estimate here.
# Do not modify the reference yet.
# =====================================================================

recommended_z_offset = -float(
    np.percentile(
        minimum_support_clearance,
        10,
    )
)


print()
print("=" * 120)
print("ESTIMATED GLOBAL ROOT-Z CORRECTION")
print("=" * 120)

print(
    f"recommended constant Z offset ~ "
    f"{recommended_z_offset*1000:+.2f} mm"
)

print(
    "This is diagnostic only; nothing was changed."
)


# =====================================================================
# CONTACT CANDIDATES
#
# Support foot should be:
#
#   low enough
#   AND moving slowly horizontally.
#
# Try several physical thresholds.
# =====================================================================

print()
print("=" * 120)
print("CONTACT CANDIDATE SWEEP")
print("=" * 120)


for threshold_mm in [
    5,
    10,
    15,
    20,
    25,
    30,
]:

    threshold = (
        threshold_mm
        / 1000.0
    )


    # Apply diagnostic global Z correction.
    lc = (
        left_clear
        + recommended_z_offset
    )

    rc = (
        right_clear
        + recommended_z_offset
    )


    left_contact = (
        (lc <= threshold)
        & (
            left_speed_xy
            < 0.20
        )
    )


    right_contact = (
        (rc <= threshold)
        & (
            right_speed_xy
            < 0.20
        )
    )


    both = (
        left_contact
        & right_contact
    )


    neither = (
        ~left_contact
        & ~right_contact
    )


    left_only = (
        left_contact
        & ~right_contact
    )


    right_only = (
        right_contact
        & ~left_contact
    )


    print(
        f"threshold={threshold_mm:2d} mm | "
        f"L={np.sum(left_contact):3d} "
        f"R={np.sum(right_contact):3d} "
        f"Lonly={np.sum(left_only):3d} "
        f"Ronly={np.sum(right_only):3d} "
        f"both={np.sum(both):3d} "
        f"neither={np.sum(neither):3d}"
    )


# =====================================================================
# SELECT 20-MM CANDIDATE FOR EVENT INSPECTION
# =====================================================================

threshold = 0.020


corrected_left_clear = (
    left_clear
    + recommended_z_offset
)

corrected_right_clear = (
    right_clear
    + recommended_z_offset
)


left_contact = (
    (corrected_left_clear <= threshold)
    & (
        left_speed_xy < 0.20
    )
)


right_contact = (
    (corrected_right_clear <= threshold)
    & (
        right_speed_xy < 0.20
    )
)


# =====================================================================
# CONTACT SWITCHES
# =====================================================================

def switches(mask):

    return np.where(
        mask[1:]
        != mask[:-1]
    )[0] + 1


left_switch = switches(
    left_contact
)

right_switch = switches(
    right_contact
)


print()
print("=" * 120)
print("20-MM CONTACT EVENTS")
print("=" * 120)

print(
    "left switches:",
    left_switch.tolist(),
)

print(
    "right switches:",
    right_switch.tolist(),
)


print()
print(
    "selected frames:"
)


selected = [
    0,
    25,
    50,
    70,
    100,
    125,
    138,
    150,
    175,
    200,
    208,
    225,
    250,
    278,
    300,
    310,
]


for frame in selected:

    if frame >= num_frames:
        continue


    print(
        f"frame={frame:03d} "
        f"rootX={root_pos[frame,0]:+.3f} "
        f"Lclr={corrected_left_clear[frame]*1000:+7.1f}mm "
        f"Rclr={corrected_right_clear[frame]*1000:+7.1f}mm "
        f"Lspd={left_speed_xy[frame]:.3f} "
        f"Rspd={right_speed_xy[frame]:.3f} "
        f"L/R="
        f"{int(left_contact[frame])}/"
        f"{int(right_contact[frame])}"
    )


# =====================================================================
# FOOT MOTION RELATIVE TO ROOT
#
# This tells us whether the demonstrated leg movement is physically
# forward/backward relative to the pelvis.
# =====================================================================

left_rel = (
    left_pos
    - root_pos
)

right_rel = (
    right_pos
    - root_pos
)


print()
print("=" * 120)
print("FOOT MOTION RELATIVE TO ROOT")
print("=" * 120)


print(
    f"LEFT relative X range: "
    f"{left_rel[:,0].min():+.3f} to "
    f"{left_rel[:,0].max():+.3f} m"
)

print(
    f"RIGHT relative X range: "
    f"{right_rel[:,0].min():+.3f} to "
    f"{right_rel[:,0].max():+.3f} m"
)


print()
print(
    "Net world foot displacement:"
)

print(
    f"LEFT : "
    f"{left_pos[-1,0]-left_pos[0,0]:+.3f} m"
)

print(
    f"RIGHT: "
    f"{right_pos[-1,0]-right_pos[0,0]:+.3f} m"
)


# =====================================================================
# SAVE DIAGNOSTIC CSV
# =====================================================================

csv_path = (
    ROOT
    / "results"
    / "tracking_reference_foot_geometry.csv"
)


matrix = np.column_stack(
    [
        np.arange(
            num_frames
        ),

        root_pos[:,0],
        root_pos[:,1],
        root_pos[:,2],

        corrected_left_clear,
        corrected_right_clear,

        left_speed_xy,
        right_speed_xy,

        left_contact.astype(
            np.int32
        ),

        right_contact.astype(
            np.int32
        ),

        left_pos,
        right_pos,
    ]
)


header = (
    "frame,"
    "root_x,root_y,root_z,"
    "left_clearance,right_clearance,"
    "left_speed_xy,right_speed_xy,"
    "left_contact,right_contact,"
    "left_x,left_y,left_z,"
    "right_x,right_y,right_z"
)


np.savetxt(
    csv_path,
    matrix,
    delimiter=",",
    header=header,
    comments="",
)


print()
print("=" * 120)
print("FINAL")
print("=" * 120)

print(
    "CSV:",
    csv_path,
)

print(
    "NO REFERENCE WAS MODIFIED."
)

print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 120)

