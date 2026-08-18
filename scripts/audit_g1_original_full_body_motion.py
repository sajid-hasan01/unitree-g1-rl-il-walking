from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

PROCESSED_15 = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_il_15dof.npz"
)

TARGET_BASENAME = (
    "B1-standtowalk_poses_120_jpos.npz"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 125)
print("G1 ORIGINAL FULL-BODY MOTION AUDIT")
print("NO DATA MODIFICATION")
print("NO PPO TRAINING")
print("=" * 125)


# ================================================================
# CURRENT 15-DOF DATASET
# ================================================================

print()
print("=" * 125)
print("CURRENT 15-DOF DATASET")
print("=" * 125)


if not PROCESSED_15.exists():

    raise FileNotFoundError(
        PROCESSED_15
    )


current = np.load(
    PROCESSED_15,
    allow_pickle=True,
)


print(
    "path:",
    PROCESSED_15,
)


print(
    "keys:",
    current.files,
)


if "source_file" in current:

    source_value = str(
        np.asarray(
            current["source_file"]
        ).reshape(-1)[0]
    )

    print(
        "source_file metadata:",
        source_value,
    )

else:

    source_value = ""

    print(
        "source_file metadata: MISSING"
    )


print()
print(
    "current controlled joints:"
)


for i, name in enumerate(
    current[
        "controlled_joint_names"
    ]
):

    print(
        f"{i:02d} {str(name)}"
    )


# ================================================================
# SEARCH FOR ORIGINAL FILE LOCALLY
# ================================================================

print()
print("=" * 125)
print("SEARCHING FOR ORIGINAL SOURCE FILE")
print("=" * 125)


matches = list(
    ROOT.rglob(
        TARGET_BASENAME
    )
)


# Also search using basename from source_file metadata.
if source_value:

    metadata_name = Path(
        source_value
    ).name


    for p in ROOT.rglob(
        metadata_name
    ):

        if p not in matches:
            matches.append(p)


print(
    "number of matches:",
    len(matches),
)


for i, path in enumerate(
    matches
):

    print(
        f"[{i}] {path}"
    )


if not matches:

    print()
    print(
        "ORIGINAL SOURCE FILE NOT FOUND LOCALLY."
    )

    print(
        "Do not modify anything."
    )

    print(
        "Send this output back so the original "
        "download/preparation path can be recovered."
    )

    raise SystemExit(0)


SOURCE = matches[0]


print()
print(
    "Using:",
    SOURCE,
)


# ================================================================
# RAW SOURCE SCHEMA
# ================================================================

raw = np.load(
    SOURCE,
    allow_pickle=True,
)


print()
print("=" * 125)
print("RAW SOURCE NPZ SCHEMA")
print("=" * 125)


for key in raw.files:

    value = raw[key]

    print(
        f"{key:36s} "
        f"shape={str(value.shape):18s} "
        f"dtype={value.dtype}"
    )


# ================================================================
# ARRAY CANDIDATES
#
# We are specifically looking for:
#
# 29 actuated G1 joints
# 36 qpos = 7 floating base + 29 joints
# other likely retargeted pose layouts
# ================================================================

print()
print("=" * 125)
print("NUMERIC MOTION ARRAY CANDIDATES")
print("=" * 125)


candidate_keys = []


for key in raw.files:

    arr = np.asarray(
        raw[key]
    )


    if arr.ndim < 2:
        continue


    if not np.issubdtype(
        arr.dtype,
        np.number,
    ):
        continue


    # Time-series-like first dimension.
    if arr.shape[0] < 20:
        continue


    candidate_keys.append(
        key
    )


    print(
        f"{key:36s} "
        f"shape={arr.shape} "
        f"min={float(np.nanmin(arr)):+.5f} "
        f"max={float(np.nanmax(arr)):+.5f}"
    )


# ================================================================
# PRIORITIZE LIKELY FULL G1 JOINT ARRAYS
# ================================================================

print()
print("=" * 125)
print("LIKELY FULL-BODY JOINT ARRAYS")
print("=" * 125)


found_likely = False


for key in candidate_keys:

    arr = np.asarray(
        raw[key]
    )


    width = int(
        arr.shape[-1]
    )


    if width in [
        29,
        36,
        35,
        120,
    ]:

        found_likely = True

        print(
            f"{key:36s} "
            f"shape={arr.shape} "
            f"last_dim={width}"
        )


if not found_likely:

    print(
        "No obvious 29/35/36/120-column "
        "numeric array detected."
    )


# ================================================================
# SEARCH NAME / LABEL ARRAYS
# ================================================================

print()
print("=" * 125)
print("POSSIBLE JOINT-NAME / LABEL ARRAYS")
print("=" * 125)


name_keys = []


for key in raw.files:

    lower = key.lower()


    if any(
        token in lower
        for token in [
            "name",
            "joint",
            "dof",
            "body",
            "link",
        ]
    ):

        arr = np.asarray(
            raw[key]
        )


        print(
            f"{key:36s} "
            f"shape={arr.shape} "
            f"dtype={arr.dtype}"
        )


        if (
            arr.ndim == 1
            and arr.size <= 200
        ):

            print(
                "   values:",
                [
                    str(x)
                    for x in arr.tolist()
                ],
            )


        name_keys.append(
            key
        )


# ================================================================
# MUJOCO G1 JOINT / ACTUATOR INVENTORY
# ================================================================

print()
print("=" * 125)
print("MUJOCO G1 ACTUATED JOINT INVENTORY")
print("=" * 125)


model = mujoco.MjModel.from_xml_path(
    str(
        MODEL_PATH
    )
)


print(
    "nq:",
    model.nq,
)

print(
    "nv:",
    model.nv,
)

print(
    "nu:",
    model.nu,
)


actuated = []


for aid in range(
    model.nu
):

    actuator_name = (
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            aid,
        )
    )


    if actuator_name is None:
        actuator_name = (
            f"actuator_{aid}"
        )


    # Transmission joint id is in actuator_trnid[:,0]
    jid = int(
        model.actuator_trnid[
            aid,
            0,
        ]
    )


    if jid >= 0:

        joint_name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )
        )

    else:

        joint_name = None


    actuated.append(
        (
            aid,
            actuator_name,
            jid,
            joint_name,
        )
    )


for (
    aid,
    actuator_name,
    jid,
    joint_name,
) in actuated:

    print(
        f"{aid:02d} "
        f"actuator={actuator_name:30s} "
        f"joint={str(joint_name)}"
    )


# ================================================================
# MODEL JOINT ORDER AFTER FLOATING BASE
# ================================================================

print()
print("=" * 125)
print("ALL MUJOCO JOINTS")
print("=" * 125)


for jid in range(
    model.njnt
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        jid,
    )


    print(
        f"{jid:02d} "
        f"name={str(name):30s} "
        f"type={int(model.jnt_type[jid])} "
        f"qposadr={int(model.jnt_qposadr[jid])} "
        f"dofadr={int(model.jnt_dofadr[jid])}"
    )


# ================================================================
# CHECK WHETHER 29 ACTUATOR JOINTS ARE UNIQUE
# ================================================================

joint_names = [
    x[3]
    for x in actuated
    if x[3] is not None
]


print()
print("=" * 125)
print("29-DOF CONTROL FEASIBILITY")
print("=" * 125)

print(
    "actuators:",
    model.nu,
)

print(
    "actuator-linked joint names:",
    len(
        joint_names
    ),
)

print(
    "unique actuator-linked joints:",
    len(
        set(
            joint_names
        )
    ),
)


if (
    model.nu == 29
    and len(
        set(
            joint_names
        )
    ) == 29
):

    print(
        "MODEL SIDE: PASS"
    )

    print(
        "The MuJoCo G1 exposes 29 distinct "
        "joint-position actuator channels."
    )

else:

    print(
        "MODEL SIDE: REVIEW"
    )


# ================================================================
# COMPARE 15 CONTROLLED WITH 29 ACTUATED
# ================================================================

current15 = [
    str(x)
    for x in current[
        "controlled_joint_names"
    ]
]


upper_or_missing = [
    name
    for name in joint_names
    if name not in current15
]


print()
print(
    "15 currently controlled:"
)

for name in current15:
    print(
        "  ",
        name,
    )


print()
print(
    "14 currently excluded/frozen:"
)

for name in upper_or_missing:
    print(
        "  ",
        name,
    )


# ================================================================
# MOTION MAGNITUDE INSPECTION
#
# For every candidate with >=29 columns, show temporal standard
# deviation of columns. Moving upper-body columns should have
# nonzero variance.
# ================================================================

print()
print("=" * 125)
print("CANDIDATE MOTION VARIANCE")
print("=" * 125)


for key in candidate_keys:

    arr = np.asarray(
        raw[key],
        dtype=np.float64,
    )


    if (
        arr.ndim != 2
        or arr.shape[1] < 29
    ):
        continue


    std = np.std(
        arr,
        axis=0,
    )


    moving = int(
        np.sum(
            std > 1e-4
        )
    )


    print()
    print(
        f"{key}:"
    )

    print(
        "  shape:",
        arr.shape,
    )

    print(
        "  moving columns (>1e-4 std):",
        moving,
        "/",
        arr.shape[1],
    )

    print(
        "  first 40 column std:"
    )

    print(
        np.array2string(
            std[
                :min(
                    40,
                    len(std)
                )
            ],

            precision=5,

            suppress_small=True,
        )
    )


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 125)
print("AUDIT COMPLETE")
print("=" * 125)

print(
    "No files were changed."
)

print(
    "No controller was changed."
)

print(
    "No PPO training was performed."
)

print("=" * 125)
