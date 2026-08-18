from pathlib import Path
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

OLD = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_il_15dof.npz"
)


raw = np.load(
    RAW,
    allow_pickle=True,
)

old = np.load(
    OLD,
    allow_pickle=True,
)


raw_names = [
    str(x)
    for x in raw["dof_names"][:15]
]

old_names = [
    str(x)
    for x in old["controlled_joint_names"]
]


raw_q = np.asarray(
    raw["dof_positions"][:, :15],
    dtype=np.float64,
)

old_q = np.asarray(
    old["joint_pos_15"],
    dtype=np.float64,
)


if raw_q.shape != old_q.shape:
    raise RuntimeError(
        f"Shape mismatch: raw={raw_q.shape}, old={old_q.shape}"
    )


print("=" * 150)
print("RAW 29-DOF SOURCE vs OLD 15-DOF PREPROCESSING AUDIT")
print("=" * 150)

print(
    "names identical:",
    raw_names == old_names,
)

print(
    "shape:",
    raw_q.shape,
)


print()
print("=" * 150)
print("PER-JOINT DIFFERENCE")
print("=" * 150)

print(
    f"{'#':>2s} "
    f"{'JOINT':28s} "
    f"{'RAW_STD':>9s} "
    f"{'OLD_STD':>9s} "
    f"{'D_MEAN':>10s} "
    f"{'D_STD':>10s} "
    f"{'D_MIN':>10s} "
    f"{'D_MAX':>10s} "
    f"{'MAXABS':>10s} "
    f"{'TYPE':>16s}"
)

print("-" * 150)


classifications = []


for j, name in enumerate(raw_names):

    r = raw_q[:, j]
    o = old_q[:, j]

    d = o - r


    raw_std = float(
        np.std(r)
    )

    old_std = float(
        np.std(o)
    )

    d_mean = float(
        np.mean(d)
    )

    d_std = float(
        np.std(d)
    )

    d_min = float(
        np.min(d)
    )

    d_max = float(
        np.max(d)
    )

    maxabs = float(
        np.max(
            np.abs(d)
        )
    )


    # -------------------------------------------------------------
    # Linear fit:
    #
    # old ~= slope * raw + intercept
    # -------------------------------------------------------------

    if raw_std > 1e-10:

        A = np.column_stack(
            [
                r,
                np.ones_like(r),
            ]
        )

        slope, intercept = (
            np.linalg.lstsq(
                A,
                o,
                rcond=None,
            )[0]
        )

        predicted = (
            slope * r
            + intercept
        )

        linear_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        o
                        - predicted
                    ) ** 2
                )
            )
        )

    else:

        slope = np.nan
        intercept = np.nan
        linear_rmse = np.nan


    # -------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------

    if maxabs < 1e-7:

        kind = "EXACT"


    elif (
        np.ptp(d) < 1e-7
    ):

        kind = "CONST_OFFSET"


    elif (
        np.max(
            np.abs(o)
        ) < 1e-7

        and np.max(
            np.abs(r)
        ) > 1e-6
    ):

        kind = "ZEROED"


    elif (
        np.isfinite(
            linear_rmse
        )
        and linear_rmse < 1e-7
    ):

        kind = "LINEAR"


    else:

        kind = "OTHER"


    classifications.append(
        kind
    )


    print(
        f"{j:2d} "
        f"{name:28s} "
        f"{raw_std:9.5f} "
        f"{old_std:9.5f} "
        f"{d_mean:+10.6f} "
        f"{d_std:10.6f} "
        f"{d_min:+10.6f} "
        f"{d_max:+10.6f} "
        f"{maxabs:10.6f} "
        f"{kind:>16s}"
    )


    if kind == "LINEAR":

        print(
            f"     LINEAR FIT: "
            f"old = {slope:+.9f} * raw "
            f"{intercept:+.9f}, "
            f"RMSE={linear_rmse:.10f}"
        )


# =====================================================================
# MAXIMUM DIFFERENCE LOCATION
# =====================================================================

difference = (
    old_q
    - raw_q
)


flat_index = int(
    np.argmax(
        np.abs(
            difference
        )
    )
)


frame, joint = np.unravel_index(
    flat_index,
    difference.shape,
)


print()
print("=" * 150)
print("GLOBAL MAX DIFFERENCE")
print("=" * 150)

print(
    "frame:",
    frame,
)

print(
    "joint:",
    joint,
    raw_names[joint],
)

print(
    "raw:",
    f"{raw_q[frame,joint]:+.9f}",
)

print(
    "old:",
    f"{old_q[frame,joint]:+.9f}",
)

print(
    "old - raw:",
    f"{difference[frame,joint]:+.9f}",
)


# =====================================================================
# STATIC RAW JOINTS
# =====================================================================

print()
print("=" * 150)
print("STATIC / NEAR-STATIC RAW JOINTS")
print("=" * 150)


for j, name in enumerate(raw_names):

    r = raw_q[:, j]
    o = old_q[:, j]


    if np.std(r) < 1e-6:

        print(
            f"{j:02d} "
            f"{name:28s} "
            f"raw_constant={np.mean(r):+.9f} "
            f"old_mean={np.mean(o):+.9f} "
            f"old_std={np.std(o):.9f} "
            f"diff={np.mean(o-r):+.9f}"
        )


# =====================================================================
# FIRST FRAME
# =====================================================================

print()
print("=" * 150)
print("FRAME 0 VALUES")
print("=" * 150)


for j, name in enumerate(raw_names):

    print(
        f"{j:02d} "
        f"{name:28s} "
        f"raw={raw_q[0,j]:+.9f} "
        f"old={old_q[0,j]:+.9f} "
        f"diff={old_q[0,j]-raw_q[0,j]:+.9f}"
    )


# =====================================================================
# ROOT CHECK - continue the validation that the builder had not
# reached because it stopped early.
# =====================================================================

raw_root_pos = np.asarray(
    raw["body_positions"][:, 0, :],
    dtype=np.float64,
)

raw_root_quat = np.asarray(
    raw["body_rotations"][:, 0, :],
    dtype=np.float64,
)

old_root_pos = np.asarray(
    old["root_positions"],
    dtype=np.float64,
)

old_root_quat = np.asarray(
    old["root_rotations"],
    dtype=np.float64,
)


root_pos_diff = float(
    np.max(
        np.abs(
            raw_root_pos
            - old_root_pos
        )
    )
)


q_direct = np.linalg.norm(
    raw_root_quat
    - old_root_quat,
    axis=1,
)

q_flipped = np.linalg.norm(
    raw_root_quat
    + old_root_quat,
    axis=1,
)


root_quat_diff = float(
    np.max(
        np.minimum(
            q_direct,
            q_flipped,
        )
    )
)


print()
print("=" * 150)
print("ROOT VALIDATION")
print("=" * 150)

print(
    "root position max diff:",
    f"{root_pos_diff:.10f}",
)

print(
    "root quaternion equivalent diff:",
    f"{root_quat_diff:.10f}",
)


# =====================================================================
# DECISION
# =====================================================================

simple = all(
    x in {
        "EXACT",
        "CONST_OFFSET",
        "ZEROED",
        "LINEAR",
    }

    for x in classifications
)


print()
print("=" * 150)
print("DECISION")
print("=" * 150)

print(
    "classifications:",
    classifications,
)


if simple:

    print(
        "PREPROCESSING DIFFERENCE: DETERMINISTIC/SIMPLE"
    )

    print(
        "The old 15-DOF transformation can be "
        "reconstructed exactly."
    )

else:

    print(
        "PREPROCESSING DIFFERENCE: NONTRIVIAL"
    )

    print(
        "Do not build the 29-DOF reference yet."
    )


print()
print(
    "NO FILES WERE MODIFIED."
)

print("=" * 150)
