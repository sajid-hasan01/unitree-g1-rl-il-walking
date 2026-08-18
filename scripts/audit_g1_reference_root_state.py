import math
from pathlib import Path

import numpy as np


DATASET = Path(
    "datasets/processed/g1_amass_walking_il_15dof.npz"
)


d = np.load(
    DATASET,
    allow_pickle=True,
)


print("=" * 105)
print("ORIGINAL AMASS-G1 ROOT REFERENCE AUDIT")
print("=" * 105)

print("Dataset:", DATASET)

print()
print("KEYS")

for k in d.files:
    print(
        f"  {k:28s} "
        f"{getattr(d[k], 'shape', None)}"
    )


fps = float(
    np.asarray(
        d["fps"]
    ).reshape(-1)[0]
)


root = np.asarray(
    d["root_positions"],
    dtype=np.float64,
)


n = root.shape[0]

duration = (
    (n - 1)
    / fps
)


print()
print("=" * 105)
print("ROOT TRANSLATION")
print("=" * 105)

print(
    f"frames   = {n}"
)

print(
    f"fps      = {fps}"
)

print(
    f"duration = {duration:.3f} s"
)


delta = (
    root[-1]
    - root[0]
)


print()
print(
    "first root:",
    root[0],
)

print(
    "last root :",
    root[-1],
)

print(
    "delta     :",
    delta,
)


print()
print(
    "average raw axis velocities:"
)

for axis, name in enumerate(
    ["axis0", "axis1", "axis2"]
):

    v = (
        delta[axis]
        / duration
    )

    print(
        f"  {name}: "
        f"{v:+.5f} m/s"
    )


# Historical project mapping:
#
#   source root axis 1
#          ↓
#   MuJoCo X
#
mapped_x = (
    root[:, 1]
    - root[0, 1]
)


mapped_vx = np.gradient(
    mapped_x,
    1.0 / fps,
)


print()
print(
    "HISTORICAL MAPPING:"
)

print(
    "source axis 1 -> MuJoCo X"
)

print(
    f"mapped final X = "
    f"{mapped_x[-1]:+.4f} m"
)

print(
    f"mapped mean VX = "
    f"{np.mean(mapped_vx):+.4f} m/s"
)

print(
    f"mapped VX min/max = "
    f"{mapped_vx.min():+.4f} / "
    f"{mapped_vx.max():+.4f}"
)


print()
print(
    "Selected mapped X/VX frames:"
)

for frame in [
    0,
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    n - 1,
]:

    if frame >= n:
        continue

    print(
        f"  frame={frame:03d} "
        f"X={mapped_x[frame]:+.4f} "
        f"VX={mapped_vx[frame]:+.4f}"
    )


# ================================================================
# ROOT ROTATION
# ================================================================

rotation_key = None

for candidate in [
    "root_rotations",
    "root_rot",
    "root_rot_quat",
]:

    if candidate in d:
        rotation_key = candidate
        break


print()
print("=" * 105)
print("ROOT ROTATION")
print("=" * 105)


if rotation_key is None:

    print(
        "NO ROOT ROTATION KEY FOUND."
    )

    raise SystemExit(0)


rot = np.asarray(
    d[rotation_key],
    dtype=np.float64,
)


print(
    "rotation key:",
    rotation_key,
)

print(
    "shape:",
    rot.shape,
)


norms = np.linalg.norm(
    rot,
    axis=1,
)


print(
    f"quaternion norm min/max = "
    f"{norms.min():.6f} / "
    f"{norms.max():.6f}"
)


print()
print(
    "Raw quaternion samples:"
)

for frame in [
    0,
    42,
    83,
    125,
    167,
    n - 1,
]:

    if frame < n:

        print(
            f"  {frame:03d}: "
            f"{rot[frame]}"
        )


# ================================================================
# Test both possible quaternion conventions.
#
# Candidate A:
#     source is already [w,x,y,z]
#
# Candidate B:
#     source is [x,y,z,w], convert to MuJoCo wxyz
# ================================================================

def normalize(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    mag = np.linalg.norm(q)

    if mag < 1e-10:
        return np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    return q / mag


def quat_matrix_wxyz(q):

    w, x, y, z = normalize(q)

    return np.array(
        [
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w),
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w),
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y),
            ],
        ],
        dtype=np.float64,
    )


def yaw_wxyz(q):

    w, x, y, z = normalize(q)

    return math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )


def evaluate_candidate(mode):

    ups = []
    yaws = []


    for raw in rot:

        if mode == "wxyz":

            q = raw

        else:

            # xyzw -> wxyz
            q = np.array(
                [
                    raw[3],
                    raw[0],
                    raw[1],
                    raw[2],
                ],
                dtype=np.float64,
            )


        R = quat_matrix_wxyz(
            q
        )

        ups.append(
            float(
                R[2, 2]
            )
        )

        yaws.append(
            math.degrees(
                yaw_wxyz(
                    q
                )
            )
        )


    return (
        np.asarray(ups),
        np.asarray(yaws),
    )


for mode in [
    "wxyz",
    "xyzw",
]:

    ups, yaws = (
        evaluate_candidate(
            mode
        )
    )


    print()
    print(
        "-" * 105
    )

    print(
        f"ASSUMING SOURCE QUATERNION = "
        f"{mode}"
    )

    print(
        f"mean up_z = "
        f"{np.mean(ups):+.4f}"
    )

    print(
        f"min/max up_z = "
        f"{np.min(ups):+.4f} / "
        f"{np.max(ups):+.4f}"
    )

    print(
        f"yaw min/max = "
        f"{np.min(yaws):+.2f} / "
        f"{np.max(yaws):+.2f} deg"
    )


    print(
        "selected frames:"
    )

    for frame in [
        0,
        42,
        83,
        125,
        167,
        n - 1,
    ]:

        if frame < n:

            print(
                f"  frame={frame:03d} "
                f"up={ups[frame]:+.4f} "
                f"yaw={yaws[frame]:+.2f}"
            )


print()
print("=" * 105)
print("IMPORTANT")
print("=" * 105)

print(
    "Do not train PPO from this audit."
)

print(
    "We will use these numbers to determine:"
)

print(
    "1. the correct AMASS -> MuJoCo forward-axis mapping,"
)

print(
    "2. the correct root quaternion convention,"
)

print(
    "3. the correct root linear/angular state for RSI,"
)

print(
    "4. whether frame-randomized training is physically valid."
)

print("=" * 105)
