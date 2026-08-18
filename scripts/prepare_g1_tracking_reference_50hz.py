from pathlib import Path
import numpy as np


SRC = Path(
    "datasets/processed/g1_amass_walking_il_15dof.npz"
)

DST = Path(
    "datasets/processed/g1_amass_walking_tracking_50hz.npz"
)

TARGET_FPS = 50.0


if not SRC.exists():
    raise SystemExit(f"Missing dataset: {SRC}")


data = np.load(
    SRC,
    allow_pickle=True,
)


print("=" * 100)
print("REFERENCE CONVERSION: 30 Hz -> 50 Hz")
print("=" * 100)

print("Input keys:")
for key in data.files:
    print(
        f"  {key:30s} "
        f"{getattr(data[key], 'shape', None)}"
    )


required = [
    "joint_pos_15",
    "controlled_joint_names",
    "fps",
]

for key in required:

    if key not in data:
        raise SystemExit(
            f"Required dataset key missing: {key}"
        )


src_q = np.asarray(
    data["joint_pos_15"],
    dtype=np.float32,
)

src_fps = float(
    np.asarray(
        data["fps"]
    ).reshape(-1)[0]
)

n_src = src_q.shape[0]

duration = (
    (n_src - 1)
    / src_fps
)

n_dst = (
    int(
        round(
            duration * TARGET_FPS
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


def interpolate_matrix(x):

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    result = np.empty(
        (
            n_dst,
            x.shape[1],
        ),
        dtype=np.float32,
    )

    for j in range(
        x.shape[1]
    ):

        result[:, j] = np.interp(
            dst_t,
            src_t,
            x[:, j],
        )

    return result


joint_pos = interpolate_matrix(
    src_q
)


# Recompute velocities from the new 50-Hz trajectory.
dt = float(
    dst_t[1] - dst_t[0]
)

joint_vel = np.gradient(
    joint_pos,
    dt,
    axis=0,
).astype(
    np.float32
)


# =====================================================================
# ROOT POSITION / VELOCITY
# =====================================================================

if "root_positions" in data:

    root_positions = interpolate_matrix(
        data["root_positions"]
    )

    root_velocity = np.gradient(
        root_positions,
        dt,
        axis=0,
    ).astype(
        np.float32
    )

else:

    root_positions = np.zeros(
        (
            n_dst,
            3,
        ),
        dtype=np.float32,
    )

    root_velocity = np.zeros_like(
        root_positions
    )


# =====================================================================
# CONTACT MASK
#
# Contacts are discrete, so use nearest reference sample.
# =====================================================================

has_contact_mask = (
    "contact_mask" in data
)

if has_contact_mask:

    src_contacts = np.asarray(
        data["contact_mask"],
        dtype=np.float32,
    )

    nearest = np.clip(
        np.rint(
            dst_t * src_fps
        ).astype(int),
        0,
        n_src - 1,
    )

    contact_mask = src_contacts[
        nearest
    ].copy()

else:

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
        joint_pos,

    joint_vel_15=
        joint_vel,

    root_positions=
        root_positions,

    root_velocity=
        root_velocity,

    contact_mask=
        contact_mask,

    has_contact_mask=np.array(
        [has_contact_mask],
        dtype=np.bool_,
    ),

    controlled_joint_names=
        np.asarray(
            data[
                "controlled_joint_names"
            ]
        ),

    fps=np.array(
        [TARGET_FPS],
        dtype=np.float32,
    ),

    source_fps=np.array(
        [src_fps],
        dtype=np.float32,
    ),

    source_frames=np.array(
        [n_src],
        dtype=np.int32,
    ),

    source_dataset=np.array(
        [str(SRC)],
    ),
)


print()
print("Saved:", DST)

print(
    "source:",
    n_src,
    "frames @",
    src_fps,
    "Hz",
)

print(
    "output:",
    n_dst,
    "frames @",
    TARGET_FPS,
    "Hz",
)

print(
    "duration:",
    f"{duration:.3f}s",
)

print(
    "joint_pos:",
    joint_pos.shape,
)

print(
    "joint_vel:",
    joint_vel.shape,
)

print(
    "contacts:",
    contact_mask.shape,
)
