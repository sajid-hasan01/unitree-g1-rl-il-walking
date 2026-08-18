from pathlib import Path
import csv
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


# ================================================================
# CONFIG
# ================================================================

REFERENCE = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
    / "medium_08_50hz_grounded.npz"
)


OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_medium08_foot_slip_source_decomposition.csv"
)


OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_medium08_foot_slip_source_decomposition.npz"
)


CONTACT_DEPTH = 1e-6

MAX_ROOT_LOWERING = 0.025


# Frozen Stage-7T detector.
CLEARANCE_ENTER = 0.005
CLEARANCE_EXIT = 0.007

VXY_ENTER = 0.40
VXY_EXIT = 0.60

VZ_ENTER = 0.15
VZ_EXIT = 0.25


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model


LEFT_GEOMS = list(
    env.left_sole_geoms
)

RIGHT_GEOMS = list(
    env.right_sole_geoms
)


FLOOR_GEOM = int(
    env.floor_geom
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


print("=" * 205)
print("G1 STAGE 7V")
print("MEDIUM_08 FOOT-SLIP SOURCE DECOMPOSITION")
print("ROOT vs LEG KINEMATIC CONTRIBUTION")
print("NO OPTIMIZATION / NO REFERENCE MODIFICATION / NO PPO")
print("=" * 205)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "reference:",
    REFERENCE,
)

print(
    "model nq/nv:",
    model.nq,
    model.nv,
)

print(
    "left foot body:",
    LEFT_BODY,
)

print(
    "right foot body:",
    RIGHT_BODY,
)


# ================================================================
# LOAD
# ================================================================

if not REFERENCE.exists():

    raise FileNotFoundError(
        REFERENCE
    )


with np.load(
    REFERENCE,
    allow_pickle=True,
) as f:

    qpos = np.asarray(
        f[
            "full_qpos"
        ],
        dtype=np.float64,
    ).copy()


    fps = (
        float(
            np.asarray(
                f[
                    "fps"
                ]
            ).reshape(
                -1
            )[0]
        )
        if "fps" in f.files
        else 50.0
    )


N = len(
    qpos
)


DT = 1.0 / fps


print(
    "frames:",
    N,
)

print(
    "fps:",
    fps,
)


# ================================================================
# HELPERS
# ================================================================

def differentiate_qpos(
    trajectory,
):

    velocity = np.zeros(
        (
            N,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        velocity[0],
        DT,
        trajectory[0],
        trajectory[1],
    )


    for frame in range(
        1,
        N - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            velocity[frame],
            2.0 * DT,
            trajectory[frame - 1],
            trajectory[frame + 1],
        )


    mujoco.mj_differentiatePos(
        model,
        velocity[-1],
        DT,
        trajectory[-2],
        trajectory[-1],
    )


    return velocity


def differentiate_array(
    values,
):

    derivative = np.zeros_like(
        values
    )


    derivative[1:-1] = (
        values[2:]
        -
        values[:-2]
    ) / (
        2.0 * DT
    )


    derivative[0] = (
        values[1]
        -
        values[0]
    ) / DT


    derivative[-1] = (
        values[-1]
        -
        values[-2]
    ) / DT


    return derivative


def sole_clearance(
    data,
    geoms,
):

    floor_z = float(
        data.geom_xpos[
            FLOOR_GEOM,
            2
        ]
    )


    value = float(
        "inf"
    )


    for gid in geoms:

        radius = float(
            model.geom_size[
                gid,
                0
            ]
        )


        bottom = (
            float(
                data.geom_xpos[
                    gid,
                    2
                ]
            )
            -
            radius
        )


        value = min(
            value,
            bottom - floor_z,
        )


    return value


def sole_center(
    data,
    geoms,
):

    return np.mean(
        data.geom_xpos[
            geoms
        ],
        axis=0,
    )


def find_segments(
    mask,
):

    segments = []

    start = None


    for frame, active in enumerate(
        mask
    ):

        if active and start is None:

            start = frame


        elif (
            not active
            and
            start is not None
        ):

            segments.append(
                (
                    start,
                    frame - 1,
                )
            )

            start = None


    if start is not None:

        segments.append(
            (
                start,
                len(mask) - 1,
            )
        )


    return segments


def cosine(
    a,
    b,
):

    na = float(
        np.linalg.norm(
            a
        )
    )


    nb = float(
        np.linalg.norm(
            b
        )
    )


    if (
        na < 1e-10
        or
        nb < 1e-10
    ):

        return float(
            "nan"
        )


    return float(
        np.dot(
            a,
            b,
        )
        /
        (
            na
            *
            nb
        )
    )


# ================================================================
# RAW CLEARANCE
# ================================================================

probe = mujoco.MjData(
    model
)


raw_left_clearance = np.zeros(
    N,
    dtype=np.float64,
)


raw_right_clearance = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    probe.qpos[:] = (
        qpos[
            frame
        ]
    )

    probe.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        probe,
    )


    raw_left_clearance[
        frame
    ] = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    raw_right_clearance[
        frame
    ] = sole_clearance(
        probe,
        RIGHT_GEOMS,
    )


# ================================================================
# GLOBAL-LOWEST PROJECTION
# ================================================================

projected_qpos = (
    qpos.copy()
)


root_shift = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    minimum = min(
        raw_left_clearance[
            frame
        ],
        raw_right_clearance[
            frame
        ],
    )


    dz = (
        -CONTACT_DEPTH
        -
        minimum
    )


    dz = min(
        0.0,
        dz,
    )


    dz = max(
        -MAX_ROOT_LOWERING,
        dz,
    )


    projected_qpos[
        frame,
        2
    ] += dz


    root_shift[
        frame
    ] = dz


qvel = differentiate_qpos(
    projected_qpos
)


# ================================================================
# PROJECTED FOOT TRAJECTORY
# ================================================================

left_clearance = np.zeros(
    N,
    dtype=np.float64,
)


right_clearance = np.zeros(
    N,
    dtype=np.float64,
)


left_position = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


right_position = np.zeros_like(
    left_position
)


for frame in range(N):

    probe.qpos[:] = (
        projected_qpos[
            frame
        ]
    )


    probe.qvel[:] = (
        qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        probe,
    )


    left_clearance[
        frame
    ] = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    right_clearance[
        frame
    ] = sole_clearance(
        probe,
        RIGHT_GEOMS,
    )


    left_position[
        frame
    ] = sole_center(
        probe,
        LEFT_GEOMS,
    )


    right_position[
        frame
    ] = sole_center(
        probe,
        RIGHT_GEOMS,
    )


left_fd_velocity = differentiate_array(
    left_position
)


right_fd_velocity = differentiate_array(
    right_position
)


left_xy = np.linalg.norm(
    left_fd_velocity[
        :,
        0:2
    ],
    axis=1,
)


right_xy = np.linalg.norm(
    right_fd_velocity[
        :,
        0:2
    ],
    axis=1,
)


left_vz = (
    left_fd_velocity[
        :,
        2
    ]
)


right_vz = (
    right_fd_velocity[
        :,
        2
    ]
)


# ================================================================
# STAGE-7T SUPPORT DETECTOR
# ================================================================

support = np.zeros(
    (
        N,
        2,
    ),
    dtype=bool,
)


for frame in range(N):

    for side in (
        0,
        1,
    ):

        if side == 0:

            clearance = (
                left_clearance[
                    frame
                ]
            )

            xy = (
                left_xy[
                    frame
                ]
            )

            vz = abs(
                left_vz[
                    frame
                ]
            )


        else:

            clearance = (
                right_clearance[
                    frame
                ]
            )

            xy = (
                right_xy[
                    frame
                ]
            )

            vz = abs(
                right_vz[
                    frame
                ]
            )


        previous = (
            frame > 0
            and
            support[
                frame - 1,
                side
            ]
        )


        if previous:

            active = (
                clearance
                <=
                CLEARANCE_EXIT

                and
                xy
                <=
                VXY_EXIT

                and
                vz
                <=
                VZ_EXIT
            )


        else:

            active = (
                clearance
                <=
                CLEARANCE_ENTER

                and
                xy
                <=
                VXY_ENTER

                and
                vz
                <=
                VZ_ENTER
            )


        support[
            frame,
            side
        ] = active


none_mask = (
    np.sum(
        support,
        axis=1,
    )
    ==
    0
)


segments = find_segments(
    none_mask
)


print()
print("=" * 205)
print("STAGE-7T NONE GAPS")
print("=" * 205)

print(
    "NONE frames:",
    int(
        np.sum(
            none_mask
        )
    ),
)

print(
    "NONE fraction:",
    f"{100*np.mean(none_mask):.1f}%",
)

print(
    "segments:",
    segments,
)


# Baseline gate from Stage 7U.
if int(
    np.sum(
        none_mask
    )
) != 36:

    raise RuntimeError(
        "medium_08 Stage-7T NONE baseline changed."
    )


# ================================================================
# JACOBIAN VELOCITY DECOMPOSITION
# ================================================================

jac_data = mujoco.MjData(
    model
)


# MuJoCo generalized-velocity addresses.
LEFT_LEG_VADDR = [
    int(
        env.vaddrs[
            i
        ]
    )
    for i in range(
        0,
        6
    )
]


RIGHT_LEG_VADDR = [
    int(
        env.vaddrs[
            i
        ]
    )
    for i in range(
        6,
        12
    )
]


print()
print(
    "left leg velocity addresses:",
    LEFT_LEG_VADDR,
)

print(
    "right leg velocity addresses:",
    RIGHT_LEG_VADDR,
)


lower_side = np.zeros(
    N,
    dtype=np.int8,
)


velocity_total = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)


velocity_base = np.zeros_like(
    velocity_total
)


velocity_base_translation = np.zeros_like(
    velocity_total
)


velocity_base_rotation = np.zeros_like(
    velocity_total
)


velocity_leg = np.zeros_like(
    velocity_total
)


velocity_other = np.zeros_like(
    velocity_total
)


finite_difference_velocity = np.zeros_like(
    velocity_total
)


base_xy = np.zeros(
    N,
    dtype=np.float64,
)


leg_xy = np.zeros(
    N,
    dtype=np.float64,
)


total_xy = np.zeros(
    N,
    dtype=np.float64,
)


fd_xy = np.zeros(
    N,
    dtype=np.float64,
)


base_leg_cos = np.full(
    N,
    np.nan,
    dtype=np.float64,
)


cancellation_fraction = np.full(
    N,
    np.nan,
    dtype=np.float64,
)


jacobian_fd_error = np.zeros(
    N,
    dtype=np.float64,
)


for frame in range(N):

    jac_data.qpos[:] = (
        projected_qpos[
            frame
        ]
    )


    jac_data.qvel[:] = (
        qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        jac_data,
    )


    # ------------------------------------------------------------
    # Select physically lower foot.
    # ------------------------------------------------------------

    if (
        left_clearance[
            frame
        ]
        <=
        right_clearance[
            frame
        ]
    ):

        side = 0

        body = (
            LEFT_BODY
        )

        point = sole_center(
            jac_data,
            LEFT_GEOMS,
        )


        leg_addresses = (
            LEFT_LEG_VADDR
        )


        fd_velocity = (
            left_fd_velocity[
                frame
            ]
        )


    else:

        side = 1

        body = (
            RIGHT_BODY
        )


        point = sole_center(
            jac_data,
            RIGHT_GEOMS,
        )


        leg_addresses = (
            RIGHT_LEG_VADDR
        )


        fd_velocity = (
            right_fd_velocity[
                frame
            ]
        )


    lower_side[
        frame
    ] = side


    finite_difference_velocity[
        frame
    ] = (
        fd_velocity
    )


    # ------------------------------------------------------------
    # Point Jacobian.
    # ------------------------------------------------------------

    jacp = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=np.float64,
    )


    jacr = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_jac(
        model,
        jac_data,
        jacp,
        jacr,
        point,
        body,
    )


    v = qvel[
        frame
    ]


    # Full point velocity.
    v_total = (
        jacp
        @
        v
    )


    # Floating base = first six generalized velocities.
    v_base = (
        jacp[
            :,
            0:6
        ]
        @
        v[
            0:6
        ]
    )


    v_base_translation = (
        jacp[
            :,
            0:3
        ]
        @
        v[
            0:3
        ]
    )


    v_base_rotation = (
        jacp[
            :,
            3:6
        ]
        @
        v[
            3:6
        ]
    )


    # Same-side leg contribution.
    v_leg = np.zeros(
        3,
        dtype=np.float64,
    )


    for address in leg_addresses:

        v_leg += (
            jacp[
                :,
                address
            ]
            *
            v[
                address
            ]
        )


    # Whatever remains should be essentially zero
    # for a foot point in this kinematic tree.
    v_other = (
        v_total
        -
        v_base
        -
        v_leg
    )


    velocity_total[
        frame
    ] = v_total


    velocity_base[
        frame
    ] = v_base


    velocity_base_translation[
        frame
    ] = v_base_translation


    velocity_base_rotation[
        frame
    ] = v_base_rotation


    velocity_leg[
        frame
    ] = v_leg


    velocity_other[
        frame
    ] = v_other


    base_xy[
        frame
    ] = np.linalg.norm(
        v_base[
            0:2
        ]
    )


    leg_xy[
        frame
    ] = np.linalg.norm(
        v_leg[
            0:2
        ]
    )


    total_xy[
        frame
    ] = np.linalg.norm(
        v_total[
            0:2
        ]
    )


    fd_xy[
        frame
    ] = np.linalg.norm(
        fd_velocity[
            0:2
        ]
    )


    base_leg_cos[
        frame
    ] = cosine(
        v_base[
            0:2
        ],
        v_leg[
            0:2
        ],
    )


    denominator = (
        base_xy[
            frame
        ]
        +
        leg_xy[
            frame
        ]
    )


    if denominator > 1e-10:

        cancellation_fraction[
            frame
        ] = (
            1.0
            -
            total_xy[
                frame
            ]
            /
            denominator
        )


    jacobian_fd_error[
        frame
    ] = np.linalg.norm(
        v_total
        -
        fd_velocity
    )


# ================================================================
# JACOBIAN SANITY
# ================================================================

print()
print("=" * 205)
print("JACOBIAN VALIDATION")
print("=" * 205)

print(
    "Jacobian vs finite-difference velocity error:"
)

print(
    "p50:",
    f"{np.percentile(jacobian_fd_error,50):.4f}",
    "m/s",
)

print(
    "p95:",
    f"{np.percentile(jacobian_fd_error,95):.4f}",
    "m/s",
)

print(
    "max:",
    f"{np.max(jacobian_fd_error):.4f}",
    "m/s",
)


print(
    "other-joint contribution p95:",
    f"{np.percentile(np.linalg.norm(velocity_other,axis=1),95):.6f}",
    "m/s",
)


# ================================================================
# GLOBAL NONE-FRAME DECOMPOSITION
# ================================================================

def print_distribution(
    name,
    values,
    mask,
):

    values = np.asarray(
        values
    )


    selected = (
        values[
            mask
        ]
    )


    print(
        f"{name:30s} "
        f"p50={np.percentile(selected,50):8.3f} "
        f"p95={np.percentile(selected,95):8.3f} "
        f"max={np.max(selected):8.3f}"
    )


print()
print("=" * 205)
print("NONE-FRAME FOOT-VELOCITY DECOMPOSITION")
print("=" * 205)


print_distribution(
    "base XY contribution",
    base_xy,
    none_mask,
)


print_distribution(
    "leg XY contribution",
    leg_xy,
    none_mask,
)


print_distribution(
    "actual Jacobian foot XY",
    total_xy,
    none_mask,
)


print_distribution(
    "finite-difference foot XY",
    fd_xy,
    none_mask,
)


print_distribution(
    "|base translation XYZ|",
    np.linalg.norm(
        velocity_base_translation,
        axis=1,
    ),
    none_mask,
)


print_distribution(
    "|base rotation XYZ|",
    np.linalg.norm(
        velocity_base_rotation,
        axis=1,
    ),
    none_mask,
)


print_distribution(
    "|leg XYZ|",
    np.linalg.norm(
        velocity_leg,
        axis=1,
    ),
    none_mask,
)


valid_cos = (
    none_mask
    &
    np.isfinite(
        base_leg_cos
    )
)


valid_cancel = (
    none_mask
    &
    np.isfinite(
        cancellation_fraction
    )
)


print()
print(
    "base-vs-leg XY cosine:"
)

print(
    " p50:",
    f"{np.percentile(base_leg_cos[valid_cos],50):+.3f}",
)

print(
    " p95:",
    f"{np.percentile(base_leg_cos[valid_cos],95):+.3f}",
)


print()
print(
    "cancellation fraction:"
)

print(
    " 0.0  = no cancellation"
)

print(
    " 1.0  = near-perfect cancellation"
)

print(
    " <0   = contributions amplify one another"
)

print(
    " p50:",
    f"{np.percentile(cancellation_fraction[valid_cancel],50):+.3f}",
)

print(
    " p95:",
    f"{np.percentile(cancellation_fraction[valid_cancel],95):+.3f}",
)


# ================================================================
# PER-GAP DECOMPOSITION
# ================================================================

print()
print("=" * 205)
print("PER UNSUPPORTED GAP")
print("=" * 205)


gap_rows = []


for gap_index, (
    start,
    end,
) in enumerate(
    segments
):

    frames = np.arange(
        start,
        end + 1,
        dtype=np.int32,
    )


    left_count = int(
        np.sum(
            lower_side[
                frames
            ]
            ==
            0
        )
    )


    right_count = int(
        np.sum(
            lower_side[
                frames
            ]
            ==
            1
        )
    )


    dominant_side = (
        "L"
        if left_count
        >=
        right_count
        else "R"
    )


    gap_cos = (
        base_leg_cos[
            frames
        ]
    )


    gap_cancel = (
        cancellation_fraction[
            frames
        ]
    )


    finite_cos = gap_cos[
        np.isfinite(
            gap_cos
        )
    ]


    finite_cancel = gap_cancel[
        np.isfinite(
            gap_cancel
        )
    ]


    row = {

        "gap":
            gap_index,

        "start":
            start,

        "end":
            end,

        "length":
            end
            -
            start
            +
            1,

        "dominant_side":
            dominant_side,

        "base_xy_p50":
            float(
                np.percentile(
                    base_xy[
                        frames
                    ],
                    50,
                )
            ),

        "base_xy_p95":
            float(
                np.percentile(
                    base_xy[
                        frames
                    ],
                    95,
                )
            ),

        "leg_xy_p50":
            float(
                np.percentile(
                    leg_xy[
                        frames
                    ],
                    50,
                )
            ),

        "leg_xy_p95":
            float(
                np.percentile(
                    leg_xy[
                        frames
                    ],
                    95,
                )
            ),

        "total_xy_p50":
            float(
                np.percentile(
                    total_xy[
                        frames
                    ],
                    50,
                )
            ),

        "total_xy_p95":
            float(
                np.percentile(
                    total_xy[
                        frames
                    ],
                    95,
                )
            ),

        "cos_p50":
            float(
                np.percentile(
                    finite_cos,
                    50,
                )
            ),

        "cancel_p50":
            float(
                np.percentile(
                    finite_cancel,
                    50,
                )
            ),

        "base_trans_p50":
            float(
                np.percentile(
                    np.linalg.norm(
                        velocity_base_translation[
                            frames
                        ],
                        axis=1,
                    ),
                    50,
                )
            ),

        "base_rot_p50":
            float(
                np.percentile(
                    np.linalg.norm(
                        velocity_base_rotation[
                            frames
                        ],
                        axis=1,
                    ),
                    50,
                )
            ),
    }


    gap_rows.append(
        row
    )


    print()
    print(
        f"GAP {gap_index:02d} "
        f"{start:03d}..{end:03d} "
        f"len={row['length']} "
        f"lower={dominant_side}"
    )


    print(
        " base XY p50/p95:",
        f"{row['base_xy_p50']:.3f}",
        "/",
        f"{row['base_xy_p95']:.3f}",
        "m/s",
    )


    print(
        " leg  XY p50/p95:",
        f"{row['leg_xy_p50']:.3f}",
        "/",
        f"{row['leg_xy_p95']:.3f}",
        "m/s",
    )


    print(
        " foot XY p50/p95:",
        f"{row['total_xy_p50']:.3f}",
        "/",
        f"{row['total_xy_p95']:.3f}",
        "m/s",
    )


    print(
        " base-leg cosine p50:",
        f"{row['cos_p50']:+.3f}",
    )


    print(
        " cancellation p50:",
        f"{row['cancel_p50']:+.3f}",
    )


    print(
        " base translational contribution p50:",
        f"{row['base_trans_p50']:.3f}",
        "m/s",
    )


    print(
        " base rotational contribution p50:",
        f"{row['base_rot_p50']:.3f}",
        "m/s",
    )


# ================================================================
# WORST FRAMES
# ================================================================

print()
print("=" * 205)
print("WORST NONE-FRAME FOOT SLIP")
print("=" * 205)


none_indices = np.where(
    none_mask
)[0]


order = none_indices[
    np.argsort(
        total_xy[
            none_indices
        ]
    )[
        ::-1
    ]
]


for rank, frame in enumerate(
    order[:15],
    start=1,
):

    side = (
        "L"
        if lower_side[
            frame
        ]
        ==
        0
        else "R"
    )


    print(
        f"{rank:02d}. "
        f"frame={frame:03d} "
        f"side={side} "
        f"baseXY={base_xy[frame]:6.3f} "
        f"legXY={leg_xy[frame]:6.3f} "
        f"footXY={total_xy[frame]:6.3f} "
        f"cos={base_leg_cos[frame]:+6.3f} "
        f"cancel={cancellation_fraction[frame]:+6.3f} "
        f"baseXYZ={np.linalg.norm(velocity_base[frame]):6.3f} "
        f"legXYZ={np.linalg.norm(velocity_leg[frame]):6.3f} "
        f"fdXY={fd_xy[frame]:6.3f}"
    )


# ================================================================
# CLASSIFICATION
# ================================================================

none_base_median = float(
    np.percentile(
        base_xy[
            none_mask
        ],
        50,
    )
)


none_leg_median = float(
    np.percentile(
        leg_xy[
            none_mask
        ],
        50,
    )
)


none_total_median = float(
    np.percentile(
        total_xy[
            none_mask
        ],
        50,
    )
)


none_cos_median = float(
    np.percentile(
        base_leg_cos[
            valid_cos
        ],
        50,
    )
)


none_cancel_median = float(
    np.percentile(
        cancellation_fraction[
            valid_cancel
        ],
        50,
    )
)


print()
print("=" * 205)
print("STAGE 7V DIAGNOSIS")
print("=" * 205)

print(
    "NONE-foot XY median:"
)

print(
    " base:",
    f"{none_base_median:.3f}",
)

print(
    " leg:",
    f"{none_leg_median:.3f}",
)

print(
    " total:",
    f"{none_total_median:.3f}",
)

print(
    "base-leg cosine median:",
    f"{none_cos_median:+.3f}",
)

print(
    "cancellation median:",
    f"{none_cancel_median:+.3f}",
)

print()


if (
    none_cos_median
    <=
    -0.60
    and
    none_base_median
    >
    1.5
    *
    none_leg_median
):

    print(
        "PRIMARY DIAGNOSIS:"
    )

    print(
        "ROOT/PELVIS MOTION DOMINATES."
    )

    print(
        "The leg tries to oppose the base motion, "
        "but cannot cancel enough of it."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Repair the root/pelvis trajectory inside "
        "the three medium_04 unsupported intervals, "
        "while preserving leg joint motion."
    )


elif (
    none_cos_median
    <=
    -0.60
    and
    0.60
    <=
    none_base_median
    /
    max(
        none_leg_median,
        1e-9,
    )
    <=
    1.67
):

    print(
        "PRIMARY DIAGNOSIS:"
    )

    print(
        "ROOT AND LEG MOTION ARE BOTH LARGE AND "
        "OPPOSE ONE ANOTHER, BUT THEIR MAGNITUDES/"
        "TIMING DO NOT CANCEL."
    )

    print()
    print(
        "THIS IS A ROOT-LEG SYNCHRONIZATION PROBLEM."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Use a stance-foot velocity constraint to "
        "derive the smallest temporally smooth root "
        "correction required for cancellation."
    )


elif (
    none_cos_median
    >
    -0.30
):

    print(
        "PRIMARY DIAGNOSIS:"
    )

    print(
        "LEG MOTION DOES NOT OPPOSE ROOT MOTION "
        "CORRECTLY."
    )

    print(
        "The retargeted leg trajectory is contributing "
        "to the stance-foot slip instead of cancelling it."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Revisit/re-retarget lower-body joint motion "
        "rather than correcting root motion alone."
    )


else:

    print(
        "PRIMARY DIAGNOSIS:"
    )

    print(
        "MIXED ROOT/LEG KINEMATIC FAILURE."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Use the per-gap decomposition above to select "
        "root-only versus root+leg repair independently "
        "for each unsupported interval."
    )


# ================================================================
# SAVE
# ================================================================

csv_rows = []


for frame in range(N):

    if not none_mask[
        frame
    ]:

        continue


    csv_rows.append(
        {

            "frame":
                frame,

            "time_s":
                frame
                /
                fps,

            "lower_side":
                "L"
                if lower_side[
                    frame
                ]
                ==
                0
                else "R",

            "left_clearance_mm":
                1000
                *
                left_clearance[
                    frame
                ],

            "right_clearance_mm":
                1000
                *
                right_clearance[
                    frame
                ],

            "base_vx":
                velocity_base[
                    frame,
                    0
                ],

            "base_vy":
                velocity_base[
                    frame,
                    1
                ],

            "base_vz":
                velocity_base[
                    frame,
                    2
                ],

            "leg_vx":
                velocity_leg[
                    frame,
                    0
                ],

            "leg_vy":
                velocity_leg[
                    frame,
                    1
                ],

            "leg_vz":
                velocity_leg[
                    frame,
                    2
                ],

            "foot_vx":
                velocity_total[
                    frame,
                    0
                ],

            "foot_vy":
                velocity_total[
                    frame,
                    1
                ],

            "foot_vz":
                velocity_total[
                    frame,
                    2
                ],

            "base_xy":
                base_xy[
                    frame
                ],

            "leg_xy":
                leg_xy[
                    frame
                ],

            "foot_xy":
                total_xy[
                    frame
                ],

            "base_leg_cos":
                base_leg_cos[
                    frame
                ],

            "cancellation":
                cancellation_fraction[
                    frame
                ],

            "jacobian_fd_error":
                jacobian_fd_error[
                    frame
                ],
        }
    )


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            csv_rows[
                0
            ].keys()
        ),
    )


    writer.writeheader()


    for row in csv_rows:

        writer.writerow(
            row
        )


np.savez_compressed(
    OUTPUT_NPZ,

    none_mask=
        none_mask,

    support=
        support,

    lower_side=
        lower_side,

    velocity_total=
        velocity_total.astype(
            np.float32
        ),

    velocity_base=
        velocity_base.astype(
            np.float32
        ),

    velocity_base_translation=
        velocity_base_translation.astype(
            np.float32
        ),

    velocity_base_rotation=
        velocity_base_rotation.astype(
            np.float32
        ),

    velocity_leg=
        velocity_leg.astype(
            np.float32
        ),

    velocity_other=
        velocity_other.astype(
            np.float32
        ),

    finite_difference_velocity=
        finite_difference_velocity.astype(
            np.float32
        ),

    base_leg_cos=
        base_leg_cos.astype(
            np.float32
        ),

    cancellation_fraction=
        cancellation_fraction.astype(
            np.float32
        ),

    jacobian_fd_error=
        jacobian_fd_error.astype(
            np.float32
        ),
)


print()
print("=" * 205)
print("OUTPUTS")
print("=" * 205)

print(
    "CSV:",
    OUTPUT_CSV,
)

print(
    "NPZ:",
    OUTPUT_NPZ,
)

print()
print(
    "NO REFERENCE MODIFIED."
)

print(
    "NO SUPPORT MASK MODIFIED."
)

print(
    "NO OPTIMIZATION."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 205)


env.close()

