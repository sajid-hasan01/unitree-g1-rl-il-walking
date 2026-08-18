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

REFERENCE_DIR = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
)


CLIPS = {
    "medium_02":
        REFERENCE_DIR
        / "medium_02_50hz_grounded.npz",

    "medium_04":
        REFERENCE_DIR
        / "medium_04_50hz_grounded.npz",

    "medium_08":
        REFERENCE_DIR
        / "medium_08_50hz_grounded.npz",
}


# Frozen Stage-7T base candidate.
CLEARANCE_ENTER = 0.005
CLEARANCE_EXIT = 0.007

VXY_ENTER = 0.40
VXY_EXIT = 0.60

VZ_ENTER = 0.15
VZ_EXIT = 0.25


# ------------------------------------------------
# Unsupported-frame rescue candidates.
#
# NONE means no rescue.
#
# Rescue is allowed ONLY when:
# - normal detector says neither foot supports,
# - candidate is the physically lower foot,
# - candidate remains close to ground,
# - candidate velocity satisfies these bounds.
# ------------------------------------------------

RESCUE_CONFIGS = [
    ("NONE", None, None),

    ("R045_018", 0.45, 0.18),
    ("R050_020", 0.50, 0.20),
    ("R055_022", 0.55, 0.22),
    ("R060_025", 0.60, 0.25),

    # Diagnostic upper bound only.
    ("R075_030", 0.75, 0.30),
]


RESCUE_CLEARANCE = 0.007


CONTACT_DEPTH = 1e-6
MAX_ROOT_LOWERING = 0.025

TRANSITION_WINDOW = 2


EXPECTED_ROOT95 = {
    "medium_02": 5.115,
    "medium_04": 4.800,
    "medium_08": 6.387,
}


OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_support_v2_unsupported_gap_audit.csv"
)


OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_support_v2_unsupported_gap_audit.npz"
)


# ================================================================
# MODEL
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


TOTAL_MASS = float(
    mujoco.mj_getTotalmass(
        model
    )
)


BODY_WEIGHT = (
    TOTAL_MASS
    *
    abs(
        float(
            model.opt.gravity[2]
        )
    )
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


print("=" * 205)
print("G1 STAGE 7U")
print("SUPPORT V2 — UNSUPPORTED-GAP AUDIT")
print("BASE SUPPORT = STAGE-7T WINNER")
print("CONSERVATIVE LOWER-FOOT RESCUE SWEEP")
print("NO OPTIMIZATION / NO REFERENCE MODIFICATION / NO PPO")
print("=" * 205)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print()
print("BASE SUPPORT V2")

print(
    "clearance enter/exit:",
    f"{1000*CLEARANCE_ENTER:.1f}",
    "/",
    f"{1000*CLEARANCE_EXIT:.1f}",
    "mm",
)

print(
    "XY enter/exit:",
    VXY_ENTER,
    "/",
    VXY_EXIT,
    "m/s",
)

print(
    "Vz enter/exit:",
    VZ_ENTER,
    "/",
    VZ_EXIT,
    "m/s",
)

print()
print(
    "rescue configs:",
    RESCUE_CONFIGS,
)


# ================================================================
# HELPERS
# ================================================================

def differentiate_qpos(
    trajectory,
    dt,
):

    n = len(
        trajectory
    )


    velocity = np.zeros(
        (
            n,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        velocity[0],
        dt,
        trajectory[0],
        trajectory[1],
    )


    for frame in range(
        1,
        n - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            velocity[frame],
            2.0 * dt,
            trajectory[frame - 1],
            trajectory[frame + 1],
        )


    mujoco.mj_differentiatePos(
        model,
        velocity[-1],
        dt,
        trajectory[-2],
        trajectory[-1],
    )


    return velocity


def differentiate_array(
    values,
    dt,
):

    derivative = np.zeros_like(
        values
    )


    derivative[1:-1] = (
        values[2:]
        -
        values[:-2]
    ) / (
        2.0 * dt
    )


    derivative[0] = (
        values[1]
        -
        values[0]
    ) / dt


    derivative[-1] = (
        values[-1]
        -
        values[-2]
    ) / dt


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


def phase_name(
    support,
):

    if support[0] and support[1]:
        return "B"

    if support[0]:
        return "L"

    if support[1]:
        return "R"

    return "N"


def transition_mask(
    support,
):

    n = len(
        support
    )


    mask = np.zeros(
        n,
        dtype=bool,
    )


    events = []


    for frame in range(
        1,
        n,
    ):

        if not np.array_equal(
            support[frame],
            support[frame - 1],
        ):

            events.append(
                frame
            )


            lo = max(
                0,
                frame
                -
                TRANSITION_WINDOW,
            )


            hi = min(
                n,
                frame
                +
                TRANSITION_WINDOW
                +
                1,
            )


            mask[lo:hi] = True


    return mask, events


def safe_p95(
    values,
    mask,
):

    if not np.any(
        mask
    ):

        return float(
            "nan"
        )


    return float(
        np.percentile(
            values[mask],
            95,
        )
    )


def find_true_segments(
    mask,
):

    segments = []

    start = None


    for i, value in enumerate(
        mask
    ):

        if value and start is None:

            start = i


        elif (
            not value
            and
            start is not None
        ):

            segments.append(
                (
                    start,
                    i - 1,
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


# ================================================================
# SUPPORT STATE MACHINE
# ================================================================

def build_support(
    left_clearance,
    right_clearance,
    left_xy,
    right_xy,
    left_vz,
    right_vz,
    rescue_vxy=None,
    rescue_vz=None,
):

    n = len(
        left_clearance
    )


    support = np.zeros(
        (
            n,
            2,
        ),
        dtype=bool,
    )


    rescued = np.zeros(
        n,
        dtype=bool,
    )


    rescue_side = np.full(
        n,
        -1,
        dtype=np.int8,
    )


    for frame in range(n):

        # --------------------------------------------------------
        # Normal hysteretic detector.
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Rescue only if normal detector produced NONE.
        # --------------------------------------------------------

        if (
            rescue_vxy is None
            or
            rescue_vz is None
        ):

            continue


        if np.any(
            support[
                frame
            ]
        ):

            continue


        # Physically lower foot only.
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

            side = 1

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


        if (
            clearance
            <=
            RESCUE_CLEARANCE
            and
            xy
            <=
            rescue_vxy
            and
            vz
            <=
            rescue_vz
        ):

            support[
                frame,
                side
            ] = True


            rescued[
                frame
            ] = True


            rescue_side[
                frame
            ] = side


    return (
        support,
        rescued,
        rescue_side,
    )


# ================================================================
# CLIP PREPROCESSING
# ================================================================

def prepare_clip(
    label,
    path,
):

    with np.load(
        path,
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


    n = len(
        qpos
    )


    dt = (
        1.0
        /
        fps
    )


    probe = mujoco.MjData(
        model
    )


    raw_left_clearance = np.zeros(
        n,
        dtype=np.float64,
    )


    raw_right_clearance = np.zeros(
        n,
        dtype=np.float64,
    )


    # ------------------------------------------------------------
    # Raw clearance.
    # ------------------------------------------------------------

    for frame in range(n):

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


    # ------------------------------------------------------------
    # Global-lowest projection.
    # ------------------------------------------------------------

    projected_qpos = (
        qpos.copy()
    )


    for frame in range(n):

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


    projected_qvel = differentiate_qpos(
        projected_qpos,
        dt,
    )


    projected_qacc = differentiate_array(
        projected_qvel,
        dt,
    )


    # ------------------------------------------------------------
    # Projected geometry.
    # ------------------------------------------------------------

    left_clearance = np.zeros(
        n,
        dtype=np.float64,
    )


    right_clearance = np.zeros(
        n,
        dtype=np.float64,
    )


    left_position = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float64,
    )


    right_position = np.zeros_like(
        left_position
    )


    for frame in range(n):

        probe.qpos[:] = (
            projected_qpos[
                frame
            ]
        )


        probe.qvel[:] = (
            projected_qvel[
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


    left_velocity = differentiate_array(
        left_position,
        dt,
    )


    right_velocity = differentiate_array(
        right_position,
        dt,
    )


    left_xy = np.linalg.norm(
        left_velocity[
            :,
            0:2
        ],
        axis=1,
    )


    right_xy = np.linalg.norm(
        right_velocity[
            :,
            0:2
        ],
        axis=1,
    )


    left_vz = (
        left_velocity[
            :,
            2
        ]
    )


    right_vz = (
        right_velocity[
            :,
            2
        ]
    )


    # ------------------------------------------------------------
    # Inverse dynamics is computed ONCE.
    # Rescue parameters do not affect qpos/qvel/qacc.
    # ------------------------------------------------------------

    inverse_data = mujoco.MjData(
        model
    )


    root_bw = np.zeros(
        n,
        dtype=np.float64,
    )


    max_tau = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        inverse_data.qpos[:] = (
            projected_qpos[
                frame
            ]
        )


        inverse_data.qvel[:] = (
            projected_qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            inverse_data,
        )


        inverse_data.qacc[:] = (
            projected_qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            inverse_data,
        )


        root_bw[
            frame
        ] = (
            np.linalg.norm(
                inverse_data.qfrc_inverse[
                    0:3
                ]
            )
            /
            BODY_WEIGHT
        )


        tau = np.asarray(
            [
                inverse_data.qfrc_inverse[
                    vadr
                ]
                for vadr in env.vaddrs
            ],
            dtype=np.float64,
        )


        max_tau[
            frame
        ] = float(
            np.max(
                np.abs(
                    tau
                )
                /
                effort
            )
        )


    root95 = float(
        np.percentile(
            root_bw,
            95,
        )
    )


    if abs(
        root95
        -
        EXPECTED_ROOT95[
            label
        ]
    ) > 0.75:

        raise RuntimeError(
            f"{label}: corrected root baseline changed "
            f"({root95:.3f})"
        )


    return {

        "label":
            label,

        "n":
            n,

        "fps":
            fps,

        "left_clearance":
            left_clearance,

        "right_clearance":
            right_clearance,

        "left_xy":
            left_xy,

        "right_xy":
            right_xy,

        "left_vz":
            left_vz,

        "right_vz":
            right_vz,

        "root_bw":
            root_bw,

        "max_tau":
            max_tau,
    }


# ================================================================
# PREPARE ALL THREE
# ================================================================

clips = {}


for label, path in (
    CLIPS.items()
):

    print()
    print("=" * 205)
    print("PREPARE:", label)
    print("=" * 205)


    clips[
        label
    ] = prepare_clip(
        label,
        path,
    )


    print(
        "corrected root p95:",
        f"{np.percentile(clips[label]['root_bw'],95):.3f}",
    )


# ================================================================
# BASE NONE-GAP MICROSCOPE
# ================================================================

print()
print("=" * 205)
print("BASE STAGE-7T UNSUPPORTED GAP MICROSCOPE")
print("=" * 205)


for label in (
    "medium_02",
    "medium_04",
    "medium_08",
):

    clip = clips[
        label
    ]


    (
        support,
        rescued,
        rescue_side,
    ) = build_support(
        clip[
            "left_clearance"
        ],
        clip[
            "right_clearance"
        ],
        clip[
            "left_xy"
        ],
        clip[
            "right_xy"
        ],
        clip[
            "left_vz"
        ],
        clip[
            "right_vz"
        ],
        None,
        None,
    )


    none = (
        np.sum(
            support,
            axis=1,
        )
        ==
        0
    )


    segments = find_true_segments(
        none
    )


    print()
    print("-" * 205)

    print(
        label,
        "NONE frames:",
        int(
            np.sum(
                none
            )
        ),
        "/",
        clip[
            "n"
        ],
        f"({100*np.mean(none):.1f}%)",
    )


    print(
        "NONE segments:",
        len(
            segments
        ),
    )


    lengths = np.asarray(
        [
            end
            -
            start
            +
            1

            for start, end in (
                segments
            )
        ],
        dtype=np.int32,
    )


    if len(
        lengths
    ):

        print(
            "gap length frames:",
            f"median={np.median(lengths):.1f}",
            f"p95={np.percentile(lengths,95):.1f}",
            f"max={np.max(lengths)}",
        )

        print(
            "gap duration max:",
            f"{1000*np.max(lengths)/clip['fps']:.1f} ms",
        )


    lower_xy = np.zeros(
        clip[
            "n"
        ],
        dtype=np.float64,
    )


    lower_vz = np.zeros(
        clip[
            "n"
        ],
        dtype=np.float64,
    )


    lower_clearance = np.zeros(
        clip[
            "n"
        ],
        dtype=np.float64,
    )


    lower_side = np.zeros(
        clip[
            "n"
        ],
        dtype=np.int8,
    )


    for frame in range(
        clip[
            "n"
        ]
    ):

        if (
            clip[
                "left_clearance"
            ][
                frame
            ]
            <=
            clip[
                "right_clearance"
            ][
                frame
            ]
        ):

            lower_side[
                frame
            ] = 0


            lower_clearance[
                frame
            ] = (
                clip[
                    "left_clearance"
                ][
                    frame
                ]
            )


            lower_xy[
                frame
            ] = (
                clip[
                    "left_xy"
                ][
                    frame
                ]
            )


            lower_vz[
                frame
            ] = abs(
                clip[
                    "left_vz"
                ][
                    frame
                ]
            )


        else:

            lower_side[
                frame
            ] = 1


            lower_clearance[
                frame
            ] = (
                clip[
                    "right_clearance"
                ][
                    frame
                ]
            )


            lower_xy[
                frame
            ] = (
                clip[
                    "right_xy"
                ][
                    frame
                ]
            )


            lower_vz[
                frame
            ] = abs(
                clip[
                    "right_vz"
                ][
                    frame
                ]
            )


    if np.any(
        none
    ):

        print()
        print(
            "LOWER FOOT DURING NONE FRAMES"
        )


        print(
            "clearance p50/p95/max [mm]:",
            f"{1000*np.percentile(lower_clearance[none],50):.3f}",
            f"{1000*np.percentile(lower_clearance[none],95):.3f}",
            f"{1000*np.max(lower_clearance[none]):.3f}",
        )


        print(
            "XY speed p50/p95/max [m/s]:",
            f"{np.percentile(lower_xy[none],50):.3f}",
            f"{np.percentile(lower_xy[none],95):.3f}",
            f"{np.max(lower_xy[none]):.3f}",
        )


        print(
            "|Vz| p50/p95/max [m/s]:",
            f"{np.percentile(lower_vz[none],50):.3f}",
            f"{np.percentile(lower_vz[none],95):.3f}",
            f"{np.max(lower_vz[none]):.3f}",
        )


        xy_fail = (
            lower_xy
            >
            VXY_ENTER
        )


        vz_fail = (
            lower_vz
            >
            VZ_ENTER
        )


        print(
            "NONE reason fractions:"
        )


        print(
            "  lower foot XY > enter:",
            f"{100*np.mean(xy_fail[none]):.1f}%",
        )


        print(
            "  lower foot |Vz| > enter:",
            f"{100*np.mean(vz_fail[none]):.1f}%",
        )


        print(
            "  BOTH velocity limits fail:",
            f"{100*np.mean((xy_fail & vz_fail)[none]):.1f}%",
        )


        print(
            "  exceeds even EXIT XY limit:",
            f"{100*np.mean((lower_xy > VXY_EXIT)[none]):.1f}%",
        )


        print(
            "  exceeds even EXIT Vz limit:",
            f"{100*np.mean((lower_vz > VZ_EXIT)[none]):.1f}%",
        )


    print()
    print(
        "PER GAP"
    )


    for gap_index, (
        start,
        end,
    ) in enumerate(
        segments
    ):

        frames = np.arange(
            start,
            end + 1,
        )


        print(
            f"{gap_index:02d}. "
            f"{start:03d}..{end:03d} "
            f"len={len(frames):2d} "
            f"dur={1000*len(frames)/clip['fps']:5.1f}ms "
            f"lowerClrMax={1000*np.max(lower_clearance[frames]):5.2f}mm "
            f"lowerXY p50/p95="
            f"{np.percentile(lower_xy[frames],50):.3f}/"
            f"{np.percentile(lower_xy[frames],95):.3f} "
            f"lowerVz p50/p95="
            f"{np.percentile(lower_vz[frames],50):.3f}/"
            f"{np.percentile(lower_vz[frames],95):.3f} "
            f"rootBW p50/p95="
            f"{np.percentile(clip['root_bw'][frames],50):.2f}/"
            f"{np.percentile(clip['root_bw'][frames],95):.2f}"
        )


# ================================================================
# RESCUE SWEEP
# ================================================================

all_rows = []


print()
print("=" * 205)
print("CONSERVATIVE RESCUE SWEEP")
print("=" * 205)


for (
    rescue_name,
    rescue_vxy,
    rescue_vz,
) in RESCUE_CONFIGS:

    print()
    print(
        "---",
        rescue_name,
        "---",
    )


    config_rows = []


    for label in (
        "medium_02",
        "medium_04",
        "medium_08",
    ):

        clip = clips[
            label
        ]


        (
            support,
            rescued,
            rescue_side,
        ) = build_support(
            clip[
                "left_clearance"
            ],
            clip[
                "right_clearance"
            ],
            clip[
                "left_xy"
            ],
            clip[
                "right_xy"
            ],
            clip[
                "left_vz"
            ],
            clip[
                "right_vz"
            ],
            rescue_vxy,
            rescue_vz,
        )


        count = np.sum(
            support,
            axis=1,
        )


        none = (
            count == 0
        )


        single = (
            count == 1
        )


        double = (
            count == 2
        )


        (
            transition,
            events,
        ) = transition_mask(
            support
        )


        steady_single = (
            single
            &
            ~transition
        )


        steady_double = (
            double
            &
            ~transition
        )


        active_xy = []


        active_vz = []


        for side in (
            0,
            1,
        ):

            active = (
                support[
                    :,
                    side
                ]
            )


            if not np.any(
                active
            ):

                continue


            if side == 0:

                active_xy.extend(
                    clip[
                        "left_xy"
                    ][
                        active
                    ].tolist()
                )


                active_vz.extend(
                    np.abs(
                        clip[
                            "left_vz"
                        ][
                            active
                        ]
                    ).tolist()
                )


            else:

                active_xy.extend(
                    clip[
                        "right_xy"
                    ][
                        active
                    ].tolist()
                )


                active_vz.extend(
                    np.abs(
                        clip[
                            "right_vz"
                        ][
                            active
                        ]
                    ).tolist()
                )


        active_xy = np.asarray(
            active_xy,
            dtype=np.float64,
        )


        active_vz = np.asarray(
            active_vz,
            dtype=np.float64,
        )


        gap_segments = find_true_segments(
            none
        )


        max_gap = (
            max(
                [
                    end
                    -
                    start
                    +
                    1

                    for start, end in gap_segments
                ]
            )

            if gap_segments
            else 0
        )


        row = {

            "rescue":
                rescue_name,

            "rescue_vxy":
                (
                    np.nan
                    if rescue_vxy is None
                    else rescue_vxy
                ),

            "rescue_vz":
                (
                    np.nan
                    if rescue_vz is None
                    else rescue_vz
                ),

            "clip":
                label,

            "none_fraction":
                float(
                    np.mean(
                        none
                    )
                ),

            "single_fraction":
                float(
                    np.mean(
                        single
                    )
                ),

            "double_fraction":
                float(
                    np.mean(
                        double
                    )
                ),

            "rescued_fraction":
                float(
                    np.mean(
                        rescued
                    )
                ),

            "events":
                len(
                    events
                ),

            "max_none_gap_frames":
                max_gap,

            "active_xy_p50":
                float(
                    np.percentile(
                        active_xy,
                        50,
                    )
                ),

            "active_xy_p95":
                float(
                    np.percentile(
                        active_xy,
                        95,
                    )
                ),

            "active_vz_p95":
                float(
                    np.percentile(
                        active_vz,
                        95,
                    )
                ),

            # Diagnostics only.
            "single_root_p95":
                safe_p95(
                    clip[
                        "root_bw"
                    ],
                    steady_single,
                ),

            "double_root_p95":
                safe_p95(
                    clip[
                        "root_bw"
                    ],
                    steady_double,
                ),

            "transition_root_p95":
                safe_p95(
                    clip[
                        "root_bw"
                    ],
                    transition,
                ),

            "tau_p95":
                float(
                    np.percentile(
                        clip[
                            "max_tau"
                        ],
                        95,
                    )
                ),
        }


        config_rows.append(
            row
        )


        all_rows.append(
            row
        )


        print(
            f"{label:10s} "
            f"NONE={100*row['none_fraction']:5.1f}% "
            f"SINGLE={100*row['single_fraction']:5.1f}% "
            f"DOUBLE={100*row['double_fraction']:5.1f}% "
            f"rescued={100*row['rescued_fraction']:5.1f}% "
            f"events={row['events']:2d} "
            f"maxGap={row['max_none_gap_frames']:2d} "
            f"activeXY95={row['active_xy_p95']:.3f} "
            f"activeVz95={row['active_vz_p95']:.3f} "
            f"S95={row['single_root_p95']:.3f} "
            f"D95={row['double_root_p95']:.3f} "
            f"T95={row['transition_root_p95']:.3f}"
        )


# ================================================================
# CROSS-CLIP PHYSICAL ACCEPTANCE
# ================================================================

print()
print("=" * 205)
print("CROSS-CLIP PHYSICAL ACCEPTANCE")
print("NO INVERSE-DYNAMICS METRIC IS USED FOR ACCEPTANCE")
print("=" * 205)


accepted = []


for (
    rescue_name,
    rescue_vxy,
    rescue_vz,
) in RESCUE_CONFIGS:

    rows = [
        row
        for row in all_rows
        if row[
            "rescue"
        ]
        ==
        rescue_name
    ]


    none_max = max(
        row[
            "none_fraction"
        ]
        for row in rows
    )


    active_xy_max = max(
        row[
            "active_xy_p95"
        ]
        for row in rows
    )


    active_vz_max = max(
        row[
            "active_vz_p95"
        ]
        for row in rows
    )


    double_min = min(
        row[
            "double_fraction"
        ]
        for row in rows
    )


    double_max = max(
        row[
            "double_fraction"
        ]
        for row in rows
    )


    max_gap = max(
        row[
            "max_none_gap_frames"
        ]
        for row in rows
    )


    event_max = max(
        row[
            "events"
        ]
        for row in rows
    )


    # ------------------------------------------------------------
    # Conservative acceptance gate.
    # ------------------------------------------------------------

    passes = (
        none_max
        <=
        0.02

        and
        active_xy_max
        <=
        0.30

        and
        active_vz_max
        <=
        0.20

        and
        double_min
        >=
        0.15

        and
        double_max
        <=
        0.45

        and
        event_max
        <=
        20
    )


    print(
        f"{rescue_name:10s} "
        f"noneMax={100*none_max:5.1f}% "
        f"activeXY95max={active_xy_max:.3f} "
        f"activeVz95max={active_vz_max:.3f} "
        f"double={100*double_min:4.1f}-"
        f"{100*double_max:4.1f}% "
        f"eventMax={event_max:2d} "
        f"noneGapMax={max_gap:2d} "
        f"PASS={passes}"
    )


    if passes:

        accepted.append(
            rescue_name
        )


# ================================================================
# FINAL DIAGNOSIS
# ================================================================

print()
print("=" * 205)
print("STAGE 7U DECISION")
print("=" * 205)


if accepted:

    print(
        "PHYSICALLY ACCEPTABLE SUPPORT-V2 "
        "RESCUE CONFIG(S):"
    )


    for value in accepted:

        print(
            " ",
            value,
        )


    # Conservative: use the first passing config,
    # because RESCUE_CONFIGS are ordered from least
    # permissive to most permissive.
    winner = accepted[
        0
    ]


    print()
    print(
        "CONSERVATIVE WINNER:",
        winner,
    )

    print()
    print(
        "RESULT:"
    )

    print(
        "A modest lower-foot rescue can eliminate "
        "unsupported gaps without accepting fast "
        "stance motion."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Freeze Support V2 with the base Stage-7T "
        "state machine plus this exact rescue rule."
    )

    print(
        "Then rerun one authoritative dynamic audit "
        "with the frozen Support-V2 timeline."
    )


else:

    print(
        "NO RESCUE CONFIG PASSED THE PHYSICAL GATE."
    )

    print()
    print(
        "RESULT:"
    )

    print(
        "The unsupported intervals are NOT merely "
        "detector dropouts."
    )

    print(
        "At those frames even the physically lowest "
        "foot is moving too quickly to represent "
        "credible stance."
    )

    print()
    print(
        "THIS IS DIRECT EVIDENCE OF RETARGETED "
        "FOOT/ROOT KINEMATIC INCONSISTENCY."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Do not weaken the support detector."
    )

    print(
        "Inspect/repair the reference foot/root motion "
        "during unsupported intervals, starting with "
        "medium_04."
    )


# ================================================================
# SAVE CSV
# ================================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            all_rows[
                0
            ].keys()
        ),
    )


    writer.writeheader()


    for row in all_rows:

        writer.writerow(
            row
        )


# ================================================================
# SAVE NPZ
# ================================================================

payload = {}


for label, clip in (
    clips.items()
):

    payload[
        f"{label}_left_clearance"
    ] = (
        clip[
            "left_clearance"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_clearance"
    ] = (
        clip[
            "right_clearance"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_left_xy"
    ] = (
        clip[
            "left_xy"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_xy"
    ] = (
        clip[
            "right_xy"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_left_vz"
    ] = (
        clip[
            "left_vz"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_vz"
    ] = (
        clip[
            "right_vz"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_root_bw"
    ] = (
        clip[
            "root_bw"
        ].astype(
            np.float32
        )
    )


payload[
    "accepted_rescues"
] = np.asarray(
    accepted,
    dtype="<U16",
)


np.savez_compressed(
    OUTPUT_NPZ,
    **payload,
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
    "NO SUPPORT MASK COMMITTED."
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
