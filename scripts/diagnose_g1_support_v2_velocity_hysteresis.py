from pathlib import Path
import csv
import itertools
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


CLEARANCE_ENTER_MM = [
    2.0,
    3.0,
    5.0,
]


VXY_ENTER = [
    0.15,
    0.25,
    0.40,
]


VZ_ENTER = [
    0.05,
    0.10,
    0.15,
]


# Hysteresis:
CLEARANCE_EXIT_EXTRA_MM = 2.0
VXY_EXIT_EXTRA = 0.20
VZ_EXIT_EXTRA = 0.10


CONTACT_DEPTH = 1e-6

MAX_ROOT_LOWERING = 0.025

TRANSITION_WINDOW = 2


EXPECTED_ROOT95 = {
    "medium_02":
        5.115,

    "medium_04":
        4.800,

    "medium_08":
        6.387,
}


OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_support_v2_velocity_hysteresis_sweep.csv"
)


OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_support_v2_velocity_hysteresis_sweep.npz"
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
            model.opt.gravity[
                2
            ]
        )
    )
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


print("=" * 205)
print("G1 STAGE 7T")
print("SUPPORT V2 — GEOMETRY + VELOCITY + HYSTERESIS")
print("27 SMALL PHYSICALLY-INTERPRETABLE CONFIGURATIONS")
print("NO OPTIMIZATION / NO REFERENCE MODIFICATION / NO PPO")
print("=" * 205)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "clearance enter [mm]:",
    CLEARANCE_ENTER_MM,
)

print(
    "XY-speed enter [m/s]:",
    VXY_ENTER,
)

print(
    "vertical-speed enter [m/s]:",
    VZ_ENTER,
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
        velocity[
            0
        ],
        dt,
        trajectory[
            0
        ],
        trajectory[
            1
        ],
    )


    for frame in range(
        1,
        n - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            velocity[
                frame
            ],
            2.0 * dt,
            trajectory[
                frame - 1
            ],
            trajectory[
                frame + 1
            ],
        )


    mujoco.mj_differentiatePos(
        model,
        velocity[
            -1
        ],
        dt,
        trajectory[
            -2
        ],
        trajectory[
            -1
        ],
    )


    return velocity


def differentiate_array(
    values,
    dt,
):

    derivative = np.zeros_like(
        values
    )


    derivative[
        1:-1
    ] = (
        values[
            2:
        ]
        -
        values[
            :-2
        ]
    ) / (
        2.0 * dt
    )


    derivative[
        0
    ] = (
        values[
            1
        ]
        -
        values[
            0
        ]
    ) / dt


    derivative[
        -1
    ] = (
        values[
            -1
        ]
        -
        values[
            -2
        ]
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


    minimum = float(
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


        minimum = min(
            minimum,
            bottom
            -
            floor_z,
        )


    return minimum


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
    mask,
):

    left = bool(
        mask[
            0
        ]
    )

    right = bool(
        mask[
            1
        ]
    )


    if left and right:
        return "B"

    if left:
        return "L"

    if right:
        return "R"

    return "N"


def make_transition_mask(
    support,
):

    n = len(
        support
    )


    result = np.zeros(
        n,
        dtype=bool,
    )


    events = []


    for frame in range(
        1,
        n,
    ):

        if not np.array_equal(
            support[
                frame
            ],
            support[
                frame - 1
            ],
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


            result[
                lo:hi
            ] = True


    return (
        result,
        events,
    )


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
            values[
                mask
            ],
            95,
        )
    )


def build_support_state_machine(
    left_clearance,
    right_clearance,
    left_xy,
    right_xy,
    left_vz,
    right_vz,
    clearance_enter,
    vxy_enter,
    vz_enter,
):

    n = len(
        left_clearance
    )


    clearance_exit = (
        clearance_enter
        +
        CLEARANCE_EXIT_EXTRA_MM
        /
        1000.0
    )


    vxy_exit = (
        vxy_enter
        +
        VXY_EXIT_EXTRA
    )


    vz_exit = (
        vz_enter
        +
        VZ_EXIT_EXTRA
    )


    support = np.zeros(
        (
            n,
            2,
        ),
        dtype=bool,
    )


    for frame in range(n):

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

                xy_speed = (
                    left_xy[
                        frame
                    ]
                )

                vertical_speed = abs(
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

                xy_speed = (
                    right_xy[
                        frame
                    ]
                )

                vertical_speed = abs(
                    right_vz[
                        frame
                    ]
                )


            previous = (
                frame
                >
                0
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
                    clearance_exit
                    and
                    xy_speed
                    <=
                    vxy_exit
                    and
                    vertical_speed
                    <=
                    vz_exit
                )


            else:

                active = (
                    clearance
                    <=
                    clearance_enter
                    and
                    xy_speed
                    <=
                    vxy_enter
                    and
                    vertical_speed
                    <=
                    vz_enter
                )


            support[
                frame,
                side
            ] = active


    return support


# ================================================================
# PROCESS ONE CLIP
# ================================================================

def process_clip(
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


        old_support = (
            np.asarray(
                f[
                    "support_mask"
                ],
                dtype=np.float64,
            )
            >
            0.5
        )


        old_contact = (
            np.asarray(
                f[
                    "contact_mask"
                ],
                dtype=np.float64,
            )
            >
            0.5
        )


        fps = (
            float(
                np.asarray(
                    f[
                        "fps"
                    ]
                ).reshape(
                    -1
                )[
                    0
                ]
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


    print()
    print("=" * 205)
    print("CLIP:", label)
    print("=" * 205)


    # ------------------------------------------------------------
    # RAW CLEARANCE
    # ------------------------------------------------------------

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
    # GLOBAL-LOWEST FIRST-CONTACT PROJECTION
    # ------------------------------------------------------------

    projected_qpos = (
        qpos.copy()
    )


    root_shift = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        minimum_clearance = min(
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
            minimum_clearance
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


    projected_qvel = differentiate_qpos(
        projected_qpos,
        dt,
    )


    projected_qacc = differentiate_array(
        projected_qvel,
        dt,
    )


    # ------------------------------------------------------------
    # PROJECTED SOLE GEOMETRY
    # ------------------------------------------------------------

    projected_left_clearance = np.zeros(
        n,
        dtype=np.float64,
    )


    projected_right_clearance = np.zeros(
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


        projected_left_clearance[
            frame
        ] = sole_clearance(
            probe,
            LEFT_GEOMS,
        )


        projected_right_clearance[
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
    # GLOBAL-LOWEST CONSISTENT INVERSE DYNAMICS
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


    print(
        "corrected root p95:",
        f"{root95:.3f}",
    )


    if abs(
        root95
        -
        EXPECTED_ROOT95[
            label
        ]
    ) > 0.75:

        raise RuntimeError(
            f"{label}: corrected baseline changed."
        )


    # ------------------------------------------------------------
    # SWEEP
    # ------------------------------------------------------------

    rows = []


    for (
        clearance_mm,
        vxy_enter,
        vz_enter,
    ) in itertools.product(
        CLEARANCE_ENTER_MM,
        VXY_ENTER,
        VZ_ENTER,
    ):

        clearance_enter = (
            clearance_mm
            /
            1000.0
        )


        support = build_support_state_machine(
            projected_left_clearance,
            projected_right_clearance,
            left_xy,
            right_xy,
            left_vz,
            right_vz,
            clearance_enter,
            vxy_enter,
            vz_enter,
        )


        count = np.sum(
            support,
            axis=1,
        )


        no_support = (
            count == 0
        )


        single = (
            count == 1
        )


        double = (
            count == 2
        )


        left_only = (
            support[
                :,
                0
            ]
            &
            ~support[
                :,
                1
            ]
        )


        right_only = (
            support[
                :,
                1
            ]
            &
            ~support[
                :,
                0
            ]
        )


        (
            transition,
            events,
        ) = make_transition_mask(
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


        # --------------------------------------------------------
        # Active-support velocities
        # --------------------------------------------------------

        stance_xy_values = []


        stance_vz_values = []


        if np.any(
            support[
                :,
                0
            ]
        ):

            stance_xy_values.extend(
                left_xy[
                    support[
                        :,
                        0
                    ]
                ].tolist()
            )


            stance_vz_values.extend(
                np.abs(
                    left_vz[
                        support[
                            :,
                            0
                        ]
                    ]
                ).tolist()
            )


        if np.any(
            support[
                :,
                1
            ]
        ):

            stance_xy_values.extend(
                right_xy[
                    support[
                        :,
                        1
                    ]
                ].tolist()
            )


            stance_vz_values.extend(
                np.abs(
                    right_vz[
                        support[
                            :,
                            1
                        ]
                    ]
                ).tolist()
            )


        stance_xy_values = np.asarray(
            stance_xy_values,
            dtype=np.float64,
        )


        stance_vz_values = np.asarray(
            stance_vz_values,
            dtype=np.float64,
        )


        # --------------------------------------------------------
        # Swing velocities during genuine single support
        # --------------------------------------------------------

        swing_xy_values = np.concatenate(
            [
                right_xy[
                    left_only
                ],
                left_xy[
                    right_only
                ],
            ]
        )


        if len(
            stance_xy_values
        ):

            stance_xy_p50 = float(
                np.percentile(
                    stance_xy_values,
                    50,
                )
            )


            stance_xy_p95 = float(
                np.percentile(
                    stance_xy_values,
                    95,
                )
            )


            stance_vz_p95 = float(
                np.percentile(
                    stance_vz_values,
                    95,
                )
            )


        else:

            stance_xy_p50 = float(
                "nan"
            )

            stance_xy_p95 = float(
                "nan"
            )

            stance_vz_p95 = float(
                "nan"
            )


        if len(
            swing_xy_values
        ):

            swing_xy_p50 = float(
                np.percentile(
                    swing_xy_values,
                    50,
                )
            )

        else:

            swing_xy_p50 = float(
                "nan"
            )


        separation = (
            swing_xy_p50
            /
            max(
                stance_xy_p50,
                1e-9,
            )
            if (
                np.isfinite(
                    swing_xy_p50
                )
                and
                np.isfinite(
                    stance_xy_p50
                )
            )
            else float(
                "nan"
            )
        )


        support_agreement = float(
            np.mean(
                np.all(
                    support
                    ==
                    old_support,
                    axis=1,
                )
            )
        )


        contact_agreement = float(
            np.mean(
                np.all(
                    support
                    ==
                    old_contact,
                    axis=1,
                )
            )
        )


        row = {

            "clip":
                label,

            "clearance_enter_mm":
                clearance_mm,

            "clearance_exit_mm":
                clearance_mm
                +
                CLEARANCE_EXIT_EXTRA_MM,

            "vxy_enter":
                vxy_enter,

            "vxy_exit":
                vxy_enter
                +
                VXY_EXIT_EXTRA,

            "vz_enter":
                vz_enter,

            "vz_exit":
                vz_enter
                +
                VZ_EXIT_EXTRA,

            "none_fraction":
                float(
                    np.mean(
                        no_support
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

            "events":
                len(
                    events
                ),

            "transition_fraction":
                float(
                    np.mean(
                        transition
                    )
                ),

            "support_agreement":
                support_agreement,

            "contact_agreement":
                contact_agreement,

            "stance_xy_p50":
                stance_xy_p50,

            "stance_xy_p95":
                stance_xy_p95,

            "stance_vz_p95":
                stance_vz_p95,

            "swing_xy_p50":
                swing_xy_p50,

            "speed_separation":
                separation,

            "single_root_p95":
                safe_p95(
                    root_bw,
                    steady_single,
                ),

            "double_root_p95":
                safe_p95(
                    root_bw,
                    steady_double,
                ),

            "transition_root_p95":
                safe_p95(
                    root_bw,
                    transition,
                ),

            "tau_p95":
                float(
                    np.percentile(
                        max_tau,
                        95,
                    )
                ),
        }


        rows.append(
            row
        )


    return {

        "rows":
            rows,

        "left_clearance":
            projected_left_clearance,

        "right_clearance":
            projected_right_clearance,

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
    }


# ================================================================
# RUN THREE CLIPS
# ================================================================

results = {}


for label, path in CLIPS.items():

    results[
        label
    ] = process_clip(
        label,
        path,
    )


# ================================================================
# CROSS-CLIP CONFIG ANALYSIS
# ================================================================

config_keys = []


for (
    clearance_mm,
    vxy_enter,
    vz_enter,
) in itertools.product(
    CLEARANCE_ENTER_MM,
    VXY_ENTER,
    VZ_ENTER,
):

    config_keys.append(
        (
            clearance_mm,
            vxy_enter,
            vz_enter,
        )
    )


cross_rows = []


for config in config_keys:

    (
        clearance_mm,
        vxy_enter,
        vz_enter,
    ) = config


    clip_rows = []


    for label in (
        "medium_02",
        "medium_04",
        "medium_08",
    ):

        row = next(
            item
            for item in results[
                label
            ][
                "rows"
            ]
            if (
                item[
                    "clearance_enter_mm"
                ]
                ==
                clearance_mm
                and
                item[
                    "vxy_enter"
                ]
                ==
                vxy_enter
                and
                item[
                    "vz_enter"
                ]
                ==
                vz_enter
            )
        )


        clip_rows.append(
            row
        )


    none_values = np.asarray(
        [
            r[
                "none_fraction"
            ]
            for r in clip_rows
        ]
    )


    double_values = np.asarray(
        [
            r[
                "double_fraction"
            ]
            for r in clip_rows
        ]
    )


    event_values = np.asarray(
        [
            r[
                "events"
            ]
            for r in clip_rows
        ]
    )


    stance_xy_values = np.asarray(
        [
            r[
                "stance_xy_p95"
            ]
            for r in clip_rows
        ]
    )


    stance_vz_values = np.asarray(
        [
            r[
                "stance_vz_p95"
            ]
            for r in clip_rows
        ]
    )


    separation_values = np.asarray(
        [
            r[
                "speed_separation"
            ]
            for r in clip_rows
        ]
    )


    # ------------------------------------------------------------
    # Physical-quality score.
    #
    # IMPORTANT:
    # No inverse-dynamics residual enters this score.
    #
    # We are selecting SUPPORT SEMANTICS,
    # not gaming the residual.
    # ------------------------------------------------------------

    penalty = 0.0


    # No-support frames are undesirable in normal walking.
    penalty += (
        10.0
        *
        float(
            np.mean(
                none_values
            )
        )
    )


    # Very little or almost continuous double support
    # is suspicious for these normal walking clips.
    for value in double_values:

        if value < 0.05:

            penalty += (
                2.0
                *
                (
                    0.05
                    -
                    value
                )
            )


        if value > 0.45:

            penalty += (
                2.0
                *
                (
                    value
                    -
                    0.45
                )
            )


    # Excessive state chatter is undesirable.
    for value in event_values:

        if value > 25:

            penalty += (
                0.02
                *
                (
                    value
                    -
                    25
                )
            )


    # Low active-support velocity is desirable.
    penalty += float(
        np.mean(
            stance_xy_values
        )
    )


    penalty += (
        1.5
        *
        float(
            np.mean(
                stance_vz_values
            )
        )
    )


    # Reward separation between swing and stance speed.
    separation_reward = float(
        np.mean(
            np.clip(
                separation_values,
                0.0,
                20.0,
            )
        )
    )


    score = (
        penalty
        -
        0.02
        *
        separation_reward
    )


    cross_rows.append(
        {

            "clearance_enter_mm":
                clearance_mm,

            "vxy_enter":
                vxy_enter,

            "vz_enter":
                vz_enter,

            "score":
                score,

            "none_mean":
                float(
                    np.mean(
                        none_values
                    )
                ),

            "none_max":
                float(
                    np.max(
                        none_values
                    )
                ),

            "double_mean":
                float(
                    np.mean(
                        double_values
                    )
                ),

            "double_min":
                float(
                    np.min(
                        double_values
                    )
                ),

            "double_max":
                float(
                    np.max(
                        double_values
                    )
                ),

            "events_mean":
                float(
                    np.mean(
                        event_values
                    )
                ),

            "stance_xy_p95_mean":
                float(
                    np.mean(
                        stance_xy_values
                    )
                ),

            "stance_vz_p95_mean":
                float(
                    np.mean(
                        stance_vz_values
                    )
                ),

            "separation_mean":
                float(
                    np.mean(
                        separation_values
                    )
                ),
        }
    )


cross_rows.sort(
    key=lambda row:
        row[
            "score"
        ]
)


# ================================================================
# TOP PHYSICAL CONFIGURATIONS
# ================================================================

print()
print("=" * 205)
print("TOP SUPPORT-V2 CONFIGURATIONS")
print("RANKED WITHOUT USING INVERSE-DYNAMICS RESIDUAL")
print("=" * 205)


print(
    f"{'RANK':>4s} "
    f"{'CLR':>5s} "
    f"{'VXY':>5s} "
    f"{'VZ':>5s} "
    f"{'SCORE':>8s} "
    f"{'NONEmean':>9s} "
    f"{'NONEmax':>8s} "
    f"{'DBLmean':>8s} "
    f"{'DBLrange':>15s} "
    f"{'EVENT':>7s} "
    f"{'STXY95':>8s} "
    f"{'STVZ95':>8s} "
    f"{'SEP':>7s}"
)


print("-" * 205)


for rank, row in enumerate(
    cross_rows[
        :10
    ],
    start=1,
):

    print(
        f"{rank:4d} "
        f"{row['clearance_enter_mm']:5.1f} "
        f"{row['vxy_enter']:5.2f} "
        f"{row['vz_enter']:5.2f} "
        f"{row['score']:8.4f} "
        f"{100*row['none_mean']:8.1f}% "
        f"{100*row['none_max']:7.1f}% "
        f"{100*row['double_mean']:7.1f}% "
        f"{100*row['double_min']:6.1f}-"
        f"{100*row['double_max']:6.1f}% "
        f"{row['events_mean']:7.1f} "
        f"{row['stance_xy_p95_mean']:8.3f} "
        f"{row['stance_vz_p95_mean']:8.3f} "
        f"{row['separation_mean']:7.2f}"
    )


# ================================================================
# SHOW BEST CONFIG PER CLIP
# ================================================================

best = cross_rows[
    0
]


best_clearance = (
    best[
        "clearance_enter_mm"
    ]
)

best_vxy = (
    best[
        "vxy_enter"
    ]
)

best_vz = (
    best[
        "vz_enter"
    ]
)


print()
print("=" * 205)
print("BEST DIAGNOSTIC CONFIG — PER CLIP")
print("NOT COMMITTED YET")
print("=" * 205)

print(
    "clearance enter:",
    f"{best_clearance:.1f} mm",
)

print(
    "clearance exit:",
    f"{best_clearance + CLEARANCE_EXIT_EXTRA_MM:.1f} mm",
)

print(
    "XY enter/exit:",
    f"{best_vxy:.2f}",
    "/",
    f"{best_vxy + VXY_EXIT_EXTRA:.2f}",
    "m/s",
)

print(
    "Vz enter/exit:",
    f"{best_vz:.2f}",
    "/",
    f"{best_vz + VZ_EXIT_EXTRA:.2f}",
    "m/s",
)


print()
print(
    f"{'CLIP':10s} "
    f"{'NONE':>7s} "
    f"{'SINGLE':>8s} "
    f"{'DOUBLE':>8s} "
    f"{'EVENTS':>7s} "
    f"{'STXY95':>8s} "
    f"{'STVZ95':>8s} "
    f"{'SWXY50':>8s} "
    f"{'SEP':>7s} "
    f"{'S95':>8s} "
    f"{'D95':>8s} "
    f"{'T95':>8s}"
)


for label in (
    "medium_02",
    "medium_04",
    "medium_08",
):

    row = next(
        item
        for item in results[
            label
        ][
            "rows"
        ]
        if (
            item[
                "clearance_enter_mm"
            ]
            ==
            best_clearance
            and
            item[
                "vxy_enter"
            ]
            ==
            best_vxy
            and
            item[
                "vz_enter"
            ]
            ==
            best_vz
        )
    )


    print(
        f"{label:10s} "
        f"{100*row['none_fraction']:6.1f}% "
        f"{100*row['single_fraction']:7.1f}% "
        f"{100*row['double_fraction']:7.1f}% "
        f"{row['events']:7d} "
        f"{row['stance_xy_p95']:8.3f} "
        f"{row['stance_vz_p95']:8.3f} "
        f"{row['swing_xy_p50']:8.3f} "
        f"{row['speed_separation']:7.2f} "
        f"{row['single_root_p95']:8.3f} "
        f"{row['double_root_p95']:8.3f} "
        f"{row['transition_root_p95']:8.3f}"
    )


# ================================================================
# SAVE ALL ROWS
# ================================================================

flat_rows = []


for label, result in (
    results.items()
):

    flat_rows.extend(
        result[
            "rows"
        ]
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
            flat_rows[
                0
            ].keys()
        ),
    )


    writer.writeheader()


    for row in flat_rows:

        writer.writerow(
            row
        )


# ================================================================
# SAVE DIAGNOSTIC NPZ
# ================================================================

payload = {

    "best_clearance_enter_mm":
        np.asarray(
            [
                best_clearance
            ],
            dtype=np.float32,
        ),

    "best_vxy_enter":
        np.asarray(
            [
                best_vxy
            ],
            dtype=np.float32,
        ),

    "best_vz_enter":
        np.asarray(
            [
                best_vz
            ],
            dtype=np.float32,
        ),
}


for label, result in (
    results.items()
):

    payload[
        f"{label}_left_clearance"
    ] = (
        result[
            "left_clearance"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_clearance"
    ] = (
        result[
            "right_clearance"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_left_xy"
    ] = (
        result[
            "left_xy"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_xy"
    ] = (
        result[
            "right_xy"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_left_vz"
    ] = (
        result[
            "left_vz"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_right_vz"
    ] = (
        result[
            "right_vz"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{label}_root_bw"
    ] = (
        result[
            "root_bw"
        ].astype(
            np.float32
        )
    )


np.savez_compressed(
    OUTPUT_NPZ,
    **payload,
)


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 205)
print("STAGE 7T DECISION")
print("=" * 205)

print(
    "The script has identified the best PHYSICAL "
    "Support-V2 candidate."
)

print(
    "It has NOT modified any dataset."
)

print(
    "It has NOT selected parameters by minimizing "
    "inverse-dynamics residual."
)

print()
print(
    "NEXT:"
)

print(
    "Inspect the best candidate above."
)

print(
    "If it has low/no unsupported frames, sensible "
    "double-support fraction, low stance velocity, and "
    "stable behavior on all three clips:"
)

print(
    "  -> freeze Support V2 using exactly that state machine."
)

print(
    "Then rerun the corrected dynamic audit with that one "
    "fixed timeline."
)

print()
print(
    "NO REFERENCE MODIFIED."
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
