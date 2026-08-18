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


TOLERANCES_MM = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
]


CONTACT_DEPTH = 1e-6

MAX_ROOT_LOWERING = 0.025

TRANSITION_WINDOW = 2


OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_support_timeline_geometry_sweep.csv"
)


OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_support_timeline_geometry_sweep.npz"
)


# ================================================================
# ENV / MODEL
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


LEFT_SET = set(
    LEFT_GEOMS
)


RIGHT_SET = set(
    RIGHT_GEOMS
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
print("G1 STAGE 7S")
print("GEOMETRY-DERIVED SUPPORT TIMELINE SENSITIVITY")
print("THRESHOLD SWEEP — NO FORCED SUPPORT LABELS")
print("NO OPTIMIZATION / NO CEM / NO PPO")
print("=" * 205)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)

print(
    "tolerances [mm]:",
    TOLERANCES_MM,
)


# ================================================================
# HELPERS
# ================================================================

def phase_name(mask):

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


def differentiate_qpos(
    qpos,
    dt,
):

    n = len(
        qpos
    )


    qvel = np.zeros(
        (
            n,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        qvel[
            0
        ],
        dt,
        qpos[
            0
        ],
        qpos[
            1
        ],
    )


    for frame in range(
        1,
        n - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            qvel[
                frame
            ],
            2.0
            *
            dt,
            qpos[
                frame - 1
            ],
            qpos[
                frame + 1
            ],
        )


    mujoco.mj_differentiatePos(
        model,
        qvel[
            -1
        ],
        dt,
        qpos[
            -2
        ],
        qpos[
            -1
        ],
    )


    return qvel


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
        2.0
        *
        dt
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


def transition_mask_from_support(
    support,
):

    n = len(
        support
    )


    mask = np.zeros(
        n,
        dtype=bool,
    )


    event_frames = []


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

            event_frames.append(
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


            mask[
                lo:hi
            ] = True


    return (
        mask,
        event_frames,
    )


def safe_percentile(
    values,
    mask,
    percentile,
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
            percentile,
        )
    )


# ================================================================
# LOAD / AUDIT ONE CLIP
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


        original_support = (
            np.asarray(
                f[
                    "support_mask"
                ],
                dtype=np.float64,
            )
            >
            0.5
        )


        original_contact = (
            np.asarray(
                f[
                    "contact_mask"
                ],
                dtype=np.float64,
            )
            >
            0.5
        )


        if "fps" in f.files:

            fps = float(
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

        else:

            fps = 50.0


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

    print(
        "path:",
        path,
    )

    print(
        "frames:",
        n,
    )


    # ------------------------------------------------------------
    # RAW GEOMETRY
    # ------------------------------------------------------------

    data = mujoco.MjData(
        model
    )


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

        data.qpos[:] = (
            qpos[
                frame
            ]
        )


        data.qvel[:] = 0.0


        mujoco.mj_forward(
            model,
            data,
        )


        left_clearance[
            frame
        ] = sole_clearance(
            data,
            LEFT_GEOMS,
        )


        right_clearance[
            frame
        ] = sole_clearance(
            data,
            RIGHT_GEOMS,
        )


        left_position[
            frame
        ] = sole_center(
            data,
            LEFT_GEOMS,
        )


        right_position[
            frame
        ] = sole_center(
            data,
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


    left_speed = np.linalg.norm(
        left_velocity,
        axis=1,
    )


    right_speed = np.linalg.norm(
        right_velocity,
        axis=1,
    )


    left_xy_speed = np.linalg.norm(
        left_velocity[
            :,
            0:2
        ],
        axis=1,
    )


    right_xy_speed = np.linalg.norm(
        right_velocity[
            :,
            0:2
        ],
        axis=1,
    )


    # ------------------------------------------------------------
    # CORRECTED GLOBAL-LOWEST PROJECTION
    # ------------------------------------------------------------

    snapped = (
        qpos.copy()
    )


    root_shift = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        minimum = min(
            left_clearance[
                frame
            ],
            right_clearance[
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


        snapped[
            frame,
            2
        ] += dz


        root_shift[
            frame
        ] = dz


    qvel = differentiate_qpos(
        snapped,
        dt,
    )


    qacc = differentiate_array(
        qvel,
        dt,
    )


    # ------------------------------------------------------------
    # CONSISTENT INVERSE DYNAMICS
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


    normal_bw = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        inverse_data.qpos[:] = (
            snapped[
                frame
            ]
        )


        inverse_data.qvel[:] = (
            qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            inverse_data,
        )


        inverse_data.qacc[:] = (
            qacc[
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


        total_normal = 0.0


        for cid in range(
            inverse_data.ncon
        ):

            con = inverse_data.contact[
                cid
            ]


            if int(
                con.efc_address
            ) < 0:

                continue


            g1 = int(
                con.geom1
            )

            g2 = int(
                con.geom2
            )


            if FLOOR_GEOM not in (
                g1,
                g2,
            ):

                continue


            other = (
                g2
                if g1
                ==
                FLOOR_GEOM
                else g1
            )


            if (
                other not in LEFT_SET
                and
                other not in RIGHT_SET
            ):

                continue


            force = np.zeros(
                6,
                dtype=np.float64,
            )


            mujoco.mj_contactForce(
                model,
                inverse_data,
                cid,
                force,
            )


            total_normal += max(
                float(
                    force[
                        0
                    ]
                ),
                0.0,
            )


        normal_bw[
            frame
        ] = (
            total_normal
            /
            BODY_WEIGHT
        )


    # ------------------------------------------------------------
    # SANITY: CORRECTED ROOT P95
    # ------------------------------------------------------------

    root95 = float(
        np.percentile(
            root_bw,
            95,
        )
    )


    print()
    print(
        "corrected global-lowest root p95:",
        f"{root95:.3f} BW",
    )


    expected_root = {
        "medium_02":
            5.115,

        "medium_04":
            4.800,

        "medium_08":
            6.387,
    }[
        label
    ]


    if abs(
        root95
        -
        expected_root
    ) > 0.75:

        raise RuntimeError(
            f"{label}: corrected root baseline changed."
        )


    # ------------------------------------------------------------
    # THRESHOLD SWEEP
    # ------------------------------------------------------------

    rows = []


    clearance_min = np.minimum(
        left_clearance,
        right_clearance,
    )


    for tolerance_mm in (
        TOLERANCES_MM
    ):

        tolerance = (
            tolerance_mm
            /
            1000.0
        )


        geometry_support = np.zeros(
            (
                n,
                2,
            ),
            dtype=bool,
        )


        geometry_support[
            :,
            0
        ] = (
            left_clearance
            <=
            clearance_min
            +
            tolerance
        )


        geometry_support[
            :,
            1
        ] = (
            right_clearance
            <=
            clearance_min
            +
            tolerance
        )


        count = np.sum(
            geometry_support,
            axis=1,
        )


        single = (
            count == 1
        )


        double = (
            count == 2
        )


        (
            transition,
            transition_events,
        ) = transition_mask_from_support(
            geometry_support
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


        left_only = (
            geometry_support[
                :,
                0
            ]
            &
            ~geometry_support[
                :,
                1
            ]
        )


        right_only = (
            geometry_support[
                :,
                1
            ]
            &
            ~geometry_support[
                :,
                0
            ]
        )


        # --------------------------------------------------------
        # Stance speed collection.
        # --------------------------------------------------------

        stance_speed = np.concatenate(
            [
                left_xy_speed[
                    geometry_support[
                        :,
                        0
                    ]
                ],

                right_xy_speed[
                    geometry_support[
                        :,
                        1
                    ]
                ],
            ]
        )


        # Swing speed only makes sense during single support.
        swing_speed = np.concatenate(
            [
                right_xy_speed[
                    left_only
                ],

                left_xy_speed[
                    right_only
                ],
            ]
        )


        if len(
            stance_speed
        ) == 0:

            stance_p50 = float(
                "nan"
            )

            stance_p95 = float(
                "nan"
            )

        else:

            stance_p50 = float(
                np.percentile(
                    stance_speed,
                    50,
                )
            )

            stance_p95 = float(
                np.percentile(
                    stance_speed,
                    95,
                )
            )


        if len(
            swing_speed
        ) == 0:

            swing_p50 = float(
                "nan"
            )

            swing_p95 = float(
                "nan"
            )

        else:

            swing_p50 = float(
                np.percentile(
                    swing_speed,
                    50,
                )
            )

            swing_p95 = float(
                np.percentile(
                    swing_speed,
                    95,
                )
            )


        speed_separation = (
            swing_p50
            /
            max(
                stance_p50,
                1e-9,
            )
            if (
                np.isfinite(
                    swing_p50
                )
                and
                np.isfinite(
                    stance_p50
                )
            )
            else float(
                "nan"
            )
        )


        support_agreement = float(
            np.mean(
                np.all(
                    geometry_support
                    ==
                    original_support,
                    axis=1,
                )
            )
        )


        contact_agreement = float(
            np.mean(
                np.all(
                    geometry_support
                    ==
                    original_contact,
                    axis=1,
                )
            )
        )


        result = {

            "clip":
                label,

            "tolerance_mm":
                tolerance_mm,

            "left_only_fraction":
                float(
                    np.mean(
                        left_only
                    )
                ),

            "right_only_fraction":
                float(
                    np.mean(
                        right_only
                    )
                ),

            "double_fraction":
                float(
                    np.mean(
                        double
                    )
                ),

            "transition_events":
                len(
                    transition_events
                ),

            "transition_frames":
                int(
                    np.sum(
                        transition
                    )
                ),

            "support_agreement":
                support_agreement,

            "contact_agreement":
                contact_agreement,

            "stance_speed_p50":
                stance_p50,

            "stance_speed_p95":
                stance_p95,

            "swing_speed_p50":
                swing_p50,

            "swing_speed_p95":
                swing_p95,

            "speed_separation":
                speed_separation,

            "single_root_p50":
                safe_percentile(
                    root_bw,
                    steady_single,
                    50,
                ),

            "single_root_p95":
                safe_percentile(
                    root_bw,
                    steady_single,
                    95,
                ),

            "double_root_p50":
                safe_percentile(
                    root_bw,
                    steady_double,
                    50,
                ),

            "double_root_p95":
                safe_percentile(
                    root_bw,
                    steady_double,
                    95,
                ),

            "transition_root_p50":
                safe_percentile(
                    root_bw,
                    transition,
                    50,
                ),

            "transition_root_p95":
                safe_percentile(
                    root_bw,
                    transition,
                    95,
                ),

            "tau_p95":
                float(
                    np.percentile(
                        max_tau,
                        95,
                    )
                ),

            "normal_p95":
                float(
                    np.percentile(
                        normal_bw,
                        95,
                    )
                ),
        }


        rows.append(
            result
        )


    # ------------------------------------------------------------
    # PRINT TABLE
    # ------------------------------------------------------------

    print()
    print(
        "GEOMETRY SUPPORT THRESHOLD SWEEP"
    )


    print(
        f"{'TOL':>6s} "
        f"{'L%':>6s} "
        f"{'R%':>6s} "
        f"{'B%':>6s} "
        f"{'EVT':>5s} "
        f"{'SUPAGR':>7s} "
        f"{'CNTAGR':>7s} "
        f"{'STv50':>7s} "
        f"{'STv95':>7s} "
        f"{'SWv50':>7s} "
        f"{'SEP':>6s} "
        f"{'S95':>8s} "
        f"{'D95':>8s} "
        f"{'T95':>8s}"
    )


    print("-" * 205)


    for row in rows:

        print(
            f"{row['tolerance_mm']:6.1f} "
            f"{100*row['left_only_fraction']:6.1f} "
            f"{100*row['right_only_fraction']:6.1f} "
            f"{100*row['double_fraction']:6.1f} "
            f"{row['transition_events']:5d} "
            f"{100*row['support_agreement']:6.1f}% "
            f"{100*row['contact_agreement']:6.1f}% "
            f"{row['stance_speed_p50']:7.3f} "
            f"{row['stance_speed_p95']:7.3f} "
            f"{row['swing_speed_p50']:7.3f} "
            f"{row['speed_separation']:6.2f} "
            f"{row['single_root_p95']:8.3f} "
            f"{row['double_root_p95']:8.3f} "
            f"{row['transition_root_p95']:8.3f}"
        )


    return {

        "rows":
            rows,

        "left_clearance":
            left_clearance,

        "right_clearance":
            right_clearance,

        "left_xy_speed":
            left_xy_speed,

        "right_xy_speed":
            right_xy_speed,

        "root_bw":
            root_bw,

        "max_tau":
            max_tau,

        "normal_bw":
            normal_bw,
    }


# ================================================================
# RUN ALL THREE
# ================================================================

all_results = {}


for label, path in CLIPS.items():

    all_results[
        label
    ] = process_clip(
        label,
        path,
    )


# ================================================================
# CROSS-CLIP TABLE BY THRESHOLD
# ================================================================

print()
print("=" * 205)
print("CROSS-CLIP SUPPORT-TIMELINE SENSITIVITY")
print("=" * 205)


for tolerance_mm in (
    TOLERANCES_MM
):

    print()
    print(
        f"--- tolerance = {tolerance_mm:.1f} mm ---"
    )


    print(
        f"{'CLIP':10s} "
        f"{'B%':>7s} "
        f"{'EVENTS':>7s} "
        f"{'STv95':>8s} "
        f"{'SWv50':>8s} "
        f"{'SEP':>7s} "
        f"{'SINGLE95':>10s} "
        f"{'DOUBLE95':>10s} "
        f"{'TRANS95':>10s}"
    )


    for label in (
        "medium_02",
        "medium_04",
        "medium_08",
    ):

        row = next(
            item
            for item in all_results[
                label
            ][
                "rows"
            ]
            if item[
                "tolerance_mm"
            ]
            ==
            tolerance_mm
        )


        print(
            f"{label:10s} "
            f"{100*row['double_fraction']:7.1f} "
            f"{row['transition_events']:7d} "
            f"{row['stance_speed_p95']:8.3f} "
            f"{row['swing_speed_p50']:8.3f} "
            f"{row['speed_separation']:7.2f} "
            f"{row['single_root_p95']:10.3f} "
            f"{row['double_root_p95']:10.3f} "
            f"{row['transition_root_p95']:10.3f}"
        )


# ================================================================
# THRESHOLD STABILITY CHECK
# ================================================================

print()
print("=" * 205)
print("THRESHOLD STABILITY CHECK")
print("=" * 205)

print(
    "We do NOT automatically choose a contact threshold here."
)

print(
    "Look for a tolerance BAND where:"
)

print(
    "1. double-support fraction changes gradually,"
)

print(
    "2. transition count is stable,"
)

print(
    "3. labelled stance-foot speed remains low,"
)

print(
    "4. swing speed remains clearly higher than stance speed,"
)

print(
    "5. SINGLE / DOUBLE / TRANS residuals are stable across "
    "neighboring thresholds."
)


# ================================================================
# SAVE CSV
# ================================================================

rows_flat = []


for label, result in (
    all_results.items()
):

    rows_flat.extend(
        result[
            "rows"
        ]
    )


fieldnames = list(
    rows_flat[
        0
    ].keys()
)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    for row in rows_flat:

        writer.writerow(
            row
        )


# ================================================================
# SAVE NPZ
# ================================================================

payload = {}


for label, result in (
    all_results.items()
):

    payload[
        f"{label}_left_clearance"
    ] = result[
        "left_clearance"
    ].astype(
        np.float32
    )


    payload[
        f"{label}_right_clearance"
    ] = result[
        "right_clearance"
    ].astype(
        np.float32
    )


    payload[
        f"{label}_left_xy_speed"
    ] = result[
        "left_xy_speed"
    ].astype(
        np.float32
    )


    payload[
        f"{label}_right_xy_speed"
    ] = result[
        "right_xy_speed"
    ].astype(
        np.float32
    )


    payload[
        f"{label}_root_bw"
    ] = result[
        "root_bw"
    ].astype(
        np.float32
    )


payload[
    "tolerances_mm"
] = np.asarray(
    TOLERANCES_MM,
    dtype=np.float32,
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
    "NO REFERENCE WAS MODIFIED."
)

print(
    "NO SUPPORT THRESHOLD WAS COMMITTED."
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
