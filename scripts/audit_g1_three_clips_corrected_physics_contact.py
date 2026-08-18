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


CONTACT_DEPTH = 1e-6

TRANSITION_WINDOW = 2

MAX_ROOT_LOWERING = 0.025


OUTPUT_NPZ = (
    ROOT
    / "results"
    / "g1_corrected_three_clip_dynamic_audit.npz"
)


OUTPUT_CSV = (
    ROOT
    / "results"
    / "g1_corrected_three_clip_dynamic_audit.csv"
)


JOINT_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",

    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",

    "waist_yaw",
    "waist_roll",
    "waist_pitch",

    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


# ================================================================
# ENV / MODEL
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model


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


print("=" * 205)
print("G1 STAGE 7R")
print("CORRECTED THREE-CLIP DYNAMIC FEASIBILITY AUDIT")
print("GLOBAL-LOWEST FIRST-CONTACT PROJECTION")
print("PHYSICS-DERIVED CONTACT/SUPPORT TIMELINE")
print("NO OPTIMIZATION / NO CEM / NO PPO")
print("=" * 205)

print(
    "MuJoCo:",
    mujoco.__version__,
)

print(
    "model nq/nv/nu:",
    model.nq,
    model.nv,
    model.nu,
)

print(
    "mass:",
    f"{TOTAL_MASS:.3f} kg",
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)


# ================================================================
# HELPERS
# ================================================================

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


def differentiate_qpos(
    trajectory,
    dt,
):

    count = len(
        trajectory
    )


    velocity = np.zeros(
        (
            count,
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
        count - 1,
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


def get_contacts(
    data,
):

    left = False
    right = False

    count = 0

    normal_force = 0.0


    for cid in range(
        data.ncon
    ):

        con = data.contact[
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


        if other in LEFT_SET:
            left = True


        if other in RIGHT_SET:
            right = True


        count += 1


        force = np.zeros(
            6,
            dtype=np.float64,
        )


        mujoco.mj_contactForce(
            model,
            data,
            cid,
            force,
        )


        normal_force += max(
            float(
                force[
                    0
                ]
            ),
            0.0,
        )


    return (
        np.asarray(
            [
                left,
                right,
            ],
            dtype=bool,
        ),
        count,
        normal_force,
    )


def build_transition_mask(
    contact,
):

    n = len(
        contact
    )


    mask = np.zeros(
        n,
        dtype=bool,
    )


    transitions = []


    for frame in range(
        1,
        n,
    ):

        if not np.array_equal(
            contact[
                frame
            ],
            contact[
                frame - 1
            ],
        ):

            transitions.append(
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
        transitions,
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


def load_reference(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


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


    return (
        qpos,
        original_support,
        original_contact,
        fps,
    )


# ================================================================
# CLIP AUDIT
# ================================================================

def audit_clip(
    label,
    path,
):

    (
        qpos,
        original_support,
        original_contact,
        fps,
    ) = load_reference(
        path
    )


    n = len(
        qpos
    )


    dt = 1.0 / fps


    if qpos.shape != (
        n,
        model.nq,
    ):

        raise RuntimeError(
            f"{label}: invalid qpos shape "
            f"{qpos.shape}"
        )


    print()
    print("=" * 205)

    print(
        "CLIP:",
        label,
    )

    print("=" * 205)

    print(
        "path:",
        path,
    )

    print(
        "frames:",
        n,
    )

    print(
        "fps:",
        fps,
    )


    # ------------------------------------------------------------
    # RAW SOLE GEOMETRY
    # ------------------------------------------------------------

    probe = mujoco.MjData(
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


    # ------------------------------------------------------------
    # ORIGINAL SINGLE-SUPPORT ORDER MISMATCH
    # ------------------------------------------------------------

    support_count = np.sum(
        original_support,
        axis=1,
    )


    original_single = (
        support_count == 1
    )


    support_order_gap = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )


    order_mismatch = np.zeros(
        n,
        dtype=bool,
    )


    forced_penetration = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        if not original_single[
            frame
        ]:

            continue


        side = int(
            np.argmax(
                original_support[
                    frame
                ]
            )
        )


        if side == 0:

            stance = (
                left_clearance[
                    frame
                ]
            )

            other = (
                right_clearance[
                    frame
                ]
            )

        else:

            stance = (
                right_clearance[
                    frame
                ]
            )

            other = (
                left_clearance[
                    frame
                ]
            )


        gap = (
            stance
            -
            other
        )


        support_order_gap[
            frame
        ] = gap


        order_mismatch[
            frame
        ] = (
            gap
            >
            0.0005
        )


        old_dz = min(
            0.0,
            -CONTACT_DEPTH
            -
            stance,
        )


        old_dz = max(
            -MAX_ROOT_LOWERING,
            old_dz,
        )


        forced_penetration[
            frame
        ] = max(
            0.0,
            -(
                other
                +
                old_dz
            ),
        )


    # ------------------------------------------------------------
    # GLOBAL-LOWEST FIRST-CONTACT PROJECTION
    # ------------------------------------------------------------

    snapped = (
        qpos.copy()
    )


    shift = np.zeros(
        n,
        dtype=np.float64,
    )


    lower_side = np.zeros(
        n,
        dtype=np.int8,
    )


    for frame in range(n):

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

        else:

            side = 1

            clearance = (
                right_clearance[
                    frame
                ]
            )


        dz = (
            -CONTACT_DEPTH
            -
            clearance
        )


        dz = min(
            0.0,
            dz,
        )


        dz = max(
            -MAX_ROOT_LOWERING,
            dz,
        )


        lower_side[
            frame
        ] = side


        shift[
            frame
        ] = dz


        snapped[
            frame,
            2
        ] += dz


    qvel = differentiate_qpos(
        snapped,
        dt,
    )


    qacc = differentiate_array(
        qvel,
        dt,
    )


    shift_velocity = differentiate_array(
        shift,
        dt,
    )


    shift_acceleration = differentiate_array(
        shift_velocity,
        dt,
    )


    # ------------------------------------------------------------
    # CONSISTENT INVERSE DYNAMICS + ACTUAL CONTACT
    # ------------------------------------------------------------

    data = mujoco.MjData(
        model
    )


    root_bw = np.zeros(
        n,
        dtype=np.float64,
    )


    root_force_components = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float64,
    )


    max_tau = np.zeros(
        n,
        dtype=np.float64,
    )


    max_tau_joint = np.zeros(
        n,
        dtype=np.int32,
    )


    normal_bw = np.zeros(
        n,
        dtype=np.float64,
    )


    actual_contact = np.zeros(
        (
            n,
            2,
        ),
        dtype=bool,
    )


    contact_count = np.zeros(
        n,
        dtype=np.int32,
    )


    for frame in range(n):

        data.qpos[:] = (
            snapped[
                frame
            ]
        )


        data.qvel[:] = (
            qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            data,
        )


        data.qacc[:] = (
            qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            data,
        )


        root_force = np.asarray(
            data.qfrc_inverse[
                0:3
            ],
            dtype=np.float64,
        )


        root_force_components[
            frame
        ] = (
            root_force
            /
            BODY_WEIGHT
        )


        root_bw[
            frame
        ] = (
            np.linalg.norm(
                root_force
            )
            /
            BODY_WEIGHT
        )


        joint_tau = np.asarray(
            [
                data.qfrc_inverse[
                    vadr
                ]
                for vadr in env.vaddrs
            ],
            dtype=np.float64,
        )


        ratio = (
            np.abs(
                joint_tau
            )
            /
            effort
        )


        max_tau[
            frame
        ] = float(
            np.max(
                ratio
            )
        )


        max_tau_joint[
            frame
        ] = int(
            np.argmax(
                ratio
            )
        )


        (
            contact,
            count,
            normal_force,
        ) = get_contacts(
            data
        )


        actual_contact[
            frame
        ] = contact


        contact_count[
            frame
        ] = count


        normal_bw[
            frame
        ] = (
            normal_force
            /
            BODY_WEIGHT
        )


    # ------------------------------------------------------------
    # PHYSICS-DERIVED SUPPORT TIMELINE
    # ------------------------------------------------------------

    (
        transition_mask,
        transition_frames,
    ) = build_transition_mask(
        actual_contact
    )


    actual_count = np.sum(
        actual_contact,
        axis=1,
    )


    actual_single = (
        actual_count == 1
    )


    actual_double = (
        actual_count >= 2
    )


    actual_none = (
        actual_count == 0
    )


    steady_single = (
        actual_single
        &
        ~transition_mask
    )


    steady_double = (
        actual_double
        &
        ~transition_mask
    )


    # ------------------------------------------------------------
    # LABEL AGREEMENT
    # ------------------------------------------------------------

    support_agreement = np.all(
        original_support
        ==
        actual_contact,
        axis=1,
    )


    contact_agreement = np.all(
        original_contact
        ==
        actual_contact,
        axis=1,
    )


    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    metrics = {

        "root50":
            float(
                np.percentile(
                    root_bw,
                    50,
                )
            ),

        "root95":
            float(
                np.percentile(
                    root_bw,
                    95,
                )
            ),

        "rootmax":
            float(
                np.max(
                    root_bw
                )
            ),

        "single50":
            safe_percentile(
                root_bw,
                steady_single,
                50,
            ),

        "single95":
            safe_percentile(
                root_bw,
                steady_single,
                95,
            ),

        "double50":
            safe_percentile(
                root_bw,
                steady_double,
                50,
            ),

        "double95":
            safe_percentile(
                root_bw,
                steady_double,
                95,
            ),

        "transition50":
            safe_percentile(
                root_bw,
                transition_mask,
                50,
            ),

        "transition95":
            safe_percentile(
                root_bw,
                transition_mask,
                95,
            ),

        "tau95":
            float(
                np.percentile(
                    max_tau,
                    95,
                )
            ),

        "overlimit":
            float(
                np.mean(
                    max_tau
                    >
                    1.0
                )
            ),

        "normal95":
            float(
                np.percentile(
                    normal_bw,
                    95,
                )
            ),

        "actual_single_frames":
            int(
                np.sum(
                    actual_single
                )
            ),

        "actual_double_frames":
            int(
                np.sum(
                    actual_double
                )
            ),

        "actual_none_frames":
            int(
                np.sum(
                    actual_none
                )
            ),

        "transition_frames_count":
            int(
                np.sum(
                    transition_mask
                )
            ),

        "transition_events":
            len(
                transition_frames
            ),

        "support_agreement":
            float(
                np.mean(
                    support_agreement
                )
            ),

        "contact_agreement":
            float(
                np.mean(
                    contact_agreement
                )
            ),

        "order_mismatch":
            int(
                np.sum(
                    order_mismatch
                )
            ),

        "original_single":
            int(
                np.sum(
                    original_single
                )
            ),

        "forced_penetration_frames":
            int(
                np.sum(
                    forced_penetration
                    >
                    1e-6
                )
            ),

        "forced_penetration_max":
            float(
                np.max(
                    forced_penetration
                )
            ),
    }


    print()
    print(
        "PHYSICS-DERIVED CONTACT COUNTS"
    )

    print(
        " single:",
        metrics[
            "actual_single_frames"
        ],
    )

    print(
        " double:",
        metrics[
            "actual_double_frames"
        ],
    )

    print(
        " none:",
        metrics[
            "actual_none_frames"
        ],
    )

    print(
        " transition events:",
        metrics[
            "transition_events"
        ],
    )

    print(
        " transition-window frames:",
        metrics[
            "transition_frames_count"
        ],
    )


    print()
    print(
        "ORIGINAL LABEL AGREEMENT"
    )

    print(
        " support-mask agreement:",
        f"{100*metrics['support_agreement']:.1f}%",
    )

    print(
        " contact-mask agreement:",
        f"{100*metrics['contact_agreement']:.1f}%",
    )

    print(
        " original single-support order mismatches:",
        f"{metrics['order_mismatch']}/"
        f"{metrics['original_single']}",
    )

    print(
        " expected-snap forced-penetration frames:",
        metrics[
            "forced_penetration_frames"
        ],
    )

    print(
        " maximum forced penetration:",
        f"{1000*metrics['forced_penetration_max']:.2f} mm",
    )


    print()
    print(
        "CORRECTED DYNAMICS"
    )

    print(
        " overall root p50/p95/max:",
        f"{metrics['root50']:.3f}",
        f"{metrics['root95']:.3f}",
        f"{metrics['rootmax']:.3f}",
        "BW",
    )

    print(
        " steady single p50/p95:",
        f"{metrics['single50']:.3f}",
        f"{metrics['single95']:.3f}",
        "BW",
    )

    print(
        " steady double p50/p95:",
        f"{metrics['double50']:.3f}",
        f"{metrics['double95']:.3f}",
        "BW",
    )

    print(
        " transition p50/p95:",
        f"{metrics['transition50']:.3f}",
        f"{metrics['transition95']:.3f}",
        "BW",
    )

    print(
        " torque p95:",
        f"{metrics['tau95']:.3f}",
    )

    print(
        " over-limit:",
        f"{100*metrics['overlimit']:.1f}%",
    )

    print(
        " normal-force p95:",
        f"{metrics['normal95']:.3f}",
        "BW",
    )


    # ------------------------------------------------------------
    # TOP CORRECTED SPIKES
    # ------------------------------------------------------------

    print()
    print(
        "TOP 12 CORRECTED ROOT-RESIDUAL FRAMES"
    )


    top = np.argsort(
        root_bw
    )[
        ::-1
    ][
        :12
    ]


    for rank, frame in enumerate(
        top,
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"frame={frame:03d} "
            f"oldSup={phase_name(original_support[frame])} "
            f"oldCnt={phase_name(original_contact[frame])} "
            f"physics={phase_name(actual_contact[frame])} "
            f"root={root_bw[frame]:7.3f} BW "
            f"F=["
            f"{root_force_components[frame,0]:+.2f},"
            f"{root_force_components[frame,1]:+.2f},"
            f"{root_force_components[frame,2]:+.2f}] "
            f"tau={max_tau[frame]:6.3f} "
            f"{JOINT_NAMES[max_tau_joint[frame]]:24s} "
            f"normal={normal_bw[frame]:6.2f} "
            f"Lclr={1000*left_clearance[frame]:6.2f}mm "
            f"Rclr={1000*right_clearance[frame]:6.2f}mm "
            f"dz={1000*shift[frame]:6.2f}mm "
            f"trans={bool(transition_mask[frame])}"
        )


    return {

        "label":
            label,

        "path":
            path,

        "fps":
            fps,

        "qpos":
            qpos,

        "snapped":
            snapped,

        "qvel":
            qvel,

        "qacc":
            qacc,

        "shift":
            shift,

        "shift_acceleration":
            shift_acceleration,

        "left_clearance":
            left_clearance,

        "right_clearance":
            right_clearance,

        "original_support":
            original_support,

        "original_contact":
            original_contact,

        "actual_contact":
            actual_contact,

        "transition_mask":
            transition_mask,

        "transition_frames":
            np.asarray(
                transition_frames,
                dtype=np.int32,
            ),

        "root_bw":
            root_bw,

        "normal_bw":
            normal_bw,

        "max_tau":
            max_tau,

        "metrics":
            metrics,
    }


# ================================================================
# RUN ALL CLIPS
# ================================================================

results = {}


for label, path in CLIPS.items():

    results[
        label
    ] = audit_clip(
        label,
        path,
    )


# ================================================================
# MEDIUM_08 SANITY GATE
# ================================================================

medium08 = results[
    "medium_08"
]


if abs(
    medium08[
        "metrics"
    ][
        "root95"
    ]
    -
    6.387
) > 0.75:

    raise RuntimeError(
        "medium_08 corrected root-p95 "
        "does not reproduce previous global-lowest result."
    )


if abs(
    medium08[
        "metrics"
    ][
        "tau95"
    ]
    -
    4.409
) > 0.75:

    raise RuntimeError(
        "medium_08 corrected torque-p95 "
        "does not reproduce previous result."
    )


# ================================================================
# MASTER COMPARISON
# ================================================================

print()
print("=" * 205)
print("STAGE 7R MASTER COMPARISON")
print("CORRECTED GLOBAL-LOWEST / PHYSICS-CONTACT AUDIT")
print("=" * 205)

print(
    f"{'CLIP':10s} "
    f"{'ROOT50':>8s} "
    f"{'ROOT95':>8s} "
    f"{'MAX':>8s} "
    f"{'SINGLE':>9s} "
    f"{'DOUBLE':>9s} "
    f"{'TRANS':>9s} "
    f"{'TAU95':>8s} "
    f"{'>LIMIT':>8s} "
    f"{'NORM95':>8s} "
    f"{'SUP-AGR':>8s}"
)

print("-" * 205)


for label in (
    "medium_02",
    "medium_04",
    "medium_08",
):

    m = results[
        label
    ][
        "metrics"
    ]


    print(
        f"{label:10s} "
        f"{m['root50']:8.3f} "
        f"{m['root95']:8.3f} "
        f"{m['rootmax']:8.3f} "
        f"{m['single95']:9.3f} "
        f"{m['double95']:9.3f} "
        f"{m['transition95']:9.3f} "
        f"{m['tau95']:8.3f} "
        f"{100*m['overlimit']:7.1f}% "
        f"{m['normal95']:8.3f} "
        f"{100*m['support_agreement']:7.1f}%"
    )


# ================================================================
# RANKING
# ================================================================

def finite_or_large(
    value,
):

    if np.isfinite(
        value
    ):

        return value

    return 1e6


def score(
    metrics,
):

    # Diagnostic ranking only.
    # Lower is better.
    #
    # Overall and transitions are weighted most because
    # those remain our current failure modes.
    return (
        1.00
        *
        finite_or_large(
            metrics[
                "root95"
            ]
        )
        +
        0.70
        *
        finite_or_large(
            metrics[
                "transition95"
            ]
        )
        +
        0.35
        *
        finite_or_large(
            metrics[
                "single95"
            ]
        )
        +
        0.25
        *
        finite_or_large(
            metrics[
                "tau95"
            ]
        )
    )


ranking = sorted(
    (
        (
            score(
                result[
                    "metrics"
                ]
            ),
            label,
        )

        for label, result in (
            results.items()
        )
    ),
    key=lambda item:
        item[
            0
        ],
)


print()
print("CORRECTED DIAGNOSTIC RANKING")


for rank, (
    value,
    label,
) in enumerate(
    ranking,
    start=1,
):

    print(
        f"{rank}. "
        f"{label} "
        f"score={value:.3f}"
    )


best_label = ranking[
    0
][
    1
]


best = results[
    best_label
][
    "metrics"
]


# ================================================================
# FINAL DECISION
# ================================================================

print()
print("=" * 205)
print("STAGE 7R DECISION")
print("=" * 205)

print(
    "best corrected clip:",
    best_label,
)

print(
    "root p95:",
    f"{best['root95']:.3f} BW",
)

print(
    "steady single p95:",
    f"{best['single95']:.3f} BW",
)

print(
    "steady double p95:",
    f"{best['double95']:.3f} BW",
)

print(
    "transition p95:",
    f"{best['transition95']:.3f} BW",
)

print(
    "torque p95:",
    f"{best['tau95']:.3f}",
)

print()


if (
    best[
        "root95"
    ]
    <=
    3.0

    and
    best[
        "single95"
    ]
    <=
    2.0

    and
    (
        not np.isfinite(
            best[
                "double95"
            ]
        )
        or
        best[
            "double95"
        ]
        <=
        2.0
    )

    and
    best[
        "transition95"
    ]
    <=
    5.0
):

    print(
        "RESULT: AFTER CORRECTING THE CONTACT AUDIT, "
        "AT LEAST ONE CLIP IS CLOSE TO DYNAMICALLY USABLE."
    )

    print(
        "NEXT:"
    )

    print(
        "Create a candidate reference using the "
        "physics-derived support/contact timeline and "
        "run zero-action forward validation."
    )


elif (
    best[
        "single95"
    ]
    <=
    2.5

    and
    best[
        "transition95"
    ]
    >
    5.0
):

    print(
        "RESULT: STEADY SUPPORT IS PHYSICALLY REASONABLE, "
        "BUT CORRECTED TRANSITIONS REMAIN THE PRIMARY PROBLEM."
    )

    print(
        "NEXT:"
    )

    print(
        "Use the physics-derived support timeline and build "
        "one transition-only smoothing/centroidal repair."
    )


elif (
    best[
        "root95"
    ]
    <=
    7.0
):

    print(
        "RESULT: THE OLD AUDIT OVERSTATED THE FAILURE, "
        "BUT SIGNIFICANT DYNAMIC INCONSISTENCY REMAINS."
    )

    print(
        "NEXT:"
    )

    print(
        "Inspect the best clip's remaining corrected top "
        "residual frames before modifying the reference."
    )


else:

    print(
        "RESULT: EVEN THE CORRECTED FIRST-CONTACT AUDIT "
        "SHOWS LARGE DYNAMIC INCONSISTENCY."
    )

    print(
        "NEXT:"
    )

    print(
        "Stop patching these retargeted references and "
        "revisit the AMASS-to-G1 retargeting/root trajectory."
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

    fieldnames = [
        "clip",
        "frame",
        "time_s",

        "original_support",
        "original_contact",
        "physics_contact",

        "transition",

        "left_clearance_mm",
        "right_clearance_mm",

        "root_shift_mm",

        "root_bw",
        "normal_bw",

        "max_tau_ratio",
    ]


    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    for label, result in (
        results.items()
    ):

        fps = result[
            "fps"
        ]


        n = len(
            result[
                "qpos"
            ]
        )


        for frame in range(n):

            writer.writerow(
                {
                    "clip":
                        label,

                    "frame":
                        frame,

                    "time_s":
                        frame
                        /
                        fps,

                    "original_support":
                        phase_name(
                            result[
                                "original_support"
                            ][
                                frame
                            ]
                        ),

                    "original_contact":
                        phase_name(
                            result[
                                "original_contact"
                            ][
                                frame
                            ]
                        ),

                    "physics_contact":
                        phase_name(
                            result[
                                "actual_contact"
                            ][
                                frame
                            ]
                        ),

                    "transition":
                        int(
                            result[
                                "transition_mask"
                            ][
                                frame
                            ]
                        ),

                    "left_clearance_mm":
                        1000
                        *
                        result[
                            "left_clearance"
                        ][
                            frame
                        ],

                    "right_clearance_mm":
                        1000
                        *
                        result[
                            "right_clearance"
                        ][
                            frame
                        ],

                    "root_shift_mm":
                        1000
                        *
                        result[
                            "shift"
                        ][
                            frame
                        ],

                    "root_bw":
                        result[
                            "root_bw"
                        ][
                            frame
                        ],

                    "normal_bw":
                        result[
                            "normal_bw"
                        ][
                            frame
                        ],

                    "max_tau_ratio":
                        result[
                            "max_tau"
                        ][
                            frame
                        ],
                }
            )


# ================================================================
# SAVE NPZ
# ================================================================

payload = {}


for label, result in (
    results.items()
):

    prefix = label


    payload[
        f"{prefix}_physics_contact"
    ] = (
        result[
            "actual_contact"
        ]
    )


    payload[
        f"{prefix}_transition_mask"
    ] = (
        result[
            "transition_mask"
        ]
    )


    payload[
        f"{prefix}_transition_frames"
    ] = (
        result[
            "transition_frames"
        ]
    )


    payload[
        f"{prefix}_root_bw"
    ] = (
        result[
            "root_bw"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{prefix}_normal_bw"
    ] = (
        result[
            "normal_bw"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{prefix}_max_tau"
    ] = (
        result[
            "max_tau"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{prefix}_root_shift"
    ] = (
        result[
            "shift"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{prefix}_left_clearance"
    ] = (
        result[
            "left_clearance"
        ].astype(
            np.float32
        )
    )


    payload[
        f"{prefix}_right_clearance"
    ] = (
        result[
            "right_clearance"
        ].astype(
            np.float32
        )
    )


    metrics = result[
        "metrics"
    ]


    payload[
        f"{prefix}_summary"
    ] = np.asarray(
        [
            metrics[
                "root50"
            ],
            metrics[
                "root95"
            ],
            metrics[
                "rootmax"
            ],
            metrics[
                "single95"
            ],
            metrics[
                "double95"
            ],
            metrics[
                "transition95"
            ],
            metrics[
                "tau95"
            ],
            metrics[
                "overlimit"
            ],
            metrics[
                "normal95"
            ],
            metrics[
                "support_agreement"
            ],
        ],
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
    "NO SOURCE REFERENCE WAS MODIFIED."
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
