from pathlib import Path
import math
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


# ================================================================
# CONFIG
# ================================================================

BASE_CANDIDATE = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_centroidal_root_candidate.npz"
)


OUTPUT_REFERENCE = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_pass2b2_lowerbody_candidate.npz"
)


OUTPUT_DIAGNOSTIC = (
    ROOT
    / "results"
    / "g1_dynamic_v2_pass2b2_lowerbody.npz"
)


CONTACT_DEPTH = 1e-6

TRANSITION_WINDOW = 2

RAMP_FRAMES = 4

HORIZON = 120


# ------------------------------------------------
# Per support phase:
# hip pitch
# hip roll
# knee
# ankle pitch
# ------------------------------------------------

JOINT_LIMITS = np.asarray(
    [
        0.080,
        0.060,
        0.100,
        0.080,
    ],
    dtype=np.float64,
)


# Dynamic objective
ROOT_FORCE_WEIGHT = 1.00
ROOT_TORQUE_WEIGHT = 0.25

# Torque penalty — much stronger than Pass 2B1
TORQUE_SOFT_WEIGHT = 0.35
TORQUE_HARD_WEIGHT = 1.25

MOTION_REG_WEIGHT = 0.40
PHASE_SIMILARITY_WEIGHT = 0.15


MAX_NFEV = 45


STARTS = [
    0,
    15,
    30,
    45,
    57,
    75,
    85,
    105,
    114,
    135,
    140,
]


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model

N = env.num_frames

DT = 1.0 / env.reference_fps


original_qpos = np.asarray(
    env.ref_full_qpos,
    dtype=np.float64,
).copy()


original_qvel = np.asarray(
    env.ref_full_qvel,
    dtype=np.float64,
).copy()


if not BASE_CANDIDATE.exists():
    raise FileNotFoundError(
        BASE_CANDIDATE
    )


with np.load(
    BASE_CANDIDATE,
    allow_pickle=True,
) as f:

    base_qpos = np.asarray(
        f["full_qpos"],
        dtype=np.float64,
    ).copy()

    base_qvel = np.asarray(
        f["full_qvel"],
        dtype=np.float64,
    ).copy()


if base_qpos.shape != original_qpos.shape:

    raise RuntimeError(
        "Pass-2B1 qpos shape mismatch."
    )


support = (
    np.asarray(
        env.ref_support,
        dtype=np.float64,
    )
    > 0.5
)


support_count = np.sum(
    support,
    axis=1,
)


single_support = (
    support_count == 1
)


double_support = (
    support_count >= 2
)


effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
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


ROOT_TORQUE_SCALE = (
    BODY_WEIGHT * 0.50
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


print("=" * 205)
print("G1 DYNAMIC V2 — PASS 2B2")
print("PASS-2B1 ROOT TRAJECTORY FROZEN")
print("SMALL SUPPORT-LEG DYNAMICS OPTIMIZATION")
print("20 VARIABLES TOTAL")
print("STRONG ACTUATOR-FEASIBILITY PENALTY")
print("NO CEM / NO PPO")
print("=" * 205)

print(
    "base candidate:",
    BASE_CANDIDATE,
)

print(
    "frames:",
    N,
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)


# ================================================================
# SUPPORT SEGMENTS
# ================================================================

segments = []


frame = 0


while frame < N:

    if not single_support[frame]:

        frame += 1
        continue


    side = int(
        np.argmax(
            support[frame]
        )
    )


    start = frame

    frame += 1


    while (
        frame < N
        and
        single_support[frame]
        and
        int(
            np.argmax(
                support[frame]
            )
        ) == side
    ):

        frame += 1


    end = frame - 1


    if end - start + 1 >= 5:

        segments.append(
            (
                start,
                end,
                side,
            )
        )


print()
print(
    "support segments:",
    len(segments),
)


for i, (
    start,
    end,
    side,
) in enumerate(segments):

    print(
        f"{i:02d}: "
        f"{'LEFT' if side == 0 else 'RIGHT'} "
        f"{start:03d}..{end:03d}"
    )


if len(segments) != 5:

    raise RuntimeError(
        "Expected exactly five stable "
        "single-support segments."
    )


# ================================================================
# FROZEN MASK
# ================================================================

transition = np.zeros(
    N,
    dtype=bool,
)


for frame in range(
    1,
    N,
):

    if not np.array_equal(
        support[frame],
        support[frame - 1],
    ):

        lo = max(
            0,
            frame - TRANSITION_WINDOW,
        )

        hi = min(
            N,
            frame
            + TRANSITION_WINDOW
            + 1,
        )

        transition[lo:hi] = True


steady_single = (
    single_support
    &
    ~transition
)


steady_double = (
    double_support
    &
    ~transition
)


frame_weight = np.full(
    N,
    0.20,
    dtype=np.float64,
)


frame_weight[
    steady_double
] = 0.30


frame_weight[
    transition
] = 0.80


frame_weight[
    steady_single
] = 1.00


print()
print("FROZEN MASK")

print(
    "steady single:",
    int(
        np.sum(
            steady_single
        )
    ),
)

print(
    "steady double:",
    int(
        np.sum(
            steady_double
        )
    ),
)

print(
    "transition:",
    int(
        np.sum(
            transition
        )
    ),
)


# ================================================================
# JOINT CHANNELS
# ================================================================

# G1 ordering:
#
# left:
# 0 hip pitch
# 1 hip roll
# 3 knee
# 4 ankle pitch
#
# right:
# 6 hip pitch
# 7 hip roll
# 9 knee
# 10 ankle pitch

LEFT_CHANNELS = [
    0,
    1,
    3,
    4,
]


RIGHT_CHANNELS = [
    6,
    7,
    9,
    10,
]


def qpos_address(
    joint_index,
):

    jid = env.joint_ids[
        joint_index
    ]

    return int(
        model.jnt_qposadr[
            jid
        ]
    )


LEFT_QPOS = [
    qpos_address(i)
    for i in LEFT_CHANNELS
]


RIGHT_QPOS = [
    qpos_address(i)
    for i in RIGHT_CHANNELS
]


# ================================================================
# SMOOTH PHASE WINDOW
# ================================================================

def smoothstep(x):

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x
        * x
        * (
            3.0
            -
            2.0
            * x
        )
    )


def support_window(
    length,
):

    result = np.zeros(
        length,
        dtype=np.float64,
    )


    for i in range(length):

        d = min(
            i,
            length - 1 - i,
        )


        alpha = min(
            1.0,
            d
            /
            float(
                max(
                    RAMP_FRAMES,
                    1,
                )
            ),
        )


        result[i] = smoothstep(
            alpha
        )


    return result


# ================================================================
# APPLY 20 PARAMETERS
# ================================================================

def build_candidate(
    flat,
):

    normalized = np.asarray(
        flat,
        dtype=np.float64,
    ).reshape(
        len(segments),
        4,
    )


    amplitudes = (
        normalized
        *
        JOINT_LIMITS[
            None,
            :
        ]
    )


    qpos = base_qpos.copy()


    for segment_index, (
        start,
        end,
        side,
    ) in enumerate(
        segments
    ):

        addresses = (
            LEFT_QPOS
            if side == 0
            else RIGHT_QPOS
        )


        window = support_window(
            end - start + 1
        )


        for local_index, frame in enumerate(
            range(
                start,
                end + 1,
            )
        ):

            w = window[
                local_index
            ]


            for channel in range(4):

                qpos[
                    frame,
                    addresses[
                        channel
                    ]
                ] += (
                    amplitudes[
                        segment_index,
                        channel
                    ]
                    *
                    w
                )


        # Physical joint limits.
        joint_indices = (
            LEFT_CHANNELS
            if side == 0
            else RIGHT_CHANNELS
        )


        for joint_index in joint_indices:

            jid = env.joint_ids[
                joint_index
            ]


            if not model.jnt_limited[
                jid
            ]:
                continue


            qadr = int(
                model.jnt_qposadr[
                    jid
                ]
            )


            lo, hi = (
                model.jnt_range[
                    jid
                ]
            )


            qpos[
                start:end+1,
                qadr
            ] = np.clip(
                qpos[
                    start:end+1,
                    qadr
                ],
                lo,
                hi,
            )


    return (
        qpos,
        normalized,
        amplitudes,
    )


# ================================================================
# DIFFERENTIATION
# ================================================================

def differentiate_qpos(
    qpos,
):

    qvel = np.zeros(
        (
            N,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        qvel[0],
        DT,
        qpos[0],
        qpos[1],
    )


    for frame in range(
        1,
        N - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            qvel[
                frame
            ],
            2.0 * DT,
            qpos[
                frame - 1
            ],
            qpos[
                frame + 1
            ],
        )


    mujoco.mj_differentiatePos(
        model,
        qvel[-1],
        DT,
        qpos[-2],
        qpos[-1],
    )


    return qvel


def differentiate_velocity(
    qvel,
):

    qacc = np.zeros_like(
        qvel
    )


    qacc[1:-1] = (
        qvel[2:]
        -
        qvel[:-2]
    ) / (
        2.0 * DT
    )


    qacc[0] = (
        qvel[1]
        -
        qvel[0]
    ) / DT


    qacc[-1] = (
        qvel[-1]
        -
        qvel[-2]
    ) / DT


    return qacc


# ================================================================
# CONTACT SNAP
# ================================================================

probe = mujoco.MjData(
    model
)


def sole_clearance(
    data,
    geoms,
):

    floor_z = float(
        data.geom_xpos[
            env.floor_geom,
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
            bottom - floor_z,
        )


    return minimum


def expected_support(
    frame,
):

    expected = (
        support[
            frame
        ].copy()
    )


    if not np.any(
        expected
    ):

        expected = (
            np.asarray(
                env.ref_contact[
                    frame
                ],
                dtype=np.float64,
            )
            > 0.5
        )


    return expected


def choose_anchor(
    qpos,
    frame,
):

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


    lc = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    rc = sole_clearance(
        probe,
        RIGHT_GEOMS,
    )


    expected = expected_support(
        frame
    )


    candidates = []


    if expected[0]:

        candidates.append(
            (
                0,
                lc,
            )
        )


    if expected[1]:

        candidates.append(
            (
                1,
                rc,
            )
        )


    if not candidates:

        candidates = [
            (
                0,
                lc,
            ),
            (
                1,
                rc,
            ),
        ]


    return min(
        candidates,
        key=lambda x:
            x[1],
    )


def contact_snap(
    qpos,
):

    snapped = qpos.copy()


    anchors = np.zeros(
        N,
        dtype=np.int8,
    )


    for frame in range(N):

        (
            anchor,
            clearance,
        ) = choose_anchor(
            qpos,
            frame,
        )


        anchors[
            frame
        ] = anchor


        dz = (
            -CONTACT_DEPTH
            - clearance
        )


        dz = min(
            0.0,
            dz,
        )


        dz = max(
            -0.025,
            dz,
        )


        snapped[
            frame,
            2
        ] += dz


    return (
        snapped,
        anchors,
    )


# ================================================================
# CONTACT CHECK
# ================================================================

def active_contacts(
    data,
):

    left = False
    right = False


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


        if env.floor_geom not in (
            g1,
            g2,
        ):

            continue


        other = (
            g2
            if g1
            == env.floor_geom
            else g1
        )


        if other in LEFT_SET:

            left = True


        elif other in RIGHT_SET:

            right = True


    return (
        left,
        right,
    )


# ================================================================
# INVERSE DYNAMICS
# ================================================================

dyn_data = mujoco.MjData(
    model
)


def dynamics(
    qpos,
):

    (
        snapped,
        anchors,
    ) = contact_snap(
        qpos
    )


    qvel = differentiate_qpos(
        snapped
    )


    qacc = differentiate_velocity(
        qvel
    )


    root_force = np.zeros(
        (
            N,
            3,
        ),
        dtype=np.float64,
    )


    root_torque = np.zeros(
        (
            N,
            3,
        ),
        dtype=np.float64,
    )


    joint_tau = np.zeros(
        (
            N,
            29,
        ),
        dtype=np.float64,
    )


    anchor_active = np.zeros(
        N,
        dtype=bool,
    )


    for frame in range(N):

        dyn_data.qpos[:] = (
            snapped[
                frame
            ]
        )


        dyn_data.qvel[:] = (
            qvel[
                frame
            ]
        )


        mujoco.mj_forward(
            model,
            dyn_data,
        )


        dyn_data.qacc[:] = (
            qacc[
                frame
            ]
        )


        mujoco.mj_inverse(
            model,
            dyn_data,
        )


        (
            left_active,
            right_active,
        ) = active_contacts(
            dyn_data
        )


        anchor_active[
            frame
        ] = (
            left_active
            if anchors[
                frame
            ] == 0
            else right_active
        )


        root_force[
            frame
        ] = (
            dyn_data.qfrc_inverse[
                0:3
            ]
        )


        root_torque[
            frame
        ] = (
            dyn_data.qfrc_inverse[
                3:6
            ]
        )


        for j, vadr in enumerate(
            env.vaddrs
        ):

            joint_tau[
                frame,
                j
            ] = (
                dyn_data.qfrc_inverse[
                    vadr
                ]
            )


    return {

        "root_force":
            root_force,

        "root_torque":
            root_torque,

        "joint_tau":
            joint_tau,

        "anchor_active":
            anchor_active,

        "qvel":
            qvel,
    }


# ================================================================
# SUMMARY
# ================================================================

def summarize(
    dyn,
):

    root_bw = (
        np.linalg.norm(
            dyn[
                "root_force"
            ],
            axis=1,
        )
        /
        BODY_WEIGHT
    )


    torque_ratio = (
        np.abs(
            dyn[
                "joint_tau"
            ]
        )
        /
        effort[
            None,
            :
        ]
    )


    max_tau = np.max(
        torque_ratio,
        axis=1,
    )


    def p95(
        mask,
    ):

        return float(
            np.percentile(
                root_bw[
                    mask
                ],
                95,
            )
        )


    return {

        "root_bw":
            root_bw,

        "max_tau":
            max_tau,

        "anchor":
            float(
                np.mean(
                    dyn[
                        "anchor_active"
                    ]
                )
            ),

        "overall":
            float(
                np.percentile(
                    root_bw,
                    95,
                )
            ),

        "single":
            p95(
                steady_single
            ),

        "double":
            p95(
                steady_double
            ),

        "transition":
            p95(
                transition
            ),

        "tau95":
            float(
                np.percentile(
                    max_tau,
                    95,
                )
            ),

        "over":
            float(
                np.mean(
                    max_tau
                    >
                    1.0
                )
            ),
    }


# ================================================================
# BASE DYNAMIC CHECK
# ================================================================

print()
print("=" * 205)
print("PASS-2B1 BASE DYNAMIC CHECK")
print("=" * 205)


base_dyn = dynamics(
    base_qpos
)


base_summary = summarize(
    base_dyn
)


print(
    "root p95:",
    f"{base_summary['overall']:.3f}",
)

print(
    "single p95:",
    f"{base_summary['single']:.3f}",
)

print(
    "transition p95:",
    f"{base_summary['transition']:.3f}",
)

print(
    "tau p95:",
    f"{base_summary['tau95']:.3f}",
)

print(
    "over-limit:",
    f"{100*base_summary['over']:.1f}%",
)


if abs(
    base_summary[
        "overall"
    ]
    - 3.019
) > 0.75:

    raise RuntimeError(
        "Pass-2B1 base did not reproduce."
    )


# ================================================================
# OBJECTIVE
# ================================================================

objective_calls = 0


def objective(
    flat,
):

    global objective_calls

    objective_calls += 1


    (
        qpos,
        normalized,
        amplitudes,
    ) = build_candidate(
        flat
    )


    dyn = dynamics(
        qpos
    )


    residual = []


    for frame in range(N):

        w = math.sqrt(
            frame_weight[
                frame
            ]
        )


        # Root translational consistency.
        residual.extend(
            (
                ROOT_FORCE_WEIGHT
                *
                w
                *
                dyn[
                    "root_force"
                ][
                    frame
                ]
                /
                BODY_WEIGHT
            ).tolist()
        )


        # Root rotational consistency.
        residual.extend(
            (
                ROOT_TORQUE_WEIGHT
                *
                w
                *
                dyn[
                    "root_torque"
                ][
                    frame
                ]
                /
                ROOT_TORQUE_SCALE
            ).tolist()
        )


        ratio = (
            np.abs(
                dyn[
                    "joint_tau"
                ][
                    frame
                ]
            )
            /
            effort
        )


        # --------------------------------------------------------
        # Soft pressure beginning at 60% authority.
        # --------------------------------------------------------

        soft = np.maximum(
            ratio - 0.60,
            0.0,
        )


        residual.extend(
            (
                TORQUE_SOFT_WEIGHT
                *
                w
                *
                soft
            ).tolist()
        )


        # --------------------------------------------------------
        # Strong penalty above the actual limit.
        # --------------------------------------------------------

        hard = np.maximum(
            ratio - 1.00,
            0.0,
        )


        residual.extend(
            (
                TORQUE_HARD_WEIGHT
                *
                w
                *
                hard
            ).tolist()
        )


    # ------------------------------------------------------------
    # Preserve original lower-body motion.
    # ------------------------------------------------------------

    residual.extend(
        (
            MOTION_REG_WEIGHT
            *
            normalized.reshape(
                -1
            )
        ).tolist()
    )


    # ------------------------------------------------------------
    # Similar gait corrections for repeated same-side phases.
    # Not equality — only mild regularization.
    # ------------------------------------------------------------

    for side in (
        0,
        1,
    ):

        indices = [
            i
            for i, segment in enumerate(
                segments
            )
            if segment[
                2
            ] == side
        ]


        for i in range(
            len(indices) - 1
        ):

            a = normalized[
                indices[i]
            ]

            b = normalized[
                indices[i+1]
            ]


            residual.extend(
                (
                    PHASE_SIMILARITY_WEIGHT
                    *
                    (
                        b - a
                    )
                ).tolist()
            )


    result = np.asarray(
        residual,
        dtype=np.float64,
    )


    if not np.all(
        np.isfinite(
            result
        )
    ):

        raise RuntimeError(
            "Non-finite objective."
        )


    if (
        objective_calls == 1
        or
        objective_calls % 20 == 0
    ):

        summary = summarize(
            dyn
        )


        print(
            f"call={objective_calls:04d} "
            f"root95={summary['overall']:.3f} "
            f"single={summary['single']:.3f} "
            f"trans={summary['transition']:.3f} "
            f"tau95={summary['tau95']:.3f} "
            f"over={100*summary['over']:.1f}%"
        )


    return result


# ================================================================
# OPTIMIZE
# ================================================================

print()
print("=" * 205)
print("RUNNING PASS-2B2 OPTIMIZATION")
print("=" * 205)


x0 = np.zeros(
    len(segments) * 4,
    dtype=np.float64,
)


result = least_squares(
    objective,

    x0=x0,

    bounds=(
        -1.0,
        1.0,
    ),

    method="trf",

    jac="2-point",

    diff_step=1e-3,

    loss="soft_l1",

    f_scale=1.0,

    x_scale="jac",

    max_nfev=MAX_NFEV,

    ftol=5e-5,
    xtol=5e-5,
    gtol=5e-5,

    verbose=1,
)


print()
print(
    "success:",
    result.success,
)

print(
    "status:",
    result.status,
)

print(
    "message:",
    result.message,
)

print(
    "nfev:",
    result.nfev,
)

print(
    "cost:",
    f"{result.cost:.5f}",
)


# ================================================================
# FINAL INVERSE RESULT
# ================================================================

(
    candidate_qpos,
    candidate_normalized,
    candidate_amplitudes,
) = build_candidate(
    result.x
)


candidate_dyn = dynamics(
    candidate_qpos
)


candidate_summary = summarize(
    candidate_dyn
)


candidate_qvel = differentiate_qpos(
    candidate_qpos
)


print()
print("=" * 205)
print("PASS-2B1 vs PASS-2B2 INVERSE DYNAMICS")
print("=" * 205)

print(
    f"{'REF':10s} "
    f"{'ROOT95':>9s} "
    f"{'SINGLE':>9s} "
    f"{'DOUBLE':>9s} "
    f"{'TRANS':>9s} "
    f"{'TAU95':>9s} "
    f"{'>LIMIT':>9s}"
)


for name, s in (
    (
        "PASS2B1",
        base_summary,
    ),
    (
        "PASS2B2",
        candidate_summary,
    ),
):

    print(
        f"{name:10s} "
        f"{s['overall']:9.3f} "
        f"{s['single']:9.3f} "
        f"{s['double']:9.3f} "
        f"{s['transition']:9.3f} "
        f"{s['tau95']:9.3f} "
        f"{100*s['over']:8.1f}%"
    )


print()
print("=" * 205)
print("LOWER-BODY CORRECTIONS")
print("=" * 205)


for i, (
    start,
    end,
    side,
) in enumerate(
    segments
):

    a = candidate_amplitudes[
        i
    ]


    print(
        f"segment={i} "
        f"{'L' if side == 0 else 'R'} "
        f"{start:03d}-{end:03d} "
        f"hipP={math.degrees(a[0]):+.2f}deg "
        f"hipR={math.degrees(a[1]):+.2f}deg "
        f"knee={math.degrees(a[2]):+.2f}deg "
        f"ankP={math.degrees(a[3]):+.2f}deg"
    )


# ================================================================
# SAVE CANDIDATE
# ================================================================

with np.load(
    BASE_CANDIDATE,
    allow_pickle=True,
) as f:

    payload = {
        key:
            f[key].copy()

        for key in f.files
    }


payload[
    "full_qpos"
] = candidate_qpos.astype(
    payload[
        "full_qpos"
    ].dtype
)


payload[
    "full_qvel"
] = candidate_qvel.astype(
    payload[
        "full_qvel"
    ].dtype
)


if "joint_pos_29" in payload:

    payload[
        "joint_pos_29"
    ] = candidate_qpos[
        :,
        7:36
    ].astype(
        payload[
            "joint_pos_29"
        ].dtype
    )


if "joint_vel_29" in payload:

    payload[
        "joint_vel_29"
    ] = candidate_qvel[
        :,
        6:35
    ].astype(
        payload[
            "joint_vel_29"
        ].dtype
    )


if "root_positions" in payload:

    payload[
        "root_positions"
    ] = candidate_qpos[
        :,
        0:3
    ].astype(
        payload[
            "root_positions"
        ].dtype
    )


if "root_quat_wxyz" in payload:

    payload[
        "root_quat_wxyz"
    ] = candidate_qpos[
        :,
        3:7
    ].astype(
        payload[
            "root_quat_wxyz"
        ].dtype
    )


payload[
    "dynamic_v2_pass"
] = np.asarray(
    "pass2b2_support_lowerbody"
)


payload[
    "pass2b2_amplitudes_rad"
] = (
    candidate_amplitudes.astype(
        np.float32
    )
)


np.savez_compressed(
    OUTPUT_REFERENCE,
    **payload,
)


# ================================================================
# FORWARD VALIDATION
# ================================================================

ZERO_ACTION = np.zeros(
    29,
    dtype=np.float32,
)


def quat_conj(q):

    return np.asarray(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ],
        dtype=np.float64,
    )


def quat_mul(a, b):

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.asarray(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )


def orientation_error_deg(
    reference,
    actual,
):

    reference = (
        reference
        /
        np.linalg.norm(
            reference
        )
    )


    actual = (
        actual
        /
        np.linalg.norm(
            actual
        )
    )


    q = quat_mul(
        reference,
        quat_conj(
            actual
        ),
    )


    if q[0] < 0:

        q = -q


    q = q / np.linalg.norm(q)


    angle = (
        2.0
        *
        math.atan2(
            np.linalg.norm(
                q[1:]
            ),
            max(
                q[0],
                1e-12,
            ),
        )
    )


    return math.degrees(
        abs(
            angle
        )
    )


def make_forward_env(
    mode,
):

    e = G129DofTrackingRestartV1(
        rsi=False,
        reset_joint_noise=0.0,
        reset_velocity_noise=0.0,
    )


    if mode == "ORIGINAL":

        return e


    if mode == "PASS2B1":

        qpos = base_qpos
        qvel = base_qvel


    elif mode == "PASS2B2":

        qpos = candidate_qpos
        qvel = candidate_qvel


    else:

        raise ValueError(
            mode
        )


    e.ref_full_qpos = qpos.copy()
    e.ref_full_qvel = qvel.copy()

    e.ref_root_pos = (
        qpos[
            :,
            0:3
        ].copy()
    )

    e.ref_root_quat = (
        qpos[
            :,
            3:7
        ].copy()
    )


    # Critical for Pass-2B2:
    # zero action must follow the NEW joint trajectory.
    e.ref_q = (
        qpos[
            :,
            7:36
        ].copy()
    )


    if hasattr(
        e,
        "ref_qd"
    ):

        if (
            isinstance(
                e.ref_qd,
                np.ndarray,
            )
            and
            e.ref_qd.shape
            ==
            qvel[
                :,
                6:35
            ].shape
        ):

            e.ref_qd = (
                qvel[
                    :,
                    6:35
                ].copy()
            )


    return e


def rollout(
    mode,
    start,
):

    e = make_forward_env(
        mode
    )


    (
        obs,
        info,
    ) = e.reset(
        options={
            "start_frame":
                int(start)
        }
    )


    max_steps = min(
        HORIZON,
        e.num_frames
        - 1
        - start,
    )


    steps = 0

    ori = 0.0
    root_error = 0.0
    qerror = 0.0

    min_up = float(
        e._up_z()
    )


    reason = ""


    while steps < max_steps:

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = e.step(
            ZERO_ACTION
        )


        steps += 1


        try:

            ref_frame = int(
                e._current_frame()
            )

        except Exception:

            ref_frame = min(
                start + steps,
                e.num_frames - 1,
            )


        ori += orientation_error_deg(
            e.ref_root_quat[
                ref_frame
            ],
            e.data.qpos[
                3:7
            ],
        )


        root_error += float(
            np.linalg.norm(
                e.data.qpos[
                    0:3
                ]
                -
                e.ref_root_pos[
                    ref_frame
                ]
            )
        )


        qerror += float(
            np.sqrt(
                np.mean(
                    (
                        e.data.qpos[
                            7:36
                        ]
                        -
                        e.ref_q[
                            ref_frame
                        ]
                    ) ** 2
                )
            )
        )


        min_up = min(
            min_up,
            float(
                e._up_z()
            ),
        )


        if terminated:

            reason = info.get(
                "termination_reason",
                "terminated",
            )

            break


        if truncated:

            reason = (
                "reference_end"
            )

            break


    if not reason:

        reason = "horizon"


    divisor = max(
        steps,
        1,
    )


    result = {

        "mode":
            mode,

        "start":
            start,

        "steps":
            steps,

        "max_steps":
            max_steps,

        "progress":
            steps
            /
            max_steps,

        "ori":
            ori
            /
            divisor,

        "root":
            root_error
            /
            divisor,

        "qerr":
            qerror
            /
            divisor,

        "minup":
            min_up,

        "reason":
            reason,
    }


    e.close()


    return result


print()
print("=" * 205)
print("FORWARD-PHYSICS VALIDATION")
print("=" * 205)


forward_results = []


for mode in (
    "ORIGINAL",
    "PASS2B1",
    "PASS2B2",
):

    for start in STARTS:

        row = rollout(
            mode,
            start,
        )


        forward_results.append(
            row
        )


        print(
            f"{mode:8s} "
            f"start={start:3d} "
            f"steps={row['steps']:3d}/"
            f"{row['max_steps']:<3d} "
            f"prog={row['progress']:.3f} "
            f"ori={row['ori']:6.2f} "
            f"root={row['root']:.3f} "
            f"qerr={row['qerr']:.3f} "
            f"minup={row['minup']:.3f} "
            f"{row['reason']}"
        )


# ================================================================
# BASELINE EXACT GATE
# ================================================================

expected = {
    0: 102,
    57: 51,
    114: 58,
}


baseline_exact = True


for start, expected_steps in (
    expected.items()
):

    row = next(
        r
        for r in forward_results
        if (
            r["mode"]
            ==
            "ORIGINAL"
            and
            r["start"]
            ==
            start
        )
    )


    if row[
        "steps"
    ] != expected_steps:

        baseline_exact = False


print()
print(
    "baseline exact 102/51/58:",
    baseline_exact,
)


if not baseline_exact:

    raise RuntimeError(
        "Forward baseline changed."
    )


# ================================================================
# AGGREGATE
# ================================================================

def aggregate(
    mode,
):

    rows = [
        r
        for r in forward_results
        if r[
            "mode"
        ] == mode
    ]


    return {

        "progress":
            float(
                np.mean(
                    [
                        r[
                            "progress"
                        ]
                        for r in rows
                    ]
                )
            ),

        "minimum":
            float(
                np.min(
                    [
                        r[
                            "progress"
                        ]
                        for r in rows
                    ]
                )
            ),

        "orientation":
            float(
                np.mean(
                    [
                        r[
                            "ori"
                        ]
                        for r in rows
                    ]
                )
            ),

        "root":
            float(
                np.mean(
                    [
                        r[
                            "root"
                        ]
                        for r in rows
                    ]
                )
            ),

        "qerr":
            float(
                np.mean(
                    [
                        r[
                            "qerr"
                        ]
                        for r in rows
                    ]
                )
            ),

        "minup":
            float(
                np.mean(
                    [
                        r[
                            "minup"
                        ]
                        for r in rows
                    ]
                )
            ),

        "done":
            int(
                np.sum(
                    [
                        r[
                            "progress"
                        ]
                        >= 0.999
                        for r in rows
                    ]
                )
            ),
    }


original_forward = aggregate(
    "ORIGINAL"
)


base_forward = aggregate(
    "PASS2B1"
)


new_forward = aggregate(
    "PASS2B2"
)


print()
print("=" * 205)
print("FORWARD AGGREGATE")
print("=" * 205)

print(
    f"{'REF':10s} "
    f"{'PROG':>8s} "
    f"{'MIN-P':>8s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'QERR':>8s} "
    f"{'MINUP':>8s} "
    f"{'DONE':>6s}"
)


for name, s in (
    (
        "ORIGINAL",
        original_forward,
    ),
    (
        "PASS2B1",
        base_forward,
    ),
    (
        "PASS2B2",
        new_forward,
    ),
):

    print(
        f"{name:10s} "
        f"{s['progress']:8.3f} "
        f"{s['minimum']:8.3f} "
        f"{s['orientation']:8.2f} "
        f"{s['root']:8.3f} "
        f"{s['qerr']:8.3f} "
        f"{s['minup']:8.3f} "
        f"{s['done']:6d}"
    )


gain_vs_original = (
    new_forward[
        "progress"
    ]
    -
    original_forward[
        "progress"
    ]
)


gain_vs_b1 = (
    new_forward[
        "progress"
    ]
    -
    base_forward[
        "progress"
    ]
)


print()
print("=" * 205)
print("FINAL PASS-2B2 VERDICT")
print("=" * 205)

print(
    "progress vs ORIGINAL:",
    f"{gain_vs_original:+.3f}",
)

print(
    "progress vs PASS2B1:",
    f"{gain_vs_b1:+.3f}",
)

print(
    "minimum progress:",
    f"{original_forward['minimum']:.3f}",
    "->",
    f"{new_forward['minimum']:.3f}",
)

print(
    "orientation:",
    f"{original_forward['orientation']:.2f}",
    "->",
    f"{new_forward['orientation']:.2f}",
)


if (
    gain_vs_original >= 0.08
    and
    candidate_summary[
        "over"
    ]
    <
    base_summary[
        "over"
    ]
):

    print()
    print(
        "RESULT: PASS-2B2 IS A MAJOR SUCCESS"
    )

    print(
        "Lower-body actuation makes the "
        "dynamics correction realizable."
    )

    print(
        "KEEP this candidate."
    )

    print(
        "NEXT: held-out forward validation before PPO."
    )


elif (
    gain_vs_original >= 0.04
    or
    (
        gain_vs_original >= 0.02
        and
        new_forward[
            "orientation"
        ]
        <
        original_forward[
            "orientation"
        ]
        - 2.0
        and
        new_forward[
            "minimum"
        ]
        >
        original_forward[
            "minimum"
        ]
    )
):

    print()
    print(
        "RESULT: PASS-2B2 IS MARGINALLY USEFUL"
    )

    print(
        "KEEP only as an intermediate candidate."
    )

    print(
        "NEXT: held-out validation; do not increase "
        "optimizer dimension."
    )


else:

    print()
    print(
        "RESULT: PASS-2B2 DOES NOT PRODUCE "
        "A MEANINGFUL FORWARD GAIN"
    )

    print(
        "STOP optimizing medium_02."
    )

    print(
        "NEXT: compare clean medium_04 "
        "using the same dynamic audit."
    )


# ================================================================
# SAVE DIAGNOSTIC
# ================================================================

np.savez_compressed(
    OUTPUT_DIAGNOSTIC,

    optimizer_x=
        result.x.astype(
            np.float32
        ),

    amplitudes=
        candidate_amplitudes.astype(
            np.float32
        ),

    pass2b1_root_bw=
        base_summary[
            "root_bw"
        ].astype(
            np.float32
        ),

    pass2b2_root_bw=
        candidate_summary[
            "root_bw"
        ].astype(
            np.float32
        ),

    original_forward_progress=
        np.asarray(
            [
                original_forward[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    pass2b1_forward_progress=
        np.asarray(
            [
                base_forward[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    pass2b2_forward_progress=
        np.asarray(
            [
                new_forward[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    baseline_exact=
        np.asarray(
            [
                baseline_exact
            ],
            dtype=np.bool_,
        ),
)


print()
print(
    "candidate:",
    OUTPUT_REFERENCE,
)

print(
    "diagnostic:",
    OUTPUT_DIAGNOSTIC,
)

print()
print(
    "PASS-2B1 ROOT TRAJECTORY WAS PRESERVED."
)

print(
    "ORIGINAL REFERENCE WAS NOT MODIFIED."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 205)


env.close()
