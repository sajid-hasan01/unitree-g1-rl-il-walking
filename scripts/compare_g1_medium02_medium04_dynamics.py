from pathlib import Path
import math
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

CONTACT_DEPTH = 1e-6
TRANSITION_WINDOW = 2
HORIZON = 120


BASE_STARTS = np.asarray(
    [
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
    ],
    dtype=np.int32,
)


ZERO_ACTION = np.zeros(
    29,
    dtype=np.float32,
)


OUTPUT = (
    ROOT
    / "results"
    / "g1_medium02_vs_medium04_dynamic_comparison.npz"
)


# ================================================================
# DEFAULT ENV = MEDIUM_02
# ================================================================

base_env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = base_env.model


MEDIUM02 = Path(
    base_env.reference_path
).resolve()


N02 = int(
    base_env.num_frames
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
    base_env.effort_limits,
    dtype=np.float64,
)


LEFT_GEOMS = list(
    base_env.left_sole_geoms
)


RIGHT_GEOMS = list(
    base_env.right_sole_geoms
)


LEFT_SET = set(
    LEFT_GEOMS
)


RIGHT_SET = set(
    RIGHT_GEOMS
)


print("=" * 205)
print("G1 CLEAN REFERENCE COMPARISON")
print("MEDIUM_02 vs MEDIUM_04")
print("SAME CONTACT-SNAPPED DYNAMIC AUDIT")
print("SAME ZERO-RESIDUAL FORWARD TEST")
print("NO OPTIMIZATION / NO CEM / NO PPO")
print("=" * 205)

print(
    "medium_02:",
    MEDIUM02,
)

print(
    "body weight:",
    f"{BODY_WEIGHT:.2f} N",
)


# ================================================================
# FIND MEDIUM_04
# ================================================================

candidates = []


for path in (
    ROOT
    / "datasets"
).rglob(
    "*.npz"
):

    name = path.name.lower()


    if (
        "medium_04"
        not in name
        and
        "medium04"
        not in name
    ):

        continue


    score = 0


    text = str(
        path
    ).lower()


    if (
        "validated_29dof_walks"
        in text
    ):

        score += 100


    if "grounded" in name:

        score += 20


    if "50hz" in name:

        score += 10


    if "candidate" in name:

        score -= 100


    if "dynamic_v2" in name:

        score -= 100


    candidates.append(
        (
            score,
            path.resolve(),
        )
    )


if not candidates:

    print()
    print(
        "Available medium files:"
    )


    for path in (
        ROOT
        / "datasets"
    ).rglob(
        "*.npz"
    ):

        if "medium" in path.name.lower():

            print(
                " ",
                path,
            )


    raise FileNotFoundError(
        "Could not locate clean medium_04 NPZ."
    )


candidates.sort(
    key=lambda x:
        (
            -x[0],
            str(
                x[1]
            ),
        )
)


print()
print(
    "medium_04 candidates:"
)


for score, path in candidates:

    print(
        f"score={score:4d}",
        path,
    )


MEDIUM04 = candidates[
    0
][1]


print()
print(
    "SELECTED medium_04:",
    MEDIUM04,
)


# ================================================================
# IDENTIFY SUPPORT / CONTACT KEYS FROM MEDIUM_02
# ================================================================

def find_matching_key(
    path,
    target,
    label,
):

    matches = []


    with np.load(
        path,
        allow_pickle=True,
    ) as f:

        for key in f.files:

            value = f[
                key
            ]


            if (
                not isinstance(
                    value,
                    np.ndarray,
                )
                or
                value.shape
                !=
                target.shape
                or
                not np.issubdtype(
                    value.dtype,
                    np.number,
                )
            ):

                continue


            if np.allclose(
                value,
                target,
                rtol=1e-6,
                atol=1e-7,
                equal_nan=True,
            ):

                matches.append(
                    key
                )


    if not matches:

        raise RuntimeError(
            f"Could not identify {label} "
            f"key in medium_02."
        )


    # Prefer names which semantically match.
    semantic = [
        k
        for k in matches
        if label.lower()
        in k.lower()
    ]


    return (
        semantic[0]
        if semantic
        else matches[0]
    )


support_key = find_matching_key(
    MEDIUM02,
    np.asarray(
        base_env.ref_support,
        dtype=np.float64,
    ),
    "support",
)


contact_key = find_matching_key(
    MEDIUM02,
    np.asarray(
        base_env.ref_contact,
        dtype=np.float64,
    ),
    "contact",
)


print()
print(
    "identified support key:",
    support_key,
)

print(
    "identified contact key:",
    contact_key,
)


# ================================================================
# REFERENCE LOADER
# ================================================================

def compute_qvel(
    qpos,
    fps,
):

    n = len(
        qpos
    )


    dt = (
        1.0
        /
        fps
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


def load_clip(
    label,
    path,
):

    with np.load(
        path,
        allow_pickle=True,
    ) as f:

        print()
        print("=" * 205)

        print(
            "LOADING",
            label,
        )

        print("=" * 205)

        print(
            "path:",
            path,
        )

        print(
            "keys:",
            list(
                f.files
            ),
        )


        if "full_qpos" not in f.files:

            raise RuntimeError(
                f"{label} has no full_qpos."
            )


        qpos = np.asarray(
            f[
                "full_qpos"
            ],
            dtype=np.float64,
        ).copy()


        if (
            qpos.ndim != 2
            or
            qpos.shape[
                1
            ] != model.nq
        ):

            raise RuntimeError(
                f"{label} qpos shape "
                f"{qpos.shape} != (*,{model.nq})"
            )


        fps = 50.0


        for key in (
            "fps",
            "reference_fps",
            "frame_rate",
            "mocap_fps",
        ):

            if key in f.files:

                value = np.asarray(
                    f[
                        key
                    ]
                ).reshape(
                    -1
                )


                if len(
                    value
                ) == 1:

                    candidate_fps = float(
                        value[
                            0
                        ]
                    )


                    if (
                        np.isfinite(
                            candidate_fps
                        )
                        and
                        10.0
                        <=
                        candidate_fps
                        <=
                        500.0
                    ):

                        fps = (
                            candidate_fps
                        )

                        break


        if "full_qvel" in f.files:

            qvel = np.asarray(
                f[
                    "full_qvel"
                ],
                dtype=np.float64,
            ).copy()

        else:

            qvel = compute_qvel(
                qpos,
                fps,
            )


        if qvel.shape != (
            len(qpos),
            model.nv,
        ):

            print(
                "full_qvel unavailable/incompatible;"
                " recomputing from qpos."
            )


            qvel = compute_qvel(
                qpos,
                fps,
            )


        if support_key not in f.files:

            raise RuntimeError(
                f"{label} does not contain "
                f"medium_02 support key "
                f"'{support_key}'."
            )


        if contact_key not in f.files:

            raise RuntimeError(
                f"{label} does not contain "
                f"medium_02 contact key "
                f"'{contact_key}'."
            )


        support = (
            np.asarray(
                f[
                    support_key
                ],
                dtype=np.float64,
            )
            > 0.5
        )


        contact = (
            np.asarray(
                f[
                    contact_key
                ],
                dtype=np.float64,
            )
            > 0.5
        )


    if support.shape != (
        len(qpos),
        2,
    ):

        raise RuntimeError(
            f"{label} support shape "
            f"{support.shape}"
        )


    if contact.shape != (
        len(qpos),
        2,
    ):

        raise RuntimeError(
            f"{label} contact shape "
            f"{contact.shape}"
        )


    print(
        "qpos:",
        qpos.shape,
    )

    print(
        "qvel:",
        qvel.shape,
    )

    print(
        "support:",
        support.shape,
    )

    print(
        "contact:",
        contact.shape,
    )

    print(
        "fps:",
        fps,
    )


    return {

        "label":
            label,

        "path":
            path,

        "qpos":
            qpos,

        "qvel":
            qvel,

        "support":
            support,

        "contact":
            contact,

        "fps":
            fps,

        "n":
            len(
                qpos
            ),
    }


clip02 = load_clip(
    "MEDIUM02",
    MEDIUM02,
)


clip04 = load_clip(
    "MEDIUM04",
    MEDIUM04,
)


# ================================================================
# DYNAMIC AUDIT
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
            base_env.floor_geom,
            2,
        ]
    )


    minimum = float(
        "inf"
    )


    for gid in geoms:

        radius = float(
            model.geom_size[
                gid,
                0,
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


def expected_support(
    clip,
    frame,
):

    expected = (
        clip[
            "support"
        ][
            frame
        ].copy()
    )


    if not np.any(
        expected
    ):

        expected = (
            clip[
                "contact"
            ][
                frame
            ].copy()
        )


    return expected


def choose_anchor(
    clip,
    frame,
):

    probe.qpos[:] = (
        clip[
            "qpos"
        ][
            frame
        ]
    )


    probe.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        probe,
    )


    left_clearance = sole_clearance(
        probe,
        LEFT_GEOMS,
    )


    right_clearance = sole_clearance(
        probe,
        RIGHT_GEOMS,
    )


    expected = expected_support(
        clip,
        frame,
    )


    candidates = []


    if expected[
        0
    ]:

        candidates.append(
            (
                0,
                left_clearance,
            )
        )


    if expected[
        1
    ]:

        candidates.append(
            (
                1,
                right_clearance,
            )
        )


    if not candidates:

        candidates = [
            (
                0,
                left_clearance,
            ),
            (
                1,
                right_clearance,
            ),
        ]


    return min(
        candidates,
        key=lambda x:
            x[
                1
            ],
    )


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


        if base_env.floor_geom not in (
            g1,
            g2,
        ):

            continue


        other = (
            g2
            if g1
            ==
            base_env.floor_geom
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


def differentiate_velocity(
    qvel,
    fps,
):

    dt = 1.0 / fps


    qacc = np.zeros_like(
        qvel
    )


    qacc[
        1:-1
    ] = (
        qvel[
            2:
        ]
        -
        qvel[
            :-2
        ]
    ) / (
        2.0
        *
        dt
    )


    qacc[
        0
    ] = (
        qvel[
            1
        ]
        -
        qvel[
            0
        ]
    ) / dt


    qacc[
        -1
    ] = (
        qvel[
            -1
        ]
        -
        qvel[
            -2
        ]
    ) / dt


    return qacc


def audit_clip(
    clip,
):

    n = clip[
        "n"
    ]


    fps = clip[
        "fps"
    ]


    qpos = clip[
        "qpos"
    ]


    support = clip[
        "support"
    ]


    support_count = np.sum(
        support,
        axis=1,
    )


    single = (
        support_count
        ==
        1
    )


    double = (
        support_count
        >=
        2
    )


    transition = np.zeros(
        n,
        dtype=bool,
    )


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


            transition[
                lo:hi
            ] = True


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


    snapped = qpos.copy()


    anchors = np.zeros(
        n,
        dtype=np.int8,
    )


    z_shift = np.zeros(
        n,
        dtype=np.float64,
    )


    for frame in range(n):

        (
            anchor,
            clearance,
        ) = choose_anchor(
            clip,
            frame,
        )


        anchors[
            frame
        ] = anchor


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
            -0.025,
            dz,
        )


        snapped[
            frame,
            2
        ] += dz


        z_shift[
            frame
        ] = dz


    qvel = compute_qvel(
        snapped,
        fps,
    )


    qacc = differentiate_velocity(
        qvel,
        fps,
    )


    data = mujoco.MjData(
        model
    )


    root_force = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float64,
    )


    joint_tau = np.zeros(
        (
            n,
            29,
        ),
        dtype=np.float64,
    )


    anchor_active = np.zeros(
        n,
        dtype=bool,
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


        (
            left_active,
            right_active,
        ) = active_contacts(
            data
        )


        anchor_active[
            frame
        ] = (
            left_active
            if anchors[
                frame
            ]
            ==
            0
            else right_active
        )


        root_force[
            frame
        ] = (
            data.qfrc_inverse[
                0:3
            ]
        )


        for j, vadr in enumerate(
            base_env.vaddrs
        ):

            joint_tau[
                frame,
                j
            ] = (
                data.qfrc_inverse[
                    vadr
                ]
            )


    root_bw = (
        np.linalg.norm(
            root_force,
            axis=1,
        )
        /
        BODY_WEIGHT
    )


    torque_ratio = (
        np.abs(
            joint_tau
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


    def safe_p95(
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
                    anchor_active
                )
            ),

        "overall50":
            float(
                np.percentile(
                    root_bw,
                    50,
                )
            ),

        "overall95":
            float(
                np.percentile(
                    root_bw,
                    95,
                )
            ),

        "maximum":
            float(
                np.max(
                    root_bw
                )
            ),

        "single95":
            safe_p95(
                steady_single
            ),

        "double95":
            safe_p95(
                steady_double
            ),

        "transition95":
            safe_p95(
                transition
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

        "shift50":
            float(
                np.percentile(
                    -1000.0
                    *
                    z_shift,
                    50,
                )
            ),

        "shift95":
            float(
                np.percentile(
                    -1000.0
                    *
                    z_shift,
                    95,
                )
            ),

        "steady_single_count":
            int(
                np.sum(
                    steady_single
                )
            ),

        "steady_double_count":
            int(
                np.sum(
                    steady_double
                )
            ),

        "transition_count":
            int(
                np.sum(
                    transition
                )
            ),
    }


print()
print("=" * 205)
print("CONSISTENT CONTACT-SNAPPED DYNAMIC AUDIT")
print("=" * 205)


audit02 = audit_clip(
    clip02
)


audit04 = audit_clip(
    clip04
)


print(
    f"{'CLIP':10s} "
    f"{'ANCH':>7s} "
    f"{'ROOT50':>9s} "
    f"{'ROOT95':>9s} "
    f"{'MAX':>9s} "
    f"{'SINGLE':>9s} "
    f"{'DOUBLE':>9s} "
    f"{'TRANS':>9s} "
    f"{'TAU95':>9s} "
    f"{'>LIMIT':>9s}"
)


for name, result in (
    (
        "MEDIUM02",
        audit02,
    ),
    (
        "MEDIUM04",
        audit04,
    ),
):

    print(
        f"{name:10s} "
        f"{100*result['anchor']:6.1f}% "
        f"{result['overall50']:9.3f} "
        f"{result['overall95']:9.3f} "
        f"{result['maximum']:9.3f} "
        f"{result['single95']:9.3f} "
        f"{result['double95']:9.3f} "
        f"{result['transition95']:9.3f} "
        f"{result['tau95']:9.3f} "
        f"{100*result['overlimit']:8.1f}%"
    )


print()
print(
    "medium_02 mask counts:",
    audit02[
        "steady_single_count"
    ],
    audit02[
        "steady_double_count"
    ],
    audit02[
        "transition_count"
    ],
)


print(
    "medium_04 mask counts:",
    audit04[
        "steady_single_count"
    ],
    audit04[
        "steady_double_count"
    ],
    audit04[
        "transition_count"
    ],
)


print(
    "root lowering p95 mm:",
    f"medium02={audit02['shift95']:.2f}",
    f"medium04={audit04['shift95']:.2f}",
)


# ================================================================
# MEDIUM02 BASELINE GATE
# ================================================================

if audit02[
    "anchor"
] < 0.95:

    raise RuntimeError(
        "Medium_02 contact audit no longer valid."
    )


if abs(
    audit02[
        "overall95"
    ]
    -
    11.255
) > 0.75:

    raise RuntimeError(
        "Medium_02 dynamic baseline changed: "
        f"{audit02['overall95']:.3f}"
    )


# ================================================================
# FORWARD ENV PATCHING
# ================================================================

def make_forward_env(
    clip,
):

    env = G129DofTrackingRestartV1(
        rsi=False,
        reset_joint_noise=0.0,
        reset_velocity_noise=0.0,
    )


    qpos = clip[
        "qpos"
    ]


    qvel = clip[
        "qvel"
    ]


    n = clip[
        "n"
    ]


    env.ref_full_qpos = (
        qpos.copy()
    )


    env.ref_full_qvel = (
        qvel.copy()
    )


    env.ref_root_pos = (
        qpos[
            :,
            0:3
        ].copy()
    )


    env.ref_root_quat = (
        qpos[
            :,
            3:7
        ].copy()
    )


    env.ref_q = (
        qpos[
            :,
            7:36
        ].copy()
    )


    if hasattr(
        env,
        "ref_qd"
    ):

        env.ref_qd = (
            qvel[
                :,
                6:35
            ].copy()
        )


    env.ref_support = (
        clip[
            "support"
        ].astype(
            np.float64
        )
    )


    env.ref_contact = (
        clip[
            "contact"
        ].astype(
            np.float64
        )
    )


    env.num_frames = int(
        n
    )


    env.reference_fps = float(
        clip[
            "fps"
        ]
    )


    env.reference_path = (
        clip[
            "path"
        ]
    )


    optional = {

        "ref_root_linvel":
            qvel[
                :,
                0:3
            ].copy(),

        "ref_root_angvel":
            qvel[
                :,
                3:6
            ].copy(),

        "ref_root_linear_velocity":
            qvel[
                :,
                0:3
            ].copy(),

        "ref_root_angular_velocity":
            qvel[
                :,
                3:6
            ].copy(),
    }


    for key, value in optional.items():

        if hasattr(
            env,
            key,
        ):

            current = getattr(
                env,
                key,
            )


            if (
                isinstance(
                    current,
                    np.ndarray,
                )
                and
                current.shape
                ==
                value.shape
            ):

                setattr(
                    env,
                    key,
                    value,
                )


    return env


# ================================================================
# QUAT ERROR
# ================================================================

def quat_conj(
    q,
):

    return np.asarray(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ],
        dtype=np.float64,
    )


def quat_mul(
    a,
    b,
):

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


    if q[
        0
    ] < 0.0:

        q = -q


    q = (
        q
        /
        np.linalg.norm(
            q
        )
    )


    angle = (
        2.0
        *
        math.atan2(
            np.linalg.norm(
                q[
                    1:4
                ]
            ),
            max(
                q[
                    0
                ],
                1e-12,
            ),
        )
    )


    return math.degrees(
        abs(
            angle
        )
    )


# ================================================================
# START-PHASE MAPPING
# ================================================================

phase_starts = (
    BASE_STARTS.astype(
        np.float64
    )
    /
    float(
        N02 - 1
    )
)


def starts_for_clip(
    clip,
):

    starts = np.rint(
        phase_starts
        *
        float(
            clip[
                "n"
            ]
            -
            1
        )
    ).astype(
        np.int32
    )


    # Preserve order and avoid accidental duplicates.
    result = []


    for start in starts:

        start = int(
            np.clip(
                start,
                0,
                clip[
                    "n"
                ]
                -
                2,
            )
        )


        if (
            not result
            or
            start
            !=
            result[
                -1
            ]
        ):

            result.append(
                start
            )


    return result


# ================================================================
# FORWARD ROLLOUT
# ================================================================

def rollout(
    clip,
    start,
):

    env = make_forward_env(
        clip
    )


    (
        obs,
        info,
    ) = env.reset(
        options={
            "start_frame":
                int(
                    start
                )
        }
    )


    reset_qpos_error = float(
        np.max(
            np.abs(
                env.data.qpos
                -
                clip[
                    "qpos"
                ][
                    start
                ]
            )
        )
    )


    reset_qvel_error = float(
        np.max(
            np.abs(
                env.data.qvel
                -
                clip[
                    "qvel"
                ][
                    start
                ]
            )
        )
    )


    if (
        reset_qpos_error
        >
        1e-6
        or
        reset_qvel_error
        >
        1e-6
    ):

        raise RuntimeError(
            f"{clip['label']} reset mismatch "
            f"at start {start}: "
            f"qpos={reset_qpos_error:.3e} "
            f"qvel={reset_qvel_error:.3e}"
        )


    max_steps = min(
        HORIZON,
        clip[
            "n"
        ]
        -
        1
        -
        start,
    )


    steps = 0

    orientation_sum = 0.0
    root_error_sum = 0.0
    qerror_sum = 0.0

    min_up = float(
        env._up_z()
    )


    reason = ""


    while steps < max_steps:

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            ZERO_ACTION
        )


        steps += 1


        try:

            frame = int(
                env._current_frame()
            )

        except Exception:

            frame = min(
                start
                +
                steps,
                clip[
                    "n"
                ]
                -
                1,
            )


        orientation_sum += (
            orientation_error_deg(
                env.ref_root_quat[
                    frame
                ],
                env.data.qpos[
                    3:7
                ],
            )
        )


        root_error_sum += float(
            np.linalg.norm(
                env.data.qpos[
                    0:3
                ]
                -
                env.ref_root_pos[
                    frame
                ]
            )
        )


        qerror_sum += float(
            np.sqrt(
                np.mean(
                    (
                        env.data.qpos[
                            7:36
                        ]
                        -
                        env.ref_q[
                            frame
                        ]
                    )
                    **
                    2
                )
            )
        )


        min_up = min(
            min_up,
            float(
                env._up_z()
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

        reason = (
            "horizon"
        )


    divisor = max(
        steps,
        1,
    )


    row = {

        "start":
            start,

        "steps":
            steps,

        "max_steps":
            max_steps,

        "progress":
            (
                steps
                /
                max_steps
            ),

        "orientation":
            (
                orientation_sum
                /
                divisor
            ),

        "root":
            (
                root_error_sum
                /
                divisor
            ),

        "qerror":
            (
                qerror_sum
                /
                divisor
            ),

        "minup":
            min_up,

        "reason":
            reason,
    }


    env.close()


    return row


def forward_clip(
    clip,
):

    starts = starts_for_clip(
        clip
    )


    rows = []


    print()
    print("=" * 205)

    print(
        "FORWARD:",
        clip[
            "label"
        ],
    )

    print("=" * 205)


    print(
        "phase-mapped starts:",
        starts,
    )


    for start in starts:

        row = rollout(
            clip,
            start,
        )


        rows.append(
            row
        )


        print(
            f"{clip['label']:8s} "
            f"start={start:3d} "
            f"steps={row['steps']:3d}/"
            f"{row['max_steps']:<3d} "
            f"prog={row['progress']:.3f} "
            f"ori={row['orientation']:6.2f} "
            f"root={row['root']:.3f} "
            f"qerr={row['qerror']:.3f} "
            f"minup={row['minup']:.3f} "
            f"{row['reason']}"
        )


    def mean(
        key,
    ):

        return float(
            np.mean(
                [
                    row[
                        key
                    ]
                    for row in rows
                ]
            )
        )


    return {

        "starts":
            np.asarray(
                starts,
                dtype=np.int32,
            ),

        "rows":
            rows,

        "progress":
            mean(
                "progress"
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
            mean(
                "orientation"
            ),

        "root":
            mean(
                "root"
            ),

        "qerror":
            mean(
                "qerror"
            ),

        "minup":
            mean(
                "minup"
            ),

        "done":
            int(
                np.sum(
                    [
                        r[
                            "progress"
                        ]
                        >=
                        0.999
                        for r in rows
                    ]
                )
            ),
    }


forward02 = forward_clip(
    clip02
)


forward04 = forward_clip(
    clip04
)


# ================================================================
# MEDIUM02 FORWARD BASELINE GATE
# ================================================================

expected = {
    0:
        102,

    57:
        51,

    114:
        58,
}


baseline_exact = True


for start, expected_steps in expected.items():

    matching = [
        row
        for row in forward02[
            "rows"
        ]
        if row[
            "start"
        ]
        ==
        start
    ]


    if (
        not matching
        or
        matching[
            0
        ][
            "steps"
        ]
        !=
        expected_steps
    ):

        baseline_exact = False


print()
print(
    "medium_02 baseline exact 102/51/58:",
    baseline_exact,
)


if not baseline_exact:

    raise RuntimeError(
        "Medium_02 forward baseline changed."
    )


# ================================================================
# COMPARISON
# ================================================================

print()
print("=" * 205)
print("CLEAN CLIP COMPARISON")
print("=" * 205)

print()
print("DYNAMIC FEASIBILITY")

print(
    f"{'CLIP':10s} "
    f"{'ROOT95':>10s} "
    f"{'SINGLE':>10s} "
    f"{'DOUBLE':>10s} "
    f"{'TRANS':>10s} "
    f"{'TAU95':>10s} "
    f"{'>LIMIT':>10s}"
)


for name, result in (
    (
        "MEDIUM02",
        audit02,
    ),
    (
        "MEDIUM04",
        audit04,
    ),
):

    print(
        f"{name:10s} "
        f"{result['overall95']:10.3f} "
        f"{result['single95']:10.3f} "
        f"{result['double95']:10.3f} "
        f"{result['transition95']:10.3f} "
        f"{result['tau95']:10.3f} "
        f"{100*result['overlimit']:9.1f}%"
    )


print()
print("FORWARD PHYSICS")

print(
    f"{'CLIP':10s} "
    f"{'PROG':>9s} "
    f"{'MIN-P':>9s} "
    f"{'ORI':>9s} "
    f"{'ROOT':>9s} "
    f"{'QERR':>9s} "
    f"{'MINUP':>9s} "
    f"{'DONE':>6s}"
)


for name, result in (
    (
        "MEDIUM02",
        forward02,
    ),
    (
        "MEDIUM04",
        forward04,
    ),
):

    print(
        f"{name:10s} "
        f"{result['progress']:9.3f} "
        f"{result['minimum']:9.3f} "
        f"{result['orientation']:9.2f} "
        f"{result['root']:9.3f} "
        f"{result['qerror']:9.3f} "
        f"{result['minup']:9.3f} "
        f"{result['done']:6d}"
    )


# ================================================================
# RATIOS / DECISION
# ================================================================

dynamic_ratio = (
    audit04[
        "overall95"
    ]
    /
    max(
        audit02[
            "overall95"
        ],
        1e-9,
    )
)


single_ratio = (
    audit04[
        "single95"
    ]
    /
    max(
        audit02[
            "single95"
        ],
        1e-9,
    )
)


tau_ratio = (
    audit04[
        "tau95"
    ]
    /
    max(
        audit02[
            "tau95"
        ],
        1e-9,
    )
)


forward_gain = (
    forward04[
        "progress"
    ]
    -
    forward02[
        "progress"
    ]
)


print()
print("=" * 205)
print("MEDIUM_04 DECISION")
print("=" * 205)

print(
    "dynamic root-p95 ratio 04/02:",
    f"{dynamic_ratio:.3f}",
)

print(
    "single-support ratio 04/02:",
    f"{single_ratio:.3f}",
)

print(
    "torque-p95 ratio 04/02:",
    f"{tau_ratio:.3f}",
)

print(
    "forward progress:",
    f"{forward02['progress']:.3f}",
    "->",
    f"{forward04['progress']:.3f}",
    f"({forward_gain:+.3f})",
)


print()


if (
    audit04[
        "anchor"
    ]
    <
    0.95
):

    print(
        "RESULT: MEDIUM_04 CONTACT AUDIT INVALID"
    )

    print(
        "Do not compare dynamics until contact "
        "validity is fixed."
    )


elif (
    dynamic_ratio
    <=
    0.50

    and
    single_ratio
    <=
    0.50

    and
    tau_ratio
    <=
    0.75
):

    print(
        "RESULT: MEDIUM_04 IS DYNAMICALLY "
        "MUCH CLEANER THAN MEDIUM_02"
    )

    print(
        "The severe medium_02 mismatch is "
        "substantially clip-specific."
    )


    if forward_gain >= 0.05:

        print(
            "MEDIUM_04 ALSO IMPROVES FORWARD PHYSICS."
        )

        print(
            "NEXT: promote medium_04 as the new "
            "reference candidate and perform its "
            "full geometry/contact/controller gate."
        )

    else:

        print(
            "Inverse dynamics is cleaner but forward "
            "survival is not yet materially better."
        )

        print(
            "NEXT: perform one small root-dynamics "
            "projection on medium_04 before deciding."
        )


elif (
    dynamic_ratio
    <=
    0.75

    or
    single_ratio
    <=
    0.75
):

    print(
        "RESULT: MEDIUM_04 IS MODERATELY CLEANER"
    )

    print(
        "Clip selection matters."
    )

    print(
        "NEXT: compare medium_08 before choosing "
        "the new base clip."
    )


elif (
    0.75
    <=
    dynamic_ratio
    <=
    1.35

    and
    0.75
    <=
    single_ratio
    <=
    1.35
):

    print(
        "RESULT: MEDIUM_04 SHOWS THE SAME "
        "DYNAMIC FAILURE CLASS"
    )

    print(
        "The issue is likely broader than medium_02."
    )

    print(
        "STOP clip-specific optimization."
    )

    print(
        "NEXT: audit the retargeting/root trajectory "
        "pipeline itself, and optionally confirm "
        "with medium_08."
    )


else:

    print(
        "RESULT: MEDIUM_04 IS NOT CLEANER"
    )

    print(
        "Do not switch references based on "
        "kinematic quality alone."
    )

    print(
        "NEXT: test medium_08 or repair the "
        "retargeting/root dynamics pipeline."
    )


# ================================================================
# SAVE
# ================================================================

np.savez_compressed(
    OUTPUT,

    medium02_path=
        np.asarray(
            str(
                MEDIUM02
            )
        ),

    medium04_path=
        np.asarray(
            str(
                MEDIUM04
            )
        ),

    medium02_dynamic=
        np.asarray(
            [
                audit02[
                    "overall95"
                ],
                audit02[
                    "single95"
                ],
                audit02[
                    "transition95"
                ],
                audit02[
                    "tau95"
                ],
                audit02[
                    "overlimit"
                ],
            ],
            dtype=np.float32,
        ),

    medium04_dynamic=
        np.asarray(
            [
                audit04[
                    "overall95"
                ],
                audit04[
                    "single95"
                ],
                audit04[
                    "transition95"
                ],
                audit04[
                    "tau95"
                ],
                audit04[
                    "overlimit"
                ],
            ],
            dtype=np.float32,
        ),

    medium02_forward=
        np.asarray(
            [
                forward02[
                    "progress"
                ],
                forward02[
                    "minimum"
                ],
                forward02[
                    "orientation"
                ],
                forward02[
                    "root"
                ],
                forward02[
                    "qerror"
                ],
            ],
            dtype=np.float32,
        ),

    medium04_forward=
        np.asarray(
            [
                forward04[
                    "progress"
                ],
                forward04[
                    "minimum"
                ],
                forward04[
                    "orientation"
                ],
                forward04[
                    "root"
                ],
                forward04[
                    "qerror"
                ],
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
    "artifact:",
    OUTPUT,
)

print()
print(
    "NO REFERENCE FILE WAS MODIFIED."
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


base_env.close()
