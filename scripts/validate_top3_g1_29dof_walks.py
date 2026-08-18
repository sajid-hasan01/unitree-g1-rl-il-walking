from pathlib import Path
import math

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

SOURCE_DIR = (
    ROOT
    / "datasets"
    / "candidate_29dof_walks_kit"
)

OUT_DIR = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
)


CANDIDATES = {
    "medium_02":
        SOURCE_DIR / "medium_02.npz",

    "medium_08":
        SOURCE_DIR / "medium_08.npz",

    "medium_04":
        SOURCE_DIR / "medium_04.npz",
}


TARGET_FPS = 50.0

TARGET_MIN_CLEARANCE = 0.003

CONTACT_CLEARANCE = 0.015

SUPPORT_SPEED = 0.30

SMOOTH_HALF_WINDOW = 4

MAX_Z_CHANGE_PER_FRAME = 0.004


FAIL_HEIGHT = 0.45
FAIL_UP = 0.40


# ================================================================
# QUATERNION HELPERS
# wxyz throughout
# ================================================================

def qnormalize(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return q / max(
        np.linalg.norm(q),
        1e-12,
    )


def qmul(a, b):

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


def qslerp(q0, q1, alpha):

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)

    dot = float(
        np.dot(q0, q1)
    )

    if dot < 0:

        q1 = -q1
        dot = -dot


    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )


    if dot > 0.9995:

        return qnormalize(
            q0
            + alpha
            * (
                q1 - q0
            )
        )


    theta = math.acos(dot)

    sin_theta = math.sin(theta)

    a = (
        math.sin(
            (1-alpha)
            * theta
        )
        / sin_theta
    )

    b = (
        math.sin(
            alpha
            * theta
        )
        / sin_theta
    )


    return qnormalize(
        a*q0
        + b*q1
    )


def yaw_from_quat(q):

    w, x, y, z = qnormalize(q)

    return math.atan2(
        2.0 * (
            w*z + x*y
        ),

        1.0
        - 2.0 * (
            y*y + z*z
        ),
    )


def quat_angle_deg(a, b):

    a = qnormalize(a)
    b = qnormalize(b)

    dot = abs(
        float(
            np.dot(a, b)
        )
    )

    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        2.0
        * math.acos(dot)
    )


# ================================================================
# MODEL
# ================================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)


sim_dt = float(
    model.opt.timestep
)


frame_skip = int(
    round(
        1.0
        / (
            TARGET_FPS
            * sim_dt
        )
    )
)


print("=" * 170)
print("TOP-3 29-DOF G1 WALK VALIDATION")
print("EXACT MUJOCO SOLE GEOMETRY + NATIVE SERVO")
print("NO PPO")
print("=" * 170)

print(
    "model nq/nv/nu:",
    model.nq,
    model.nv,
    model.nu,
)

print(
    "simulation dt:",
    sim_dt,
)

print(
    "50-Hz frame skip:",
    frame_skip,
)


# ================================================================
# MODEL JOINT ORDER
# ================================================================

model_names = []

joint_ids = []

qaddrs = []

vaddrs = []

actuator_ids = []


for aid in range(
    model.nu
):

    jid = int(
        model.actuator_trnid[
            aid,
            0,
        ]
    )


    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        jid,
    )


    model_names.append(
        str(name)
    )

    joint_ids.append(
        jid
    )

    actuator_ids.append(
        aid
    )


    qaddrs.append(
        int(
            model.jnt_qposadr[
                jid
            ]
        )
    )

    vaddrs.append(
        int(
            model.jnt_dofadr[
                jid
            ]
        )
    )


if len(model_names) != 29:

    raise RuntimeError(
        "Expected 29 actuated G1 joints."
    )


# ================================================================
# FOOT GEOMETRY
# ================================================================

left_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_foot",
)

right_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_foot",
)

floor_gid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor",
)


if min(
    left_site,
    right_site,
    floor_gid,
) < 0:

    raise RuntimeError(
        "Required foot/floor geometry not found."
    )


left_body = int(
    model.site_bodyid[
        left_site
    ]
)

right_body = int(
    model.site_bodyid[
        right_site
    ]
)


floor_z = float(
    model.geom_pos[
        floor_gid,
        2,
    ]
)


def get_sole_geoms(
    body_id,
):

    result = []


    for gid in range(
        model.ngeom
    ):

        if (
            int(
                model.geom_bodyid[
                    gid
                ]
            )
            != body_id
        ):

            continue


        if (
            int(
                model.geom_type[
                    gid
                ]
            )
            != int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            )
        ):

            continue


        radius = float(
            model.geom_size[
                gid,
                0,
            ]
        )


        if radius <= 0.010:

            result.append(
                gid
            )


    return sorted(
        result
    )


left_sole = get_sole_geoms(
    left_body
)

right_sole = get_sole_geoms(
    right_body
)


print()
print(
    "left true sole spheres:",
    left_sole,
)

print(
    "right true sole spheres:",
    right_sole,
)


if len(left_sole) != 4:

    raise RuntimeError(
        "Expected four left sole spheres."
    )


if len(right_sole) != 4:

    raise RuntimeError(
        "Expected four right sole spheres."
    )


# ================================================================
# TEMPLATE QPOS
# ================================================================

if model.nkey > 0:

    template_qpos = (
        model.key_qpos[
            0
        ].copy()
    )

else:

    template_qpos = np.zeros(
        model.nq,
        dtype=np.float64,
    )

    template_qpos[2] = 0.80
    template_qpos[3] = 1.0


# ================================================================
# SOLE CLEARANCE
# ================================================================

def sole_clearance(
    data,
    geom_ids,
):

    values = []


    for gid in geom_ids:

        z = float(
            data.geom_xpos[
                gid,
                2,
            ]
        )

        radius = float(
            model.geom_size[
                gid,
                0,
            ]
        )


        values.append(
            z
            - radius
            - floor_z
        )


    return float(
        min(values)
    )


# ================================================================
# BUILD ONE CANDIDATE
# ================================================================

def build_reference(
    label,
    source_path,
):

    print()
    print("=" * 170)
    print(
        "BUILDING:",
        label,
    )
    print("=" * 170)


    raw = np.load(
        source_path,
        allow_pickle=True,
    )


    names = [
        str(x)
        for x in raw[
            "dof_names"
        ]
    ]


    if names != model_names:

        print(
            "source names:",
            names,
        )

        print(
            "model names:",
            model_names,
        )

        raise RuntimeError(
            f"{label}: 29-DOF order mismatch."
        )


    raw_q = np.asarray(
        raw["dof_positions"],
        dtype=np.float64,
    )


    body_names = [
        str(x)
        for x in raw[
            "body_names"
        ]
    ]


    pelvis_index = (
        body_names.index(
            "pelvis"
        )
    )


    raw_root = np.asarray(
        raw["body_positions"],
        dtype=np.float64,
    )[
        :,
        pelvis_index,
        :
    ]


    raw_quat = np.asarray(
        raw["body_rotations"],
        dtype=np.float64,
    )[
        :,
        pelvis_index,
        :
    ]


    source_fps = float(
        np.asarray(
            raw["fps"]
        ).reshape(-1)[0]
    )


    n_source = len(
        raw_q
    )


    duration = (
        (n_source - 1)
        / source_fps
    )


    n_target = (
        int(
            round(
                duration
                * TARGET_FPS
            )
        )
        + 1
    )


    source_t = np.linspace(
        0.0,
        duration,
        n_source,
    )


    target_t = np.linspace(
        0.0,
        duration,
        n_target,
    )


    # ============================================================
    # RESAMPLE JOINT POSITIONS
    # ============================================================

    q = np.zeros(
        (
            n_target,
            29,
        ),
        dtype=np.float64,
    )


    for j in range(29):

        q[:, j] = np.interp(
            target_t,
            source_t,
            raw_q[:, j],
        )


    # ============================================================
    # RESAMPLE ROOT POSITION
    # ============================================================

    root_raw = np.zeros(
        (
            n_target,
            3,
        ),
        dtype=np.float64,
    )


    for axis in range(3):

        root_raw[
            :,
            axis,
        ] = np.interp(
            target_t,
            source_t,
            raw_root[
                :,
                axis,
            ],
        )


    # ============================================================
    # ROOT QUAT SLERP
    # ============================================================

    root_quat_source = np.zeros(
        (
            n_target,
            4,
        ),
        dtype=np.float64,
    )


    for k, t in enumerate(
        target_t
    ):

        source_position = (
            t * source_fps
        )


        i0 = int(
            np.floor(
                source_position
            )
        )


        i0 = int(
            np.clip(
                i0,
                0,
                n_source - 1,
            )
        )


        i1 = min(
            i0 + 1,
            n_source - 1,
        )


        alpha = float(
            source_position
            - i0
        )


        root_quat_source[
            k
        ] = qslerp(
            raw_quat[
                i0
            ],
            raw_quat[
                i1
            ],
            alpha,
        )


    # ============================================================
    # AUTOMATIC FORWARD ALIGNMENT
    #
    # Whatever direction this KIT clip travels becomes +X.
    # ============================================================

    travel = (
        root_raw[
            -1,
            :2,
        ]
        -
        root_raw[
            0,
            :2,
        ]
    )


    travel_heading = math.atan2(
        travel[1],
        travel[0],
    )


    c = math.cos(
        travel_heading
    )

    s = math.sin(
        travel_heading
    )


    relative = (
        root_raw
        - root_raw[
            0
        ]
    )


    root = np.zeros_like(
        root_raw
    )


    root[:, 0] = (
        c * relative[:, 0]
        +
        s * relative[:, 1]
    )


    root[:, 1] = (
        -s * relative[:, 0]
        +
        c * relative[:, 1]
    )


    # Preserve actual source pelvis Z.
    root[:, 2] = (
        root_raw[:, 2]
    )


    # world rotation by -travel_heading
    half = (
        -travel_heading
        / 2.0
    )


    q_align = np.asarray(
        [
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ],
        dtype=np.float64,
    )


    root_quat = np.zeros_like(
        root_quat_source
    )


    for i in range(
        n_target
    ):

        root_quat[
            i
        ] = qnormalize(
            qmul(
                q_align,
                root_quat_source[
                    i
                ],
            )
        )


        if (
            i > 0
            and np.dot(
                root_quat[
                    i-1
                ],
                root_quat[
                    i
                ],
            ) < 0
        ):

            root_quat[
                i
            ] *= -1.0


    # ============================================================
    # FK QPOS HELPER
    # ============================================================

    def frame_qpos(
        frame,
        root_position,
    ):

        result = (
            template_qpos.copy()
        )


        result[
            0:3
        ] = root_position[
            frame
        ]


        result[
            3:7
        ] = root_quat[
            frame
        ]


        for j, qadr in enumerate(
            qaddrs
        ):

            result[
                qadr
            ] = q[
                frame,
                j,
            ]


        mujoco.mj_normalizeQuat(
            model,
            result,
        )


        return result


    # ============================================================
    # RAW SOLE CLEARANCE
    # ============================================================

    data = mujoco.MjData(
        model
    )


    left_raw_clear = np.zeros(
        n_target,
        dtype=np.float64,
    )


    right_raw_clear = np.zeros(
        n_target,
        dtype=np.float64,
    )


    for i in range(
        n_target
    ):

        data.qpos[:] = frame_qpos(
            i,
            root,
        )

        data.qvel[:] = 0.0


        mujoco.mj_forward(
            model,
            data,
        )


        left_raw_clear[
            i
        ] = sole_clearance(
            data,
            left_sole,
        )


        right_raw_clear[
            i
        ] = sole_clearance(
            data,
            right_sole,
        )


    lowest = np.minimum(
        left_raw_clear,
        right_raw_clear,
    )


    required_dz = (
        TARGET_MIN_CLEARANCE
        - lowest
    )


    # ============================================================
    # V4 TRIANGULAR SMOOTHING
    # ============================================================

    h = SMOOTH_HALF_WINDOW


    kernel = np.concatenate(
        [
            np.arange(
                1,
                h + 2,
                dtype=np.float64,
            ),

            np.arange(
                h,
                0,
                -1,
                dtype=np.float64,
            ),
        ]
    )


    kernel /= np.sum(
        kernel
    )


    padded = np.pad(
        required_dz,
        (
            h,
            h,
        ),
        mode="edge",
    )


    smooth = np.convolve(
        padded,
        kernel,
        mode="valid",
    )


    dz = np.maximum(
        smooth,
        required_dz,
    )


    # ============================================================
    # FORWARD/BACKWARD RATE ENVELOPE
    # Only raises neighbouring samples.
    # ============================================================

    for _ in range(5):

        for i in range(
            1,
            n_target,
        ):

            minimum = (
                dz[
                    i-1
                ]
                -
                MAX_Z_CHANGE_PER_FRAME
            )


            if dz[i] < minimum:

                dz[i] = minimum


        for i in range(
            n_target - 2,
            -1,
            -1,
        ):

            minimum = (
                dz[
                    i+1
                ]
                -
                MAX_Z_CHANGE_PER_FRAME
            )


            if dz[i] < minimum:

                dz[i] = minimum


        dz = np.maximum(
            dz,
            required_dz,
        )


    grounded_root = (
        root.copy()
    )


    grounded_root[
        :,
        2,
    ] += dz


    # ============================================================
    # FULL QPOS
    # ============================================================

    full_qpos = np.zeros(
        (
            n_target,
            model.nq,
        ),
        dtype=np.float64,
    )


    for i in range(
        n_target
    ):

        full_qpos[
            i
        ] = frame_qpos(
            i,
            grounded_root,
        )


    # ============================================================
    # QVEL VIA MUJOCO
    # ============================================================

    dt = (
        1.0
        / TARGET_FPS
    )


    full_qvel = np.zeros(
        (
            n_target,
            model.nv,
        ),
        dtype=np.float64,
    )


    mujoco.mj_differentiatePos(
        model,
        full_qvel[
            0
        ],
        dt,
        full_qpos[
            0
        ],
        full_qpos[
            1
        ],
    )


    for i in range(
        1,
        n_target - 1,
    ):

        mujoco.mj_differentiatePos(
            model,
            full_qvel[
                i
            ],

            2.0 * dt,

            full_qpos[
                i-1
            ],

            full_qpos[
                i+1
            ],
        )


    mujoco.mj_differentiatePos(
        model,
        full_qvel[
            -1
        ],
        dt,
        full_qpos[
            -2
        ],
        full_qpos[
            -1
        ],
    )


    joint_vel = np.zeros(
        (
            n_target,
            29,
        ),
        dtype=np.float64,
    )


    for j, vadr in enumerate(
        vaddrs
    ):

        joint_vel[
            :,
            j,
        ] = full_qvel[
            :,
            vadr,
        ]


    # ============================================================
    # EXACT FINAL FOOT GEOMETRY
    # ============================================================

    left_clear = np.zeros(
        n_target,
        dtype=np.float64,
    )

    right_clear = np.zeros(
        n_target,
        dtype=np.float64,
    )


    left_pos = np.zeros(
        (
            n_target,
            3,
        ),
        dtype=np.float64,
    )

    right_pos = np.zeros(
        (
            n_target,
            3,
        ),
        dtype=np.float64,
    )


    for i in range(
        n_target
    ):

        data.qpos[:] = full_qpos[
            i
        ]

        data.qvel[:] = full_qvel[
            i
        ]


        mujoco.mj_forward(
            model,
            data,
        )


        left_clear[
            i
        ] = sole_clearance(
            data,
            left_sole,
        )


        right_clear[
            i
        ] = sole_clearance(
            data,
            right_sole,
        )


        left_pos[
            i
        ] = data.site_xpos[
            left_site
        ]


        right_pos[
            i
        ] = data.site_xpos[
            right_site
        ]


    left_vel = np.gradient(
        left_pos,
        dt,
        axis=0,
    )


    right_vel = np.gradient(
        right_pos,
        dt,
        axis=0,
    )


    left_speed = np.linalg.norm(
        left_vel[
            :,
            :2,
        ],
        axis=1,
    )


    right_speed = np.linalg.norm(
        right_vel[
            :,
            :2,
        ],
        axis=1,
    )


    left_contact = (
        left_clear
        <= CONTACT_CLEARANCE
    )


    right_contact = (
        right_clear
        <= CONTACT_CLEARANCE
    )


    left_support = (
        left_contact
        &
        (
            left_speed
            <= SUPPORT_SPEED
        )
    )


    right_support = (
        right_contact
        &
        (
            right_speed
            <= SUPPORT_SPEED
        )
    )


    contact_mask = np.column_stack(
        [
            left_contact,
            right_contact,
        ]
    ).astype(
        np.float32
    )


    support_mask = np.column_stack(
        [
            left_support,
            right_support,
        ]
    ).astype(
        np.float32
    )


    # ============================================================
    # GEOMETRY METRICS
    # ============================================================

    all_clear = np.concatenate(
        [
            left_clear,
            right_clear,
        ]
    )


    contact_coverage = float(
        np.mean(
            left_contact
            |
            right_contact
        )
    )


    support_coverage = float(
        np.mean(
            left_support
            |
            right_support
        )
    )


    double_contact = float(
        np.mean(
            left_contact
            &
            right_contact
        )
    )


    no_contact = float(
        np.mean(
            ~(
                left_contact
                |
                right_contact
            )
        )
    )


    left_support_fraction = float(
        np.mean(
            left_support
        )
    )


    right_support_fraction = float(
        np.mean(
            right_support
        )
    )


    support_symmetry = (

        min(
            left_support_fraction,
            right_support_fraction,
        )

        /
        max(
            left_support_fraction,
            right_support_fraction,
            1e-8,
        )
    )


    stance_speeds = np.concatenate(
        [
            left_speed[
                left_support
            ],

            right_speed[
                right_support
            ],
        ]
    )


    if len(
        stance_speeds
    ):

        stance_slip_median = float(
            np.median(
                stance_speeds
            )
        )

        stance_slip_p95 = float(
            np.percentile(
                stance_speeds,
                95,
            )
        )

    else:

        stance_slip_median = 999.0
        stance_slip_p95 = 999.0


    left_swing = left_clear[
        ~left_contact
    ]

    right_swing = right_clear[
        ~right_contact
    ]


    left_swing95 = float(
        np.percentile(
            left_swing,
            95,
        )
        if len(left_swing)
        else 0.0
    )


    right_swing95 = float(
        np.percentile(
            right_swing,
            95,
        )
        if len(right_swing)
        else 0.0
    )


    swing_symmetry = (

        min(
            left_swing95,
            right_swing95,
        )

        /
        max(
            left_swing95,
            right_swing95,
            1e-8,
        )
    )


    yaw_values = np.unwrap(
        np.asarray(
            [
                yaw_from_quat(x)
                for x in root_quat
            ]
        )
    )


    yaw_range_deg = math.degrees(
        float(
            np.ptp(
                yaw_values
            )
        )
    )


    # ============================================================
    # SAVE REFERENCE
    # ============================================================

    output = (
        OUT_DIR
        /
        (
            label
            + "_50hz_grounded.npz"
        )
    )


    np.savez(
        output,

        fps=
            np.asarray(
                [TARGET_FPS],
                dtype=np.float32,
            ),

        source_fps=
            np.asarray(
                [source_fps],
                dtype=np.float32,
            ),

        source_file=
            np.asarray(
                [str(source_path)]
            ),

        dof_names=
            np.asarray(
                names
            ),

        joint_pos_29=
            q.astype(
                np.float32
            ),

        joint_vel_29=
            joint_vel.astype(
                np.float32
            ),

        full_qpos=
            full_qpos.astype(
                np.float32
            ),

        full_qvel=
            full_qvel.astype(
                np.float32
            ),

        root_positions=
            grounded_root.astype(
                np.float32
            ),

        root_quat_wxyz=
            root_quat.astype(
                np.float32
            ),

        left_foot_pos=
            left_pos.astype(
                np.float32
            ),

        right_foot_pos=
            right_pos.astype(
                np.float32
            ),

        left_foot_vel=
            left_vel.astype(
                np.float32
            ),

        right_foot_vel=
            right_vel.astype(
                np.float32
            ),

        left_sole_clearance=
            left_clear.astype(
                np.float32
            ),

        right_sole_clearance=
            right_clear.astype(
                np.float32
            ),

        contact_mask=
            contact_mask,

        support_mask=
            support_mask,

        root_z_correction=
            dz.astype(
                np.float32
            ),

        travel_heading_source_rad=
            np.asarray(
                [travel_heading],
                dtype=np.float32,
            ),
    )


    geometry = {

        "label":
            label,

        "output":
            output,

        "frames":
            n_target,

        "duration":
            duration,

        "travel":
            float(
                grounded_root[
                    -1,
                    0,
                ]
                -
                grounded_root[
                    0,
                    0,
                ]
            ),

        "lateral":
            float(
                grounded_root[
                    -1,
                    1,
                ]
                -
                grounded_root[
                    0,
                    1,
                ]
            ),

        "yaw_range":
            yaw_range_deg,

        "dz_min":
            float(
                np.min(dz)
            ),

        "dz_med":
            float(
                np.median(dz)
            ),

        "dz_max":
            float(
                np.max(dz)
            ),

        "dz_rate":
            float(
                np.max(
                    np.abs(
                        np.diff(dz)
                    )
                )
            ),

        "penetration":
            int(
                np.sum(
                    all_clear
                    < -0.001
                )
            ),

        "contact":
            contact_coverage,

        "support":
            support_coverage,

        "double_contact":
            double_contact,

        "no_contact":
            no_contact,

        "left_support":
            left_support_fraction,

        "right_support":
            right_support_fraction,

        "support_symmetry":
            support_symmetry,

        "slip_median":
            stance_slip_median,

        "slip_p95":
            stance_slip_p95,

        "left_swing":
            left_swing95,

        "right_swing":
            right_swing95,

        "swing_symmetry":
            swing_symmetry,

        "q":
            q,

        "qvel":
            joint_vel,

        "full_qpos":
            full_qpos,

        "full_qvel":
            full_qvel,

        "root":
            grounded_root,

        "quat":
            root_quat,
    }


    geometry[
        "exact_pass"
    ] = bool(

        geometry[
            "penetration"
        ] == 0

        and geometry[
            "contact"
        ] >= 0.65

        and geometry[
            "support"
        ] >= 0.45

        and geometry[
            "support_symmetry"
        ] >= 0.60

        and geometry[
            "dz_max"
        ] <= 0.12
    )


    return geometry


# ================================================================
# BUILD ALL THREE
# ================================================================

references = {}


for label, path in (
    CANDIDATES.items()
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    references[
        label
    ] = build_reference(
        label,
        path,
    )


# ================================================================
# GEOMETRY TABLE
# ================================================================

print()
print("=" * 195)
print("EXACT MUJOCO GROUNDING / CONTACT RESULTS")
print("=" * 195)

print(
    f"{'NAME':10s} "
    f"{'PASS':>5s} "
    f"{'SEC':>6s} "
    f"{'FWD':>7s} "
    f"{'YAW':>7s} "
    f"{'DZ-MED':>8s} "
    f"{'DZ-MAX':>8s} "
    f"{'PEN':>4s} "
    f"{'CONTACT':>8s} "
    f"{'SUPPORT':>8s} "
    f"{'L-SUP':>7s} "
    f"{'R-SUP':>7s} "
    f"{'SUP-SYM':>7s} "
    f"{'SLIP50':>7s} "
    f"{'SLIP95':>7s} "
    f"{'L-SWING':>8s} "
    f"{'R-SWING':>8s}"
)

print("-" * 195)


for label in CANDIDATES:

    g = references[
        label
    ]


    print(
        f"{label:10s} "
        f"{str(g['exact_pass']):>5s} "
        f"{g['duration']:6.2f} "
        f"{g['travel']:7.3f} "
        f"{g['yaw_range']:7.2f} "
        f"{1000*g['dz_med']:8.1f} "
        f"{1000*g['dz_max']:8.1f} "
        f"{g['penetration']:4d} "
        f"{100*g['contact']:7.1f}% "
        f"{100*g['support']:7.1f}% "
        f"{100*g['left_support']:6.1f}% "
        f"{100*g['right_support']:6.1f}% "
        f"{g['support_symmetry']:7.3f} "
        f"{g['slip_median']:7.3f} "
        f"{g['slip_p95']:7.3f} "
        f"{1000*g['left_swing']:8.1f} "
        f"{1000*g['right_swing']:8.1f}"
    )


# ================================================================
# OPEN-LOOP NATIVE SERVO SCREEN
#
# This is not expected to solve walking.
# It only asks which clean motion is dynamically least hostile.
# ================================================================

pelvis_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pelvis",
)


def up_z(data):

    return float(
        data.xmat[
            pelvis_id
        ].reshape(
            3,
            3,
        )[2, 2]
    )


def failed(data):

    return (
        float(
            data.qpos[2]
        ) < FAIL_HEIGHT

        or up_z(
            data
        ) < FAIL_UP
    )


def actual_joint_q(data):

    return np.asarray(
        [
            data.qpos[
                qadr
            ]
            for qadr
            in qaddrs
        ],
        dtype=np.float64,
    )


def rollout(
    ref,
    start,
):

    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    data.qpos[:] = ref[
        "full_qpos"
    ][
        start
    ]


    data.qvel[:] = ref[
        "full_qvel"
    ][
        start
    ]


    mujoco.mj_forward(
        model,
        data,
    )


    current = int(
        start
    )


    steps = 0

    minimum_up = up_z(
        data
    )


    ori2 = 0.0
    qerr2 = 0.0
    root2 = 0.0


    while True:

        nxt = min(
            current + 1,
            len(
                ref[
                    "q"
                ]
            ) - 1,
        )


        for _ in range(
            frame_skip
        ):

            for j, aid in enumerate(
                actuator_ids
            ):

                data.ctrl[
                    aid
                ] = ref[
                    "q"
                ][
                    nxt,
                    j,
                ]


            mujoco.mj_step(
                model,
                data,
            )


        current = nxt
        steps += 1


        minimum_up = min(
            minimum_up,
            up_z(
                data
            ),
        )


        ori = quat_angle_deg(
            data.qpos[
                3:7
            ],

            ref[
                "quat"
            ][
                current
            ],
        )


        qerr = float(
            np.sqrt(
                np.mean(
                    (
                        actual_joint_q(
                            data
                        )
                        -
                        ref[
                            "q"
                        ][
                            current
                        ]
                    ) ** 2
                )
            )
        )


        root_error = float(
            np.linalg.norm(
                data.qpos[
                    0:3
                ]
                -
                ref[
                    "root"
                ][
                    current
                ]
            )
        )


        ori2 += ori ** 2
        qerr2 += qerr ** 2
        root2 += root_error ** 2


        if failed(
            data
        ):

            done = False
            break


        if current >= (
            len(
                ref[
                    "q"
                ]
            )
            - 1
        ):

            done = True
            break


    remaining = max(
        len(
            ref[
                "q"
            ]
        )
        - 1
        - start,
        1,
    )


    return {

        "steps":
            steps,

        "done":
            done,

        "progress":
            min(
                1.0,
                steps / remaining,
            ),

        "min_up":
            minimum_up,

        "ori":
            math.sqrt(
                ori2
                / max(
                    steps,
                    1,
                )
            ),

        "qerr":
            math.sqrt(
                qerr2
                / max(
                    steps,
                    1,
                )
            ),

        "root":
            math.sqrt(
                root2
                / max(
                    steps,
                    1,
                )
            ),
    }


dynamic_results = {}


print()
print("=" * 170)
print("NATIVE POSITION-SERVO OPEN-LOOP SCREEN")
print("NO PPO")
print("=" * 170)


for label in CANDIDATES:

    ref = references[
        label
    ]


    n = len(
        ref["q"]
    )


    starts = sorted(
        set(
            [
                0,

                int(
                    round(
                        0.25
                        * (
                            n - 1
                        )
                    )
                ),

                int(
                    round(
                        0.50
                        * (
                            n - 1
                        )
                    )
                ),
            ]
        )
    )


    rows = []


    for start in starts:

        r = rollout(
            ref,
            start,
        )


        rows.append(
            r
        )


        print(
            f"{label:10s} "
            f"start={start:03d} "
            f"steps={r['steps']:3d} "
            f"done={r['done']} "
            f"prog={r['progress']:.3f} "
            f"minUp={r['min_up']:.3f} "
            f"ori={r['ori']:.2f} "
            f"qerr={r['qerr']:.3f} "
            f"root={r['root']:.3f}"
        )


    dynamic_results[
        label
    ] = {

        "done":
            sum(
                int(
                    x["done"]
                )
                for x in rows
            ),

        "progress":
            float(
                np.mean(
                    [
                        x["progress"]
                        for x in rows
                    ]
                )
            ),

        "orientation":
            float(
                np.mean(
                    [
                        x["ori"]
                        for x in rows
                    ]
                )
            ),

        "qerr":
            float(
                np.mean(
                    [
                        x["qerr"]
                        for x in rows
                    ]
                )
            ),

        "root":
            float(
                np.mean(
                    [
                        x["root"]
                        for x in rows
                    ]
                )
            ),

        "min_up":
            float(
                np.mean(
                    [
                        x["min_up"]
                        for x in rows
                    ]
                )
            ),
    }


# ================================================================
# DYNAMIC SUMMARY
# ================================================================

print()
print("=" * 140)
print("DYNAMIC SCREEN SUMMARY")
print("=" * 140)

print(
    f"{'NAME':10s} "
    f"{'DONE':>5s} "
    f"{'PROG':>7s} "
    f"{'ORI':>8s} "
    f"{'QERR':>7s} "
    f"{'ROOT':>7s} "
    f"{'MINUP':>7s}"
)

print("-" * 140)


for label in CANDIDATES:

    r = dynamic_results[
        label
    ]


    print(
        f"{label:10s} "
        f"{r['done']:5d} "
        f"{r['progress']:7.3f} "
        f"{r['orientation']:8.2f} "
        f"{r['qerr']:7.3f} "
        f"{r['root']:7.3f} "
        f"{r['min_up']:7.3f}"
    )


# ================================================================
# FINAL SELECTION
#
# Priority:
#
# 1. clean exact geometry
# 2. open-loop normalized progress
# 3. support quality
# 4. lower orientation error
# 5. lower required root correction
#
# ================================================================

def ranking_key(label):

    g = references[
        label
    ]

    d = dynamic_results[
        label
    ]


    return (

        int(
            g[
                "exact_pass"
            ]
        ),

        d[
            "done"
        ],

        d[
            "progress"
        ],

        g[
            "support"
        ],

        g[
            "support_symmetry"
        ],

        -d[
            "orientation"
        ],

        -g[
            "dz_max"
        ],
    )


ordered = sorted(
    CANDIDATES.keys(),
    key=ranking_key,
    reverse=True,
)


print()
print("=" * 170)
print("FINAL CLEAN-WALK DATASET SELECTION")
print("=" * 170)


for rank, label in enumerate(
    ordered,
    1,
):

    g = references[
        label
    ]

    d = dynamic_results[
        label
    ]


    print(
        f"{rank}. {label}"
    )

    print(
        f"   exact geometry pass : "
        f"{g['exact_pass']}"
    )

    print(
        f"   forward travel      : "
        f"{g['travel']:.3f} m"
    )

    print(
        f"   contact coverage    : "
        f"{100*g['contact']:.1f}%"
    )

    print(
        f"   support coverage    : "
        f"{100*g['support']:.1f}%"
    )

    print(
        f"   support symmetry    : "
        f"{g['support_symmetry']:.3f}"
    )

    print(
        f"   stance slip median  : "
        f"{g['slip_median']:.3f} m/s"
    )

    print(
        f"   max root correction : "
        f"{1000*g['dz_max']:.1f} mm"
    )

    print(
        f"   dynamic progress    : "
        f"{d['progress']:.3f}"
    )

    print(
        f"   dynamic orientation : "
        f"{d['orientation']:.2f} deg"
    )

    print(
        f"   reference file      : "
        f"{g['output']}"
    )

    print()


winner = ordered[
    0
]


wg = references[
    winner
]


print(
    "WINNER:",
    winner,
)

print(
    "WINNER REFERENCE:",
    wg[
        "output"
    ],
)


if not wg[
    "exact_pass"
]:

    print()
    print(
        "FINAL DECISION: NONE OF THE TOP-3 "
        "PASSES EXACT CONTACT VALIDATION."
    )

    print(
        "Do NOT train yet."
    )

else:

    print()
    print(
        "FINAL DECISION: CLEAN 29-DOF "
        "FORWARD-WALK REFERENCE FOUND."
    )

    print(
        "This is the reference to use for "
        "the clean controller restart."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 170)
