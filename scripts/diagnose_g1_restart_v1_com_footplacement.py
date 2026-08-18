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
    quat_error_vector,
)


# ================================================================
# CONFIG
# ================================================================

TRAIN_STARTS = [
    0,
    30,
    57,
    85,
    114,
    140,
]

HELDOUT_STARTS = [
    15,
    45,
    75,
    105,
    135,
]

ALL_STARTS = (
    TRAIN_STARTS
    + HELDOUT_STARTS
)

HORIZON = 120

CEM_POPULATION = 144
CEM_ELITE = 18
CEM_ITERATIONS = 6

INITIAL_STD = 0.60
MIN_STD = 0.055
PARAM_LIMIT = 3.0

SEED = 20260814


# ================================================================
# CONTROLLER PARAMETERIZATION
#
# Left sagittal:
#   hip pitch / knee / ankle pitch
#   3 x 9 = 27
#
# Right sagittal:
#   3 x 9 = 27
#
# Left frontal:
#   hip roll / ankle roll
#   2 x 6 = 12
#
# Right frontal:
#   2 x 6 = 12
#
# Waist pitch:
#   1 x 4 = 4
#
# Waist roll:
#   1 x 4 = 4
#
# Yaw:
#   hip common / hip differential / waist yaw
#   3 x 2 = 6
#
# TOTAL = 92
# ================================================================

LEFT_SAG_SHAPE = (3, 9)
RIGHT_SAG_SHAPE = (3, 9)

LEFT_FRONT_SHAPE = (2, 6)
RIGHT_FRONT_SHAPE = (2, 6)

WAIST_PITCH_SHAPE = (1, 4)
WAIST_ROLL_SHAPE = (1, 4)

YAW_SHAPE = (3, 2)


PARAM_DIM = (
    int(np.prod(LEFT_SAG_SHAPE))
    + int(np.prod(RIGHT_SAG_SHAPE))
    + int(np.prod(LEFT_FRONT_SHAPE))
    + int(np.prod(RIGHT_FRONT_SHAPE))
    + int(np.prod(WAIST_PITCH_SHAPE))
    + int(np.prod(WAIST_ROLL_SHAPE))
    + int(np.prod(YAW_SHAPE))
)


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


G = abs(
    float(
        env.model.opt.gravity[2]
    )
)


# ================================================================
# COM HELPERS
# ================================================================

_jac_com = np.zeros(
    (
        3,
        env.model.nv,
    ),
    dtype=np.float64,
)


def whole_body_com(
    data,
):

    # pelvis subtree == complete articulated G1.
    return data.subtree_com[
        env.pelvis_id
    ].copy()


def whole_body_com_velocity(
    data,
):

    _jac_com[:] = 0.0

    mujoco.mj_jacSubtreeCom(
        env.model,
        data,
        _jac_com,
        env.pelvis_id,
    )

    return (
        _jac_com
        @ data.qvel
    )


# ================================================================
# SUPPORT CENTER
# ================================================================

def support_center(
    left,
    right,
    flags,
):

    l = float(flags[0])
    r = float(flags[1])

    if (
        l > 0.5
        and
        r > 0.5
    ):
        return (
            0.5
            * (
                left
                + right
            )
        )

    if l > 0.5:
        return left.copy()

    if r > 0.5:
        return right.copy()

    # Flight / ambiguous transition fallback.
    return (
        0.5
        * (
            left
            + right
        )
    )


# ================================================================
# PRECOMPUTE REFERENCE COM / SUPPORT / FEET
# ================================================================

print("=" * 180)
print("PRECOMPUTING REFERENCE COM + SUPPORT GEOMETRY")
print("=" * 180)


cache = mujoco.MjData(
    env.model
)


N = env.num_frames


ref_com = np.zeros(
    (
        N,
        3,
    ),
    dtype=np.float64,
)

ref_com_vel = np.zeros_like(
    ref_com
)

ref_support_center = np.zeros_like(
    ref_com
)


ref_pelvis_pos = np.zeros_like(
    ref_com
)

ref_pelvis_R = np.zeros(
    (
        N,
        3,
        3,
    ),
    dtype=np.float64,
)


ref_left_local = np.zeros_like(
    ref_com
)

ref_right_local = np.zeros_like(
    ref_com
)


for frame in range(N):

    cache.qpos[:] = (
        env.ref_full_qpos[
            frame
        ]
    )

    cache.qvel[:] = (
        env.ref_full_qvel[
            frame
        ]
    )

    mujoco.mj_forward(
        env.model,
        cache,
    )

    ref_com[
        frame
    ] = whole_body_com(
        cache
    )

    ref_com_vel[
        frame
    ] = whole_body_com_velocity(
        cache
    )


    pelvis_pos = (
        cache.xpos[
            env.pelvis_id
        ].copy()
    )

    R = (
        cache.xmat[
            env.pelvis_id
        ].reshape(
            3,
            3,
        ).copy()
    )


    ref_pelvis_pos[
        frame
    ] = pelvis_pos

    ref_pelvis_R[
        frame
    ] = R


    left_world = (
        env.ref_left_foot_pos[
            frame
        ]
    )

    right_world = (
        env.ref_right_foot_pos[
            frame
        ]
    )


    ref_left_local[
        frame
    ] = (
        R.T
        @ (
            left_world
            - pelvis_pos
        )
    )


    ref_right_local[
        frame
    ] = (
        R.T
        @ (
            right_world
            - pelvis_pos
        )
    )


    support_flags = (
        env.ref_support[
            frame
        ]
    )


    # If stable support mask has no active foot,
    # use reference contact mask for this instant.
    if np.sum(
        support_flags
    ) < 0.5:

        support_flags = (
            env.ref_contact[
                frame
            ]
        )


    ref_support_center[
        frame
    ] = support_center(
        left_world,
        right_world,
        support_flags,
    )


print(
    "reference frames:",
    N,
)

print(
    "COM start:",
    np.round(
        ref_com[0],
        4,
    ),
)

print(
    "COM end:",
    np.round(
        ref_com[-1],
        4,
    ),
)


# ================================================================
# PARAMETER DECODER
# ================================================================

def decode(
    theta,
):

    theta = np.asarray(
        theta,
        dtype=np.float64,
    )

    if theta.shape != (
        PARAM_DIM,
    ):
        raise ValueError(
            theta.shape
        )


    offset = 0


    def take(shape):

        nonlocal offset

        count = int(
            np.prod(shape)
        )

        result = theta[
            offset:
            offset + count
        ].reshape(shape)

        offset += count

        return result


    return (
        take(
            LEFT_SAG_SHAPE
        ),
        take(
            RIGHT_SAG_SHAPE
        ),
        take(
            LEFT_FRONT_SHAPE
        ),
        take(
            RIGHT_FRONT_SHAPE
        ),
        take(
            WAIST_PITCH_SHAPE
        ),
        take(
            WAIST_ROLL_SHAPE
        ),
        take(
            YAW_SHAPE
        ),
    )


# ================================================================
# BALANCE STATE
# ================================================================

def balance_state():

    frame = int(
        env._current_frame
    )


    # ------------------------------------------------------------
    # Pelvis orientation state.
    # ------------------------------------------------------------

    orientation_error = (
        quat_error_vector(
            env.ref_root_quat[
                frame
            ],
            env.data.qpos[
                3:7
            ],
        )
    )


    angular_velocity_error = (
        env.data.qvel[
            3:6
        ]
        -
        env.ref_full_qvel[
            frame,
            3:6
        ]
    )


    pelvis_pos = (
        env.data.xpos[
            env.pelvis_id
        ].copy()
    )

    R = (
        env._root_rotation_matrix()
    )


    # ------------------------------------------------------------
    # COM.
    # ------------------------------------------------------------

    com = whole_body_com(
        env.data
    )

    com_vel = (
        whole_body_com_velocity(
            env.data
        )
    )


    # ------------------------------------------------------------
    # Current foot states.
    # ------------------------------------------------------------

    left_world = (
        env.data.site_xpos[
            env.left_foot_site
        ].copy()
    )

    right_world = (
        env.data.site_xpos[
            env.right_foot_site
        ].copy()
    )


    actual_contact = (
        env._actual_contacts()
    )


    ref_support = (
        env.ref_support[
            frame
        ]
    )


    # Actual support selection prefers real contact.
    actual_support_flags = (
        actual_contact.copy()
    )


    if np.sum(
        actual_support_flags
    ) < 0.5:

        actual_support_flags = (
            ref_support.copy()
        )


    actual_support = (
        support_center(
            left_world,
            right_world,
            actual_support_flags,
        )
    )


    target_support = (
        ref_support_center[
            frame
        ]
    )


    # ------------------------------------------------------------
    # COM relative to support.
    # Each is expressed in its own pelvis frame before comparison.
    # ------------------------------------------------------------

    actual_com_rel = (
        R.T
        @ (
            com
            - actual_support
        )
    )


    ref_R = (
        ref_pelvis_R[
            frame
        ]
    )


    target_com_rel = (
        ref_R.T
        @ (
            ref_com[
                frame
            ]
            - target_support
        )
    )


    com_support_error = (
        actual_com_rel
        - target_com_rel
    )


    # ------------------------------------------------------------
    # Extrapolated COM.
    # ------------------------------------------------------------

    actual_height = max(
        float(
            com[2]
            - actual_support[2]
        ),
        0.35,
    )


    ref_height = max(
        float(
            ref_com[
                frame,
                2
            ]
            - target_support[
                2
            ]
        ),
        0.35,
    )


    omega_actual = math.sqrt(
        G
        / actual_height
    )


    omega_ref = math.sqrt(
        G
        / ref_height
    )


    xcom = (
        com.copy()
    )

    xcom[
        :2
    ] += (
        com_vel[
            :2
        ]
        / omega_actual
    )


    target_xcom = (
        ref_com[
            frame
        ].copy()
    )

    target_xcom[
        :2
    ] += (
        ref_com_vel[
            frame,
            :2
        ]
        / omega_ref
    )


    actual_xcom_rel = (
        R.T
        @ (
            xcom
            - actual_support
        )
    )


    target_xcom_rel = (
        ref_R.T
        @ (
            target_xcom
            - target_support
        )
    )


    xcom_support_error = (
        actual_xcom_rel
        - target_xcom_rel
    )


    # ------------------------------------------------------------
    # Pelvis/root Z.
    # ------------------------------------------------------------

    root_z_error = float(
        env.data.qpos[2]
        -
        env.ref_root_pos[
            frame,
            2
        ]
    )


    root_z_velocity_error = float(
        env.data.qvel[2]
        -
        env.ref_full_qvel[
            frame,
            2
        ]
    )


    # ------------------------------------------------------------
    # Swing / stance foot placement error in pelvis frame.
    # ------------------------------------------------------------

    actual_left_local = (
        R.T
        @ (
            left_world
            - pelvis_pos
        )
    )


    actual_right_local = (
        R.T
        @ (
            right_world
            - pelvis_pos
        )
    )


    left_foot_error = (
        actual_left_local
        -
        ref_left_local[
            frame
        ]
    )


    right_foot_error = (
        actual_right_local
        -
        ref_right_local[
            frame
        ]
    )


    left_support_signal = max(
        float(
            ref_support[0]
        ),
        float(
            actual_contact[0]
        ),
    )


    right_support_signal = max(
        float(
            ref_support[1]
        ),
        float(
            actual_contact[1]
        ),
    )


    # ------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------

    left_sag = np.asarray(
        [
            orientation_error[1] / 0.35,
            angular_velocity_error[1] / 1.50,

            com_support_error[0] / 0.15,
            xcom_support_error[0] / 0.20,

            root_z_error / 0.12,
            root_z_velocity_error / 0.55,

            left_foot_error[0] / 0.12,
            left_foot_error[2] / 0.10,

            2.0 * left_support_signal - 1.0,
        ],
        dtype=np.float64,
    )


    right_sag = np.asarray(
        [
            orientation_error[1] / 0.35,
            angular_velocity_error[1] / 1.50,

            com_support_error[0] / 0.15,
            xcom_support_error[0] / 0.20,

            root_z_error / 0.12,
            root_z_velocity_error / 0.55,

            right_foot_error[0] / 0.12,
            right_foot_error[2] / 0.10,

            2.0 * right_support_signal - 1.0,
        ],
        dtype=np.float64,
    )


    left_front = np.asarray(
        [
            orientation_error[0] / 0.35,
            angular_velocity_error[0] / 1.50,

            com_support_error[1] / 0.12,
            xcom_support_error[1] / 0.16,

            left_foot_error[1] / 0.10,

            2.0 * left_support_signal - 1.0,
        ],
        dtype=np.float64,
    )


    right_front = np.asarray(
        [
            orientation_error[0] / 0.35,
            angular_velocity_error[0] / 1.50,

            com_support_error[1] / 0.12,
            xcom_support_error[1] / 0.16,

            right_foot_error[1] / 0.10,

            2.0 * right_support_signal - 1.0,
        ],
        dtype=np.float64,
    )


    waist_pitch = np.asarray(
        [
            orientation_error[1] / 0.35,
            angular_velocity_error[1] / 1.50,
            com_support_error[0] / 0.15,
            xcom_support_error[0] / 0.20,
        ],
        dtype=np.float64,
    )


    waist_roll = np.asarray(
        [
            orientation_error[0] / 0.35,
            angular_velocity_error[0] / 1.50,
            com_support_error[1] / 0.12,
            xcom_support_error[1] / 0.16,
        ],
        dtype=np.float64,
    )


    yaw = np.asarray(
        [
            orientation_error[2] / 0.35,
            angular_velocity_error[2] / 1.50,
        ],
        dtype=np.float64,
    )


    features = (
        left_sag,
        right_sag,
        left_front,
        right_front,
        waist_pitch,
        waist_roll,
        yaw,
    )


    features = tuple(
        np.clip(
            x,
            -2.5,
            2.5,
        )
        for x in features
    )


    diagnostics = {
        "com_error":
            float(
                np.linalg.norm(
                    com_support_error[
                        :2
                    ]
                )
            ),

        "xcom_error":
            float(
                np.linalg.norm(
                    xcom_support_error[
                        :2
                    ]
                )
            ),

        "com_x":
            float(
                com_support_error[0]
            ),

        "com_y":
            float(
                com_support_error[1]
            ),

        "xcom_x":
            float(
                xcom_support_error[0]
            ),

        "xcom_y":
            float(
                xcom_support_error[1]
            ),

        "left_support":
            left_support_signal,

        "right_support":
            right_support_signal,
    }


    return (
        features,
        diagnostics,
    )


# ================================================================
# CONTROLLER
# ================================================================

def feedback_action(
    theta,
):

    (
        left_sag_gain,
        right_sag_gain,
        left_front_gain,
        right_front_gain,
        waist_pitch_gain,
        waist_roll_gain,
        yaw_gain,
    ) = decode(
        theta
    )


    (
        features,
        diagnostics,
    ) = balance_state()


    (
        left_sag_f,
        right_sag_f,
        left_front_f,
        right_front_f,
        waist_pitch_f,
        waist_roll_f,
        yaw_f,
    ) = features


    left_sag_out = (
        left_sag_gain
        @ left_sag_f
    )

    right_sag_out = (
        right_sag_gain
        @ right_sag_f
    )


    left_front_out = (
        left_front_gain
        @ left_front_f
    )

    right_front_out = (
        right_front_gain
        @ right_front_f
    )


    waist_pitch_out = float(
        (
            waist_pitch_gain
            @ waist_pitch_f
        )[0]
    )


    waist_roll_out = float(
        (
            waist_roll_gain
            @ waist_roll_f
        )[0]
    )


    yaw_out = (
        yaw_gain
        @ yaw_f
    )


    left_support = (
        diagnostics[
            "left_support"
        ]
    )

    right_support = (
        diagnostics[
            "right_support"
        ]
    )


    # ------------------------------------------------------------
    # Role-dependent authority:
    #
    # Swing leg:
    #   more hip/knee placement authority.
    #
    # Stance leg:
    #   more ankle stabilization authority.
    # ------------------------------------------------------------

    left_swing_gate = (
        1.0
        - 0.25
        * left_support
    )

    right_swing_gate = (
        1.0
        - 0.25
        * right_support
    )


    left_ankle_gate = (
        0.45
        + 0.55
        * left_support
    )

    right_ankle_gate = (
        0.45
        + 0.55
        * right_support
    )


    action = np.zeros(
        29,
        dtype=np.float64,
    )


    # LEFT sagittal
    action[0] += (
        left_swing_gate
        * left_sag_out[0]
    )

    action[3] += (
        left_swing_gate
        * left_sag_out[1]
    )

    action[4] += (
        left_ankle_gate
        * left_sag_out[2]
    )


    # RIGHT sagittal
    action[6] += (
        right_swing_gate
        * right_sag_out[0]
    )

    action[9] += (
        right_swing_gate
        * right_sag_out[1]
    )

    action[10] += (
        right_ankle_gate
        * right_sag_out[2]
    )


    # LEFT frontal
    action[1] += (
        left_swing_gate
        * left_front_out[0]
    )

    action[5] += (
        left_ankle_gate
        * left_front_out[1]
    )


    # RIGHT frontal
    action[7] += (
        right_swing_gate
        * right_front_out[0]
    )

    action[11] += (
        right_ankle_gate
        * right_front_out[1]
    )


    # Waist
    action[14] += (
        waist_pitch_out
    )

    action[13] += (
        waist_roll_out
    )


    # Yaw
    hip_yaw_common = (
        yaw_out[0]
    )

    hip_yaw_diff = (
        yaw_out[1]
    )

    waist_yaw = (
        yaw_out[2]
    )


    action[2] += (
        left_swing_gate
        * (
            hip_yaw_common
            +
            hip_yaw_diff
        )
    )


    action[8] += (
        right_swing_gate
        * (
            hip_yaw_common
            -
            hip_yaw_diff
        )
    )


    action[12] += (
        waist_yaw
    )


    return np.tanh(
        action
    ).astype(
        np.float32
    )


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    theta,
    start,
):

    env.reset(
        options={
            "start_frame":
                int(start)
        }
    )


    remaining = (
        env.num_frames
        - 1
        - start
    )


    max_steps = min(
        HORIZON,
        remaining,
    )


    steps = 0

    min_up = env._up_z()

    ori_sum = 0.0

    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0

    com_sum = 0.0
    xcom_sum = 0.0

    q_sum = 0.0

    action_sum = 0.0

    clipped = 0
    action_values = 0

    reason = ""
    completed = False


    while steps < max_steps:

        if theta is None:

            action = np.zeros(
                29,
                dtype=np.float32,
            )

        else:

            action = feedback_action(
                theta
            )


        clipped += int(
            np.sum(
                np.abs(action)
                >= 0.999
            )
        )

        action_values += 29


        action_sum += float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        action,
                        dtype=np.float64,
                    ) ** 2
                )
            )
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        steps += 1


        frame = int(
            env._current_frame
        )


        root_error = (
            env.data.qpos[
                0:3
            ]
            -
            env.ref_root_pos[
                frame
            ]
        )


        x_sum += abs(
            float(
                root_error[0]
            )
        )

        y_sum += abs(
            float(
                root_error[1]
            )
        )

        z_sum += abs(
            float(
                root_error[2]
            )
        )


        (
            _,
            balance_diag,
        ) = balance_state()


        com_sum += float(
            balance_diag[
                "com_error"
            ]
        )


        xcom_sum += float(
            balance_diag[
                "xcom_error"
            ]
        )


        terms = info[
            "reward_terms"
        ]


        ori_sum += float(
            terms[
                "root_orientation_deg"
            ]
        )


        q_sum += float(
            terms[
                "q_error"
            ]
        )


        min_up = min(
            min_up,
            float(
                info["up"]
            ),
        )


        if terminated:

            reason = info[
                "termination_reason"
            ]

            break


        if truncated:

            reason = "reference_end"
            completed = True

            break


    if not reason:
        reason = "horizon"


    divisor = max(
        steps,
        1,
    )


    progress = (
        steps
        / max_steps
    )


    metrics = {
        "start":
            start,

        "steps":
            steps,

        "max_steps":
            max_steps,

        "progress":
            progress,

        "orientation":
            ori_sum / divisor,

        "min_up":
            min_up,

        "x":
            x_sum / divisor,

        "y":
            y_sum / divisor,

        "z":
            z_sum / divisor,

        "com":
            com_sum / divisor,

        "xcom":
            xcom_sum / divisor,

        "q":
            q_sum / divisor,

        "action":
            action_sum / divisor,

        "clip":
            clipped
            / max(
                action_values,
                1,
            ),

        "completed":
            completed,

        "reason":
            reason,
    }


    # ------------------------------------------------------------
    # Diagnostic objective.
    #
    # Survival dominates.
    # COM/support quality prevents "upright but dynamically wrong"
    # solutions from winning.
    # ------------------------------------------------------------

    score = (
        150.0
        * metrics[
            "progress"
        ]

        + 25.0
        * metrics[
            "min_up"
        ]

        - 0.20
        * metrics[
            "orientation"
        ]

        - 8.0
        * metrics[
            "x"
        ]

        - 14.0
        * metrics[
            "y"
        ]

        - 18.0
        * metrics[
            "z"
        ]

        - 18.0
        * metrics[
            "com"
        ]

        - 14.0
        * metrics[
            "xcom"
        ]

        - 2.0
        * metrics[
            "q"
        ]

        - 1.2
        * metrics[
            "action"
        ]

        - 30.0
        * metrics[
            "clip"
        ]
    )


    if completed:
        score += 25.0


    metrics[
        "score"
    ] = score


    return metrics


# ================================================================
# ZERO-ACTION BASELINE
# ================================================================

print()
print("=" * 180)
print("COM / FOOT-PLACEMENT ZERO-ACTION BASELINE")
print("=" * 180)


baseline = {}


for start in ALL_STARTS:

    r = rollout(
        None,
        start,
    )

    baseline[
        start
    ] = r


    print(
        f"start={start:03d} "
        f"steps={r['steps']:3d}/"
        f"{r['max_steps']:3d} "
        f"prog={r['progress']:.3f} "
        f"up={r['min_up']:.3f} "
        f"ori={r['orientation']:.2f} "
        f"COM={r['com']:.3f} "
        f"XCOM={r['xcom']:.3f} "
        f"x={r['x']:.3f} "
        f"y={r['y']:.3f} "
        f"reason={r['reason']}"
    )


# ================================================================
# CEM EVALUATION
# ================================================================

def evaluate(
    theta,
):

    rows = [
        rollout(
            theta,
            start,
        )
        for start
        in TRAIN_STARTS
    ]


    scores = np.asarray(
        [
            r["score"]
            for r in rows
        ],
        dtype=np.float64,
    )


    progress = np.asarray(
        [
            r["progress"]
            for r in rows
        ],
        dtype=np.float64,
    )


    # Encourage performance across the entire start distribution,
    # not only a single favorable section.
    robust_score = (
        float(
            np.mean(scores)
        )

        + 35.0
        * float(
            np.min(progress)
        )

        + 15.0
        * float(
            np.percentile(
                progress,
                25,
            )
        )
    )


    return (
        robust_score,
        rows,
    )


# ================================================================
# CEM
# ================================================================

print()
print("=" * 180)
print("COM + FOOT-PLACEMENT CEM")
print("TRAIN STARTS:", TRAIN_STARTS)
print("=" * 180)


rng = np.random.default_rng(
    SEED
)


mean = np.zeros(
    PARAM_DIM,
    dtype=np.float64,
)

std = np.full(
    PARAM_DIM,
    INITIAL_STD,
    dtype=np.float64,
)


best_theta = (
    mean.copy()
)

best_score = -np.inf


for iteration in range(
    1,
    CEM_ITERATIONS + 1,
):

    population = (
        mean[None, :]
        +
        std[None, :]
        * rng.standard_normal(
            (
                CEM_POPULATION,
                PARAM_DIM,
            )
        )
    )


    population = np.clip(
        population,
        -PARAM_LIMIT,
        PARAM_LIMIT,
    )


    scores = np.empty(
        CEM_POPULATION,
        dtype=np.float64,
    )

    mean_progress = np.empty(
        CEM_POPULATION,
        dtype=np.float64,
    )

    minimum_progress = np.empty(
        CEM_POPULATION,
        dtype=np.float64,
    )


    for i in range(
        CEM_POPULATION
    ):

        score, rows = evaluate(
            population[i]
        )

        scores[i] = score


        p = np.asarray(
            [
                r["progress"]
                for r in rows
            ],
            dtype=np.float64,
        )


        mean_progress[i] = (
            np.mean(p)
        )

        minimum_progress[i] = (
            np.min(p)
        )


    order = np.argsort(
        scores
    )[::-1]


    elites = population[
        order[
            :CEM_ELITE
        ]
    ]


    elite_mean = np.mean(
        elites,
        axis=0,
    )

    elite_std = np.std(
        elites,
        axis=0,
    )


    mean = (
        0.70
        * elite_mean
        +
        0.30
        * mean
    )


    std = (
        0.70
        * elite_std
        +
        0.30
        * std
    )


    std = np.maximum(
        std,
        MIN_STD,
    )


    idx = int(
        order[0]
    )


    if scores[idx] > best_score:

        best_score = float(
            scores[idx]
        )

        best_theta = (
            population[
                idx
            ].copy()
        )


    print(
        f"iter={iteration} "
        f"best={scores[idx]:+.2f} "
        f"eliteMean="
        f"{np.mean(scores[order[:CEM_ELITE]]):+.2f} "
        f"meanProg={mean_progress[idx]:.3f} "
        f"minProg={minimum_progress[idx]:.3f} "
        f"std={np.mean(std):.3f}"
    )


# ================================================================
# CONFIRMATION
# ================================================================

print()
print("=" * 190)
print("COM + FOOT-PLACEMENT CONTROLLER CONFIRMATION")
print("=" * 190)


best = {}


for start in ALL_STARTS:

    r = rollout(
        best_theta,
        start,
    )

    best[
        start
    ] = r


    b = baseline[
        start
    ]


    group = (
        "TRAIN"
        if start in TRAIN_STARTS
        else "HOLD"
    )


    print(
        f"{group:5s} "
        f"start={start:03d} "
        f"BASE={b['steps']:3d}/"
        f"{b['max_steps']:3d} "
        f"COM={r['steps']:3d}/"
        f"{r['max_steps']:3d} "
        f"delta={r['steps']-b['steps']:+4d} "
        f"prog={b['progress']:.3f}"
        f"->{r['progress']:.3f} "
        f"ori={b['orientation']:.2f}"
        f"->{r['orientation']:.2f} "
        f"COMerr={b['com']:.3f}"
        f"->{r['com']:.3f} "
        f"XCOM={b['xcom']:.3f}"
        f"->{r['xcom']:.3f} "
        f"up={b['min_up']:.3f}"
        f"->{r['min_up']:.3f} "
        f"clip={r['clip']:.3f} "
        f"reason={r['reason']}"
    )


# ================================================================
# AGGREGATION
# ================================================================

def aggregate(
    table,
    starts,
):

    rows = [
        table[
            s
        ]
        for s in starts
    ]


    def mean(key):

        return float(
            np.mean(
                [
                    x[key]
                    for x in rows
                ]
            )
        )


    return {
        "steps":
            mean("steps"),

        "progress":
            mean("progress"),

        "orientation":
            mean("orientation"),

        "min_up":
            mean("min_up"),

        "x":
            mean("x"),

        "y":
            mean("y"),

        "z":
            mean("z"),

        "com":
            mean("com"),

        "xcom":
            mean("xcom"),

        "clip":
            mean("clip"),

        "completed":
            int(
                sum(
                    int(
                        x["completed"]
                    )
                    for x in rows
                )
            ),
    }


base_all = aggregate(
    baseline,
    ALL_STARTS,
)

new_all = aggregate(
    best,
    ALL_STARTS,
)


base_train = aggregate(
    baseline,
    TRAIN_STARTS,
)

new_train = aggregate(
    best,
    TRAIN_STARTS,
)


base_hold = aggregate(
    baseline,
    HELDOUT_STARTS,
)

new_hold = aggregate(
    best,
    HELDOUT_STARTS,
)


print()
print("=" * 180)
print("COM + FOOT-PLACEMENT SUMMARY")
print("=" * 180)

print(
    f"ALL progress      : "
    f"{base_all['progress']:.3f}"
    f" -> "
    f"{new_all['progress']:.3f}"
)

print(
    f"TRAIN progress    : "
    f"{base_train['progress']:.3f}"
    f" -> "
    f"{new_train['progress']:.3f}"
)

print(
    f"HELD-OUT progress : "
    f"{base_hold['progress']:.3f}"
    f" -> "
    f"{new_hold['progress']:.3f}"
)

print(
    f"orientation       : "
    f"{base_all['orientation']:.2f}"
    f" -> "
    f"{new_all['orientation']:.2f}"
)

print(
    f"COM-support error : "
    f"{base_all['com']:.3f}"
    f" -> "
    f"{new_all['com']:.3f}"
)

print(
    f"XCoM error        : "
    f"{base_all['xcom']:.3f}"
    f" -> "
    f"{new_all['xcom']:.3f}"
)

print(
    f"X root error      : "
    f"{base_all['x']:.3f}"
    f" -> "
    f"{new_all['x']:.3f}"
)

print(
    f"Y root error      : "
    f"{base_all['y']:.3f}"
    f" -> "
    f"{new_all['y']:.3f}"
)

print(
    f"Z root error      : "
    f"{base_all['z']:.3f}"
    f" -> "
    f"{new_all['z']:.3f}"
)

print(
    f"mean min-up       : "
    f"{base_all['min_up']:.3f}"
    f" -> "
    f"{new_all['min_up']:.3f}"
)

print(
    f"clipping          : "
    f"{100*new_all['clip']:.2f}%"
)

print(
    f"completions       : "
    f"{base_all['completed']}"
    f" -> "
    f"{new_all['completed']}"
)


# ================================================================
# DECISION
# ================================================================

all_gain = (
    new_all["progress"]
    -
    base_all["progress"]
)


hold_gain = (
    new_hold["progress"]
    -
    base_hold["progress"]
)


orientation_change = (
    new_all["orientation"]
    -
    base_all["orientation"]
)


com_change = (
    new_all["com"]
    -
    base_all["com"]
)


xcom_change = (
    new_all["xcom"]
    -
    base_all["xcom"]
)


worst_regression = min(
    best[s]["steps"]
    -
    baseline[s]["steps"]

    for s in ALL_STARTS
)


print()
print("=" * 180)
print("COM / FOOT-PLACEMENT AUTHORITY DECISION")
print("=" * 180)

print(
    "all-start progress gain:",
    f"{all_gain:+.3f}",
)

print(
    "held-out progress gain:",
    f"{hold_gain:+.3f}",
)

print(
    "orientation change:",
    f"{orientation_change:+.2f} deg",
)

print(
    "COM-support error change:",
    f"{com_change:+.3f} m",
)

print(
    "XCoM error change:",
    f"{xcom_change:+.3f} m",
)

print(
    "worst per-start step regression:",
    f"{worst_regression:+d}",
)

print(
    "clipping:",
    f"{100*new_all['clip']:.2f}%",
)


strong = (
    all_gain >= 0.12
    and
    hold_gain >= 0.08
    and
    orientation_change <= 4.0
    and
    worst_regression >= -5
    and
    new_all["clip"] <= 0.08
)


marginal = (
    all_gain >= 0.07
    and
    hold_gain >= 0.04
    and
    orientation_change <= 7.0
    and
    worst_regression >= -10
)


print()


if strong:

    print(
        "COM / FOOT-PLACEMENT AUTHORITY: STRONG"
    )

    print(
        "The balance representation now generalizes "
        "across unseen motion phases."
    )

    print(
        "NEXT: promote COM/support/foot-placement "
        "features into the Gym environment observation."
    )

    print(
        "Then run one final environment validation "
        "before the first PPO pilot."
    )


elif marginal:

    print(
        "COM / FOOT-PLACEMENT AUTHORITY: MARGINAL"
    )

    print(
        "The representation helps but is still not "
        "robust enough for PPO."
    )

    print(
        "NEXT: inspect support transitions and "
        "swing-foot timing / phase adaptation."
    )


else:

    print(
        "COM / FOOT-PLACEMENT AUTHORITY: INSUFFICIENT"
    )

    print(
        "Do NOT train PPO."
    )

    print(
        "If this fails, fixed-time motion phase "
        "becomes the next primary suspect."
    )


# ================================================================
# SAVE DIAGNOSTIC
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_com_footplacement_best.npz"
)


np.savez(
    output,

    theta=
        best_theta.astype(
            np.float32
        ),

    train_starts=
        np.asarray(
            TRAIN_STARTS,
            dtype=np.int32,
        ),

    heldout_starts=
        np.asarray(
            HELDOUT_STARTS,
            dtype=np.int32,
        ),

    all_gain=
        np.asarray(
            [all_gain],
            dtype=np.float32,
        ),

    heldout_gain=
        np.asarray(
            [hold_gain],
            dtype=np.float32,
        ),

    orientation_change=
        np.asarray(
            [orientation_change],
            dtype=np.float32,
        ),

    com_change=
        np.asarray(
            [com_change],
            dtype=np.float32,
        ),

    xcom_change=
        np.asarray(
            [xcom_change],
            dtype=np.float32,
        ),
)


print()
print(
    "diagnostic controller:",
    output,
)

print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 180)


env.close()
