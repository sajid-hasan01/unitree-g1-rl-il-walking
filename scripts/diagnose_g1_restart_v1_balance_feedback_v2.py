from pathlib import Path
import math
import sys

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
    57,
    114,
]

HELDOUT_STARTS = [
    30,
    85,
]

EXTRA_HELDOUT_STARTS = [
    15,
    140,
]

ALL_STARTS = (
    TRAIN_STARTS
    + HELDOUT_STARTS
    + EXTRA_HELDOUT_STARTS
)


HORIZON = 120

CEM_POPULATION = 128
CEM_ELITE = 16
CEM_ITERATIONS = 6

INITIAL_STD = 0.65
MIN_STD = 0.06

PARAM_LIMIT = 3.0

SEED = 20260814


# ================================================================
# PARAMETERIZATION
#
# Left sagittal:
# 3 joints x 7 features = 21
#
# Right sagittal:
# 3 joints x 7 features = 21
#
# Waist pitch:
# 2 features = 2
#
# Frontal:
# 5 outputs x 4 features = 20
#
# Yaw:
# 3 outputs x 2 features = 6
#
# TOTAL = 70
# ================================================================

LEFT_SAG_SHAPE = (
    3,
    7,
)

RIGHT_SAG_SHAPE = (
    3,
    7,
)

WAIST_PITCH_SHAPE = (
    1,
    2,
)

FRONTAL_SHAPE = (
    5,
    4,
)

YAW_SHAPE = (
    3,
    2,
)


PARAM_DIM = (
    int(np.prod(LEFT_SAG_SHAPE))
    +
    int(np.prod(RIGHT_SAG_SHAPE))
    +
    int(np.prod(WAIST_PITCH_SHAPE))
    +
    int(np.prod(FRONTAL_SHAPE))
    +
    int(np.prod(YAW_SHAPE))
)


env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


print("=" * 175)
print("RESTART V1 — BALANCE FEEDBACK V2")
print("VERTICAL FEEDBACK + INDEPENDENT LEG SAGITTAL CONTROL")
print("NO PPO")
print("=" * 175)

print("reference:", env.reference_path)
print("actions:", env.action_space.shape)
print("observations:", env.observation_space.shape)
print("parameter dimension:", PARAM_DIM)
print("train starts:", TRAIN_STARTS)
print("held-out starts:", HELDOUT_STARTS)
print("extra held-out starts:", EXTRA_HELDOUT_STARTS)


# ================================================================
# DECODE
# ================================================================

def decode(theta):

    theta = np.asarray(
        theta,
        dtype=np.float64,
    )

    if theta.shape != (PARAM_DIM,):
        raise ValueError(theta.shape)

    offset = 0

    def take(shape):

        nonlocal offset

        n = int(
            np.prod(shape)
        )

        result = theta[
            offset:
            offset + n
        ].reshape(shape)

        offset += n

        return result


    left_sag = take(
        LEFT_SAG_SHAPE
    )

    right_sag = take(
        RIGHT_SAG_SHAPE
    )

    waist_pitch = take(
        WAIST_PITCH_SHAPE
    )

    frontal = take(
        FRONTAL_SHAPE
    )

    yaw = take(
        YAW_SHAPE
    )

    return (
        left_sag,
        right_sag,
        waist_pitch,
        frontal,
        yaw,
    )


# ================================================================
# STATE
# ================================================================

def state_features():

    frame = int(
        env._current_frame
    )

    ori = quat_error_vector(
        env.ref_root_quat[
            frame
        ],
        env.data.qpos[
            3:7
        ],
    )

    ang = (
        env.data.qvel[
            3:6
        ]
        -
        env.ref_full_qvel[
            frame,
            3:6
        ]
    )

    R = env._root_rotation_matrix()

    pos = (
        R.T
        @ (
            env.data.qpos[
                0:3
            ]
            -
            env.ref_root_pos[
                frame
            ]
        )
    )

    vel = (
        R.T
        @ (
            env.data.qvel[
                0:3
            ]
            -
            env.ref_full_qvel[
                frame,
                0:3
            ]
        )
    )


    # Foot vertical-error signal.
    left_foot_world_error = (
        env.data.site_xpos[
            env.left_foot_site
        ]
        -
        env.ref_left_foot_pos[
            frame
        ]
    )

    right_foot_world_error = (
        env.data.site_xpos[
            env.right_foot_site
        ]
        -
        env.ref_right_foot_pos[
            frame
        ]
    )


    left_foot_local = (
        R.T
        @ left_foot_world_error
    )

    right_foot_local = (
        R.T
        @ right_foot_world_error
    )


    # ------------------------------------------------------------
    # Sagittal features:
    #
    # pitch
    # pitch velocity
    # X error
    # X velocity error
    # Z error
    # Z velocity error
    # own-foot Z error
    # ------------------------------------------------------------

    left_sag = np.asarray(
        [
            ori[1] / 0.35,
            ang[1] / 1.50,

            pos[0] / 0.30,
            vel[0] / 0.80,

            pos[2] / 0.15,
            vel[2] / 0.60,

            left_foot_local[2] / 0.10,
        ],
        dtype=np.float64,
    )


    right_sag = np.asarray(
        [
            ori[1] / 0.35,
            ang[1] / 1.50,

            pos[0] / 0.30,
            vel[0] / 0.80,

            pos[2] / 0.15,
            vel[2] / 0.60,

            right_foot_local[2] / 0.10,
        ],
        dtype=np.float64,
    )


    waist_pitch = np.asarray(
        [
            ori[1] / 0.35,
            ang[1] / 1.50,
        ],
        dtype=np.float64,
    )


    frontal = np.asarray(
        [
            ori[0] / 0.35,
            ang[0] / 1.50,

            pos[1] / 0.20,
            vel[1] / 0.60,
        ],
        dtype=np.float64,
    )


    yaw = np.asarray(
        [
            ori[2] / 0.35,
            ang[2] / 1.50,
        ],
        dtype=np.float64,
    )


    return (
        np.clip(
            left_sag,
            -2.5,
            2.5,
        ),

        np.clip(
            right_sag,
            -2.5,
            2.5,
        ),

        np.clip(
            waist_pitch,
            -2.5,
            2.5,
        ),

        np.clip(
            frontal,
            -2.5,
            2.5,
        ),

        np.clip(
            yaw,
            -2.5,
            2.5,
        ),
    )


# ================================================================
# CLOSED-LOOP CONTROLLER
# ================================================================

def feedback_action(theta):

    (
        left_gain,
        right_gain,
        waist_pitch_gain,
        frontal_gain,
        yaw_gain,
    ) = decode(theta)


    (
        left_features,
        right_features,
        waist_pitch_features,
        frontal_features,
        yaw_features,
    ) = state_features()


    left_out = (
        left_gain
        @ left_features
    )

    right_out = (
        right_gain
        @ right_features
    )

    waist_pitch_out = float(
        (
            waist_pitch_gain
            @ waist_pitch_features
        )[0]
    )

    frontal_out = (
        frontal_gain
        @ frontal_features
    )

    yaw_out = (
        yaw_gain
        @ yaw_features
    )


    frame = int(
        env._current_frame
    )

    ref_support = (
        env.ref_support[
            frame
        ]
    )

    contact = (
        env._actual_contacts()
    )


    left_support = max(
        float(
            ref_support[0]
        ),
        float(
            contact[0]
        ),
    )

    right_support = max(
        float(
            ref_support[1]
        ),
        float(
            contact[1]
        ),
    )


    left_gate = (
        0.40
        +
        0.60
        * left_support
    )

    right_gate = (
        0.40
        +
        0.60
        * right_support
    )


    action = np.zeros(
        29,
        dtype=np.float64,
    )


    # ------------------------------------------------------------
    # Independent left sagittal
    # hip pitch / knee / ankle pitch
    # ------------------------------------------------------------

    action[0] += (
        left_gate
        * left_out[0]
    )

    action[3] += (
        left_gate
        * left_out[1]
    )

    action[4] += (
        left_gate
        * left_out[2]
    )


    # ------------------------------------------------------------
    # Independent right sagittal
    # ------------------------------------------------------------

    action[6] += (
        right_gate
        * right_out[0]
    )

    action[9] += (
        right_gate
        * right_out[1]
    )

    action[10] += (
        right_gate
        * right_out[2]
    )


    action[14] += (
        waist_pitch_out
    )


    # ------------------------------------------------------------
    # Frontal
    # ------------------------------------------------------------

    hip_roll_common = (
        frontal_out[0]
    )

    hip_roll_diff = (
        frontal_out[1]
    )

    ankle_roll_common = (
        frontal_out[2]
    )

    ankle_roll_diff = (
        frontal_out[3]
    )

    waist_roll = (
        frontal_out[4]
    )


    action[1] += (
        left_gate
        * (
            hip_roll_common
            +
            hip_roll_diff
        )
    )

    action[7] += (
        right_gate
        * (
            hip_roll_common
            -
            hip_roll_diff
        )
    )


    action[5] += (
        left_gate
        * (
            ankle_roll_common
            +
            ankle_roll_diff
        )
    )

    action[11] += (
        right_gate
        * (
            ankle_roll_common
            -
            ankle_roll_diff
        )
    )


    action[13] += (
        waist_roll
    )


    # ------------------------------------------------------------
    # Yaw
    # ------------------------------------------------------------

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
        left_gate
        * (
            hip_yaw_common
            +
            hip_yaw_diff
        )
    )

    action[8] += (
        right_gate
        * (
            hip_yaw_common
            -
            hip_yaw_diff
        )
    )


    action[12] += (
        waist_yaw
    )


    # Keep official/validated action scale unchanged.
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
                np.abs(
                    action
                )
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

            reason = (
                "reference_end"
            )

            completed = True

            break


    if not reason:
        reason = "horizon"


    d = max(
        steps,
        1,
    )


    progress = (
        steps
        / max_steps
    )


    ori = (
        ori_sum
        / d
    )


    xerr = (
        x_sum
        / d
    )

    yerr = (
        y_sum
        / d
    )

    zerr = (
        z_sum
        / d
    )

    qerr = (
        q_sum
        / d
    )

    action_rms = (
        action_sum
        / d
    )

    clip_fraction = (
        clipped
        / max(
            action_values,
            1,
        )
    )


    # ============================================================
    # OBJECTIVE
    #
    # Survival still dominates.
    # Vertical collapse is now explicitly penalized.
    # ============================================================

    score = (
        130.0
        * progress

        +
        30.0
        * min_up

        -
        0.23
        * ori

        -
        10.0
        * xerr

        -
        16.0
        * yerr

        -
        24.0
        * zerr

        -
        2.0
        * qerr

        -
        1.5
        * action_rms

        -
        25.0
        * clip_fraction
    )


    if completed:

        score += 20.0


    return {
        "start":
            start,

        "steps":
            steps,

        "max_steps":
            max_steps,

        "progress":
            progress,

        "min_up":
            min_up,

        "orientation":
            ori,

        "x":
            xerr,

        "y":
            yerr,

        "z":
            zerr,

        "q":
            qerr,

        "action":
            action_rms,

        "clip":
            clip_fraction,

        "completed":
            completed,

        "reason":
            reason,

        "score":
            score,
    }


# ================================================================
# BASELINE
# ================================================================

print()
print("=" * 175)
print("ZERO-ACTION BASELINE")
print("=" * 175)


baseline = {}


for start in ALL_STARTS:

    r = rollout(
        None,
        start,
    )

    baseline[start] = r


    print(
        f"start={start:03d} "
        f"steps={r['steps']:3d}/"
        f"{r['max_steps']:3d} "
        f"prog={r['progress']:.3f} "
        f"up={r['min_up']:.3f} "
        f"ori={r['orientation']:.2f} "
        f"x={r['x']:.3f} "
        f"y={r['y']:.3f} "
        f"z={r['z']:.3f} "
        f"reason={r['reason']}"
    )


# ================================================================
# CEM OBJECTIVE
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
            x["score"]
            for x in rows
        ],
        dtype=np.float64,
    )


    progress = np.asarray(
        [
            x["progress"]
            for x in rows
        ],
        dtype=np.float64,
    )


    robust = (
        float(
            np.mean(
                scores
            )
        )

        +
        30.0
        * float(
            np.min(
                progress
            )
        )
    )


    return (
        robust,
        rows,
    )


# ================================================================
# CEM
# ================================================================

print()
print("=" * 175)
print("BALANCE FEEDBACK V2 CEM")
print("=" * 175)


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


    progress_means = np.empty(
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


        progress_means[i] = float(
            np.mean(
                [
                    x["progress"]
                    for x in rows
                ]
            )
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
        f"bestProg="
        f"{progress_means[idx]:.3f} "
        f"std={np.mean(std):.3f}"
    )


# ================================================================
# CONFIRMATION
# ================================================================

print()
print("=" * 175)
print("BALANCE FEEDBACK V2 CONFIRMATION")
print("=" * 175)


best = {}


for start in ALL_STARTS:

    r = rollout(
        best_theta,
        start,
    )

    best[start] = r


    b = baseline[
        start
    ]


    print(
        f"start={start:03d} "
        f"BASE={b['steps']:3d}/"
        f"{b['max_steps']:3d} "
        f"V2={r['steps']:3d}/"
        f"{r['max_steps']:3d} "
        f"delta={r['steps']-b['steps']:+4d} "
        f"prog={b['progress']:.3f}"
        f"->{r['progress']:.3f} "
        f"ori={b['orientation']:.2f}"
        f"->{r['orientation']:.2f} "
        f"z={b['z']:.3f}"
        f"->{r['z']:.3f} "
        f"minUp={b['min_up']:.3f}"
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
        table[s]
        for s in starts
    ]


    return {
        "steps":
            float(
                np.mean(
                    [
                        x["steps"]
                        for x in rows
                    ]
                )
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
                        x["orientation"]
                        for x in rows
                    ]
                )
            ),

        "x":
            float(
                np.mean(
                    [
                        x["x"]
                        for x in rows
                    ]
                )
            ),

        "y":
            float(
                np.mean(
                    [
                        x["y"]
                        for x in rows
                    ]
                )
            ),

        "z":
            float(
                np.mean(
                    [
                        x["z"]
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

        "clip":
            float(
                np.mean(
                    [
                        x["clip"]
                        for x in rows
                    ]
                )
            ),

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

v2_all = aggregate(
    best,
    ALL_STARTS,
)


base_hold = aggregate(
    baseline,
    HELDOUT_STARTS,
)

v2_hold = aggregate(
    best,
    HELDOUT_STARTS,
)


base_extra = aggregate(
    baseline,
    EXTRA_HELDOUT_STARTS,
)

v2_extra = aggregate(
    best,
    EXTRA_HELDOUT_STARTS,
)


print()
print("=" * 175)
print("BALANCE FEEDBACK V2 SUMMARY")
print("=" * 175)

print(
    f"ALL progress      : "
    f"{base_all['progress']:.3f}"
    f" -> "
    f"{v2_all['progress']:.3f}"
)

print(
    f"HELD-OUT progress : "
    f"{base_hold['progress']:.3f}"
    f" -> "
    f"{v2_hold['progress']:.3f}"
)

print(
    f"EXTRA held-out    : "
    f"{base_extra['progress']:.3f}"
    f" -> "
    f"{v2_extra['progress']:.3f}"
)

print(
    f"orientation       : "
    f"{base_all['orientation']:.2f}"
    f" -> "
    f"{v2_all['orientation']:.2f}"
)

print(
    f"X error           : "
    f"{base_all['x']:.3f}"
    f" -> "
    f"{v2_all['x']:.3f}"
)

print(
    f"Y error           : "
    f"{base_all['y']:.3f}"
    f" -> "
    f"{v2_all['y']:.3f}"
)

print(
    f"Z error           : "
    f"{base_all['z']:.3f}"
    f" -> "
    f"{v2_all['z']:.3f}"
)

print(
    f"mean min-up       : "
    f"{base_all['min_up']:.3f}"
    f" -> "
    f"{v2_all['min_up']:.3f}"
)

print(
    f"clipping          : "
    f"{100*v2_all['clip']:.2f}%"
)

print(
    f"completions       : "
    f"{base_all['completed']}"
    f" -> "
    f"{v2_all['completed']}"
)


# ================================================================
# DECISION
# ================================================================

all_gain = (
    v2_all["progress"]
    -
    base_all["progress"]
)

hold_gain = (
    v2_hold["progress"]
    -
    base_hold["progress"]
)

extra_gain = (
    v2_extra["progress"]
    -
    base_extra["progress"]
)

ori_change = (
    v2_all["orientation"]
    -
    base_all["orientation"]
)

z_change = (
    v2_all["z"]
    -
    base_all["z"]
)


worst_regression = min(
    best[s]["steps"]
    -
    baseline[s]["steps"]

    for s in ALL_STARTS
)


print()
print("=" * 175)
print("BALANCE FEEDBACK V2 DECISION")
print("=" * 175)

print(
    "all-start gain:",
    f"{all_gain:+.3f}",
)

print(
    "held-out gain:",
    f"{hold_gain:+.3f}",
)

print(
    "extra held-out gain:",
    f"{extra_gain:+.3f}",
)

print(
    "orientation change:",
    f"{ori_change:+.2f} deg",
)

print(
    "vertical error change:",
    f"{z_change:+.3f} m",
)

print(
    "worst step regression:",
    f"{worst_regression:+d}",
)

print(
    "clipping:",
    f"{100*v2_all['clip']:.2f}%",
)


strong = (
    all_gain >= 0.15
    and
    hold_gain >= 0.08
    and
    extra_gain >= 0.05
    and
    ori_change <= 4.0
    and
    z_change <= 0.02
    and
    worst_regression >= -5
    and
    v2_all["clip"] <= 0.08
)


marginal = (
    all_gain >= 0.08
    and
    hold_gain >= 0.04
    and
    ori_change <= 7.0
    and
    worst_regression >= -10
)


if strong:

    print()
    print(
        "BALANCE REPRESENTATION: STRONG"
    )

    print(
        "Vertical state and independent leg "
        "feedback materially improve robust control."
    )

    print(
        "NEXT: first small PPO pilot."
    )


elif marginal:

    print()
    print(
        "BALANCE REPRESENTATION: MARGINAL"
    )

    print(
        "The representation helps, but robust "
        "authority is still incomplete."
    )

    print(
        "NEXT: add foot-placement / capture-point "
        "features before PPO."
    )


else:

    print()
    print(
        "BALANCE REPRESENTATION: INSUFFICIENT"
    )

    print(
        "Do NOT train PPO."
    )

    print(
        "NEXT: move from root-only feedback to "
        "explicit foot-placement / COM control."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_balance_feedback_v2_best.npz"
)


np.savez(
    output,

    theta=
        best_theta.astype(
            np.float32
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

    extra_heldout_gain=
        np.asarray(
            [extra_gain],
            dtype=np.float32,
        ),

    orientation_change=
        np.asarray(
            [ori_change],
            dtype=np.float32,
        ),

    vertical_error_change=
        np.asarray(
            [z_change],
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

print("=" * 175)


env.close()
