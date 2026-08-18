from pathlib import Path
import math
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
    quat_error_vector,
)


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_STARTS = [
    0,
    57,
    114,
]


# These starts are NOT used by CEM.
# They test whether the feedback controller generalizes.
HELDOUT_STARTS = [
    30,
    85,
]


ALL_STARTS = (
    TRAIN_STARTS
    + HELDOUT_STARTS
)


HORIZON = 120


CEM_POPULATION = 96
CEM_ELITE = 12
CEM_ITERATIONS = 5

INITIAL_STD = 0.70

MIN_STD = 0.08

PARAM_LIMIT = 3.0

SEED = 20260814


# ================================================================
# PARAMETERIZATION
#
# 16 sagittal
# 20 frontal
#  6 yaw
# ----------------
# 42 total feedback parameters
#
# This is NOT an open-loop action sequence.
#
# Every group output is computed from the CURRENT ROBOT STATE.
# ================================================================

SAGITTAL_SHAPE = (
    4,
    4,
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
    np.prod(
        SAGITTAL_SHAPE
    )
    +
    np.prod(
        FRONTAL_SHAPE
    )
    +
    np.prod(
        YAW_SHAPE
    )
)


# ================================================================
# ENVIRONMENT
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


print("=" * 165)
print("29-DOF RESTART V1 STATE-FEEDBACK AUTHORITY TEST")
print("NO PPO TRAINING")
print("=" * 165)

print(
    "reference:",
    env.reference_path,
)

print(
    "frames:",
    env.num_frames,
)

print(
    "action dim:",
    env.action_space.shape,
)

print(
    "observation dim:",
    env.observation_space.shape,
)

print(
    "feedback parameter dimension:",
    PARAM_DIM,
)

print(
    "training starts:",
    TRAIN_STARTS,
)

print(
    "held-out starts:",
    HELDOUT_STARTS,
)

print(
    "horizon:",
    HORIZON,
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
            f"Expected ({PARAM_DIM},), "
            f"got {theta.shape}"
        )


    offset = 0


    sagittal_size = int(
        np.prod(
            SAGITTAL_SHAPE
        )
    )


    sagittal = theta[
        offset:
        offset
        + sagittal_size
    ].reshape(
        SAGITTAL_SHAPE
    )


    offset += sagittal_size


    frontal_size = int(
        np.prod(
            FRONTAL_SHAPE
        )
    )


    frontal = theta[
        offset:
        offset
        + frontal_size
    ].reshape(
        FRONTAL_SHAPE
    )


    offset += frontal_size


    yaw_size = int(
        np.prod(
            YAW_SHAPE
        )
    )


    yaw = theta[
        offset:
        offset
        + yaw_size
    ].reshape(
        YAW_SHAPE
    )


    return (
        sagittal,
        frontal,
        yaw,
    )


# ================================================================
# CURRENT STATE FEATURES
# ================================================================

def state_features():

    frame = int(
        env._current_frame
    )


    # ------------------------------------------------------------
    # Orientation error:
    #
    # [roll-like, pitch-like, yaw-like]
    # small-angle representation.
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
            3:6,
        ]
    )


    R = env._root_rotation_matrix()


    root_position_error_body = (
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


    root_velocity_error_body = (
        R.T
        @ (
            env.data.qvel[
                0:3
            ]
            -
            env.ref_full_qvel[
                frame,
                0:3,
            ]
        )
    )


    # ------------------------------------------------------------
    # Normalize features into approximately O(1) ranges.
    # CEM then searches feedback gain values rather than fighting
    # wildly different physical units.
    # ------------------------------------------------------------

    sagittal = np.asarray(
        [
            orientation_error[
                1
            ] / 0.35,

            angular_velocity_error[
                1
            ] / 1.50,

            root_position_error_body[
                0
            ] / 0.30,

            root_velocity_error_body[
                0
            ] / 0.80,
        ],
        dtype=np.float64,
    )


    frontal = np.asarray(
        [
            orientation_error[
                0
            ] / 0.35,

            angular_velocity_error[
                0
            ] / 1.50,

            root_position_error_body[
                1
            ] / 0.20,

            root_velocity_error_body[
                1
            ] / 0.60,
        ],
        dtype=np.float64,
    )


    yaw = np.asarray(
        [
            orientation_error[
                2
            ] / 0.35,

            angular_velocity_error[
                2
            ] / 1.50,
        ],
        dtype=np.float64,
    )


    sagittal = np.clip(
        sagittal,
        -2.5,
        +2.5,
    )

    frontal = np.clip(
        frontal,
        -2.5,
        +2.5,
    )

    yaw = np.clip(
        yaw,
        -2.5,
        +2.5,
    )


    return (
        sagittal,
        frontal,
        yaw,
    )


# ================================================================
# STATE-FEEDBACK POLICY
#
# Critical point:
#
# action(t) = f(current state error)
#
# not:
#
# action(t) = pre-recorded sequence[t]
# ================================================================

def feedback_action(
    theta,
):

    (
        sagittal_gain,
        frontal_gain,
        yaw_gain,
    ) = decode(
        theta
    )


    (
        sagittal_features,
        frontal_features,
        yaw_features,
    ) = state_features()


    sagittal_output = (
        sagittal_gain
        @ sagittal_features
    )


    frontal_output = (
        frontal_gain
        @ frontal_features
    )


    yaw_output = (
        yaw_gain
        @ yaw_features
    )


    frame = int(
        env._current_frame
    )


    # ------------------------------------------------------------
    # Support-aware gain gating.
    #
    # Support leg gets full balance authority.
    # Swing leg still gets 45%, so its trajectory is not frozen.
    # ------------------------------------------------------------

    reference_support = (
        env.ref_support[
            frame
        ]
    )


    actual_contact = (
        env._actual_contacts()
    )


    left_support_signal = max(
        float(
            reference_support[
                0
            ]
        ),
        float(
            actual_contact[
                0
            ]
        ),
    )


    right_support_signal = max(
        float(
            reference_support[
                1
            ]
        ),
        float(
            actual_contact[
                1
            ]
        ),
    )


    left_gate = (
        0.45
        + 0.55
        * left_support_signal
    )


    right_gate = (
        0.45
        + 0.55
        * right_support_signal
    )


    action = np.zeros(
        29,
        dtype=np.float64,
    )


    # ============================================================
    # SAGITTAL
    #
    # output:
    # 0 hip pitch
    # 1 knee
    # 2 ankle pitch
    # 3 waist pitch
    # ============================================================

    hip_pitch = (
        sagittal_output[
            0
        ]
    )

    knee = (
        sagittal_output[
            1
        ]
    )

    ankle_pitch = (
        sagittal_output[
            2
        ]
    )

    waist_pitch = (
        sagittal_output[
            3
        ]
    )


    action[
        0
    ] += (
        hip_pitch
        * left_gate
    )

    action[
        6
    ] += (
        hip_pitch
        * right_gate
    )


    action[
        3
    ] += (
        knee
        * left_gate
    )

    action[
        9
    ] += (
        knee
        * right_gate
    )


    action[
        4
    ] += (
        ankle_pitch
        * left_gate
    )

    action[
        10
    ] += (
        ankle_pitch
        * right_gate
    )


    action[
        14
    ] += waist_pitch


    # ============================================================
    # FRONTAL
    #
    # output:
    # 0 hip-roll common
    # 1 hip-roll differential
    # 2 ankle-roll common
    # 3 ankle-roll differential
    # 4 waist roll
    #
    # Both common and differential modes are included so we don't
    # assume the correct left/right sign convention in advance.
    # CEM determines it.
    # ============================================================

    hip_roll_common = (
        frontal_output[
            0
        ]
    )

    hip_roll_diff = (
        frontal_output[
            1
        ]
    )

    ankle_roll_common = (
        frontal_output[
            2
        ]
    )

    ankle_roll_diff = (
        frontal_output[
            3
        ]
    )

    waist_roll = (
        frontal_output[
            4
        ]
    )


    action[
        1
    ] += (
        (
            hip_roll_common
            +
            hip_roll_diff
        )
        * left_gate
    )


    action[
        7
    ] += (
        (
            hip_roll_common
            -
            hip_roll_diff
        )
        * right_gate
    )


    action[
        5
    ] += (
        (
            ankle_roll_common
            +
            ankle_roll_diff
        )
        * left_gate
    )


    action[
        11
    ] += (
        (
            ankle_roll_common
            -
            ankle_roll_diff
        )
        * right_gate
    )


    action[
        13
    ] += waist_roll


    # ============================================================
    # YAW
    #
    # output:
    # 0 hip-yaw common
    # 1 hip-yaw differential
    # 2 waist yaw
    # ============================================================

    hip_yaw_common = (
        yaw_output[
            0
        ]
    )

    hip_yaw_diff = (
        yaw_output[
            1
        ]
    )

    waist_yaw = (
        yaw_output[
            2
        ]
    )


    action[
        2
    ] += (
        (
            hip_yaw_common
            +
            hip_yaw_diff
        )
        * left_gate
    )


    action[
        8
    ] += (
        (
            hip_yaw_common
            -
            hip_yaw_diff
        )
        * right_gate
    )


    action[
        12
    ] += waist_yaw


    # Smooth bounded normalized action.
    action = np.tanh(
        action
    )


    return action.astype(
        np.float32
    )


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    theta,
    start,
    collect=False,
):

    obs, info = env.reset(
        options={
            "start_frame":
                int(
                    start
                ),
        },
    )


    remaining = (
        env.num_frames
        - 1
        - int(
            start
        )
    )


    maximum_steps = min(
        HORIZON,
        remaining,
    )


    if maximum_steps <= 0:

        raise RuntimeError(
            "Invalid rollout horizon."
        )


    previous_action = np.zeros(
        29,
        dtype=np.float64,
    )


    steps = 0

    min_up = env._up_z()

    orientation_sum = 0.0

    root_sum = 0.0

    q_sum = 0.0

    qd_sum = 0.0

    action_sum = 0.0

    action_rate_sum = 0.0

    max_y_error = 0.0

    terminated_reason = ""

    completed = False


    history = []


    while (
        steps
        < maximum_steps
    ):

        if theta is None:

            action = np.zeros(
                29,
                dtype=np.float32,
            )

        else:

            action = feedback_action(
                theta
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


        terms = info[
            "reward_terms"
        ]


        up = float(
            info[
                "up"
            ]
        )


        min_up = min(
            min_up,
            up,
        )


        orientation_sum += float(
            terms[
                "root_orientation_deg"
            ]
        )


        root_sum += float(
            terms[
                "root_position_error"
            ]
        )


        q_sum += float(
            terms[
                "q_error"
            ]
        )


        qd_sum += float(
            terms[
                "qd_error"
            ]
        )


        action64 = np.asarray(
            action,
            dtype=np.float64,
        )


        action_sum += float(
            np.sqrt(
                np.mean(
                    action64 ** 2
                )
            )
        )


        action_rate_sum += float(
            np.sqrt(
                np.mean(
                    (
                        action64
                        - previous_action
                    ) ** 2
                )
            )
        )


        previous_action = (
            action64.copy()
        )


        frame = int(
            env._current_frame
        )


        y_error = abs(
            float(
                env.data.qpos[
                    1
                ]
                -
                env.ref_root_pos[
                    frame,
                    1
                ]
            )
        )


        max_y_error = max(
            max_y_error,
            y_error,
        )


        if collect:

            history.append(
                (
                    frame,
                    up,
                    float(
                        terms[
                            "root_orientation_deg"
                        ]
                    ),
                    float(
                        terms[
                            "root_position_error"
                        ]
                    ),
                    y_error,
                    float(
                        terms[
                            "q_error"
                        ]
                    ),
                )
            )


        if terminated:

            terminated_reason = (
                info[
                    "termination_reason"
                ]
            )

            break


        if truncated:

            completed = True
            terminated_reason = (
                "reference_end"
            )

            break


    if (
        not terminated_reason
        and steps >= maximum_steps
    ):

        terminated_reason = (
            "horizon"
        )


    progress = (
        steps
        / maximum_steps
    )


    mean_orientation = (
        orientation_sum
        / max(
            steps,
            1,
        )
    )


    mean_root = (
        root_sum
        / max(
            steps,
            1,
        )
    )


    mean_q = (
        q_sum
        / max(
            steps,
            1,
        )
    )


    mean_qd = (
        qd_sum
        / max(
            steps,
            1,
        )
    )


    mean_action = (
        action_sum
        / max(
            steps,
            1,
        )
    )


    mean_action_rate = (
        action_rate_sum
        / max(
            steps,
            1,
        )
    )


    # ============================================================
    # OBJECTIVE
    #
    # Survival dominates, but "falling differently" is penalized.
    # ============================================================

    score = (

        120.0
        * progress

        + 25.0
        * min_up

        - 0.25
        * mean_orientation

        - 10.0
        * mean_root

        - 12.0
        * max_y_error

        - 2.0
        * mean_q

        - 1.0
        * mean_qd

        - 2.0
        * mean_action

        - 1.0
        * mean_action_rate
    )


    if completed:

        score += 20.0


    if terminated_reason in (
        "orientation",
        "root_height",
    ):

        score -= (
            10.0
            * (
                1.0
                - progress
            )
        )


    return {

        "start":
            int(
                start
            ),

        "max_steps":
            int(
                maximum_steps
            ),

        "steps":
            int(
                steps
            ),

        "progress":
            float(
                progress
            ),

        "completed":
            bool(
                completed
            ),

        "reason":
            terminated_reason,

        "min_up":
            float(
                min_up
            ),

        "orientation":
            float(
                mean_orientation
            ),

        "root":
            float(
                mean_root
            ),

        "q":
            float(
                mean_q
            ),

        "qd":
            float(
                mean_qd
            ),

        "max_y":
            float(
                max_y_error
            ),

        "action":
            float(
                mean_action
            ),

        "action_rate":
            float(
                mean_action_rate
            ),

        "score":
            float(
                score
            ),

        "history":
            history,
    }


# ================================================================
# MULTI-START OBJECTIVE
# ================================================================

def evaluate_theta(
    theta,
    starts,
):

    rows = [
        rollout(
            theta,
            start,
            collect=False,
        )
        for start
        in starts
    ]


    mean_score = float(
        np.mean(
            [
                r[
                    "score"
                ]
                for r
                in rows
            ]
        )
    )


    min_progress = float(
        np.min(
            [
                r[
                    "progress"
                ]
                for r
                in rows
            ]
        )
    )


    # Prevent CEM from succeeding on only one start state.
    robust_score = (
        mean_score
        + 25.0
        * min_progress
    )


    return (
        robust_score,
        rows,
    )


# ================================================================
# ZERO-ACTION BASELINE
# ================================================================

print()
print("=" * 165)
print("ZERO-ACTION FINITE-HORIZON BASELINE")
print("=" * 165)


baseline_rows = {}


for start in ALL_STARTS:

    result = rollout(
        None,
        start,
        collect=False,
    )


    baseline_rows[
        start
    ] = result


    print(
        f"start={start:03d} "
        f"steps={result['steps']:3d}/"
        f"{result['max_steps']:3d} "
        f"prog={result['progress']:.3f} "
        f"minUp={result['min_up']:.3f} "
        f"ori={result['orientation']:.2f} "
        f"root={result['root']:.3f} "
        f"maxY={result['max_y']:.3f} "
        f"q={result['q']:.3f} "
        f"reason={result['reason']}"
    )


# ================================================================
# CEM
# ================================================================

print()
print("=" * 165)
print("STATE-FEEDBACK CEM")
print("TRAIN STARTS ONLY:", TRAIN_STARTS)
print("=" * 165)


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

best_rows = None


for iteration in range(
    1,
    CEM_ITERATIONS + 1,
):

    population = (
        mean[
            None,
            :
        ]
        +
        std[
            None,
            :
        ]
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
        +PARAM_LIMIT,
    )


    scores = np.empty(
        CEM_POPULATION,
        dtype=np.float64,
    )


    iteration_rows = []


    for i in range(
        CEM_POPULATION
    ):

        score, rows = (
            evaluate_theta(
                population[
                    i
                ],
                TRAIN_STARTS,
            )
        )


        scores[
            i
        ] = score

        iteration_rows.append(
            rows
        )


    order = np.argsort(
        scores
    )[
        ::-1
    ]


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


    iteration_best_index = int(
        order[
            0
        ]
    )


    iteration_best_score = float(
        scores[
            iteration_best_index
        ]
    )


    if (
        iteration_best_score
        >
        best_score
    ):

        best_score = (
            iteration_best_score
        )

        best_theta = (
            population[
                iteration_best_index
            ].copy()
        )

        best_rows = (
            iteration_rows[
                iteration_best_index
            ]
        )


    elite_score_mean = float(
        np.mean(
            scores[
                order[
                    :CEM_ELITE
                ]
            ]
        )
    )


    iteration_best_progress = float(
        np.mean(
            [
                x[
                    "progress"
                ]
                for x in
                iteration_rows[
                    iteration_best_index
                ]
            ]
        )
    )


    iteration_best_orientation = float(
        np.mean(
            [
                x[
                    "orientation"
                ]
                for x in
                iteration_rows[
                    iteration_best_index
                ]
            ]
        )
    )


    print(
        f"iter={iteration} "
        f"bestScore={iteration_best_score:+8.2f} "
        f"eliteMean={elite_score_mean:+8.2f} "
        f"meanProg={iteration_best_progress:.3f} "
        f"meanOri={iteration_best_orientation:.2f} "
        f"stdMean={np.mean(std):.3f}"
    )


# ================================================================
# CONFIRM GLOBAL BEST ON TRAINING + HELD-OUT STARTS
# ================================================================

print()
print("=" * 165)
print("BEST STATE-FEEDBACK CONTROLLER CONFIRMATION")
print("=" * 165)


best_results = {}


for start in ALL_STARTS:

    result = rollout(
        best_theta,
        start,
        collect=False,
    )


    best_results[
        start
    ] = result


    base = baseline_rows[
        start
    ]


    print(
        f"start={start:03d} "
        f"BASE={base['steps']:3d}/"
        f"{base['max_steps']:3d} "
        f"FB={result['steps']:3d}/"
        f"{result['max_steps']:3d} "
        f"delta={result['steps']-base['steps']:+4d} "
        f"prog={base['progress']:.3f}"
        f"->{result['progress']:.3f} "
        f"ori={base['orientation']:.2f}"
        f"->{result['orientation']:.2f} "
        f"root={base['root']:.3f}"
        f"->{result['root']:.3f} "
        f"minUp={base['min_up']:.3f}"
        f"->{result['min_up']:.3f} "
        f"maxY={base['max_y']:.3f}"
        f"->{result['max_y']:.3f} "
        f"reason={result['reason']}"
    )


# ================================================================
# AGGREGATES
# ================================================================

def aggregate(
    table,
    starts,
):

    rows = [
        table[
            start
        ]
        for start
        in starts
    ]


    return {

        "steps":
            float(
                np.mean(
                    [
                        r[
                            "steps"
                        ]
                        for r
                        in rows
                    ]
                )
            ),

        "progress":
            float(
                np.mean(
                    [
                        r[
                            "progress"
                        ]
                        for r
                        in rows
                    ]
                )
            ),

        "orientation":
            float(
                np.mean(
                    [
                        r[
                            "orientation"
                        ]
                        for r
                        in rows
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
                        for r
                        in rows
                    ]
                )
            ),

        "min_up":
            float(
                np.mean(
                    [
                        r[
                            "min_up"
                        ]
                        for r
                        in rows
                    ]
                )
            ),

        "max_y":
            float(
                np.mean(
                    [
                        r[
                            "max_y"
                        ]
                        for r
                        in rows
                    ]
                )
            ),

        "completed":
            int(
                sum(
                    int(
                        r[
                            "completed"
                        ]
                    )
                    for r
                    in rows
                )
            ),
    }


base_all = aggregate(
    baseline_rows,
    ALL_STARTS,
)

best_all = aggregate(
    best_results,
    ALL_STARTS,
)


base_train = aggregate(
    baseline_rows,
    TRAIN_STARTS,
)

best_train = aggregate(
    best_results,
    TRAIN_STARTS,
)


base_hold = aggregate(
    baseline_rows,
    HELDOUT_STARTS,
)

best_hold = aggregate(
    best_results,
    HELDOUT_STARTS,
)


print()
print("=" * 165)
print("STATE-FEEDBACK AUTHORITY SUMMARY")
print("=" * 165)


print(
    "ALL STARTS"
)

print(
    f"  mean steps      : "
    f"{base_all['steps']:.1f}"
    f" -> "
    f"{best_all['steps']:.1f}"
)

print(
    f"  mean progress   : "
    f"{base_all['progress']:.3f}"
    f" -> "
    f"{best_all['progress']:.3f}"
)

print(
    f"  mean orientation: "
    f"{base_all['orientation']:.2f}"
    f" -> "
    f"{best_all['orientation']:.2f} deg"
)

print(
    f"  mean root error : "
    f"{base_all['root']:.3f}"
    f" -> "
    f"{best_all['root']:.3f} m"
)

print(
    f"  mean min-up     : "
    f"{base_all['min_up']:.3f}"
    f" -> "
    f"{best_all['min_up']:.3f}"
)

print(
    f"  mean max-Y      : "
    f"{base_all['max_y']:.3f}"
    f" -> "
    f"{best_all['max_y']:.3f} m"
)

print(
    f"  completions     : "
    f"{base_all['completed']}"
    f" -> "
    f"{best_all['completed']}"
)


print()
print(
    "TRAIN STARTS"
)

print(
    f"  progress: "
    f"{base_train['progress']:.3f}"
    f" -> "
    f"{best_train['progress']:.3f}"
)


print()
print(
    "HELD-OUT STARTS"
)

print(
    f"  progress: "
    f"{base_hold['progress']:.3f}"
    f" -> "
    f"{best_hold['progress']:.3f}"
)

print(
    f"  orientation: "
    f"{base_hold['orientation']:.2f}"
    f" -> "
    f"{best_hold['orientation']:.2f}"
)


# ================================================================
# REGRESSION CHECK
# ================================================================

worst_step_regression = min(
    (
        best_results[
            s
        ][
            "steps"
        ]
        -
        baseline_rows[
            s
        ][
            "steps"
        ]
    )
    for s in ALL_STARTS
)


progress_gain = (
    best_all[
        "progress"
    ]
    -
    base_all[
        "progress"
    ]
)


heldout_progress_gain = (
    best_hold[
        "progress"
    ]
    -
    base_hold[
        "progress"
    ]
)


orientation_change = (
    best_all[
        "orientation"
    ]
    -
    base_all[
        "orientation"
    ]
)


root_change = (
    best_all[
        "root"
    ]
    -
    base_all[
        "root"
    ]
)


minup_change = (
    best_all[
        "min_up"
    ]
    -
    base_all[
        "min_up"
    ]
)


print()
print("=" * 165)
print("STATE-FEEDBACK AUTHORITY DECISION")
print("=" * 165)

print(
    "progress gain:",
    f"{progress_gain:+.3f}",
)

print(
    "held-out progress gain:",
    f"{heldout_progress_gain:+.3f}",
)

print(
    "orientation change:",
    f"{orientation_change:+.2f} deg",
)

print(
    "root-error change:",
    f"{root_change:+.3f} m",
)

print(
    "min-up change:",
    f"{minup_change:+.3f}",
)

print(
    "worst per-start step regression:",
    f"{worst_step_regression:+d}",
)


# ================================================================
# DECISION RULE
# ================================================================

strong = (

    progress_gain
    >= 0.10

    and heldout_progress_gain
    >= 0.06

    and orientation_change
    <= 5.0

    and root_change
    <= 0.05

    and minup_change
    >= -0.02

    and worst_step_regression
    >= -5
)


marginal = (

    progress_gain
    >= 0.05

    and orientation_change
    <= 8.0

    and worst_step_regression
    >= -10
)


if strong:

    print()
    print(
        "STATE-FEEDBACK AUTHORITY: STRONG"
    )

    print(
        "Residual position actions have demonstrated "
        "real closed-loop corrective authority."
    )

    print(
        "NEXT: first small PPO pilot on Restart V1."
    )


elif marginal:

    print()
    print(
        "STATE-FEEDBACK AUTHORITY: MARGINAL"
    )

    print(
        "Closed-loop feedback helps, but the available "
        "authority is still limited."
    )

    print(
        "NEXT: inspect action-scale multiplier / "
        "balance-channel representation before PPO."
    )


else:

    print()
    print(
        "STATE-FEEDBACK AUTHORITY: INSUFFICIENT"
    )

    print(
        "Do NOT start PPO."
    )

    print(
        "NEXT: change the control/action formulation "
        "before training."
    )


# ================================================================
# SAVE BEST CONTROLLER
#
# Diagnostic artifact only — NOT a trained model.
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_feedback_authority_best.npz"
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

    best_score=
        np.asarray(
            [
                best_score
            ],
            dtype=np.float32,
        ),

    progress_gain=
        np.asarray(
            [
                progress_gain
            ],
            dtype=np.float32,
        ),

    heldout_progress_gain=
        np.asarray(
            [
                heldout_progress_gain
            ],
            dtype=np.float32,
        ),

    orientation_change=
        np.asarray(
            [
                orientation_change
            ],
            dtype=np.float32,
        ),

    root_change=
        np.asarray(
            [
                root_change
            ],
            dtype=np.float32,
        ),

    minup_change=
        np.asarray(
            [
                minup_change
            ],
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

print("=" * 165)


env.close()
