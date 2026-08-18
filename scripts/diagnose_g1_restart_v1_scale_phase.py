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


BEST_PATH = (
    ROOT
    / "results"
    / "g1_restart_v1_feedback_authority_best.npz"
)


if not BEST_PATH.exists():

    raise FileNotFoundError(
        BEST_PATH
    )


saved = np.load(
    BEST_PATH,
    allow_pickle=True,
)


theta = np.asarray(
    saved["theta"],
    dtype=np.float64,
)


STARTS = [
    0,
    30,
    57,
    85,
    114,
]


HELDOUT_STARTS = [
    30,
    85,
]


MULTIPLIERS = [
    0.00,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
]


HORIZON = 120


# ================================================================
# SAME FEEDBACK PARAMETERIZATION AS PREVIOUS TEST
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
    int(
        np.prod(
            SAGITTAL_SHAPE
        )
    )
    +
    int(
        np.prod(
            FRONTAL_SHAPE
        )
    )
    +
    int(
        np.prod(
            YAW_SHAPE
        )
    )
)


if theta.shape != (
    PARAM_DIM,
):

    raise RuntimeError(
        f"Expected theta ({PARAM_DIM},), "
        f"got {theta.shape}"
    )


env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


# ================================================================
# DECODE
# ================================================================

def decode(
    x,
):

    offset = 0


    sagittal_size = int(
        np.prod(
            SAGITTAL_SHAPE
        )
    )


    sagittal = x[
        offset:
        offset + sagittal_size
    ].reshape(
        SAGITTAL_SHAPE
    )


    offset += sagittal_size


    frontal_size = int(
        np.prod(
            FRONTAL_SHAPE
        )
    )


    frontal = x[
        offset:
        offset + frontal_size
    ].reshape(
        FRONTAL_SHAPE
    )


    offset += frontal_size


    yaw_size = int(
        np.prod(
            YAW_SHAPE
        )
    )


    yaw = x[
        offset:
        offset + yaw_size
    ].reshape(
        YAW_SHAPE
    )


    return (
        sagittal,
        frontal,
        yaw,
    )


SAGITTAL_GAIN, FRONTAL_GAIN, YAW_GAIN = (
    decode(
        theta
    )
)


# ================================================================
# STATE FEATURES
# ================================================================

def get_state_features():

    frame = int(
        env._current_frame
    )


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


    position_error_body = (
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


    velocity_error_body = (
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


    sagittal = np.asarray(
        [
            orientation_error[
                1
            ] / 0.35,

            angular_velocity_error[
                1
            ] / 1.50,

            position_error_body[
                0
            ] / 0.30,

            velocity_error_body[
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

            position_error_body[
                1
            ] / 0.20,

            velocity_error_body[
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


    return (
        np.clip(
            sagittal,
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
# ORIGINAL FEEDBACK CONTROLLER
# ================================================================

def raw_feedback_action():

    (
        sagittal_features,
        frontal_features,
        yaw_features,
    ) = get_state_features()


    sagittal_output = (
        SAGITTAL_GAIN
        @ sagittal_features
    )


    frontal_output = (
        FRONTAL_GAIN
        @ frontal_features
    )


    yaw_output = (
        YAW_GAIN
        @ yaw_features
    )


    frame = int(
        env._current_frame
    )


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
        +
        0.55
        * left_support_signal
    )


    right_gate = (
        0.45
        +
        0.55
        * right_support_signal
    )


    action = np.zeros(
        29,
        dtype=np.float64,
    )


    # ------------------------------------------------------------
    # SAGITTAL
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # FRONTAL
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # YAW
    # ------------------------------------------------------------

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


    return np.tanh(
        action
    )


# ================================================================
# MULTIPLIED ACTION
# ================================================================

def scaled_feedback_action(
    multiplier,
):

    if multiplier <= 0.0:

        return np.zeros(
            29,
            dtype=np.float32,
        )


    raw = raw_feedback_action()


    scaled = (
        multiplier
        * raw
    )


    return np.clip(
        scaled,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    start,
    multiplier,
):

    obs, info = env.reset(
        options={
            "start_frame":
                int(
                    start
                ),
        }
    )


    remaining = (
        env.num_frames
        - 1
        - start
    )


    maximum_steps = min(
        HORIZON,
        remaining,
    )


    steps = 0


    orientation_sum = 0.0

    q_sum = 0.0

    min_up = env._up_z()


    abs_x_sum = 0.0
    abs_y_sum = 0.0
    abs_z_sum = 0.0


    x_sq = 0.0
    y_sq = 0.0
    z_sq = 0.0


    max_abs_x = 0.0
    max_abs_y = 0.0
    max_abs_z = 0.0


    final_x_error = 0.0
    final_y_error = 0.0
    final_z_error = 0.0


    action_rms_sum = 0.0

    clipped_values = 0
    action_values = 0


    reason = ""


    while (
        steps
        < maximum_steps
    ):

        action = (
            scaled_feedback_action(
                multiplier
            )
        )


        clipped_values += int(
            np.sum(
                np.abs(
                    action
                )
                >= 0.999
            )
        )


        action_values += 29


        action_rms_sum += float(
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


        ex = float(
            root_error[
                0
            ]
        )

        ey = float(
            root_error[
                1
            ]
        )

        ez = float(
            root_error[
                2
            ]
        )


        final_x_error = ex
        final_y_error = ey
        final_z_error = ez


        abs_x_sum += abs(
            ex
        )

        abs_y_sum += abs(
            ey
        )

        abs_z_sum += abs(
            ez
        )


        x_sq += (
            ex ** 2
        )

        y_sq += (
            ey ** 2
        )

        z_sq += (
            ez ** 2
        )


        max_abs_x = max(
            max_abs_x,
            abs(
                ex
            ),
        )

        max_abs_y = max(
            max_abs_y,
            abs(
                ey
            ),
        )

        max_abs_z = max(
            max_abs_z,
            abs(
                ez
            ),
        )


        terms = info[
            "reward_terms"
        ]


        orientation_sum += float(
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
                info[
                    "up"
                ]
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

            break


    if not reason:

        reason = "horizon"


    divisor = max(
        steps,
        1,
    )


    return {

        "start":
            start,

        "mult":
            multiplier,

        "steps":
            steps,

        "max_steps":
            maximum_steps,

        "progress":
            steps
            / maximum_steps,

        "orientation":
            orientation_sum
            / divisor,

        "q":
            q_sum
            / divisor,

        "min_up":
            min_up,

        "mae_x":
            abs_x_sum
            / divisor,

        "mae_y":
            abs_y_sum
            / divisor,

        "mae_z":
            abs_z_sum
            / divisor,

        "rmse_x":
            math.sqrt(
                x_sq
                / divisor
            ),

        "rmse_y":
            math.sqrt(
                y_sq
                / divisor
            ),

        "rmse_z":
            math.sqrt(
                z_sq
                / divisor
            ),

        "max_x":
            max_abs_x,

        "max_y":
            max_abs_y,

        "max_z":
            max_abs_z,

        "final_x":
            final_x_error,

        "final_y":
            final_y_error,

        "final_z":
            final_z_error,

        "action_rms":
            action_rms_sum
            / divisor,

        "clip_fraction":
            (
                clipped_values
                / action_values

                if action_values
                else 0.0
            ),

        "reason":
            reason,
    }


# ================================================================
# RUN
# ================================================================

print("=" * 185)
print("RESTART V1 ACTION-SCALE + ROOT-ERROR DIAGNOSTIC")
print("NO PPO")
print("=" * 185)

print(
    "reference:",
    env.reference_path,
)

print(
    "feedback artifact:",
    BEST_PATH,
)

print(
    "starts:",
    STARTS,
)

print(
    "multipliers:",
    MULTIPLIERS,
)


results = []


for multiplier in MULTIPLIERS:

    for start in STARTS:

        result = rollout(
            start,
            multiplier,
        )

        results.append(
            result
        )


# ================================================================
# DETAIL
# ================================================================

print()
print("=" * 205)
print("PER-START SCALE RESULTS")
print("=" * 205)

print(
    f"{'MULT':>5s} "
    f"{'START':>5s} "
    f"{'STEP':>5s} "
    f"{'PROG':>6s} "
    f"{'ORI':>7s} "
    f"{'MINUP':>7s} "
    f"{'X-MAE':>7s} "
    f"{'Y-MAE':>7s} "
    f"{'Z-MAE':>7s} "
    f"{'X-END':>8s} "
    f"{'Y-END':>8s} "
    f"{'Z-END':>8s} "
    f"{'X-MAX':>7s} "
    f"{'ACT':>6s} "
    f"{'CLIP':>6s} "
    f"{'REASON':>14s}"
)

print("-" * 205)


for r in results:

    print(
        f"{r['mult']:5.2f} "
        f"{r['start']:5d} "
        f"{r['steps']:5d} "
        f"{r['progress']:6.3f} "
        f"{r['orientation']:7.2f} "
        f"{r['min_up']:7.3f} "
        f"{r['mae_x']:7.3f} "
        f"{r['mae_y']:7.3f} "
        f"{r['mae_z']:7.3f} "
        f"{r['final_x']:+8.3f} "
        f"{r['final_y']:+8.3f} "
        f"{r['final_z']:+8.3f} "
        f"{r['max_x']:7.3f} "
        f"{r['action_rms']:6.3f} "
        f"{r['clip_fraction']:6.3f} "
        f"{r['reason']:>14s}"
    )


# ================================================================
# AGGREGATE
# ================================================================

def aggregate(
    multiplier,
    starts,
):

    rows = [
        r
        for r in results

        if (
            r[
                "mult"
            ]
            == multiplier

            and r[
                "start"
            ]
            in starts
        )
    ]


    return {

        "progress":
            float(
                np.mean(
                    [
                        x[
                            "progress"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "steps":
            float(
                np.mean(
                    [
                        x[
                            "steps"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "orientation":
            float(
                np.mean(
                    [
                        x[
                            "orientation"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "min_up":
            float(
                np.mean(
                    [
                        x[
                            "min_up"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "x":
            float(
                np.mean(
                    [
                        x[
                            "mae_x"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "y":
            float(
                np.mean(
                    [
                        x[
                            "mae_y"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "z":
            float(
                np.mean(
                    [
                        x[
                            "mae_z"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "final_x":
            float(
                np.mean(
                    [
                        abs(
                            x[
                                "final_x"
                            ]
                        )
                        for x
                        in rows
                    ]
                )
            ),

        "action":
            float(
                np.mean(
                    [
                        x[
                            "action_rms"
                        ]
                        for x
                        in rows
                    ]
                )
            ),

        "clip":
            float(
                np.mean(
                    [
                        x[
                            "clip_fraction"
                        ]
                        for x
                        in rows
                    ]
                )
            ),
    }


summary = {
    multiplier:
        aggregate(
            multiplier,
            STARTS,
        )
    for multiplier
    in MULTIPLIERS
}


heldout = {
    multiplier:
        aggregate(
            multiplier,
            HELDOUT_STARTS,
        )
    for multiplier
    in MULTIPLIERS
}


print()
print("=" * 175)
print("ACTION-SCALE AGGREGATE")
print("=" * 175)

print(
    f"{'MULT':>5s} "
    f"{'STEPS':>7s} "
    f"{'PROG':>7s} "
    f"{'H-PROG':>7s} "
    f"{'ORI':>8s} "
    f"{'MINUP':>7s} "
    f"{'X-MAE':>7s} "
    f"{'Y-MAE':>7s} "
    f"{'Z-MAE':>7s} "
    f"{'|XEND|':>8s} "
    f"{'ACT':>7s} "
    f"{'CLIP':>7s}"
)

print("-" * 175)


for multiplier in MULTIPLIERS:

    s = summary[
        multiplier
    ]

    h = heldout[
        multiplier
    ]


    print(
        f"{multiplier:5.2f} "
        f"{s['steps']:7.1f} "
        f"{s['progress']:7.3f} "
        f"{h['progress']:7.3f} "
        f"{s['orientation']:8.2f} "
        f"{s['min_up']:7.3f} "
        f"{s['x']:7.3f} "
        f"{s['y']:7.3f} "
        f"{s['z']:7.3f} "
        f"{s['final_x']:8.3f} "
        f"{s['action']:7.3f} "
        f"{s['clip']:7.3f}"
    )


# ================================================================
# BEST MULTIPLIER
#
# Held-out generalization gets first priority.
# ================================================================

nonzero = [
    x
    for x in MULTIPLIERS
    if x > 0.0
]


best_multiplier = max(
    nonzero,

    key=lambda m: (
        heldout[
            m
        ][
            "progress"
        ],

        summary[
            m
        ][
            "progress"
        ],

        -summary[
            m
        ][
            "orientation"
        ],

        -summary[
            m
        ][
            "x"
        ],
    ),
)


base = summary[
    0.00
]

best = summary[
    best_multiplier
]

base_hold = heldout[
    0.00
]

best_hold = heldout[
    best_multiplier
]


# ================================================================
# ROOT ERROR DOMINANCE
# ================================================================

root_axis_total = (
    best[
        "x"
    ]
    +
    best[
        "y"
    ]
    +
    best[
        "z"
    ]
)


x_fraction = (
    best[
        "x"
    ]
    / max(
        root_axis_total,
        1e-12,
    )
)


progress_gain = (
    best[
        "progress"
    ]
    -
    base[
        "progress"
    ]
)


heldout_gain = (
    best_hold[
        "progress"
    ]
    -
    base_hold[
        "progress"
    ]
)


orientation_change = (
    best[
        "orientation"
    ]
    -
    base[
        "orientation"
    ]
)


print()
print("=" * 175)
print("SCALE / PHASE DIAGNOSTIC DECISION")
print("=" * 175)

print(
    "best multiplier:",
    f"{best_multiplier:.2f}x",
)

print(
    "all-start progress:",
    f"{base['progress']:.3f}",
    "->",
    f"{best['progress']:.3f}",
    f"({progress_gain:+.3f})",
)

print(
    "held-out progress:",
    f"{base_hold['progress']:.3f}",
    "->",
    f"{best_hold['progress']:.3f}",
    f"({heldout_gain:+.3f})",
)

print(
    "orientation:",
    f"{base['orientation']:.2f}",
    "->",
    f"{best['orientation']:.2f}",
    f"({orientation_change:+.2f} deg)",
)

print(
    "mean |X error|:",
    f"{base['x']:.3f}",
    "->",
    f"{best['x']:.3f}",
)

print(
    "mean |Y error|:",
    f"{base['y']:.3f}",
    "->",
    f"{best['y']:.3f}",
)

print(
    "mean |Z error|:",
    f"{base['z']:.3f}",
    "->",
    f"{best['z']:.3f}",
)

print(
    "X fraction of axis-wise root error:",
    f"{100*x_fraction:.1f}%",
)

print(
    "mean absolute final X error:",
    f"{best['final_x']:.3f} m",
)

print(
    "action clipping:",
    f"{100*best['clip']:.2f}%",
)


# ================================================================
# INTERPRETATION
# ================================================================

print()


if (
    heldout_gain
    >= 0.06

    and progress_gain
    >= 0.10

    and orientation_change
    <= 5.0

    and best[
        "clip"
    ] < 0.10

    and best_multiplier
    != 1.00
):

    print(
        "RESULT: ACTION SCALE WAS TOO CONSERVATIVE"
    )

    print(
        f"Use approximately {best_multiplier:.2f}x "
        "the current residual authority."
    )

    print(
        "NEXT: re-optimize state-feedback gains at "
        "this multiplier before PPO."
    )


elif (
    x_fraction
    >= 0.65

    and best[
        "x"
    ]
    >= 0.10

    and best[
        "y"
    ]
    <= 0.15

    and best[
        "z"
    ]
    <= 0.15
):

    print(
        "RESULT: FORWARD / PHASE ERROR DOMINATES"
    )

    print(
        "The robot can remain dynamically stable, "
        "but fixed-time reference progression is "
        "creating substantial forward-position error."
    )

    print(
        "NEXT: add controlled phase adaptation / "
        "reference-speed correction."
    )

    print(
        "Do NOT solve this by simply increasing "
        "residual authority."
    )


elif (
    heldout_gain
    < 0.06
):

    print(
        "RESULT: GENERALIZATION IS STILL LIMITED"
    )

    print(
        "Changing action magnitude alone does not "
        "solve the held-out-state problem."
    )

    print(
        "NEXT: improve the balance-channel/state "
        "representation before PPO."
    )


else:

    print(
        "RESULT: AUTHORITY IS ADEQUATE BUT NOT YET "
        "A CLEAN PPO GATE"
    )

    print(
        "NEXT: rerun state-feedback optimization "
        "using the best multiplier and decomposed "
        "root/phase objective."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_scale_phase_diagnostic.npz"
)


np.savez(
    output,

    multipliers=
        np.asarray(
            MULTIPLIERS,
            dtype=np.float32,
        ),

    best_multiplier=
        np.asarray(
            [
                best_multiplier
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

    heldout_gain=
        np.asarray(
            [
                heldout_gain
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

    x_fraction=
        np.asarray(
            [
                x_fraction
            ],
            dtype=np.float32,
        ),
)


print()
print(
    "diagnostic artifact:",
    output,
)

print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 175)


env.close()
