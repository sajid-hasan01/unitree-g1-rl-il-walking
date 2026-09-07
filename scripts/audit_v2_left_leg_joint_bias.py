from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)

SEED = 425
RESIDUAL_SMOOTHING = 0.35

LEFT_SAGITTAL = np.array(
    [0, 3, 4],
    dtype=np.int64,
)

RIGHT_SAGITTAL = np.array(
    [6, 9, 10],
    dtype=np.int64,
)

SELECTED_STEPS = [
    175,
    250,
    325,
    395,
    435,
    500,
    575,
    624,
    650,
    675,
    680,
    681,
    682,
    683,
    689,
    698,
    708,
    716,
    720,
    723,
    728,
    735,
    741,
    750,
]

RANGES = [
    ("early", 151, 300),
    ("middle", 301, 649),
    ("late", 650, 750),
    ("worst_pen_region", 675, 700),
    ("late_orientation_region", 708, 735),
]


def validate_model_spaces(model, env):

    if model.observation_space.shape != env.observation_space.shape:
        raise RuntimeError(
            "Observation-space shape mismatch."
        )

    if model.action_space.shape != env.action_space.shape:
        raise RuntimeError(
            "Action-space shape mismatch."
        )

    if not np.allclose(
        model.observation_space.low,
        env.observation_space.low,
    ):
        raise RuntimeError(
            "Observation lower-bound mismatch."
        )

    if not np.allclose(
        model.observation_space.high,
        env.observation_space.high,
    ):
        raise RuntimeError(
            "Observation upper-bound mismatch."
        )

    if not np.allclose(
        model.action_space.low,
        env.action_space.low,
    ):
        raise RuntimeError(
            "Action lower-bound mismatch."
        )

    if not np.allclose(
        model.action_space.high,
        env.action_space.high,
    ):
        raise RuntimeError(
            "Action upper-bound mismatch."
        )


def foot_gap(
    distances,
    heel_slice,
    toe_slice,
):

    return float(
        np.mean(
            distances[
                heel_slice
            ]
        )
        -
        np.mean(
            distances[
                toe_slice
            ]
        )
    )


def collect_rollout(
    env,
    model,
):

    obs, info = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP
        },
    )

    if int(
        info[
            "start_step"
        ]
    ) != TRAIN_START_STEP:
        raise RuntimeError(
            "Unexpected reset start step."
        )

    names = list(
        env.baseline.JOINT_NAMES[
            :12
        ]
    )

    rows = []

    while (
        env.current_step
        <
        TRAIN_END_STEP
    ):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
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

        base = np.asarray(
            env.current_metadata[
                "base_joints"
            ][
                :12
            ],
            dtype=np.float64,
        ).copy()

        applied = np.asarray(
            env.current_metadata[
                "applied_joints"
            ][
                :12
            ],
            dtype=np.float64,
        ).copy()

        residual = np.asarray(
            env.previous_residual,
            dtype=np.float64,
        ).copy()

        signed = np.asarray(
            info[
                "signed_distances"
            ],
            dtype=np.float64,
        ).copy()

        flat = np.asarray(
            info[
                "flat_reference_distances"
            ],
            dtype=np.float64,
        ).copy()

        rows.append(
            {
                "step":
                    int(
                        info[
                            "step"
                        ]
                    ),

                "x":
                    float(
                        info[
                            "world_x"
                        ]
                    ),

                "reward":
                    float(
                        reward
                    ),

                "action":
                    action
                    .astype(
                        np.float64
                    )
                    .copy(),

                "base":
                    base,

                "applied":
                    applied,

                "residual":
                    residual,

                "penetration":
                    float(
                        np.max(
                            np.maximum(
                                -signed,
                                0.0,
                            )
                        )
                    ),

                "penetration_cost":
                    float(
                        info[
                            "penetration_cost"
                        ]
                    ),

                "left_gap":
                    foot_gap(
                        signed,
                        slice(
                            0,
                            2,
                        ),
                        slice(
                            2,
                            4,
                        ),
                    ),

                "flat_left_gap":
                    foot_gap(
                        flat,
                        slice(
                            0,
                            2,
                        ),
                        slice(
                            2,
                            4,
                        ),
                    ),
            }
        )

        if terminated:
            raise RuntimeError(
                f"Unexpected termination at step {info['step']}."
            )

        if (
            truncated
            and
            env.current_step
            <
            TRAIN_END_STEP
        ):
            raise RuntimeError(
                f"Unexpected early truncation at step {info['step']}."
            )

    return (
        names,
        rows,
    )


def arrays(
    rows,
):

    result = {}

    for key in [
        "step",
        "x",
        "reward",
        "penetration",
        "penetration_cost",
        "left_gap",
        "flat_left_gap",
    ]:

        dtype = (
            np.int64
            if key == "step"
            else np.float64
        )

        result[
            key
        ] = np.asarray(
            [
                row[
                    key
                ]
                for row in rows
            ],
            dtype=dtype,
        )

    for key in [
        "action",
        "base",
        "applied",
        "residual",
    ]:

        result[
            key
        ] = np.asarray(
            [
                row[
                    key
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

    return result


def print_consistency(
    data,
):

    error = np.max(
        np.abs(
            (
                data[
                    "applied"
                ]
                -
                data[
                    "base"
                ]
            )
            -
            data[
                "residual"
            ]
        )
    )

    print()
    print(
        "=" * 105
    )

    print(
        "CONSISTENCY CHECKS"
    )

    print(
        "=" * 105
    )

    print(
        "Transitions:",
        len(
            data[
                "step"
            ]
        ),
    )

    print(
        "Step range:",
        (
            f"{data['step'][0]} -> "
            f"{data['step'][-1]}"
        ),
    )

    print(
        "max |(applied - BC) - residual|:",
        f"{error:.12e} rad",
    )

    print(
        "max |action|:",
        f"{np.max(np.abs(data['action'])):.9f}",
    )

    print(
        "near-boundary action samples |a|>=0.95:",
        int(
            np.count_nonzero(
                np.abs(
                    data[
                        "action"
                    ]
                )
                >=
                0.95
            )
        ),
    )

    print(
        "max penetration:",
        (
            f"{np.max(data['penetration']) * 1000:.3f} mm"
        ),
    )


def print_joint_table(
    names,
    data,
):

    print()
    print(
        "=" * 150
    )

    print(
        "FULL-TRAJECTORY PPO JOINT BIAS RELATIVE TO SAME-PHASE BC"
    )

    print(
        "=" * 150
    )

    print(
        "idx | joint                         | "
        "mean dQ | median dQ | mean|dQ| | "
        "p05 dQ | p95 dQ | max|dQ| | "
        "mean action | max|action| | |a|>=.95"
    )

    print(
        "-" * 150
    )

    for (
        index,
        name,
    ) in enumerate(
        names
    ):

        dq = data[
            "residual"
        ][
            :,
            index
        ]

        action = data[
            "action"
        ][
            :,
            index
        ]

        near_bound = int(
            np.count_nonzero(
                np.abs(
                    action
                )
                >=
                0.95
            )
        )

        print(
            f"{index:3d} | "
            f"{name:29s} | "
            f"{np.mean(dq):+7.4f} | "
            f"{np.median(dq):+9.4f} | "
            f"{np.mean(np.abs(dq)):8.4f} | "
            f"{np.percentile(dq, 5):+7.4f} | "
            f"{np.percentile(dq, 95):+7.4f} | "
            f"{np.max(np.abs(dq)):8.4f} | "
            f"{np.mean(action):+11.4f} | "
            f"{np.max(np.abs(action)):11.4f} | "
            f"{near_bound:8d}"
        )


def print_left_right_asymmetry(
    data,
):

    labels = [
        "hip pitch",
        "hip roll",
        "hip yaw",
        "knee",
        "ankle pitch",
        "ankle roll",
    ]

    print()
    print(
        "=" * 110
    )

    print(
        "LEFT-vs-RIGHT RESIDUAL ASYMMETRY"
    )

    print(
        "=" * 110
    )

    print(
        "joint type       | "
        "left mean dQ | right mean dQ | "
        "left mean|dQ| | right mean|dQ| | mean L-R"
    )

    print(
        "-" * 110
    )

    for (
        local_index,
        label,
    ) in enumerate(
        labels
    ):

        left_index = (
            local_index
        )

        right_index = (
            local_index
            +
            6
        )

        left = data[
            "residual"
        ][
            :,
            left_index
        ]

        right = data[
            "residual"
        ][
            :,
            right_index
        ]

        print(
            f"{label:16s} | "
            f"{np.mean(left):+12.4f} | "
            f"{np.mean(right):+13.4f} | "
            f"{np.mean(np.abs(left)):13.4f} | "
            f"{np.mean(np.abs(right)):14.4f} | "
            f"{np.mean(left - right):+9.4f}"
        )


def print_region_summary(
    data,
):

    steps = data[
        "step"
    ]

    residual = data[
        "residual"
    ]

    print()
    print(
        "=" * 130
    )

    print(
        "SAGITTAL RESIDUALS BY TRAJECTORY REGION"
    )

    print(
        "=" * 130
    )

    print(
        "region                   | n   | "
        "L hipP | L knee | L ankleP | "
        "R hipP | R knee | R ankleP | "
        "L sagittal RMS"
    )

    print(
        "-" * 130
    )

    for (
        name,
        start,
        end,
    ) in RANGES:

        mask = (
            (
                steps
                >=
                start
            )
            &
            (
                steps
                <=
                end
            )
        )

        left_rms = np.sqrt(
            np.mean(
                residual[
                    mask
                ][
                    :,
                    LEFT_SAGITTAL
                ]
                **
                2
            )
        )

        print(
            f"{name:24s} | "
            f"{np.count_nonzero(mask):3d} | "
            f"{np.mean(residual[mask, 0]):+6.3f} | "
            f"{np.mean(residual[mask, 3]):+6.3f} | "
            f"{np.mean(residual[mask, 4]):+8.3f} | "
            f"{np.mean(residual[mask, 6]):+6.3f} | "
            f"{np.mean(residual[mask, 9]):+6.3f} | "
            f"{np.mean(residual[mask, 10]):+8.3f} | "
            f"{left_rms:14.4f}"
        )


def print_selected(
    data,
):

    print()
    print(
        "=" * 170
    )

    print(
        "SELECTED FRAMES: LEFT-LEG SAGITTAL POSTURE"
    )

    print(
        "=" * 170
    )

    print(
        "step | x       | pen mm | Lgap | flatL | "
        "BC hipP | PPO hipP | dHipP | "
        "BC knee | PPO knee | dKnee | "
        "BC ankP | PPO ankP | dAnkP | "
        "L residual norm"
    )

    print(
        "-" * 170
    )

    for step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            data[
                "step"
            ]
            ==
            step
        )[
            0
        ]

        if len(
            matches
        ) != 1:
            continue

        index = int(
            matches[
                0
            ]
        )

        norm = np.linalg.norm(
            data[
                "residual"
            ][
                index,
                LEFT_SAGITTAL
            ]
        )

        print(
            f"{step:4d} | "
            f"{data['x'][index]:+7.3f} | "
            f"{data['penetration'][index] * 1000:6.1f} | "
            f"{data['left_gap'][index] * 1000:+6.1f} | "
            f"{data['flat_left_gap'][index] * 1000:+6.1f} | "
            f"{data['base'][index, 0]:+7.3f} | "
            f"{data['applied'][index, 0]:+8.3f} | "
            f"{data['residual'][index, 0]:+6.3f} | "
            f"{data['base'][index, 3]:+7.3f} | "
            f"{data['applied'][index, 3]:+8.3f} | "
            f"{data['residual'][index, 3]:+6.3f} | "
            f"{data['base'][index, 4]:+7.3f} | "
            f"{data['applied'][index, 4]:+8.3f} | "
            f"{data['residual'][index, 4]:+6.3f} | "
            f"{norm:15.4f}"
        )


def print_top_left_deviations(
    data,
    count=20,
):

    residual = data[
        "residual"
    ]

    norm = np.linalg.norm(
        residual[
            :,
            LEFT_SAGITTAL
        ],
        axis=1,
    )

    order = np.argsort(
        norm
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 135
    )

    print(
        "TOP LEFT-SAGITTAL DEVIATIONS FROM BC"
    )

    print(
        "=" * 135
    )

    print(
        "rank | step | pen mm | Lgap | "
        "dHipPitch | dKnee | dAnklePitch | "
        "norm | aHipPitch | aKnee | aAnklePitch"
    )

    print(
        "-" * 135
    )

    for (
        rank,
        index,
    ) in enumerate(
        order,
        start=1,
    ):

        print(
            f"{rank:4d} | "
            f"{data['step'][index]:4d} | "
            f"{data['penetration'][index] * 1000:6.1f} | "
            f"{data['left_gap'][index] * 1000:+6.1f} | "
            f"{residual[index, 0]:+9.4f} | "
            f"{residual[index, 3]:+6.4f} | "
            f"{residual[index, 4]:+11.4f} | "
            f"{norm[index]:5.4f} | "
            f"{data['action'][index, 0]:+9.4f} | "
            f"{data['action'][index, 3]:+6.4f} | "
            f"{data['action'][index, 4]:+11.4f}"
        )


def print_worst_penetration(
    data,
    count=15,
):

    residual = data[
        "residual"
    ]

    order = np.argsort(
        data[
            "penetration"
        ]
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 140
    )

    print(
        "WORST REMAINING PENETRATION FRAMES + SAGITTAL RESIDUALS"
    )

    print(
        "=" * 140
    )

    print(
        "rank | step | pen mm | penCost | Lgap | "
        "L dHipP | L dKnee | L dAnkP | "
        "R dHipP | R dKnee | R dAnkP"
    )

    print(
        "-" * 140
    )

    for (
        rank,
        index,
    ) in enumerate(
        order,
        start=1,
    ):

        print(
            f"{rank:4d} | "
            f"{data['step'][index]:4d} | "
            f"{data['penetration'][index] * 1000:6.1f} | "
            f"{data['penetration_cost'][index]:7.4f} | "
            f"{data['left_gap'][index] * 1000:+6.1f} | "
            f"{residual[index, 0]:+7.4f} | "
            f"{residual[index, 3]:+7.4f} | "
            f"{residual[index, 4]:+8.4f} | "
            f"{residual[index, 6]:+7.4f} | "
            f"{residual[index, 9]:+7.4f} | "
            f"{residual[index, 10]:+8.4f}"
        )


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Reward-V2 PPO checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 110
    )

    print(
        "UNITREE G1 REWARD-V2 LEFT-LEG POSTURE / JOINT-BIAS AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Purpose: identify exactly which residual joints "
        "create the visible left-leg bend/pull."
    )

    print(
        "No training. No reward modification. No mj_step()."
    )

    print(
        "=" * 110
    )

    env = G1KinematicUnevenEnv(
        episode_length=
            (
                TRAIN_END_STEP
                -
                TRAIN_START_STEP
                +
                1
            ),

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    try:

        print()
        print(
            "Loading Reward-V2 PPO..."
        )

        model = PPO.load(
            MODEL_PATH,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            model,
            env,
        )

        print(
            "Model-space validation: PASS"
        )

        print(
            "Running deterministic Reward-V2 trajectory..."
        )

        names, rows = collect_rollout(
            env,
            model,
        )

        data = arrays(
            rows
        )

        print_consistency(
            data
        )

        print_joint_table(
            names,
            data,
        )

        print_left_right_asymmetry(
            data
        )

        print_region_summary(
            data
        )

        print_selected(
            data
        )

        print_top_left_deviations(
            data,
            count=20,
        )

        print_worst_penetration(
            data,
            count=15,
        )

        print()
        print(
            "=" * 110
        )

        print(
            "AUDIT COMPLETE"
        )

        print(
            "=" * 110
        )

        print(
            "No training was performed."
        )

        print(
            "No environment/model file was modified."
        )

        print(
            "Review measured hip-pitch, knee, and "
            "ankle-pitch biases before changing Reward V2."
        )

        print(
            "=" * 110
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()