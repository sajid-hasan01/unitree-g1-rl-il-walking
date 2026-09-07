from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
    PENETRATION_WEIGHT,
)


# ============================================================
# EXISTING TRAINED V2 PPO
#
# THIS SCRIPT DOES NOT TRAIN OR MODIFY IT.
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)


SEED = 425
RESIDUAL_SMOOTHING = 0.35

EVAL_STATES = (
    TRAIN_END_STEP
    -
    TRAIN_START_STEP
    +
    1
)

EVAL_TRANSITIONS = (
    TRAIN_END_STEP
    -
    TRAIN_START_STEP
)


# ============================================================
# CANDIDATE WORST-SPHERE PENETRATION TERM
#
# Existing V2:
#
#   mean over all 8 phase-aware penetration costs
#
# Candidate additional term:
#
#   worst_cost =
#       (
#           max phase-aware penetration excess
#           /
#           PENETRATION_SCALE
#       ) ** 2
#
# PENETRATION_SCALE stays exactly the same as V2.
# ============================================================

WORST_WEIGHT_VALUES = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
]


# ============================================================
# KNOWN DIAGNOSTIC FRAMES
# ============================================================

SELECTED_STEPS = [
    395,
    435,
    487,
    488,
    624,
    680,
    681,
    682,
    683,
    689,
    690,
    698,
    699,
    708,
    715,
    716,
    720,
    723,
    728,
    741,
]


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model_spaces(
    model,
    env,
):

    if (
        model.observation_space.shape
        !=
        env.observation_space.shape
    ):
        raise RuntimeError(
            "Observation-space shape mismatch."
        )

    if (
        model.action_space.shape
        !=
        env.action_space.shape
    ):
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


# ============================================================
# PHASE-AWARE PENETRATION
# ============================================================

def penetration_terms(
    signed_distances,
    flat_reference,
):

    signed_distances = np.asarray(
        signed_distances,
        dtype=np.float64,
    )

    flat_reference = np.asarray(
        flat_reference,
        dtype=np.float64,
    )

    if signed_distances.shape != (8,):
        raise RuntimeError(
            "Expected 8 signed foot distances."
        )

    if flat_reference.shape != (8,):
        raise RuntimeError(
            "Expected 8 flat-reference distances."
        )


    # --------------------------------------------------------
    # ACTUAL RAW TERRAIN PENETRATION
    # --------------------------------------------------------

    actual_penetration = np.maximum(
        -signed_distances,
        0.0,
    )


    # --------------------------------------------------------
    # EXACT V2 PHASE-AWARE ALLOWED FLOOR
    #
    # This preserves negative same-phase flat-BC geometry.
    # --------------------------------------------------------

    allowed_floor = np.minimum(
        flat_reference,
        0.0,
    )


    # --------------------------------------------------------
    # EXACT V2 PHASE-AWARE PENETRATION EXCESS
    # --------------------------------------------------------

    penetration_excess = np.maximum(
        allowed_floor
        -
        signed_distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )


    # --------------------------------------------------------
    # EXISTING V2 MEAN PENETRATION COST
    # --------------------------------------------------------

    mean_cost = float(
        np.mean(
            (
                penetration_excess
                /
                PENETRATION_SCALE
            )
            **
            2
        )
    )


    # --------------------------------------------------------
    # CANDIDATE WORST-SPHERE PENETRATION COST
    # --------------------------------------------------------

    max_excess = float(
        np.max(
            penetration_excess
        )
    )

    worst_cost = float(
        (
            max_excess
            /
            PENETRATION_SCALE
        )
        **
        2
    )


    return {
        "actual_max_penetration":
            float(
                np.max(
                    actual_penetration
                )
            ),

        "mean_cost":
            mean_cost,

        "max_excess":
            max_excess,

        "worst_cost":
            worst_cost,

        "penetration_excess":
            penetration_excess,
    }


# ============================================================
# DETERMINISTIC ROLLOUT
# ============================================================

def run_rollout(
    env,
    model,
    use_ppo,
):

    observation, reset_info = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP,
        },
    )

    if (
        int(
            reset_info[
                "start_step"
            ]
        )
        !=
        TRAIN_START_STEP
    ):
        raise RuntimeError(
            "Unexpected reset start step."
        )

    records = []

    while (
        env.current_step
        <
        TRAIN_END_STEP
    ):

        if use_ppo:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

        else:

            action = np.zeros(
                12,
                dtype=np.float32,
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

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

        pen = penetration_terms(
            signed,
            flat,
        )

        env_penetration_cost = float(
            info[
                "penetration_cost"
            ]
        )

        if not np.isclose(
            pen[
                "mean_cost"
            ],
            env_penetration_cost,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Offline reconstruction of V2 "
                "penetration cost does not match "
                "environment value."
            )

        records.append(
            {
                "step":
                    int(
                        info[
                            "step"
                        ]
                    ),

                "world_x":
                    float(
                        info[
                            "world_x"
                        ]
                    ),

                "base_reward":
                    float(
                        reward
                    ),

                "actual_max_penetration":
                    pen[
                        "actual_max_penetration"
                    ],

                "mean_penetration_cost":
                    pen[
                        "mean_cost"
                    ],

                "max_phase_excess":
                    pen[
                        "max_excess"
                    ],

                "worst_penetration_cost":
                    pen[
                        "worst_cost"
                    ],

                "signed_distances":
                    signed,

                "flat_reference":
                    flat,
            }
        )

        if terminated:
            raise RuntimeError(
                "Unexpected termination at step "
                f"{info['step']}."
            )

        if (
            truncated
            and
            env.current_step
            <
            TRAIN_END_STEP
        ):
            raise RuntimeError(
                "Unexpected early truncation at step "
                f"{info['step']}."
            )

    if len(records) != EVAL_TRANSITIONS:
        raise RuntimeError(
            "Unexpected transition count: "
            f"{len(records)} != {EVAL_TRANSITIONS}"
        )

    return records


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

def to_arrays(
    records,
):

    scalar_keys = [
        "step",
        "world_x",
        "base_reward",
        "actual_max_penetration",
        "mean_penetration_cost",
        "max_phase_excess",
        "worst_penetration_cost",
    ]

    result = {}

    for key in scalar_keys:

        dtype = (
            np.int64
            if key == "step"
            else np.float64
        )

        result[
            key
        ] = np.asarray(
            [
                record[
                    key
                ]
                for record
                in records
            ],
            dtype=dtype,
        )

    result[
        "signed_distances"
    ] = np.asarray(
        [
            record[
                "signed_distances"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    result[
        "flat_reference"
    ] = np.asarray(
        [
            record[
                "flat_reference"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    return result


# ============================================================
# FAIR-COMPARISON VALIDATION
# ============================================================

def validate_fair_comparison(
    bc,
    ppo,
):

    if not np.array_equal(
        bc[
            "step"
        ],
        ppo[
            "step"
        ],
    ):
        raise RuntimeError(
            "BC/PPO step arrays differ."
        )

    max_x_difference = float(
        np.max(
            np.abs(
                bc[
                    "world_x"
                ]
                -
                ppo[
                    "world_x"
                ]
            )
        )
    )

    max_flat_difference = float(
        np.max(
            np.abs(
                bc[
                    "flat_reference"
                ]
                -
                ppo[
                    "flat_reference"
                ]
            )
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "FAIR-COMPARISON INVARIANTS"
    )

    print(
        "=" * 110
    )

    print(
        "Transitions:",
        len(
            bc[
                "step"
            ]
        ),
    )

    print(
        "Maximum world-X difference:",
        f"{max_x_difference:.12f}",
    )

    print(
        "Maximum flat-reference difference:",
        f"{max_flat_difference:.12f}",
    )

    if max_x_difference > 1e-12:
        raise RuntimeError(
            "World-X invariant FAILED."
        )

    if max_flat_difference > 1e-12:
        raise RuntimeError(
            "Flat-reference invariant FAILED."
        )

    print(
        "PASS"
    )


# ============================================================
# REWARD RESCORING
# ============================================================

def rescore(
    bc,
    ppo,
    weight,
):

    bc_adjusted = (
        bc[
            "base_reward"
        ]
        -
        float(
            weight
        )
        *
        bc[
            "worst_penetration_cost"
        ]
    )

    ppo_adjusted = (
        ppo[
            "base_reward"
        ]
        -
        float(
            weight
        )
        *
        ppo[
            "worst_penetration_cost"
        ]
    )

    delta = (
        ppo_adjusted
        -
        bc_adjusted
    )

    epsilon = 1e-9

    better = int(
        np.count_nonzero(
            delta
            >
            epsilon
        )
    )

    same = int(
        np.count_nonzero(
            np.abs(
                delta
            )
            <=
            epsilon
        )
    )

    worse = int(
        np.count_nonzero(
            delta
            <
            -epsilon
        )
    )

    return {
        "bc_mean":
            float(
                np.mean(
                    bc_adjusted
                )
            ),

        "ppo_mean":
            float(
                np.mean(
                    ppo_adjusted
                )
            ),

        "delta_mean":
            float(
                np.mean(
                    delta
                )
            ),

        "better":
            better,

        "same":
            same,

        "worse":
            worse,
    }


# ============================================================
# THRESHOLD WEIGHT
# ============================================================

def threshold_weight(
    bc_reward,
    ppo_reward,
    bc_cost,
    ppo_cost,
):

    reward_advantage = (
        ppo_reward
        -
        bc_reward
    )

    if reward_advantage <= 0.0:
        return (
            "already BC-preferred"
        )

    cost_difference = (
        ppo_cost
        -
        bc_cost
    )

    if cost_difference <= 0.0:
        return (
            "cannot flip with positive weight"
        )

    threshold = (
        reward_advantage
        /
        cost_difference
    )

    return (
        f"{threshold:.6f}"
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def print_global_summary(
    bc,
    ppo,
):

    print()
    print(
        "=" * 110
    )

    print(
        "CURRENT V2 PENETRATION SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        "Existing penetration weight:",
        PENETRATION_WEIGHT,
    )

    print(
        "Penetration scale:",
        f"{PENETRATION_SCALE * 1000:.1f} mm",
    )

    print(
        "Tolerance:",
        f"{FLAT_DISTANCE_TOLERANCE * 1000:.1f} mm",
    )

    print()

    print(
        "BC:"
    )

    print(
        "  mean V2 average penetration cost:",
        (
            f"{np.mean(bc['mean_penetration_cost']):.6f}"
        ),
    )

    print(
        "  mean candidate worst-sphere cost:",
        (
            f"{np.mean(bc['worst_penetration_cost']):.6f}"
        ),
    )

    print(
        "  max actual penetration:",
        (
            f"{np.max(bc['actual_max_penetration']) * 1000:.3f} mm"
        ),
    )

    print(
        "  max phase-aware excess:",
        (
            f"{np.max(bc['max_phase_excess']) * 1000:.3f} mm"
        ),
    )

    print()

    print(
        "V2 PPO:"
    )

    print(
        "  mean V2 average penetration cost:",
        (
            f"{np.mean(ppo['mean_penetration_cost']):.6f}"
        ),
    )

    print(
        "  mean candidate worst-sphere cost:",
        (
            f"{np.mean(ppo['worst_penetration_cost']):.6f}"
        ),
    )

    print(
        "  max actual penetration:",
        (
            f"{np.max(ppo['actual_max_penetration']) * 1000:.3f} mm"
        ),
    )

    print(
        "  max phase-aware excess:",
        (
            f"{np.max(ppo['max_phase_excess']) * 1000:.3f} mm"
        ),
    )

    print()

    print(
        "Base Reward-V2 mean:"
    )

    print(
        "  BC :",
        f"{np.mean(bc['base_reward']):+.6f}",
    )

    print(
        "  PPO:",
        f"{np.mean(ppo['base_reward']):+.6f}",
    )

    print(
        "  PPO - BC:",
        (
            f"{np.mean(ppo['base_reward']) - np.mean(bc['base_reward']):+.6f}"
        ),
    )


# ============================================================
# WEIGHT SWEEP
# ============================================================

def print_weight_sweep(
    bc,
    ppo,
):

    print()
    print(
        "=" * 110
    )

    print(
        "CANDIDATE WORST-SPHERE WEIGHT SWEEP"
    )

    print(
        "=" * 110
    )

    print(
        "weight | BC meanR | PPO meanR | "
        "deltaR   | PPO better | same | PPO worse"
    )

    print(
        "-" * 90
    )

    for weight in (
        WORST_WEIGHT_VALUES
    ):

        result = rescore(
            bc,
            ppo,
            weight,
        )

        print(
            f"{weight:6.2f} | "
            f"{result['bc_mean']:+8.4f} | "
            f"{result['ppo_mean']:+9.4f} | "
            f"{result['delta_mean']:+8.4f} | "
            f"{result['better']:10d} | "
            f"{result['same']:4d} | "
            f"{result['worse']:9d}"
        )


# ============================================================
# SELECTED FRAME TABLE
# ============================================================

def print_selected_frames(
    bc,
    ppo,
):

    print()
    print(
        "=" * 155
    )

    print(
        "SELECTED PENETRATION FRAMES"
    )

    print(
        "=" * 155
    )

    print(
        "step | x       | "
        "BC pen | PPO pen | "
        "BC phaseEx | PPO phaseEx | "
        "BC avgCost | PPO avgCost | "
        "BC worst | PPO worst | "
        "base PPO-BC"
    )

    print(
        "-" * 155
    )

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            ppo[
                "step"
            ]
            ==
            requested_step
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

        reward_advantage = (
            ppo[
                "base_reward"
            ][
                index
            ]
            -
            bc[
                "base_reward"
            ][
                index
            ]
        )

        print(
            f"{requested_step:4d} | "
            f"{ppo['world_x'][index]:+7.3f} | "
            f"{bc['actual_max_penetration'][index] * 1000:6.1f} | "
            f"{ppo['actual_max_penetration'][index] * 1000:7.1f} | "
            f"{bc['max_phase_excess'][index] * 1000:10.1f} | "
            f"{ppo['max_phase_excess'][index] * 1000:11.1f} | "
            f"{bc['mean_penetration_cost'][index]:10.4f} | "
            f"{ppo['mean_penetration_cost'][index]:11.4f} | "
            f"{bc['worst_penetration_cost'][index]:8.4f} | "
            f"{ppo['worst_penetration_cost'][index]:9.4f} | "
            f"{reward_advantage:+10.4f}"
        )


# ============================================================
# FRAME-SPECIFIC THRESHOLDS
# ============================================================

def print_thresholds(
    bc,
    ppo,
):

    print()
    print(
        "=" * 110
    )

    print(
        "MINIMUM WORST-SPHERE WEIGHT TO REMOVE PPO "
        "REWARD ADVANTAGE AT SELECTED FRAMES"
    )

    print(
        "=" * 110
    )

    print(
        "step | threshold"
    )

    print(
        "-" * 60
    )

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            ppo[
                "step"
            ]
            ==
            requested_step
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

        threshold = threshold_weight(
            bc[
                "base_reward"
            ][
                index
            ],

            ppo[
                "base_reward"
            ][
                index
            ],

            bc[
                "worst_penetration_cost"
            ][
                index
            ],

            ppo[
                "worst_penetration_cost"
            ][
                index
            ],
        )

        print(
            f"{requested_step:4d} | "
            f"{threshold}"
        )


# ============================================================
# WORST CURRENT PPO FRAMES
# ============================================================

def print_worst_ppo_frames(
    bc,
    ppo,
    count=20,
):

    order = np.argsort(
        ppo[
            "max_phase_excess"
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
        "WORST CURRENT V2 PPO PHASE-AWARE PENETRATION FRAMES"
    )

    print(
        "=" * 140
    )

    print(
        "rank | step | x       | "
        "BC pen | PPO pen | "
        "BC phaseEx | PPO phaseEx | "
        "PPO avgCost | PPO worstCost | "
        "base PPO-BC"
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

        advantage = (
            ppo[
                "base_reward"
            ][
                index
            ]
            -
            bc[
                "base_reward"
            ][
                index
            ]
        )

        print(
            f"{rank:4d} | "
            f"{ppo['step'][index]:4d} | "
            f"{ppo['world_x'][index]:+7.3f} | "
            f"{bc['actual_max_penetration'][index] * 1000:6.1f} | "
            f"{ppo['actual_max_penetration'][index] * 1000:7.1f} | "
            f"{bc['max_phase_excess'][index] * 1000:10.1f} | "
            f"{ppo['max_phase_excess'][index] * 1000:11.1f} | "
            f"{ppo['mean_penetration_cost'][index]:11.4f} | "
            f"{ppo['worst_penetration_cost'][index]:13.4f} | "
            f"{advantage:+10.4f}"
        )


# ============================================================
# HIGH-SEVERITY FRAME COUNTS
# ============================================================

def print_severity_counts(
    bc,
    ppo,
):

    thresholds_mm = [
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "ACTUAL MAX-PENETRATION SEVERITY COUNTS"
    )

    print(
        "=" * 100
    )

    print(
        "threshold | BC frames | PPO frames"
    )

    print(
        "-" * 50
    )

    for threshold_mm in (
        thresholds_mm
    ):

        threshold_m = (
            threshold_mm
            /
            1000.0
        )

        bc_count = int(
            np.count_nonzero(
                bc[
                    "actual_max_penetration"
                ]
                >=
                threshold_m
            )
        )

        ppo_count = int(
            np.count_nonzero(
                ppo[
                    "actual_max_penetration"
                ]
                >=
                threshold_m
            )
        )

        print(
            f"{threshold_mm:8.1f} | "
            f"{bc_count:9d} | "
            f"{ppo_count:10d}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Reward-V2 PPO checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 120
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "WORST-SPHERE PENETRATION PENALTY OFFLINE SWEEP"
    )

    print(
        "=" * 120
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Reference range:",
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
    )

    print(
        "Transitions:",
        EVAL_TRANSITIONS,
    )

    print()

    print(
        "Current V2 penetration objective:"
    )

    print(
        "  mean squared phase-aware excess "
        "across 8 sole spheres"
    )

    print()

    print(
        "Candidate additional objective:"
    )

    print(
        "  squared WORST phase-aware excess "
        "among the 8 sole spheres"
    )

    print()

    print(
        "This experiment changes ONLY offline reward scoring."
    )

    print(
        "No training."
    )

    print(
        "No action changes."
    )

    print(
        "No hip/knee changes."
    )

    print(
        "No gait/posture penalty."
    )

    print(
        "No environment modification."
    )

    print(
        "No mujoco.mj_step()."
    )

    print(
        "=" * 120
    )


    # ========================================================
    # ENVIRONMENTS
    # ========================================================

    bc_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    ppo_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

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
            ppo_env,
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # BC
        # ====================================================

        print()
        print(
            "Running BC-only zero-residual trajectory..."
        )

        bc_records = run_rollout(
            bc_env,
            model=None,
            use_ppo=False,
        )


        # ====================================================
        # PPO
        # ====================================================

        print(
            "Running deterministic Reward-V2 PPO trajectory..."
        )

        ppo_records = run_rollout(
            ppo_env,
            model=model,
            use_ppo=True,
        )


        bc = to_arrays(
            bc_records
        )

        ppo = to_arrays(
            ppo_records
        )


        # ====================================================
        # CHECKS
        # ====================================================

        validate_fair_comparison(
            bc,
            ppo,
        )


        # ====================================================
        # REPORTS
        # ====================================================

        print_global_summary(
            bc,
            ppo,
        )

        print_weight_sweep(
            bc,
            ppo,
        )

        print_selected_frames(
            bc,
            ppo,
        )

        print_thresholds(
            bc,
            ppo,
        )

        print_worst_ppo_frames(
            bc,
            ppo,
            count=20,
        )

        print_severity_counts(
            bc,
            ppo,
        )


        print()
        print(
            "=" * 120
        )

        print(
            "WORST-SPHERE PENETRATION SWEEP COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No checkpoint was modified."
        )

        print(
            "No environment file was modified."
        )

        print()

        print(
            "Do not select a weight until "
            "the complete output is reviewed."
        )

        print(
            "=" * 120
        )


    finally:

        bc_env.close()
        ppo_env.close()


if __name__ == "__main__":
    main()