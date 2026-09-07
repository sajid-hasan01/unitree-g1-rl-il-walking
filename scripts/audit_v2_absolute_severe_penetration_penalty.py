from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# EXACT ORIGINAL REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    PENETRATION_SCALE,
)


# ============================================================
# CHECKPOINTS
# ============================================================

ORIGINAL_V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)

BALANCED_V2_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_smoke_seed425.zip"
)


# ============================================================
# EVALUATION
# ============================================================

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
# CANDIDATE ABSOLUTE-SEVERE PENETRATION THRESHOLDS
#
# THESE ARE HYPERPARAMETER CANDIDATES.
#
# They are NOT physical facts.
#
# The goal is to find a threshold that leaves small/normal
# penetration handling to existing phase-aware Reward V2,
# while adding a strong signal against visually severe
# penetration.
# ============================================================

THRESHOLDS_MM = [
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
]


# ============================================================
# CANDIDATE WEIGHTS
# ============================================================

WEIGHTS = [
    0.25,
    0.50,
    1.00,
    2.00,
    4.00,
    6.00,
    8.00,
]


# ============================================================
# DIAGNOSTIC REGIONS
# ============================================================

REGIONS = [
    (
        "middle_486_495",
        486,
        495,
    ),

    (
        "middle_529_533",
        529,
        533,
    ),

    (
        "old_failure_675_700",
        675,
        700,
    ),

    (
        "late_708_735",
        708,
        735,
    ),
]


SELECTED_STEPS = [
    395,
    435,

    486,
    487,
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,

    529,
    530,
    531,
    532,
    533,

    624,

    679,
    680,
    681,
    682,
    683,
    689,
    690,
    698,
    699,
    700,

    708,
    710,
    711,
    713,
    715,
    716,
    720,
    723,
    728,

    730,
    731,
    732,
    733,

    741,
    750,
]


# ============================================================
# HELPERS
# ============================================================

def require(
    condition,
    message,
):

    if not condition:

        raise RuntimeError(
            message
        )


def validate_model_spaces(
    model,
    env,
    name,
):

    require(
        model.observation_space.shape
        ==
        env.observation_space.shape,
        f"{name}: observation shape mismatch.",
    )

    require(
        model.action_space.shape
        ==
        env.action_space.shape,
        f"{name}: action shape mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.low,
            env.observation_space.low,
        ),
        f"{name}: observation low mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.high,
            env.observation_space.high,
        ),
        f"{name}: observation high mismatch.",
    )

    require(
        np.allclose(
            model.action_space.low,
            env.action_space.low,
        ),
        f"{name}: action low mismatch.",
    )

    require(
        np.allclose(
            model.action_space.high,
            env.action_space.high,
        ),
        f"{name}: action high mismatch.",
    )


# ============================================================
# ROLLOUT
# ============================================================

def run_rollout(
    env,
    model=None,
):

    observation, reset_info = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP,
        },
    )

    require(
        int(
            reset_info[
                "start_step"
            ]
        )
        ==
        TRAIN_START_STEP,
        "Unexpected start step.",
    )

    records = []

    for transition_index in range(
        EVAL_TRANSITIONS
    ):

        if model is None:

            action = np.zeros(
                12,
                dtype=np.float32,
            )

        else:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
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

        require(
            terminated
            is False,
            (
                "Unexpected termination "
                f"at step {info['step']}."
            ),
        )

        if (
            transition_index
            <
            EVAL_TRANSITIONS - 1
        ):

            require(
                truncated
                is False,
                (
                    "Unexpected early truncation "
                    f"at step {info['step']}."
                ),
            )

        distances = np.asarray(
            info[
                "signed_distances"
            ],
            dtype=np.float64,
        )

        actual_penetration = np.maximum(
            -distances,
            0.0,
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

                "reward":
                    float(
                        reward
                    ),

                "max_actual_penetration":
                    float(
                        np.max(
                            actual_penetration
                        )
                    ),

                "max_phase_excess":
                    float(
                        info[
                            "max_penetration_excess_m"
                        ]
                    ),

                "flat_reference":
                    np.asarray(
                        info[
                            "flat_reference_distances"
                        ],
                        dtype=np.float64,
                    ).copy(),
            }
        )

    require(
        len(
            records
        )
        ==
        EVAL_TRANSITIONS,
        "Unexpected rollout length.",
    )

    return records


# ============================================================
# ARRAY CONVERSION
# ============================================================

def to_arrays(
    records,
):

    output = {}

    for key in [
        "step",
        "world_x",
        "reward",
        "max_actual_penetration",
        "max_phase_excess",
    ]:

        dtype = (
            np.int64
            if key == "step"
            else np.float64
        )

        output[
            key
        ] = np.asarray(
            [
                row[
                    key
                ]
                for row in records
            ],
            dtype=dtype,
        )

    output[
        "flat_reference"
    ] = np.asarray(
        [
            row[
                "flat_reference"
            ]
            for row in records
        ],
        dtype=np.float64,
    )

    return output


# ============================================================
# CANDIDATE ABSOLUTE-SEVERE COST
# ============================================================

def severe_cost(
    max_actual_penetration,
    threshold_mm,
):

    threshold_m = (
        float(
            threshold_mm
        )
        /
        1000.0
    )

    excess = np.maximum(
        max_actual_penetration
        -
        threshold_m,
        0.0,
    )

    cost = (
        excess
        /
        PENETRATION_SCALE
    ) ** 2

    return (
        excess,
        cost,
    )


# ============================================================
# FAIR COMPARISON
# ============================================================

def validate_fair_comparison(
    bc,
    original,
    balanced,
):

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

    for (
        name,
        candidate,
    ) in [
        (
            "Original V2",
            original,
        ),

        (
            "Balanced V2",
            balanced,
        ),
    ]:

        step_diff = float(
            np.max(
                np.abs(
                    bc[
                        "step"
                    ]
                    -
                    candidate[
                        "step"
                    ]
                )
            )
        )

        x_diff = float(
            np.max(
                np.abs(
                    bc[
                        "world_x"
                    ]
                    -
                    candidate[
                        "world_x"
                    ]
                )
            )
        )

        flat_diff = float(
            np.max(
                np.abs(
                    bc[
                        "flat_reference"
                    ]
                    -
                    candidate[
                        "flat_reference"
                    ]
                )
            )
        )

        print(
            f"{name:16s} | "
            f"stepDiff={step_diff:.1f} | "
            f"xDiff={x_diff:.12f} | "
            f"flatDiff={flat_diff:.12f}"
        )

        require(
            step_diff
            ==
            0.0,
            f"{name}: step invariant failed.",
        )

        require(
            x_diff
            <=
            1e-12,
            f"{name}: X invariant failed.",
        )

        require(
            flat_diff
            <=
            1e-12,
            f"{name}: flat invariant failed.",
        )

    print(
        "PASS"
    )


# ============================================================
# BASE SUMMARY
# ============================================================

def print_base_summary(
    results,
):

    print()
    print(
        "=" * 110
    )

    print(
        "CURRENT RAW-PENETRATION SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        "policy       | mean maxPen | maximum | "
        ">=20 | >=30 | >=40 | >=50 | >=60 | mean Reward"
    )

    print(
        "-" * 110
    )

    for name in [
        "BC",
        "V2",
        "BALANCED",
    ]:

        r = results[
            name
        ]

        pen = r[
            "max_actual_penetration"
        ]

        print(
            f"{name:12s} | "
            f"{np.mean(pen) * 1000:11.3f} | "
            f"{np.max(pen) * 1000:7.3f} | "
            f"{np.count_nonzero(pen >= 0.020):4d} | "
            f"{np.count_nonzero(pen >= 0.030):4d} | "
            f"{np.count_nonzero(pen >= 0.040):4d} | "
            f"{np.count_nonzero(pen >= 0.050):4d} | "
            f"{np.count_nonzero(pen >= 0.060):4d} | "
            f"{np.mean(r['reward']):+11.5f}"
        )


# ============================================================
# THRESHOLD DIAGNOSTICS
# ============================================================

def print_threshold_summary(
    results,
):

    print()
    print(
        "=" * 150
    )

    print(
        "ABSOLUTE-SEVERE COST BY THRESHOLD"
    )

    print(
        "=" * 150
    )

    print(
        "threshold | policy       | "
        "activeFrames | meanCost | maxCost | "
        "meanPenaltyInput mm | maxPenaltyInput mm"
    )

    print(
        "-" * 150
    )

    for threshold_mm in (
        THRESHOLDS_MM
    ):

        for name in [
            "BC",
            "V2",
            "BALANCED",
        ]:

            r = results[
                name
            ]

            excess, cost = severe_cost(
                r[
                    "max_actual_penetration"
                ],
                threshold_mm,
            )

            active = (
                excess
                >
                0.0
            )

            print(
                f"{threshold_mm:9.1f} | "
                f"{name:12s} | "
                f"{np.count_nonzero(active):12d} | "
                f"{np.mean(cost):8.5f} | "
                f"{np.max(cost):7.4f} | "
                f"{np.mean(excess) * 1000:19.3f} | "
                f"{np.max(excess) * 1000:18.3f}"
            )

        print(
            "-" * 150
        )


# ============================================================
# WEIGHT SWEEP
# ============================================================

def print_weight_sweep(
    results,
):

    bc = results[
        "BC"
    ]

    bal = results[
        "BALANCED"
    ]

    severe_mask_40 = (
        bal[
            "max_actual_penetration"
        ]
        >=
        0.040
    )

    safe_mask_20 = (
        bal[
            "max_actual_penetration"
        ]
        <=
        0.020
    )

    print()
    print(
        "=" * 180
    )

    print(
        "THRESHOLD / WEIGHT SWEEP ON BALANCED POLICY"
    )

    print(
        "=" * 180
    )

    print(
        "Tmm | weight | meanR all | meanR >=40 | "
        ">=40 R<0 | >=40 BC-pref | "
        "safe<=20 affected | "
        "mean added penalty all | max added penalty"
    )

    print(
        "-" * 180
    )

    for threshold_mm in (
        THRESHOLDS_MM
    ):

        _, bc_cost = severe_cost(
            bc[
                "max_actual_penetration"
            ],
            threshold_mm,
        )

        _, bal_cost = severe_cost(
            bal[
                "max_actual_penetration"
            ],
            threshold_mm,
        )

        for weight in (
            WEIGHTS
        ):

            adjusted_bc = (
                bc[
                    "reward"
                ]
                -
                float(
                    weight
                )
                *
                bc_cost
            )

            adjusted_bal = (
                bal[
                    "reward"
                ]
                -
                float(
                    weight
                )
                *
                bal_cost
            )

            added_penalty = (
                float(
                    weight
                )
                *
                bal_cost
            )

            severe_negative = int(
                np.count_nonzero(
                    adjusted_bal[
                        severe_mask_40
                    ]
                    <
                    0.0
                )
            )

            severe_bc_preferred = int(
                np.count_nonzero(
                    adjusted_bal[
                        severe_mask_40
                    ]
                    <
                    adjusted_bc[
                        severe_mask_40
                    ]
                )
            )

            safe_affected = int(
                np.count_nonzero(
                    added_penalty[
                        safe_mask_20
                    ]
                    >
                    1e-12
                )
            )

            print(
                f"{threshold_mm:3.0f} | "
                f"{weight:6.2f} | "
                f"{np.mean(adjusted_bal):+9.4f} | "
                f"{np.mean(adjusted_bal[severe_mask_40]):+11.4f} | "
                f"{severe_negative:8d}/"
                f"{np.count_nonzero(severe_mask_40):2d} | "
                f"{severe_bc_preferred:12d}/"
                f"{np.count_nonzero(severe_mask_40):2d} | "
                f"{safe_affected:17d} | "
                f"{np.mean(added_penalty):22.5f} | "
                f"{np.max(added_penalty):17.5f}"
            )

        print(
            "-" * 180
        )


# ============================================================
# REGION EFFECT
# ============================================================

def print_region_effect(
    results,
):

    bal = results[
        "BALANCED"
    ]

    print()
    print(
        "=" * 150
    )

    print(
        "BALANCED POLICY: CANDIDATE COST BY PROBLEM REGION"
    )

    print(
        "=" * 150
    )

    print(
        "region                  | Tmm | "
        "active | meanPen | maxPen | "
        "meanCost | maxCost"
    )

    print(
        "-" * 150
    )

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        region_mask = (
            (
                bal[
                    "step"
                ]
                >=
                start
            )
            &
            (
                bal[
                    "step"
                ]
                <=
                end
            )
        )

        for threshold_mm in (
            THRESHOLDS_MM
        ):

            excess, cost = severe_cost(
                bal[
                    "max_actual_penetration"
                ][
                    region_mask
                ],
                threshold_mm,
            )

            pen = (
                bal[
                    "max_actual_penetration"
                ][
                    region_mask
                ]
            )

            print(
                f"{region_name:23s} | "
                f"{threshold_mm:3.0f} | "
                f"{np.count_nonzero(excess > 0.0):6d} | "
                f"{np.mean(pen) * 1000:7.2f} | "
                f"{np.max(pen) * 1000:6.2f} | "
                f"{np.mean(cost):8.5f} | "
                f"{np.max(cost):7.4f}"
            )

        print(
            "-" * 150
        )


# ============================================================
# SELECTED FRAME COST
# ============================================================

def print_selected_frames(
    results,
):

    bal = results[
        "BALANCED"
    ]

    print()
    print(
        "=" * 180
    )

    print(
        "SELECTED BALANCED-V2 FRAMES"
    )

    print(
        "=" * 180
    )

    print(
        "step | pen mm | baseR   | "
        "T20 cost | T25 cost | T30 cost | "
        "T35 cost | T40 cost"
    )

    print(
        "-" * 180
    )

    for requested_step in (
        SELECTED_STEPS
    ):

        matches = np.where(
            bal[
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

        values = []

        for threshold_mm in (
            THRESHOLDS_MM
        ):

            _, cost = severe_cost(
                np.asarray(
                    [
                        bal[
                            "max_actual_penetration"
                        ][
                            index
                        ]
                    ],
                    dtype=np.float64,
                ),
                threshold_mm,
            )

            values.append(
                float(
                    cost[
                        0
                    ]
                )
            )

        print(
            f"{requested_step:4d} | "
            f"{bal['max_actual_penetration'][index] * 1000:6.1f} | "
            f"{bal['reward'][index]:+7.3f} | "
            f"{values[0]:8.4f} | "
            f"{values[1]:8.4f} | "
            f"{values[2]:8.4f} | "
            f"{values[3]:8.4f} | "
            f"{values[4]:8.4f}"
        )


# ============================================================
# GOOD-FRAME PROTECTION
# ============================================================

def print_good_frame_protection(
    results,
):

    bal = results[
        "BALANCED"
    ]

    print()
    print(
        "=" * 120
    )

    print(
        "GOOD-FRAME PROTECTION CHECK"
    )

    print(
        "=" * 120
    )

    categories = [
        (
            "<= 10 mm",
            bal[
                "max_actual_penetration"
            ]
            <=
            0.010,
        ),

        (
            "<= 20 mm",
            bal[
                "max_actual_penetration"
            ]
            <=
            0.020,
        ),

        (
            "20-30 mm",
            (
                bal[
                    "max_actual_penetration"
                ]
                >
                0.020
            )
            &
            (
                bal[
                    "max_actual_penetration"
                ]
                <=
                0.030
            ),
        ),

        (
            "30-40 mm",
            (
                bal[
                    "max_actual_penetration"
                ]
                >
                0.030
            )
            &
            (
                bal[
                    "max_actual_penetration"
                ]
                <
                0.040
            ),
        ),

        (
            ">= 40 mm",
            bal[
                "max_actual_penetration"
            ]
            >=
            0.040,
        ),
    ]

    print(
        "category   | frames | mean base Reward"
    )

    print(
        "-" * 70
    )

    for (
        label,
        mask,
    ) in categories:

        count = int(
            np.count_nonzero(
                mask
            )
        )

        mean_reward = (
            float(
                np.mean(
                    bal[
                        "reward"
                    ][
                        mask
                    ]
                )
            )
            if count
            else
            0.0
        )

        print(
            f"{label:10s} | "
            f"{count:6d} | "
            f"{mean_reward:+16.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [
        ORIGINAL_V2_MODEL,
        BALANCED_V2_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required checkpoint not found:\n"
                f"{path}"
            )

    print(
        "=" * 120
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "ABSOLUTE SEVERE-PENETRATION PENALTY OFFLINE SWEEP"
    )

    print(
        "=" * 120
    )

    print(
        "Purpose:"
    )

    print(
        "  Keep Reward-V2 phase-aware penetration handling"
    )

    print(
        "  PLUS test an independent safeguard against"
    )

    print(
        "  visually severe ABSOLUTE raw terrain penetration."
    )

    print()

    print(
        "Candidate cost:"
    )

    print(
        "  max_actual_penetration = "
        "maximum raw penetration among 8 sole spheres"
    )

    print()

    print(
        "  severe_excess = "
        "max(max_actual_penetration - threshold, 0)"
    )

    print()

    print(
        "  severe_cost = "
        "(severe_excess / PENETRATION_SCALE)^2"
    )

    print()

    print(
        "Existing V2 penetration scale:",
        (
            f"{PENETRATION_SCALE * 1000:.1f} mm"
        ),
    )

    print(
        "Threshold candidates [mm]:",
        THRESHOLDS_MM,
    )

    print(
        "Weight candidates:",
        WEIGHTS,
    )

    print()

    print(
        "No PPO training."
    )

    print(
        "No environment modification."
    )

    print(
        "No posture/hip/knee modification."
    )

    print(
        "No action-bound modification."
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

    original_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    balanced_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        print()
        print(
            "Loading original Reward-V2 PPO..."
        )

        original_model = PPO.load(
            ORIGINAL_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced-start Reward-V2 PPO..."
        )

        balanced_model = PPO.load(
            BALANCED_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            original_model,
            original_env,
            "Original V2",
        )

        validate_model_spaces(
            balanced_model,
            balanced_env,
            "Balanced V2",
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # ROLLOUTS
        # ====================================================

        print()

        print(
            "Running BC-only..."
        )

        bc = to_arrays(
            run_rollout(
                bc_env,
                model=None,
            )
        )

        print(
            "Running original Reward-V2 PPO..."
        )

        original = to_arrays(
            run_rollout(
                original_env,
                model=original_model,
            )
        )

        print(
            "Running balanced-start Reward-V2 PPO..."
        )

        balanced = to_arrays(
            run_rollout(
                balanced_env,
                model=balanced_model,
            )
        )


        # ====================================================
        # FAIR COMPARISON
        # ====================================================

        validate_fair_comparison(
            bc,
            original,
            balanced,
        )


        # ====================================================
        # REPORTS
        # ====================================================

        results = {
            "BC":
                bc,

            "V2":
                original,

            "BALANCED":
                balanced,
        }

        print_base_summary(
            results
        )

        print_threshold_summary(
            results
        )

        print_good_frame_protection(
            results
        )

        print_weight_sweep(
            results
        )

        print_region_effect(
            results
        )

        print_selected_frames(
            results
        )


        print()
        print(
            "=" * 120
        )

        print(
            "ABSOLUTE SEVERE-PENETRATION SWEEP COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            "No PPO training was performed."
        )

        print()

        print(
            "Do NOT choose a threshold or weight "
            "until the complete sweep is reviewed."
        )

        print()

        print(
            "Selection goal:"
        )

        print(
            "  1. Leave low-penetration frames untouched."
        )

        print(
            "  2. Strongly punish >=40 mm failures."
        )

        print(
            "  3. Preserve the useful Reward-V2 "
            "phase-aware reference handling."
        )

        print(
            "  4. Keep the change penetration-only."
        )

        print(
            "=" * 120
        )


    finally:

        bc_env.close()
        original_env.close()
        balanced_env.close()


if __name__ == "__main__":
    main()