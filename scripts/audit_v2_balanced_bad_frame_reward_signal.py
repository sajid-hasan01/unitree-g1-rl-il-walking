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
    DEFICIT_WEIGHT,
    PENETRATION_WEIGHT,
    EXCESS_WEIGHT,
    RESIDUAL_WEIGHT,
    SMOOTHNESS_WEIGHT,
    ACTION_WEIGHT,
    JOINT_CLIP_WEIGHT,
)


# ============================================================
# MODELS
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
# IMPORTANT FRAMES
#
# These come directly from the deterministic comparison.
#
# 486-495:
#   middle-region deterioration
#
# 530-532:
#   new balanced hotspot
#
# 679-700:
#   old V2 severe region that balanced training improved
#
# 708-733:
#   late region, including new 730-733 hotspot
# ============================================================

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

    530,
    531,
    532,

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


REGIONS = [
    (
        "middle_hotspot_486_495",
        486,
        495,
    ),

    (
        "middle_hotspot_530_532",
        530,
        532,
    ),

    (
        "old_late_failure_675_700",
        675,
        700,
    ),

    (
        "late_708_735",
        708,
        735,
    ),
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
        f"{name}: observation-space shape mismatch.",
    )

    require(
        model.action_space.shape
        ==
        env.action_space.shape,
        f"{name}: action-space shape mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.low,
            env.observation_space.low,
        ),
        f"{name}: observation lower bounds mismatch.",
    )

    require(
        np.allclose(
            model.observation_space.high,
            env.observation_space.high,
        ),
        f"{name}: observation upper bounds mismatch.",
    )

    require(
        np.allclose(
            model.action_space.low,
            env.action_space.low,
        ),
        f"{name}: action lower bounds mismatch.",
    )

    require(
        np.allclose(
            model.action_space.high,
            env.action_space.high,
        ),
        f"{name}: action upper bounds mismatch.",
    )


# ============================================================
# EXACT REWARD RECONSTRUCTION
# ============================================================

def reconstruct_reward(
    info,
):

    return float(
        1.0

        -
        DEFICIT_WEIGHT
        *
        float(
            info[
                "deficit_cost"
            ]
        )

        -
        PENETRATION_WEIGHT
        *
        float(
            info[
                "penetration_cost"
            ]
        )

        -
        EXCESS_WEIGHT
        *
        float(
            info[
                "excess_cost"
            ]
        )

        -
        RESIDUAL_WEIGHT
        *
        float(
            info[
                "residual_cost"
            ]
        )

        -
        SMOOTHNESS_WEIGHT
        *
        float(
            info[
                "smoothness_cost"
            ]
        )

        -
        ACTION_WEIGHT
        *
        float(
            info[
                "action_cost"
            ]
        )

        -
        JOINT_CLIP_WEIGHT
        *
        float(
            info[
                "joint_clip_cost"
            ]
        )
    )


def weighted_components(
    info,
):

    return {
        "deficit_penalty":
            DEFICIT_WEIGHT
            *
            float(
                info[
                    "deficit_cost"
                ]
            ),

        "penetration_penalty":
            PENETRATION_WEIGHT
            *
            float(
                info[
                    "penetration_cost"
                ]
            ),

        "excess_penalty":
            EXCESS_WEIGHT
            *
            float(
                info[
                    "excess_cost"
                ]
            ),

        "residual_penalty":
            RESIDUAL_WEIGHT
            *
            float(
                info[
                    "residual_cost"
                ]
            ),

        "smoothness_penalty":
            SMOOTHNESS_WEIGHT
            *
            float(
                info[
                    "smoothness_cost"
                ]
            ),

        "action_penalty":
            ACTION_WEIGHT
            *
            float(
                info[
                    "action_cost"
                ]
            ),

        "clip_penalty":
            JOINT_CLIP_WEIGHT
            *
            float(
                info[
                    "joint_clip_cost"
                ]
            ),
    }


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
        "Unexpected evaluation start.",
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
                "Unexpected kinematic termination "
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

        reconstructed = reconstruct_reward(
            info
        )

        require(
            np.isclose(
                reconstructed,
                float(
                    reward
                ),
                rtol=1e-6,
                atol=1e-8,
            ),
            (
                "Reward reconstruction mismatch "
                f"at step {info['step']}: "
                f"env={reward}, "
                f"reconstructed={reconstructed}"
            ),
        )

        signed = np.asarray(
            info[
                "signed_distances"
            ],
            dtype=np.float64,
        )

        actual_penetration = np.maximum(
            -signed,
            0.0,
        )

        components = weighted_components(
            info
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

                "flat":
                    np.asarray(
                        info[
                            "flat_reference_distances"
                        ],
                        dtype=np.float64,
                    ).copy(),

                "reward":
                    float(
                        reward
                    ),

                "max_penetration":
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

                "deficit_cost":
                    float(
                        info[
                            "deficit_cost"
                        ]
                    ),

                "penetration_cost":
                    float(
                        info[
                            "penetration_cost"
                        ]
                    ),

                "excess_cost":
                    float(
                        info[
                            "excess_cost"
                        ]
                    ),

                "residual_cost":
                    float(
                        info[
                            "residual_cost"
                        ]
                    ),

                "smoothness_cost":
                    float(
                        info[
                            "smoothness_cost"
                        ]
                    ),

                "action_cost":
                    float(
                        info[
                            "action_cost"
                        ]
                    ),

                "joint_clip_cost":
                    float(
                        info[
                            "joint_clip_cost"
                        ]
                    ),

                **components,
            }
        )

    require(
        len(
            records
        )
        ==
        EVAL_TRANSITIONS,
        (
            "Unexpected transition count: "
            f"{len(records)}"
        ),
    )

    return records


# ============================================================
# ARRAYS
# ============================================================

def to_arrays(
    records,
):

    keys = [
        "step",
        "world_x",
        "reward",
        "max_penetration",
        "max_phase_excess",

        "deficit_cost",
        "penetration_cost",
        "excess_cost",
        "residual_cost",
        "smoothness_cost",
        "action_cost",
        "joint_clip_cost",

        "deficit_penalty",
        "penetration_penalty",
        "excess_penalty",
        "residual_penalty",
        "smoothness_penalty",
        "action_penalty",
        "clip_penalty",
    ]

    output = {}

    for key in keys:

        dtype = (
            np.int64
            if key == "step"
            else np.float64
        )

        output[
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

    output[
        "flat"
    ] = np.asarray(
        [
            record[
                "flat"
            ]
            for record
            in records
        ],
        dtype=np.float64,
    )

    return output


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
        "=" * 120
    )

    print(
        "FAIR-COMPARISON INVARIANTS"
    )

    print(
        "=" * 120
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
                        "flat"
                    ]
                    -
                    candidate[
                        "flat"
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
# GLOBAL REWARD COMPONENT SUMMARY
# ============================================================

def print_global_summary(
    results,
):

    print()
    print(
        "=" * 165
    )

    print(
        "GLOBAL REWARD-SIGNAL SUMMARY"
    )

    print(
        "=" * 165
    )

    print(
        "policy       | meanR    | "
        "meanPen | maxPen | phaseEx | "
        "defPen | penPen | excessPen | "
        "resPen | smoothPen | actionPen"
    )

    print(
        "-" * 165
    )

    for name in [
        "BC",
        "V2",
        "BALANCED",
    ]:

        r = results[
            name
        ]

        print(
            f"{name:12s} | "
            f"{np.mean(r['reward']):+8.4f} | "
            f"{np.mean(r['max_penetration']) * 1000:7.2f} | "
            f"{np.max(r['max_penetration']) * 1000:6.2f} | "
            f"{np.max(r['max_phase_excess']) * 1000:7.2f} | "
            f"{np.mean(r['deficit_penalty']):6.3f} | "
            f"{np.mean(r['penetration_penalty']):6.3f} | "
            f"{np.mean(r['excess_penalty']):9.4f} | "
            f"{np.mean(r['residual_penalty']):6.4f} | "
            f"{np.mean(r['smoothness_penalty']):9.5f} | "
            f"{np.mean(r['action_penalty']):9.5f}"
        )


# ============================================================
# REGION SUMMARY
# ============================================================

def print_regions(
    results,
):

    print()
    print(
        "=" * 155
    )

    print(
        "PROBLEM-REGION REWARD SIGNAL"
    )

    print(
        "=" * 155
    )

    print(
        "region                     | policy       | "
        "meanR   | maxPen | phaseEx | "
        "defPen | penPen | resPen"
    )

    print(
        "-" * 155
    )

    for (
        region_name,
        start,
        end,
    ) in REGIONS:

        for name in [
            "BC",
            "V2",
            "BALANCED",
        ]:

            r = results[
                name
            ]

            mask = (
                (
                    r[
                        "step"
                    ]
                    >=
                    start
                )
                &
                (
                    r[
                        "step"
                    ]
                    <=
                    end
                )
            )

            print(
                f"{region_name:26s} | "
                f"{name:12s} | "
                f"{np.mean(r['reward'][mask]):+7.3f} | "
                f"{np.max(r['max_penetration'][mask]) * 1000:6.1f} | "
                f"{np.max(r['max_phase_excess'][mask]) * 1000:7.1f} | "
                f"{np.mean(r['deficit_penalty'][mask]):6.3f} | "
                f"{np.mean(r['penetration_penalty'][mask]):6.3f} | "
                f"{np.mean(r['residual_penalty'][mask]):6.4f}"
            )

        print(
            "-" * 155
        )


# ============================================================
# SELECTED FRAMES
# ============================================================

def print_selected_frames(
    results,
):

    print()
    print(
        "=" * 190
    )

    print(
        "SELECTED-FRAME REWARD DECOMPOSITION"
    )

    print(
        "=" * 190
    )

    print(
        "step | policy       | pen mm | phase mm | "
        "reward  | vsBC    | "
        "defPen | penPen | resPen | smooth | action"
    )

    print(
        "-" * 190
    )

    bc = results[
        "BC"
    ]

    for requested_step in (
        SELECTED_STEPS
    ):

        bc_match = np.where(
            bc[
                "step"
            ]
            ==
            requested_step
        )[
            0
        ]

        if len(
            bc_match
        ) != 1:
            continue

        bc_i = int(
            bc_match[
                0
            ]
        )

        bc_reward = float(
            bc[
                "reward"
            ][
                bc_i
            ]
        )

        for name in [
            "BC",
            "V2",
            "BALANCED",
        ]:

            r = results[
                name
            ]

            matches = np.where(
                r[
                    "step"
                ]
                ==
                requested_step
            )[
                0
            ]

            i = int(
                matches[
                    0
                ]
            )

            print(
                f"{requested_step:4d} | "
                f"{name:12s} | "
                f"{r['max_penetration'][i] * 1000:6.1f} | "
                f"{r['max_phase_excess'][i] * 1000:8.1f} | "
                f"{r['reward'][i]:+7.3f} | "
                f"{r['reward'][i] - bc_reward:+7.3f} | "
                f"{r['deficit_penalty'][i]:6.3f} | "
                f"{r['penetration_penalty'][i]:6.3f} | "
                f"{r['residual_penalty'][i]:6.4f} | "
                f"{r['smoothness_penalty'][i]:6.4f} | "
                f"{r['action_penalty'][i]:6.4f}"
            )

        print(
            "-" * 190
        )


# ============================================================
# WORST BALANCED FRAMES
# ============================================================

def print_worst_balanced(
    results,
    count=25,
):

    bc = results[
        "BC"
    ]

    old = results[
        "V2"
    ]

    bal = results[
        "BALANCED"
    ]

    order = np.argsort(
        bal[
            "max_penetration"
        ]
    )[
        ::-1
    ][
        :count
    ]

    print()
    print(
        "=" * 170
    )

    print(
        "WORST BALANCED-V2 FRAMES: "
        "IS REWARD V2 ALREADY PUNISHING THEM?"
    )

    print(
        "=" * 170
    )

    print(
        "rank | step | BAL pen | BAL phase | "
        "BC R     | V2 R    | BAL R   | "
        "BAL-BC | verdict"
    )

    print(
        "-" * 170
    )

    already_bc_preferred = 0
    balanced_preferred = 0

    for (
        rank,
        i,
    ) in enumerate(
        order,
        start=1,
    ):

        step = int(
            bal[
                "step"
            ][
                i
            ]
        )

        bc_matches = np.where(
            bc[
                "step"
            ]
            ==
            step
        )[
            0
        ]

        old_matches = np.where(
            old[
                "step"
            ]
            ==
            step
        )[
            0
        ]

        bc_i = int(
            bc_matches[
                0
            ]
        )

        old_i = int(
            old_matches[
                0
            ]
        )

        delta = float(
            bal[
                "reward"
            ][
                i
            ]
            -
            bc[
                "reward"
            ][
                bc_i
            ]
        )

        if delta < 0.0:

            verdict = (
                "ALREADY BC-PREFERRED"
            )

            already_bc_preferred += 1

        else:

            verdict = (
                "BALANCED STILL REWARDED"
            )

            balanced_preferred += 1

        print(
            f"{rank:4d} | "
            f"{step:4d} | "
            f"{bal['max_penetration'][i] * 1000:7.2f} | "
            f"{bal['max_phase_excess'][i] * 1000:9.2f} | "
            f"{bc['reward'][bc_i]:+8.3f} | "
            f"{old['reward'][old_i]:+7.3f} | "
            f"{bal['reward'][i]:+7.3f} | "
            f"{delta:+7.3f} | "
            f"{verdict}"
        )

    print()

    print(
        "Top-frame verdict counts:"
    )

    print(
        "  already BC-preferred:",
        already_bc_preferred,
    )

    print(
        "  balanced still rewarded:",
        balanced_preferred,
    )


# ============================================================
# SEVERE BALANCED FRAMES >= 40 MM
# ============================================================

def print_severe_signal(
    results,
):

    bc = results[
        "BC"
    ]

    bal = results[
        "BALANCED"
    ]

    threshold_m = 0.040

    severe_indices = np.where(
        bal[
            "max_penetration"
        ]
        >=
        threshold_m
    )[
        0
    ]

    bc_preferred = 0

    balanced_preferred = 0

    deltas = []

    for i in severe_indices:

        step = int(
            bal[
                "step"
            ][
                i
            ]
        )

        bc_i = int(
            np.where(
                bc[
                    "step"
                ]
                ==
                step
            )[
                0
            ][
                0
            ]
        )

        delta = float(
            bal[
                "reward"
            ][
                i
            ]
            -
            bc[
                "reward"
            ][
                bc_i
            ]
        )

        deltas.append(
            delta
        )

        if delta < 0.0:

            bc_preferred += 1

        else:

            balanced_preferred += 1

    deltas = np.asarray(
        deltas,
        dtype=np.float64,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "SEVERE BALANCED PENETRATION SIGNAL (>= 40 MM)"
    )

    print(
        "=" * 110
    )

    print(
        "Severe frames:",
        len(
            severe_indices
        ),
    )

    print(
        "Reward already BC-prefers:",
        bc_preferred,
    )

    print(
        "Reward still prefers balanced:",
        balanced_preferred,
    )

    if deltas.size:

        print(
            "Mean BAL-BC reward delta:",
            f"{np.mean(deltas):+.6f}",
        )

        print(
            "Minimum BAL-BC reward delta:",
            f"{np.min(deltas):+.6f}",
        )

        print(
            "Maximum BAL-BC reward delta:",
            f"{np.max(deltas):+.6f}",
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for model_path in [
        ORIGINAL_V2_MODEL,
        BALANCED_V2_MODEL,
    ]:

        if not model_path.exists():

            raise FileNotFoundError(
                "Required PPO checkpoint not found:\n"
                f"{model_path}"
            )

    print(
        "=" * 120
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "BALANCED BAD-FRAME REWARD-SIGNAL AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        "Purpose:"
    )

    print(
        "  Determine whether Reward V2 already "
        "punishes the remaining/new penetration "
        "failures strongly enough."
    )

    print()

    print(
        "Original V2:",
        ORIGINAL_V2_MODEL,
    )

    print(
        "Balanced V2:",
        BALANCED_V2_MODEL,
    )

    print()

    print(
        "No PPO training."
    )

    print(
        "No reward modification."
    )

    print(
        "No action modification."
    )

    print(
        "No posture penalty."
    )

    print(
        "No mj_step()."
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
            "Loading original V2 PPO..."
        )

        original_model = PPO.load(
            ORIGINAL_V2_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading balanced V2 PPO..."
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
            "Running original V2..."
        )

        original = to_arrays(
            run_rollout(
                original_env,
                model=original_model,
            )
        )

        print(
            "Running balanced V2..."
        )

        balanced = to_arrays(
            run_rollout(
                balanced_env,
                model=balanced_model,
            )
        )


        # ====================================================
        # VALIDATE
        # ====================================================

        validate_fair_comparison(
            bc,
            original,
            balanced,
        )


        # ====================================================
        # REPORT
        # ====================================================

        results = {
            "BC":
                bc,

            "V2":
                original,

            "BALANCED":
                balanced,
        }

        print_global_summary(
            results
        )

        print_regions(
            results
        )

        print_selected_frames(
            results
        )

        print_worst_balanced(
            results,
            count=25,
        )

        print_severe_signal(
            results
        )


        print()
        print(
            "=" * 120
        )

        print(
            "REWARD-SIGNAL AUDIT COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            "No PPO training was performed."
        )

        print(
            "Decision rule:"
        )

        print(
            "  If severe balanced failures are "
            "already BC-preferred by Reward V2, "
            "do NOT add another penetration term yet."
        )

        print(
            "  In that case, additional learning "
            "under the corrected sampling distribution "
            "is the cleaner next experiment."
        )

        print(
            "  If severe failures are still rewarded "
            "over BC, inspect the reward formulation "
            "before increasing training."
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