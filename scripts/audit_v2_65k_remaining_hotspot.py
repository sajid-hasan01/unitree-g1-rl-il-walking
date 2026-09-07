from __future__ import annotations

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
# EXACT REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    RESIDUAL_SCALES,
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

BALANCED_V2_8K_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_smoke_seed425.zip"
)

BALANCED_V2_65K_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_balanced_long65k_seed425.zip"
)


# ============================================================
# SETTINGS
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
# TARGET REGIONS
# ============================================================

HOTSPOT_START = 680

HOTSPOT_END = 701

CORE_START = 685

CORE_END = 700


# ============================================================
# JOINT NAMES
# ============================================================

JOINT_NAMES = [
    "L_HipPitch",
    "L_HipRoll",
    "L_HipYaw",
    "L_Knee",
    "L_AnklePitch",
    "L_AnkleRoll",

    "R_HipPitch",
    "R_HipRoll",
    "R_HipYaw",
    "R_Knee",
    "R_AnklePitch",
    "R_AnkleRoll",
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
# WEIGHTED REWARD COMPONENTS
# ============================================================

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
# REWARD RECONSTRUCTION
# ============================================================

def reconstruct_reward(
    info,
):

    c = weighted_components(
        info
    )

    return float(
        1.0

        -
        c[
            "deficit_penalty"
        ]

        -
        c[
            "penetration_penalty"
        ]

        -
        c[
            "excess_penalty"
        ]

        -
        c[
            "residual_penalty"
        ]

        -
        c[
            "smoothness_penalty"
        ]

        -
        c[
            "action_penalty"
        ]

        -
        c[
            "clip_penalty"
        ]
    )


# ============================================================
# ROLLOUT
# ============================================================

def run_rollout(
    env,
    model,
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

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        require(
            action.shape
            ==
            (12,),
            (
                "Unexpected action shape: "
                f"{action.shape}"
            ),
        )

        require(
            env.action_space.contains(
                action
            ),
            "Policy action outside action space.",
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


        # ----------------------------------------------------
        # VERIFY EXACT V2 REWARD
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CURRENT GEOMETRY
        # ----------------------------------------------------

        signed = np.asarray(
            env.current_distances,
            dtype=np.float64,
        )

        require(
            signed.shape
            ==
            (8,),
            (
                "Expected 8 sole signed distances, "
                f"got {signed.shape}."
            ),
        )

        penetration = np.maximum(
            -signed,
            0.0,
        )

        max_raw_penetration = float(
            np.max(
                penetration
            )
        )


        # ----------------------------------------------------
        # RESIDUAL / ACTION UTILIZATION
        # ----------------------------------------------------

        residual = np.asarray(
            env.previous_residual,
            dtype=np.float64,
        ).copy()

        normalized_residual = (
            residual
            /
            np.asarray(
                RESIDUAL_SCALES,
                dtype=np.float64,
            )
        )

        abs_action = np.abs(
            action.astype(
                np.float64
            )
        )

        abs_normalized_residual = np.abs(
            normalized_residual
        )

        action_near_bound = (
            abs_action
            >=
            0.95
        )

        residual_near_scale = (
            abs_normalized_residual
            >=
            0.95
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

                "reward":
                    float(
                        reward
                    ),

                "max_pen":
                    max_raw_penetration,

                "phase_excess":
                    float(
                        info[
                            "max_penetration_excess_m"
                        ]
                    ),

                "deficit_penalty":
                    components[
                        "deficit_penalty"
                    ],

                "penetration_penalty":
                    components[
                        "penetration_penalty"
                    ],

                "excess_penalty":
                    components[
                        "excess_penalty"
                    ],

                "residual_penalty":
                    components[
                        "residual_penalty"
                    ],

                "smoothness_penalty":
                    components[
                        "smoothness_penalty"
                    ],

                "action_penalty":
                    components[
                        "action_penalty"
                    ],

                "clip_penalty":
                    components[
                        "clip_penalty"
                    ],

                "action":
                    action
                    .astype(
                        np.float64
                    )
                    .copy(),

                "residual":
                    residual,

                "normalized_residual":
                    normalized_residual,

                "max_abs_action":
                    float(
                        np.max(
                            abs_action
                        )
                    ),

                "action_bound_count":
                    int(
                        np.count_nonzero(
                            action_near_bound
                        )
                    ),

                "max_abs_normalized_residual":
                    float(
                        np.max(
                            abs_normalized_residual
                        )
                    ),

                "residual_bound_count":
                    int(
                        np.count_nonzero(
                            residual_near_scale
                        )
                    ),
            }
        )

    return records


# ============================================================
# INDEX
# ============================================================

def index_by_step(
    records,
):

    return {
        int(
            record[
                "step"
            ]
        ):
            record

        for record
        in records
    }


# ============================================================
# HOTSPOT TABLE
# ============================================================

def print_hotspot(
    eight,
    sixty_five,
):

    eight_by_step = index_by_step(
        eight
    )

    sixty_five_by_step = index_by_step(
        sixty_five
    )

    print()
    print(
        "=" * 190
    )

    print(
        "REMAINING HOTSPOT: BALANCED V2 8K VS 65K"
    )

    print(
        "=" * 190
    )

    print(
        "step | "
        "8Kpen | 65Kpen | "
        "8K R   | 65K R  | dR     | "
        "65K phase | "
        "defPen | penPen | resPen | "
        "max|a| | a>=.95 | "
        "max|r/s| | r>=.95"
    )

    print(
        "-" * 190
    )

    for step in range(
        HOTSPOT_START,
        HOTSPOT_END
        +
        1,
    ):

        r8 = eight_by_step[
            step
        ]

        r65 = sixty_five_by_step[
            step
        ]

        print(
            f"{step:4d} | "
            f"{r8['max_pen'] * 1000:5.1f} | "
            f"{r65['max_pen'] * 1000:6.1f} | "
            f"{r8['reward']:+6.3f} | "
            f"{r65['reward']:+6.3f} | "
            f"{r65['reward'] - r8['reward']:+6.3f} | "
            f"{r65['phase_excess'] * 1000:9.1f} | "
            f"{r65['deficit_penalty']:6.3f} | "
            f"{r65['penetration_penalty']:6.3f} | "
            f"{r65['residual_penalty']:6.4f} | "
            f"{r65['max_abs_action']:6.3f} | "
            f"{r65['action_bound_count']:6d} | "
            f"{r65['max_abs_normalized_residual']:8.3f} | "
            f"{r65['residual_bound_count']:6d}"
        )


# ============================================================
# CORE REGION DECISION
# ============================================================

def print_core_decision(
    eight,
    sixty_five,
):

    eight_by_step = index_by_step(
        eight
    )

    sixty_five_by_step = index_by_step(
        sixty_five
    )

    worse_penetration_steps = []

    reward_prefers_65k = []

    reward_prefers_8k = []

    saturated_steps = []

    for step in range(
        CORE_START,
        CORE_END
        +
        1,
    ):

        r8 = eight_by_step[
            step
        ]

        r65 = sixty_five_by_step[
            step
        ]

        if (
            r65[
                "max_pen"
            ]
            >
            r8[
                "max_pen"
            ]
            +
            0.001
        ):

            worse_penetration_steps.append(
                step
            )

            if (
                r65[
                    "reward"
                ]
                >
                r8[
                    "reward"
                ]
            ):

                reward_prefers_65k.append(
                    step
                )

            else:

                reward_prefers_8k.append(
                    step
                )

        if (
            r65[
                "action_bound_count"
            ]
            >
            0
            or
            r65[
                "residual_bound_count"
            ]
            >
            0
        ):

            saturated_steps.append(
                step
            )


    print()
    print(
        "=" * 125
    )

    print(
        "CORE HOTSPOT DECISION DATA"
    )

    print(
        "=" * 125
    )

    print(
        "Core region:",
        f"{CORE_START} -> {CORE_END}",
    )

    print()

    print(
        "65K has >1 mm more penetration "
        "than 8K at:"
    )

    print(
        worse_penetration_steps
    )

    print()

    print(
        "Among those worse-penetration steps:"
    )

    print(
        "  V2 total reward still prefers 65K:",
        reward_prefers_65k,
    )

    print(
        "  V2 total reward prefers 8K:",
        reward_prefers_8k,
    )

    print()

    print(
        "65K steps with at least one action "
        "or residual >=95% of scale:"
    )

    print(
        saturated_steps
    )

    print()

    print(
        "Interpretation rule:"
    )

    print(
        "  A) Worse penetration + higher 65K reward:"
    )

    print(
        "     current V2 objective is accepting "
        "the penetration because other terms compensate."
    )

    print()

    print(
        "  B) Worse penetration + lower 65K reward:"
    )

    print(
        "     current V2 reward already dislikes "
        "the 65K solution; more optimization may help."
    )

    print()

    print(
        "  C) Persistent action/residual saturation:"
    )

    print(
        "     inspect correction capacity / policy "
        "strategy before simply extending training."
    )

    print(
        "=" * 125
    )


# ============================================================
# JOINT UTILIZATION
# ============================================================

def print_joint_utilization(
    records,
):

    by_step = index_by_step(
        records
    )

    region = [
        by_step[
            step
        ]

        for step
        in range(
            CORE_START,
            CORE_END
            +
            1,
        )
    ]

    actions = np.asarray(
        [
            row[
                "action"
            ]

            for row
            in region
        ],
        dtype=np.float64,
    )

    residuals = np.asarray(
        [
            row[
                "residual"
            ]

            for row
            in region
        ],
        dtype=np.float64,
    )

    normalized = (
        residuals
        /
        np.asarray(
            RESIDUAL_SCALES,
            dtype=np.float64,
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "65K CORE-HOTSPOT JOINT UTILIZATION"
    )

    print(
        "=" * 120
    )

    print(
        "joint         | mean action | max|action| | "
        "mean residual | max|r/scale|"
    )

    print(
        "-" * 120
    )

    for joint_index, joint_name in enumerate(
        JOINT_NAMES
    ):

        print(
            f"{joint_name:13s} | "
            f"{np.mean(actions[:, joint_index]):+11.4f} | "
            f"{np.max(np.abs(actions[:, joint_index])):11.4f} | "
            f"{np.mean(residuals[:, joint_index]):+13.4f} | "
            f"{np.max(np.abs(normalized[:, joint_index])):12.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [
        BALANCED_V2_8K_MODEL,
        BALANCED_V2_65K_MODEL,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required PPO checkpoint not found:\n"
                f"{path}"
            )


    print(
        "=" * 120
    )

    print(
        "UNITREE G1 BALANCED REWARD-V2 65K "
        "REMAINING HOTSPOT AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        "Purpose:"
    )

    print(
        "  Diagnose the remaining 689-700 penetration "
        "cluster before changing reward, action bounds, "
        "or training budget."
    )

    print()

    print(
        "No training."
    )

    print(
        "No reward modification."
    )

    print(
        "No action modification."
    )

    print(
        "No residual-scale modification."
    )

    print(
        "No mj_step()."
    )

    print(
        "=" * 120
    )


    env8 = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    env65 = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        print()
        print(
            "Loading Balanced V2 8K..."
        )

        model8 = PPO.load(
            BALANCED_V2_8K_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading Balanced V2 65K..."
        )

        model65 = PPO.load(
            BALANCED_V2_65K_MODEL,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            model8,
            env8,
            "Balanced V2 8K",
        )

        validate_model_spaces(
            model65,
            env65,
            "Balanced V2 65K",
        )

        print(
            "Model-space validation: PASS"
        )

        print()
        print(
            "Running deterministic 8K rollout..."
        )

        records8 = run_rollout(
            env8,
            model8,
        )

        print(
            "Running deterministic 65K rollout..."
        )

        records65 = run_rollout(
            env65,
            model65,
        )

        require(
            len(
                records8
            )
            ==
            EVAL_TRANSITIONS,
            "Unexpected 8K rollout length.",
        )

        require(
            len(
                records65
            )
            ==
            EVAL_TRANSITIONS,
            "Unexpected 65K rollout length.",
        )

        print(
            "Reward reconstruction: PASS"
        )

        print_hotspot(
            records8,
            records65,
        )

        print_core_decision(
            records8,
            records65,
        )

        print_joint_utilization(
            records65,
        )

        print()
        print(
            "=" * 120
        )

        print(
            "65K REMAINING-HOTSPOT AUDIT COMPLETE"
        )

        print(
            "=" * 120
        )

        print(
            "No checkpoint was modified."
        )

        print(
            "Do not extend training or change V2 "
            "until this output is reviewed."
        )

        print(
            "=" * 120
        )

    finally:

        env8.close()
        env65.close()


if __name__ == "__main__":
    main()