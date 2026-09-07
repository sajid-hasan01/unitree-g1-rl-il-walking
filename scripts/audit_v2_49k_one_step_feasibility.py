from __future__ import annotations

from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
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
# EXACT V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
    PENETRATION_SCALE,
    FLAT_DISTANCE_TOLERANCE,
    signed_geom_distance,
)


# ============================================================
# CURRENT PENETRATION-FIRST CHAMPION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "v2_balanced_checkpointed65k_seed425_checkpoints"
    / "v2_balanced_checkpointed65k_seed425_049152_steps.zip"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 425

RESIDUAL_SMOOTHING = 0.35

TARGET_STEPS = [
    527,
    717,
    719,
]

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
# DIFFERENTIAL EVOLUTION SETTINGS
#
# Diagnostic search only.
# No PPO training.
# ============================================================

DE_MAXITER = 60

DE_POPSIZE = 8

DE_TOL = 1e-7


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


def geom_name(
    model,
    geom_id,
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        int(
            geom_id
        ),
    )

    if name is None:
        return f"geom_{geom_id}"

    return str(
        name
    )


# ============================================================
# WORST FOOT / TERRAIN CONTACT
# ============================================================

def get_worst_contact(
    env,
):

    best = None

    for (
        sphere_index,
        sphere,
    ) in enumerate(
        env.foot_spheres
    ):

        for terrain_geom_id in (
            env.terrain_geom_ids
        ):

            distance = signed_geom_distance(
                env.model,
                env.data,
                sphere[
                    "geom_id"
                ],
                terrain_geom_id,
            )

            item = {
                "sphere_index":
                    sphere_index,

                "sphere_geom_id":
                    int(
                        sphere[
                            "geom_id"
                        ]
                    ),

                "side":
                    str(
                        sphere[
                            "side"
                        ]
                    ),

                "region":
                    str(
                        sphere[
                            "region"
                        ]
                    ),

                "terrain_geom_id":
                    int(
                        terrain_geom_id
                    ),

                "terrain_name":
                    geom_name(
                        env.model,
                        terrain_geom_id,
                    ),

                "signed_distance":
                    float(
                        distance
                    ),
            }

            if (
                best is None
                or
                item[
                    "signed_distance"
                ]
                <
                best[
                    "signed_distance"
                ]
            ):

                best = item

    require(
        best is not None,
        "No foot/terrain distance found.",
    )

    return best


# ============================================================
# GEOMETRY
# ============================================================

def geometry_metrics(
    env,
    step,
):

    current = np.asarray(
        env.current_distances,
        dtype=np.float64,
    )

    flat = np.asarray(
        env.flat_reference_distances[
            step
        ],
        dtype=np.float64,
    )

    penetration = np.maximum(
        -current,
        0.0,
    )

    max_penetration = float(
        np.max(
            penetration
        )
    )

    deficit = np.maximum(
        flat
        -
        current
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    allowed_floor = np.minimum(
        flat,
        0.0,
    )

    phase_excess = np.maximum(
        allowed_floor
        -
        current
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    phase_cost = float(
        np.mean(
            (
                phase_excess
                /
                PENETRATION_SCALE
            )
            **
            2
        )
    )

    return {
        "max_penetration":
            max_penetration,

        "mean_deficit":
            float(
                np.mean(
                    deficit
                )
            ),

        "max_deficit":
            float(
                np.max(
                    deficit
                )
            ),

        "phase_cost":
            phase_cost,

        "max_phase_excess":
            float(
                np.max(
                    phase_excess
                )
            ),
    }


# ============================================================
# SNAPSHOT POLICY PREDECESSOR STATES
#
# To evaluate target t, we preserve the exact residual state
# at t-1 and the policy action that originally generated t.
# ============================================================

def collect_target_snapshots(
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
        "Wrong rollout start.",
    )

    snapshots = {}

    while (
        env.current_step
        <
        TRAIN_END_STEP
    ):

        source_step = int(
            env.current_step
        )

        target_step = (
            source_step
            +
            1
        )

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
            f"Unexpected action shape: {action.shape}",
        )

        previous_residual = (
            env.previous_residual
            .astype(
                np.float64
            )
            .copy()
        )

        if target_step in TARGET_STEPS:

            snapshots[
                target_step
            ] = {
                "source_step":
                    source_step,

                "target_step":
                    target_step,

                "previous_residual":
                    previous_residual,

                "policy_action":
                    action
                    .astype(
                        np.float64
                    )
                    .copy(),
            }

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

        if target_step in TARGET_STEPS:

            geometry = geometry_metrics(
                env,
                target_step,
            )

            contact = get_worst_contact(
                env
            )

            snapshots[
                target_step
            ][
                "actual_reward"
            ] = float(
                reward
            )

            snapshots[
                target_step
            ][
                "actual_geometry"
            ] = geometry

            snapshots[
                target_step
            ][
                "actual_contact"
            ] = contact

            snapshots[
                target_step
            ][
                "actual_residual"
            ] = (
                env.previous_residual
                .astype(
                    np.float64
                )
                .copy()
            )

    require(
        set(
            snapshots.keys()
        )
        ==
        set(
            TARGET_STEPS
        ),
        (
            "Failed to collect all target snapshots. "
            f"Found {sorted(snapshots.keys())}"
        ),
    )

    return snapshots


# ============================================================
# APPLY ONE CANDIDATE ACTION
#
# Exact same transition rule as env.step():
#
# desired = action * residual_scales
#
# r_t =
#   (1-alpha) r_(t-1)
#   +
#   alpha * desired
# ============================================================

def evaluate_candidate(
    env,
    snapshot,
    candidate_action,
):

    action = np.asarray(
        candidate_action,
        dtype=np.float64,
    )

    action = np.clip(
        action,
        -1.0,
        1.0,
    )

    previous_residual = np.asarray(
        snapshot[
            "previous_residual"
        ],
        dtype=np.float64,
    )

    desired_residual = (
        action
        *
        np.asarray(
            env.residual_scales,
            dtype=np.float64,
        )
    )

    alpha = float(
        env.residual_smoothing
    )

    new_residual = (
        (
            1.0
            -
            alpha
        )
        *
        previous_residual
        +
        alpha
        *
        desired_residual
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Reconstruct exact target state
    # --------------------------------------------------------

    target_step = int(
        snapshot[
            "target_step"
        ]
    )

    env.current_step = (
        target_step
    )

    env.previous_residual = (
        new_residual.copy()
    )

    env._apply_uneven_state(
        target_step,
        new_residual,
    )


    # --------------------------------------------------------
    # Exact immediate V2 reward
    # --------------------------------------------------------

    (
        reward,
        reward_info,
    ) = env._reward(
        action,
        previous_residual,
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = geometry_metrics(
        env,
        target_step,
    )

    contact = get_worst_contact(
        env
    )


    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    normalized_residual = (
        new_residual.astype(
            np.float64
        )
        /
        np.asarray(
            env.residual_scales,
            dtype=np.float64,
        )
    )

    return {
        "action":
            action.copy(),

        "residual":
            new_residual
            .astype(
                np.float64
            )
            .copy(),

        "normalized_residual":
            normalized_residual,

        "reward":
            float(
                reward
            ),

        "reward_info":
            reward_info,

        "geometry":
            geometry,

        "contact":
            contact,

        "max_abs_action":
            float(
                np.max(
                    np.abs(
                        action
                    )
                )
            ),

        "action_sat_count":
            int(
                np.count_nonzero(
                    np.abs(
                        action
                    )
                    >=
                    0.95
                )
            ),

        "max_abs_normalized_residual":
            float(
                np.max(
                    np.abs(
                        normalized_residual
                    )
                )
            ),

        "residual_sat_count":
            int(
                np.count_nonzero(
                    np.abs(
                        normalized_residual
                    )
                    >=
                    0.95
                )
            ),

        "clip_cost":
            float(
                reward_info[
                    "joint_clip_cost"
                ]
            ),
    }


# ============================================================
# VERIFY RECONSTRUCTION
#
# The exact policy action evaluated through our candidate
# function must reproduce the original rollout.
# ============================================================

def validate_policy_reconstruction(
    env,
    snapshot,
):

    result = evaluate_candidate(
        env,
        snapshot,
        snapshot[
            "policy_action"
        ],
    )

    actual_geometry = snapshot[
        "actual_geometry"
    ]

    require(
        abs(
            result[
                "geometry"
            ][
                "max_penetration"
            ]
            -
            actual_geometry[
                "max_penetration"
            ]
        )
        <
        1e-6,
        (
            "Policy max-penetration reconstruction "
            f"failed at step "
            f"{snapshot['target_step']}."
        ),
    )

    require(
        abs(
            result[
                "reward"
            ]
            -
            snapshot[
                "actual_reward"
            ]
        )
        <
        1e-5,
        (
            "Policy reward reconstruction failed at "
            f"step {snapshot['target_step']}. "
            f"original={snapshot['actual_reward']}, "
            f"reconstructed={result['reward']}"
        ),
    )

    return result


# ============================================================
# OPTIMIZATION OBJECTIVES
# ============================================================

def max_penetration_objective(
    action,
    env,
    snapshot,
):

    result = evaluate_candidate(
        env,
        snapshot,
        action,
    )

    max_penetration = (
        result[
            "geometry"
        ][
            "max_penetration"
        ]
    )

    # Strongly discourage any mechanical joint clipping.
    clip_penalty = (
        100.0
        *
        result[
            "clip_cost"
        ]
    )

    return float(
        max_penetration
        +
        clip_penalty
    )


def negative_reward_objective(
    action,
    env,
    snapshot,
):

    result = evaluate_candidate(
        env,
        snapshot,
        action,
    )

    return float(
        -result[
            "reward"
        ]
    )


# ============================================================
# RUN DE
# ============================================================

def optimize_action(
    env,
    snapshot,
    objective,
    objective_name,
):

    bounds = [
        (
            -1.0,
            1.0,
        )
        for _
        in range(
            12
        )
    ]

    start_time = (
        time.perf_counter()
    )

    result = differential_evolution(
        func=
            objective,

        bounds=
            bounds,

        args=(
            env,
            snapshot,
        ),

        strategy=
            "best1bin",

        maxiter=
            DE_MAXITER,

        popsize=
            DE_POPSIZE,

        tol=
            DE_TOL,

        mutation=(
            0.5,
            1.0,
        ),

        recombination=
            0.7,

        seed=
            SEED,

        polish=
            False,

        updating=
            "immediate",

        workers=
            1,

        disp=
            False,
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    final = evaluate_candidate(
        env,
        snapshot,
        result.x,
    )

    return {
        "objective_name":
            objective_name,

        "optimizer_success":
            bool(
                result.success
            ),

        "optimizer_message":
            str(
                result.message
            ),

        "optimizer_fun":
            float(
                result.fun
            ),

        "nfev":
            int(
                result.nfev
            ),

        "nit":
            int(
                result.nit
            ),

        "elapsed":
            float(
                elapsed
            ),

        "candidate":
            final,
    }


# ============================================================
# JOINT NAMES
# ============================================================

def joint_names(
    env,
):

    names = list(
        env.baseline.JOINT_NAMES[
            :12
        ]
    )

    require(
        len(
            names
        )
        ==
        12,
        "Expected 12 leg joint names.",
    )

    return [
        str(
            name
        )
        for name
        in names
    ]


# ============================================================
# ACTION DELTA REPORT
# ============================================================

def print_action_difference(
    env,
    policy_action,
    optimized_action,
):

    names = joint_names(
        env
    )

    policy = np.asarray(
        policy_action,
        dtype=np.float64,
    )

    optimized = np.asarray(
        optimized_action,
        dtype=np.float64,
    )

    delta = (
        optimized
        -
        policy
    )

    order = np.argsort(
        -np.abs(
            delta
        )
    )

    print(
        "  Largest action changes:"
    )

    for index in order[
        :6
    ]:

        print(
            f"    {names[index]:27s} "
            f"policy={policy[index]:+7.3f} "
            f"opt={optimized[index]:+7.3f} "
            f"delta={delta[index]:+7.3f}"
        )


# ============================================================
# RESULT REPORT
# ============================================================

def print_candidate(
    env,
    label,
    result,
    policy_action,
):

    geometry = result[
        "geometry"
    ]

    contact = result[
        "contact"
    ]

    print()
    print(
        f"{label}"
    )

    print(
        f"  max penetration       : "
        f"{geometry['max_penetration'] * 1000:.3f} mm"
    )

    print(
        f"  mean deficit          : "
        f"{geometry['mean_deficit'] * 1000:.3f} mm"
    )

    print(
        f"  max deficit           : "
        f"{geometry['max_deficit'] * 1000:.3f} mm"
    )

    print(
        f"  phase cost            : "
        f"{geometry['phase_cost']:.8f}"
    )

    print(
        f"  max phase excess      : "
        f"{geometry['max_phase_excess'] * 1000:.3f} mm"
    )

    print(
        f"  immediate V2 reward   : "
        f"{result['reward']:+.6f}"
    )

    print(
        f"  worst contact         : "
        f"{contact['side']}-"
        f"{contact['region']}@"
        f"{contact['terrain_name']}"
    )

    print(
        f"  max |action|          : "
        f"{result['max_abs_action']:.4f}"
    )

    print(
        f"  action >=95% count    : "
        f"{result['action_sat_count']}"
    )

    print(
        f"  max |residual/scale|  : "
        f"{result['max_abs_normalized_residual']:.4f}"
    )

    print(
        f"  residual >=95% count  : "
        f"{result['residual_sat_count']}"
    )

    print(
        f"  joint clip cost       : "
        f"{result['clip_cost']:.8f}"
    )

    print_action_difference(
        env,
        policy_action,
        result[
            "action"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "49K checkpoint not found:\n"
            f"{MODEL_PATH}"
        )


    print(
        "=" * 130
    )

    print(
        "UNITREE G1 V2-49K "
        "ONE-STEP ACTION FEASIBILITY AUDIT"
    )

    print(
        "=" * 130
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "  Test whether the remaining severe penetration "
        "frames can be improved from their exact predecessor "
        "states using only legal current V2 actions."
    )

    print()

    print(
        "Targets:",
        TARGET_STEPS,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print()

    print(
        "For every target we perform two independent searches:"
    )

    print(
        "  1. minimize raw maximum penetration"
    )

    print(
        "  2. maximize exact immediate Reward-V2"
    )

    print()

    print(
        "No PPO training."
    )

    print(
        "No reward modification."
    )

    print(
        "No residual-scale modification."
    )

    print(
        "No action-bound modification."
    )

    print(
        "No mj_step()."
    )

    print(
        "=" * 130
    )


    rollout_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=
            False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    optimization_env = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=
            False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )


    try:

        print()
        print(
            "Loading V2-49K model..."
        )

        model = PPO.load(
            MODEL_PATH,
            env=None,
            device="cpu",
            force_reset=True,
        )

        require(
            model.observation_space.shape
            ==
            rollout_env.observation_space.shape,
            "Observation-space mismatch.",
        )

        require(
            model.action_space.shape
            ==
            rollout_env.action_space.shape,
            "Action-space mismatch.",
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # COLLECT EXACT PREDECESSOR STATES
        # ====================================================

        print()
        print(
            "Collecting exact policy predecessor states..."
        )

        snapshots = collect_target_snapshots(
            rollout_env,
            model,
        )

        print(
            "Predecessor-state collection: PASS"
        )


        # ====================================================
        # TARGET-BY-TARGET
        # ====================================================

        for target_step in TARGET_STEPS:

            snapshot = snapshots[
                target_step
            ]

            print()
            print(
                "=" * 130
            )

            print(
                f"TARGET STEP {target_step}"
            )

            print(
                "=" * 130
            )

            print(
                f"Source step: "
                f"{snapshot['source_step']}"
            )


            # ------------------------------------------------
            # VERIFY EXACT POLICY RECONSTRUCTION
            # ------------------------------------------------

            policy_result = (
                validate_policy_reconstruction(
                    optimization_env,
                    snapshot,
                )
            )

            print(
                "Policy transition reconstruction: PASS"
            )

            print_candidate(
                optimization_env,
                "ORIGINAL 49K POLICY ACTION",
                policy_result,
                snapshot[
                    "policy_action"
                ],
            )


            # ------------------------------------------------
            # MAX-PENETRATION OPTIMIZATION
            # ------------------------------------------------

            print()
            print(
                "Running bounded DE: "
                "MINIMUM RAW MAX PENETRATION..."
            )

            penetration_search = optimize_action(
                optimization_env,
                snapshot,
                max_penetration_objective,
                "minimum_max_penetration",
            )

            print(
                f"  evaluations : "
                f"{penetration_search['nfev']}"
            )

            print(
                f"  iterations  : "
                f"{penetration_search['nit']}"
            )

            print(
                f"  elapsed     : "
                f"{penetration_search['elapsed']:.2f} s"
            )

            print_candidate(
                optimization_env,
                "BEST ONE-STEP PENETRATION ACTION",
                penetration_search[
                    "candidate"
                ],
                snapshot[
                    "policy_action"
                ],
            )


            # ------------------------------------------------
            # EXACT REWARD-V2 OPTIMIZATION
            # ------------------------------------------------

            print()
            print(
                "Running bounded DE: "
                "MAXIMUM IMMEDIATE REWARD-V2..."
            )

            reward_search = optimize_action(
                optimization_env,
                snapshot,
                negative_reward_objective,
                "maximum_immediate_reward_v2",
            )

            print(
                f"  evaluations : "
                f"{reward_search['nfev']}"
            )

            print(
                f"  iterations  : "
                f"{reward_search['nit']}"
            )

            print(
                f"  elapsed     : "
                f"{reward_search['elapsed']:.2f} s"
            )

            print_candidate(
                optimization_env,
                "BEST ONE-STEP REWARD-V2 ACTION",
                reward_search[
                    "candidate"
                ],
                snapshot[
                    "policy_action"
                ],
            )


            # ------------------------------------------------
            # SUMMARY FOR THIS TARGET
            # ------------------------------------------------

            original_pen = (
                policy_result[
                    "geometry"
                ][
                    "max_penetration"
                ]
            )

            best_pen = (
                penetration_search[
                    "candidate"
                ][
                    "geometry"
                ][
                    "max_penetration"
                ]
            )

            reward_pen = (
                reward_search[
                    "candidate"
                ][
                    "geometry"
                ][
                    "max_penetration"
                ]
            )

            original_reward = (
                policy_result[
                    "reward"
                ]
            )

            best_reward = (
                reward_search[
                    "candidate"
                ][
                    "reward"
                ]
            )

            print()
            print(
                "-" * 130
            )

            print(
                f"STEP {target_step} DECISION DATA"
            )

            print(
                "-" * 130
            )

            print(
                f"Policy max penetration      : "
                f"{original_pen * 1000:.3f} mm"
            )

            print(
                f"Minimum feasible one-step   : "
                f"{best_pen * 1000:.3f} mm"
            )

            print(
                f"Reward-opt action max pen   : "
                f"{reward_pen * 1000:.3f} mm"
            )

            print(
                f"Policy immediate reward     : "
                f"{original_reward:+.6f}"
            )

            print(
                f"Best immediate V2 reward    : "
                f"{best_reward:+.6f}"
            )

            print(
                f"Raw penetration improvement : "
                f"{(original_pen - best_pen) * 1000:.3f} mm"
            )

            print(
                "-" * 130
            )


        # ====================================================
        # FINAL INTERPRETATION RULE
        # ====================================================

        print()
        print(
            "=" * 130
        )

        print(
            "INTERPRETATION"
        )

        print(
            "=" * 130
        )

        print(
            "If bounded one-step optimization reduces a "
            "30-42 mm failure substantially while keeping "
            "clip cost zero:"
        )

        print(
            "  -> current residual/action bounds are sufficient "
            "for that immediate state."
        )

        print(
            "  -> the PPO policy is using a poor saturated "
            "action combination."
        )

        print()

        print(
            "If even the penetration optimizer cannot "
            "substantially improve a failure:"
        )

        print(
            "  -> a single action from t-1 is insufficient."
        )

        print(
            "  -> next test must optimize a multi-step "
            "anticipation window before changing bounds."
        )

        print()

        print(
            "If reward optimization improves reward but keeps "
            "large penetration:"
        )

        print(
            "  -> inspect the V2 objective trade-off at that "
            "specific state before fine-tuning."
        )

        print(
            "=" * 130
        )

        print()

        print(
            "ONE-STEP FEASIBILITY AUDIT COMPLETE"
        )

        print(
            "No model was modified."
        )


    finally:

        rollout_env.close()

        optimization_env.close()


if __name__ == "__main__":
    main()