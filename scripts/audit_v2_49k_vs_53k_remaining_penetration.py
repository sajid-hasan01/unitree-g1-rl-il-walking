from __future__ import annotations

from pathlib import Path
import sys

import mujoco
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
    FLAT_DISTANCE_TOLERANCE,
    DEFICIT_SCALE,
    PENETRATION_SCALE,
    DEFICIT_WEIGHT,
    PENETRATION_WEIGHT,
    EXCESS_WEIGHT,
    RESIDUAL_WEIGHT,
    SMOOTHNESS_WEIGHT,
    ACTION_WEIGHT,
    JOINT_CLIP_WEIGHT,
    signed_geom_distance,
)


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "v2_balanced_checkpointed65k_seed425_checkpoints"
)

MODEL_49K = (
    CHECKPOINT_DIR
    / "v2_balanced_checkpointed65k_seed425_049152_steps.zip"
)

MODEL_53K = (
    CHECKPOINT_DIR
    / "v2_balanced_checkpointed65k_seed425_053248_steps.zip"
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

SEVERE_THRESHOLD_M = 0.020

IMPORTANT_THRESHOLD_M = 0.030

VERY_SEVERE_THRESHOLD_M = 0.040

SATURATION_THRESHOLD = 0.95


# ============================================================
# EXPECTED CHECKPOINT METRICS
#
# From the checkpointed trainer output.
# Used only to verify that the intended files and evaluation
# conditions are being reproduced.
# ============================================================

EXPECTED = {
    "V2-49K": {
        "penetrating_frames":
            52,

        "frames_ge_20mm":
            13,

        "frames_ge_30mm":
            5,

        "frames_ge_40mm":
            1,

        "frames_ge_50mm":
            0,

        "max_penetration_mm":
            41.709,
    },

    "V2-53K": {
        "penetrating_frames":
            46,

        "frames_ge_20mm":
            13,

        "frames_ge_30mm":
            5,

        "frames_ge_40mm":
            1,

        "frames_ge_50mm":
            0,

        "max_penetration_mm":
            42.689,
    },
}


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
# GEOMETRY NAME
# ============================================================

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

        return (
            f"geom_{int(geom_id)}"
        )

    return str(
        name
    )


# ============================================================
# EXACT WORST FOOT-SPHERE / TERRAIN PAIR
#
# This recomputes every sphere-to-terrain-block distance.
#
# It also verifies that the result agrees with the
# environment's current_distances vector.
# ============================================================

def get_worst_contact(
    env,
):

    current_distances = np.asarray(
        env.current_distances,
        dtype=np.float64,
    )

    require(
        current_distances.shape
        ==
        (8,),
        (
            "Expected 8 current sole distances, "
            f"got {current_distances.shape}."
        ),
    )

    per_sphere = []

    for (
        sphere_index,
        sphere,
    ) in enumerate(
        env.foot_spheres
    ):

        best_distance = np.inf

        best_terrain_geom_id = None

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

            if (
                distance
                <
                best_distance
            ):

                best_distance = float(
                    distance
                )

                best_terrain_geom_id = int(
                    terrain_geom_id
                )

        require(
            best_terrain_geom_id
            is not None,
            "No terrain geom found.",
        )

        require(
            np.isclose(
                best_distance,
                current_distances[
                    sphere_index
                ],
                rtol=1e-5,
                atol=1e-7,
            ),
            (
                "Exact geometry-distance recomputation "
                "does not match environment distance "
                f"for sphere {sphere_index}: "
                f"recomputed={best_distance}, "
                f"environment="
                f"{current_distances[sphere_index]}"
            ),
        )

        per_sphere.append(
            {
                "sphere_index":
                    int(
                        sphere_index
                    ),

                "sphere_geom_id":
                    int(
                        sphere[
                            "geom_id"
                        ]
                    ),

                "sphere_geom_name":
                    geom_name(
                        env.model,
                        sphere[
                            "geom_id"
                        ],
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

                "local_pos":
                    np.asarray(
                        sphere[
                            "local_pos"
                        ],
                        dtype=np.float64,
                    ).copy(),

                "terrain_geom_id":
                    best_terrain_geom_id,

                "terrain_geom_name":
                    geom_name(
                        env.model,
                        best_terrain_geom_id,
                    ),

                "signed_distance":
                    float(
                        best_distance
                    ),
            }
        )

    worst = min(
        per_sphere,
        key=
            lambda item:
                item[
                    "signed_distance"
                ],
    )

    return (
        worst,
        per_sphere,
    )


# ============================================================
# STATE GEOMETRY METRICS
# ============================================================

def calculate_state_geometry(
    current_distances,
    flat_reference,
):

    current = np.asarray(
        current_distances,
        dtype=np.float64,
    )

    flat = np.asarray(
        flat_reference,
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

        "penetration":
            penetration,

        "deficit":
            deficit,

        "phase_excess":
            phase_excess,

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
# WEIGHTED REWARD COMPONENTS
# ============================================================

def weighted_reward_components(
    info,
):

    return {
        "deficit":
            float(
                DEFICIT_WEIGHT
                *
                info[
                    "deficit_cost"
                ]
            ),

        "penetration":
            float(
                PENETRATION_WEIGHT
                *
                info[
                    "penetration_cost"
                ]
            ),

        "excess":
            float(
                EXCESS_WEIGHT
                *
                info[
                    "excess_cost"
                ]
            ),

        "residual":
            float(
                RESIDUAL_WEIGHT
                *
                info[
                    "residual_cost"
                ]
            ),

        "smoothness":
            float(
                SMOOTHNESS_WEIGHT
                *
                info[
                    "smoothness_cost"
                ]
            ),

        "action":
            float(
                ACTION_WEIGHT
                *
                info[
                    "action_cost"
                ]
            ),

        "clip":
            float(
                JOINT_CLIP_WEIGHT
                *
                info[
                    "joint_clip_cost"
                ]
            ),
    }


# ============================================================
# EXACT REWARD RECONSTRUCTION
# ============================================================

def reconstruct_reward(
    info,
):

    c = weighted_reward_components(
        info
    )

    reward = float(
        1.0

        -
        c[
            "deficit"
        ]

        -
        c[
            "penetration"
        ]

        -
        c[
            "excess"
        ]

        -
        c[
            "residual"
        ]

        -
        c[
            "smoothness"
        ]

        -
        c[
            "action"
        ]

        -
        c[
            "clip"
        ]
    )

    return reward


# ============================================================
# JOINT SATURATION NAMES
# ============================================================

def get_saturation(
    env,
    action,
    residual,
):

    joint_names = list(
        env.baseline.JOINT_NAMES[
            :12
        ]
    )

    require(
        len(
            joint_names
        )
        ==
        12,
        "Expected 12 leg joint names.",
    )

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    normalized_residual = (
        residual
        /
        np.asarray(
            env.residual_scales,
            dtype=np.float64,
        )
    )

    action_indices = np.where(
        np.abs(
            action
        )
        >=
        SATURATION_THRESHOLD
    )[
        0
    ]

    residual_indices = np.where(
        np.abs(
            normalized_residual
        )
        >=
        SATURATION_THRESHOLD
    )[
        0
    ]

    action_names = [
        str(
            joint_names[
                index
            ]
        )

        for index
        in action_indices
    ]

    residual_names = [
        str(
            joint_names[
                index
            ]
        )

        for index
        in residual_indices
    ]

    return {
        "max_abs_action":
            float(
                np.max(
                    np.abs(
                        action
                    )
                )
            ),

        "action_count":
            int(
                len(
                    action_indices
                )
            ),

        "action_names":
            action_names,

        "max_abs_residual_normalized":
            float(
                np.max(
                    np.abs(
                        normalized_residual
                    )
                )
            ),

        "residual_count":
            int(
                len(
                    residual_indices
                )
            ),

        "residual_names":
            residual_names,

        "normalized_residual":
            normalized_residual,
    }


# ============================================================
# INITIAL STATE RECORD
#
# Step 150 has no PPO transition reward because it is the
# reset state. It is included for geometry counts only.
# ============================================================

def initial_state_record(
    env,
):

    current = np.asarray(
        env.current_distances,
        dtype=np.float64,
    ).copy()

    flat = np.asarray(
        env.flat_reference_distances[
            env.current_step
        ],
        dtype=np.float64,
    ).copy()

    geometry = calculate_state_geometry(
        current,
        flat,
    )

    worst, _ = get_worst_contact(
        env
    )

    return {
        "step":
            int(
                env.current_step
            ),

        "world_x":
            float(
                env.current_metadata[
                    "world_x"
                ]
            ),

        "distances":
            current,

        "flat":
            flat,

        "max_penetration":
            geometry[
                "max_penetration"
            ],

        "phase_cost":
            geometry[
                "phase_cost"
            ],

        "max_phase_excess":
            geometry[
                "max_phase_excess"
            ],

        "deficit":
            geometry[
                "deficit"
            ],

        "worst_contact":
            worst,

        "reward":
            None,

        "reward_components":
            None,

        "action":
            None,

        "residual":
            np.zeros(
                12,
                dtype=np.float64,
            ),

        "saturation":
            None,

        "joint_clip_cost":
            0.0,
    }


# ============================================================
# DETERMINISTIC ROLLOUT
# ============================================================

def run_rollout(
    env,
    model,
    label,
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
        (
            f"{label}: wrong start step."
        ),
    )

    records = [
        initial_state_record(
            env
        )
    ]

    transition_rewards = []


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
                f"{label}: unexpected action shape "
                f"{action.shape}."
            ),
        )

        require(
            env.action_space.contains(
                action
            ),
            (
                f"{label}: action outside action space."
            ),
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
                f"{label}: unexpected termination "
                f"at step {info['step']}."
            ),
        )

        if (
            transition_index
            <
            EVAL_TRANSITIONS
            -
            1
        ):

            require(
                truncated
                is False,
                (
                    f"{label}: early truncation "
                    f"at step {info['step']}."
                ),
            )

        require(
            np.all(
                np.isfinite(
                    observation
                )
            ),
            (
                f"{label}: non-finite observation."
            ),
        )

        require(
            np.isfinite(
                reward
            ),
            (
                f"{label}: non-finite reward."
            ),
        )


        # ----------------------------------------------------
        # EXACT REWARD CHECK
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
                f"{label}: reward reconstruction "
                f"mismatch at step {info['step']}. "
                f"env={reward}, "
                f"reconstructed={reconstructed}"
            ),
        )


        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        current = np.asarray(
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

        geometry = calculate_state_geometry(
            current,
            flat,
        )

        worst, _ = get_worst_contact(
            env
        )


        # ----------------------------------------------------
        # ACTION / RESIDUAL SATURATION
        # ----------------------------------------------------

        residual = np.asarray(
            info[
                "residual"
            ],
            dtype=np.float64,
        ).copy()

        saturation = get_saturation(
            env,
            action,
            residual,
        )

        components = weighted_reward_components(
            info
        )


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

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

                "distances":
                    current,

                "flat":
                    flat,

                "max_penetration":
                    geometry[
                        "max_penetration"
                    ],

                "phase_cost":
                    geometry[
                        "phase_cost"
                    ],

                "max_phase_excess":
                    geometry[
                        "max_phase_excess"
                    ],

                "deficit":
                    geometry[
                        "deficit"
                    ],

                "worst_contact":
                    worst,

                "reward":
                    float(
                        reward
                    ),

                "reward_components":
                    components,

                "action":
                    action
                    .astype(
                        np.float64
                    )
                    .copy(),

                "residual":
                    residual,

                "saturation":
                    saturation,

                "joint_clip_cost":
                    float(
                        info[
                            "joint_clip_cost"
                        ]
                    ),
            }
        )

        transition_rewards.append(
            float(
                reward
            )
        )


    require(
        len(
            records
        )
        ==
        EVAL_STATES,
        (
            f"{label}: expected "
            f"{EVAL_STATES} states, "
            f"got {len(records)}."
        ),
    )

    require(
        records[
            0
        ][
            "step"
        ]
        ==
        TRAIN_START_STEP,
        (
            f"{label}: wrong first state."
        ),
    )

    require(
        records[
            -1
        ][
            "step"
        ]
        ==
        TRAIN_END_STEP,
        (
            f"{label}: wrong final state."
        ),
    )

    return {
        "label":
            label,

        "records":
            records,

        "mean_reward":
            float(
                np.mean(
                    transition_rewards
                )
            ),
    }


# ============================================================
# GLOBAL METRICS
# ============================================================

def global_metrics(
    rollout,
):

    records = rollout[
        "records"
    ]

    max_pen = np.asarray(
        [
            record[
                "max_penetration"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    all_distances = np.asarray(
        [
            record[
                "distances"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    all_deficit = np.asarray(
        [
            record[
                "deficit"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    phase_cost = np.asarray(
        [
            record[
                "phase_cost"
            ]

            for record
            in records
        ],
        dtype=np.float64,
    )

    penetration = np.maximum(
        -all_distances,
        0.0,
    )

    positive_penetration = penetration[
        penetration
        >
        0.0
    ]

    hotspot = np.asarray(
        [
            record[
                "max_penetration"
            ]

            for record
            in records

            if
            685
            <=
            record[
                "step"
            ]
            <=
            700
        ],
        dtype=np.float64,
    )

    return {
        "penetrating_frames":
            int(
                np.count_nonzero(
                    max_pen
                    >
                    0.0
                )
            ),

        "penetrating_samples":
            int(
                np.count_nonzero(
                    penetration
                    >
                    0.0
                )
            ),

        "mean_positive_penetration":
            float(
                np.mean(
                    positive_penetration
                )
                if
                positive_penetration.size
                else
                0.0
            ),

        "mean_frame_max_penetration":
            float(
                np.mean(
                    max_pen
                )
            ),

        "max_penetration":
            float(
                np.max(
                    max_pen
                )
            ),

        "frames_ge_20mm":
            int(
                np.count_nonzero(
                    max_pen
                    >=
                    0.020
                )
            ),

        "frames_ge_30mm":
            int(
                np.count_nonzero(
                    max_pen
                    >=
                    0.030
                )
            ),

        "frames_ge_40mm":
            int(
                np.count_nonzero(
                    max_pen
                    >=
                    0.040
                )
            ),

        "frames_ge_50mm":
            int(
                np.count_nonzero(
                    max_pen
                    >=
                    0.050
                )
            ),

        "frames_ge_60mm":
            int(
                np.count_nonzero(
                    max_pen
                    >=
                    0.060
                )
            ),

        "mean_deficit":
            float(
                np.mean(
                    all_deficit
                )
            ),

        "max_deficit":
            float(
                np.max(
                    all_deficit
                )
            ),

        "squared_deficit":
            float(
                np.sum(
                    all_deficit
                    **
                    2
                )
            ),

        "mean_phase_cost":
            float(
                np.mean(
                    phase_cost
                )
            ),

        "mean_reward":
            float(
                rollout[
                    "mean_reward"
                ]
            ),

        "hotspot_max":
            float(
                np.max(
                    hotspot
                )
            ),

        "hotspot_penetrating_frames":
            int(
                np.count_nonzero(
                    hotspot
                    >
                    0.0
                )
            ),

        "max_joint_clip_cost":
            float(
                max(
                    record[
                        "joint_clip_cost"
                    ]

                    for record
                    in records
                )
            ),
    }


# ============================================================
# VALIDATE AGAINST CHECKPOINTED TRAINER
# ============================================================

def validate_expected_metrics(
    label,
    metrics,
):

    expected = EXPECTED[
        label
    ]

    integer_fields = [
        "penetrating_frames",
        "frames_ge_20mm",
        "frames_ge_30mm",
        "frames_ge_40mm",
        "frames_ge_50mm",
    ]

    for field in integer_fields:

        require(
            metrics[
                field
            ]
            ==
            expected[
                field
            ],
            (
                f"{label}: expected {field}="
                f"{expected[field]}, got "
                f"{metrics[field]}."
            ),
        )

    actual_max_mm = (
        metrics[
            "max_penetration"
        ]
        *
        1000.0
    )

    require(
        abs(
            actual_max_mm
            -
            expected[
                "max_penetration_mm"
            ]
        )
        <
        0.02,
        (
            f"{label}: max penetration does not "
            "match checkpointed evaluation. "
            f"Expected approximately "
            f"{expected['max_penetration_mm']:.3f} mm, "
            f"got {actual_max_mm:.3f} mm."
        ),
    )


# ============================================================
# INDEX
# ============================================================

def index_records(
    rollout,
):

    return {
        int(
            record[
                "step"
            ]
        ):
            record

        for record
        in rollout[
            "records"
        ]
    }


# ============================================================
# CONTACT LABEL
# ============================================================

def contact_label(
    record,
):

    contact = record[
        "worst_contact"
    ]

    return (
        f"{contact['side']}-"
        f"{contact['region']}@"
        f"{contact['terrain_geom_name']}"
    )


# ============================================================
# GLOBAL TABLE
# ============================================================

def print_global_comparison(
    metrics49,
    metrics53,
):

    print()
    print(
        "=" * 165
    )

    print(
        "GLOBAL DETERMINISTIC COMPARISON"
    )

    print(
        "=" * 165
    )

    print(
        "policy | penFrm | penSamp | meanPen | meanMax | "
        "maxPen | >=20 | >=30 | >=40 | >=50 | "
        "phaseCost | sqDef | reward | hot685-700 | clip"
    )

    print(
        "-" * 165
    )

    for (
        label,
        metrics,
    ) in [
        (
            "V2-49K",
            metrics49,
        ),
        (
            "V2-53K",
            metrics53,
        ),
    ]:

        print(
            f"{label:7s} | "
            f"{metrics['penetrating_frames']:6d} | "
            f"{metrics['penetrating_samples']:7d} | "
            f"{metrics['mean_positive_penetration'] * 1000:7.3f} | "
            f"{metrics['mean_frame_max_penetration'] * 1000:7.3f} | "
            f"{metrics['max_penetration'] * 1000:6.3f} | "
            f"{metrics['frames_ge_20mm']:4d} | "
            f"{metrics['frames_ge_30mm']:4d} | "
            f"{metrics['frames_ge_40mm']:4d} | "
            f"{metrics['frames_ge_50mm']:4d} | "
            f"{metrics['mean_phase_cost']:9.6f} | "
            f"{metrics['squared_deficit']:7.6f} | "
            f"{metrics['mean_reward']:+7.6f} | "
            f"{metrics['hotspot_max'] * 1000:10.3f} | "
            f"{metrics['max_joint_clip_cost']:.6f}"
        )


# ============================================================
# SEVERE FRAME TABLE FOR ONE MODEL
# ============================================================

def print_severe_frames(
    rollout,
):

    print()
    print(
        "=" * 190
    )

    print(
        f"{rollout['label']} — ALL FRAMES >=20 mm"
    )

    print(
        "=" * 190
    )

    print(
        "step | x       | maxPen | contact                         | "
        "reward  | phaseEx | defPen | penPen | resPen | "
        "max|a| | aSat | max|r/s| | rSat"
    )

    print(
        "-" * 190
    )

    severe_records = [
        record

        for record
        in rollout[
            "records"
        ]

        if
        record[
            "max_penetration"
        ]
        >=
        SEVERE_THRESHOLD_M
    ]

    severe_records.sort(
        key=
            lambda record:
                record[
                    "step"
                ]
    )

    for record in severe_records:

        if (
            record[
                "reward"
            ]
            is None
        ):

            reward_text = (
                "  N/A  "
            )

            deficit_penalty = 0.0

            penetration_penalty = 0.0

            residual_penalty = 0.0

            max_action = 0.0

            action_count = 0

            max_residual = 0.0

            residual_count = 0

        else:

            reward_text = (
                f"{record['reward']:+7.3f}"
            )

            deficit_penalty = (
                record[
                    "reward_components"
                ][
                    "deficit"
                ]
            )

            penetration_penalty = (
                record[
                    "reward_components"
                ][
                    "penetration"
                ]
            )

            residual_penalty = (
                record[
                    "reward_components"
                ][
                    "residual"
                ]
            )

            max_action = (
                record[
                    "saturation"
                ][
                    "max_abs_action"
                ]
            )

            action_count = (
                record[
                    "saturation"
                ][
                    "action_count"
                ]
            )

            max_residual = (
                record[
                    "saturation"
                ][
                    "max_abs_residual_normalized"
                ]
            )

            residual_count = (
                record[
                    "saturation"
                ][
                    "residual_count"
                ]
            )

        print(
            f"{record['step']:4d} | "
            f"{record['world_x']:+7.3f} | "
            f"{record['max_penetration'] * 1000:6.2f} | "
            f"{contact_label(record):31s} | "
            f"{reward_text} | "
            f"{record['max_phase_excess'] * 1000:7.2f} | "
            f"{deficit_penalty:6.3f} | "
            f"{penetration_penalty:6.3f} | "
            f"{residual_penalty:6.4f} | "
            f"{max_action:6.3f} | "
            f"{action_count:4d} | "
            f"{max_residual:8.3f} | "
            f"{residual_count:4d}"
        )


# ============================================================
# UNION COMPARISON
#
# Every state at >=20 mm in either policy.
# ============================================================

def print_union_comparison(
    rollout49,
    rollout53,
):

    records49 = index_records(
        rollout49
    )

    records53 = index_records(
        rollout53
    )

    steps = sorted(
        {
            step

            for step in range(
                TRAIN_START_STEP,
                TRAIN_END_STEP
                +
                1
            )

            if (
                records49[
                    step
                ][
                    "max_penetration"
                ]
                >=
                SEVERE_THRESHOLD_M

                or

                records53[
                    step
                ][
                    "max_penetration"
                ]
                >=
                SEVERE_THRESHOLD_M
            )
        }
    )

    print()
    print(
        "=" * 190
    )

    print(
        "UNION OF >=20 mm FRAMES: 49K VS 53K"
    )

    print(
        "=" * 190
    )

    print(
        "step | 49Pen | 49 contact                       | 49R    | "
        "53Pen | 53 contact                       | 53R    | "
        "53-49 pen"
    )

    print(
        "-" * 190
    )

    for step in steps:

        r49 = records49[
            step
        ]

        r53 = records53[
            step
        ]

        reward49 = (
            "N/A"
            if
            r49[
                "reward"
            ]
            is None
            else
            f"{r49['reward']:+.3f}"
        )

        reward53 = (
            "N/A"
            if
            r53[
                "reward"
            ]
            is None
            else
            f"{r53['reward']:+.3f}"
        )

        difference_mm = (
            (
                r53[
                    "max_penetration"
                ]
                -
                r49[
                    "max_penetration"
                ]
            )
            *
            1000.0
        )

        print(
            f"{step:4d} | "
            f"{r49['max_penetration'] * 1000:5.1f} | "
            f"{contact_label(r49):32s} | "
            f"{reward49:6s} | "
            f"{r53['max_penetration'] * 1000:5.1f} | "
            f"{contact_label(r53):32s} | "
            f"{reward53:6s} | "
            f"{difference_mm:+9.2f}"
        )


# ============================================================
# IMPORTANT >=30 mm FRAMES
# ============================================================

def print_important_frames(
    rollout,
):

    records = [
        record

        for record
        in rollout[
            "records"
        ]

        if
        record[
            "max_penetration"
        ]
        >=
        IMPORTANT_THRESHOLD_M
    ]

    records.sort(
        key=
            lambda record:
                record[
                    "max_penetration"
                ],
        reverse=True,
    )

    print()
    print(
        "=" * 145
    )

    print(
        f"{rollout['label']} — >=30 mm FRAMES, WORST FIRST"
    )

    print(
        "=" * 145
    )

    for rank, record in enumerate(
        records,
        start=1,
    ):

        contact = (
            record[
                "worst_contact"
            ]
        )

        print(
            f"{rank:2d}. "
            f"step={record['step']:3d} | "
            f"x={record['world_x']:+.3f} | "
            f"pen={record['max_penetration'] * 1000:7.3f} mm | "
            f"{contact['side']}-{contact['region']} | "
            f"sphereGeom={contact['sphere_geom_id']} | "
            f"terrain={contact['terrain_geom_name']} | "
            f"reward="
            f"{record['reward']:+.6f}"
        )


# ============================================================
# SATURATED JOINTS AT >=30 mm FRAMES
# ============================================================

def print_saturation_details(
    rollout,
):

    records = [
        record

        for record
        in rollout[
            "records"
        ]

        if (
            record[
                "max_penetration"
            ]
            >=
            IMPORTANT_THRESHOLD_M
            and
            record[
                "saturation"
            ]
            is not None
        )
    ]

    print()
    print(
        "=" * 150
    )

    print(
        f"{rollout['label']} — SATURATION AT >=30 mm FRAMES"
    )

    print(
        "=" * 150
    )

    for record in records:

        saturation = (
            record[
                "saturation"
            ]
        )

        action_names = (
            ", ".join(
                saturation[
                    "action_names"
                ]
            )
            if
            saturation[
                "action_names"
            ]
            else
            "none"
        )

        residual_names = (
            ", ".join(
                saturation[
                    "residual_names"
                ]
            )
            if
            saturation[
                "residual_names"
            ]
            else
            "none"
        )

        print()
        print(
            f"step {record['step']} | "
            f"penetration="
            f"{record['max_penetration'] * 1000:.3f} mm"
        )

        print(
            "  action >=95%:"
        )

        print(
            f"    {action_names}"
        )

        print(
            "  residual >=95% scale:"
        )

        print(
            f"    {residual_names}"
        )


# ============================================================
# SAME-STEP REWARD/PENETRATION ALIGNMENT
#
# Diagnostic only:
# total reward contains several objectives.
# ============================================================

def print_reward_alignment(
    rollout49,
    rollout53,
):

    records49 = index_records(
        rollout49
    )

    records53 = index_records(
        rollout53
    )

    lower_pen_higher_reward = []

    lower_pen_lower_reward = []

    for step in range(
        TRAIN_START_STEP
        +
        1,
        TRAIN_END_STEP
        +
        1,
    ):

        r49 = records49[
            step
        ]

        r53 = records53[
            step
        ]

        difference = (
            r53[
                "max_penetration"
            ]
            -
            r49[
                "max_penetration"
            ]
        )

        if (
            abs(
                difference
            )
            <=
            0.001
        ):

            continue

        if difference > 0.0:

            lower = r49

            higher = r53

            lower_label = "49K"

        else:

            lower = r53

            higher = r49

            lower_label = "53K"

        if (
            lower[
                "reward"
            ]
            >
            higher[
                "reward"
            ]
        ):

            lower_pen_higher_reward.append(
                (
                    step,
                    lower_label,
                    lower[
                        "max_penetration"
                    ],
                    higher[
                        "max_penetration"
                    ],
                    lower[
                        "reward"
                    ],
                    higher[
                        "reward"
                    ],
                )
            )

        else:

            lower_pen_lower_reward.append(
                (
                    step,
                    lower_label,
                    lower[
                        "max_penetration"
                    ],
                    higher[
                        "max_penetration"
                    ],
                    lower[
                        "reward"
                    ],
                    higher[
                        "reward"
                    ],
                )
            )


    print()
    print(
        "=" * 125
    )

    print(
        "SAME-STEP TOTAL-REWARD / PENETRATION ALIGNMENT"
    )

    print(
        "=" * 125
    )

    print(
        "Only frames with >1 mm penetration difference "
        "are counted."
    )

    print()

    print(
        "Lower-penetration policy ALSO has higher reward:",
        len(
            lower_pen_higher_reward
        ),
    )

    print(
        "Lower-penetration policy has lower/equal reward:",
        len(
            lower_pen_lower_reward
        ),
    )

    if lower_pen_lower_reward:

        print()

        print(
            "Frames where total reward does NOT simply "
            "follow lower raw max penetration:"
        )

        for (
            step,
            lower_label,
            low_pen,
            high_pen,
            low_reward,
            high_reward,
        ) in lower_pen_lower_reward:

            print(
                f"  step={step:3d} | "
                f"lowerPenPolicy={lower_label} | "
                f"lowPen={low_pen * 1000:6.2f} mm | "
                f"highPen={high_pen * 1000:6.2f} mm | "
                f"lowR={low_reward:+.4f} | "
                f"highR={high_reward:+.4f}"
            )

    print()

    print(
        "Important:"
    )

    print(
        "  This is only a total-reward diagnostic."
    )

    print(
        "  V2 reward also contains deficit, clearance, "
        "residual, smoothness and action costs."
    )


# ============================================================
# SUMMARY
# ============================================================

def print_decision_summary(
    metrics49,
    metrics53,
):

    print()
    print(
        "=" * 130
    )

    print(
        "49K VS 53K DECISION DATA"
    )

    print(
        "=" * 130
    )

    print(
        "49K:"
    )

    print(
        f"  penetrating frames : "
        f"{metrics49['penetrating_frames']}"
    )

    print(
        f"  max penetration    : "
        f"{metrics49['max_penetration'] * 1000:.3f} mm"
    )

    print(
        f"  >=30 mm frames     : "
        f"{metrics49['frames_ge_30mm']}"
    )

    print(
        f"  >=40 mm frames     : "
        f"{metrics49['frames_ge_40mm']}"
    )

    print(
        f"  phase cost         : "
        f"{metrics49['mean_phase_cost']:.6f}"
    )

    print(
        f"  squared deficit    : "
        f"{metrics49['squared_deficit']:.6f}"
    )

    print(
        f"  mean Reward-V2     : "
        f"{metrics49['mean_reward']:+.6f}"
    )

    print()

    print(
        "53K:"
    )

    print(
        f"  penetrating frames : "
        f"{metrics53['penetrating_frames']}"
    )

    print(
        f"  max penetration    : "
        f"{metrics53['max_penetration'] * 1000:.3f} mm"
    )

    print(
        f"  >=30 mm frames     : "
        f"{metrics53['frames_ge_30mm']}"
    )

    print(
        f"  >=40 mm frames     : "
        f"{metrics53['frames_ge_40mm']}"
    )

    print(
        f"  phase cost         : "
        f"{metrics53['mean_phase_cost']:.6f}"
    )

    print(
        f"  squared deficit    : "
        f"{metrics53['squared_deficit']:.6f}"
    )

    print(
        f"  mean Reward-V2     : "
        f"{metrics53['mean_reward']:+.6f}"
    )

    print()

    print(
        "Both checkpoints:"
    )

    print(
        f"  685-700 max penetration: "
        f"{metrics49['hotspot_max'] * 1000:.3f} / "
        f"{metrics53['hotspot_max'] * 1000:.3f} mm"
    )

    print()

    print(
        "Do NOT fine-tune, alter V2, enlarge residual "
        "bounds, or train V3 until the exact remaining "
        "failure locations above are reviewed."
    )

    print(
        "=" * 130
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for path in [
        MODEL_49K,
        MODEL_53K,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Required checkpoint not found:\n"
                f"{path}"
            )


    print(
        "=" * 130
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "BALANCED V2 — 49K VS 53K REMAINING "
        "PENETRATION AUDIT"
    )

    print(
        "=" * 130
    )

    print()

    print(
        "49K checkpoint:"
    )

    print(
        MODEL_49K
    )

    print()

    print(
        "53K checkpoint:"
    )

    print(
        MODEL_53K
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        f"  explicit trajectory: "
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}"
    )

    print(
        f"  smoothing: "
        f"{RESIDUAL_SMOOTHING}"
    )

    print(
        "  deterministic PPO actions"
    )

    print(
        "  same Reward-V2 environment"
    )

    print(
        "  same terrain/root/reference"
    )

    print(
        "  exact MuJoCo sphere-to-terrain distances"
    )

    print(
        "  no training"
    )

    print(
        "  no reward change"
    )

    print(
        "  no residual-bound change"
    )

    print(
        "  no mj_step()"
    )

    print(
        "=" * 130
    )


    env49 = G1KinematicUnevenEnv(
        episode_length=
            EVAL_STATES,

        random_start=
            False,

        residual_smoothing=
            RESIDUAL_SMOOTHING,
    )

    env53 = G1KinematicUnevenEnv(
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
            "Loading 49K..."
        )

        model49 = PPO.load(
            MODEL_49K,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "Loading 53K..."
        )

        model53 = PPO.load(
            MODEL_53K,
            env=None,
            device="cpu",
            force_reset=True,
        )

        validate_model_spaces(
            model49,
            env49,
            "V2-49K",
        )

        validate_model_spaces(
            model53,
            env53,
            "V2-53K",
        )

        print(
            "Model-space validation: PASS"
        )


        # ====================================================
        # ROLLOUTS
        # ====================================================

        print()
        print(
            "Running deterministic 49K rollout..."
        )

        rollout49 = run_rollout(
            env49,
            model49,
            "V2-49K",
        )

        print(
            "Running deterministic 53K rollout..."
        )

        rollout53 = run_rollout(
            env53,
            model53,
            "V2-53K",
        )

        print(
            "Reward reconstruction: PASS"
        )

        print(
            "Exact sphere/terrain distance verification: PASS"
        )


        # ====================================================
        # GLOBAL METRICS
        # ====================================================

        metrics49 = global_metrics(
            rollout49
        )

        metrics53 = global_metrics(
            rollout53
        )

        validate_expected_metrics(
            "V2-49K",
            metrics49,
        )

        validate_expected_metrics(
            "V2-53K",
            metrics53,
        )

        print(
            "Checkpoint metric reproduction: PASS"
        )


        # ====================================================
        # REPORTS
        # ====================================================

        print_global_comparison(
            metrics49,
            metrics53,
        )

        print_severe_frames(
            rollout49
        )

        print_severe_frames(
            rollout53
        )

        print_union_comparison(
            rollout49,
            rollout53,
        )

        print_important_frames(
            rollout49
        )

        print_important_frames(
            rollout53
        )

        print_saturation_details(
            rollout49
        )

        print_saturation_details(
            rollout53
        )

        print_reward_alignment(
            rollout49,
            rollout53,
        )

        print_decision_summary(
            metrics49,
            metrics53,
        )


        print()
        print(
            "=" * 130
        )

        print(
            "49K VS 53K REMAINING-PENETRATION "
            "AUDIT COMPLETE"
        )

        print(
            "=" * 130
        )

        print(
            "No checkpoint was modified."
        )

        print(
            "No PPO training was performed."
        )

        print(
            "=" * 130
        )


    finally:

        env49.close()
        env53.close()


if __name__ == "__main__":
    main()
