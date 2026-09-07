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
    FLAT_DISTANCE_TOLERANCE,
    PENETRATION_SCALE,
    PENETRATION_WEIGHT,
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)

SEED = 425
RESIDUAL_SMOOTHING = 0.35
EVAL_STATES = TRAIN_END_STEP - TRAIN_START_STEP + 1
EVAL_TRANSITIONS = TRAIN_END_STEP - TRAIN_START_STEP

LEFT_HEEL = [0, 1]
LEFT_TOE = [2, 3]
RIGHT_HEEL = [4, 5]
RIGHT_TOE = [6, 7]

GROUPS = {
    "left": [0, 1, 2, 3],
    "right": [4, 5, 6, 7],
    "heel": [0, 1, 4, 5],
    "toe": [2, 3, 6, 7],
}

SELECTED_STEPS = [395, 435, 624, 708, 716, 720, 723, 728, 741]


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else 0.0


def safe_median(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if values.size else 0.0


def safe_max(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(values)) if values.size else 0.0


def percent_reduction(baseline, ppo):
    baseline = float(baseline)
    ppo = float(ppo)

    if abs(baseline) < 1e-12:
        return 0.0

    return 100.0 * (baseline - ppo) / baseline


def validate_model_spaces(model, env):
    checks = [
        (
            "observation shape",
            model.observation_space.shape,
            env.observation_space.shape,
        ),
        (
            "action shape",
            model.action_space.shape,
            env.action_space.shape,
        ),
    ]

    for name, saved, current in checks:
        if saved != current:
            raise RuntimeError(
                f"{name} mismatch: saved={saved}, env={current}"
            )

    if not np.allclose(
        model.observation_space.low,
        env.observation_space.low,
    ):
        raise RuntimeError(
            "Observation lower bounds mismatch."
        )

    if not np.allclose(
        model.observation_space.high,
        env.observation_space.high,
    ):
        raise RuntimeError(
            "Observation upper bounds mismatch."
        )

    if not np.allclose(
        model.action_space.low,
        env.action_space.low,
    ):
        raise RuntimeError(
            "Action lower bounds mismatch."
        )

    if not np.allclose(
        model.action_space.high,
        env.action_space.high,
    ):
        raise RuntimeError(
            "Action upper bounds mismatch."
        )


def run_rollout(env, model=None, use_ppo=False):

    obs, reset_info = env.reset(
        seed=SEED,
        options={
            "start_step":
                TRAIN_START_STEP
        },
    )

    if int(reset_info["start_step"]) != TRAIN_START_STEP:
        raise RuntimeError(
            "Incorrect evaluation start step."
        )

    steps = [
        int(env.current_step)
    ]

    frames = [
        float(
            env.current_metadata[
                "motion_frame"
            ]
        )
    ]

    world_x = [
        float(
            env.current_metadata[
                "world_x"
            ]
        )
    ]

    distances = [
        env.current_distances
        .astype(np.float64)
        .copy()
    ]

    flat = [
        env.flat_reference_distances[
            env.current_step
        ]
        .astype(np.float64)
        .copy()
    ]

    residuals = [
        env.previous_residual
        .astype(np.float64)
        .copy()
    ]

    actions = []
    rewards = []
    penetration_costs = []
    clip_costs = []

    for transition_index in range(
        EVAL_TRANSITIONS
    ):

        if use_ppo:

            action, _ = model.predict(
                obs,
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

        if action.shape != (12,):
            raise RuntimeError(
                f"Unexpected action shape: {action.shape}"
            )

        if not env.action_space.contains(
            action
        ):
            raise RuntimeError(
                "Action outside environment action space."
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

        if not np.all(
            np.isfinite(
                obs
            )
        ):
            raise RuntimeError(
                "Observation contains NaN/Inf."
            )

        if not np.isfinite(
            reward
        ):
            raise RuntimeError(
                "Reward contains NaN/Inf."
            )

        actions.append(
            action
            .astype(np.float64)
            .copy()
        )

        rewards.append(
            float(
                reward
            )
        )

        penetration_costs.append(
            float(
                info[
                    "penetration_cost"
                ]
            )
        )

        clip_costs.append(
            float(
                info[
                    "joint_clip_cost"
                ]
            )
        )

        steps.append(
            int(
                info[
                    "step"
                ]
            )
        )

        frames.append(
            float(
                info[
                    "motion_frame"
                ]
            )
        )

        world_x.append(
            float(
                info[
                    "world_x"
                ]
            )
        )

        distances.append(
            np.asarray(
                info[
                    "signed_distances"
                ],
                dtype=np.float64,
            ).copy()
        )

        flat.append(
            np.asarray(
                info[
                    "flat_reference_distances"
                ],
                dtype=np.float64,
            ).copy()
        )

        residuals.append(
            np.asarray(
                info[
                    "residual"
                ],
                dtype=np.float64,
            ).copy()
        )

        if (
            transition_index
            <
            EVAL_TRANSITIONS
            -
            1
            and
            (
                terminated
                or
                truncated
            )
        ):
            raise RuntimeError(
                f"Evaluation ended early at step {info['step']}."
            )

    result = {
        "steps":
            np.asarray(
                steps,
                dtype=np.int32,
            ),

        "frames":
            np.asarray(
                frames,
                dtype=np.float64,
            ),

        "world_x":
            np.asarray(
                world_x,
                dtype=np.float64,
            ),

        "distances":
            np.asarray(
                distances,
                dtype=np.float64,
            ),

        "flat":
            np.asarray(
                flat,
                dtype=np.float64,
            ),

        "residuals":
            np.asarray(
                residuals,
                dtype=np.float64,
            ),

        "actions":
            np.asarray(
                actions,
                dtype=np.float64,
            ),

        "rewards":
            np.asarray(
                rewards,
                dtype=np.float64,
            ),

        "penetration_costs":
            np.asarray(
                penetration_costs,
                dtype=np.float64,
            ),

        "clip_costs":
            np.asarray(
                clip_costs,
                dtype=np.float64,
            ),
    }

    expected_shapes = {
        "steps":
            (
                EVAL_STATES,
            ),

        "distances":
            (
                EVAL_STATES,
                8,
            ),

        "flat":
            (
                EVAL_STATES,
                8,
            ),

        "residuals":
            (
                EVAL_STATES,
                12,
            ),

        "actions":
            (
                EVAL_TRANSITIONS,
                12,
            ),

        "rewards":
            (
                EVAL_TRANSITIONS,
            ),
    }

    for key, expected in (
        expected_shapes.items()
    ):

        if result[
            key
        ].shape != expected:

            raise RuntimeError(
                f"{key} shape mismatch: "
                f"{result[key].shape} != {expected}"
            )

    return result


def compute_geometry(
    distances,
    flat,
):

    penetration = np.maximum(
        -distances,
        0.0,
    )

    penetration_mask = (
        penetration
        >
        0.0
    )

    penetration_values = (
        penetration[
            penetration_mask
        ]
    )

    frame_penetration = np.max(
        penetration,
        axis=1,
    )

    deficit = np.maximum(
        flat
        -
        distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    deficit_mask = (
        deficit
        >
        0.0
    )

    deficit_values = (
        deficit[
            deficit_mask
        ]
    )

    frame_deficit = np.max(
        deficit,
        axis=1,
    )

    allowed_floor = np.minimum(
        flat,
        0.0,
    )

    phase_pen = np.maximum(
        allowed_floor
        -
        distances
        -
        FLAT_DISTANCE_TOLERANCE,
        0.0,
    )

    phase_pen_mask = (
        phase_pen
        >
        0.0
    )

    phase_pen_values = (
        phase_pen[
            phase_pen_mask
        ]
    )

    frame_phase_pen = np.max(
        phase_pen,
        axis=1,
    )

    frame_phase_cost = np.mean(
        (
            phase_pen
            /
            PENETRATION_SCALE
        )
        **
        2,
        axis=1,
    )

    groups = {}

    for (
        name,
        indices,
    ) in GROUPS.items():

        gp = penetration[
            :,
            indices
        ]

        gd = deficit[
            :,
            indices
        ]

        gphase = phase_pen[
            :,
            indices
        ]

        groups[
            name
        ] = {
            "penetrating":
                int(
                    np.sum(
                        gp
                        >
                        0.0
                    )
                ),

            "mean_depth":
                safe_mean(
                    gp[
                        gp
                        >
                        0.0
                    ]
                ),

            "max_depth":
                safe_max(
                    gp[
                        gp
                        >
                        0.0
                    ]
                ),

            "deficit":
                int(
                    np.sum(
                        gd
                        >
                        0.0
                    )
                ),

            "phase_pen":
                int(
                    np.sum(
                        gphase
                        >
                        0.0
                    )
                ),
        }

    return {
        "penetrating_frames":
            int(
                np.sum(
                    np.any(
                        penetration_mask,
                        axis=1,
                    )
                )
            ),

        "penetrating_frame_rate":
            float(
                np.mean(
                    np.any(
                        penetration_mask,
                        axis=1,
                    )
                )
            ),

        "penetrating_samples":
            int(
                np.sum(
                    penetration_mask
                )
            ),

        "penetrating_sample_rate":
            float(
                np.mean(
                    penetration_mask
                )
            ),

        "mean_depth":
            safe_mean(
                penetration_values
            ),

        "median_depth":
            safe_median(
                penetration_values
            ),

        "max_depth":
            safe_max(
                penetration_values
            ),

        "deficit_frames":
            int(
                np.sum(
                    np.any(
                        deficit_mask,
                        axis=1,
                    )
                )
            ),

        "deficit_frame_rate":
            float(
                np.mean(
                    np.any(
                        deficit_mask,
                        axis=1,
                    )
                )
            ),

        "mean_positive_deficit":
            safe_mean(
                deficit_values
            ),

        "mean_all_deficit":
            float(
                np.mean(
                    deficit
                )
            ),

        "max_deficit":
            safe_max(
                deficit_values
            ),

        "squared_deficit":
            float(
                np.sum(
                    deficit
                    **
                    2
                )
            ),

        "phase_pen_frames":
            int(
                np.sum(
                    np.any(
                        phase_pen_mask,
                        axis=1,
                    )
                )
            ),

        "phase_pen_frame_rate":
            float(
                np.mean(
                    np.any(
                        phase_pen_mask,
                        axis=1,
                    )
                )
            ),

        "phase_pen_samples":
            int(
                np.sum(
                    phase_pen_mask
                )
            ),

        "mean_phase_pen":
            safe_mean(
                phase_pen_values
            ),

        "max_phase_pen":
            safe_max(
                phase_pen_values
            ),

        "mean_phase_cost":
            float(
                np.mean(
                    frame_phase_cost
                )
            ),

        "groups":
            groups,

        "_frame_pen":
            frame_penetration,

        "_frame_deficit":
            frame_deficit,

        "_frame_phase_pen":
            frame_phase_pen,
    }


def foot_gap(
    distances,
    heel_indices,
    toe_indices,
):

    return (
        np.mean(
            distances[
                :,
                heel_indices
            ],
            axis=1,
        )
        -
        np.mean(
            distances[
                :,
                toe_indices
            ],
            axis=1,
        )
    )


def compute_foot_shape(
    distances,
    flat,
):

    left_gap = foot_gap(
        distances,
        LEFT_HEEL,
        LEFT_TOE,
    )

    right_gap = foot_gap(
        distances,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    flat_left_gap = foot_gap(
        flat,
        LEFT_HEEL,
        LEFT_TOE,
    )

    flat_right_gap = foot_gap(
        flat,
        RIGHT_HEEL,
        RIGHT_TOE,
    )

    left_extra = (
        left_gap
        -
        flat_left_gap
    )

    right_extra = (
        right_gap
        -
        flat_right_gap
    )

    return {
        "left_mean_gap":
            float(
                np.mean(
                    left_gap
                )
            ),

        "left_median_gap":
            float(
                np.median(
                    left_gap
                )
            ),

        "left_max_gap":
            float(
                np.max(
                    left_gap
                )
            ),

        "left_mean_extra":
            float(
                np.mean(
                    left_extra
                )
            ),

        "left_max_extra":
            float(
                np.max(
                    left_extra
                )
            ),

        "right_mean_gap":
            float(
                np.mean(
                    right_gap
                )
            ),

        "right_median_gap":
            float(
                np.median(
                    right_gap
                )
            ),

        "right_max_gap":
            float(
                np.max(
                    right_gap
                )
            ),

        "right_mean_extra":
            float(
                np.mean(
                    right_extra
                )
            ),

        "right_max_extra":
            float(
                np.max(
                    right_extra
                )
            ),

        "_left_gap":
            left_gap,

        "_flat_left_gap":
            flat_left_gap,

        "_left_extra":
            left_extra,
    }


def print_geometry(
    title,
    m,
):

    print()
    print(
        title
    )

    print(
        "-" * 100
    )

    print(
        "Penetrating frames:",
        f"{m['penetrating_frames']}/{EVAL_STATES}",
        f"({100.0 * m['penetrating_frame_rate']:.2f}%)",
    )

    print(
        "Penetrating samples:",
        f"{m['penetrating_samples']}/{EVAL_STATES * 8}",
        f"({100.0 * m['penetrating_sample_rate']:.2f}%)",
    )

    print(
        "Mean / median / max penetration:",
        f"{m['mean_depth'] * 1000:.3f} /",
        f"{m['median_depth'] * 1000:.3f} /",
        f"{m['max_depth'] * 1000:.3f} mm",
    )

    print(
        "Deficit frames:",
        f"{m['deficit_frames']}/{EVAL_STATES}",
        f"({100.0 * m['deficit_frame_rate']:.2f}%)",
    )

    print(
        "Mean positive / all-sample / max deficit:",
        f"{m['mean_positive_deficit'] * 1000:.3f} /",
        f"{m['mean_all_deficit'] * 1000:.3f} /",
        f"{m['max_deficit'] * 1000:.3f} mm",
    )

    print(
        "Total squared deficit:",
        f"{m['squared_deficit']:.8f} m^2",
    )

    print(
        "V2 phase-aware penetration frames:",
        f"{m['phase_pen_frames']}/{EVAL_STATES}",
        f"({100.0 * m['phase_pen_frame_rate']:.2f}%)",
    )

    print(
        "V2 phase-aware penetration samples:",
        m[
            "phase_pen_samples"
        ],
    )

    print(
        "Mean / max phase-aware excess penetration:",
        f"{m['mean_phase_pen'] * 1000:.3f} /",
        f"{m['max_phase_pen'] * 1000:.3f} mm",
    )

    print(
        "Mean V2 penetration cost:",
        f"{m['mean_phase_cost']:.6f}",
    )

    print()
    print(
        "Groups:"
    )

    for name in [
        "left",
        "right",
        "heel",
        "toe",
    ]:

        g = m[
            "groups"
        ][
            name
        ]

        print(
            f"  {name:5s} | "
            f"penetrating={g['penetrating']:4d} | "
            f"mean={g['mean_depth'] * 1000:7.3f}mm | "
            f"max={g['max_depth'] * 1000:7.3f}mm | "
            f"deficit={g['deficit']:4d} | "
            f"phasePen={g['phase_pen']:4d}"
        )


def print_foot_shape(
    title,
    s,
):

    print()
    print(
        title
    )

    print(
        "-" * 100
    )

    print(
        "LEFT gap mean / median / max:",
        f"{s['left_mean_gap'] * 1000:+.3f} /",
        f"{s['left_median_gap'] * 1000:+.3f} /",
        f"{s['left_max_gap'] * 1000:+.3f} mm",
    )

    print(
        "LEFT extra toe-down vs flat BC mean / max:",
        f"{s['left_mean_extra'] * 1000:+.3f} /",
        f"{s['left_max_extra'] * 1000:+.3f} mm",
    )

    print(
        "RIGHT gap mean / median / max:",
        f"{s['right_mean_gap'] * 1000:+.3f} /",
        f"{s['right_median_gap'] * 1000:+.3f} /",
        f"{s['right_max_gap'] * 1000:+.3f} mm",
    )

    print(
        "RIGHT extra toe-down vs flat BC mean / max:",
        f"{s['right_mean_extra'] * 1000:+.3f} /",
        f"{s['right_max_extra'] * 1000:+.3f} mm",
    )


def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Reward-V2 PPO model not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        "=" * 100
    )

    print(
        "UNITREE G1 REWARD-V2 "
        "DETERMINISTIC BC-vs-PPO EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Range:",
        f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
    )

    print(
        "States:",
        EVAL_STATES,
    )

    print(
        "Transitions:",
        EVAL_TRANSITIONS,
    )

    print(
        "Residual smoothing:",
        RESIDUAL_SMOOTHING,
    )

    print(
        "Flat-reference tolerance:",
        f"{FLAT_DISTANCE_TOLERANCE * 1000:.1f} mm",
    )

    print(
        "V2 penetration scale / weight:",
        (
            f"{PENETRATION_SCALE * 1000:.1f} mm "
            f"/ {PENETRATION_WEIGHT}"
        ),
    )

    print(
        "Physics: NONE -- mj_forward only"
    )

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

        print()
        print(
            "Running BC-only zero-residual trajectory..."
        )

        bc = run_rollout(
            bc_env,
            model=None,
            use_ppo=False,
        )

        print(
            "Running deterministic Reward-V2 PPO trajectory..."
        )

        ppo = run_rollout(
            ppo_env,
            model=model,
            use_ppo=True,
        )

        step_diff = float(
            np.max(
                np.abs(
                    bc[
                        "steps"
                    ]
                    -
                    ppo[
                        "steps"
                    ]
                )
            )
        )

        frame_diff = float(
            np.max(
                np.abs(
                    bc[
                        "frames"
                    ]
                    -
                    ppo[
                        "frames"
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
                    ppo[
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
                    ppo[
                        "flat"
                    ]
                )
            )
        )

        print()
        print(
            "Fair-comparison invariants:"
        )

        print(
            "  max step difference:",
            step_diff,
        )

        print(
            "  max motion-frame difference:",
            frame_diff,
        )

        print(
            "  max world-X difference:",
            x_diff,
        )

        print(
            "  max flat-reference difference:",
            flat_diff,
        )

        if (
            step_diff
            !=
            0.0
            or
            frame_diff
            >
            1e-12
            or
            x_diff
            >
            1e-12
            or
            flat_diff
            >
            1e-12
        ):

            raise RuntimeError(
                "Fair-comparison invariants FAILED."
            )

        bc_g = compute_geometry(
            bc[
                "distances"
            ],
            bc[
                "flat"
            ],
        )

        ppo_g = compute_geometry(
            ppo[
                "distances"
            ],
            ppo[
                "flat"
            ],
        )

        bc_s = compute_foot_shape(
            bc[
                "distances"
            ],
            bc[
                "flat"
            ],
        )

        ppo_s = compute_foot_shape(
            ppo[
                "distances"
            ],
            ppo[
                "flat"
            ],
        )

        print()
        print(
            "=" * 100
        )

        print(
            "GEOMETRIC RESULTS"
        )

        print(
            "=" * 100
        )

        print_geometry(
            "BC-ONLY",
            bc_g,
        )

        print_geometry(
            "DETERMINISTIC REWARD-V2 PPO",
            ppo_g,
        )

        print()
        print(
            "=" * 100
        )

        print(
            "BC -> REWARD-V2 PPO CHANGE"
        )

        print(
            "=" * 100
        )

        print(
            "Penetrating-frame reduction:",
            (
                f"{percent_reduction(bc_g['penetrating_frames'], ppo_g['penetrating_frames']):+.2f}%"
            ),
        )

        print(
            "Penetrating-sample reduction:",
            (
                f"{percent_reduction(bc_g['penetrating_samples'], ppo_g['penetrating_samples']):+.2f}%"
            ),
        )

        print(
            "Mean penetration-depth reduction:",
            (
                f"{percent_reduction(bc_g['mean_depth'], ppo_g['mean_depth']):+.2f}%"
            ),
        )

        print(
            "Maximum penetration-depth reduction:",
            (
                f"{percent_reduction(bc_g['max_depth'], ppo_g['max_depth']):+.2f}%"
            ),
        )

        print(
            "Mean all-sample deficit reduction:",
            (
                f"{percent_reduction(bc_g['mean_all_deficit'], ppo_g['mean_all_deficit']):+.2f}%"
            ),
        )

        print(
            "Maximum deficit reduction:",
            (
                f"{percent_reduction(bc_g['max_deficit'], ppo_g['max_deficit']):+.2f}%"
            ),
        )

        print(
            "Squared-deficit reduction:",
            (
                f"{percent_reduction(bc_g['squared_deficit'], ppo_g['squared_deficit']):+.2f}%"
            ),
        )

        print(
            "V2 penetration-cost reduction:",
            (
                f"{percent_reduction(bc_g['mean_phase_cost'], ppo_g['mean_phase_cost']):+.2f}%"
            ),
        )

        print(
            "Maximum phase-aware penetration reduction:",
            (
                f"{percent_reduction(bc_g['max_phase_pen'], ppo_g['max_phase_pen']):+.2f}%"
            ),
        )

        print()
        print(
            "Reward:"
        )

        print(
            "  BC mean:",
            f"{np.mean(bc['rewards']):+.6f}",
        )

        print(
            "  PPO mean:",
            f"{np.mean(ppo['rewards']):+.6f}",
        )

        print(
            "  change:",
            (
                f"{np.mean(ppo['rewards']) - np.mean(bc['rewards']):+.6f}"
            ),
        )

        transition_residuals = (
            ppo[
                "residuals"
            ][
                1:
            ]
        )

        normalized_residuals = (
            transition_residuals
            /
            ppo_env.residual_scales
            .astype(
                np.float64
            )[
                None,
                :
            ]
        )

        print()
        print(
            "PPO policy behavior:"
        )

        print(
            "  mean |action|:",
            (
                f"{np.mean(np.abs(ppo['actions'])):.6f}"
            ),
        )

        print(
            "  max |action|:",
            (
                f"{np.max(np.abs(ppo['actions'])):.6f}"
            ),
        )

        print(
            "  mean |residual|:",
            (
                f"{np.mean(np.abs(transition_residuals)):.6f} rad"
            ),
        )

        print(
            "  max |residual|:",
            (
                f"{np.max(np.abs(transition_residuals)):.6f} rad"
            ),
        )

        print(
            "  max normalized residual:",
            (
                f"{np.max(np.abs(normalized_residuals)):.6f}"
            ),
        )

        print(
            "  mean |residual delta|:",
            (
                f"{np.mean(np.abs(np.diff(ppo['residuals'], axis=0))):.6f} rad"
            ),
        )

        print(
            "  max joint clip cost:",
            (
                f"{np.max(ppo['clip_costs']):.9f}"
            ),
        )

        print()
        print(
            "=" * 100
        )

        print(
            "FOOT-SHAPE RESULTS"
        )

        print(
            "=" * 100
        )

        print_foot_shape(
            "BC-ONLY",
            bc_s,
        )

        print_foot_shape(
            "DETERMINISTIC REWARD-V2 PPO",
            ppo_s,
        )

        frame_improvement = (
            bc_g[
                "_frame_deficit"
            ]
            -
            ppo_g[
                "_frame_deficit"
            ]
        )

        better = int(
            np.sum(
                frame_improvement
                >
                1e-9
            )
        )

        same = int(
            np.sum(
                np.abs(
                    frame_improvement
                )
                <=
                1e-9
            )
        )

        worse = int(
            np.sum(
                frame_improvement
                <
                -1e-9
            )
        )

        print()
        print(
            "Per-frame max-deficit outcome:"
        )

        print(
            "  PPO better:",
            better,
        )

        print(
            "  same:",
            same,
        )

        print(
            "  PPO worse:",
            worse,
        )

        print()
        print(
            "Selected diagnostic frames:"
        )

        print(
            "step | x       | BC pen | PPO pen | "
            "BC def | PPO def | phasePen | "
            "L gap | flatL | L extra"
        )

        print(
            "-" * 100
        )

        for requested_step in (
            SELECTED_STEPS
        ):

            matches = np.where(
                ppo[
                    "steps"
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

            i = int(
                matches[
                    0
                ]
            )

            print(
                f"{requested_step:4d} | "
                f"{ppo['world_x'][i]:+7.3f} | "
                f"{bc_g['_frame_pen'][i] * 1000:6.1f} | "
                f"{ppo_g['_frame_pen'][i] * 1000:7.1f} | "
                f"{bc_g['_frame_deficit'][i] * 1000:6.1f} | "
                f"{ppo_g['_frame_deficit'][i] * 1000:7.1f} | "
                f"{ppo_g['_frame_phase_pen'][i] * 1000:8.1f} | "
                f"{ppo_s['_left_gap'][i] * 1000:+6.1f} | "
                f"{ppo_s['_flat_left_gap'][i] * 1000:+6.1f} | "
                f"{ppo_s['_left_extra'][i] * 1000:+7.1f}"
            )

        worst_indices = np.argsort(
            ppo_g[
                "_frame_pen"
            ]
        )[
            ::-1
        ][
            :10
        ]

        print()
        print(
            "10 worst remaining Reward-V2 PPO penetration frames:"
        )

        print(
            "step | frame   | x       | "
            "BC pen mm | PPO pen mm | "
            "phase excess mm | max deficit mm"
        )

        print(
            "-" * 92
        )

        for i in (
            worst_indices
        ):

            print(
                f"{ppo['steps'][i]:4d} | "
                f"{ppo['frames'][i]:7.2f} | "
                f"{ppo['world_x'][i]:+7.3f} | "
                f"{bc_g['_frame_pen'][i] * 1000:9.3f} | "
                f"{ppo_g['_frame_pen'][i] * 1000:10.3f} | "
                f"{ppo_g['_frame_phase_pen'][i] * 1000:15.3f} | "
                f"{ppo_g['_frame_deficit'][i] * 1000:14.3f}"
            )

        print()
        print(
            "=" * 100
        )

        print(
            "REWARD-V2 DETERMINISTIC EVALUATION COMPLETE"
        )

        print(
            "=" * 100
        )

        print(
            "No PPO training was performed."
        )

        print(
            "No mujoco.mj_step() was called."
        )

        print(
            "Do not increase training budget "
            "until these results are reviewed."
        )

        print(
            "=" * 100
        )

    finally:

        bc_env.close()
        ppo_env.close()


if __name__ == "__main__":
    main()