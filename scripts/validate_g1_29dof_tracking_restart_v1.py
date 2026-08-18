from pathlib import Path
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


def section(
    title,
):

    print()
    print("=" * 145)
    print(title)
    print("=" * 145)


env = G129DofTrackingRestartV1(
    rsi=True,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


# ================================================================
# BASIC AUDIT
# ================================================================

section(
    "RESTART V1 ENVIRONMENT AUDIT"
)


print(
    "reference:",
    env.reference_path,
)

print(
    "frames:",
    env.num_frames,
)

print(
    "reference FPS:",
    env.reference_fps,
)

print(
    "sim dt:",
    env.sim_dt,
)

print(
    "frame skip:",
    env.frame_skip,
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
    "left sole geoms:",
    env.left_sole_geoms,
)

print(
    "right sole geoms:",
    env.right_sole_geoms,
)


if env.action_space.shape != (
    29,
):

    raise RuntimeError(
        "Action dimension is not 29."
    )


if env.num_observations < 150:

    raise RuntimeError(
        "Observation unexpectedly small."
    )


# ================================================================
# ACTION SCALE AUDIT
# ================================================================

section(
    "ACTUATOR-DERIVED ACTION SCALE"
)


print(
    f"{'#':>2s} "
    f"{'JOINT':28s} "
    f"{'KP':>7s} "
    f"{'KD':>8s} "
    f"{'EFFORT':>8s} "
    f"{'SCALE':>8s} "
    f"{'DEG':>7s}"
)


for j, name in enumerate(
    env.ref_names
):

    print(
        f"{j:2d} "
        f"{name:28s} "
        f"{env.stiffness[j]:7.1f} "
        f"{env.damping[j]:8.3f} "
        f"{env.effort_limits[j]:8.1f} "
        f"{env.action_scale[j]:8.5f} "
        f"{math.degrees(env.action_scale[j]):7.3f}"
    )


expected_scale = (
    0.25
    * env.effort_limits
    / env.stiffness
)


scale_error = float(
    np.max(
        np.abs(
            expected_scale
            - env.action_scale
        )
    )
)


print()
print(
    "max formula error:",
    f"{scale_error:.12f}",
)


if scale_error > 1e-12:

    raise RuntimeError(
        "Action-scale formula mismatch."
    )


# ================================================================
# RESET / OBSERVATION CHECK
# ================================================================

section(
    "OBSERVATION + RESET CHECK"
)


for start in [
    0,
    57,
    114,
]:

    obs, info = env.reset(
        seed=123,
        options={
            "start_frame":
                start,
        },
    )


    finite = bool(
        np.all(
            np.isfinite(
                obs
            )
        )
    )


    q_error = float(
        np.sqrt(
            np.mean(
                (
                    env._joint_q()
                    -
                    env.ref_q[
                        start
                    ]
                ) ** 2
            )
        )
    )


    root_error = float(
        np.linalg.norm(
            env.data.qpos[
                0:3
            ]
            -
            env.ref_root_pos[
                start
            ]
        )
    )


    print(
        f"start={start:03d} "
        f"obsFinite={finite} "
        f"obsMin={np.min(obs):+.3f} "
        f"obsMax={np.max(obs):+.3f} "
        f"qErr={q_error:.9f} "
        f"rootErr={root_error:.9f}"
    )


    if not finite:

        raise RuntimeError(
            "Observation contains NaN/Inf."
        )


    if q_error > 1e-6:

        raise RuntimeError(
            "RSI reset joint mismatch."
        )


    if root_error > 1e-6:

        raise RuntimeError(
            "RSI reset root mismatch."
        )


# ================================================================
# ZERO ACTION ROLLOUT
#
# Must reproduce our previously measured open-loop reference
# tracking behavior approximately.
# ================================================================

def zero_rollout(
    start,
):

    obs, info = env.reset(
        options={
            "start_frame":
                start,
        }
    )


    zero = np.zeros(
        29,
        dtype=np.float32,
    )


    steps = 0

    total_reward = 0.0

    max_abs_action_target_delta = 0.0

    max_q_error = 0.0

    max_root_error = 0.0

    max_orientation = 0.0

    min_up = env._up_z()

    reason = ""


    while True:

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero
        )


        steps += 1

        total_reward += float(
            reward
        )


        expected_target = (
            env.ref_q[
                env._current_frame
            ]
        )


        target_delta = float(
            np.max(
                np.abs(
                    env.last_target
                    - expected_target
                )
            )
        )


        max_abs_action_target_delta = max(
            max_abs_action_target_delta,
            target_delta,
        )


        terms = info[
            "reward_terms"
        ]


        max_q_error = max(
            max_q_error,
            float(
                terms[
                    "q_error"
                ]
            ),
        )


        max_root_error = max(
            max_root_error,
            float(
                terms[
                    "root_position_error"
                ]
            ),
        )


        max_orientation = max(
            max_orientation,
            float(
                terms[
                    "root_orientation_deg"
                ]
            ),
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

            reason = "reference_end"

            break


        if steps > env.num_frames + 5:

            raise RuntimeError(
                "Rollout exceeded reference length."
            )


    return {
        "start":
            start,

        "steps":
            steps,

        "reward":
            total_reward,

        "min_up":
            min_up,

        "max_q":
            max_q_error,

        "max_root":
            max_root_error,

        "max_ori":
            max_orientation,

        "target_delta":
            max_abs_action_target_delta,

        "reason":
            reason,
    }


section(
    "ZERO-ACTION CAUSAL BASELINE"
)


zero_results = []


for start in [
    0,
    57,
    114,
]:

    r = zero_rollout(
        start
    )

    zero_results.append(
        r
    )


    print(
        f"start={r['start']:03d} "
        f"steps={r['steps']:3d} "
        f"minUp={r['min_up']:.3f} "
        f"maxQ={r['max_q']:.3f} "
        f"maxRoot={r['max_root']:.3f} "
        f"maxOri={r['max_ori']:.1f} "
        f"targetDelta={r['target_delta']:.9f} "
        f"reason={r['reason']}"
    )


# Known values from the exact medium_02 native-servo screen were:
#
# start 0   ~102
# start 57  ~51
# start 114 ~58
#
# Allow modest tolerance for environment implementation details.

expected_ranges = {
    0:
        (
            95,
            110,
        ),

    57:
        (
            45,
            60,
        ),

    114:
        (
            50,
            66,
        ),
}


baseline_match = True


for r in zero_results:

    lo, hi = expected_ranges[
        r["start"]
    ]


    if not (
        lo
        <= r["steps"]
        <= hi
    ):

        baseline_match = False


    if r[
        "target_delta"
    ] > 1e-9:

        baseline_match = False


print()
print(
    "known-baseline reproduction:",
    baseline_match,
)


# ================================================================
# ACTION TARGET AUTHORITY
# ================================================================

section(
    "ACTION TARGET AUTHORITY"
)


env.reset(
    options={
        "start_frame":
            40,
    }
)


reference_target = (
    env.ref_q[
        41
    ]
)


plus = env._action_target(
    np.ones(
        29,
        dtype=np.float64,
    ),
    41,
)

minus = env._action_target(
    -np.ones(
        29,
        dtype=np.float64,
    ),
    41,
)


for j, name in enumerate(
    env.ref_names
):

    positive_delta = (
        plus[j]
        - reference_target[j]
    )

    negative_delta = (
        minus[j]
        - reference_target[j]
    )


    print(
        f"{j:02d} "
        f"{name:28s} "
        f"-1={negative_delta:+.5f}rad "
        f"+1={positive_delta:+.5f}rad"
    )


# At least most joints must have nonzero authority.
authority_count = int(
    np.sum(
        (
            np.abs(
                plus
                - reference_target
            )
            > 1e-6
        )
        |
        (
            np.abs(
                minus
                - reference_target
            )
            > 1e-6
        )
    )
)


print()
print(
    "joints with nonzero residual authority:",
    authority_count,
    "/29",
)


if authority_count < 27:

    raise RuntimeError(
        "Unexpected loss of action authority."
    )


# ================================================================
# RANDOM ACTION SANITY
# ================================================================

section(
    "SMALL RANDOM-ACTION SANITY"
)


obs, info = env.reset(
    seed=20260814,
    options={
        "start_frame":
            0,
    },
)


rng = np.random.default_rng(
    20260814
)


random_steps = 0

random_finite = True

random_reason = ""


for _ in range(
    100
):

    action = np.clip(
        0.10
        * rng.standard_normal(
            29
        ),
        -1.0,
        1.0,
    ).astype(
        np.float32
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


    random_steps += 1


    if (
        not np.all(
            np.isfinite(
                obs
            )
        )
        or not np.isfinite(
            reward
        )
    ):

        random_finite = False
        break


    if terminated:

        random_reason = info[
            "termination_reason"
        ]

        break


    if truncated:

        random_reason = "reference_end"
        break


print(
    "steps:",
    random_steps,
)

print(
    "finite:",
    random_finite,
)

print(
    "end reason:",
    random_reason,
)

print(
    "final frame:",
    env._current_frame,
)

print(
    "final up:",
    f"{env._up_z():.3f}",
)


if not random_finite:

    raise RuntimeError(
        "Random action produced NaN/Inf."
    )


# ================================================================
# RSI DISTRIBUTION
# ================================================================

section(
    "RSI RESET DISTRIBUTION"
)


starts = []


for seed in range(
    64
):

    obs, info = env.reset(
        seed=seed
    )

    starts.append(
        int(
            info[
                "start_frame"
            ]
        )
    )


starts = np.asarray(
    starts,
    dtype=np.int64,
)


print(
    "samples:",
    len(
        starts
    ),
)

print(
    "min start:",
    int(
        np.min(
            starts
        )
    ),
)

print(
    "max start:",
    int(
        np.max(
            starts
        )
    ),
)

print(
    "unique starts:",
    len(
        np.unique(
            starts
        )
    ),
)

print(
    "mean start:",
    f"{np.mean(starts):.2f}",
)


if len(
    np.unique(
        starts
    )
) < 20:

    raise RuntimeError(
        "RSI distribution is unexpectedly narrow."
    )


# ================================================================
# FINAL DECISION
# ================================================================

section(
    "RESTART V1 VALIDATION DECISION"
)


print(
    "29 actions:",
    env.action_space.shape
    == (
        29,
    ),
)

print(
    "observation finite:",
    True,
)

print(
    "actuator-derived scale:",
    scale_error
    <= 1e-12,
)

print(
    "zero-action reproduces baseline:",
    baseline_match,
)

print(
    "action authority:",
    f"{authority_count}/29",
)

print(
    "small random actions finite:",
    random_finite,
)

print(
    "RSI functioning:",
    len(
        np.unique(
            starts
        )
    ) >= 20,
)


validation_pass = (
    env.action_space.shape
    == (
        29,
    )

    and scale_error
    <= 1e-12

    and baseline_match

    and authority_count
    >= 27

    and random_finite

    and len(
        np.unique(
            starts
        )
    ) >= 20
)


print()


if validation_pass:

    print(
        "ENVIRONMENT VALIDATION: PASS"
    )

    print(
        "Restart V1 is internally consistent."
    )

    print(
        "NEXT: finite-horizon state-feedback "
        "authority test."
    )

    print(
        "DO NOT START PPO YET."
    )


else:

    print(
        "ENVIRONMENT VALIDATION: FAIL"
    )

    print(
        "Do not train."
    )

    print(
        "Fix the failed validation before "
        "changing rewards or PPO settings."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 145)


env.close()

