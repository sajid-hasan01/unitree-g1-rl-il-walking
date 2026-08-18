import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


# =====================================================================
# TEST SETUP
# =====================================================================

START_FRAME = 0
WARMUP_STEPS = 45

NUM_SEGMENTS = 5
SEGMENT_STEPS = 7
HORIZON = NUM_SEGMENTS * SEGMENT_STEPS

SCREEN_POPULATION = 80
SCREEN_ELITES = 10
SCREEN_ITERATIONS = 3

FULL_POPULATION = 160
FULL_ELITES = 20
FULL_ITERATIONS = 4

INITIAL_STD = 0.40
MIN_STD = 0.06

SEED = 20260814


# =====================================================================
# JOINT GROUPS
#
# 0 left hip pitch
# 1 left hip roll
# 2 left hip yaw
# 3 left knee
# 4 left ankle pitch
# 5 left ankle roll
#
# 6 right hip pitch
# 7 right hip roll
# 8 right hip yaw
# 9 right knee
# 10 right ankle pitch
# 11 right ankle roll
#
# 12 waist yaw
# 13 waist roll
# 14 waist pitch
# =====================================================================

SAGITTAL = [
    0, 3, 4,
    6, 9, 10,
    14,
]

FRONTAL = [
    1, 5,
    7, 11,
    13,
]

YAW = [
    2,
    8,
    12,
]


def multiplier(
    indices=None,
    factor=1.0,
):

    result = np.ones(
        15,
        dtype=np.float64,
    )

    if indices is not None:

        result[
            indices
        ] = factor

    return result


CONFIGS = {
    "BASE_1.0":
        multiplier(),

    "SAGITTAL_1.5":
        multiplier(
            SAGITTAL,
            1.5,
        ),

    "SAGITTAL_2.0":
        multiplier(
            SAGITTAL,
            2.0,
        ),

    "FRONTAL_1.5":
        multiplier(
            FRONTAL,
            1.5,
        ),

    "FRONTAL_2.0":
        multiplier(
            FRONTAL,
            2.0,
        ),

    "BALANCE_1.5":
        multiplier(
            SAGITTAL + FRONTAL,
            1.5,
        ),

    "BALANCE_2.0":
        multiplier(
            SAGITTAL + FRONTAL,
            2.0,
        ),

    "ALL_1.5":
        multiplier(
            list(
                range(15)
            ),
            1.5,
        ),
}


# =====================================================================
# GET ORIGINAL ACTION SCALE
# =====================================================================

probe = G1ClosedLoopTrackingEnvV3(

    fixed_start_frame=
        START_FRAME,

    random_reference_start=
        False,

    reset_position_noise=
        0.0,

    reset_velocity_noise=
        0.0,

    control_hz=
        50.0,

    target_smoothing=
        0.20,

    max_episode_steps=
        400,
)


BASE_SCALE = np.asarray(
    probe.action_scale,
    dtype=np.float64,
).copy()


JOINT_NAMES = [
    item["name"]
    for item
    in probe.joint_info
]


probe.close()


print("=" * 135)
print("G1 V4 GROUPED RESIDUAL AUTHORITY SWEEP")
print("NO PPO TRAINING")
print("=" * 135)

print()
print("ORIGINAL PHYSICAL ACTION SCALES")

for i in range(15):

    print(
        f"{i:02d} "
        f"{JOINT_NAMES[i]:28s} "
        f"{BASE_SCALE[i]:.4f} rad "
        f"{np.degrees(BASE_SCALE[i]):.2f} deg"
    )


# =====================================================================
# ENV
# =====================================================================

def make_env(
    scale_multiplier,
):

    env = G1ClosedLoopTrackingEnvV3(

        fixed_start_frame=
            START_FRAME,

        random_reference_start=
            False,

        reset_position_noise=
            0.0,

        reset_velocity_noise=
            0.0,

        control_hz=
            50.0,

        target_smoothing=
            0.20,

        max_episode_steps=
            400,
    )


    env.action_scale = (
        BASE_SCALE
        * scale_multiplier
    ).astype(
        np.float64
    )


    return env


# =====================================================================
# WARMUP
# =====================================================================

def warmup(
    env,
):

    obs, info = env.reset(
        seed=1234
    )


    zero = np.zeros(
        15,
        dtype=np.float32,
    )


    for _ in range(
        WARMUP_STEPS
    ):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero
        )


        if terminated or truncated:

            raise RuntimeError(
                "Robot failed before "
                "authority branch."
            )


    return obs, info


# =====================================================================
# ACTION EXPANSION
# =====================================================================

def expand(
    segments,
):

    return np.repeat(
        segments,
        SEGMENT_STEPS,
        axis=0,
    ).astype(
        np.float32
    )


# =====================================================================
# BRANCH OBJECTIVE
#
# Compared with the previous authority test:
#
# - orientation penalty is much stronger
# - root error penalty is much stronger
# - lateral drift is explicitly penalized
# - survival still matters
#
# This prevents CEM from receiving a high score merely for
# "falling later in a worse direction."
# =====================================================================

def evaluate_candidate(
    scale_multiplier,
    segments,
):

    env = make_env(
        scale_multiplier
    )


    obs, info = warmup(
        env
    )


    survived = 0

    min_up = 1.0
    final_up = float(
        info["up_z"]
    )

    max_orientation = 0.0
    max_y = 0.0

    root_values = []
    velocity_values = []
    q_values = []


    terminated_flag = False


    for action in expand(
        segments
    ):

        action = np.clip(
            action,
            -1.0,
            1.0,
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


        survived += 1


        up = float(
            info["up_z"]
        )


        orientation = abs(
            float(
                info[
                    "orientation_error_deg"
                ]
            )
        )


        y = abs(
            float(
                info["y"]
            )
        )


        min_up = min(
            min_up,
            up,
        )


        final_up = up


        max_orientation = max(
            max_orientation,
            orientation,
        )


        max_y = max(
            max_y,
            y,
        )


        root_values.append(
            float(
                info[
                    "root_position_error"
                ]
            )
        )


        velocity_values.append(
            float(
                info[
                    "root_velocity_error"
                ]
            )
        )


        q_values.append(
            float(
                info[
                    "q_error_rms"
                ]
            )
        )


        if terminated or truncated:

            terminated_flag = True
            break


    mean_root = float(
        np.mean(
            root_values
        )
    ) if root_values else 999.0


    mean_velocity = float(
        np.mean(
            velocity_values
        )
    ) if velocity_values else 999.0


    mean_q = float(
        np.mean(
            q_values
        )
    ) if q_values else 999.0


    mean_action = float(
        np.mean(
            np.abs(
                segments
            )
        )
    )


    score = (

        10.0
        * survived

        + 100.0
        * min_up

        + 30.0
        * final_up

        - 1.50
        * max_orientation

        - 20.0
        * mean_root

        - 4.0
        * mean_velocity

        - 25.0
        * max_y

        - 0.50
        * mean_q

        - 0.30
        * mean_action
    )


    if terminated_flag:

        score -= 30.0


    metrics = {

        "survived":
            survived,

        "min_up":
            min_up,

        "final_up":
            final_up,

        "orientation":
            max_orientation,

        "max_y":
            max_y,

        "root":
            mean_root,

        "velocity":
            mean_velocity,

        "q":
            mean_q,

        "action":
            mean_action,

        "terminated":
            terminated_flag,
    }


    env.close()


    return (
        float(score),
        metrics,
    )


# =====================================================================
# COMPLETE EPISODE
# =====================================================================

def total_episode(
    scale_multiplier,
    segments,
):

    env = make_env(
        scale_multiplier
    )


    obs, info = warmup(
        env
    )


    terminated = False
    truncated = False


    steps = WARMUP_STEPS


    min_up = float(
        info["up_z"]
    )


    max_orientation = abs(
        float(
            info[
                "orientation_error_deg"
            ]
        )
    )


    max_y = abs(
        float(
            info["y"]
        )
    )


    max_action = 0.0


    for action in expand(
        segments
    ):

        action = np.clip(
            action,
            -1.0,
            1.0,
        )


        max_action = max(
            max_action,

            float(
                np.max(
                    np.abs(
                        action
                    )
                )
            ),
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


        min_up = min(
            min_up,
            float(
                info["up_z"]
            ),
        )


        max_orientation = max(
            max_orientation,

            abs(
                float(
                    info[
                        "orientation_error_deg"
                    ]
                )
            ),
        )


        max_y = max(
            max_y,

            abs(
                float(
                    info["y"]
                )
            ),
        )


        if terminated or truncated:
            break


    zero = np.zeros(
        15,
        dtype=np.float32,
    )


    while not (
        terminated
        or truncated
    ):

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


        min_up = min(
            min_up,
            float(
                info["up_z"]
            ),
        )


        max_orientation = max(
            max_orientation,

            abs(
                float(
                    info[
                        "orientation_error_deg"
                    ]
                )
            ),
        )


        max_y = max(
            max_y,

            abs(
                float(
                    info["y"]
                )
            ),
        )


    result = {

        "steps":
            steps,

        "completed":
            bool(
                info.get(
                    "completed",
                    False,
                )
            ),

        "min_up":
            min_up,

        "orientation":
            max_orientation,

        "max_y":
            max_y,

        "final_ref":
            int(
                info[
                    "reference_index"
                ]
            ),

        "max_action":
            max_action,
    }


    env.close()

    return result


# =====================================================================
# CEM
# =====================================================================

def cem_search(
    scale_multiplier,
    population,
    elites_count,
    iterations,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    mean = np.zeros(
        (
            NUM_SEGMENTS,
            15,
        ),
        dtype=np.float64,
    )


    std = np.full(
        (
            NUM_SEGMENTS,
            15,
        ),
        INITIAL_STD,
        dtype=np.float64,
    )


    zero = np.zeros(
        (
            NUM_SEGMENTS,
            15,
        ),
        dtype=np.float32,
    )


    best_score, best_metrics = (
        evaluate_candidate(
            scale_multiplier,
            zero,
        )
    )


    best_segments = (
        zero.copy()
    )


    for iteration in range(
        iterations
    ):

        candidates = rng.normal(
            loc=mean,
            scale=std,

            size=(
                population,
                NUM_SEGMENTS,
                15,
            ),
        )


        candidates = np.clip(
            candidates,
            -1.0,
            1.0,
        )


        scores = np.zeros(
            population,
            dtype=np.float64,
        )


        for i in range(
            population
        ):

            score, metrics = (
                evaluate_candidate(
                    scale_multiplier,
                    candidates[i],
                )
            )


            scores[i] = score


            if score > best_score:

                best_score = float(
                    score
                )

                best_segments = (
                    candidates[i]
                    .astype(
                        np.float32
                    )
                    .copy()
                )

                best_metrics = (
                    metrics.copy()
                )


        elite_indices = np.argsort(
            scores
        )[
            -elites_count:
        ]


        elite = candidates[
            elite_indices
        ]


        elite_mean = np.mean(
            elite,
            axis=0,
        )


        elite_std = np.std(
            elite,
            axis=0,
        )


        mean = (
            0.25
            * mean

            + 0.75
            * elite_mean
        )


        std = (
            0.25
            * std

            + 0.75
            * elite_std
        )


        std = np.maximum(
            std,
            MIN_STD,
        )


    total = total_episode(
        scale_multiplier,
        best_segments,
    )


    return {
        "score":
            best_score,

        "segments":
            best_segments,

        "branch":
            best_metrics,

        "total":
            total,
    }


# =====================================================================
# ZERO BASELINE
# =====================================================================

zero_segments = np.zeros(
    (
        NUM_SEGMENTS,
        15,
    ),
    dtype=np.float32,
)


baseline_multiplier = (
    CONFIGS[
        "BASE_1.0"
    ]
)


baseline_branch_score, baseline_branch = (
    evaluate_candidate(
        baseline_multiplier,
        zero_segments,
    )
)


baseline_total = total_episode(
    baseline_multiplier,
    zero_segments,
)


print()
print("=" * 135)
print("ZERO ACTION BASELINE")
print("=" * 135)

print(
    "episode steps:",
    baseline_total["steps"],
)

print(
    "min up:",
    f"{baseline_total['min_up']:.3f}",
)

print(
    "max orientation:",
    f"{baseline_total['orientation']:.2f}",
)

print(
    "max Y:",
    f"{baseline_total['max_y']:.3f}",
)


# =====================================================================
# SCREEN ALL SCALE CONFIGURATIONS
# =====================================================================

screen_results = {}


print()
print("=" * 135)
print("GROUP SCALE SCREEN")
print("=" * 135)


for config_index, (
    name,
    scale_multiplier,
) in enumerate(
    CONFIGS.items()
):

    result = cem_search(

        scale_multiplier,

        population=
            SCREEN_POPULATION,

        elites_count=
            SCREEN_ELITES,

        iterations=
            SCREEN_ITERATIONS,

        seed=
            SEED
            + config_index,
    )


    screen_results[
        name
    ] = result


    branch = result[
        "branch"
    ]


    total = result[
        "total"
    ]


    print(
        f"{name:18s} "
        f"steps={total['steps']:3d} "
        f"delta={total['steps']-baseline_total['steps']:+3d} "
        f"minUp={total['min_up']:.3f} "
        f"ori={total['orientation']:6.1f} "
        f"maxY={total['max_y']:.3f} "
        f"branch={branch['survived']:02d}/{HORIZON} "
        f"root={branch['root']:.3f} "
        f"score={result['score']:+.1f}",
        flush=True,
    )


# =====================================================================
# PICK SCREEN WINNER
#
# We prioritize:
#
# 1. episode survival
# 2. uprightness
# 3. lower orientation error
# 4. lower lateral drift
# =====================================================================

best_name = max(

    screen_results,

    key=lambda name: (

        screen_results[
            name
        ]["total"]["steps"],

        screen_results[
            name
        ]["total"]["min_up"],

        -screen_results[
            name
        ]["total"]["orientation"],

        -screen_results[
            name
        ]["total"]["max_y"],
    ),
)


best_multiplier = CONFIGS[
    best_name
]


print()
print("=" * 135)
print("SCREEN WINNER")
print("=" * 135)

print(
    "configuration:",
    best_name,
)


# =====================================================================
# FULL CEM CONFIRMATION OF WINNER
# =====================================================================

full = cem_search(

    best_multiplier,

    population=
        FULL_POPULATION,

    elites_count=
        FULL_ELITES,

    iterations=
        FULL_ITERATIONS,

    seed=
        SEED + 999,
)


branch = full[
    "branch"
]


total = full[
    "total"
]


print()
print("=" * 135)
print("BEST CONFIG FULL CEM")
print("=" * 135)

print(
    "configuration:",
    best_name,
)

print(
    "branch survived:",
    f"{branch['survived']}/{HORIZON}",
)

print(
    "branch min up:",
    f"{branch['min_up']:.3f}",
)

print(
    "branch final up:",
    f"{branch['final_up']:.3f}",
)

print(
    "branch max orientation:",
    f"{branch['orientation']:.2f}",
)

print(
    "branch root error:",
    f"{branch['root']:.3f}",
)

print(
    "branch velocity error:",
    f"{branch['velocity']:.3f}",
)

print(
    "branch max Y:",
    f"{branch['max_y']:.3f}",
)

print()
print(
    "episode steps:",
    total["steps"],
)

print(
    "delta steps:",
    total["steps"]
    - baseline_total["steps"],
)

print(
    "episode min up:",
    f"{total['min_up']:.3f}",
)

print(
    "episode max orientation:",
    f"{total['orientation']:.2f}",
)

print(
    "episode max Y:",
    f"{total['max_y']:.3f}",
)

print(
    "completed:",
    total["completed"],
)


# =====================================================================
# PHYSICAL SCALES OF WINNING CONFIG
# =====================================================================

winning_scale = (
    BASE_SCALE
    * best_multiplier
)


print()
print("=" * 135)
print("WINNING PHYSICAL ACTION SCALES")
print("=" * 135)


for i in range(15):

    print(
        f"{i:02d} "
        f"{JOINT_NAMES[i]:28s} "
        f"old={BASE_SCALE[i]:.4f}rad "
        f"new={winning_scale[i]:.4f}rad "
        f"newDeg={np.degrees(winning_scale[i]):.2f}"
    )


# =====================================================================
# DECISION
# =====================================================================

delta_steps = (
    total["steps"]
    - baseline_total["steps"]
)


orientation_change = (
    total["orientation"]
    - baseline_total["orientation"]
)


up_change = (
    total["min_up"]
    - baseline_total["min_up"]
)


print()
print("=" * 135)
print("SCALE AUTHORITY DECISION")
print("=" * 135)

print(
    "baseline steps:",
    baseline_total["steps"],
)

print(
    "best steps:",
    total["steps"],
)

print(
    "delta steps:",
    delta_steps,
)

print(
    "orientation change:",
    f"{orientation_change:+.2f} deg",
)

print(
    "min-up change:",
    f"{up_change:+.3f}",
)


if (
    delta_steps >= 12
    and orientation_change <= 15.0
    and up_change >= 0.04
):

    print()
    print(
        "AUTHORITY RESULT: STRONG"
    )

    print(
        "Selected scale widening gives "
        "genuine recovery authority."
    )

    print(
        "Next step: permanently apply ONLY "
        "the winning joint-group scale and "
        "redesign PPO reward/credit assignment."
    )


elif (
    delta_steps >= 6
    and orientation_change <= 20.0
):

    print()
    print(
        "AUTHORITY RESULT: IMPROVED BUT MARGINAL"
    )

    print(
        "The winning group helps, but residual "
        "position control remains near its limit."
    )

    print(
        "Next step: one focused controller test "
        "before PPO."
    )


else:

    print()
    print(
        "AUTHORITY RESULT: STILL INSUFFICIENT"
    )

    print(
        "Increasing joint-position residual scale "
        "does not create a real recovery controller."
    )

    print(
        "Do not train PPO again."
    )

    print(
        "Next step: change controller representation "
        "rather than increasing reward or timesteps."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 135)
