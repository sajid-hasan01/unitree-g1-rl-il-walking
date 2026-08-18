import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from envs.g1_closed_loop_tracking_env_v3 import (
    G1ClosedLoopTrackingEnvV3,
)


# =====================================================================
# TEST CONFIG
# =====================================================================

START_FRAME = 0

# Zero-reference controller is still healthy here,
# but collapse begins soon afterward.
WARMUP_STEPS = 45


# Search over:
#
#   5 piecewise-constant action segments
#   x 7 control steps each
#
# = 35-step intervention
#
NUM_SEGMENTS = 5
SEGMENT_STEPS = 7

SEARCH_HORIZON = (
    NUM_SEGMENTS
    * SEGMENT_STEPS
)


POPULATION = 160
ELITES = 20
ITERATIONS = 4

INITIAL_STD = 0.35

MIN_STD = 0.06

SEED = 2026


# =====================================================================
# ENV FACTORY
# =====================================================================

def make_env():

    return G1ClosedLoopTrackingEnvV3(

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


# =====================================================================
# EXPAND SEGMENTS
# =====================================================================

def expand_segments(
    segments,
):

    sequence = []

    for segment in segments:

        for _ in range(
            SEGMENT_STEPS
        ):

            sequence.append(
                segment.copy()
            )

    return np.asarray(
        sequence,
        dtype=np.float32,
    )


# =====================================================================
# ZERO WARMUP
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


    for step in range(
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


        if (
            terminated
            or truncated
        ):

            raise RuntimeError(
                "Robot terminated before "
                "the authority-test branch."
            )


    return (
        obs,
        info,
    )


# =====================================================================
# CANDIDATE EVALUATION
# =====================================================================

def evaluate_candidate(
    segments,
):

    env = make_env()

    obs, info = warmup(
        env
    )


    sequence = expand_segments(
        segments
    )


    survived = 0

    min_up = 1.0

    final_up = float(
        info["up_z"]
    )

    max_orientation = 0.0

    root_errors = []
    velocity_errors = []
    q_errors = []

    terminated_flag = False


    for action in sequence:

        action = np.clip(
            action,
            -1.0,
            +1.0,
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


        root_error = float(
            info[
                "root_position_error"
            ]
        )


        velocity_error = float(
            info[
                "root_velocity_error"
            ]
        )


        q_error = float(
            info[
                "q_error_rms"
            ]
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


        root_errors.append(
            root_error
        )


        velocity_errors.append(
            velocity_error
        )


        q_errors.append(
            q_error
        )


        if (
            terminated
            or truncated
        ):

            terminated_flag = True

            break


    mean_root = float(
        np.mean(
            root_errors
        )
        if root_errors
        else 999.0
    )


    mean_velocity = float(
        np.mean(
            velocity_errors
        )
        if velocity_errors
        else 999.0
    )


    mean_q = float(
        np.mean(
            q_errors
        )
        if q_errors
        else 999.0
    )


    mean_action = float(
        np.mean(
            np.abs(
                segments
            )
        )
    )


    # -------------------------------------------------------------
    # AUTHORITY OBJECTIVE
    #
    # Survival/uprightness intentionally dominate.
    #
    # We are NOT training a final locomotion policy here.
    #
    # We only want to know whether residual control has enough
    # physical authority to alter the collapse.
    # -------------------------------------------------------------

    score = (
        8.0
        * survived

        + 50.0
        * min_up

        + 20.0
        * final_up

        - 0.35
        * max_orientation

        - 5.0
        * mean_root

        - 2.0
        * mean_velocity

        - 0.50
        * mean_q

        - 0.20
        * mean_action
    )


    if terminated_flag:

        score -= 40.0


    metrics = {

        "score":
            float(
                score
            ),

        "survived":
            survived,

        "min_up":
            min_up,

        "final_up":
            final_up,

        "max_orientation":
            max_orientation,

        "mean_root":
            mean_root,

        "mean_velocity":
            mean_velocity,

        "mean_q":
            mean_q,

        "mean_action":
            mean_action,

        "terminated":
            terminated_flag,
    }


    env.close()


    return (
        float(
            score
        ),
        metrics,
    )


# =====================================================================
# COMPLETE ROLLOUT
#
# After the searched intervention finishes, return to zero action
# and measure total episode survival.
# =====================================================================

def total_survival(
    segments,
):

    env = make_env()

    obs, info = warmup(
        env
    )


    sequence = expand_segments(
        segments
    )


    episode_steps = (
        WARMUP_STEPS
    )


    intervention_steps = 0

    min_up = float(
        info["up_z"]
    )

    max_orientation = 0.0

    max_action = 0.0


    for action in sequence:

        action = np.clip(
            action,
            -1.0,
            +1.0,
        ).astype(
            np.float32
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


        episode_steps += 1
        intervention_steps += 1


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


        if (
            terminated
            or truncated
        ):

            break


    # -------------------------------------------------------------
    # Return to raw reference after intervention.
    # -------------------------------------------------------------

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


        episode_steps += 1


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


    result = {

        "steps":
            episode_steps,

        "intervention":
            intervention_steps,

        "completed":
            bool(
                info.get(
                    "completed",
                    False,
                )
            ),

        "min_up":
            min_up,

        "max_orientation":
            max_orientation,

        "final_ref":
            int(
                info[
                    "reference_index"
                ]
            ),

        "final_x":
            float(
                info["x"]
            ),

        "max_action":
            max_action,
    }


    env.close()

    return result


# =====================================================================
# BASELINE
# =====================================================================

zero_segments = np.zeros(
    (
        NUM_SEGMENTS,
        15,
    ),
    dtype=np.float32,
)


baseline_score, baseline_metrics = (
    evaluate_candidate(
        zero_segments
    )
)


baseline_total = total_survival(
    zero_segments
)


print("=" * 130)
print("G1 V4 RESIDUAL ACTION-AUTHORITY TEST")
print("CEM SEARCH - NO PPO TRAINING")
print("=" * 130)

print(
    "start frame:",
    START_FRAME,
)

print(
    "warmup:",
    WARMUP_STEPS,
)

print(
    "search horizon:",
    SEARCH_HORIZON,
)

print(
    "segments:",
    NUM_SEGMENTS,
)

print(
    "segment length:",
    SEGMENT_STEPS,
)


print()
print("=" * 130)
print("ZERO-ACTION BASELINE")
print("=" * 130)

print(
    "branch score:",
    f"{baseline_score:.3f}",
)

print(
    "branch survived:",
    baseline_metrics[
        "survived"
    ],
    "/",
    SEARCH_HORIZON,
)

print(
    "branch min up:",
    f"{baseline_metrics['min_up']:.3f}",
)

print(
    "branch final up:",
    f"{baseline_metrics['final_up']:.3f}",
)

print(
    "branch max orientation:",
    f"{baseline_metrics['max_orientation']:.2f}",
    "deg",
)

print(
    "total episode steps:",
    baseline_total[
        "steps"
    ],
)

print(
    "total min up:",
    f"{baseline_total['min_up']:.3f}",
)


# =====================================================================
# CEM
# =====================================================================

rng = np.random.default_rng(
    SEED
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


best_score = (
    baseline_score
)

best_segments = (
    zero_segments.copy()
)

best_metrics = (
    baseline_metrics.copy()
)


print()
print("=" * 130)
print("CEM SEARCH")
print("=" * 130)


for iteration in range(
    1,
    ITERATIONS + 1,
):

    candidates = rng.normal(
        loc=mean,
        scale=std,
        size=(
            POPULATION,
            NUM_SEGMENTS,
            15,
        ),
    )


    candidates = np.clip(
        candidates,
        -1.0,
        +1.0,
    )


    scores = np.zeros(
        POPULATION,
        dtype=np.float64,
    )


    metrics_list = []


    for i in range(
        POPULATION
    ):

        score, metrics = (
            evaluate_candidate(
                candidates[i]
            )
        )


        scores[i] = score

        metrics_list.append(
            metrics
        )


        if score > best_score:

            best_score = (
                float(
                    score
                )
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


    elite_ids = np.argsort(
        scores
    )[-ELITES:]


    elites = candidates[
        elite_ids
    ]


    elite_mean = np.mean(
        elites,
        axis=0,
    )


    elite_std = np.std(
        elites,
        axis=0,
    )


    # Smoothed CEM update.
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


    iteration_best_id = int(
        np.argmax(
            scores
        )
    )


    iteration_metrics = (
        metrics_list[
            iteration_best_id
        ]
    )


    print(
        f"iteration={iteration} "
        f"best_score={scores[iteration_best_id]:+.3f} "
        f"global={best_score:+.3f} "
        f"survived="
        f"{iteration_metrics['survived']:02d}/"
        f"{SEARCH_HORIZON} "
        f"minUp="
        f"{iteration_metrics['min_up']:.3f} "
        f"finalUp="
        f"{iteration_metrics['final_up']:.3f} "
        f"ori="
        f"{iteration_metrics['max_orientation']:.1f} "
        f"root="
        f"{iteration_metrics['mean_root']:.3f} "
        f"stdMean="
        f"{np.mean(std):.3f}",
        flush=True,
    )


# =====================================================================
# FINAL BEST ROLLOUT
# =====================================================================

best_total = total_survival(
    best_segments
)


print()
print("=" * 130)
print("BEST SEARCHED INTERVENTION")
print("=" * 130)

print(
    "branch score:",
    f"{best_score:.3f}",
)

print(
    "branch survived:",
    best_metrics[
        "survived"
    ],
    "/",
    SEARCH_HORIZON,
)

print(
    "branch min up:",
    f"{best_metrics['min_up']:.3f}",
)

print(
    "branch final up:",
    f"{best_metrics['final_up']:.3f}",
)

print(
    "branch max orientation:",
    f"{best_metrics['max_orientation']:.2f}",
    "deg",
)

print(
    "branch mean root error:",
    f"{best_metrics['mean_root']:.3f}",
)

print(
    "branch mean velocity error:",
    f"{best_metrics['mean_velocity']:.3f}",
)

print(
    "mean normalized action:",
    f"{best_metrics['mean_action']:.3f}",
)


print()
print(
    "TOTAL EPISODE:"
)

print(
    "steps:",
    best_total[
        "steps"
    ],
)

print(
    "completed:",
    best_total[
        "completed"
    ],
)

print(
    "min up:",
    f"{best_total['min_up']:.3f}",
)

print(
    "max orientation:",
    f"{best_total['max_orientation']:.2f}",
    "deg",
)

print(
    "max normalized action:",
    f"{best_total['max_action']:.3f}",
)


# =====================================================================
# ACTION TABLE
# =====================================================================

probe = make_env()

probe.reset(
    seed=1234
)


scale = np.asarray(
    getattr(
        probe,
        "action_scale",
        np.ones(
            15,
            dtype=np.float32,
        ),
    ),
    dtype=np.float64,
).reshape(
    -1
)


if scale.size != 15:

    scale = np.ones(
        15,
        dtype=np.float64,
    )


joint_names = [
    item["name"]
    for item in probe.joint_info
]


probe.close()


print()
print("=" * 130)
print("BEST ACTION SEGMENTS")
print("=" * 130)


for segment_id in range(
    NUM_SEGMENTS
):

    print()
    print(
        f"SEGMENT {segment_id + 1} "
        f"(steps "
        f"{WARMUP_STEPS + segment_id*SEGMENT_STEPS + 1}"
        f"-"
        f"{WARMUP_STEPS + (segment_id+1)*SEGMENT_STEPS}"
        f")"
    )


    for j in range(15):

        normalized = float(
            best_segments[
                segment_id,
                j,
            ]
        )


        physical_rad = (
            normalized
            * scale[j]
        )


        physical_deg = (
            np.degrees(
                physical_rad
            )
        )


        print(
            f"  {j:02d} "
            f"{joint_names[j]:28s} "
            f"a={normalized:+.3f} "
            f"dq={physical_rad:+.4f}rad "
            f"({physical_deg:+.2f}deg)"
        )


# =====================================================================
# DECISION
# =====================================================================

delta_steps = (
    best_total["steps"]
    - baseline_total["steps"]
)


delta_branch_up = (
    best_metrics["min_up"]
    - baseline_metrics["min_up"]
)


print()
print("=" * 130)
print("AUTHORITY DECISION")
print("=" * 130)

print(
    "baseline total steps:",
    baseline_total[
        "steps"
    ],
)

print(
    "best total steps:",
    best_total[
        "steps"
    ],
)

print(
    "delta steps:",
    delta_steps,
)

print(
    "baseline branch min up:",
    f"{baseline_metrics['min_up']:.3f}",
)

print(
    "best branch min up:",
    f"{best_metrics['min_up']:.3f}",
)

print(
    "delta min up:",
    f"{delta_branch_up:+.3f}",
)


if (
    delta_steps >= 10
    or (
        best_metrics[
            "survived"
        ]
        == SEARCH_HORIZON

        and delta_branch_up
        >= 0.08
    )
):

    print()
    print(
        "RESIDUAL AUTHORITY: STRONG"
    )

    print(
        "The existing joint-position residual "
        "controller CAN materially alter the collapse."
    )

    print(
        "Next bottleneck = reward / credit assignment."
    )


elif (
    delta_steps >= 3
    or delta_branch_up >= 0.04
):

    print()
    print(
        "RESIDUAL AUTHORITY: WEAK"
    )

    print(
        "Residual control can influence the fall, "
        "but current authority may be marginal."
    )

    print(
        "Next step = inspect/widen selected joint "
        "action scales before more PPO."
    )


else:

    print()
    print(
        "RESIDUAL AUTHORITY: INSUFFICIENT"
    )

    print(
        "The current residual action space cannot "
        "meaningfully change the failure."
    )

    print(
        "Do NOT reward-tune PPO yet."
    )

    print(
        "Next step = change action authority/"
        "controller representation."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 130)
