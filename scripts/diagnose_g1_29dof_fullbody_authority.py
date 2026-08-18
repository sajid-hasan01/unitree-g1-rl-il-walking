import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

DATASET = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_29dof_v1_grounded.npz"
)


# ================================================================
# EXPERIMENT
# ================================================================

START_FRAME = 0

WARMUP_STEPS = 45

NUM_SEGMENTS = 5
SEGMENT_STEPS = 7

HORIZON = (
    NUM_SEGMENTS
    * SEGMENT_STEPS
)

SCREEN_POP = 80
SCREEN_ELITES = 10
SCREEN_ITERATIONS = 3

FULL_POP = 160
FULL_ELITES = 20
FULL_ITERATIONS = 4

INITIAL_STD = 0.40
MIN_STD = 0.06

SEED = 20260814

FAIL_HEIGHT = 0.45
FAIL_UP = 0.40


# ================================================================
# LOAD
# ================================================================

d = np.load(
    DATASET,
    allow_pickle=True,
)


ref_q = np.asarray(
    d["joint_pos_29"],
    dtype=np.float64,
)

ref_qd = np.asarray(
    d["joint_vel_29"],
    dtype=np.float64,
)

ref_full_qpos = np.asarray(
    d["full_qpos"],
    dtype=np.float64,
)

ref_full_qvel = np.asarray(
    d["full_qvel"],
    dtype=np.float64,
)

ref_root = np.asarray(
    d["root_positions"],
    dtype=np.float64,
)

ref_quat = np.asarray(
    d["root_quat_wxyz"],
    dtype=np.float64,
)

names = [
    str(x)
    for x in d["dof_names"]
]


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)


if len(names) != 29:

    raise RuntimeError(
        "Expected 29 reference joints."
    )


# ================================================================
# CONTROL RATE
# ================================================================

CONTROL_HZ = 50.0

sim_dt = float(
    model.opt.timestep
)

FRAME_SKIP = int(
    round(
        1.0
        / (
            CONTROL_HZ
            * sim_dt
        )
    )
)


# ================================================================
# JOINT / ACTUATOR ADDRESSES
# ================================================================

joint_ids = []
qaddrs = []
vaddrs = []
aids = []


for name in names:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )


    if jid < 0 or aid < 0:

        raise RuntimeError(
            f"Missing joint or actuator: {name}"
        )


    joint_ids.append(
        jid
    )

    qaddrs.append(
        int(
            model.jnt_qposadr[
                jid
            ]
        )
    )

    vaddrs.append(
        int(
            model.jnt_dofadr[
                jid
            ]
        )
    )

    aids.append(
        aid
    )


# ================================================================
# PHYSICAL RESIDUAL SCALES
#
# First 15 preserve scales already used in the previous diagnostic.
#
# Upper body uses moderate residual authority:
# shoulders/elbows roughly 5.7-6.9 degrees.
#
# Wrists are excluded from the search because the demonstration
# showed essentially zero wrist movement.
# ================================================================

ACTION_SCALE = np.array(
    [
        # legs + waist
        0.100,
        0.070,
        0.070,
        0.100,
        0.055,
        0.055,

        0.100,
        0.070,
        0.070,
        0.100,
        0.055,
        0.055,

        0.050,
        0.050,
        0.050,

        # left arm
        0.120,
        0.100,
        0.100,
        0.120,

        0.050,
        0.050,
        0.050,

        # right arm
        0.120,
        0.100,
        0.100,
        0.120,

        0.050,
        0.050,
        0.050,
    ],
    dtype=np.float64,
)


LOWER15 = list(
    range(15)
)

ARMS8 = [
    15,
    16,
    17,
    18,

    22,
    23,
    24,
    25,
]


FULL23 = (
    LOWER15
    + ARMS8
)


CONFIGS = {
    "LOWER15":
        {
            "indices": LOWER15,
            "scale_multiplier": 1.0,
        },

    "ARMS8":
        {
            "indices": ARMS8,
            "scale_multiplier": 1.0,
        },

    "ARMS8_1.5":
        {
            "indices": ARMS8,
            "scale_multiplier": 1.5,
        },

    "FULL23":
        {
            "indices": FULL23,
            "scale_multiplier": 1.0,
        },

    "FULL23_ARMS1.5":
        {
            "indices": FULL23,
            "scale_multiplier": 1.0,
            "arm_multiplier": 1.5,
        },
}


pelvis_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pelvis",
)


# ================================================================
# HELPERS
# ================================================================

def get_up(
    data,
):

    return float(
        data.xmat[
            pelvis_id
        ].reshape(
            3,
            3,
        )[2, 2]
    )


def quat_error_deg(
    actual,
    desired,
):

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    desired = np.asarray(
        desired,
        dtype=np.float64,
    )


    actual /= max(
        np.linalg.norm(actual),
        1e-12,
    )

    desired /= max(
        np.linalg.norm(desired),
        1e-12,
    )


    dot = abs(
        float(
            np.dot(
                actual,
                desired,
            )
        )
    )


    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )


    return math.degrees(
        2.0
        * math.acos(dot)
    )


def actual_joint_q(
    data,
):

    return np.asarray(
        [
            data.qpos[qadr]
            for qadr in qaddrs
        ],
        dtype=np.float64,
    )


def actual_joint_qd(
    data,
):

    return np.asarray(
        [
            data.qvel[vadr]
            for vadr in vaddrs
        ],
        dtype=np.float64,
    )


def physical_failure(
    data,
):

    return (
        float(
            data.qpos[2]
        ) < FAIL_HEIGHT

        or get_up(data)
        < FAIL_UP
    )


def clip_target(
    joint_index,
    value,
):

    jid = joint_ids[
        joint_index
    ]

    aid = aids[
        joint_index
    ]


    result = float(
        value
    )


    if model.jnt_limited[
        jid
    ]:

        lo, hi = (
            model.jnt_range[
                jid
            ]
        )

        result = float(
            np.clip(
                result,
                lo,
                hi,
            )
        )


    if model.actuator_ctrllimited[
        aid
    ]:

        lo, hi = (
            model.actuator_ctrlrange[
                aid
            ]
        )

        result = float(
            np.clip(
                result,
                lo,
                hi,
            )
        )


    return result


def apply_targets(
    data,
    ref_index,
    residual=None,
):

    if residual is None:

        residual = np.zeros(
            29,
            dtype=np.float64,
        )


    for j in range(29):

        target = (
            ref_q[
                ref_index,
                j,
            ]
            + residual[j]
        )


        data.ctrl[
            aids[j]
        ] = clip_target(
            j,
            target,
        )


def sim_control_step(
    data,
):

    for _ in range(
        FRAME_SKIP
    ):

        mujoco.mj_step(
            model,
            data,
        )


# ================================================================
# RESET + WARMUP
# ================================================================

def create_warmed_state():

    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    data.qpos[:] = (
        ref_full_qpos[
            START_FRAME
        ]
    )


    data.qvel[:] = (
        ref_full_qvel[
            START_FRAME
        ]
    )


    apply_targets(
        data,
        START_FRAME,
    )


    mujoco.mj_forward(
        model,
        data,
    )


    current_ref = (
        START_FRAME
    )


    for _ in range(
        WARMUP_STEPS
    ):

        current_ref += 1


        apply_targets(
            data,
            current_ref,
        )


        sim_control_step(
            data
        )


        if physical_failure(
            data
        ):

            raise RuntimeError(
                "Reference failed before CEM branch."
            )


    return data


def make_candidate_data():

    # Replaying warmup guarantees a deterministic branch state.
    return create_warmed_state()


# ================================================================
# CONFIG SCALE
# ================================================================

def configuration_scales(
    config,
):

    result = (
        ACTION_SCALE.copy()
        * float(
            config.get(
                "scale_multiplier",
                1.0,
            )
        )
    )


    # For FULL23_ARMS1.5 only arms receive extra scaling.
    if (
        "arm_multiplier"
        in config
    ):

        arm_multiplier = float(
            config[
                "arm_multiplier"
            ]
        )


        result[
            ARMS8
        ] = (
            ACTION_SCALE[
                ARMS8
            ]
            * arm_multiplier
        )


        # Lower body stays at original 1x.
        result[
            LOWER15
        ] = ACTION_SCALE[
            LOWER15
        ]


    return result


# ================================================================
# EXPAND CEM ACTIONS
# ================================================================

def build_sequence(
    compact_segments,
    indices,
    scales,
):

    sequence = []


    for segment in compact_segments:

        residual = np.zeros(
            29,
            dtype=np.float64,
        )


        for local_index, joint_index in enumerate(
            indices
        ):

            normalized = float(
                np.clip(
                    segment[
                        local_index
                    ],
                    -1.0,
                    1.0,
                )
            )


            residual[
                joint_index
            ] = (
                normalized
                * scales[
                    joint_index
                ]
            )


        for _ in range(
            SEGMENT_STEPS
        ):

            sequence.append(
                residual.copy()
            )


    return sequence


# ================================================================
# BRANCH EVALUATION
# ================================================================

def evaluate_candidate(
    compact_segments,
    config,
):

    indices = config[
        "indices"
    ]


    scales = configuration_scales(
        config
    )


    data = make_candidate_data()


    sequence = build_sequence(
        compact_segments,
        indices,
        scales,
    )


    survived = 0

    min_up = get_up(
        data
    )

    final_up = min_up

    max_orientation = 0.0
    max_y = 0.0

    root_values = []
    velocity_values = []
    q_values = []

    current_ref = (
        START_FRAME
        + WARMUP_STEPS
    )


    failed = False


    for residual in sequence:

        current_ref = min(
            current_ref + 1,
            len(ref_q) - 1,
        )


        apply_targets(
            data,
            current_ref,
            residual,
        )


        sim_control_step(
            data
        )


        survived += 1


        up = get_up(
            data
        )


        orientation = quat_error_deg(
            data.qpos[
                3:7
            ],

            ref_quat[
                current_ref
            ],
        )


        root_error = float(
            np.linalg.norm(
                data.qpos[
                    0:3
                ]
                - ref_root[
                    current_ref
                ]
            )
        )


        velocity_error = float(
            np.linalg.norm(
                data.qvel[
                    0:3
                ]
                - ref_full_qvel[
                    current_ref,
                    0:3
                ]
            )
        )


        q_error = float(
            np.sqrt(
                np.mean(
                    (
                        actual_joint_q(
                            data
                        )
                        - ref_q[
                            current_ref
                        ]
                    ) ** 2
                )
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

            abs(
                float(
                    data.qpos[1]
                    - ref_root[
                        current_ref,
                        1,
                    ]
                )
            ),
        )


        root_values.append(
            root_error
        )

        velocity_values.append(
            velocity_error
        )

        q_values.append(
            q_error
        )


        if (
            physical_failure(
                data
            )

            or current_ref
            >= len(ref_q) - 1
        ):

            failed = physical_failure(
                data
            )

            break


    mean_root = float(
        np.mean(
            root_values
        )
        if root_values
        else 999.0
    )


    mean_velocity = float(
        np.mean(
            velocity_values
        )
        if velocity_values
        else 999.0
    )


    mean_q = float(
        np.mean(
            q_values
        )
        if q_values
        else 999.0
    )


    mean_action = float(
        np.mean(
            np.abs(
                compact_segments
            )
        )
    )


    # ============================================================
    # Objective strongly penalizes "falling differently."
    # ============================================================

    score = (

        12.0
        * survived

        + 120.0
        * min_up

        + 40.0
        * final_up

        - 1.75
        * max_orientation

        - 25.0
        * mean_root

        - 3.0
        * mean_velocity

        - 30.0
        * max_y

        - 0.50
        * mean_q

        - 0.25
        * mean_action
    )


    if failed:

        score -= 30.0


    return (
        float(
            score
        ),

        {
            "survived":
                survived,

            "min_up":
                min_up,

            "final_up":
                final_up,

            "orientation":
                max_orientation,

            "root":
                mean_root,

            "velocity":
                mean_velocity,

            "max_y":
                max_y,

            "q":
                mean_q,

            "action":
                mean_action,
        },
    )


# ================================================================
# COMPLETE EPISODE
# ================================================================

def run_complete(
    compact_segments,
    config,
):

    indices = config[
        "indices"
    ]

    scales = configuration_scales(
        config
    )


    data = make_candidate_data()


    current_ref = (
        START_FRAME
        + WARMUP_STEPS
    )


    steps = (
        WARMUP_STEPS
    )


    min_up = get_up(
        data
    )

    max_orientation = 0.0

    max_y = 0.0


    sequence = build_sequence(
        compact_segments,
        indices,
        scales,
    )


    # Intervention.
    for residual in sequence:

        current_ref = min(
            current_ref + 1,
            len(ref_q) - 1,
        )


        apply_targets(
            data,
            current_ref,
            residual,
        )


        sim_control_step(
            data
        )


        steps += 1


        up = get_up(
            data
        )


        orientation = quat_error_deg(
            data.qpos[
                3:7
            ],

            ref_quat[
                current_ref
            ],
        )


        min_up = min(
            min_up,
            up,
        )


        max_orientation = max(
            max_orientation,
            orientation,
        )


        max_y = max(
            max_y,

            abs(
                float(
                    data.qpos[1]
                    - ref_root[
                        current_ref,
                        1,
                    ]
                )
            ),
        )


        if (
            physical_failure(
                data
            )

            or current_ref
            >= len(ref_q) - 1
        ):

            break


    # Return to raw full-body reference.
    while (
        not physical_failure(
            data
        )

        and current_ref
        < len(ref_q) - 1
    ):

        current_ref += 1


        apply_targets(
            data,
            current_ref,
        )


        sim_control_step(
            data
        )


        steps += 1


        up = get_up(
            data
        )


        orientation = quat_error_deg(
            data.qpos[
                3:7
            ],

            ref_quat[
                current_ref
            ],
        )


        min_up = min(
            min_up,
            up,
        )


        max_orientation = max(
            max_orientation,
            orientation,
        )


        max_y = max(
            max_y,

            abs(
                float(
                    data.qpos[1]
                    - ref_root[
                        current_ref,
                        1,
                    ]
                )
            ),
        )


    return {
        "steps":
            steps,

        "completed":
            bool(
                current_ref
                >= len(ref_q) - 1

                and not physical_failure(
                    data
                )
            ),

        "min_up":
            min_up,

        "orientation":
            max_orientation,

        "max_y":
            max_y,

        "final_ref":
            current_ref,
    }


# ================================================================
# TRUE ZERO-RESIDUAL BASELINE
# ================================================================

zero_config = {
    "indices": [],
    "scale_multiplier": 1.0,
}


zero_segments = np.zeros(
    (
        NUM_SEGMENTS,
        0,
    ),
    dtype=np.float64,
)


baseline = run_complete(
    zero_segments,
    zero_config,
)


print("=" * 145)
print("29-DOF FULL-BODY RESIDUAL AUTHORITY TEST")
print("NO PPO TRAINING")
print("=" * 145)


print()
print("=" * 145)
print("ZERO-RESIDUAL BASELINE")
print("=" * 145)

print(
    "steps:",
    baseline["steps"],
)

print(
    "completed:",
    baseline["completed"],
)

print(
    "min up:",
    f"{baseline['min_up']:.3f}",
)

print(
    "max orientation:",
    f"{baseline['orientation']:.2f}",
)

print(
    "max Y error:",
    f"{baseline['max_y']:.3f}",
)


# ================================================================
# CEM
# ================================================================

def cem(
    config,
    population,
    elite_count,
    iterations,
    seed,
):

    indices = config[
        "indices"
    ]

    dim = len(
        indices
    )


    rng = np.random.default_rng(
        seed
    )


    mean = np.zeros(
        (
            NUM_SEGMENTS,
            dim,
        ),
        dtype=np.float64,
    )


    std = np.full(
        (
            NUM_SEGMENTS,
            dim,
        ),
        INITIAL_STD,
        dtype=np.float64,
    )


    zero = np.zeros_like(
        mean
    )


    best_score, best_metrics = (
        evaluate_candidate(
            zero,
            config,
        )
    )


    best_segments = (
        zero.copy()
    )


    for iteration in range(
        1,
        iterations + 1,
    ):

        candidates = rng.normal(
            loc=mean,
            scale=std,

            size=(
                population,
                NUM_SEGMENTS,
                dim,
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
                    candidates[i],
                    config,
                )
            )


            scores[i] = score


            if score > best_score:

                best_score = (
                    float(
                        score
                    )
                )

                best_segments = (
                    candidates[i].copy()
                )

                best_metrics = (
                    metrics.copy()
                )


        elite_ids = np.argsort(
            scores
        )[
            -elite_count:
        ]


        elite = candidates[
            elite_ids
        ]


        mean = (
            0.25
            * mean

            + 0.75
            * np.mean(
                elite,
                axis=0,
            )
        )


        std = (
            0.25
            * std

            + 0.75
            * np.std(
                elite,
                axis=0,
            )
        )


        std = np.maximum(
            std,
            MIN_STD,
        )


    total = run_complete(
        best_segments,
        config,
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


# ================================================================
# SCREEN
# ================================================================

results = {}


print()
print("=" * 145)
print("GROUP AUTHORITY SCREEN")
print("=" * 145)


for k, (
    name,
    config,
) in enumerate(
    CONFIGS.items()
):

    result = cem(

        config,

        population=
            SCREEN_POP,

        elite_count=
            SCREEN_ELITES,

        iterations=
            SCREEN_ITERATIONS,

        seed=
            SEED + k,
    )


    results[
        name
    ] = result


    b = result[
        "branch"
    ]

    t = result[
        "total"
    ]


    print(
        f"{name:20s} "
        f"steps={t['steps']:3d} "
        f"delta={t['steps']-baseline['steps']:+3d} "
        f"minUp={t['min_up']:.3f} "
        f"ori={t['orientation']:6.1f} "
        f"maxY={t['max_y']:.3f} "
        f"branch={b['survived']:02d}/{HORIZON} "
        f"root={b['root']:.3f} "
        f"vel={b['velocity']:.3f} "
        f"score={result['score']:+.1f}",
        flush=True,
    )


# ================================================================
# SCREEN WINNER
# ================================================================

winner_name = max(

    results.keys(),

    key=lambda name: (

        results[
            name
        ]["total"]["steps"],

        results[
            name
        ]["total"]["min_up"],

        -results[
            name
        ]["total"]["orientation"],

        -results[
            name
        ]["total"]["max_y"],
    ),
)


print()
print("=" * 145)
print("SCREEN WINNER")
print("=" * 145)

print(
    "configuration:",
    winner_name,
)


# ================================================================
# FULL CONFIRMATION
# ================================================================

winner = cem(

    CONFIGS[
        winner_name
    ],

    population=
        FULL_POP,

    elite_count=
        FULL_ELITES,

    iterations=
        FULL_ITERATIONS,

    seed=
        SEED + 999,
)


b = winner[
    "branch"
]

t = winner[
    "total"
]


print()
print("=" * 145)
print("WINNER FULL CEM")
print("=" * 145)

print(
    "configuration:",
    winner_name,
)

print(
    "branch survived:",
    f"{b['survived']}/{HORIZON}",
)

print(
    "branch min up:",
    f"{b['min_up']:.3f}",
)

print(
    "branch final up:",
    f"{b['final_up']:.3f}",
)

print(
    "branch orientation:",
    f"{b['orientation']:.2f}",
)

print(
    "branch root error:",
    f"{b['root']:.3f}",
)

print(
    "branch velocity error:",
    f"{b['velocity']:.3f}",
)

print(
    "branch max Y:",
    f"{b['max_y']:.3f}",
)


print()
print(
    "episode steps:",
    t["steps"],
)

print(
    "delta steps:",
    t["steps"]
    - baseline["steps"],
)

print(
    "episode min up:",
    f"{t['min_up']:.3f}",
)

print(
    "episode max orientation:",
    f"{t['orientation']:.2f}",
)

print(
    "episode max Y:",
    f"{t['max_y']:.3f}",
)

print(
    "completed:",
    t["completed"],
)


# ================================================================
# ARM CONTRIBUTION
# ================================================================

lower = results[
    "LOWER15"
]["total"]


arm_names = [
    "ARMS8",
    "ARMS8_1.5",
    "FULL23",
    "FULL23_ARMS1.5",
]


best_arm_name = max(

    arm_names,

    key=lambda name: (
        results[
            name
        ]["total"]["steps"],

        -results[
            name
        ]["total"]["orientation"],
    ),
)


best_arm = results[
    best_arm_name
]["total"]


print()
print("=" * 145)
print("ARM CONTRIBUTION")
print("=" * 145)

print(
    "best lower-only steps:",
    lower["steps"],
)

print(
    "best arm-containing config:",
    best_arm_name,
)

print(
    "arm-containing steps:",
    best_arm["steps"],
)

print(
    "difference vs lower-only:",
    best_arm["steps"]
    - lower["steps"],
)

print(
    "lower-only orientation:",
    f"{lower['orientation']:.2f}",
)

print(
    "arm-containing orientation:",
    f"{best_arm['orientation']:.2f}",
)


# ================================================================
# DECISION
# ================================================================

delta = (
    t["steps"]
    - baseline["steps"]
)


ori_change = (
    t["orientation"]
    - baseline["orientation"]
)


up_change = (
    t["min_up"]
    - baseline["min_up"]
)


print()
print("=" * 145)
print("FULL-BODY AUTHORITY DECISION")
print("=" * 145)

print(
    "baseline steps:",
    baseline["steps"],
)

print(
    "winner steps:",
    t["steps"],
)

print(
    "delta steps:",
    delta,
)

print(
    "orientation change:",
    f"{ori_change:+.2f} deg",
)

print(
    "min-up change:",
    f"{up_change:+.3f}",
)


if (
    delta >= 12
    and ori_change <= 15.0
    and up_change >= 0.04
):

    print()
    print(
        "FULL-BODY POSITION AUTHORITY: STRONG"
    )

    print(
        "The 29-DOF/23-moving-joint residual "
        "controller can materially stabilize "
        "the reference."
    )

    print(
        "NEXT: build 29-DOF state-aware PPO "
        "tracking environment."
    )


elif (
    delta >= 6
    and ori_change <= 15.0
):

    print()
    print(
        "FULL-BODY POSITION AUTHORITY: MARGINAL"
    )

    print(
        "Upper-body feedback helps, but "
        "joint-position residual authority "
        "is still near its limit."
    )

    print(
        "NEXT: inspect controller gains/action "
        "representation before PPO."
    )


else:

    print()
    print(
        "FULL-BODY POSITION AUTHORITY: INSUFFICIENT"
    )

    print(
        "Adding upper-body joint-position "
        "feedback does not create a clean "
        "recovery controller."
    )

    print(
        "DO NOT TRAIN PPO."
    )

    print(
        "NEXT: change control formulation rather "
        "than adding reward terms or timesteps."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 145)
