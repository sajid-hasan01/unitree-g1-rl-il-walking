from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


# ================================================================
# CONFIG
# ================================================================

CANDIDATE_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "medium_02_dynamic_v2_centroidal_root_candidate.npz"
)


STARTS = [
    0,
    15,
    30,
    45,
    57,
    75,
    85,
    105,
    114,
    135,
    140,
]


HORIZON = 120


ZERO_ACTION = np.zeros(
    29,
    dtype=np.float32,
)


# ================================================================
# LOAD CANDIDATE
# ================================================================

if not CANDIDATE_PATH.exists():

    raise FileNotFoundError(
        CANDIDATE_PATH
    )


with np.load(
    CANDIDATE_PATH,
    allow_pickle=True,
) as f:

    if "full_qpos" not in f.files:

        raise RuntimeError(
            "Candidate has no full_qpos."
        )


    if "full_qvel" not in f.files:

        raise RuntimeError(
            "Candidate has no full_qvel."
        )


    candidate_qpos = np.asarray(
        f[
            "full_qpos"
        ],
        dtype=np.float64,
    ).copy()


    candidate_qvel = np.asarray(
        f[
            "full_qvel"
        ],
        dtype=np.float64,
    ).copy()


print("=" * 190)
print("G1 DYNAMIC V2 PASS-2B1")
print("FORWARD-PHYSICS VALIDATION")
print("ORIGINAL vs DIRECT ROOT-DYNAMICS CANDIDATE")
print("ZERO RESIDUAL ACTION")
print("NO CEM / NO PPO")
print("=" * 190)

print(
    "candidate:",
    CANDIDATE_PATH,
)

print(
    "candidate qpos:",
    candidate_qpos.shape,
)

print(
    "candidate qvel:",
    candidate_qvel.shape,
)


# ================================================================
# MAKE ENVIRONMENT
# ================================================================

def make_env(
    candidate=False,
):

    env = G129DofTrackingRestartV1(
        rsi=False,
        reset_joint_noise=0.0,
        reset_velocity_noise=0.0,
    )


    if not candidate:

        return env


    if (
        candidate_qpos.shape
        != env.ref_full_qpos.shape
    ):

        raise RuntimeError(
            "Candidate qpos shape mismatch: "
            f"{candidate_qpos.shape} "
            f"vs {env.ref_full_qpos.shape}"
        )


    if (
        candidate_qvel.shape
        != env.ref_full_qvel.shape
    ):

        raise RuntimeError(
            "Candidate qvel shape mismatch."
        )


    # ------------------------------------------------------------
    # Candidate modifies root only.
    # Joint reference trajectory remains unchanged.
    # ------------------------------------------------------------

    env.ref_full_qpos = (
        candidate_qpos.copy()
    )


    env.ref_full_qvel = (
        candidate_qvel.copy()
    )


    env.ref_root_pos = (
        candidate_qpos[
            :,
            0:3
        ].copy()
    )


    env.ref_root_quat = (
        candidate_qpos[
            :,
            3:7
        ].copy()
    )


    # Patch additional cached root velocity arrays if present.
    optional_replacements = {

        "ref_root_linvel":
            candidate_qvel[
                :,
                0:3
            ].copy(),

        "ref_root_angvel":
            candidate_qvel[
                :,
                3:6
            ].copy(),

        "ref_root_linear_velocity":
            candidate_qvel[
                :,
                0:3
            ].copy(),

        "ref_root_angular_velocity":
            candidate_qvel[
                :,
                3:6
            ].copy(),
    }


    for name, value in (
        optional_replacements.items()
    ):

        if hasattr(
            env,
            name,
        ):

            current = getattr(
                env,
                name,
            )


            if (
                isinstance(
                    current,
                    np.ndarray,
                )
                and
                current.shape
                ==
                value.shape
            ):

                setattr(
                    env,
                    name,
                    value,
                )


    env.reference_path = (
        CANDIDATE_PATH
    )


    return env


# ================================================================
# VERIFY CANDIDATE RESET
# ================================================================

candidate_env = make_env(
    candidate=True
)


(
    obs,
    info,
) = candidate_env.reset(
    options={
        "start_frame":
            0
    }
)


reset_qpos_error = float(
    np.max(
        np.abs(
            candidate_env.data.qpos
            -
            candidate_qpos[
                0
            ]
        )
    )
)


reset_qvel_error = float(
    np.max(
        np.abs(
            candidate_env.data.qvel
            -
            candidate_qvel[
                0
            ]
        )
    )
)


print()
print("=" * 190)
print("CANDIDATE RESET VALIDATION")
print("=" * 190)

print(
    "qpos max error:",
    f"{reset_qpos_error:.3e}",
)

print(
    "qvel max error:",
    f"{reset_qvel_error:.3e}",
)


if (
    reset_qpos_error
    > 1e-6
):

    raise RuntimeError(
        "Candidate qpos is not being used "
        "by reset()."
    )


if (
    reset_qvel_error
    > 1e-6
):

    raise RuntimeError(
        "Candidate qvel is not being used "
        "by reset()."
    )


candidate_env.close()


# ================================================================
# CONTACT HELPER
# ================================================================

def actual_contact(
    env,
):

    try:

        c = env._actual_contacts()

        arr = np.asarray(
            c
        ).reshape(
            -1
        )


        if len(
            arr
        ) >= 2:

            return bool(
                arr[0]
                or
                arr[1]
            )


    except Exception:

        pass


    # Fallback: direct MuJoCo check.
    left_set = set(
        env.left_sole_geoms
    )


    right_set = set(
        env.right_sole_geoms
    )


    for cid in range(
        env.data.ncon
    ):

        con = (
            env.data.contact[
                cid
            ]
        )


        if int(
            con.efc_address
        ) < 0:

            continue


        g1 = int(
            con.geom1
        )


        g2 = int(
            con.geom2
        )


        if env.floor_geom not in (
            g1,
            g2,
        ):

            continue


        other = (
            g2
            if g1
            == env.floor_geom
            else g1
        )


        if (
            other in left_set
            or
            other in right_set
        ):

            return True


    return False


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    mode,
    start,
):

    env = make_env(
        candidate=(
            mode
            ==
            "PASS2B1"
        )
    )


    (
        obs,
        info,
    ) = env.reset(
        options={
            "start_frame":
                int(
                    start
                )
        }
    )


    maximum_steps = min(
        HORIZON,
        env.num_frames
        - 1
        - start,
    )


    initial_root_x = float(
        env.data.qpos[
            0
        ]
    )


    steps = 0

    orientation_sum = 0.0
    qerror_sum = 0.0
    root_error_sum = 0.0

    contact_steps = 0

    min_up = float(
        env._up_z()
    )


    reason = ""


    while steps < maximum_steps:

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            ZERO_ACTION
        )


        steps += 1


        # --------------------------------------------------------
        # Metrics already proven to work in previous diagnostics.
        # --------------------------------------------------------

        terms = info[
            "reward_terms"
        ]


        orientation_sum += float(
            terms[
                "root_orientation_deg"
            ]
        )


        qerror_sum += float(
            terms[
                "q_error"
            ]
        )


        min_up = min(
            min_up,
            float(
                info[
                    "up"
                ]
            ),
        )


        if actual_contact(
            env
        ):

            contact_steps += 1


        # --------------------------------------------------------
        # Compute root error directly against current reference.
        # --------------------------------------------------------

        try:

            frame = int(
                env._current_frame()
            )

        except Exception:

            frame = min(
                start + steps,
                env.num_frames - 1,
            )


        root_error_sum += float(
            np.linalg.norm(
                env.data.qpos[
                    0:3
                ]
                -
                env.ref_root_pos[
                    frame
                ]
            )
        )


        if terminated:

            reason = info.get(
                "termination_reason",
                "terminated",
            )

            break


        if truncated:

            reason = (
                "reference_end"
            )

            break


    if not reason:

        reason = "horizon"


    divisor = max(
        steps,
        1,
    )


    actual_dx = float(
        env.data.qpos[
            0
        ]
        -
        initial_root_x
    )


    result = {

        "mode":
            mode,

        "start":
            start,

        "steps":
            steps,

        "maximum_steps":
            maximum_steps,

        "progress":
            steps
            /
            maximum_steps,

        "orientation":
            orientation_sum
            /
            divisor,

        "qerror":
            qerror_sum
            /
            divisor,

        "root_error":
            root_error_sum
            /
            divisor,

        "min_up":
            min_up,

        "contact":
            contact_steps
            /
            divisor,

        "dx":
            actual_dx,

        "reason":
            reason,
    }


    env.close()


    return result


# ================================================================
# RUN
# ================================================================

results = []


for mode in (
    "ORIGINAL",
    "PASS2B1",
):

    print()
    print("=" * 190)
    print(
        "RUNNING:",
        mode,
    )
    print("=" * 190)


    for start in STARTS:

        result = rollout(
            mode,
            start,
        )


        results.append(
            result
        )


        print(
            f"{mode:8s} "
            f"start={start:3d} "
            f"steps="
            f"{result['steps']:3d}/"
            f"{result['maximum_steps']:<3d} "
            f"prog={result['progress']:.3f} "
            f"ori={result['orientation']:6.2f} "
            f"root={result['root_error']:.3f} "
            f"qerr={result['qerror']:.3f} "
            f"minup={result['min_up']:.3f} "
            f"contact={result['contact']:.3f} "
            f"dx={result['dx']:+.3f} "
            f"{result['reason']}"
        )


# ================================================================
# BASELINE CONSISTENCY
# ================================================================

expected = {
    0:
        102,

    57:
        51,

    114:
        58,
}


baseline_exact = True


for start, expected_steps in (
    expected.items()
):

    row = next(
        r
        for r in results
        if (
            r[
                "mode"
            ]
            ==
            "ORIGINAL"

            and
            r[
                "start"
            ]
            ==
            start
        )
    )


    if (
        row[
            "steps"
        ]
        !=
        expected_steps
    ):

        baseline_exact = False


print()
print(
    "baseline exact 102/51/58:",
    baseline_exact,
)


if not baseline_exact:

    raise RuntimeError(
        "Original forward baseline changed. "
        "Do not interpret candidate comparison."
    )


# ================================================================
# AGGREGATE
# ================================================================

def aggregate(
    mode,
):

    rows = [
        r
        for r in results
        if r[
            "mode"
        ] == mode
    ]


    def mean(
        key,
    ):

        return float(
            np.mean(
                [
                    r[
                        key
                    ]
                    for r in rows
                ]
            )
        )


    return {

        "steps":
            mean(
                "steps"
            ),

        "progress":
            mean(
                "progress"
            ),

        "orientation":
            mean(
                "orientation"
            ),

        "root_error":
            mean(
                "root_error"
            ),

        "qerror":
            mean(
                "qerror"
            ),

        "min_up":
            mean(
                "min_up"
            ),

        "contact":
            mean(
                "contact"
            ),

        "dx":
            mean(
                "dx"
            ),

        "minimum_progress":
            float(
                np.min(
                    [
                        r[
                            "progress"
                        ]
                        for r in rows
                    ]
                )
            ),

        "completions":
            int(
                np.sum(
                    [
                        r[
                            "progress"
                        ]
                        >=
                        0.999
                        for r in rows
                    ]
                )
            ),
    }


original = aggregate(
    "ORIGINAL"
)


candidate = aggregate(
    "PASS2B1"
)


print()
print("=" * 190)
print("FORWARD-PHYSICS AGGREGATE")
print("=" * 190)

print(
    f"{'REF':10s} "
    f"{'STEPS':>7s} "
    f"{'PROG':>7s} "
    f"{'MIN-P':>7s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'QERR':>8s} "
    f"{'MINUP':>8s} "
    f"{'CONTACT':>8s} "
    f"{'DX':>8s} "
    f"{'DONE':>5s}"
)

print("-" * 190)


for name, result in (
    (
        "ORIGINAL",
        original,
    ),
    (
        "PASS2B1",
        candidate,
    ),
):

    print(
        f"{name:10s} "
        f"{result['steps']:7.1f} "
        f"{result['progress']:7.3f} "
        f"{result['minimum_progress']:7.3f} "
        f"{result['orientation']:8.2f} "
        f"{result['root_error']:8.3f} "
        f"{result['qerror']:8.3f} "
        f"{result['min_up']:8.3f} "
        f"{result['contact']:8.3f} "
        f"{result['dx']:8.3f} "
        f"{result['completions']:5d}"
    )


# ================================================================
# DECISION
# ================================================================

progress_gain = (
    candidate[
        "progress"
    ]
    -
    original[
        "progress"
    ]
)


orientation_change = (
    candidate[
        "orientation"
    ]
    -
    original[
        "orientation"
    ]
)


root_change = (
    candidate[
        "root_error"
    ]
    -
    original[
        "root_error"
    ]
)


print()
print("=" * 190)
print("PASS-2B1 FORWARD VALIDATION DECISION")
print("=" * 190)

print(
    "progress:",
    f"{original['progress']:.3f}",
    "->",
    f"{candidate['progress']:.3f}",
    f"({progress_gain:+.3f})",
)

print(
    "orientation:",
    f"{original['orientation']:.2f}",
    "->",
    f"{candidate['orientation']:.2f}",
    "deg",
    f"({orientation_change:+.2f})",
)

print(
    "root error:",
    f"{original['root_error']:.3f}",
    "->",
    f"{candidate['root_error']:.3f}",
    "m",
    f"({root_change:+.3f})",
)

print(
    "minimum progress:",
    f"{original['minimum_progress']:.3f}",
    "->",
    f"{candidate['minimum_progress']:.3f}",
)

print(
    "completion count:",
    original[
        "completions"
    ],
    "->",
    candidate[
        "completions"
    ],
)


print()


if (
    progress_gain
    >= 0.10

    and
    orientation_change
    <= 5.0

    and
    root_change
    <= 0.08
):

    print(
        "RESULT: PASS-2B1 FORWARD VALIDATION PASSES"
    )

    print(
        "The inverse-dynamics improvement translates "
        "to substantially better forward physics."
    )

    print(
        "KEEP this candidate as the base for Pass 2B2."
    )

    print(
        "NEXT: add a SMALL lower-body spline set "
        "with stronger actuator-limit penalties."
    )


elif (
    progress_gain
    >= 0.04

    and
    orientation_change
    <= 7.0
):

    print(
        "RESULT: PASS-2B1 FORWARD EFFECT IS MARGINAL "
        "BUT POSITIVE"
    )

    print(
        "The dynamics objective has predictive value, "
        "but the reference still needs lower-body repair."
    )

    print(
        "KEEP as an intermediate candidate."
    )

    print(
        "NEXT: Pass 2B2, but keep the parameter count small."
    )


elif (
    progress_gain
    > -0.03

    and
    candidate[
        "orientation"
    ]
    <
    original[
        "orientation"
    ]
):

    print(
        "RESULT: SURVIVAL IS FLAT BUT ORIENTATION IMPROVES"
    )

    print(
        "Do not train PPO yet."
    )

    print(
        "Pass 2B2 may still be justified, "
        "but actuator feasibility must be targeted."
    )


else:

    print(
        "RESULT: PASS-2B1 DOES NOT GENERALIZE "
        "TO FORWARD PHYSICS"
    )

    print(
        "Do NOT expand the optimizer yet."
    )

    print(
        "NEXT: compare medium_04 dynamic feasibility "
        "against medium_02 before additional optimization."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_dynamic_v2_pass2b1_forward_validation.npz"
)


np.savez_compressed(
    output,

    starts=
        np.asarray(
            STARTS,
            dtype=np.int32,
        ),

    original_progress=
        np.asarray(
            [
                original[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    candidate_progress=
        np.asarray(
            [
                candidate[
                    "progress"
                ]
            ],
            dtype=np.float32,
        ),

    progress_gain=
        np.asarray(
            [
                progress_gain
            ],
            dtype=np.float32,
        ),

    original_orientation=
        np.asarray(
            [
                original[
                    "orientation"
                ]
            ],
            dtype=np.float32,
        ),

    candidate_orientation=
        np.asarray(
            [
                candidate[
                    "orientation"
                ]
            ],
            dtype=np.float32,
        ),

    baseline_exact=
        np.asarray(
            [
                baseline_exact
            ],
            dtype=np.bool_,
        ),
)


print()
print(
    "artifact:",
    output,
)

print()
print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 190)
