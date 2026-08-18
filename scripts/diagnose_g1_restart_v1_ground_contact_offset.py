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


OFFSETS_MM = [
    0.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    12.0,
    15.0,
]


HORIZON = 120


# ================================================================
# ENV
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


model = env.model
data = env.data


BASE_FLOOR_Z = float(
    model.geom_pos[
        env.floor_geom,
        2,
    ]
)


LEFT_SOLES = set(
    env.left_sole_geoms
)

RIGHT_SOLES = set(
    env.right_sole_geoms
)


print("=" * 190)
print("G1 RESTART V1 — TRUE GROUND-CONTACT GAP DIAGNOSTIC")
print("NO DATASET MODIFICATION")
print("NO CEM")
print("NO PPO")
print("=" * 190)

print(
    "MuJoCo:",
    mujoco.__version__,
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
    "floor geom:",
    env.floor_geom,
)

print(
    "base floor Z:",
    f"{BASE_FLOOR_Z:.6f}",
)

print(
    "floor margin:",
    f"{model.geom_margin[env.floor_geom]:.6f}",
)

print(
    "floor gap:",
    f"{model.geom_gap[env.floor_geom]:.6f}",
)


print()
print("SOLE MARGIN / GAP")

for gid in (
    env.left_sole_geoms
    +
    env.right_sole_geoms
):

    print(
        f"geom={gid:02d} "
        f"margin={model.geom_margin[gid]:.6f} "
        f"gap={model.geom_gap[gid]:.6f}"
    )


# ================================================================
# CLEARANCE
# ================================================================

def geom_clearance(
    d,
    gid,
    floor_z,
):

    return float(
        d.geom_xpos[
            gid,
            2,
        ]
        -
        model.geom_size[
            gid,
            0,
        ]
        -
        floor_z
    )


def foot_clearance(
    d,
    geoms,
    floor_z,
):

    return min(
        geom_clearance(
            d,
            gid,
            floor_z,
        )
        for gid in geoms
    )


# ================================================================
# REFERENCE SUPPORT-CLEARANCE AUDIT
# ================================================================

print()
print("=" * 190)
print("REFERENCE SUPPORT-FOOT CLEARANCE")
print("ORIGINAL FLOOR")
print("=" * 190)


left_support_clearance = []
right_support_clearance = []

left_contact_clearance = []
right_contact_clearance = []


for frame in range(
    env.num_frames
):

    data.qpos[:] = (
        env.ref_full_qpos[
            frame
        ]
    )

    data.qvel[:] = (
        env.ref_full_qvel[
            frame
        ]
    )


    mujoco.mj_forward(
        model,
        data,
    )


    lc = foot_clearance(
        data,
        env.left_sole_geoms,
        BASE_FLOOR_Z,
    )


    rc = foot_clearance(
        data,
        env.right_sole_geoms,
        BASE_FLOOR_Z,
    )


    if (
        env.ref_support[
            frame,
            0,
        ]
        > 0.5
    ):
        left_support_clearance.append(
            lc
        )


    if (
        env.ref_support[
            frame,
            1,
        ]
        > 0.5
    ):
        right_support_clearance.append(
            rc
        )


    if (
        env.ref_contact[
            frame,
            0,
        ]
        > 0.5
    ):
        left_contact_clearance.append(
            lc
        )


    if (
        env.ref_contact[
            frame,
            1,
        ]
        > 0.5
    ):
        right_contact_clearance.append(
            rc
        )


def report_clearance(
    name,
    values,
):

    x = (
        1000.0
        * np.asarray(
            values,
            dtype=np.float64,
        )
    )


    print(
        f"{name:28s} "
        f"N={len(x):3d} "
        f"min={np.min(x):6.2f}mm "
        f"p50={np.percentile(x,50):6.2f}mm "
        f"p90={np.percentile(x,90):6.2f}mm "
        f"p95={np.percentile(x,95):6.2f}mm "
        f"max={np.max(x):6.2f}mm"
    )


report_clearance(
    "LEFT stable support",
    left_support_clearance,
)

report_clearance(
    "RIGHT stable support",
    right_support_clearance,
)

report_clearance(
    "LEFT contact label",
    left_contact_clearance,
)

report_clearance(
    "RIGHT contact label",
    right_contact_clearance,
)


# ================================================================
# ACTIVE SOLE CONTACTS
# ================================================================

def active_sole_contacts():

    left = False
    right = False

    active_count = 0


    for cid in range(
        data.ncon
    ):

        con = data.contact[
            cid
        ]


        # Active force-generating constraint only.
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
            if g1 == env.floor_geom
            else g1
        )


        if other in LEFT_SOLES:

            left = True
            active_count += 1


        elif other in RIGHT_SOLES:

            right = True
            active_count += 1


    return (
        left,
        right,
        active_count,
    )


# ================================================================
# ZERO-RESIDUAL ROLLOUT
# ================================================================

ZERO = np.zeros(
    29,
    dtype=np.float32,
)


def rollout(
    offset_mm,
    start,
):

    floor_z = (
        BASE_FLOOR_Z
        +
        offset_mm
        / 1000.0
    )


    model.geom_pos[
        env.floor_geom,
        2,
    ] = floor_z


    env.floor_z = floor_z


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


    # Ensure geometry/contact state reflects moved floor.
    mujoco.mj_forward(
        model,
        data,
    )


    (
        initial_left,
        initial_right,
        initial_count,
    ) = active_sole_contacts()


    max_steps = min(
        HORIZON,
        env.num_frames
        - 1
        - start,
    )


    steps = 0

    contact_steps = 0
    double_contact_steps = 0

    left_contact_steps = 0
    right_contact_steps = 0


    orientation_sum = 0.0

    min_up = env._up_z()

    q_error_sum = 0.0


    reason = ""


    while steps < max_steps:

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            ZERO
        )


        steps += 1


        (
            left,
            right,
            count,
        ) = active_sole_contacts()


        if left or right:
            contact_steps += 1

        if left and right:
            double_contact_steps += 1

        if left:
            left_contact_steps += 1

        if right:
            right_contact_steps += 1


        terms = info[
            "reward_terms"
        ]


        orientation_sum += float(
            terms[
                "root_orientation_deg"
            ]
        )


        q_error_sum += float(
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


        if terminated:

            reason = info[
                "termination_reason"
            ]

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


    return {

        "offset":
            offset_mm,

        "start":
            start,

        "steps":
            steps,

        "max_steps":
            max_steps,

        "progress":
            steps
            / max_steps,

        "initial_contact":
            int(
                initial_left
                or initial_right
            ),

        "initial_left":
            int(
                initial_left
            ),

        "initial_right":
            int(
                initial_right
            ),

        "contact_fraction":
            contact_steps
            / divisor,

        "double_fraction":
            double_contact_steps
            / divisor,

        "left_fraction":
            left_contact_steps
            / divisor,

        "right_fraction":
            right_contact_steps
            / divisor,

        "orientation":
            orientation_sum
            / divisor,

        "q_error":
            q_error_sum
            / divisor,

        "min_up":
            min_up,

        "reason":
            reason,
    }


# ================================================================
# RUN SWEEP
# ================================================================

results = []


print()
print("=" * 190)
print("GROUND OFFSET SWEEP")
print("=" * 190)


for offset in OFFSETS_MM:

    for start in STARTS:

        r = rollout(
            offset,
            start,
        )

        results.append(
            r
        )


# Restore original floor after experiment.
model.geom_pos[
    env.floor_geom,
    2,
] = BASE_FLOOR_Z

env.floor_z = (
    BASE_FLOOR_Z
)


# ================================================================
# DETAIL
# ================================================================

print()
print("=" * 190)
print("PER-START RESULTS")
print("=" * 190)

print(
    f"{'OFF':>6s} "
    f"{'START':>5s} "
    f"{'STEPS':>8s} "
    f"{'PROG':>6s} "
    f"{'INIT':>5s} "
    f"{'CONT':>6s} "
    f"{'DOUBLE':>7s} "
    f"{'ORI':>7s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'REASON':>14s}"
)

print("-" * 190)


for r in results:

    print(
        f"{r['offset']:6.1f} "
        f"{r['start']:5d} "
        f"{r['steps']:3d}/"
        f"{r['max_steps']:<3d} "
        f"{r['progress']:6.3f} "
        f"{r['initial_contact']:5d} "
        f"{r['contact_fraction']:6.3f} "
        f"{r['double_fraction']:7.3f} "
        f"{r['orientation']:7.2f} "
        f"{r['min_up']:7.3f} "
        f"{r['q_error']:7.3f} "
        f"{r['reason']:>14s}"
    )


# ================================================================
# AGGREGATE
# ================================================================

def aggregate(
    offset,
):

    rows = [
        r
        for r in results
        if abs(
            r[
                "offset"
            ]
            - offset
        ) < 1e-9
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

        "progress":
            mean(
                "progress"
            ),

        "steps":
            mean(
                "steps"
            ),

        "initial_contact":
            mean(
                "initial_contact"
            ),

        "contact":
            mean(
                "contact_fraction"
            ),

        "double":
            mean(
                "double_fraction"
            ),

        "orientation":
            mean(
                "orientation"
            ),

        "min_up":
            mean(
                "min_up"
            ),

        "q_error":
            mean(
                "q_error"
            ),

        "min_progress":
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
    }


summary = {
    offset:
        aggregate(
            offset
        )
    for offset in OFFSETS_MM
}


print()
print("=" * 175)
print("GROUND CONTACT OFFSET SUMMARY")
print("=" * 175)

print(
    f"{'OFF-MM':>7s} "
    f"{'STEPS':>7s} "
    f"{'PROG':>7s} "
    f"{'MIN-P':>7s} "
    f"{'INIT-C':>7s} "
    f"{'CONTACT':>8s} "
    f"{'DOUBLE':>8s} "
    f"{'ORI':>8s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s}"
)

print("-" * 175)


for offset in OFFSETS_MM:

    s = summary[
        offset
    ]


    print(
        f"{offset:7.1f} "
        f"{s['steps']:7.1f} "
        f"{s['progress']:7.3f} "
        f"{s['min_progress']:7.3f} "
        f"{s['initial_contact']:7.3f} "
        f"{s['contact']:8.3f} "
        f"{s['double']:8.3f} "
        f"{s['orientation']:8.2f} "
        f"{s['min_up']:7.3f} "
        f"{s['q_error']:7.3f}"
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


baseline_consistent = True


for start, expected_steps in expected.items():

    row = next(
        r
        for r in results

        if (
            abs(
                r["offset"]
            )
            < 1e-9

            and
            r["start"]
            == start
        )
    )


    if row[
        "steps"
    ] != expected_steps:

        baseline_consistent = False


print()
print(
    "baseline exact 102/51/58:",
    baseline_consistent,
)


# ================================================================
# DECISION
# ================================================================

base = summary[
    0.0
]


best_offset = max(
    OFFSETS_MM,

    key=lambda x: (
        summary[x][
            "progress"
        ],

        summary[x][
            "min_progress"
        ],

        -summary[x][
            "orientation"
        ],

        -x,
    ),
)


best = summary[
    best_offset
]


gain = (
    best[
        "progress"
    ]
    -
    base[
        "progress"
    ]
)


orientation_change = (
    best[
        "orientation"
    ]
    -
    base[
        "orientation"
    ]
)


print()
print("=" * 175)
print("GROUND-CONTACT GAP DECISION")
print("=" * 175)

print(
    "baseline progress:",
    f"{base['progress']:.3f}",
)

print(
    "best floor offset:",
    f"{best_offset:.1f} mm",
)

print(
    "best progress:",
    f"{best['progress']:.3f}",
)

print(
    "progress gain:",
    f"{gain:+.3f}",
)

print(
    "initial-contact rate:",
    f"{100*base['initial_contact']:.1f}%",
    "->",
    f"{100*best['initial_contact']:.1f}%",
)

print(
    "dynamic contact coverage:",
    f"{100*base['contact']:.1f}%",
    "->",
    f"{100*best['contact']:.1f}%",
)

print(
    "orientation change:",
    f"{orientation_change:+.2f} deg",
)


print()


if (
    best_offset > 0.0

    and gain >= 0.08

    and orientation_change <= 5.0
):

    print(
        "RESULT: SUPPORT-CONTACT GAP IS A MAJOR BOTTLENECK"
    )

    print(
        "Closing the reference-to-floor gap materially "
        "improves the actual MuJoCo dynamics."
    )

    print(
        "NEXT: build a support-aware contact-compatible "
        "grounded reference V2."
    )

    print(
        "Do NOT simply keep the floor raised permanently."
    )


elif (
    best_offset > 0.0

    and gain >= 0.04
):

    print(
        "RESULT: SUPPORT-CONTACT GAP HAS A MARGINAL EFFECT"
    )

    print(
        "Contact-compatible grounding helps, but is "
        "not sufficient by itself."
    )


else:

    print(
        "RESULT: CONTACT GAP IS NOT THE PRIMARY FAILURE"
    )

    print(
        "The 3–15 mm support clearance is not enough "
        "to explain the dynamic instability."
    )

    print(
        "NEXT: redo inverse dynamics with explicitly "
        "contact-snapped support states."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_ground_contact_offset.npz"
)


np.savez(
    output,

    offsets_mm=
        np.asarray(
            OFFSETS_MM,
            dtype=np.float32,
        ),

    starts=
        np.asarray(
            STARTS,
            dtype=np.int32,
        ),

    best_offset_mm=
        np.asarray(
            [
                best_offset
            ],
            dtype=np.float32,
        ),

    progress_gain=
        np.asarray(
            [
                gain
            ],
            dtype=np.float32,
        ),

    baseline_consistent=
        np.asarray(
            [
                baseline_consistent
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

print("=" * 175)


env.close()
