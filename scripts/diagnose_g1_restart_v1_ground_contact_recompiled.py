from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


SCENE = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


if not SCENE.exists():
    raise FileNotFoundError(
        SCENE
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
# GET BASE MODEL INFORMATION
# ================================================================

probe = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


BASE_MODEL = probe.model


FLOOR_NAME = mujoco.mj_id2name(
    BASE_MODEL,
    mujoco.mjtObj.mjOBJ_GEOM,
    probe.floor_geom,
)


if FLOOR_NAME is None:
    raise RuntimeError(
        "Floor geom has no name."
    )


BASE_FLOOR_Z = float(
    BASE_MODEL.geom_pos[
        probe.floor_geom,
        2,
    ]
)


EXPECTED_TOPOLOGY = (
    BASE_MODEL.nq,
    BASE_MODEL.nv,
    BASE_MODEL.nu,
    BASE_MODEL.nbody,
    BASE_MODEL.ngeom,
    BASE_MODEL.nsite,
)


print("=" * 180)
print("CORRECTED RECOMPILED GROUND-CONTACT GAP TEST")
print("FRESH MjModel PER FLOOR OFFSET")
print("NO CEM")
print("NO PPO")
print("=" * 180)

print(
    "MuJoCo version:",
    mujoco.__version__,
)

print(
    "scene:",
    SCENE,
)

print(
    "floor name:",
    FLOOR_NAME,
)

print(
    "original floor Z:",
    f"{BASE_FLOOR_Z:.6f}",
)

print(
    "topology:",
    EXPECTED_TOPOLOGY,
)


probe.close()


# ================================================================
# COMPILE MODIFIED MODEL
# ================================================================

def compile_model(
    offset_mm,
):

    spec = mujoco.MjSpec.from_file(
        str(
            SCENE
        )
    )


    # Avoid depending on named-access support:
    # find floor through the documented spec.geoms list.
    floor_spec = None


    for geom in spec.geoms:

        if geom.name == FLOOR_NAME:

            floor_spec = geom
            break


    if floor_spec is None:

        available = [
            g.name
            for g in spec.geoms
            if g.name
        ]

        raise RuntimeError(
            "Could not find floor geom "
            f"{FLOOR_NAME!r} in MjSpec. "
            f"Available named geoms include: "
            f"{available[:20]}"
        )


    original_spec_z = float(
        floor_spec.pos[
            2
        ]
    )


    requested_z = (
        original_spec_z
        +
        float(
            offset_mm
        )
        / 1000.0
    )


    floor_spec.pos[
        2
    ] = requested_z


    model = spec.compile()


    topology = (
        model.nq,
        model.nv,
        model.nu,
        model.nbody,
        model.ngeom,
        model.nsite,
    )


    if topology != EXPECTED_TOPOLOGY:

        raise RuntimeError(
            "Recompiled model topology changed.\n"
            f"expected={EXPECTED_TOPOLOGY}\n"
            f"actual={topology}"
        )


    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FLOOR_NAME,
    )


    if floor_id < 0:

        raise RuntimeError(
            "Floor missing after recompilation."
        )


    compiled_z = float(
        model.geom_pos[
            floor_id,
            2,
        ]
    )


    error = abs(
        compiled_z
        -
        requested_z
    )


    if error > 1e-8:

        raise RuntimeError(
            "Compiled floor position does not "
            "match requested position.\n"
            f"requested={requested_z:.9f}\n"
            f"compiled={compiled_z:.9f}"
        )


    return (
        model,
        floor_id,
        original_spec_z,
        compiled_z,
    )


# ================================================================
# ENVIRONMENT WITH RECOMPILED MODEL
# ================================================================

def make_env(
    offset_mm,
):

    env = G129DofTrackingRestartV1(
        rsi=False,
        reset_joint_noise=0.0,
        reset_velocity_noise=0.0,
    )


    (
        model,
        floor_id,
        original_spec_z,
        compiled_z,
    ) = compile_model(
        offset_mm
    )


    # ------------------------------------------------------------
    # Topology is identical; replace physics model/data only.
    # ------------------------------------------------------------

    env.model = model

    env.data = mujoco.MjData(
        model
    )


    env.floor_geom = (
        floor_id
    )


    env.floor_z = (
        compiled_z
    )


    # ------------------------------------------------------------
    # Verify all important cached IDs still identify
    # the same objects after recompilation.
    # ------------------------------------------------------------

    for j, name in enumerate(
        env.ref_names
    ):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )


        if jid != env.joint_ids[
            j
        ]:

            raise RuntimeError(
                f"Joint ID changed for {name}: "
                f"{env.joint_ids[j]} -> {jid}"
            )


    return (
        env,
        compiled_z,
    )


# ================================================================
# ACTIVE CONTACT DETECTOR
# ================================================================

def active_sole_contacts(
    env,
):

    left_set = set(
        env.left_sole_geoms
    )

    right_set = set(
        env.right_sole_geoms
    )


    left = False
    right = False

    active_contacts = 0


    for cid in range(
        env.data.ncon
    ):

        contact = env.data.contact[
            cid
        ]


        # Only a force-generating active constraint.
        if int(
            contact.efc_address
        ) < 0:

            continue


        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
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


        if other in left_set:

            left = True
            active_contacts += 1


        elif other in right_set:

            right = True
            active_contacts += 1


    return (
        left,
        right,
        active_contacts,
    )


# ================================================================
# MINIMUM SOLE CLEARANCE
# ================================================================

def sole_clearance(
    env,
    geoms,
):

    floor_z = float(
        env.model.geom_pos[
            env.floor_geom,
            2,
        ]
    )


    values = []


    for gid in geoms:

        # These are sphere sole geoms:
        # size[0] = radius.
        bottom = (
            env.data.geom_xpos[
                gid,
                2,
            ]
            -
            env.model.geom_size[
                gid,
                0,
            ]
        )


        values.append(
            float(
                bottom
                - floor_z
            )
        )


    return min(
        values
    )


# ================================================================
# ZERO-ACTION ROLLOUT
# ================================================================

ZERO = np.zeros(
    29,
    dtype=np.float32,
)


def rollout(
    env,
    start,
):

    (
        obs,
        info,
    ) = env.reset(
        options={
            "start_frame":
                int(
                    start
                ),
        }
    )


    # Generate contacts for exact reset configuration.
    mujoco.mj_forward(
        env.model,
        env.data,
    )


    (
        initial_left,
        initial_right,
        initial_n,
    ) = active_sole_contacts(
        env
    )


    initial_left_clearance = (
        sole_clearance(
            env,
            env.left_sole_geoms,
        )
    )


    initial_right_clearance = (
        sole_clearance(
            env,
            env.right_sole_geoms,
        )
    )


    maximum_steps = min(
        HORIZON,
        env.num_frames
        - 1
        - start,
    )


    steps = 0

    contact_steps = 0
    double_steps = 0

    first_contact_step = None


    ori_sum = 0.0
    q_sum = 0.0

    min_up = env._up_z()

    reason = ""


    while steps < maximum_steps:

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
            nactive,
        ) = active_sole_contacts(
            env
        )


        if left or right:

            contact_steps += 1


            if first_contact_step is None:

                first_contact_step = (
                    steps
                )


        if left and right:

            double_steps += 1


        terms = info[
            "reward_terms"
        ]


        ori_sum += float(
            terms[
                "root_orientation_deg"
            ]
        )


        q_sum += float(
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

        "start":
            start,

        "steps":
            steps,

        "max_steps":
            maximum_steps,

        "progress":
            steps
            / maximum_steps,

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

        "initial_active_count":
            initial_n,

        "left_clearance":
            initial_left_clearance,

        "right_clearance":
            initial_right_clearance,

        "contact_fraction":
            contact_steps
            / divisor,

        "double_fraction":
            double_steps
            / divisor,

        "first_contact_step":
            (
                first_contact_step
                if first_contact_step is not None
                else -1
            ),

        "orientation":
            ori_sum
            / divisor,

        "q_error":
            q_sum
            / divisor,

        "min_up":
            min_up,

        "reason":
            reason,
    }


# ================================================================
# RUN
# ================================================================

results = []


print()
print("=" * 190)
print("MODEL RECOMPILATION AUDIT")
print("=" * 190)


for offset in OFFSETS_MM:

    env, compiled_floor_z = (
        make_env(
            offset
        )
    )


    actual_offset = (
        1000.0
        * (
            compiled_floor_z
            -
            BASE_FLOOR_Z
        )
    )


    print(
        f"requested={offset:5.1f} mm "
        f"compiledFloorZ="
        f"{compiled_floor_z:+.6f} m "
        f"actualOffset="
        f"{actual_offset:+6.2f} mm"
    )


    for start in STARTS:

        r = rollout(
            env,
            start,
        )


        r[
            "offset"
        ] = offset


        results.append(
            r
        )


    env.close()


# ================================================================
# DETAILS
# ================================================================

print()
print("=" * 210)
print("RECOMPILED FLOOR — PER-START RESULTS")
print("=" * 210)

print(
    f"{'OFF':>5s} "
    f"{'ST':>4s} "
    f"{'STEPS':>8s} "
    f"{'PROG':>6s} "
    f"{'INIT':>4s} "
    f"{'L-C':>7s} "
    f"{'R-C':>7s} "
    f"{'FIRST':>5s} "
    f"{'CONT':>6s} "
    f"{'DBL':>6s} "
    f"{'ORI':>7s} "
    f"{'UP':>6s} "
    f"{'QERR':>6s} "
    f"{'REASON':>14s}"
)

print("-" * 210)


for r in results:

    print(
        f"{r['offset']:5.1f} "
        f"{r['start']:4d} "
        f"{r['steps']:3d}/"
        f"{r['max_steps']:<3d} "
        f"{r['progress']:6.3f} "
        f"{r['initial_contact']:4d} "
        f"{1000*r['left_clearance']:+7.2f} "
        f"{1000*r['right_clearance']:+7.2f} "
        f"{r['first_contact_step']:5d} "
        f"{r['contact_fraction']:6.3f} "
        f"{r['double_fraction']:6.3f} "
        f"{r['orientation']:7.2f} "
        f"{r['min_up']:6.3f} "
        f"{r['q_error']:6.3f} "
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
            r["offset"]
            - offset
        ) < 1e-9
    ]


    def mean(
        key,
    ):

        return float(
            np.mean(
                [
                    r[key]
                    for r in rows
                ]
            )
        )


    positive_first = [
        r[
            "first_contact_step"
        ]
        for r in rows
        if r[
            "first_contact_step"
        ] >= 0
    ]


    return {

        "steps":
            mean(
                "steps"
            ),

        "progress":
            mean(
                "progress"
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

        "initial":
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

        "q":
            mean(
                "q_error"
            ),

        "first":
            (
                float(
                    np.mean(
                        positive_first
                    )
                )
                if positive_first
                else float("nan")
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
print("=" * 180)
print("RECOMPILED GROUND-CONTACT SUMMARY")
print("=" * 180)

print(
    f"{'OFF':>6s} "
    f"{'STEPS':>7s} "
    f"{'PROG':>7s} "
    f"{'MIN-P':>7s} "
    f"{'INIT':>7s} "
    f"{'CONT':>7s} "
    f"{'DBL':>7s} "
    f"{'FIRST':>7s} "
    f"{'ORI':>8s} "
    f"{'UP':>7s} "
    f"{'QERR':>7s}"
)

print("-" * 180)


for offset in OFFSETS_MM:

    s = summary[
        offset
    ]


    print(
        f"{offset:6.1f} "
        f"{s['steps']:7.1f} "
        f"{s['progress']:7.3f} "
        f"{s['min_progress']:7.3f} "
        f"{s['initial']:7.3f} "
        f"{s['contact']:7.3f} "
        f"{s['double']:7.3f} "
        f"{s['first']:7.2f} "
        f"{s['orientation']:8.2f} "
        f"{s['min_up']:7.3f} "
        f"{s['q']:7.3f}"
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

    r = next(
        x
        for x in results
        if (
            abs(
                x["offset"]
            ) < 1e-9

            and x[
                "start"
            ] == start
        )
    )


    if r[
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

    key=lambda offset: (
        summary[
            offset
        ][
            "progress"
        ],

        summary[
            offset
        ][
            "min_progress"
        ],

        -summary[
            offset
        ][
            "orientation"
        ],

        -offset,
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


ori_change = (
    best[
        "orientation"
    ]
    -
    base[
        "orientation"
    ]
)


print()
print("=" * 180)
print("CORRECTED GROUND-CONTACT DECISION")
print("=" * 180)

print(
    "baseline progress:",
    f"{base['progress']:.3f}",
)

print(
    "best offset:",
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
    "initial contact:",
    f"{100*base['initial']:.1f}%",
    "->",
    f"{100*best['initial']:.1f}%",
)

print(
    "dynamic contact:",
    f"{100*base['contact']:.1f}%",
    "->",
    f"{100*best['contact']:.1f}%",
)

print(
    "orientation change:",
    f"{ori_change:+.2f} deg",
)


print()


if (
    best_offset
    <= 7.0

    and best_offset
    > 0.0

    and gain
    >= 0.08

    and ori_change
    <= 5.0
):

    print(
        "RESULT: SMALL SUPPORT GAP IS A MAJOR BOTTLENECK"
    )

    print(
        "A physically small contact-compatible "
        "grounding correction materially improves "
        "the G1 dynamics."
    )

    print(
        "NEXT: rebuild the reference with "
        "support-compatible grounding."
    )


elif (
    best_offset
    > 7.0

    and gain
    >= 0.08
):

    print(
        "RESULT: LARGE FLOOR OFFSET HELPS, "
        "BUT IS NOT A CLEAN GROUNDING FIX"
    )

    print(
        "The improvement requires substantial "
        "penetration/contact alteration."
    )

    print(
        "NEXT: inspect support labels and "
        "foot-contact geometry individually."
    )


elif (
    best_offset
    > 0.0

    and gain
    >= 0.04
):

    print(
        "RESULT: CONTACT GAP HAS A MARGINAL EFFECT"
    )

    print(
        "Ground contact timing contributes, "
        "but is not the primary bottleneck."
    )


else:

    print(
        "RESULT: INITIAL SUPPORT GAP IS NOT "
        "THE PRIMARY BOTTLENECK"
    )

    print(
        "The corrected recompiled test does not "
        "show a material survival improvement."
    )

    print(
        "NEXT: perform contact-snapped "
        "inverse-dynamics feasibility analysis."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_ground_contact_recompiled.npz"
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

    gain=
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

print("=" * 180)
