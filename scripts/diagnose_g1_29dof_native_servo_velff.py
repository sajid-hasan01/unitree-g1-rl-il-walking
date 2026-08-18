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


STARTS = [
    0,
    50,
    100,
    150,
    200,
]


ALPHAS = [
    0.00,
    0.25,
    0.50,
    1.00,
]


CONTROL_HZ = 50.0

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
        "Expected 29 joints."
    )


sim_dt = float(
    model.opt.timestep
)


frame_skip = int(
    round(
        1.0
        / (
            CONTROL_HZ
            * sim_dt
        )
    )
)


# ================================================================
# ADDRESSES + COMPILED SERVO PARAMETERS
# ================================================================

joint_ids = []
qaddrs = []
vaddrs = []
aids = []

kp = []
kd = []

effort_limits = []


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
            f"Missing joint/actuator: {name}"
        )


    joint_ids.append(
        jid
    )

    aids.append(
        aid
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


    gain = float(
        model.actuator_gainprm[
            aid,
            0,
        ]
    )


    b0 = float(
        model.actuator_biasprm[
            aid,
            0,
        ]
    )

    b1 = float(
        model.actuator_biasprm[
            aid,
            1,
        ]
    )

    b2 = float(
        model.actuator_biasprm[
            aid,
            2,
        ]
    )


    # ------------------------------------------------------------
    # Must have:
    #
    # p = Kp*u - Kp*q - Kd*qdot
    # ------------------------------------------------------------

    if abs(
        b0
    ) > 1e-8:

        raise RuntimeError(
            f"Unexpected actuator b0 for {name}: {b0}"
        )


    if abs(
        b1 + gain
    ) > 1e-5:

        raise RuntimeError(
            f"Actuator is not expected position servo: "
            f"{name}, gain={gain}, b1={b1}"
        )


    if b2 > 1e-8:

        raise RuntimeError(
            f"Unexpected positive velocity bias: "
            f"{name}, b2={b2}"
        )


    kp.append(
        gain
    )

    kd.append(
        -b2
    )


    if hasattr(
        model,
        "jnt_actfrcrange"
    ):

        lo = float(
            model.jnt_actfrcrange[
                jid,
                0,
            ]
        )

        hi = float(
            model.jnt_actfrcrange[
                jid,
                1,
            ]
        )

        limit = max(
            abs(lo),
            abs(hi),
        )

    else:

        raise RuntimeError(
            "Model does not expose jnt_actfrcrange."
        )


    effort_limits.append(
        limit
    )


kp = np.asarray(
    kp,
    dtype=np.float64,
)

kd = np.asarray(
    kd,
    dtype=np.float64,
)

effort_limits = np.asarray(
    effort_limits,
    dtype=np.float64,
)


# ================================================================
# SERVO AUDIT
# ================================================================

print("=" * 160)
print("G1 NATIVE POSITION SERVO + VELOCITY FEEDFORWARD")
print("NO PPO")
print("=" * 160)


print(
    "frames:",
    len(ref_q),
)

print(
    "control Hz:",
    CONTROL_HZ,
)

print(
    "sim dt:",
    sim_dt,
)

print(
    "frame skip:",
    frame_skip,
)


print()
print("=" * 160)
print("COMPILED NATIVE SERVO")
print("=" * 160)

print(
    f"{'#':>2s} "
    f"{'JOINT':28s} "
    f"{'KP':>8s} "
    f"{'KD':>9s} "
    f"{'KD/KP':>10s} "
    f"{'EFFORT':>8s}"
)

print("-" * 160)


for j in range(29):

    print(
        f"{j:2d} "
        f"{names[j]:28s} "
        f"{kp[j]:8.2f} "
        f"{kd[j]:9.3f} "
        f"{kd[j]/kp[j]:10.5f} "
        f"{effort_limits[j]:8.1f}"
    )


# ================================================================
# HELPERS
# ================================================================

pelvis_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pelvis",
)


def joint_q(
    data,
):

    return np.asarray(
        [
            data.qpos[qadr]
            for qadr in qaddrs
        ],
        dtype=np.float64,
    )


def joint_qd(
    data,
):

    return np.asarray(
        [
            data.qvel[vadr]
            for vadr in vaddrs
        ],
        dtype=np.float64,
    )


def up_z(
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
    qa,
    qb,
):

    qa = np.asarray(
        qa,
        dtype=np.float64,
    )

    qb = np.asarray(
        qb,
        dtype=np.float64,
    )


    qa /= max(
        np.linalg.norm(qa),
        1e-12,
    )

    qb /= max(
        np.linalg.norm(qb),
        1e-12,
    )


    dot = abs(
        float(
            np.dot(
                qa,
                qb,
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


def physical_failure(
    data,
):

    return (
        float(
            data.qpos[2]
        ) < FAIL_HEIGHT

        or up_z(
            data
        ) < FAIL_UP
    )


# ================================================================
# COMMAND
#
# Native actuator:
#
# tau = kp*(ctrl-q) - kd*qdot
#
# Choose:
#
# ctrl = qref + alpha*(kd/kp)*qdref
#
# alpha=1 gives:
#
# tau =
# kp*(qref-q)
# + kd*(qdref-qdot)
# ================================================================

def command(
    data,
    ref_index,
    alpha,
):

    desired = (
        ref_q[
            ref_index
        ]
        +
        alpha
        * (
            kd / kp
        )
        * ref_qd[
            ref_index
        ]
    )


    clipped = np.zeros(
        29,
        dtype=np.bool_,
    )


    for j, aid in enumerate(
        aids
    ):

        target = float(
            desired[j]
        )


        if model.actuator_ctrllimited[
            aid
        ]:

            lo, hi = (
                model.actuator_ctrlrange[
                    aid
                ]
            )


            before = target


            target = float(
                np.clip(
                    target,
                    lo,
                    hi,
                )
            )


            if abs(
                target - before
            ) > 1e-10:

                clipped[j] = True


        data.ctrl[
            aid
        ] = target


    q = joint_q(
        data
    )

    qd = joint_qd(
        data
    )


    # Requested actuator force according to compiled affine law.
    requested_force = (
        kp
        * (
            np.asarray(
                [
                    data.ctrl[aid]
                    for aid in aids
                ],
                dtype=np.float64,
            )
            - q
        )
        -
        kd
        * qd
    )


    requested_saturated = (
        np.abs(
            requested_force
        )
        > effort_limits
    )


    shift = (
        np.asarray(
            [
                data.ctrl[aid]
                for aid in aids
            ],
            dtype=np.float64,
        )
        - ref_q[
            ref_index
        ]
    )


    return (
        requested_saturated,
        clipped,
        shift,
    )


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    alpha,
    start,
):

    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    data.qpos[:] = (
        ref_full_qpos[
            start
        ]
    )

    data.qvel[:] = (
        ref_full_qvel[
            start
        ]
    )


    mujoco.mj_forward(
        model,
        data,
    )


    initial_x = float(
        data.qpos[0]
    )

    initial_y = float(
        data.qpos[1]
    )


    current_ref = int(
        start
    )

    steps = 0

    min_up = up_z(
        data
    )

    max_y = 0.0

    qerr2 = 0.0
    qderr2 = 0.0
    orierr2 = 0.0
    rooterr2 = 0.0

    requested_sat_count = 0
    requested_samples = 0

    ctrl_clip_count = 0
    ctrl_samples = 0

    shift2 = 0.0
    shift_samples = 0

    max_shift = 0.0

    max_actual_force_ratio = 0.0


    while True:

        next_ref = min(
            current_ref + 1,
            len(ref_q) - 1,
        )


        for _ in range(
            frame_skip
        ):

            (
                requested_sat,
                ctrl_clipped,
                shift,
            ) = command(
                data,
                next_ref,
                alpha,
            )


            requested_sat_count += int(
                np.sum(
                    requested_sat
                )
            )

            requested_samples += 29


            ctrl_clip_count += int(
                np.sum(
                    ctrl_clipped
                )
            )

            ctrl_samples += 29


            shift2 += float(
                np.sum(
                    shift ** 2
                )
            )

            shift_samples += 29


            max_shift = max(
                max_shift,
                float(
                    np.max(
                        np.abs(
                            shift
                        )
                    )
                ),
            )


            mujoco.mj_step(
                model,
                data,
            )


            actual_ratio = []


            for j, vadr in enumerate(
                vaddrs
            ):

                actual_ratio.append(
                    abs(
                        float(
                            data.qfrc_actuator[
                                vadr
                            ]
                        )
                    )
                    / effort_limits[j]
                )


            max_actual_force_ratio = max(
                max_actual_force_ratio,
                max(
                    actual_ratio
                ),
            )


        current_ref = (
            next_ref
        )

        steps += 1


        q = joint_q(
            data
        )

        qd = joint_qd(
            data
        )


        qerr = float(
            np.sqrt(
                np.mean(
                    (
                        q
                        - ref_q[
                            current_ref
                        ]
                    ) ** 2
                )
            )
        )


        qderr = float(
            np.sqrt(
                np.mean(
                    (
                        qd
                        - ref_qd[
                            current_ref
                        ]
                    ) ** 2
                )
            )
        )


        ori = quat_error_deg(
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


        up = up_z(
            data
        )


        min_up = min(
            min_up,
            up,
        )


        max_y = max(
            max_y,

            abs(
                float(
                    data.qpos[
                        1
                    ]
                    - initial_y
                )
            ),
        )


        qerr2 += qerr ** 2
        qderr2 += qderr ** 2
        orierr2 += ori ** 2
        rooterr2 += root_error ** 2


        if (
            steps == 1
            or steps % 25 == 0
        ):

            print(
                f"alpha={alpha:.2f} "
                f"start={start:03d} "
                f"step={steps:03d} "
                f"ref={current_ref:03d} "
                f"x={data.qpos[0]-initial_x:+.3f} "
                f"vx={data.qvel[0]:+.3f} "
                f"y={data.qpos[1]-initial_y:+.3f} "
                f"z={data.qpos[2]:.3f} "
                f"up={up:.3f} "
                f"q={qerr:.3f} "
                f"qd={qderr:.3f} "
                f"ori={ori:.1f}"
            )


        if physical_failure(
            data
        ):

            completed = False
            break


        if current_ref >= (
            len(ref_q) - 1
        ):

            completed = True
            break


    return {

        "alpha":
            alpha,

        "start":
            start,

        "steps":
            steps,

        "done":
            completed,

        "progress":
            min(
                1.0,
                steps
                / max(
                    len(ref_q)
                    - 1
                    - start,
                    1,
                ),
            ),

        "dx":
            float(
                data.qpos[0]
                - initial_x
            ),

        "max_y":
            max_y,

        "min_up":
            min_up,

        "qerr":
            math.sqrt(
                qerr2
                / max(
                    steps,
                    1,
                )
            ),

        "qderr":
            math.sqrt(
                qderr2
                / max(
                    steps,
                    1,
                )
            ),

        "ori":
            math.sqrt(
                orierr2
                / max(
                    steps,
                    1,
                )
            ),

        "root":
            math.sqrt(
                rooterr2
                / max(
                    steps,
                    1,
                )
            ),

        "request_sat":
            (
                requested_sat_count
                / requested_samples
                if requested_samples
                else 0.0
            ),

        "ctrl_clip":
            (
                ctrl_clip_count
                / ctrl_samples
                if ctrl_samples
                else 0.0
            ),

        "shift_rms":
            math.sqrt(
                shift2
                / max(
                    shift_samples,
                    1,
                )
            ),

        "shift_max":
            max_shift,

        "force_ratio":
            max_actual_force_ratio,
    }


# ================================================================
# RUN
# ================================================================

results = []


print()
print("=" * 160)
print("VELOCITY FEEDFORWARD SWEEP")
print("=" * 160)


for alpha in ALPHAS:

    for start in STARTS:

        results.append(
            rollout(
                alpha,
                start,
            )
        )


# ================================================================
# DETAIL
# ================================================================

print()
print("=" * 175)
print("VELOCITY FEEDFORWARD DETAIL")
print("=" * 175)

print(
    f"{'ALPHA':>5s} "
    f"{'START':>5s} "
    f"{'STEP':>5s} "
    f"{'DONE':>5s} "
    f"{'PROG':>6s} "
    f"{'DX':>8s} "
    f"{'MAXY':>7s} "
    f"{'MINUP':>7s} "
    f"{'QERR':>7s} "
    f"{'QDERR':>7s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'REQSAT':>7s} "
    f"{'CLIP':>7s} "
    f"{'SHIFTR':>7s} "
    f"{'SHIFTMAX':>8s} "
    f"{'FMAX':>6s}"
)

print("-" * 175)


for r in results:

    print(
        f"{r['alpha']:5.2f} "
        f"{r['start']:5d} "
        f"{r['steps']:5d} "
        f"{str(r['done']):>5s} "
        f"{r['progress']:6.3f} "
        f"{r['dx']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['min_up']:7.3f} "
        f"{r['qerr']:7.3f} "
        f"{r['qderr']:7.3f} "
        f"{r['ori']:8.2f} "
        f"{r['root']:8.3f} "
        f"{r['request_sat']:7.3f} "
        f"{r['ctrl_clip']:7.3f} "
        f"{r['shift_rms']:7.3f} "
        f"{r['shift_max']:8.3f} "
        f"{r['force_ratio']:6.3f}"
    )


# ================================================================
# AGGREGATE
# ================================================================

print()
print("=" * 150)
print("VELOCITY FEEDFORWARD AGGREGATE")
print("=" * 150)

print(
    f"{'ALPHA':>5s} "
    f"{'DONE':>5s} "
    f"{'PROG':>7s} "
    f"{'QERR':>7s} "
    f"{'QDERR':>7s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'MINUP':>7s} "
    f"{'REQSAT':>7s} "
    f"{'CLIP':>7s}"
)

print("-" * 150)


summaries = {}


for alpha in ALPHAS:

    rows = [
        r
        for r in results
        if r[
            "alpha"
        ] == alpha
    ]


    s = {

        "done":
            sum(
                int(
                    r["done"]
                )
                for r in rows
            ),

        "prog":
            float(
                np.mean(
                    [
                        r["progress"]
                        for r in rows
                    ]
                )
            ),

        "qerr":
            float(
                np.mean(
                    [
                        r["qerr"]
                        for r in rows
                    ]
                )
            ),

        "qderr":
            float(
                np.mean(
                    [
                        r["qderr"]
                        for r in rows
                    ]
                )
            ),

        "ori":
            float(
                np.mean(
                    [
                        r["ori"]
                        for r in rows
                    ]
                )
            ),

        "root":
            float(
                np.mean(
                    [
                        r["root"]
                        for r in rows
                    ]
                )
            ),

        "minup":
            float(
                np.mean(
                    [
                        r["min_up"]
                        for r in rows
                    ]
                )
            ),

        "reqsat":
            float(
                np.mean(
                    [
                        r["request_sat"]
                        for r in rows
                    ]
                )
            ),

        "clip":
            float(
                np.mean(
                    [
                        r["ctrl_clip"]
                        for r in rows
                    ]
                )
            ),
    }


    summaries[
        alpha
    ] = s


    print(
        f"{alpha:5.2f} "
        f"{s['done']:5d} "
        f"{s['prog']:7.3f} "
        f"{s['qerr']:7.3f} "
        f"{s['qderr']:7.3f} "
        f"{s['ori']:8.2f} "
        f"{s['root']:8.3f} "
        f"{s['minup']:7.3f} "
        f"{s['reqsat']:7.3f} "
        f"{s['clip']:7.3f}"
    )


# ================================================================
# FRAME 0
# ================================================================

print()
print("=" * 140)
print("FRAME-0 COMPARISON")
print("=" * 140)


for alpha in ALPHAS:

    r = next(
        item
        for item in results

        if (
            item["alpha"]
            == alpha

            and item["start"]
            == 0
        )
    )


    print(
        f"alpha={alpha:.2f} "
        f"steps={r['steps']:3d} "
        f"qerr={r['qerr']:.3f} "
        f"qderr={r['qderr']:.3f} "
        f"ori={r['ori']:.2f} "
        f"root={r['root']:.3f} "
        f"minUp={r['min_up']:.3f} "
        f"reqSat={r['request_sat']:.3f} "
        f"shiftMax={r['shift_max']:.3f}"
    )


# ================================================================
# DECISION
# ================================================================

base = summaries[
    0.00
]


best_alpha = max(
    ALPHAS,

    key=lambda a: (
        summaries[a]["done"],
        summaries[a]["prog"],
        -summaries[a]["ori"],
        -summaries[a]["root"],
        -summaries[a]["qderr"],
    ),
)


best = summaries[
    best_alpha
]


print()
print("=" * 150)
print("VELOCITY FEEDFORWARD DECISION")
print("=" * 150)

print(
    "best alpha:",
    f"{best_alpha:.2f}",
)

print(
    "completion:",
    f"{base['done']}/5",
    "->",
    f"{best['done']}/5",
)

print(
    "progress:",
    f"{base['prog']:.3f}",
    "->",
    f"{best['prog']:.3f}",
)

print(
    "joint velocity error:",
    f"{base['qderr']:.3f}",
    "->",
    f"{best['qderr']:.3f}",
)

print(
    "orientation:",
    f"{base['ori']:.2f}",
    "->",
    f"{best['ori']:.2f}",
)

print(
    "root:",
    f"{base['root']:.3f}",
    "->",
    f"{best['root']:.3f}",
)

print(
    "requested-force saturation:",
    f"{best['reqsat']:.3f}",
)

print(
    "control-range clipping:",
    f"{best['clip']:.3f}",
)


if (
    best_alpha > 0.0

    and (
        best["prog"]
        >= base["prog"] + 0.08

        or best["done"]
        > base["done"]
    )

    and best["ori"]
    <= base["ori"] + 5.0
):

    print()
    print(
        "RESULT: VELOCITY FEEDFORWARD HELPS"
    )

    print(
        "Use this alpha in the next "
        "closed-loop controller."
    )


elif (
    best_alpha > 0.0

    and best["qderr"]
    < base["qderr"] * 0.80
):

    print()
    print(
        "RESULT: VELOCITY TRACKING IMPROVED, "
        "BUT BALANCE DID NOT"
    )

    print(
        "Reference velocity tracking is better, "
        "but whole-body stabilization remains "
        "the bottleneck."
    )


else:

    print()
    print(
        "RESULT: VELOCITY FEEDFORWARD NOT USEFUL"
    )

    print(
        "The native position servo is already "
        "the better low-level controller."
    )

    print(
        "Next bottleneck is dynamic whole-body "
        "tracking / phase stabilization, not "
        "actuator gain semantics."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 150)
