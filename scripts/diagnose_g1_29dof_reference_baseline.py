import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz_29dof_v1_grounded.npz"
)

MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


STARTS = [
    0,
    50,
    100,
    150,
    200,
]


CONTROL_HZ = 50.0


# =====================================================================
# LOAD
# =====================================================================

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

data = mujoco.MjData(
    model
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


if frame_skip <= 0:

    raise RuntimeError(
        "Invalid frame_skip."
    )


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
            f"Missing joint/actuator: {name}"
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

    aids.append(aid)


def joint_q():

    return np.asarray(
        [
            data.qpos[qadr]
            for qadr
            in qaddrs
        ],
        dtype=np.float64,
    )


def joint_qd():

    return np.asarray(
        [
            data.qvel[vadr]
            for vadr
            in vaddrs
        ],
        dtype=np.float64,
    )


def up_z():

    mat = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        mat,
        data.qpos[
            3:7
        ],
    )

    return float(
        mat.reshape(
            3,
            3,
        )[2, 2]
    )


def quat_angle_deg(
    qa,
    qb,
):

    qa = qa / max(
        np.linalg.norm(qa),
        1e-12,
    )

    qb = qb / max(
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

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    return math.degrees(
        2.0
        * math.acos(dot)
    )


def run(
    start,
):

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


    data.ctrl[:] = 0.0


    for j, aid in enumerate(
        aids
    ):

        target = ref_q[
            start,
            j,
        ]


        if (
            model.actuator_ctrllimited[
                aid
            ]
        ):

            lo, hi = (
                model.actuator_ctrlrange[
                    aid
                ]
            )

            target = float(
                np.clip(
                    target,
                    lo,
                    hi,
                )
            )


        data.ctrl[
            aid
        ] = target


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


    steps = 0

    min_up = up_z()

    max_y = 0.0

    qerr2 = 0.0
    qderr2 = 0.0
    orierr2 = 0.0
    rooterr2 = 0.0

    final_ref = start

    completed = False


    print()
    print(
        f"START={start:03d}"
    )


    while True:

        next_ref = min(
            start + steps + 1,
            len(ref_q) - 1,
        )


        for j, aid in enumerate(
            aids
        ):

            target = ref_q[
                next_ref,
                j,
            ]


            if (
                model.actuator_ctrllimited[
                    aid
                ]
            ):

                lo, hi = (
                    model.actuator_ctrlrange[
                        aid
                    ]
                )

                target = float(
                    np.clip(
                        target,
                        lo,
                        hi,
                    )
                )


            data.ctrl[
                aid
            ] = target


        for _ in range(
            frame_skip
        ):

            mujoco.mj_step(
                model,
                data,
            )


        steps += 1

        final_ref = next_ref


        q = joint_q()

        qd = joint_qd()


        qerr = float(
            np.sqrt(
                np.mean(
                    (
                        q
                        - ref_q[
                            final_ref
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
                            final_ref
                        ]
                    ) ** 2
                )
            )
        )


        orierr = quat_angle_deg(
            data.qpos[
                3:7
            ],

            ref_quat[
                final_ref
            ],
        )


        rooterr = float(
            np.linalg.norm(
                data.qpos[
                    0:3
                ]
                - (
                    ref_root[
                        final_ref
                    ]
                    - ref_root[
                        start
                    ]
                    + ref_root[
                        start
                    ]
                )
            )
        )


        up = up_z()


        min_up = min(
            min_up,
            up,
        )


        max_y = max(
            max_y,

            abs(
                float(
                    data.qpos[1]
                    - initial_y
                )
            ),
        )


        qerr2 += (
            qerr ** 2
        )

        qderr2 += (
            qderr ** 2
        )

        orierr2 += (
            orierr ** 2
        )

        rooterr2 += (
            rooterr ** 2
        )


        z = float(
            data.qpos[2]
        )


        if (
            steps == 1
            or steps % 25 == 0
        ):

            print(
                f"  step={steps:03d} "
                f"ref={final_ref:03d} "
                f"x={data.qpos[0]-initial_x:+.3f} "
                f"vx={data.qvel[0]:+.3f} "
                f"y={data.qpos[1]-initial_y:+.3f} "
                f"z={z:.3f} "
                f"up={up:.3f} "
                f"qerr={qerr:.3f} "
                f"ori={orierr:.1f}"
            )


        # Same physical-failure idea as earlier diagnostics.
        if (
            z < 0.45
            or up < 0.40
        ):

            break


        if final_ref >= (
            len(ref_q) - 1
        ):

            completed = True
            break


    result = {

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
            float(
                np.sqrt(
                    qerr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "qderr":
            float(
                np.sqrt(
                    qderr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "ori":
            float(
                np.sqrt(
                    orierr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),

        "root":
            float(
                np.sqrt(
                    rooterr2
                    / max(
                        steps,
                        1,
                    )
                )
            ),
    }


    return result


print("=" * 150)
print("FULL 29-DOF REFERENCE PHYSICS BASELINE")
print("ALL 29 JOINTS FOLLOW THE DEMONSTRATION")
print("NO PPO")
print("=" * 150)

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


results = [
    run(start)
    for start in STARTS
]


print()
print("=" * 150)
print("29-DOF BASELINE SUMMARY")
print("=" * 150)

print(
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
    f"{'ROOT':>8s}"
)

print("-" * 150)


for r in results:

    print(
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
        f"{r['root']:8.3f}"
    )


print()
print(
    "completed:",
    sum(
        int(
            r["done"]
        )
        for r in results
    ),
    "/",
    len(results),
)

print(
    "mean progress:",
    f"{np.mean([r['progress'] for r in results]):.3f}",
)

print(
    "mean q error:",
    f"{np.mean([r['qerr'] for r in results]):.3f}",
)

print(
    "mean orientation error:",
    f"{np.mean([r['ori'] for r in results]):.2f} deg",
)

print(
    "mean root error:",
    f"{np.mean([r['root'] for r in results]):.3f} m",
)


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 150)
