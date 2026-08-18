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


CONTROL_HZ = 50.0

FAIL_HEIGHT = 0.45
FAIL_UP = 0.40


# ================================================================
# UNITREE G1 LOW-LEVEL EXAMPLE GAINS
#
# Joint order is the exact 29-DOF order already validated against
# the reference dataset and MuJoCo model.
# ================================================================

KP = np.asarray(
    [
        # left leg
        60, 60, 60, 100, 40, 40,

        # right leg
        60, 60, 60, 100, 40, 40,

        # waist
        60, 40, 40,

        # left arm
        40, 40, 40, 40, 40, 40, 40,

        # right arm
        40, 40, 40, 40, 40, 40, 40,
    ],
    dtype=np.float64,
)


KD = np.asarray(
    [
        # left leg
        1, 1, 1, 2, 1, 1,

        # right leg
        1, 1, 1, 2, 1, 1,

        # waist
        1, 1, 1,

        # left arm
        1, 1, 1, 1, 1, 1, 1,

        # right arm
        1, 1, 1, 1, 1, 1, 1,
    ],
    dtype=np.float64,
)


CONTROLLERS = {
    "OLD_ANGLE_AS_CTRL":
        None,

    "PD_0.5X":
        0.5,

    "PD_1.0X":
        1.0,
}


# ================================================================
# LOAD
# ================================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        MODEL_PATH
    )

if not DATASET.exists():
    raise FileNotFoundError(
        DATASET
    )


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


if ref_q.shape[1] != 29:
    raise RuntimeError(
        "Expected 29 reference joints."
    )

if len(names) != 29:
    raise RuntimeError(
        "Expected 29 joint names."
    )


# ================================================================
# MODEL
# ================================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
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


# ================================================================
# JOINT/ACTUATOR MAP
# ================================================================

joint_ids = []
qaddrs = []
vaddrs = []
aids = []

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


    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    if aid < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
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


    # ------------------------------------------------------------
    # Exact force limit from THIS local G1 model.
    #
    # Its MJCF joints define actuatorfrcrange.
    # ------------------------------------------------------------

    if hasattr(
        model,
        "jnt_actfrcrange",
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

        # Fallback matching the same local XML.
        fallback = {
            "left_hip_pitch_joint": 88,
            "left_hip_roll_joint": 139,
            "left_hip_yaw_joint": 88,
            "left_knee_joint": 139,
            "left_ankle_pitch_joint": 50,
            "left_ankle_roll_joint": 50,

            "right_hip_pitch_joint": 88,
            "right_hip_roll_joint": 139,
            "right_hip_yaw_joint": 88,
            "right_knee_joint": 139,
            "right_ankle_pitch_joint": 50,
            "right_ankle_roll_joint": 50,

            "waist_yaw_joint": 88,
            "waist_roll_joint": 50,
            "waist_pitch_joint": 50,

            "left_shoulder_pitch_joint": 25,
            "left_shoulder_roll_joint": 25,
            "left_shoulder_yaw_joint": 25,
            "left_elbow_joint": 25,
            "left_wrist_roll_joint": 25,
            "left_wrist_pitch_joint": 5,
            "left_wrist_yaw_joint": 5,

            "right_shoulder_pitch_joint": 25,
            "right_shoulder_roll_joint": 25,
            "right_shoulder_yaw_joint": 25,
            "right_elbow_joint": 25,
            "right_wrist_roll_joint": 25,
            "right_wrist_pitch_joint": 5,
            "right_wrist_yaw_joint": 5,
        }

        limit = float(
            fallback[name]
        )


    if limit <= 0.0:
        raise RuntimeError(
            f"Invalid effort limit for {name}: {limit}"
        )


    effort_limits.append(
        limit
    )


effort_limits = np.asarray(
    effort_limits,
    dtype=np.float64,
)


# ================================================================
# ACTUATOR SEMANTICS AUDIT
# ================================================================

print("=" * 155)
print("G1 ACTUATOR SEMANTICS + CORRECT PD BASELINE")
print("=" * 155)

print(
    "nq/nv/nu:",
    model.nq,
    model.nv,
    model.nu,
)

print(
    "reference frames:",
    len(ref_q),
)

print(
    "control Hz:",
    CONTROL_HZ,
)

print(
    "MuJoCo dt:",
    sim_dt,
)

print(
    "frame skip:",
    frame_skip,
)


print()
print("=" * 155)
print("COMPILED ACTUATOR AUDIT")
print("=" * 155)

print(
    f"{'#':>2s} "
    f"{'JOINT':28s} "
    f"{'GAIN0':>8s} "
    f"{'BIAS0':>8s} "
    f"{'BIAS1':>8s} "
    f"{'BIAS2':>8s} "
    f"{'GEAR':>7s} "
    f"{'EFFORT':>8s}"
)


direct_motor_count = 0


for j, aid in enumerate(
    aids
):

    gain0 = float(
        model.actuator_gainprm[
            aid,
            0,
        ]
    )

    bias0 = float(
        model.actuator_biasprm[
            aid,
            0,
        ]
    )

    bias1 = float(
        model.actuator_biasprm[
            aid,
            1,
        ]
    )

    bias2 = float(
        model.actuator_biasprm[
            aid,
            2,
        ]
    )

    gear = float(
        model.actuator_gear[
            aid,
            0,
        ]
    )


    is_direct = (
        abs(
            gain0 - 1.0
        ) < 1e-9

        and abs(
            bias0
        ) < 1e-9

        and abs(
            bias1
        ) < 1e-9

        and abs(
            bias2
        ) < 1e-9
    )


    if is_direct:
        direct_motor_count += 1


    print(
        f"{j:2d} "
        f"{names[j]:28s} "
        f"{gain0:8.3f} "
        f"{bias0:8.3f} "
        f"{bias1:8.3f} "
        f"{bias2:8.3f} "
        f"{gear:7.3f} "
        f"{effort_limits[j]:8.1f}"
    )


print()
print(
    "direct-drive motor-like actuators:",
    direct_motor_count,
    "/",
    29,
)


if direct_motor_count != 29:

    print(
        "WARNING: Not all actuators compiled as simple "
        "direct-drive motors. Inspect before interpreting PD."
    )


# ================================================================
# STATE HELPERS
# ================================================================

pelvis_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pelvis",
)


def get_joint_q(
    data,
):

    return np.asarray(
        [
            data.qpos[qadr]
            for qadr in qaddrs
        ],
        dtype=np.float64,
    )


def get_joint_qd(
    data,
):

    return np.asarray(
        [
            data.qvel[vadr]
            for vadr in vaddrs
        ],
        dtype=np.float64,
    )


def get_up_z(
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


def orientation_error_deg(
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


    actual = (
        actual
        / max(
            np.linalg.norm(actual),
            1e-12,
        )
    )

    desired = (
        desired
        / max(
            np.linalg.norm(desired),
            1e-12,
        )
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


def failed(
    data,
):

    return (
        float(
            data.qpos[2]
        ) < FAIL_HEIGHT

        or get_up_z(
            data
        ) < FAIL_UP
    )


# ================================================================
# CONTROLLER
# ================================================================

def apply_old_wrong_control(
    data,
    ref_index,
):

    # ------------------------------------------------------------
    # Reproduces the previous interpretation:
    #
    #   ctrl = desired angle
    #
    # On a direct-drive <motor>, this is NOT position control.
    # ------------------------------------------------------------

    for j, aid in enumerate(
        aids
    ):

        data.ctrl[
            aid
        ] = ref_q[
            ref_index,
            j,
        ]


def apply_pd_control(
    data,
    ref_index,
    gain_scale,
):

    q = get_joint_q(
        data
    )

    qd = get_joint_qd(
        data
    )


    q_target = ref_q[
        ref_index
    ]

    qd_target = ref_qd[
        ref_index
    ]


    tau_raw = (

        (
            KP
            * gain_scale
        )
        * (
            q_target
            - q
        )

        +

        (
            KD
            * gain_scale
        )
        * (
            qd_target
            - qd
        )
    )


    tau = np.clip(
        tau_raw,
        -effort_limits,
        +effort_limits,
    )


    for j, aid in enumerate(
        aids
    ):

        data.ctrl[
            aid
        ] = tau[
            j
        ]


    saturated = (
        np.abs(
            tau_raw
        )
        > effort_limits
    )


    return (
        tau,
        saturated,
    )


# ================================================================
# ROLLOUT
# ================================================================

def rollout(
    controller_name,
    gain_scale,
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

    min_up = get_up_z(
        data
    )

    max_y = 0.0

    qerr2 = 0.0
    qderr2 = 0.0
    orierr2 = 0.0
    rooterr2 = 0.0

    saturation_count = 0

    torque_samples = 0

    max_torque_ratio = 0.0


    print()
    print(
        f"{controller_name} "
        f"START={start:03d}"
    )


    while True:

        next_ref = min(
            current_ref + 1,
            len(ref_q) - 1,
        )


        # --------------------------------------------------------
        # Low-level control is updated at every MuJoCo simulation
        # step, while q_ref/qd_ref update at 50 Hz.
        # --------------------------------------------------------

        for _ in range(
            frame_skip
        ):

            if gain_scale is None:

                apply_old_wrong_control(
                    data,
                    next_ref,
                )

            else:

                tau, saturated = (
                    apply_pd_control(
                        data,
                        next_ref,
                        gain_scale,
                    )
                )


                saturation_count += int(
                    np.sum(
                        saturated
                    )
                )


                torque_samples += 29


                ratio = np.max(
                    np.abs(
                        tau
                    )
                    / effort_limits
                )


                max_torque_ratio = max(
                    max_torque_ratio,
                    float(
                        ratio
                    ),
                )


            mujoco.mj_step(
                model,
                data,
            )


        current_ref = next_ref

        steps += 1


        q = get_joint_q(
            data
        )

        qd = get_joint_qd(
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


        orierr = (
            orientation_error_deg(
                data.qpos[
                    3:7
                ],

                ref_quat[
                    current_ref
                ],
            )
        )


        rooterr = float(
            np.linalg.norm(
                data.qpos[
                    0:3
                ]
                - ref_root[
                    current_ref
                ]
            )
        )


        up = get_up_z(
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


        if (
            steps == 1
            or steps % 25 == 0
        ):

            print(
                f"  step={steps:03d} "
                f"ref={current_ref:03d} "
                f"x={data.qpos[0]-initial_x:+.3f} "
                f"vx={data.qvel[0]:+.3f} "
                f"y={data.qpos[1]-initial_y:+.3f} "
                f"z={data.qpos[2]:.3f} "
                f"up={up:.3f} "
                f"qerr={qerr:.3f} "
                f"ori={orierr:.1f}"
            )


        if failed(
            data
        ):

            completed = False
            break


        if current_ref >= (
            len(ref_q) - 1
        ):

            completed = True
            break


    saturation_fraction = (

        saturation_count
        / torque_samples

        if torque_samples > 0
        else 0.0
    )


    return {

        "controller":
            controller_name,

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

        "sat":
            saturation_fraction,

        "tau_ratio":
            max_torque_ratio,
    }


# ================================================================
# RUN ALL
# ================================================================

all_results = []


print()
print("=" * 155)
print("RUNNING CONTROLLER COMPARISON")
print("NO PPO")
print("=" * 155)


for controller_name, gain_scale in (
    CONTROLLERS.items()
):

    for start in STARTS:

        result = rollout(
            controller_name,
            gain_scale,
            start,
        )


        all_results.append(
            result
        )


# ================================================================
# DETAIL TABLE
# ================================================================

print()
print("=" * 165)
print("PD SERVO BASELINE SUMMARY")
print("=" * 165)

print(
    f"{'CONTROLLER':20s} "
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
    f"{'SAT':>7s} "
    f"{'TAUMAX':>7s}"
)

print("-" * 165)


for r in all_results:

    print(
        f"{r['controller']:20s} "
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
        f"{r['sat']:7.3f} "
        f"{r['tau_ratio']:7.3f}"
    )


# ================================================================
# AGGREGATE
# ================================================================

print()
print("=" * 140)
print("CONTROLLER AGGREGATE")
print("=" * 140)

print(
    f"{'CONTROLLER':20s} "
    f"{'DONE':>5s} "
    f"{'PROG':>7s} "
    f"{'QERR':>7s} "
    f"{'QDERR':>7s} "
    f"{'ORI':>8s} "
    f"{'ROOT':>8s} "
    f"{'MINUP':>7s} "
    f"{'SAT':>7s}"
)

print("-" * 140)


summaries = {}


for controller_name in (
    CONTROLLERS.keys()
):

    rows = [
        r
        for r in all_results
        if r[
            "controller"
        ] == controller_name
    ]


    summary = {

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

        "sat":
            float(
                np.mean(
                    [
                        r["sat"]
                        for r in rows
                    ]
                )
            ),
    }


    summaries[
        controller_name
    ] = summary


    print(
        f"{controller_name:20s} "
        f"{summary['done']:5d} "
        f"{summary['prog']:7.3f} "
        f"{summary['qerr']:7.3f} "
        f"{summary['qderr']:7.3f} "
        f"{summary['ori']:8.2f} "
        f"{summary['root']:8.3f} "
        f"{summary['minup']:7.3f} "
        f"{summary['sat']:7.3f}"
    )


# ================================================================
# FRAME-0 COMPARISON
# ================================================================

print()
print("=" * 140)
print("FRAME-0 CAUSAL COMPARISON")
print("=" * 140)


for controller_name in (
    CONTROLLERS.keys()
):

    row = next(
        r
        for r in all_results

        if (
            r["controller"]
            == controller_name

            and r["start"]
            == 0
        )
    )


    print(
        f"{controller_name:20s} "
        f"steps={row['steps']:3d} "
        f"done={row['done']} "
        f"qerr={row['qerr']:.4f} "
        f"ori={row['ori']:.2f} "
        f"root={row['root']:.3f} "
        f"minUp={row['min_up']:.3f} "
        f"sat={row['sat']:.3f}"
    )


# ================================================================
# DECISION
# ================================================================

old = summaries[
    "OLD_ANGLE_AS_CTRL"
]

pd05 = summaries[
    "PD_0.5X"
]

pd10 = summaries[
    "PD_1.0X"
]


best_name = max(
    [
        "PD_0.5X",
        "PD_1.0X",
    ],

    key=lambda name: (
        summaries[name]["done"],
        summaries[name]["prog"],
        -summaries[name]["ori"],
        -summaries[name]["root"],
    ),
)


best = summaries[
    best_name
]


print()
print("=" * 140)
print("SERVO DECISION")
print("=" * 140)

print(
    "best PD controller:",
    best_name,
)

print(
    "completion:",
    f"{old['done']}/5",
    "->",
    f"{best['done']}/5",
)

print(
    "mean progress:",
    f"{old['prog']:.3f}",
    "->",
    f"{best['prog']:.3f}",
)

print(
    "mean q error:",
    f"{old['qerr']:.3f}",
    "->",
    f"{best['qerr']:.3f}",
)

print(
    "mean orientation error:",
    f"{old['ori']:.2f}",
    "->",
    f"{best['ori']:.2f}",
)

print(
    "mean root error:",
    f"{old['root']:.3f}",
    "->",
    f"{best['root']:.3f}",
)

print(
    "PD torque saturation:",
    f"{best['sat']:.3f}",
)


if (
    best["done"] >= 3

    or best["prog"]
    >= old["prog"] + 0.20

    or (
        best["ori"]
        <= old["ori"] * 0.65

        and best["qerr"]
        <= old["qerr"] * 0.65
    )
):

    print()
    print(
        "RESULT: LOW-LEVEL SERVO BUG CONFIRMED"
    )

    print(
        "Directly writing joint angles into motor ctrl "
        "was a major cause of the previous failure."
    )

    print(
        "NEXT: rebuild the closed-loop tracking "
        "environment around PD torque servos."
    )

else:

    print()
    print(
        "RESULT: PD SERVO ALONE IS NOT SUFFICIENT"
    )

    print(
        "But actuator semantics are now correct."
    )

    print(
        "NEXT: diagnose gains / tracking dynamics "
        "before any PPO."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 140)
