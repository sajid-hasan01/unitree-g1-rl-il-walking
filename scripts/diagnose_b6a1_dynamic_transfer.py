from pathlib import Path
import sys
import argparse
import csv
import math
import numpy as np
import mujoco

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from scripts.evaluate_deterministic_left_lift import (
    add_args,
    make_env,
)

label = sys.argv[1]
csv_path = Path(sys.argv[2])

parser = argparse.ArgumentParser(add_help=False)
add_args(parser)
args = parser.parse_args([])

args.target_clearance = 0.026
args.swing_z_weight = 1.00
args.swing_ik_gain = 0.90
args.swing_ik_max_delta = 0.14
args.max_steps = 650

env = make_env(args)
obs, info = env.reset()

m = env.model
d = env.data

LEFT_BODY = int(env.left_foot_body)
LEFT_SITE = int(env.left_foot_site)

# ------------------------------------------------------------
# Locate LEFT hip pitch joint + corresponding position actuator
# ------------------------------------------------------------

hip_jid = mujoco.mj_name2id(
    m,
    mujoco.mjtObj.mjOBJ_JOINT,
    "left_hip_pitch_joint",
)

if hip_jid < 0:
    raise RuntimeError("left_hip_pitch_joint not found")

hip_qadr = int(
    m.jnt_qposadr[hip_jid]
)

hip_act = None

for aid in range(m.nu):

    if int(m.actuator_trnid[aid][0]) == hip_jid:
        hip_act = aid
        break

if hip_act is None:
    raise RuntimeError(
        "Could not identify left hip pitch actuator"
    )

# ------------------------------------------------------------
# Sole spheres
# ------------------------------------------------------------

SOLE = []

for gid in range(m.ngeom):

    if int(m.geom_bodyid[gid]) != LEFT_BODY:
        continue

    if int(m.geom_type[gid]) != int(
        mujoco.mjtGeom.mjGEOM_SPHERE
    ):
        continue

    radius = float(
        m.geom_size[gid][0]
    )

    if radius <= 0.010:
        SOLE.append(gid)

SOLE = sorted(SOLE)

if len(SOLE) != 4:
    raise RuntimeError(
        f"Expected four sole spheres, found {SOLE}"
    )

floor_gid = mujoco.mj_name2id(
    m,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor",
)

floor_z = float(
    d.geom_xpos[floor_gid][2]
)


def sole_clearance():

    vals = {}

    for gid in SOLE:

        vals[gid] = (
            float(d.geom_xpos[gid][2])
            - float(m.geom_size[gid][0])
            - floor_z
        )

    limiting = min(
        vals,
        key=vals.get,
    )

    return (
        float(vals[limiting]),
        int(limiting),
        vals,
    )


def foot_tilt():

    R = np.asarray(
        d.site_xmat[LEFT_SITE],
        dtype=float,
    ).reshape(3, 3)

    z = R[:, 2]

    pitch = math.degrees(
        math.atan2(
            -float(z[0]),
            float(z[2]),
        )
    )

    roll = math.degrees(
        math.atan2(
            float(z[1]),
            float(z[2]),
        )
    )

    return pitch, roll


def whole_body_com():

    mass = np.asarray(
        m.body_mass[1:],
        dtype=float,
    )

    pos = np.asarray(
        d.xipos[1:],
        dtype=float,
    )

    return (
        mass[:, None] * pos
    ).sum(axis=0) / mass.sum()


initial_site = np.asarray(
    d.site_xpos[LEFT_SITE],
    dtype=float,
).copy()

initial_hip = float(
    d.qpos[hip_qadr]
)

rows = []

first_air = None
touchdown_step = None
shared_step = None

while True:

    obs, reward, terminated, truncated, info = env.step()

    step = int(
        info.get(
            "episode_step",
            len(rows) + 1,
        )
    )

    sw = float(
        info.get(
            "swing_env",
            0.0,
        )
    )

    contact = bool(
        info.get(
            "left_contact",
            False,
        )
    )

    force = float(
        info.get(
            "left_normal_force",
            0.0,
        )
    )

    td = bool(
        info.get(
            "touchdown_latched",
            False,
        )
    )

    shared = bool(
        info.get(
            "shared_support_latched",
            False,
        )
    )

    if (
        first_air is None
        and not contact
        and force <= 1.0
        and sw > 0.001
    ):
        first_air = step

    if touchdown_step is None and td:
        touchdown_step = step

    if shared_step is None and shared:
        shared_step = step

    sole, limiting, corners = (
        sole_clearance()
    )

    site = np.asarray(
        d.site_xpos[LEFT_SITE],
        dtype=float,
    ).copy()

    pitch, roll = foot_tilt()

    com = whole_body_com()

    hip_actual = float(
        d.qpos[hip_qadr]
    )

    hip_ctrl = float(
        d.ctrl[hip_act]
    )

    rows.append(
        {
            "label": label,
            "step": step,
            "sw": sw,

            "hip_ctrl_deg":
                math.degrees(hip_ctrl),

            "hip_actual_deg":
                math.degrees(hip_actual),

            "hip_actual_from_reset_deg":
                math.degrees(
                    hip_actual - initial_hip
                ),

            "hip_tracking_error_deg":
                math.degrees(
                    hip_ctrl - hip_actual
                ),

            "site_x_mm":
                1000.0 * (
                    site[0]
                    - initial_site[0]
                ),

            "site_y_mm":
                1000.0 * (
                    site[1]
                    - initial_site[1]
                ),

            "site_lift_mm":
                1000.0
                * float(
                    info.get(
                        "left_foot_clearance",
                        0.0,
                    )
                ),

            "sole_mm":
                1000.0 * sole,

            "limiting_geom":
                limiting,

            "g15_mm":
                1000.0
                * corners.get(15, float("nan")),

            "g16_mm":
                1000.0
                * corners.get(16, float("nan")),

            "g17_mm":
                1000.0
                * corners.get(17, float("nan")),

            "g18_mm":
                1000.0
                * corners.get(18, float("nan")),

            "foot_pitch_deg":
                pitch,

            "foot_roll_deg":
                roll,

            "contact":
                int(contact),

            "force_N":
                force,

            "touchdown":
                int(td),

            "shared":
                int(shared),

            "com_x_mm":
                1000.0 * com[0],

            "com_y_mm":
                1000.0 * com[1],

            "root_x_mm":
                1000.0 * float(d.qpos[0]),

            "root_y_mm":
                1000.0 * float(d.qpos[1]),

            "root_xv":
                float(d.qvel[0]),

            "root_yv":
                float(d.qvel[1]),
        }
    )

    if terminated or truncated:
        break


csv_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


max_sole = max(
    rows,
    key=lambda r: r["sole_mm"],
)

max_forward = max(
    rows,
    key=lambda r: r["site_x_mm"],
)

print()
print("=" * 100)
print(f"{label} SUMMARY")
print("=" * 100)

print(
    f"steps              = {len(rows)}"
)

print(
    f"first air          = {first_air}"
)

print(
    f"touchdown          = {touchdown_step}"
)

print(
    f"shared support     = {shared_step}"
)

print(
    f"max true sole      = "
    f"{max_sole['sole_mm']:.4f} mm "
    f"@ step {max_sole['step']}"
)

print(
    f"max forward X      = "
    f"{max_forward['site_x_mm']:.3f} mm "
    f"@ step {max_forward['step']}"
)

print(
    f"CSV                = {csv_path}"
)

env.close()
