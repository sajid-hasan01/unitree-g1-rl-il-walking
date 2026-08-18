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

CONTACT_DEPTH = 0.00010       # 0.10 mm
TRANSITION_WINDOW = 2

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)

model = env.model

N = env.num_frames
DT = 1.0 / env.reference_fps

TOTAL_MASS = float(
    mujoco.mj_getTotalmass(model)
)

GRAVITY = abs(
    float(model.opt.gravity[2])
)

BODY_WEIGHT = TOTAL_MASS * GRAVITY


LEFT = set(env.left_sole_geoms)
RIGHT = set(env.right_sole_geoms)


print("=" * 185)
print("CONTACT-SNAPPED INVERSE-DYNAMICS AUDIT")
print("REFERENCE DATASET IS NOT MODIFIED")
print("NO CEM / NO PPO")
print("=" * 185)

print("reference:", env.reference_path)
print("frames:", N)
print("mass:", f"{TOTAL_MASS:.3f} kg")
print("body weight:", f"{BODY_WEIGHT:.2f} N")
print(
    "target support penetration:",
    f"{1000*CONTACT_DEPTH:.3f} mm",
)


# ================================================================
# REFERENCE ACCELERATION
# ================================================================

ref_vel = np.asarray(
    env.ref_full_qvel,
    dtype=np.float64,
)

ref_acc = np.zeros_like(ref_vel)

ref_acc[1:-1] = (
    ref_vel[2:] - ref_vel[:-2]
) / (2.0 * DT)

ref_acc[0] = (
    ref_vel[1] - ref_vel[0]
) / DT

ref_acc[-1] = (
    ref_vel[-1] - ref_vel[-2]
) / DT


# ================================================================
# GEOMETRY HELPERS
# ================================================================

def sole_clearance(data, geoms):

    floor_z = float(
        data.geom_xpos[
            env.floor_geom,
            2,
        ]
    )

    values = []

    for gid in geoms:

        radius = float(
            model.geom_size[gid, 0]
        )

        bottom = float(
            data.geom_xpos[gid, 2]
            - radius
        )

        values.append(
            bottom - floor_z
        )

    return min(values)


def active_contacts(data):

    left = False
    right = False

    count = 0

    for cid in range(data.ncon):

        con = data.contact[cid]

        if int(con.efc_address) < 0:
            continue

        g1 = int(con.geom1)
        g2 = int(con.geom2)

        if env.floor_geom not in (g1, g2):
            continue

        other = (
            g2
            if g1 == env.floor_geom
            else g1
        )

        if other in LEFT:
            left = True
            count += 1

        elif other in RIGHT:
            right = True
            count += 1

    return left, right, count


# ================================================================
# SUPPORT SIDE
#
# For double-support labels we anchor the geometrically closer
# foot rather than forcing the higher foot deep into the floor.
# ================================================================

probe = mujoco.MjData(model)


def choose_anchor(frame):

    probe.qpos[:] = env.ref_full_qpos[frame]
    probe.qvel[:] = env.ref_full_qvel[frame]

    mujoco.mj_forward(
        model,
        probe,
    )

    lc = sole_clearance(
        probe,
        env.left_sole_geoms,
    )

    rc = sole_clearance(
        probe,
        env.right_sole_geoms,
    )

    support = np.asarray(
        env.ref_support[frame],
        dtype=np.float64,
    )

    if np.sum(support) < 0.5:

        support = np.asarray(
            env.ref_contact[frame],
            dtype=np.float64,
        )

    candidates = []

    if support[0] > 0.5:
        candidates.append(("L", lc))

    if support[1] > 0.5:
        candidates.append(("R", rc))

    # Defensive fallback.
    if not candidates:

        candidates = [
            ("L", lc),
            ("R", rc),
        ]

    side, clearance = min(
        candidates,
        key=lambda x: x[1],
    )

    return (
        side,
        float(clearance),
        float(lc),
        float(rc),
    )


# ================================================================
# SUPPORT TRANSITION MASK
# ================================================================

transition = np.zeros(
    N,
    dtype=bool,
)


for frame in range(1, N):

    changed = not np.array_equal(
        env.ref_support[frame],
        env.ref_support[frame - 1],
    )

    if changed:

        lo = max(
            0,
            frame - TRANSITION_WINDOW,
        )

        hi = min(
            N,
            frame + TRANSITION_WINDOW + 1,
        )

        transition[lo:hi] = True


# ================================================================
# AUDIT FUNCTION
# ================================================================

def run_audit(snapped):

    data = mujoco.MjData(model)

    root_force = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    root_torque = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    tau = np.zeros(
        (N, 29),
        dtype=np.float64,
    )

    contact_normal = np.zeros(
        N,
        dtype=np.float64,
    )

    active = np.zeros(
        N,
        dtype=bool,
    )

    anchor_active = np.zeros(
        N,
        dtype=bool,
    )

    z_shift = np.zeros(
        N,
        dtype=np.float64,
    )

    anchor_code = np.zeros(
        N,
        dtype=np.int8,
    )


    force6 = np.zeros(
        6,
        dtype=np.float64,
    )


    for frame in range(N):

        qpos = np.asarray(
            env.ref_full_qpos[frame],
            dtype=np.float64,
        ).copy()


        side, clearance, lc, rc = (
            choose_anchor(frame)
        )


        if side == "L":
            anchor_code[frame] = 0
        else:
            anchor_code[frame] = 1


        if snapped:

            # desired anchor sole clearance = -CONTACT_DEPTH
            dz = (
                -CONTACT_DEPTH
                - clearance
            )

            # Only lower the root. Do not artificially lift it.
            dz = min(
                0.0,
                dz,
            )

            # Safety bound; current reference is far inside this.
            dz = max(
                dz,
                -0.020,
            )

            qpos[2] += dz

            z_shift[frame] = dz


        data.qpos[:] = qpos
        data.qvel[:] = env.ref_full_qvel[frame]


        # Generate collision / constraint structure.
        mujoco.mj_forward(
            model,
            data,
        )


        # mj_forward overwrites acceleration; restore desired qacc.
        data.qacc[:] = ref_acc[frame]


        mujoco.mj_inverse(
            model,
            data,
        )


        left, right, count = active_contacts(
            data
        )


        active[frame] = (
            left or right
        )


        if side == "L":
            anchor_active[frame] = left
        else:
            anchor_active[frame] = right


        root_force[frame] = (
            data.qfrc_inverse[0:3]
        )

        root_torque[frame] = (
            data.qfrc_inverse[3:6]
        )


        for j, vadr in enumerate(
            env.vaddrs
        ):

            tau[frame, j] = (
                data.qfrc_inverse[vadr]
            )


        # Inverse constraint contact forces.
        for cid in range(data.ncon):

            con = data.contact[cid]

            if int(con.efc_address) < 0:
                continue

            g1 = int(con.geom1)
            g2 = int(con.geom2)

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

            if (
                other not in LEFT
                and other not in RIGHT
            ):
                continue

            force6[:] = 0.0

            mujoco.mj_contactForce(
                model,
                data,
                cid,
                force6,
            )

            contact_normal[frame] += abs(
                float(force6[0])
            )


    return {
        "root_force":
            root_force,

        "root_torque":
            root_torque,

        "tau":
            tau,

        "contact_normal":
            contact_normal,

        "active":
            active,

        "anchor_active":
            anchor_active,

        "z_shift":
            z_shift,

        "anchor":
            anchor_code,
    }


print()
print("Running original reference audit...")

original = run_audit(
    snapped=False
)


print("Running contact-snapped audit...")

snapped = run_audit(
    snapped=True
)


# ================================================================
# METRICS
# ================================================================

effort = np.asarray(
    env.effort_limits,
    dtype=np.float64,
)


def metrics(result):

    force_norm = np.linalg.norm(
        result["root_force"],
        axis=1,
    )

    torque_norm = np.linalg.norm(
        result["root_torque"],
        axis=1,
    )

    force_bw = (
        force_norm / BODY_WEIGHT
    )

    torque_ratio = (
        np.abs(result["tau"])
        /
        effort[None, :]
    )

    max_joint_ratio = np.max(
        torque_ratio,
        axis=1,
    )

    normal_bw = (
        result["contact_normal"]
        /
        BODY_WEIGHT
    )

    return {
        "force_norm":
            force_norm,

        "force_bw":
            force_bw,

        "torque_norm":
            torque_norm,

        "torque_ratio":
            torque_ratio,

        "max_joint_ratio":
            max_joint_ratio,

        "normal_bw":
            normal_bw,
    }


orig_m = metrics(original)
snap_m = metrics(snapped)


def pct(x, p):
    return float(
        np.percentile(
            x,
            p,
        )
    )


# ================================================================
# SUMMARY
# ================================================================

print()
print("=" * 185)
print("CONTACT-SNAPPED INVERSE-DYNAMICS SUMMARY")
print("=" * 185)

print(
    "Original active-contact frames:",
    f"{100*np.mean(original['active']):.1f}%",
)

print(
    "Snapped active-contact frames:",
    f"{100*np.mean(snapped['active']):.1f}%",
)

print(
    "Snapped anchor-contact frames:",
    f"{100*np.mean(snapped['anchor_active']):.1f}%",
)


shift_mm = (
    -1000.0
    * snapped["z_shift"]
)


print()
print("ROOT LOWERING REQUIRED")

print(
    f"p50={pct(shift_mm,50):.2f} mm "
    f"p90={pct(shift_mm,90):.2f} mm "
    f"p95={pct(shift_mm,95):.2f} mm "
    f"max={np.max(shift_mm):.2f} mm"
)


print()
print("FLOATING-BASE RESIDUAL / BODY WEIGHT")

print(
    "ORIGINAL "
    f"p50={pct(orig_m['force_bw'],50):.3f} "
    f"p95={pct(orig_m['force_bw'],95):.3f} "
    f"max={np.max(orig_m['force_bw']):.3f}"
)

print(
    "SNAPPED  "
    f"p50={pct(snap_m['force_bw'],50):.3f} "
    f"p95={pct(snap_m['force_bw'],95):.3f} "
    f"max={np.max(snap_m['force_bw']):.3f}"
)


print()
print("INVERSE CONTACT NORMAL / BODY WEIGHT")

print(
    "ORIGINAL "
    f"p50={pct(orig_m['normal_bw'],50):.3f} "
    f"p95={pct(orig_m['normal_bw'],95):.3f} "
    f"max={np.max(orig_m['normal_bw']):.3f}"
)

print(
    "SNAPPED  "
    f"p50={pct(snap_m['normal_bw'],50):.3f} "
    f"p95={pct(snap_m['normal_bw'],95):.3f} "
    f"max={np.max(snap_m['normal_bw']):.3f}"
)


print()
print("MAX REQUIRED JOINT TORQUE / LIMIT")

print(
    "ORIGINAL "
    f"p50={pct(orig_m['max_joint_ratio'],50):.3f} "
    f"p95={pct(orig_m['max_joint_ratio'],95):.3f} "
    f"max={np.max(orig_m['max_joint_ratio']):.3f}"
)

print(
    "SNAPPED  "
    f"p50={pct(snap_m['max_joint_ratio'],50):.3f} "
    f"p95={pct(snap_m['max_joint_ratio'],95):.3f} "
    f"max={np.max(snap_m['max_joint_ratio']):.3f}"
)


over_limit = (
    snap_m["max_joint_ratio"] > 1.0
)


print(
    "Snapped frames requiring >100% effort:",
    int(np.sum(over_limit)),
    "/",
    N,
    f"({100*np.mean(over_limit):.1f}%)",
)


# ================================================================
# STEADY VS TRANSITION
# ================================================================

steady = ~transition


print()
print("=" * 185)
print("SUPPORT-TRANSITION RESIDUAL ANALYSIS")
print("=" * 185)

print(
    "transition-window frames:",
    int(np.sum(transition)),
)

print(
    "steady-support frames:",
    int(np.sum(steady)),
)


transition_p95 = pct(
    snap_m["force_bw"][transition],
    95,
) if np.any(transition) else float("nan")


steady_p95 = pct(
    snap_m["force_bw"][steady],
    95,
) if np.any(steady) else float("nan")


print(
    "snapped root residual p95 — steady:",
    f"{steady_p95:.3f} BW",
)

print(
    "snapped root residual p95 — transition:",
    f"{transition_p95:.3f} BW",
)


# ================================================================
# TOP HIGH-RESIDUAL FRAMES
# ================================================================

print()
print("=" * 185)
print("TOP 20 CONTACT-SNAPPED RESIDUAL FRAMES")
print("=" * 185)

order = np.argsort(
    snap_m["force_bw"]
)[::-1][:20]


print(
    f"{'FRAME':>5s} "
    f"{'BW':>7s} "
    f"{'FN/BW':>7s} "
    f"{'TAU/L':>7s} "
    f"{'SHIFT':>8s} "
    f"{'ANCH':>5s} "
    f"{'SUP-L':>5s} "
    f"{'SUP-R':>5s} "
    f"{'TRANS':>5s}"
)


for frame in order:

    anchor = (
        "L"
        if snapped["anchor"][frame] == 0
        else "R"
    )

    print(
        f"{frame:5d} "
        f"{snap_m['force_bw'][frame]:7.3f} "
        f"{snap_m['normal_bw'][frame]:7.3f} "
        f"{snap_m['max_joint_ratio'][frame]:7.3f} "
        f"{shift_mm[frame]:8.2f} "
        f"{anchor:>5s} "
        f"{int(env.ref_support[frame,0] > .5):5d} "
        f"{int(env.ref_support[frame,1] > .5):5d} "
        f"{int(transition[frame]):5d}"
    )


# ================================================================
# DECISION
# ================================================================

active_rate = float(
    np.mean(
        snapped["anchor_active"]
    )
)

p95_bw = pct(
    snap_m["force_bw"],
    95,
)

max_bw = float(
    np.max(
        snap_m["force_bw"]
    )
)

tau_p95 = pct(
    snap_m["max_joint_ratio"],
    95,
)

overlimit_fraction = float(
    np.mean(
        over_limit
    )
)


print()
print("=" * 185)
print("CONTACT-SNAPPED FEASIBILITY DECISION")
print("=" * 185)

print(
    "anchor contact validity:",
    f"{100*active_rate:.1f}%",
)

print(
    "root residual p95:",
    f"{p95_bw:.3f} BW",
)

print(
    "root residual max:",
    f"{max_bw:.3f} BW",
)

print(
    "required torque p95/limit:",
    f"{tau_p95:.3f}",
)

print(
    "over-limit frame fraction:",
    f"{100*overlimit_fraction:.1f}%",
)

print(
    "steady residual p95:",
    f"{steady_p95:.3f} BW",
)

print(
    "transition residual p95:",
    f"{transition_p95:.3f} BW",
)


print()


if active_rate < 0.95:

    print(
        "RESULT: DIAGNOSTIC CONTACT VALIDITY FAILED"
    )

    print(
        "Do not interpret inverse-dynamics residuals."
    )

    print(
        "The expected support foot was not active "
        "in enough snapped frames."
    )


elif (
    p95_bw <= 0.15
    and
    max_bw <= 0.40
    and
    tau_p95 <= 0.80
    and
    overlimit_fraction <= 0.01
):

    print(
        "RESULT: REFERENCE DYNAMIC FEASIBILITY PASSES"
    )

    print(
        "The previous ~1-BW root residual was mainly "
        "a missing-contact artifact."
    )

    print(
        "NEXT: one final support-transition/controller "
        "validation, then the first small PPO pilot."
    )


elif (
    steady_p95 <= 0.20
    and
    transition_p95 > max(
        0.25,
        1.5 * steady_p95,
    )
):

    print(
        "RESULT: SUPPORT TRANSITIONS ARE THE PRIMARY "
        "DYNAMIC BOTTLENECK"
    )

    print(
        "Steady stance is comparatively feasible, "
        "but contact-change frames require much "
        "larger unexplained root forces."
    )

    print(
        "NEXT: inspect transition timing, impact "
        "velocity and double-support blending."
    )


elif (
    p95_bw >= 0.30
    or
    max_bw >= 0.75
):

    print(
        "RESULT: GENUINE DYNAMIC INCONSISTENCY REMAINS"
    )

    print(
        "Even after creating an active support "
        "constraint, the reference needs substantial "
        "unexplained floating-base force."
    )

    print(
        "NEXT: dynamically project / optimize the "
        "reference before PPO."
    )


elif (
    tau_p95 >= 0.90
    or
    overlimit_fraction >= 0.10
):

    print(
        "RESULT: ACTUATOR FEASIBILITY BOTTLENECK"
    )

    print(
        "The contact-compatible reference requires "
        "too much joint effort."
    )


else:

    print(
        "RESULT: DYNAMIC FEASIBILITY IS MARGINAL"
    )

    print(
        "Do not start PPO yet."
    )

    print(
        "NEXT: inspect the high-residual frames "
        "printed above."
    )


# ================================================================
# SAVE
# ================================================================

output = (
    ROOT
    / "results"
    / "g1_restart_v1_contact_snapped_inverse.npz"
)


np.savez(
    output,

    original_root_force=
        original["root_force"].astype(
            np.float32
        ),

    snapped_root_force=
        snapped["root_force"].astype(
            np.float32
        ),

    snapped_tau=
        snapped["tau"].astype(
            np.float32
        ),

    snapped_contact_normal=
        snapped["contact_normal"].astype(
            np.float32
        ),

    snapped_z_shift=
        snapped["z_shift"].astype(
            np.float32
        ),

    transition_mask=
        transition,

    p95_root_bw=
        np.asarray(
            [p95_bw],
            dtype=np.float32,
        ),

    steady_p95_bw=
        np.asarray(
            [steady_p95],
            dtype=np.float32,
        ),

    transition_p95_bw=
        np.asarray(
            [transition_p95],
            dtype=np.float32,
        ),

    active_rate=
        np.asarray(
            [active_rate],
            dtype=np.float32,
        ),
)


print()
print(
    "artifact:",
    output,
)

print()
print(
    "REFERENCE FILE WAS NOT MODIFIED."
)

print(
    "NO CEM."
)

print(
    "NO PPO."
)

print("=" * 185)


env.close()
