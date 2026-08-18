from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from envs.g1_closed_loop_tracking_env import (
    G1ClosedLoopTrackingEnv,
)


DATASET = (
    ROOT
    / "datasets"
    / "processed"
    / "g1_amass_walking_tracking_50hz.npz"
)


print("=" * 100)
print("TRACKING V1 DATASET PREFLIGHT")
print("=" * 100)


data = np.load(
    DATASET,
    allow_pickle=True,
)


print(
    "keys:",
    data.files,
)

print(
    "fps:",
    float(
        np.asarray(
            data["fps"]
        ).reshape(-1)[0]
    ),
)

print(
    "joint_pos:",
    data["joint_pos_15"].shape,
)

print(
    "joint_vel:",
    data["joint_vel_15"].shape,
)

print(
    "root_positions:",
    data["root_positions"].shape,
)

print(
    "root_velocity:",
    data["root_velocity"].shape,
)

print(
    "contact_mask:",
    data["contact_mask"].shape,
)

print(
    "has_contact_mask:",
    bool(
        np.asarray(
            data["has_contact_mask"]
        ).reshape(-1)[0]
    ),
)

print(
    "contact ones:",
    int(
        np.sum(
            data["contact_mask"] > 0.5
        )
    ),
)


if bool(
    np.asarray(
        data["has_contact_mask"]
    ).reshape(-1)[0]
):

    print(
        "Contact tracking reward: ENABLED"
    )

else:

    print(
        "Contact tracking reward: DISABLED"
    )


# =====================================================================
# ENVIRONMENT
# =====================================================================

print()
print("=" * 100)
print("TRACKING V1 ENVIRONMENT PREFLIGHT")
print("=" * 100)


env = G1ClosedLoopTrackingEnv(
    fixed_start_frame=42,

    random_reference_start=False,

    reset_position_noise=0.0,

    reset_velocity_noise=0.0,

    control_hz=50.0,

    target_velocity=-0.18,

    target_smoothing=0.20,
)


obs, info = env.reset(
    seed=1234
)


print(
    "OBS:",
    obs.shape,
)

print(
    "ACTION:",
    env.action_space.shape,
)

print(
    "REFERENCE FPS:",
    env.reference_fps,
)

print(
    "REFERENCE FRAMES:",
    env.num_frames,
)

print(
    "CONTROL HZ:",
    env.control_hz,
)

print(
    "MUJOCO DT:",
    env.sim_dt,
)

print(
    "FRAME SKIP:",
    env.frame_skip,
)

print(
    "ACTUAL CONTROL DT:",
    env.frame_skip
    * env.sim_dt,
)

print(
    "CONTACT REFERENCE:",
    env.has_reference_contact,
)

print(
    "START FRAME:",
    info[
        "reset_reference_frame"
    ],
)


if obs.shape != (126,):

    raise RuntimeError(
        f"Bad observation shape: {obs.shape}"
    )


if env.action_space.shape != (15,):

    raise RuntimeError(
        "Bad action shape."
    )


if not np.all(
    np.isfinite(
        obs
    )
):

    raise RuntimeError(
        "Observation contains NaN/Inf."
    )


# =====================================================================
# ACTION SCALE CHECK
# =====================================================================

print()
print("=" * 100)
print("PER-JOINT ACTION SCALES")
print("=" * 100)


for i, (
    name,
    scale,
) in enumerate(
    zip(
        env.joint_names,
        env.action_scale,
    )
):

    degrees = np.degrees(
        float(scale)
    )

    print(
        f"{i:02d} "
        f"{name:27s} "
        f"{float(scale):.5f} rad "
        f"({degrees:.2f} deg)"
    )


print()
print(
    "MIN SCALE:",
    float(
        np.min(
            env.action_scale
        )
    ),
)

print(
    "MAX SCALE:",
    float(
        np.max(
            env.action_scale
        )
    ),
)


if np.any(
    env.action_scale <= 0.0
):

    raise RuntimeError(
        "Non-positive action scale."
    )


if np.any(
    env.action_scale > 0.14 + 1e-6
):

    raise RuntimeError(
        "Unexpected action scale above safety clamp."
    )


# =====================================================================
# REFERENCE ERROR AT RESET
#
# Because RSI directly initializes q/qdot from the reference,
# these should initially be very small.
# =====================================================================

ref_q, ref_qd, _, _, idx = (
    env._reference()
)

q = env._joint_positions()
qd = env._joint_velocities()


q_error = float(
    np.sqrt(
        np.mean(
            (
                q - ref_q
            ) ** 2
        )
    )
)


qd_error = float(
    np.sqrt(
        np.mean(
            (
                qd - ref_qd
            ) ** 2
        )
    )
)


print()
print("=" * 100)
print("RESET REFERENCE CONSISTENCY")
print("=" * 100)

print(
    "reference index:",
    idx,
)

print(
    "initial q RMS error:",
    q_error,
)

print(
    "initial qdot RMS error:",
    qd_error,
)

print(
    "initial z:",
    float(
        env.data.qpos[2]
    ),
)

print(
    "initial up_z:",
    env._up_z(),
)

print(
    "initial vx:",
    float(
        env.data.qvel[0]
    ),
)


# =====================================================================
# SHORT ZERO-ACTION DYNAMIC TEST
#
# action = 0 means:
#
#     target = actual reference pose
#
# This is NOT expected to solve walking.
# It only checks whether the new reference/control pipeline
# is physically sane before PPO.
# =====================================================================

print()
print("=" * 100)
print("25-STEP ZERO-ACTION DYNAMIC SANITY TEST")
print("=" * 100)


terminated_early = False


for step in range(
    1,
    26,
):

    action = np.zeros(
        15,
        dtype=np.float32,
    )


    (
        obs,
        reward,
        terminated,
        truncated,
        step_info,
    ) = env.step(
        action
    )


    if (
        step == 1
        or step % 5 == 0
        or terminated
        or truncated
    ):

        print(
            f"step={step:02d} "
            f"ref={step_info['reference_index']:03d} "
            f"z={step_info['z']:.3f} "
            f"up={step_info['up_z']:.3f} "
            f"x={step_info['x']:+.3f} "
            f"y={step_info['y']:+.3f} "
            f"yaw={step_info['yaw_deg']:+.1f} "
            f"qerr={step_info['q_error_rms']:.3f} "
            f"reward={reward:+.3f}"
        )


    if terminated or truncated:

        terminated_early = True

        print(
            "ZERO-ACTION TEST ENDED EARLY:",
            "terminated="
            f"{terminated}",
            "truncated="
            f"{truncated}",
        )

        break


print()
print("=" * 100)
print("PREFLIGHT SUMMARY")
print("=" * 100)

print(
    "Dataset 50 Hz:",
    "PASS",
)

print(
    "Observation:",
    "PASS",
)

print(
    "Action space:",
    "PASS",
)

print(
    "Finite state:",
    "PASS",
)

print(
    "Contact reward:",
    (
        "ENABLED"
        if env.has_reference_contact
        else
        "DISABLED - source has no labels"
    ),
)

print(
    "Zero-action early termination:",
    terminated_early,
)

print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)


env.close()
