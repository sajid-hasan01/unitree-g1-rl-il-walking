from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env_v3b import (
    G1PhaseBCResidualEnvV3B,
    geom_min_z,
)


def controlled_index(env: G1PhaseBCResidualEnvV3B, joint_name: str) -> int:
    if joint_name not in env.joint_names:
        raise RuntimeError(f"Joint not found in controlled joints: {joint_name}")
    return env.joint_names.index(joint_name)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--side", choices=["left", "right"], default="right")

    parser.add_argument("--lift_start", type=int, default=80)
    parser.add_argument("--lift_ramp", type=int, default=120)
    parser.add_argument("--hold_steps", type=int, default=220)

    parser.add_argument("--support_roll", type=float, default=0.08)
    parser.add_argument("--hip_pitch_delta", type=float, default=-0.20)
    parser.add_argument("--knee_delta", type=float, default=0.42)
    parser.add_argument("--ankle_pitch_delta", type=float, default=-0.22)
    parser.add_argument("--ankle_roll_delta", type=float, default=0.04)

    parser.add_argument("--target_smoothing", type=float, default=0.04)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    env = G1PhaseBCResidualEnvV3B(
        render_mode="human" if args.render else None,
        start_frame=90,
        gait_scale=0.0,
        residual_scale=0.0,
        target_smoothing=args.target_smoothing,
        transition_steps=1,
        target_velocity=0.0,
        max_episode_steps=args.steps,
    )

    obs, info = env.reset()

    if args.side == "right":
        swing = "right"
        support = "left"
        swing_geoms = env.right_foot_geom_ids
        support_geoms = env.left_foot_geom_ids
        support_roll_sign = +1.0
    else:
        swing = "left"
        support = "right"
        swing_geoms = env.left_foot_geom_ids
        support_geoms = env.right_foot_geom_ids
        support_roll_sign = -1.0

    swing_hip_pitch = controlled_index(env, f"{swing}_hip_pitch_joint")
    swing_knee = controlled_index(env, f"{swing}_knee_joint")
    swing_ankle_pitch = controlled_index(env, f"{swing}_ankle_pitch_joint")
    swing_ankle_roll = controlled_index(env, f"{swing}_ankle_roll_joint")
    support_hip_roll = controlled_index(env, f"{support}_hip_roll_joint")
    support_ankle_roll = controlled_index(env, f"{support}_ankle_roll_joint")

    base_target = env.stand_joint_values.copy()
    current_target = base_target.copy()

    print("=" * 100)
    print("DIRECT SINGLE-FOOT LIFT DIAGNOSTIC")
    print("=" * 100)
    print("Swing side:", swing)
    print("Support side:", support)
    print("Support roll:", args.support_roll)
    print("Hip pitch delta:", args.hip_pitch_delta)
    print("Knee delta:", args.knee_delta)
    print("Ankle pitch delta:", args.ankle_pitch_delta)
    print("Ankle roll delta:", args.ankle_roll_delta)
    print("=" * 100)

    max_swing_clearance = 0.0
    min_z = float(info["base_z"])
    min_up_z = float(info["up_z"])
    max_x = float(info["base_x"])
    final_info = info

    stand_swing_z = geom_min_z(env.data, swing_geoms)
    stand_support_z = geom_min_z(env.data, support_geoms)

    for step in range(args.steps):
        if step < args.lift_start:
            alpha = 0.0
        else:
            alpha = smoothstep((step - args.lift_start) / max(args.lift_ramp, 1))

        desired = base_target.copy()

        # Shift body weight toward support leg.
        desired[support_hip_roll] += support_roll_sign * args.support_roll
        desired[support_ankle_roll] -= support_roll_sign * args.support_roll * 0.6

        # Lift swing leg with explicit pitch-pattern target.
        desired[swing_hip_pitch] += args.hip_pitch_delta * alpha
        desired[swing_knee] += args.knee_delta * alpha
        desired[swing_ankle_pitch] += args.ankle_pitch_delta * alpha
        desired[swing_ankle_roll] += args.ankle_roll_delta * alpha

        current_target = (
            (1.0 - args.target_smoothing) * current_target
            + args.target_smoothing * desired
        )

        env._apply_target(current_target)

        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)

        if env.viewer is not None:
            if env.viewer.is_running():
                env.viewer.sync()

        base_x = float(env.data.qpos[0])
        base_z = float(env.data.qpos[2])
        up_z = env._get_up_z()

        swing_z = geom_min_z(env.data, swing_geoms)
        support_z = geom_min_z(env.data, support_geoms)

        swing_clearance = max(0.0, swing_z - stand_swing_z)
        support_clearance = max(0.0, support_z - stand_support_z)

        max_swing_clearance = max(max_swing_clearance, swing_clearance)
        min_z = min(min_z, base_z)
        min_up_z = min(min_up_z, up_z)
        max_x = max(max_x, base_x)

        terminated = bool(base_z < env.fall_height or up_z < env.fall_up_z)

        final_info = {
            "step": step + 1,
            "base_x": base_x,
            "base_z": base_z,
            "up_z": up_z,
            "terminated": terminated,
        }

        if step % args.print_every == 0 or terminated:
            print(
                f"step={step:04d} "
                f"x={base_x:+.3f} "
                f"z={base_z:+.3f} "
                f"up_z={up_z:+.3f} "
                f"alpha={alpha:.2f} "
                f"swing_z={swing_z:+.4f} "
                f"support_z={support_z:+.4f} "
                f"swing_clear={swing_clearance:+.4f} "
                f"support_clear={support_clearance:+.4f} "
                f"fallen={terminated}"
            )

        if terminated:
            break

    print()
    print("=" * 100)
    print("DIRECT SINGLE-FOOT LIFT SUMMARY")
    print("=" * 100)
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Max x:", f"{max_x:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Min z:", f"{min_z:+.3f}")
    print("Min up_z:", f"{min_up_z:+.3f}")
    print("Max swing foot clearance:", f"{max_swing_clearance:+.4f}")
    print("Terminated:", final_info["terminated"])
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
