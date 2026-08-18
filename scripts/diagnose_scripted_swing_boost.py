from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env_v3b import G1PhaseBCResidualEnvV3B


LEFT_SWING_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
]

RIGHT_SWING_JOINTS = [
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
]


def joint_indices(joint_names: list[str], selected: list[str]) -> list[int]:
    indices: list[int] = []

    for name in selected:
        if name not in joint_names:
            raise RuntimeError(f"Joint not found in controlled joints: {name}")
        indices.append(joint_names.index(name))

    return indices


def build_swing_boost_action(
    env: G1PhaseBCResidualEnvV3B,
    boost_scale: float,
    boost_clip: float,
    include_roll_yaw: bool,
) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    phase_idx = env.phase_index % len(env.phase_bc_targets)

    desired_left_swing = bool(env.reference_left_swing[phase_idx])
    desired_right_swing = bool(env.reference_right_swing[phase_idx])

    phase_target = env.phase_bc_targets[phase_idx]
    stand_target = env.stand_joint_values

    if include_roll_yaw:
        left_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ]
        right_names = [
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
    else:
        left_names = LEFT_SWING_JOINTS
        right_names = RIGHT_SWING_JOINTS

    left_indices = joint_indices(env.joint_names, left_names)
    right_indices = joint_indices(env.joint_names, right_names)

    if desired_left_swing:
        for idx in left_indices:
            desired_extra_rad = boost_scale * (phase_target[idx] - stand_target[idx])
            action[idx] += desired_extra_rad / max(env.residual_scale, 1e-6)

    if desired_right_swing:
        for idx in right_indices:
            desired_extra_rad = boost_scale * (phase_target[idx] - stand_target[idx])
            action[idx] += desired_extra_rad / max(env.residual_scale, 1e-6)

    return np.clip(action, -boost_clip, boost_clip).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_residual_ppo_v3/models/phase_bc_residual_ppo_v3a_stable_drift_20k.zip",
    )

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=600)

    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--gait_scale", type=float, default=0.18)
    parser.add_argument("--residual_scale", type=float, default=0.16)
    parser.add_argument("--target_smoothing", type=float, default=0.06)
    parser.add_argument("--transition_steps", type=int, default=260)
    parser.add_argument("--target_velocity", type=float, default=0.020)

    parser.add_argument("--policy_weight", type=float, default=1.0)
    parser.add_argument("--boost_scale", type=float, default=0.25)
    parser.add_argument("--boost_clip", type=float, default=0.35)
    parser.add_argument("--include_roll_yaw", action="store_true")

    parser.add_argument("--print_every", type=int, default=20)

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = G1PhaseBCResidualEnvV3B(
        render_mode="human" if args.render else None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()

    print("=" * 100)
    print("SCRIPTED SWING-BOOST DIAGNOSTIC")
    print("=" * 100)
    print("Model:", model_path)
    print("Steps:", args.steps)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target velocity:", args.target_velocity)
    print("Policy weight:", args.policy_weight)
    print("Boost scale:", args.boost_scale)
    print("Boost clip:", args.boost_clip)
    print("Include roll/yaw:", args.include_roll_yaw)
    print("=" * 100)

    total_reward = 0.0
    final_info = info

    max_x = float(info["base_x"])
    min_z = float(info["base_z"])
    min_up_z = float(info["up_z"])

    both_contact_steps = 0
    left_only_steps = 0
    right_only_steps = 0
    no_contact_steps = 0

    desired_left_swing_steps = 0
    desired_right_swing_steps = 0

    actual_left_clearance_steps = 0
    actual_right_clearance_steps = 0

    max_left_clearance = 0.0
    max_right_clearance = 0.0

    max_action_abs = 0.0

    for step in range(args.steps):
        policy_action, _ = model.predict(obs, deterministic=args.deterministic)
        policy_action = np.asarray(policy_action, dtype=np.float32)

        boost_action = build_swing_boost_action(
            env=env,
            boost_scale=args.boost_scale,
            boost_clip=args.boost_clip,
            include_roll_yaw=args.include_roll_yaw,
        )

        final_action = args.policy_weight * policy_action + boost_action
        final_action = np.clip(final_action, -1.0, 1.0).astype(np.float32)

        max_action_abs = max(max_action_abs, float(np.max(np.abs(final_action))))

        obs, reward, terminated, truncated, info = env.step(final_action)

        total_reward += float(reward)
        final_info = info

        max_x = max(max_x, float(info["base_x"]))
        min_z = min(min_z, float(info["base_z"]))
        min_up_z = min(min_up_z, float(info["up_z"]))

        left_contact = bool(info["left_contact"])
        right_contact = bool(info["right_contact"])

        if left_contact and right_contact:
            both_contact_steps += 1
        elif left_contact and not right_contact:
            left_only_steps += 1
        elif right_contact and not left_contact:
            right_only_steps += 1
        else:
            no_contact_steps += 1

        if int(info["desired_left_swing"]) == 1:
            desired_left_swing_steps += 1

        if int(info["desired_right_swing"]) == 1:
            desired_right_swing_steps += 1

        if float(info["left_clearance"]) > 0.015:
            actual_left_clearance_steps += 1

        if float(info["right_clearance"]) > 0.015:
            actual_right_clearance_steps += 1

        max_left_clearance = max(max_left_clearance, float(info["left_clearance"]))
        max_right_clearance = max(max_right_clearance, float(info["right_clearance"]))

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"x={info['base_x']:+.3f} "
                f"z={info['base_z']:+.3f} "
                f"up_z={info['up_z']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"DL={info['desired_left_swing']} "
                f"DR={info['desired_right_swing']} "
                f"Lc={int(info['left_contact'])} "
                f"Rc={int(info['right_contact'])} "
                f"Lclear={info['left_clearance']:.4f} "
                f"Rclear={info['right_clearance']:.4f} "
                f"act_abs={info['action_mean_abs']:.3f} "
                f"boost_abs={float(np.mean(np.abs(boost_action))):.3f} "
                f"reward={reward:+.3f} "
                f"fallen={terminated}"
            )

        if terminated or truncated:
            break

    print()
    print("=" * 100)
    print("SCRIPTED SWING-BOOST SUMMARY")
    print("=" * 100)
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Max x:", f"{max_x:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Min z:", f"{min_z:+.3f}")
    print("Min up_z:", f"{min_up_z:+.3f}")
    print("Total reward:", f"{total_reward:+.3f}")
    print("Terminated:", final_info["terminated"])
    print("Truncated:", final_info["truncated"])
    print()
    print("Both contact steps:", both_contact_steps)
    print("Left-only contact steps:", left_only_steps)
    print("Right-only contact steps:", right_only_steps)
    print("No contact steps:", no_contact_steps)
    print("Desired left swing steps:", desired_left_swing_steps)
    print("Desired right swing steps:", desired_right_swing_steps)
    print("Actual left clearance > 0.015 steps:", actual_left_clearance_steps)
    print("Actual right clearance > 0.015 steps:", actual_right_clearance_steps)
    print("Max left clearance:", f"{max_left_clearance:+.4f}")
    print("Max right clearance:", f"{max_right_clearance:+.4f}")
    print("Max action abs:", f"{max_action_abs:.4f}")
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
