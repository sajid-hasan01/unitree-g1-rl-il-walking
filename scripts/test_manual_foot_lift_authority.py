import argparse
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)


def set_upper_body_hold(env):
    for item in env.upper_body_actuators:
        env.data.ctrl[item["actuator_id"]] = env._clip_ctrl(
            item["actuator_id"],
            item["target_qpos"],
        )


def initialize_state(env, start_mode, reference_frame):
    obs, info = env.reset()

    if start_mode == "stand":
        env.data.qpos[:] = env.stand_qpos
        env.data.qvel[:] = 0.0
        env.data.ctrl[:] = 0.0
        env.data.qpos[0] = 0.0
        env.data.qpos[1] = 0.0
        env.data.qpos[2] = float(env.stand_qpos[2] + env.height_offset)
        env.data.qpos[3:7] = env._yaw_to_quat_wxyz(np.deg2rad(env.initial_yaw_degrees))

        for i, qpos_address in enumerate(env.joint_qpos_addresses):
            joint_pos = float(env.data.qpos[qpos_address])
            env.data.ctrl[env.actuator_ids[i]] = env._clip_ctrl(env.actuator_ids[i], joint_pos)

        set_upper_body_hold(env)
        mujoco.mj_forward(env.model, env.data)

    elif start_mode == "reference":
        env.episode_step = env.initial_stand_steps + env.transition_steps
        env._apply_reference_state_initialization(reference_frame)
        env.episode_step = env.initial_stand_steps + env.transition_steps
        env.motion_frame = float(reference_frame)
        mujoco.mj_forward(env.model, env.data)

    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    env._update_previous_foot_positions()

    left_pos = env._get_site_position(env.left_foot_site_id)
    right_pos = env._get_site_position(env.right_foot_site_id)
    env.ground_foot_height = float(min(left_pos[2], right_pos[2]))

    return {
        "left_site": left_pos.copy(),
        "right_site": right_pos.copy(),
        "root_z": float(env.data.qpos[2]),
        "up_z": float(env._get_up_z()),
    }


def get_joint_index(env, joint_name):
    try:
        return env.controlled_joint_names.index(joint_name)
    except ValueError as exc:
        raise RuntimeError(f"Joint not controlled: {joint_name}") from exc


def run_trial(env, start_mode, reference_frame, leg, offsets, rollout_steps, frame_skip):
    initialize_state(env, start_mode, reference_frame)

    base_targets = np.zeros(env.num_actions, dtype=np.float64)
    for i, qpos_address in enumerate(env.joint_qpos_addresses):
        base_targets[i] = float(env.data.qpos[qpos_address])

    trial_targets = base_targets.copy()

    prefix = "left" if leg == "left" else "right"

    joint_names = [
        f"{prefix}_hip_pitch_joint",
        f"{prefix}_hip_roll_joint",
        f"{prefix}_knee_joint",
        f"{prefix}_ankle_pitch_joint",
        f"{prefix}_ankle_roll_joint",
    ]

    offset_vector = {
        joint_names[0]: float(offsets[0]),
        joint_names[1]: float(offsets[1]),
        joint_names[2]: float(offsets[2]),
        joint_names[3]: float(offsets[3]),
        joint_names[4]: float(offsets[4]),
    }

    joint_indices = {
        name: get_joint_index(env, name)
        for name in joint_names
    }

    max_relative_clearance = -1e9
    max_absolute_clearance = -1e9
    best_step = 0
    ever_swing_collision_free = False
    final_summary = {}

    for step in range(rollout_steps):
        alpha = min(1.0, (step + 1) / max(rollout_steps * 0.35, 1.0))

        trial_targets[:] = base_targets[:]

        for name, delta in offset_vector.items():
            trial_targets[joint_indices[name]] += alpha * delta

        for i, actuator_id in enumerate(env.actuator_ids):
            env.data.ctrl[actuator_id] = env._clip_ctrl(actuator_id, trial_targets[i])

        set_upper_body_hold(env)

        for _ in range(frame_skip):
            mujoco.mj_step(env.model, env.data)

        left_pos = env._get_site_position(env.left_foot_site_id)
        right_pos = env._get_site_position(env.right_foot_site_id)
        left_contact, right_contact = env._get_foot_contacts()

        if leg == "left":
            relative_clearance = float(left_pos[2] - right_pos[2])
            absolute_clearance = float(left_pos[2])
            target_collision_contact = bool(left_contact)
            support_collision_contact = bool(right_contact)
        else:
            relative_clearance = float(right_pos[2] - left_pos[2])
            absolute_clearance = float(right_pos[2])
            target_collision_contact = bool(right_contact)
            support_collision_contact = bool(left_contact)

        if relative_clearance > max_relative_clearance:
            max_relative_clearance = relative_clearance
            max_absolute_clearance = absolute_clearance
            best_step = step + 1

        if relative_clearance > 0.035 and not target_collision_contact:
            ever_swing_collision_free = True

        final_summary = {
            "root_z": float(env.data.qpos[2]),
            "up_z": float(env._get_up_z()),
            "left_z": float(left_pos[2]),
            "right_z": float(right_pos[2]),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "target_contact": target_collision_contact,
            "support_contact": support_collision_contact,
            "relative_clearance": relative_clearance,
            "absolute_clearance": absolute_clearance,
        }

    return {
        "leg": leg,
        "offsets": offsets,
        "max_relative_clearance": max_relative_clearance,
        "max_absolute_clearance": max_absolute_clearance,
        "best_step": best_step,
        "ever_swing_collision_free": ever_swing_collision_free,
        "final": final_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manual foot-lift authority test for Unitree G1 MuJoCo. "
            "This bypasses PPO and sweeps direct joint target offsets to check whether "
            "the controlled actuators and collision geometry can physically lift each foot."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--start_mode", type=str, default="stand", choices=["stand", "reference"])
    parser.add_argument("--reference_frame", type=int, default=25)
    parser.add_argument("--target_velocity", type=float, default=-0.08)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--reference_start_frame", type=int, default=25)
    parser.add_argument("--height_offset", type=float, default=0.10)
    parser.add_argument("--reference_speed", type=float, default=0.08)
    parser.add_argument("--action_scale", type=float, default=0.060)
    parser.add_argument("--action_target_smoothing", type=float, default=0.25)
    parser.add_argument("--initial_stand_steps", type=int, default=120)
    parser.add_argument("--transition_steps", type=int, default=700)
    parser.add_argument("--rollout_steps", type=int, default=80)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--leg", type=str, default="both", choices=["left", "right", "both"])

    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        initial_yaw_degrees=args.initial_yaw_degrees,
        reference_start_frame=args.reference_start_frame,
        height_offset=args.height_offset,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        include_contact_phase_observation=True,
        use_reference_contact_mask=True,
    )

    legs = ["left", "right"] if args.leg == "both" else [args.leg]

    # Offset order:
    # hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll
    candidate_values = {
        "hip_pitch": [-0.55, -0.30, 0.30, 0.55],
        "hip_roll": [-0.25, 0.0, 0.25],
        "knee": [-0.80, -0.45, 0.45, 0.80],
        "ankle_pitch": [-0.45, -0.20, 0.20, 0.45],
        "ankle_roll": [-0.20, 0.0, 0.20],
    }

    offset_candidates = list(
        itertools.product(
            candidate_values["hip_pitch"],
            candidate_values["hip_roll"],
            candidate_values["knee"],
            candidate_values["ankle_pitch"],
            candidate_values["ankle_roll"],
        )
    )

    print()
    print("=" * 110)
    print("MANUAL FOOT-LIFT AUTHORITY TEST")
    print("=" * 110)
    print("Dataset:", args.dataset_path)
    print("Start mode:", args.start_mode)
    print("Reference frame:", args.reference_frame)
    print("Yaw:", args.initial_yaw_degrees)
    print("Reference start frame:", args.reference_start_frame)
    print("Height offset:", args.height_offset)
    print("Rollout steps:", args.rollout_steps)
    print("Frame skip:", args.frame_skip)
    print("Candidate trials per leg:", len(offset_candidates))
    print("Offset order: hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll")
    print()

    initial = initialize_state(env, args.start_mode, args.reference_frame)
    print("Initial state:")
    print("  root_z:", initial["root_z"])
    print("  up_z:", initial["up_z"])
    print("  left_site:", initial["left_site"])
    print("  right_site:", initial["right_site"])
    print()

    all_results = []

    for leg in legs:
        print(f"Testing {leg} foot...")
        leg_results = []

        for offsets in offset_candidates:
            result = run_trial(
                env=env,
                start_mode=args.start_mode,
                reference_frame=args.reference_frame,
                leg=leg,
                offsets=offsets,
                rollout_steps=args.rollout_steps,
                frame_skip=args.frame_skip,
            )
            leg_results.append(result)

        leg_results.sort(key=lambda item: item["max_relative_clearance"], reverse=True)
        all_results.extend(leg_results)

        print()
        print(f"Top {min(args.top_k, len(leg_results))} {leg} foot-lift commands:")
        print(
            "rank | max_rel_clear | abs_z | best_step | collision_free | "
            "offsets(hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll) | "
            "final root/up | final L/R z | final L/R contact"
        )
        print("-" * 110)

        for rank, result in enumerate(leg_results[: args.top_k], start=1):
            final = result["final"]
            print(
                f"{rank:04d} | "
                f"{result['max_relative_clearance']:+.4f} | "
                f"{result['max_absolute_clearance']:+.4f} | "
                f"{result['best_step']:04d} | "
                f"{str(result['ever_swing_collision_free']):>14} | "
                f"{tuple(round(x, 3) for x in result['offsets'])} | "
                f"{final['root_z']:.3f}/{final['up_z']:.3f} | "
                f"{final['left_z']:.3f}/{final['right_z']:.3f} | "
                f"{int(final['left_contact'])}/{int(final['right_contact'])}"
            )

        print()

    all_results.sort(key=lambda item: item["max_relative_clearance"], reverse=True)
    best = all_results[0]

    print("=" * 110)
    print("BEST OVERALL:")
    print(best)
    print()
    if best["max_relative_clearance"] >= 0.045:
        print("DIAGNOSIS: The actuators CAN mechanically lift a foot. Next fix should change reward/control curriculum.")
    else:
        print("DIAGNOSIS: Manual commands also fail to create clear foot lift. Next fix must modify MuJoCo actuator strength, geometry, or reference scaling.")
    print("=" * 110)

    env.close()


if __name__ == "__main__":
    main()
