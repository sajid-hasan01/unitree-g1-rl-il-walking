import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_INPUT = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)

DEFAULT_OUTPUT = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)


def apply_reference_pose(env, frame_index, height_offset, yaw_degrees):
    frame_index = int(frame_index) % env.num_frames

    env.data.qpos[:] = env.stand_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0

    if env.has_reference_root_positions:
        env.data.qpos[2] = float(env.reference_root_positions[frame_index, 2] + height_offset)
    else:
        env.data.qpos[2] = float(env.stand_qpos[2] + height_offset)

    env.data.qpos[3:7] = env._yaw_to_quat_wxyz(np.deg2rad(yaw_degrees))

    joint_pos = env.reference_joint_positions[frame_index]

    for i, qpos_address in enumerate(env.joint_qpos_addresses):
        env.data.qpos[qpos_address] = float(joint_pos[i])

    for i, actuator_id in enumerate(env.actuator_ids):
        env.data.ctrl[actuator_id] = env._clip_ctrl(actuator_id, joint_pos[i])

    for item in env.upper_body_actuators:
        env.data.ctrl[item["actuator_id"]] = env._clip_ctrl(
            item["actuator_id"],
            item["target_qpos"],
        )

    mujoco.mj_forward(env.model, env.data)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a functional gait-phase contact mask from reference foot-site "
            "relative height. The lower foot is stance; the higher foot is swing; "
            "both are stance only when their heights are close."
        )
    )

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--height_offset", type=float, default=0.10)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--double_support_threshold", type=float, default=0.025)
    parser.add_argument("--print_frames", type=int, default=80)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    env = G1DynamicWalkingEnv(
        dataset_path=str(input_path),
        reference_mode="cyclic",
        target_forward_velocity=-0.08,
        action_scale=0.060,
        action_target_smoothing=0.25,
        height_offset=args.height_offset,
        reference_speed=0.08,
        initial_stand_steps=120,
        transition_steps=700,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        use_reference_contact_mask=False,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    original = np.load(input_path, allow_pickle=True)
    output_data = {key: original[key] for key in original.files}

    if "contact_mask" in output_data:
        output_data["original_contact_mask"] = output_data["contact_mask"]

    phase_contact_mask = np.zeros((env.num_frames, 2), dtype=np.float32)
    left_clearance = np.zeros(env.num_frames, dtype=np.float32)
    right_clearance = np.zeros(env.num_frames, dtype=np.float32)

    for frame in range(env.num_frames):
        apply_reference_pose(
            env,
            frame,
            height_offset=args.height_offset,
            yaw_degrees=args.initial_yaw_degrees,
        )

        left_pos = env._get_site_position(env.left_foot_site_id)
        right_pos = env._get_site_position(env.right_foot_site_id)

        ground = float(min(left_pos[2], right_pos[2]))
        lclr = float(max(left_pos[2] - ground, 0.0))
        rclr = float(max(right_pos[2] - ground, 0.0))

        left_clearance[frame] = lclr
        right_clearance[frame] = rclr

        if lclr <= args.double_support_threshold and rclr <= args.double_support_threshold:
            left_contact = True
            right_contact = True
        elif lclr < rclr:
            left_contact = True
            right_contact = False
        else:
            left_contact = False
            right_contact = True

        phase_contact_mask[frame, 0] = 1.0 if left_contact else 0.0
        phase_contact_mask[frame, 1] = 1.0 if right_contact else 0.0

    output_data["contact_mask"] = phase_contact_mask
    output_data["phase_left_foot_clearance"] = left_clearance
    output_data["phase_right_foot_clearance"] = right_clearance
    output_data["contact_mask_source"] = np.array(["functional_relative_foot_height_phase"])
    output_data["double_support_threshold"] = np.array([args.double_support_threshold], dtype=np.float32)
    output_data["phase_height_offset"] = np.array([args.height_offset], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_data)

    print()
    print("=" * 90)
    print("FUNCTIONAL PHASE CONTACT DATASET CREATED")
    print("=" * 90)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Frames:", env.num_frames)
    print("height_offset:", args.height_offset)
    print("double_support_threshold:", args.double_support_threshold)
    print(
        "contact fractions:",
        "left=", float(phase_contact_mask[:, 0].mean()),
        "right=", float(phase_contact_mask[:, 1].mean()),
        "double=", float(np.mean(np.sum(phase_contact_mask, axis=1) == 2)),
        "flight=", float(np.mean(np.sum(phase_contact_mask, axis=1) == 0)),
    )
    print("left clearance min/mean/max:", float(left_clearance.min()), float(left_clearance.mean()), float(left_clearance.max()))
    print("right clearance min/mean/max:", float(right_clearance.min()), float(right_clearance.mean()), float(right_clearance.max()))
    print()
    print("First frames:")
    print("frame | Lmask Rmask | Lclr Rclr")
    for i in range(min(args.print_frames, env.num_frames)):
        print(
            f"{i:05d} | {int(phase_contact_mask[i,0])}     {int(phase_contact_mask[i,1])}     | "
            f"{left_clearance[i]:.4f} {right_clearance[i]:.4f}"
        )
    print("=" * 90)

    env.close()


if __name__ == "__main__":
    main()
