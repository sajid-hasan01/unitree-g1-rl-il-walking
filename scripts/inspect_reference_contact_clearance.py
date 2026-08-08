import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)


def apply_reference_frame(env, frame_index, use_reference_root_height=True, height_offset=0.02, yaw_degrees=0.0):
    frame_index = int(frame_index) % env.num_frames

    env.data.qpos[:] = env.stand_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0

    env.data.qpos[0] = 0.0
    env.data.qpos[1] = 0.0

    if use_reference_root_height and env.has_reference_root_positions:
        env.data.qpos[2] = float(env.reference_root_positions[frame_index, 2] + height_offset)
    else:
        env.data.qpos[2] = float(env.stand_qpos[2])

    env.data.qpos[3:7] = env._yaw_to_quat_wxyz(np.deg2rad(yaw_degrees))

    joint_pos = env.reference_joint_positions[frame_index]

    for i, qpos_address in enumerate(env.joint_qpos_addresses):
        env.data.qpos[qpos_address] = float(joint_pos[i])

    for i, actuator_id in enumerate(env.actuator_ids):
        target = env._clip_ctrl(actuator_id, joint_pos[i])
        env.data.ctrl[actuator_id] = target

    for item in env.upper_body_actuators:
        target = env._clip_ctrl(item["actuator_id"], item["target_qpos"])
        env.data.ctrl[item["actuator_id"]] = target

    mujoco.mj_forward(env.model, env.data)


def circular_shift(values, shift):
    return np.roll(values, int(shift), axis=0)


def match_rate(mask_contact, measured_contact):
    return float(np.mean(mask_contact.astype(bool) == measured_contact.astype(bool)))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect OpenHE/processed G1 reference contact mask vs kinematic foot clearance. "
            "This does not train and does not use PPO."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--target_velocity", type=float, default=-0.10)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--reference_speed", type=float, default=0.28)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--clearance_threshold", type=float, default=0.015)
    parser.add_argument("--max_shift", type=int, default=40)
    parser.add_argument("--print_frames", type=int, default=40)

    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        action_scale=0.055,
        action_target_smoothing=0.35,
        reference_speed=args.reference_speed,
        initial_stand_steps=70,
        transition_steps=220,
        include_contact_phase_observation=True,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    if not env.has_reference_contact_mask:
        raise RuntimeError("Dataset has no contact_mask.")

    left_z = []
    right_z = []

    for frame in range(env.num_frames):
        apply_reference_frame(
            env,
            frame,
            use_reference_root_height=True,
            height_offset=args.height_offset,
            yaw_degrees=args.initial_yaw_degrees,
        )

        left_pos = env._get_site_position(env.left_foot_site_id)
        right_pos = env._get_site_position(env.right_foot_site_id)

        left_z.append(float(left_pos[2]))
        right_z.append(float(right_pos[2]))

    left_z = np.asarray(left_z, dtype=np.float64)
    right_z = np.asarray(right_z, dtype=np.float64)

    ground_z = float(min(np.min(left_z), np.min(right_z)))
    left_clearance = np.maximum(left_z - ground_z, 0.0)
    right_clearance = np.maximum(right_z - ground_z, 0.0)

    measured_left_contact = left_clearance <= args.clearance_threshold
    measured_right_contact = right_clearance <= args.clearance_threshold

    mask_left = env.reference_contact_mask[:, 0] > 0.5
    mask_right = env.reference_contact_mask[:, 1] > 0.5

    print()
    print("=" * 90)
    print("REFERENCE CONTACT / FOOT-CLEARANCE INSPECTION")
    print("=" * 90)
    print("Dataset:", args.dataset_path)
    print("Frames:", env.num_frames)
    print("FPS:", env.fps)
    print("Yaw degrees:", args.initial_yaw_degrees)
    print("Height offset:", args.height_offset)
    print("Ground z estimated from reference:", ground_z)
    print("Clearance threshold:", args.clearance_threshold)
    print()

    print("Foot clearance summary:")
    print(f"  left  min/mean/max: {left_clearance.min():.5f} / {left_clearance.mean():.5f} / {left_clearance.max():.5f}")
    print(f"  right min/mean/max: {right_clearance.min():.5f} / {right_clearance.mean():.5f} / {right_clearance.max():.5f}")
    print()

    print("Contact fraction:")
    print(f"  mask left contact fraction:      {np.mean(mask_left):.3f}")
    print(f"  mask right contact fraction:     {np.mean(mask_right):.3f}")
    print(f"  measured left contact fraction:  {np.mean(measured_left_contact):.3f}")
    print(f"  measured right contact fraction: {np.mean(measured_right_contact):.3f}")
    print()

    direct_left = match_rate(mask_left, measured_left_contact)
    direct_right = match_rate(mask_right, measured_right_contact)

    inverted_left = match_rate(~mask_left, measured_left_contact)
    inverted_right = match_rate(~mask_right, measured_right_contact)

    swapped_left = match_rate(mask_right, measured_left_contact)
    swapped_right = match_rate(mask_left, measured_right_contact)

    print("Mask interpretation agreement:")
    print(f"  direct:   left={direct_left:.3f}, right={direct_right:.3f}, avg={(direct_left + direct_right) / 2:.3f}")
    print(f"  inverted: left={inverted_left:.3f}, right={inverted_right:.3f}, avg={(inverted_left + inverted_right) / 2:.3f}")
    print(f"  swapped:  left={swapped_left:.3f}, right={swapped_right:.3f}, avg={(swapped_left + swapped_right) / 2:.3f}")
    print()

    best = {
        "score": -1.0,
        "shift": 0,
        "mode": "direct",
        "left": 0.0,
        "right": 0.0,
    }

    for shift in range(-args.max_shift, args.max_shift + 1):
        shifted_left = circular_shift(mask_left, shift)
        shifted_right = circular_shift(mask_right, shift)

        candidates = [
            ("direct", shifted_left, shifted_right),
            ("inverted", ~shifted_left, ~shifted_right),
            ("swapped", shifted_right, shifted_left),
            ("swapped_inverted", ~shifted_right, ~shifted_left),
        ]

        for mode, cand_left, cand_right in candidates:
            l_score = match_rate(cand_left, measured_left_contact)
            r_score = match_rate(cand_right, measured_right_contact)
            score = 0.5 * (l_score + r_score)

            if score > best["score"]:
                best = {
                    "score": score,
                    "shift": shift,
                    "mode": mode,
                    "left": l_score,
                    "right": r_score,
                }

    print("Best contact-mask alignment search:")
    print(
        f"  mode={best['mode']}, shift={best['shift']} frames, "
        f"left={best['left']:.3f}, right={best['right']:.3f}, avg={best['score']:.3f}"
    )
    print()

    n = min(args.print_frames, env.num_frames)
    print(f"First {n} frames:")
    print("frame | maskL maskR | measL measR | Lclr  Rclr")
    for frame in range(n):
        print(
            f"{frame:05d} | "
            f"{int(mask_left[frame])}     {int(mask_right[frame])}     | "
            f"{int(measured_left_contact[frame])}     {int(measured_right_contact[frame])}     | "
            f"{left_clearance[frame]:.4f} {right_clearance[frame]:.4f}"
        )

    print()
    print("Frames where mask expects swing but kinematic clearance is still near ground:")
    bad_left = np.where((~mask_left) & measured_left_contact)[0]
    bad_right = np.where((~mask_right) & measured_right_contact)[0]
    print("  left expected swing but measured contact:", bad_left[:40].tolist(), "count=", len(bad_left))
    print("  right expected swing but measured contact:", bad_right[:40].tolist(), "count=", len(bad_right))

    print()
    if left_clearance.max() < 0.025 and right_clearance.max() < 0.025:
        print("DIAGNOSIS: reference foot lift is extremely small. The processed dataset is probably too flat for dynamic walking.")
    elif best["score"] < 0.70:
        print("DIAGNOSIS: contact_mask does not align well with kinematic foot clearance. Contact labels/phasing are likely wrong.")
    else:
        print("DIAGNOSIS: kinematic clearance and contact mask are reasonably aligned; failure is likely actuator/dynamics speed/transition.")

    env.close()
    print("=" * 90)


if __name__ == "__main__":
    main()
