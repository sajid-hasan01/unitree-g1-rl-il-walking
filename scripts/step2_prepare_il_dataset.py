import os
import argparse
import numpy as np


CONTROLLED_15_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


def normalize_name(name):
    name = str(name)

    if name.endswith("_joint"):
        return name

    return name + "_joint"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            "datasets",
            "raw",
            "amass_g1",
            "g1",
            "ACCAD",
            "Female1Walking_c3d",
            "B1-standtowalk_poses_120_jpos.npz",
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "datasets",
            "processed",
            "g1_amass_walking_il_15dof.npz",
        ),
    )

    parser.add_argument(
        "--left_hip_roll_offset",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--right_hip_roll_offset",
        type=float,
        default=-0.07,
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input AMASS file not found: {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = np.load(args.input, allow_pickle=True)

    fps = float(data["fps"][0])

    dof_names = [normalize_name(name) for name in data["dof_names"]]
    dof_positions = data["dof_positions"].astype(np.float32)
    dof_velocities = data["dof_velocities"].astype(np.float32)

    body_names = [str(name) for name in data["body_names"]]
    body_positions = data["body_positions"].astype(np.float32)
    body_rotations = data["body_rotations"].astype(np.float32)

    if "pelvis" in body_names:
        pelvis_index = body_names.index("pelvis")
    else:
        pelvis_index = 0

    root_positions = body_positions[:, pelvis_index, :]
    root_rotations = body_rotations[:, pelvis_index, :]

    joint_indices = []

    for joint_name in CONTROLLED_15_JOINTS:
        if joint_name not in dof_names:
            raise ValueError(f"Missing joint in AMASS file: {joint_name}")

        joint_indices.append(dof_names.index(joint_name))

    joint_pos_15 = dof_positions[:, joint_indices].copy()
    joint_vel_15 = dof_velocities[:, joint_indices].copy()

    left_hip_roll_id = CONTROLLED_15_JOINTS.index("left_hip_roll_joint")
    right_hip_roll_id = CONTROLLED_15_JOINTS.index("right_hip_roll_joint")

    joint_pos_15[:, left_hip_roll_id] += args.left_hip_roll_offset
    joint_pos_15[:, right_hip_roll_id] += args.right_hip_roll_offset

    num_frames = joint_pos_15.shape[0]

    progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    phase = 2.0 * np.pi * progress

    il_observations = np.stack(
        [
            np.sin(phase).astype(np.float32),
            np.cos(phase).astype(np.float32),
            progress.astype(np.float32),
        ],
        axis=1,
    )

    il_actions = joint_pos_15.astype(np.float32)

    np.savez_compressed(
        args.output,
        fps=np.array([fps], dtype=np.float32),
        controlled_joint_names=np.array(CONTROLLED_15_JOINTS),
        il_observations=il_observations,
        il_actions=il_actions,
        joint_pos_15=joint_pos_15.astype(np.float32),
        joint_vel_15=joint_vel_15.astype(np.float32),
        root_positions=root_positions.astype(np.float32),
        root_rotations=root_rotations.astype(np.float32),
        source_file=np.array([args.input]),
    )

    print("IL dataset prepared successfully.")
    print("Input:", args.input)
    print("Output:", args.output)
    print("FPS:", fps)
    print("Frames:", num_frames)
    print("il_observations:", il_observations.shape)
    print("il_actions:", il_actions.shape)
    print("joint_pos_15:", joint_pos_15.shape)
    print("joint_vel_15:", joint_vel_15.shape)

    print("\nControlled joint order:")
    for i, joint_name in enumerate(CONTROLLED_15_JOINTS):
        print(i, joint_name)


if __name__ == "__main__":
    main()