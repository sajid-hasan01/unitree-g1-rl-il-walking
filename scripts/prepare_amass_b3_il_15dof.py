from __future__ import annotations

import argparse
import math
from pathlib import Path

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


def decode_names(arr) -> list[str]:
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def rotation_2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def choose_root_body_index(body_names: list[str]) -> int:
    for word in ["pelvis", "torso", "trunk", "base", "waist"]:
        for i, name in enumerate(body_names):
            if word in name.lower():
                return i
    return 0


def build_il_observations(
    joint_pos_15: np.ndarray,
    joint_vel_15: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_frames = joint_pos_15.shape[0]

    q_current = joint_pos_15[:-1].astype(np.float32)
    qvel_current = joint_vel_15[:-1].astype(np.float32)
    q_next = joint_pos_15[1:].astype(np.float32)
    qvel_next = joint_vel_15[1:].astype(np.float32)

    prev_action = np.zeros_like(q_current, dtype=np.float32)
    prev_action[0] = q_current[0]
    prev_action[1:] = q_current[:-1]

    phase = np.arange(total_frames - 1, dtype=np.float32) / float(max(total_frames - 1, 1))
    phase_sin = np.sin(2.0 * np.pi * phase).reshape(-1, 1).astype(np.float32)
    phase_cos = np.cos(2.0 * np.pi * phase).reshape(-1, 1).astype(np.float32)

    il_observations = np.concatenate(
        [
            phase_sin,
            phase_cos,
            q_current,
            qvel_current,
            prev_action,
        ],
        axis=1,
    ).astype(np.float32)

    il_actions = q_next.astype(np.float32)

    return il_observations, il_actions, q_current, qvel_current, qvel_next


def align_root_motion(
    body_positions: np.ndarray,
    body_names: list[str],
    forward_axis: str,
    root_motion_scale: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    root_idx = choose_root_body_index(body_names)
    root_name = body_names[root_idx]

    root_xyz = body_positions[:, root_idx, :3].astype(np.float64)
    root_xy = root_xyz[:, :2]

    total_delta = root_xy[-1] - root_xy[0]
    heading = math.atan2(float(total_delta[1]), float(total_delta[0]))

    if forward_axis == "pos_x":
        target_angle = 0.0
    elif forward_axis == "neg_x":
        target_angle = math.pi
    else:
        raise ValueError("forward_axis must be pos_x or neg_x")

    rot = rotation_2d(target_angle - heading)

    aligned = np.zeros_like(root_xyz, dtype=np.float32)

    for i in range(len(root_xyz)):
        raw_delta_xy = root_xy[i] - root_xy[0]
        aligned_xy = rot @ raw_delta_xy

        aligned[i, 0] = float(aligned_xy[0] * root_motion_scale)
        aligned[i, 1] = float(aligned_xy[1] * root_motion_scale)
        aligned[i, 2] = float(root_xyz[i, 2] - root_xyz[0, 2])

    root_velocity = np.zeros_like(aligned, dtype=np.float32)
    if len(aligned) > 1:
        root_velocity[1:] = aligned[1:] - aligned[:-1]
        root_velocity[0] = root_velocity[1]

    return aligned, root_velocity, root_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_npz",
        type=str,
        default="experiments/amass_b3_15dof_il/raw/B3-walk1_poses_120_jpos.npz",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument("--forward_axis", choices=["pos_x", "neg_x"], default="pos_x")
    parser.add_argument("--root_motion_scale", type=float, default=0.45)
    args = parser.parse_args()

    raw_path = Path(args.raw_npz)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw AMASS B3 file not found: {raw_path}")

    data = np.load(raw_path, allow_pickle=True)

    required_keys = [
        "dof_names",
        "dof_positions",
        "dof_velocities",
        "body_names",
        "body_positions",
    ]

    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Missing required key: {key}")

    dof_names = decode_names(data["dof_names"])
    dof_positions = np.asarray(data["dof_positions"], dtype=np.float32)
    dof_velocities = np.asarray(data["dof_velocities"], dtype=np.float32)

    name_to_index = {name: i for i, name in enumerate(dof_names)}
    controlled_indices = []

    for joint_name in CONTROLLED_15_JOINTS:
        if joint_name not in name_to_index:
            raise KeyError(f"Joint not found in raw file: {joint_name}")
        controlled_indices.append(name_to_index[joint_name])

    joint_pos_15 = dof_positions[:, controlled_indices].astype(np.float32)
    joint_vel_15 = dof_velocities[:, controlled_indices].astype(np.float32)

    il_observations, il_actions, q_current, qvel_current, qvel_next = build_il_observations(
        joint_pos_15=joint_pos_15,
        joint_vel_15=joint_vel_15,
    )

    body_names = decode_names(data["body_names"])
    body_positions = np.asarray(data["body_positions"], dtype=np.float32)

    root_positions, root_velocity, root_body_name = align_root_motion(
        body_positions=body_positions,
        body_names=body_names,
        forward_axis=args.forward_axis,
        root_motion_scale=args.root_motion_scale,
    )

    fps = 30.0
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])

    np.savez_compressed(
        out_path,
        source_raw_npz=str(raw_path),
        source_motion_name="AMASS ACCAD Female1Walking B3-walk1",
        fps=np.array([fps], dtype=np.float32),
        controlled_joint_names=np.array(CONTROLLED_15_JOINTS),
        controlled_indices=np.array(controlled_indices, dtype=np.int64),
        joint_pos_15=joint_pos_15,
        joint_vel_15=joint_vel_15,
        il_observations=il_observations,
        il_actions=il_actions,
        q_current=q_current,
        qvel_current=qvel_current,
        qvel_next=qvel_next,
        root_positions=root_positions,
        root_velocity=root_velocity,
        root_body_name=np.array([root_body_name]),
        forward_axis=np.array([args.forward_axis]),
        root_motion_scale=np.array([args.root_motion_scale], dtype=np.float32),
    )

    displacement = np.linalg.norm(root_positions[-1, :2] - root_positions[0, :2])
    duration = joint_pos_15.shape[0] / fps

    print("=" * 100)
    print("Prepared AMASS B3 15-DOF IL dataset")
    print("=" * 100)
    print("Raw file:", raw_path)
    print("Output:", out_path)
    print("Frames:", joint_pos_15.shape[0])
    print("FPS:", fps)
    print("Duration:", f"{duration:.2f}s")
    print("Controlled joints:", len(CONTROLLED_15_JOINTS))
    print("joint_pos_15:", joint_pos_15.shape)
    print("joint_vel_15:", joint_vel_15.shape)
    print("il_observations:", il_observations.shape)
    print("il_actions:", il_actions.shape)
    print("Root body:", root_body_name)
    print("Aligned displacement:", f"{displacement:.3f}m")
    print("Forward axis:", args.forward_axis)
    print("=" * 100)


if __name__ == "__main__":
    main()
