import argparse
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


DEFAULT_INPUT = (
    "datasets/raw/amass_g1_candidates/g1/BioMotionLab_NTroje/"
    "rub007/0005_normal_walk1_poses_120_jpos.npz"
)

DEFAULT_OUTPUT = "datasets/processed/g1_amass_cyclic_walk_il_15dof.npz"


def decode_names(names_array):
    names = []

    for item in names_array:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))

    return names


def get_fps(data):
    if "fps" not in data:
        return 30.0

    fps_array = np.asarray(data["fps"]).reshape(-1)

    if len(fps_array) == 0:
        return 30.0

    return float(fps_array[0])


def get_required_array(data, key):
    if key not in data:
        raise KeyError(f"Missing required key in AMASS file: {key}")

    return np.asarray(data[key], dtype=np.float32)


def build_joint_index_map(dof_names):
    joint_indices = []

    for joint_name in CONTROLLED_15_JOINTS:
        if joint_name not in dof_names:
            raise ValueError(
                f"Controlled joint not found in AMASS dof_names: {joint_name}"
            )

        joint_indices.append(dof_names.index(joint_name))

    return joint_indices


def compute_phase_observations(num_frames):
    frame_indices = np.arange(num_frames, dtype=np.float32)
    phase = frame_indices / float(num_frames)

    phase_angle = 2.0 * np.pi * phase

    phase_sin = np.sin(phase_angle)
    phase_cos = np.cos(phase_angle)

    il_observations = np.stack(
        [
            phase_sin,
            phase_cos,
            phase,
        ],
        axis=1,
    ).astype(np.float32)

    return il_observations


def summarize_motion(root_positions, fps):
    if root_positions is None:
        return {
            "root_dx": 0.0,
            "root_dy": 0.0,
            "root_dz": 0.0,
            "root_xy_displacement": 0.0,
            "root_avg_xy_speed": 0.0,
        }

    root_delta = root_positions[-1] - root_positions[0]

    root_dx = float(root_delta[0])
    root_dy = float(root_delta[1])
    root_dz = float(root_delta[2])

    root_xy_displacement = float(np.linalg.norm(root_delta[:2]))
    duration = root_positions.shape[0] / fps

    if duration > 0:
        root_avg_xy_speed = root_xy_displacement / duration
    else:
        root_avg_xy_speed = 0.0

    return {
        "root_dx": root_dx,
        "root_dy": root_dy,
        "root_dz": root_dz,
        "root_xy_displacement": root_xy_displacement,
        "root_avg_xy_speed": root_avg_xy_speed,
    }


def extract_root_positions(data):
    if "body_positions" not in data:
        return None

    body_positions = np.asarray(data["body_positions"], dtype=np.float32)

    if body_positions.ndim != 3 or body_positions.shape[2] < 3:
        return None

    body_index = 0

    if "body_names" in data:
        body_names = decode_names(data["body_names"])

        for preferred_name in ["pelvis", "torso_link", "base_link", "root"]:
            if preferred_name in body_names:
                body_index = body_names.index(preferred_name)
                break

    return body_positions[:, body_index, :3]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input AMASS file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(input_path, allow_pickle=True)

    fps = get_fps(data)

    dof_positions = get_required_array(data, "dof_positions")
    dof_velocities = get_required_array(data, "dof_velocities")

    if "dof_names" not in data:
        raise KeyError("Missing required key in AMASS file: dof_names")

    dof_names = decode_names(data["dof_names"])

    joint_indices = build_joint_index_map(dof_names)

    joint_pos_15 = dof_positions[:, joint_indices].astype(np.float32)
    joint_vel_15 = dof_velocities[:, joint_indices].astype(np.float32)

    num_frames = joint_pos_15.shape[0]

    il_observations = compute_phase_observations(num_frames)

    # For behavior cloning style compatibility:
    # action = desired 15-DOF reference joint target.
    il_actions = joint_pos_15.copy()

    root_positions = extract_root_positions(data)
    root_summary = summarize_motion(root_positions, fps)

    np.savez_compressed(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        frames=np.array([num_frames], dtype=np.int32),
        is_cyclic=np.array([True], dtype=np.bool_),
        source_file=np.array([str(input_path)]),
        controlled_joint_names=np.array(CONTROLLED_15_JOINTS),
        dof_names=np.array(dof_names),
        joint_indices_15=np.array(joint_indices, dtype=np.int32),
        joint_pos_15=joint_pos_15,
        joint_vel_15=joint_vel_15,
        il_observations=il_observations,
        il_actions=il_actions,
        root_dx=np.array([root_summary["root_dx"]], dtype=np.float32),
        root_dy=np.array([root_summary["root_dy"]], dtype=np.float32),
        root_dz=np.array([root_summary["root_dz"]], dtype=np.float32),
        root_xy_displacement=np.array(
            [root_summary["root_xy_displacement"]],
            dtype=np.float32,
        ),
        root_avg_xy_speed=np.array(
            [root_summary["root_avg_xy_speed"]],
            dtype=np.float32,
        ),
    )

    duration = num_frames / fps

    print()
    print("Cyclic AMASS IL dataset prepared successfully.")
    print("Input:", input_path)
    print("Output:", output_path)
    print()
    print("FPS:", fps)
    print("Frames:", num_frames)
    print("Duration:", round(duration, 4), "sec")
    print("joint_pos_15:", joint_pos_15.shape)
    print("joint_vel_15:", joint_vel_15.shape)
    print("il_observations:", il_observations.shape)
    print("il_actions:", il_actions.shape)
    print()
    print("Root dx:", round(root_summary["root_dx"], 4))
    print("Root dy:", round(root_summary["root_dy"], 4))
    print("Root dz:", round(root_summary["root_dz"], 4))
    print("Root xy displacement:", round(root_summary["root_xy_displacement"], 4))
    print("Root avg xy speed:", round(root_summary["root_avg_xy_speed"], 4), "m/s")
    print()
    print("Controlled joints:")
    for i, joint_name in enumerate(CONTROLLED_15_JOINTS):
        print(f"{i:02d}: {joint_name}")


if __name__ == "__main__":
    main()