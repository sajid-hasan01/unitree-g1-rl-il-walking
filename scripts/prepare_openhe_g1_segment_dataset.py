import argparse
import pathlib
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

REPO_ID = "openhe/g1-retargeted-motions"
REPO_TYPE = "dataset"

DEFAULT_REPO_FILE = "lafan1_retargeted/walk3_subject4.pkl"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "datasets" / "raw" / "openhe_g1_retargeted_motions"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_15dof.npz"
)


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


OPENHE_23_DOF_JOINT_NAMES = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def download_file(repo_file, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_file,
        local_dir=str(local_dir),
    )

    return Path(local_path)


def load_motion(local_path):
    original_posix_path = pathlib.PosixPath

    try:
        pathlib.PosixPath = pathlib.WindowsPath

        try:
            data = joblib.load(local_path)
        except Exception:
            with open(local_path, "rb") as file:
                data = pickle.load(file)

    finally:
        pathlib.PosixPath = original_posix_path

    if not isinstance(data, dict):
        raise ValueError(f"Loaded object is not dict. Type: {type(data)}")

    keys = list(data.keys())

    if len(keys) == 0:
        raise ValueError("Loaded dictionary is empty.")

    first_key = keys[0]
    first_value = data[first_key]

    if isinstance(first_value, dict):
        motion_key = str(first_key)
        motion = first_value
    else:
        motion_key = "direct"
        motion = data

    if not isinstance(motion, dict):
        raise ValueError(f"Motion object is not dict. Type: {type(motion)}")

    return motion_key, motion


def get_fps(motion):
    if "fps" not in motion:
        return 30.0

    fps = motion["fps"]

    if isinstance(fps, np.ndarray):
        fps = fps.reshape(-1)[0]

    return float(fps)


def finite_difference_velocity(position, fps):
    velocity = np.zeros_like(position, dtype=np.float32)

    if len(position) <= 1:
        return velocity

    velocity[1:] = (position[1:] - position[:-1]) * fps
    velocity[0] = velocity[1]

    return velocity


def make_phase_features(num_frames):
    phase = np.arange(num_frames, dtype=np.float32) / max(num_frames - 1, 1)
    sin_phase = np.sin(2.0 * np.pi * phase)
    cos_phase = np.cos(2.0 * np.pi * phase)

    return phase, sin_phase, cos_phase


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_file", type=str, default=DEFAULT_REPO_FILE)
    parser.add_argument("--local_dir", type=str, default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--start_frame", type=int, default=1320)
    parser.add_argument("--end_frame", type=int, default=1620)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))

    args = parser.parse_args()

    local_path = download_file(
        repo_file=args.repo_file,
        local_dir=Path(args.local_dir),
    )

    motion_key, motion = load_motion(local_path)

    required_keys = ["root_trans_offset", "root_rot", "dof", "contact_mask"]

    for key in required_keys:
        if key not in motion:
            raise KeyError(f"Missing required key: {key}")

    root_positions_full = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    root_rot_full = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_full = np.asarray(motion["dof"], dtype=np.float32)
    contact_mask_full = np.asarray(motion["contact_mask"], dtype=np.float32)
    fps = get_fps(motion)

    total_frames = min(
        len(root_positions_full),
        len(root_rot_full),
        len(dof_full),
        len(contact_mask_full),
    )

    start_frame = max(0, args.start_frame)
    end_frame = min(args.end_frame, total_frames)

    if end_frame <= start_frame + 2:
        raise ValueError("Invalid start_frame/end_frame range.")

    root_positions = root_positions_full[start_frame:end_frame].copy()
    root_rot = root_rot_full[start_frame:end_frame].copy()
    dof_23 = dof_full[start_frame:end_frame].copy()
    contact_mask = contact_mask_full[start_frame:end_frame].copy()

    if dof_23.shape[1] < 15:
        raise ValueError(f"Expected at least 15 DOF, got shape: {dof_23.shape}")

    joint_pos_15 = dof_23[:, :15].astype(np.float32)
    joint_vel_15 = finite_difference_velocity(joint_pos_15, fps)

    root_positions_relative = root_positions.copy()
    root_positions_relative[:, 0] -= root_positions_relative[0, 0]
    root_positions_relative[:, 1] -= root_positions_relative[0, 1]

    root_velocity = finite_difference_velocity(root_positions_relative[:, :3], fps)

    num_frames = joint_pos_15.shape[0]
    phase, sin_phase, cos_phase = make_phase_features(num_frames)

    il_observations = np.stack(
        [
            phase,
            root_velocity[:, 0],
            root_velocity[:, 1],
        ],
        axis=1,
    ).astype(np.float32)

    il_actions = joint_pos_15.copy().astype(np.float32)

    root_delta = root_positions_relative[-1, :3] - root_positions_relative[0, :3]
    xy_displacement = float(np.linalg.norm(root_delta[:2]))
    duration_sec = num_frames / fps
    avg_xy_speed = xy_displacement / duration_sec

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        joint_pos_15=joint_pos_15,
        joint_vel_15=joint_vel_15,
        il_observations=il_observations,
        il_actions=il_actions,
        root_positions=root_positions_relative.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        root_velocity=root_velocity.astype(np.float32),
        contact_mask=contact_mask.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source_total_frames=np.array([total_frames], dtype=np.int32),
        source_start_frame=np.array([start_frame], dtype=np.int32),
        source_end_frame=np.array([end_frame], dtype=np.int32),
        source_xy_displacement=np.array([xy_displacement], dtype=np.float32),
        source_avg_xy_speed=np.array([avg_xy_speed], dtype=np.float32),
        source_repo_file=np.array([args.repo_file]),
        source_motion_key=np.array([motion_key]),
        controlled_joint_names=np.array(CONTROLLED_15_JOINTS),
        openhe_23_dof_joint_names=np.array(OPENHE_23_DOF_JOINT_NAMES),
        is_cyclic=np.array([True]),
    )

    print()
    print("OpenHE segment converted successfully.")
    print("Source repo file:", args.repo_file)
    print("Motion key:", motion_key)
    print("Local source path:", local_path)
    print("Output:", output_path)
    print()
    print("Frames:", num_frames)
    print("FPS:", fps)
    print("Duration:", round(duration_sec, 4), "sec")
    print("joint_pos_15:", joint_pos_15.shape)
    print("joint_vel_15:", joint_vel_15.shape)
    print("root_positions:", root_positions_relative.shape)
    print("root_rot:", root_rot.shape)
    print("contact_mask:", contact_mask.shape)
    print()
    print("Root delta:", root_delta)
    print("XY displacement:", round(xy_displacement, 4), "m")
    print("Average XY speed:", round(avg_xy_speed, 4), "m/s")
    print()
    print("Controlled joints:")
    for i, joint_name in enumerate(CONTROLLED_15_JOINTS):
        print(f"{i:02d}: {joint_name}")


if __name__ == "__main__":
    main()