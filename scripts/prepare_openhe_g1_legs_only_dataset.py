import argparse
import pathlib
import pickle
from pathlib import Path

import joblib
import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "openhe/g1-retargeted-motions"
REPO_TYPE = "dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO_FILE = "lafan1_retargeted/walk3_subject4.pkl"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "datasets" / "raw" / "openhe_g1_retargeted_motions"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
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
        return str(first_key), first_value

    return "direct", data


def get_fps(motion):
    fps = motion.get("fps", 30.0)

    if isinstance(fps, np.ndarray):
        fps = fps.reshape(-1)[0]

    return float(fps)


def smooth_array(x, window):
    if window <= 1:
        return x.astype(np.float32)

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)

    for i in range(len(x)):
        out[i] = np.mean(padded[i : i + window], axis=0)

    return out.astype(np.float32)


def finite_difference_velocity(position, fps):
    velocity = np.zeros_like(position, dtype=np.float32)

    if len(position) <= 1:
        return velocity

    velocity[1:] = (position[1:] - position[:-1]) * fps
    velocity[0] = velocity[1]

    return velocity


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_file", type=str, default=DEFAULT_REPO_FILE)
    parser.add_argument("--local_dir", type=str, default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--start_frame", type=int, default=1320)
    parser.add_argument("--end_frame", type=int, default=1620)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))

    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--leg_scale", type=float, default=0.60)

    args = parser.parse_args()

    local_path = download_file(args.repo_file, Path(args.local_dir))
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

    start = max(0, args.start_frame)
    end = min(
        args.end_frame,
        len(dof_full),
        len(root_positions_full),
        len(root_rot_full),
        len(contact_mask_full),
    )

    if end <= start + 2:
        raise ValueError("Invalid frame range.")

    root_positions = root_positions_full[start:end].copy()
    root_rot = root_rot_full[start:end].copy()
    dof_23 = dof_full[start:end].copy()
    contact_mask = contact_mask_full[start:end].copy()

    if dof_23.shape[1] < 15:
        raise ValueError(f"Expected at least 15 DOF, got {dof_23.shape}")

    # Use only 12 leg joints from OpenHE.
    legs_12 = dof_23[:, :12].copy()

    # Reduce extreme motion to reduce shaking.
    mean_legs = np.mean(legs_12, axis=0, keepdims=True)
    legs_12 = mean_legs + args.leg_scale * (legs_12 - mean_legs)

    # Smooth leg motion.
    legs_12 = smooth_array(legs_12, args.smooth_window)

    # Hold waist stable to reduce left/right body tilt.
    waist_3 = np.zeros((legs_12.shape[0], 3), dtype=np.float32)

    joint_pos_15 = np.concatenate([legs_12, waist_3], axis=1).astype(np.float32)
    joint_vel_15 = finite_difference_velocity(joint_pos_15, fps)

    root_positions_relative = root_positions.copy()
    root_positions_relative[:, 0] -= root_positions_relative[0, 0]
    root_positions_relative[:, 1] -= root_positions_relative[0, 1]

    root_velocity = finite_difference_velocity(root_positions_relative[:, :3], fps)

    num_frames = joint_pos_15.shape[0]
    phase = np.arange(num_frames, dtype=np.float32) / max(num_frames - 1, 1)

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
        source_total_frames=np.array([len(dof_full)], dtype=np.int32),
        source_start_frame=np.array([start], dtype=np.int32),
        source_end_frame=np.array([end], dtype=np.int32),
        source_xy_displacement=np.array([xy_displacement], dtype=np.float32),
        source_avg_xy_speed=np.array([avg_xy_speed], dtype=np.float32),
        source_repo_file=np.array([args.repo_file]),
        source_motion_key=np.array([motion_key]),
        controlled_joint_names=np.array(CONTROLLED_15_JOINTS),
        is_cyclic=np.array([True]),
        waist_held_stable=np.array([True]),
        leg_scale=np.array([args.leg_scale], dtype=np.float32),
        smooth_window=np.array([args.smooth_window], dtype=np.int32),
    )

    print()
    print("OpenHE legs-only dataset created.")
    print("Output:", output_path)
    print("Frames:", num_frames)
    print("FPS:", fps)
    print("joint_pos_15:", joint_pos_15.shape)
    print("joint_vel_15:", joint_vel_15.shape)
    print("root_positions:", root_positions_relative.shape)
    print("contact_mask:", contact_mask.shape)
    print("waist held stable: True")
    print("leg_scale:", args.leg_scale)
    print("smooth_window:", args.smooth_window)
    print("xy displacement:", round(xy_displacement, 4), "m")
    print("avg xy speed:", round(avg_xy_speed, 4), "m/s")
    print("root delta:", root_delta)


if __name__ == "__main__":
    main()
