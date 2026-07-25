import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "ember-lab-berkeley/AMASS_Retargeted_for_G1"
REPO_TYPE = "dataset"

DEFAULT_HF_FILE = "g1/BioMotionLab_NTroje/rub007/0005_normal_walk1_poses_120_jpos.npz"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "datasets" / "raw" / "amass_g1_candidates"
DEFAULT_MODEL_XML = PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


G1_JOINT_NAMES_29 = [
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
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def get_first_scalar(data, key, default):
    if key not in data:
        return default

    value = np.asarray(data[key]).reshape(-1)

    if len(value) == 0:
        return default

    return float(value[0])


def decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def find_local_file(local_dir, hf_file):
    expected_path = local_dir / hf_file

    if expected_path.exists():
        return expected_path

    matches = list(local_dir.rglob(Path(hf_file).name))

    if matches:
        return matches[0]

    return None


def download_or_find_file(hf_file, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)

    existing = find_local_file(local_dir, hf_file)

    if existing is not None:
        return existing

    print("File not found locally. Downloading from Hugging Face...")
    print("HF file:", hf_file)

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=hf_file,
        local_dir=str(local_dir),
    )

    return Path(downloaded_path)


def get_stand_qpos(model):
    if model.nkey > 0:
        return model.key_qpos[0].copy()

    return model.qpos0.copy()


def get_joint_qpos_address(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        return None

    return int(model.jnt_qposadr[joint_id])


def load_joint_positions(npz_data):
    possible_keys = [
        "dof_positions",
        "joint_positions",
        "jpos",
        "qpos",
    ]

    for key in possible_keys:
        if key in npz_data:
            array = np.asarray(npz_data[key], dtype=np.float32)

            if array.ndim == 2:
                return key, array

    raise KeyError(
        "Could not find joint positions. Expected one of: "
        "dof_positions, joint_positions, jpos, qpos"
    )


def load_root_positions(npz_data):
    possible_root_keys = [
        "root_positions",
        "root_pos",
        "root_translation",
        "root_trans",
        "trans",
        "translation",
    ]

    for key in possible_root_keys:
        if key in npz_data:
            array = np.asarray(npz_data[key], dtype=np.float32)

            if array.ndim == 2 and array.shape[1] >= 3:
                return key, array[:, :3]

    if "body_positions" in npz_data:
        body_positions = np.asarray(npz_data["body_positions"], dtype=np.float32)

        if body_positions.ndim != 3 or body_positions.shape[2] < 3:
            return None, None

        body_index = 0

        if "body_names" in npz_data:
            body_names = [decode_name(name) for name in npz_data["body_names"]]

            preferred_names = [
                "pelvis",
                "torso_link",
                "base_link",
                "root",
            ]

            for preferred_name in preferred_names:
                if preferred_name in body_names:
                    body_index = body_names.index(preferred_name)
                    break

        return "body_positions", body_positions[:, body_index, :3]

    return None, None


def set_joint_positions(model, qpos, joint_positions_frame):
    joint_count = min(len(joint_positions_frame), len(G1_JOINT_NAMES_29))

    for i in range(joint_count):
        joint_name = G1_JOINT_NAMES_29[i]
        qpos_address = get_joint_qpos_address(model, joint_name)

        if qpos_address is None:
            continue

        qpos[qpos_address] = joint_positions_frame[i]

    return qpos


def print_motion_summary(local_file, npz_data, joint_key, joint_positions, root_key, root_positions, fps):
    print()
    print("AMASS candidate loaded")
    print("File:", local_file)
    print("Joint key:", joint_key)
    print("Joint position shape:", joint_positions.shape)
    print("FPS:", fps)
    print("Frames:", joint_positions.shape[0])
    print("Duration:", round(joint_positions.shape[0] / fps, 4), "sec")

    print()
    print("Available NPZ keys:")
    for key in npz_data.keys():
        try:
            shape = np.asarray(npz_data[key]).shape
            print(f"  {key}: {shape}")
        except Exception:
            print(f"  {key}: unreadable")

    if root_positions is not None:
        delta = root_positions[-1] - root_positions[0]
        xy_displacement = float(np.linalg.norm(delta[:2]))

        print()
        print("Root source:", root_key)
        print("Root dx:", round(float(delta[0]), 4))
        print("Root dy:", round(float(delta[1]), 4))
        print("Root dz:", round(float(delta[2]), 4))
        print("Root xy displacement:", round(xy_displacement, 4))
        print("Average root xy speed:", round(xy_displacement / (joint_positions.shape[0] / fps), 4), "m/s")

        if xy_displacement < 0.10:
            print()
            print("NOTE: Root displacement is very small.")
            print("This is probably walking-in-place or treadmill-style motion.")
            print("Still check the leg stepping pattern visually.")
    else:
        print()
        print("No root trajectory found. The replay will use the standing root pose.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_file", type=str, default=DEFAULT_HF_FILE)
    parser.add_argument("--local_dir", type=str, default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--model_xml", type=str, default=str(DEFAULT_MODEL_XML))

    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--root_xy_scale", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)

    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    model_xml = Path(args.model_xml)

    if not model_xml.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_xml}")

    local_file = download_or_find_file(args.hf_file, local_dir)

    npz_data = np.load(local_file, allow_pickle=True)

    fps = get_first_scalar(npz_data, "fps", 30.0)

    joint_key, joint_positions = load_joint_positions(npz_data)
    root_key, root_positions = load_root_positions(npz_data)

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    stand_qpos = get_stand_qpos(model)

    print_motion_summary(
        local_file=local_file,
        npz_data=npz_data,
        joint_key=joint_key,
        joint_positions=joint_positions,
        root_key=root_key,
        root_positions=root_positions,
        fps=fps,
    )

    total_frames = joint_positions.shape[0]

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    root_start = None
    root_z_offset = 0.0

    if root_positions is not None:
        root_start = root_positions[0].copy()
        root_z_offset = stand_qpos[2] - root_start[2] + args.height_offset

    print()
    print("Viewer opening...")
    print("Close the MuJoCo viewer window to stop.")
    print()

    frame_time = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0

        while viewer.is_running():
            qpos = stand_qpos.copy()

            if root_positions is not None:
                root = root_positions[frame].copy()

                qpos[0] = (root[0] - root_start[0]) * args.root_xy_scale
                qpos[1] = (root[1] - root_start[1]) * args.root_xy_scale
                qpos[2] = root[2] + root_z_offset
            else:
                qpos[2] = stand_qpos[2] + args.height_offset

            qpos = set_joint_positions(
                model=model,
                qpos=qpos,
                joint_positions_frame=joint_positions[frame],
            )

            data.qpos[:] = qpos
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            if frame % 30 == 0:
                print(
                    f"frame={frame:04d}/{total_frames - 1}, "
                    f"time={frame / fps:.2f}s, "
                    f"root_x={data.qpos[0]:.3f}, "
                    f"root_y={data.qpos[1]:.3f}, "
                    f"root_z={data.qpos[2]:.3f}"
                )

            frame += 1

            if frame >= total_frames:
                if args.loop:
                    frame = 0
                else:
                    break

            time.sleep(frame_time * args.slowmo)

    print()
    print("Visualization finished.")


if __name__ == "__main__":
    main()