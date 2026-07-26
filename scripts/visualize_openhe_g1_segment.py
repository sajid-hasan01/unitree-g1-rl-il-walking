import argparse
import pathlib
import pickle
import sys
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

REPO_ID = "openhe/g1-retargeted-motions"
REPO_TYPE = "dataset"

DEFAULT_REPO_FILE = "lafan1_retargeted/walk1_subject1.pkl"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "datasets" / "raw" / "openhe_g1_retargeted_motions"
DEFAULT_MODEL_XML = (
    PROJECT_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


# OpenHE 23-DOF assumption:
# 12 leg joints + 3 waist joints + 8 arm joints.
#
# This is the most likely order for this retargeted G1 dataset.
# We are using visualization first to verify whether this mapping is correct.
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

    return motion_key, motion


def get_fps(motion):
    if "fps" not in motion:
        return 30.0

    fps = motion["fps"]

    if isinstance(fps, np.ndarray):
        fps = fps.reshape(-1)[0]

    return float(fps)


def get_stand_qpos(model):
    if model.nkey > 0:
        return model.key_qpos[0].copy()

    qpos = model.qpos0.copy()
    qpos[3] = 1.0
    return qpos


def get_joint_qpos_address(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        return None

    return int(model.jnt_qposadr[joint_id])


def set_dof_pose(model, qpos, dof_frame):
    used = 0
    missing = []

    joint_count = min(len(dof_frame), len(OPENHE_23_DOF_JOINT_NAMES))

    for i in range(joint_count):
        joint_name = OPENHE_23_DOF_JOINT_NAMES[i]
        qpos_address = get_joint_qpos_address(model, joint_name)

        if qpos_address is None:
            missing.append(joint_name)
            continue

        qpos[qpos_address] = float(dof_frame[i])
        used += 1

    return qpos, used, missing


def print_summary(repo_file, local_path, motion_key, motion, start_frame, end_frame):
    root = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    contact = np.asarray(motion["contact_mask"]) if "contact_mask" in motion else None
    fps = get_fps(motion)

    root_segment = root[start_frame:end_frame]
    delta = root_segment[-1] - root_segment[0]
    xy_disp = float(np.linalg.norm(delta[:2]))
    duration = (end_frame - start_frame) / fps
    avg_speed = xy_disp / duration

    print()
    print("OpenHE G1 segment loaded")
    print("Repo file:", repo_file)
    print("Local path:", local_path)
    print("Motion key:", motion_key)
    print("Keys:", sorted(list(motion.keys())))
    print("FPS:", fps)
    print("Root shape:", root.shape)
    print("DOF shape:", dof.shape)
    if contact is not None:
        print("Contact shape:", contact.shape)
    print()
    print("Segment:", start_frame, "to", end_frame)
    print("Duration:", round(duration, 4), "sec")
    print("dx:", round(float(delta[0]), 4))
    print("dy:", round(float(delta[1]), 4))
    print("dz:", round(float(delta[2]), 4))
    print("xy displacement:", round(xy_disp, 4), "m")
    print("avg xy speed:", round(avg_speed, 4), "m/s")
    print()
    print("23-DOF mapping used:")
    for i, name in enumerate(OPENHE_23_DOF_JOINT_NAMES):
        print(f"{i:02d}: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_file", type=str, default=DEFAULT_REPO_FILE)
    parser.add_argument("--local_dir", type=str, default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--model_xml", type=str, default=str(DEFAULT_MODEL_XML))

    parser.add_argument("--start_frame", type=int, default=2820)
    parser.add_argument("--end_frame", type=int, default=3120)

    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--use_root_xy", action="store_true")
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--root_xy_scale", type=float, default=1.0)

    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    model_xml = Path(args.model_xml)

    if not model_xml.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_xml}")

    local_path = download_file(args.repo_file, local_dir)
    motion_key, motion = load_motion(local_path)

    root = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    fps = get_fps(motion)

    start_frame = max(0, args.start_frame)
    end_frame = min(args.end_frame, len(dof), len(root))

    if end_frame <= start_frame + 1:
        raise ValueError("Invalid start/end frame range.")

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    stand_qpos = get_stand_qpos(model)

    print_summary(
        repo_file=args.repo_file,
        local_path=local_path,
        motion_key=motion_key,
        motion=motion,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    root_start = root[start_frame].copy()

    qpos_test = stand_qpos.copy()
    qpos_test, used, missing = set_dof_pose(model, qpos_test, dof[start_frame])

    print()
    print("Mapped joints used:", used)
    print("Missing joints:", missing)
    print()
    print("Viewer opening...")
    print("Close the MuJoCo viewer window to stop.")
    print()

    frame = start_frame
    frame_time = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            qpos = stand_qpos.copy()

            if args.use_root_xy:
                root_frame = root[frame]
                qpos[0] = (root_frame[0] - root_start[0]) * args.root_xy_scale
                qpos[1] = (root_frame[1] - root_start[1]) * args.root_xy_scale
            else:
                qpos[0] = 0.0
                qpos[1] = 0.0

            qpos[2] = float(stand_qpos[2] + args.height_offset)

            qpos[3] = 1.0
            qpos[4] = 0.0
            qpos[5] = 0.0
            qpos[6] = 0.0

            qpos, used, missing = set_dof_pose(model, qpos, dof[frame])

            data.qpos[:] = qpos
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            if (frame - start_frame) % 30 == 0:
                local_frame = frame - start_frame
                print(
                    f"frame={frame}, "
                    f"local={local_frame}/{end_frame - start_frame}, "
                    f"time={local_frame / fps:.2f}s, "
                    f"x={data.qpos[0]:.3f}, "
                    f"y={data.qpos[1]:.3f}, "
                    f"z={data.qpos[2]:.3f}"
                )

            frame += 1

            if frame >= end_frame:
                if args.loop:
                    frame = start_frame
                else:
                    break

            time.sleep(frame_time * args.slowmo)

    print()
    print("Visualization finished.")


if __name__ == "__main__":
    main()
