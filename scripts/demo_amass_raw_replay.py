from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def decode_names(arr: np.ndarray) -> list[str]:
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def choose_root_body_index(body_names: list[str]) -> int:
    preferred = ["pelvis", "torso", "trunk", "base", "waist"]
    lower_names = [n.lower() for n in body_names]

    for word in preferred:
        for i, name in enumerate(lower_names):
            if word in name:
                return i

    return 0


def find_main_motion_array(data: np.lib.npyio.NpzFile, model: mujoco.MjModel) -> tuple[str, np.ndarray, str]:
    keys = list(data.files)

    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == model.nq:
            return k, arr.astype(np.float64), "full_qpos"

    preferred_words = ["dof_positions", "jpos", "joint", "pose", "poses", "dof"]
    for preferred_key in preferred_words:
        for k in keys:
            arr = data[k]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            if preferred_key in k.lower() and arr.shape[1] == model.nu:
                return k, arr.astype(np.float64), "actuator_joint_pos"

    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= model.nu:
            return k, arr[:, : model.nu].astype(np.float64), "actuator_joint_pos"

    raise RuntimeError("Could not find usable motion array.")


def actuator_joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    qpos_addresses: list[int] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_addresses.append(qpos_adr)
    return qpos_addresses


def print_body_displacement_stats(body_names: list[str], body_positions: np.ndarray) -> None:
    print("\nBody displacement stats:")
    stats = []
    for i, name in enumerate(body_names):
        p0 = body_positions[0, i, :3]
        p1 = body_positions[-1, i, :3]
        delta = p1 - p0
        xy = float(np.linalg.norm(delta[:2]))
        stats.append((xy, i, name, delta))

    stats.sort(reverse=True, key=lambda x: x[0])

    for xy, i, name, delta in stats[:10]:
        print(
            f"  body[{i:02d}] {name:<30} "
            f"dx={delta[0]:+.3f} dy={delta[1]:+.3f} dz={delta[2]:+.3f} xy={xy:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay AMASS-retargeted G1 candidate in MuJoCo.")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")

    parser.add_argument("--sleep_time", type=float, default=0.025)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--root_motion_scale", type=float, default=0.45)
    parser.add_argument("--max_frames", type=int, default=900)
    parser.add_argument("--print_every", type=int, default=60)

    parser.add_argument("--use_body_root", action="store_true")
    parser.add_argument("--root_body_index", type=int, default=-1)
    parser.add_argument("--apply_root_z", action="store_true")
    parser.add_argument("--apply_root_quat", action="store_true")

    parser.add_argument("--cam_distance", type=float, default=5.0)
    parser.add_argument("--cam_azimuth", type=float, default=140.0)
    parser.add_argument("--cam_elevation", type=float, default=-20.0)
    parser.add_argument("--cam_lookat_z", type=float, default=0.75)

    args = parser.parse_args()

    npz_path = Path(args.npz)
    model_path = Path(args.model_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        stand_qpos = model.key_qpos[0].copy()
    else:
        stand_qpos = np.zeros(model.nq, dtype=np.float64)
        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79

    npz = np.load(npz_path, allow_pickle=True)

    print("=" * 100)
    print("AMASS G1 REPLAY WITH OPTIONAL BODY ROOT MOTION")
    print("File:", npz_path)
    print("MuJoCo model:", model_path)
    print("nq=", model.nq, "nv=", model.nv, "nu=", model.nu)
    print("=" * 100)

    print("Available keys:")
    for k in npz.files:
        arr = npz[k]
        print(f"  {k}: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")

    motion_key, motion, mode = find_main_motion_array(npz, model)
    print("\nSelected motion key:", motion_key)
    print("Motion mode:", mode)
    print("Motion shape:", motion.shape)

    fps = 120.0
    if "fps" in npz.files:
        fps_arr = np.asarray(npz["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])
    print("fps:", fps)

    body_positions = None
    body_rotations = None
    body_names = []

    if "body_positions" in npz.files and "body_names" in npz.files:
        body_positions = np.asarray(npz["body_positions"], dtype=np.float64)
        body_names = decode_names(npz["body_names"])
        print("\nBody names:")
        for i, name in enumerate(body_names):
            print(f"  body[{i:02d}] {name}")

        print_body_displacement_stats(body_names, body_positions)

    if "body_rotations" in npz.files:
        body_rotations = np.asarray(npz["body_rotations"], dtype=np.float64)

    root_body_index = args.root_body_index
    if root_body_index < 0 and body_names:
        root_body_index = choose_root_body_index(body_names)

    if body_names and root_body_index >= 0:
        print("\nChosen root body index:", root_body_index)
        print("Chosen root body name:", body_names[root_body_index])

    qpos_adrs = actuator_joint_qpos_addresses(model)

    num_frames = motion.shape[0]
    max_frames = min(args.max_frames, num_frames)

    root0 = None
    if body_positions is not None and root_body_index >= 0:
        root0 = body_positions[0, root_body_index, :3].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, args.cam_lookat_z]
        viewer.sync()

        shown = 0
        frame = 0

        while viewer.is_running() and shown < max_frames:
            qpos = stand_qpos.copy()

            if mode == "full_qpos":
                qpos[:] = motion[frame, : model.nq]
            elif mode == "actuator_joint_pos":
                joint_values = motion[frame, : model.nu]
                for i, qadr in enumerate(qpos_adrs):
                    qpos[qadr] = joint_values[i]
            else:
                raise RuntimeError(f"Unknown mode: {mode}")

            if args.use_body_root and body_positions is not None and root0 is not None:
                root_now = body_positions[frame, root_body_index, :3]
                delta = (root_now - root0) * args.root_motion_scale

                qpos[0] = stand_qpos[0] + delta[0]
                qpos[1] = stand_qpos[1] + delta[1]

                if args.apply_root_z:
                    qpos[2] = stand_qpos[2] + delta[2]
                    if qpos[2] < 0.45 or qpos[2] > 1.20:
                        qpos[2] = stand_qpos[2]
                else:
                    qpos[2] = stand_qpos[2]

                if args.apply_root_quat and body_rotations is not None:
                    quat = body_rotations[frame, root_body_index, :4].copy()
                    norm = np.linalg.norm(quat)
                    if norm > 1e-6:
                        qpos[3:7] = quat / norm

            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

            if shown % args.print_every == 0:
                print(
                    f"frame={frame:04d}/{num_frames} "
                    f"root=({data.qpos[0]:+.3f},{data.qpos[1]:+.3f},{data.qpos[2]:+.3f})"
                )

            time.sleep(args.sleep_time)

            frame = (frame + max(1, args.frame_stride)) % num_frames
            shown += 1

    print("\nReplay finished.")


if __name__ == "__main__":
    main()
