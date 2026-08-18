from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def decode_names(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def choose_root_body_index(body_names: list[str]) -> int:
    for word in ["pelvis", "torso", "trunk", "base", "waist"]:
        for i, name in enumerate(body_names):
            if word in name.lower():
                return i
    return 0


def actuator_joint_qpos_addresses(model: mujoco.MjModel) -> list[int]:
    addresses = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def rotation_2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--model_path", default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--root_motion_scale", type=float, default=0.45)
    parser.add_argument("--forward_axis", choices=["neg_x", "pos_x"], default="neg_x")
    parser.add_argument("--sleep_time", type=float, default=0.025)
    parser.add_argument("--max_frames", type=int, default=900)
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument("--cam_distance", type=float, default=6.0)
    parser.add_argument("--cam_azimuth", type=float, default=140.0)
    parser.add_argument("--cam_elevation", type=float, default=-20.0)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    model_path = Path(args.model_path)

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        stand_qpos = model.key_qpos[0].copy()
    else:
        stand_qpos = np.zeros(model.nq, dtype=np.float64)
        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79

    d = np.load(npz_path, allow_pickle=True)

    dof_positions = np.asarray(d["dof_positions"], dtype=np.float64)
    body_positions = np.asarray(d["body_positions"], dtype=np.float64)
    body_names = decode_names(d["body_names"])

    root_index = choose_root_body_index(body_names)
    root_name = body_names[root_index]

    qpos_adrs = actuator_joint_qpos_addresses(model)

    root_xy = body_positions[:, root_index, :2]
    raw_total_delta = root_xy[-1] - root_xy[0]
    raw_heading_angle = math.atan2(float(raw_total_delta[1]), float(raw_total_delta[0]))

    target_angle = math.pi if args.forward_axis == "neg_x" else 0.0
    align_theta = target_angle - raw_heading_angle
    rot = rotation_2d(align_theta)

    raw_disp = float(np.linalg.norm(raw_total_delta))
    duration = len(dof_positions) / float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else len(dof_positions) / 30.0

    print("=" * 100)
    print("AMASS ALIGNED REPLAY")
    print("File:", npz_path)
    print("Frames:", len(dof_positions))
    print("Duration:", f"{duration:.2f}s")
    print("Root body:", root_name)
    print("Raw dx/dy:", f"{raw_total_delta[0]:+.3f}", f"{raw_total_delta[1]:+.3f}")
    print("Raw displacement:", f"{raw_disp:.3f}m")
    print("Aligned forward axis:", args.forward_axis)
    print("Root motion scale:", args.root_motion_scale)
    print("=" * 100)

    max_frames = min(args.max_frames, len(dof_positions))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.sync()

        for frame in range(max_frames):
            if not viewer.is_running():
                break

            qpos = stand_qpos.copy()

            joint_values = dof_positions[frame, : model.nu]
            for i, qadr in enumerate(qpos_adrs):
                qpos[qadr] = joint_values[i]

            raw_delta = root_xy[frame] - root_xy[0]
            aligned_delta = rot @ raw_delta

            qpos[0] = stand_qpos[0] + args.root_motion_scale * aligned_delta[0]
            qpos[1] = stand_qpos[1] + args.root_motion_scale * aligned_delta[1]
            qpos[2] = stand_qpos[2]

            data.qpos[:] = qpos
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            if frame % args.print_every == 0:
                print(
                    f"frame={frame:04d}/{len(dof_positions)} "
                    f"root=({data.qpos[0]:+.3f},{data.qpos[1]:+.3f},{data.qpos[2]:+.3f})"
                )

            time.sleep(args.sleep_time)

    print("Replay finished.")


if __name__ == "__main__":
    main()
