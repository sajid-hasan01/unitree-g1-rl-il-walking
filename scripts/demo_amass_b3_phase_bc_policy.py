from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


class PhaseBCPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_dim))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_phase_features(num_frames: int, num_harmonics: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

    features = [progress]

    for k in range(1, num_harmonics + 1):
        angle = 2.0 * math.pi * k * progress
        features.append(np.sin(angle).astype(np.float32))
        features.append(np.cos(angle).astype(np.float32))

    return np.concatenate(features, axis=1).astype(np.float32)


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        if joint_id < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {joint_name}")

        addresses.append(int(model.jnt_qposadr[joint_id]))

    return addresses


def apply_joint_positions(qpos: np.ndarray, qpos_addresses: list[int], values: np.ndarray) -> None:
    for value, qadr in zip(values, qpos_addresses):
        qpos[qadr] = float(value)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_il_phase_bc/models/g1_amass_b3_phase_bc_15dof_best.pt",
    )
    parser.add_argument(
        "--mujoco_model",
        default="third_party/mujoco_menagerie/unitree_g1/scene.xml",
    )
    parser.add_argument("--mode", choices=["reference", "predicted"], default="predicted")
    parser.add_argument("--sleep_time", type=float, default=0.025)
    parser.add_argument("--max_frames", type=int, default=229)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--cam_distance", type=float, default=6.0)
    parser.add_argument("--cam_azimuth", type=float, default=140.0)
    parser.add_argument("--cam_elevation", type=float, default=-20.0)
    parser.add_argument("--use_root_z", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    mujoco_model_path = Path(args.mujoco_model)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not mujoco_model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {mujoco_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = np.load(dataset_path, allow_pickle=True)

    joint_pos_15 = np.asarray(dataset["joint_pos_15"], dtype=np.float32)
    root_positions = np.asarray(dataset["root_positions"], dtype=np.float32)
    joint_names = [str(x) for x in dataset["controlled_joint_names"]]

    fps = 30.0
    if "fps" in dataset.files:
        fps_arr = np.asarray(dataset["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])

    checkpoint = load_checkpoint(model_path, device)

    input_dim = int(checkpoint["input_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dims = list(checkpoint["hidden_dims"])
    num_harmonics = int(checkpoint["num_harmonics"])

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)

    policy = PhaseBCPolicy(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    phase_features = build_phase_features(
        num_frames=len(joint_pos_15),
        num_harmonics=num_harmonics,
    )

    phase_features_norm = (phase_features - x_mean) / x_std

    with torch.no_grad():
        x_tensor = torch.tensor(phase_features_norm, dtype=torch.float32, device=device)
        pred_norm = policy(x_tensor).cpu().numpy()

    pred_joint_pos_15 = (pred_norm * y_std + y_mean).astype(np.float32)

    model = mujoco.MjModel.from_xml_path(str(mujoco_model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        stand_qpos = model.key_qpos[0].copy()
    else:
        stand_qpos = np.zeros(model.nq, dtype=np.float64)
        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79

    qpos_addresses = joint_qpos_addresses(model, joint_names)

    total_frames = min(args.max_frames, len(joint_pos_15))

    print("=" * 100)
    print("AMASS B3 15-DOF PHASE-BC VISUAL EVALUATION")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("Model:", model_path)
    print("MuJoCo model:", mujoco_model_path)
    print("Mode:", args.mode)
    print("Device:", device)
    print("Frames:", total_frames)
    print("FPS:", fps)
    print("Input dim:", input_dim)
    print("Action dim:", action_dim)
    print("Num harmonics:", num_harmonics)
    print("Best epoch:", checkpoint.get("best_epoch", "unknown"))
    print("Best val MSE:", checkpoint.get("best_val_mse_rad2", "unknown"))
    print("=" * 100)

    errors = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.sync()

        for frame in range(total_frames):
            if not viewer.is_running():
                break

            if args.mode == "reference":
                display_q = joint_pos_15[frame]
            else:
                display_q = pred_joint_pos_15[frame]

            qpos = stand_qpos.copy()
            apply_joint_positions(qpos, qpos_addresses, display_q)

            if frame < len(root_positions):
                qpos[0] = stand_qpos[0] + float(root_positions[frame, 0])
                qpos[1] = stand_qpos[1] + float(root_positions[frame, 1])

                if args.use_root_z:
                    qpos[2] = stand_qpos[2] + float(root_positions[frame, 2])
                else:
                    qpos[2] = stand_qpos[2]

            data.qpos[:] = qpos
            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            ref_q = joint_pos_15[frame]
            err = display_q - ref_q
            mse = float(np.mean(err * err))
            mae = float(np.mean(np.abs(err)))
            max_abs = float(np.max(np.abs(err)))
            errors.append((mse, mae, max_abs))

            if frame % args.print_every == 0:
                print(
                    f"frame={frame:04d}/{total_frames} "
                    f"root=({data.qpos[0]:+.3f},{data.qpos[1]:+.3f},{data.qpos[2]:+.3f}) "
                    f"mse={mse:.8f} mae={mae:.8f} max_abs={max_abs:.8f}"
                )

            time.sleep(args.sleep_time)

    if errors:
        mse_values = np.array([x[0] for x in errors], dtype=np.float64)
        mae_values = np.array([x[1] for x in errors], dtype=np.float64)
        max_values = np.array([x[2] for x in errors], dtype=np.float64)

        print()
        print("=" * 100)
        print("PHASE-BC VISUAL EVALUATION SUMMARY")
        print("=" * 100)
        print("Mode:", args.mode)
        print("Mean MSE:", f"{np.mean(mse_values):.8f}")
        print("Mean MAE:", f"{np.mean(mae_values):.8f}")
        print("Max MSE:", f"{np.max(mse_values):.8f}")
        print("Max MAE:", f"{np.max(mae_values):.8f}")
        print("Max Abs Error:", f"{np.max(max_values):.8f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
