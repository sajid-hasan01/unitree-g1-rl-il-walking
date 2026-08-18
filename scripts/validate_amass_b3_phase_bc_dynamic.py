from __future__ import annotations

import argparse
import csv
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


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_phase_features(num_frames: int, num_harmonics: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

    features = [progress]

    for k in range(1, num_harmonics + 1):
        angle = 2.0 * math.pi * k * progress
        features.append(np.sin(angle).astype(np.float32))
        features.append(np.cos(angle).astype(np.float32))

    return np.concatenate(features, axis=1).astype(np.float32)


def joint_to_actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    mapping: dict[str, int] = {}

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        if joint_name is not None:
            mapping[joint_name] = actuator_id

    return mapping


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")

        addresses.append(int(model.jnt_qposadr[joint_id]))

    return addresses


def make_neutral_ctrl_from_stand(model: mujoco.MjModel, stand_qpos: np.ndarray) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        ctrl[actuator_id] = stand_qpos[qadr]

    return ctrl


def clip_ctrl(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    clipped = ctrl.copy()

    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            low, high = model.actuator_ctrlrange[i]
            clipped[i] = np.clip(clipped[i], low, high)

    return clipped


def get_body_id(model: mujoco.MjModel) -> int:
    for name in ["pelvis", "torso_link", "waist_yaw_link", "base"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return body_id

    return 1


def get_up_z(data: mujoco.MjData, body_id: int) -> float:
    mat = data.xmat[body_id].reshape(3, 3)
    return float(mat[2, 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument(
        "--policy",
        default="experiments/amass_b3_15dof_il_phase_bc/models/g1_amass_b3_phase_bc_15dof_best.pt",
    )
    parser.add_argument(
        "--mujoco_model",
        default="third_party/mujoco_menagerie/unitree_g1/scene.xml",
    )
    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_il_phase_bc/reports/dynamic_validation.csv",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--target_smoothing", type=float, default=0.20)
    parser.add_argument("--warmup_steps", type=int, default=90)
    parser.add_argument("--sleep_time", type=float, default=0.0)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_up_z", type=float, default=0.50)
    parser.add_argument("--cam_distance", type=float, default=5.5)
    parser.add_argument("--cam_azimuth", type=float, default=140.0)
    parser.add_argument("--cam_elevation", type=float, default=-20.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    policy_path = Path(args.policy)
    mujoco_model_path = Path(args.mujoco_model)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    if not mujoco_model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {mujoco_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = np.load(dataset_path, allow_pickle=True)
    joint_pos_15 = np.asarray(dataset["joint_pos_15"], dtype=np.float32)
    joint_names = [str(x) for x in dataset["controlled_joint_names"]]

    fps = 30.0
    if "fps" in dataset.files:
        fps_arr = np.asarray(dataset["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])

    checkpoint = load_checkpoint(policy_path, device)

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

    pred_joint_pos_15 = (pred_norm * y_std + y_mean).astype(np.float64)

    model = mujoco.MjModel.from_xml_path(str(mujoco_model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        stand_qpos = model.key_qpos[0].copy()
    else:
        stand_qpos = np.zeros(model.nq, dtype=np.float64)
        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79

    joint_to_act = joint_to_actuator_map(model)
    qpos_addresses = joint_qpos_addresses(model, joint_names)
    body_id = get_body_id(model)

    controlled_actuator_ids = []

    for joint_name in joint_names:
        if joint_name not in joint_to_act:
            raise RuntimeError(f"No actuator found for joint: {joint_name}")
        controlled_actuator_ids.append(joint_to_act[joint_name])

    neutral_ctrl = make_neutral_ctrl_from_stand(model, stand_qpos)

    # Initialize robot in the first predicted pose, but leave root free.
    data.qpos[:] = stand_qpos
    for qadr, value in zip(qpos_addresses, pred_joint_pos_15[0]):
        data.qpos[qadr] = float(value)

    data.qvel[:] = 0.0
    data.ctrl[:] = neutral_ctrl
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)
    control_dt = 1.0 / fps
    frame_skip = max(1, int(round(control_dt / sim_dt)))

    print("=" * 100)
    print("DYNAMIC VALIDATION: AMASS B3 15-DOF PHASE-BC")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("Policy:", policy_path)
    print("MuJoCo model:", mujoco_model_path)
    print("CSV:", csv_path)
    print("Device:", device)
    print("Frames per cycle:", len(pred_joint_pos_15))
    print("Cycles:", args.cycles)
    print("FPS:", fps)
    print("MuJoCo timestep:", sim_dt)
    print("Control timestep:", control_dt)
    print("Frame skip:", frame_skip)
    print("Action scale:", args.action_scale)
    print("Target smoothing:", args.target_smoothing)
    print("Warmup steps:", args.warmup_steps)
    print("Controlled joints:", len(joint_names))
    print("Controlled actuators:", controlled_actuator_ids)
    print("Fall height:", args.fall_height)
    print("Fall up_z:", args.fall_up_z)
    print("=" * 100)

    rows = []
    current_target = pred_joint_pos_15[0].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.sync()

        # Warmup hold.
        warmup_ctrl = neutral_ctrl.copy()
        for act_id, value in zip(controlled_actuator_ids, pred_joint_pos_15[0]):
            warmup_ctrl[act_id] = float(value)
        warmup_ctrl = clip_ctrl(model, warmup_ctrl)

        for _ in range(args.warmup_steps):
            if not viewer.is_running():
                break
            data.ctrl[:] = warmup_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()
            if args.real_time:
                time.sleep(sim_dt)

        global_step = 0
        fallen = False

        total_control_steps = len(pred_joint_pos_15) * args.cycles

        for control_step in range(total_control_steps):
            if not viewer.is_running():
                break

            frame = control_step % len(pred_joint_pos_15)
            raw_target = pred_joint_pos_15[frame]

            # Optionally blend toward stand pose for safer dynamic tracking.
            safe_target = pred_joint_pos_15[0] + args.action_scale * (raw_target - pred_joint_pos_15[0])

            current_target = (
                (1.0 - args.target_smoothing) * current_target
                + args.target_smoothing * safe_target
            )

            ctrl = neutral_ctrl.copy()

            for act_id, value in zip(controlled_actuator_ids, current_target):
                ctrl[act_id] = float(value)

            ctrl = clip_ctrl(model, ctrl)
            data.ctrl[:] = ctrl

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
                global_step += 1

            viewer.sync()

            base_x = float(data.qpos[0])
            base_y = float(data.qpos[1])
            base_z = float(data.qpos[2])
            up_z = get_up_z(data, body_id)

            ctrl_error = 0.0
            actual_q = []
            for qadr in qpos_addresses:
                actual_q.append(float(data.qpos[qadr]))
            actual_q = np.array(actual_q, dtype=np.float64)
            ctrl_error = float(np.mean(np.abs(actual_q - current_target)))

            fallen = base_z < args.fall_height or up_z < args.fall_up_z

            row = {
                "control_step": control_step,
                "global_step": global_step,
                "cycle": control_step // len(pred_joint_pos_15),
                "frame": frame,
                "base_x": base_x,
                "base_y": base_y,
                "base_z": base_z,
                "up_z": up_z,
                "ctrl_mae": ctrl_error,
                "fallen": int(fallen),
            }
            rows.append(row)

            if control_step % args.print_every == 0 or fallen:
                print(
                    f"step={control_step:04d}/{total_control_steps} "
                    f"cycle={row['cycle']} frame={frame:03d} "
                    f"x={base_x:+.3f} y={base_y:+.3f} z={base_z:+.3f} "
                    f"up_z={up_z:+.3f} ctrl_mae={ctrl_error:.4f} "
                    f"fallen={fallen}"
                )

            if fallen:
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        final = rows[-1]
        max_x = max(row["base_x"] for row in rows)
        min_z = min(row["base_z"] for row in rows)
        min_up_z = min(row["up_z"] for row in rows)
        duration_sec = rows[-1]["global_step"] * sim_dt

        print()
        print("=" * 100)
        print("DYNAMIC VALIDATION SUMMARY")
        print("=" * 100)
        print("CSV saved:", csv_path)
        print("Duration simulated:", f"{duration_sec:.2f}s")
        print("Control steps completed:", len(rows))
        print("Final x:", f"{final['base_x']:+.3f}")
        print("Final y:", f"{final['base_y']:+.3f}")
        print("Final z:", f"{final['base_z']:+.3f}")
        print("Max x:", f"{max_x:+.3f}")
        print("Min base z:", f"{min_z:+.3f}")
        print("Min up_z:", f"{min_up_z:+.3f}")
        print("Fallen:", bool(final["fallen"]))
        print("=" * 100)


if __name__ == "__main__":
    main()
