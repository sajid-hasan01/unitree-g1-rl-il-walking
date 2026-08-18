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
    mapping = {}

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


def make_ctrl_from_qpos(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        ctrl[actuator_id] = qpos[qadr]

    return ctrl


def clip_ctrl(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    out = ctrl.copy()

    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            low, high = model.actuator_ctrlrange[i]
            out[i] = np.clip(out[i], low, high)

    return out


def get_body_id(model: mujoco.MjModel) -> int:
    for name in ["pelvis", "torso_link", "waist_yaw_link", "base"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return body_id
    return 1


def get_up_z(data: mujoco.MjData, body_id: int) -> float:
    mat = data.xmat[body_id].reshape(3, 3)
    return float(mat[2, 2])


def rollout_sequence(seq: np.ndarray, start_frame: int, reverse_time: bool) -> np.ndarray:
    start_frame = int(np.clip(start_frame, 0, len(seq) - 1))
    rolled = np.concatenate([seq[start_frame:], seq[:start_frame]], axis=0)

    if reverse_time:
        rolled = rolled[::-1].copy()

    return rolled


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

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--reverse_time", action="store_true")
    parser.add_argument("--cycles", type=int, default=1)

    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--target_smoothing", type=float, default=0.08)
    parser.add_argument("--stand_hold_steps", type=int, default=90)
    parser.add_argument("--transition_steps", type=int, default=150)

    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_up_z", type=float, default=0.50)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--sleep_time", type=float, default=0.0)

    parser.add_argument("--cam_distance", type=float, default=5.5)
    parser.add_argument("--cam_azimuth", type=float, default=140.0)
    parser.add_argument("--cam_elevation", type=float, default=-20.0)

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    policy_path = Path(args.policy)
    mujoco_model_path = Path(args.mujoco_model)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    if not mujoco_model_path.exists():
        raise FileNotFoundError(mujoco_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = np.load(dataset_path, allow_pickle=True)
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

    policy = PhaseBCPolicy(input_dim, action_dim, hidden_dims).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    num_frames = int(np.asarray(dataset["joint_pos_15"]).shape[0])

    phase_features = build_phase_features(num_frames=num_frames, num_harmonics=num_harmonics)
    phase_norm = (phase_features - x_mean) / x_std

    with torch.no_grad():
        phase_tensor = torch.tensor(phase_norm, dtype=torch.float32, device=device)
        pred_norm = policy(phase_tensor).cpu().numpy()

    pred_joint_pos_15 = (pred_norm * y_std + y_mean).astype(np.float64)
    pred_joint_pos_15 = rollout_sequence(
        seq=pred_joint_pos_15,
        start_frame=args.start_frame,
        reverse_time=args.reverse_time,
    )

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

    controlled_actuator_ids = []
    for joint_name in joint_names:
        if joint_name not in joint_to_act:
            raise RuntimeError(f"No actuator found for joint: {joint_name}")
        controlled_actuator_ids.append(joint_to_act[joint_name])

    stand_ctrl = make_ctrl_from_qpos(model, stand_qpos)
    stand_joint_values = np.array([stand_qpos[qadr] for qadr in qpos_addresses], dtype=np.float64)

    body_id = get_body_id(model)
    sim_dt = float(model.opt.timestep)
    control_dt = 1.0 / fps
    frame_skip = max(1, int(round(control_dt / sim_dt)))

    data.qpos[:] = stand_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = clip_ctrl(model, stand_ctrl)
    mujoco.mj_forward(model, data)

    print("=" * 100)
    print("DYNAMIC VALIDATION V2: STAND-TO-PHASE-BC GAIT")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("Policy:", policy_path)
    print("Start frame:", args.start_frame)
    print("Reverse time:", args.reverse_time)
    print("Cycles:", args.cycles)
    print("Action scale:", args.action_scale)
    print("Target smoothing:", args.target_smoothing)
    print("Stand hold steps:", args.stand_hold_steps)
    print("Transition steps:", args.transition_steps)
    print("FPS:", fps)
    print("Frame skip:", frame_skip)
    print("Controlled actuators:", controlled_actuator_ids)
    print("=" * 100)

    current_target = stand_joint_values.copy()
    total_control_steps = len(pred_joint_pos_15) * args.cycles

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.cam_distance
        viewer.cam.azimuth = args.cam_azimuth
        viewer.cam.elevation = args.cam_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.sync()

        # Hold stable standing first.
        for step in range(args.stand_hold_steps):
            if not viewer.is_running():
                break

            data.ctrl[:] = clip_ctrl(model, stand_ctrl)
            mujoco.mj_step(model, data)
            viewer.sync()

            if args.real_time:
                time.sleep(sim_dt)

        fallen = False
        global_step = 0

        for control_step in range(total_control_steps):
            if not viewer.is_running():
                break

            frame = control_step % len(pred_joint_pos_15)

            raw_pose = pred_joint_pos_15[frame]

            # Critical correction:
            # scale the gait around the MuJoCo stand pose, not around frame 0.
            gait_target = stand_joint_values + args.action_scale * (raw_pose - stand_joint_values)

            if control_step < args.transition_steps:
                alpha = control_step / float(max(args.transition_steps, 1))
                alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
                desired_target = (1.0 - alpha) * stand_joint_values + alpha * gait_target
            else:
                desired_target = gait_target

            current_target = (
                (1.0 - args.target_smoothing) * current_target
                + args.target_smoothing * desired_target
            )

            ctrl = stand_ctrl.copy()
            for act_id, value in zip(controlled_actuator_ids, current_target):
                ctrl[act_id] = float(value)

            data.ctrl[:] = clip_ctrl(model, ctrl)

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
                global_step += 1

            viewer.sync()

            base_x = float(data.qpos[0])
            base_y = float(data.qpos[1])
            base_z = float(data.qpos[2])
            up_z = get_up_z(data, body_id)

            actual_q = np.array([float(data.qpos[qadr]) for qadr in qpos_addresses], dtype=np.float64)
            ctrl_mae = float(np.mean(np.abs(actual_q - current_target)))

            fallen = base_z < args.fall_height or up_z < args.fall_up_z

            if control_step % args.print_every == 0 or fallen:
                print(
                    f"step={control_step:04d}/{total_control_steps} "
                    f"frame={frame:03d} "
                    f"x={base_x:+.3f} y={base_y:+.3f} z={base_z:+.3f} "
                    f"up_z={up_z:+.3f} ctrl_mae={ctrl_mae:.4f} "
                    f"fallen={fallen}"
                )

            if fallen:
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    print()
    print("=" * 100)
    print("DYNAMIC VALIDATION V2 SUMMARY")
    print("=" * 100)
    print("Control steps completed:", control_step + 1)
    print("Global sim steps:", global_step)
    print("Final x:", f"{base_x:+.3f}")
    print("Final y:", f"{base_y:+.3f}")
    print("Final z:", f"{base_z:+.3f}")
    print("Final up_z:", f"{up_z:+.3f}")
    print("Fallen:", fallen)
    print("=" * 100)


if __name__ == "__main__":
    main()
