import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer


MODEL_XML_PATH = os.path.join(
    "third_party",
    "mujoco_menagerie",
    "unitree_g1",
    "scene.xml",
)


class BCWalkingPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def get_joint_qpos_addresses(model, joint_names):
    qpos_addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")

        qpos_addresses.append(model.jnt_qposadr[joint_id])

    return qpos_addresses


def build_bc_observation(frame, total_frames):
    progress = frame / max(total_frames - 1, 1)
    phase = 2.0 * np.pi * progress

    obs = np.array(
        [
            np.sin(phase),
            np.cos(phase),
            progress,
        ],
        dtype=np.float32,
    )

    return obs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        type=str,
        default=os.path.join("models", "g1_bc_walking_policy.pt"),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(
            "datasets",
            "processed",
            "g1_amass_walking_il_15dof.npz",
        ),
    )

    parser.add_argument("--playback_speed", type=float, default=0.45)

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Positive value lifts the robot, negative value lowers it.",
    )

    parser.add_argument(
        "--forward_scale",
        type=float,
        default=1.0,
        help="Scales AMASS root forward movement.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.policy):
        raise FileNotFoundError(f"BC policy not found: {args.policy}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Processed IL dataset not found: {args.dataset}")

    if not os.path.exists(MODEL_XML_PATH):
        raise FileNotFoundError(f"MuJoCo model not found: {MODEL_XML_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.policy,
        map_location=device,
        weights_only=False,
    )

    dataset = np.load(args.dataset, allow_pickle=True)

    root_positions = dataset["root_positions"].astype(np.float32)
    fps = float(dataset["fps"][0])
    total_frames = root_positions.shape[0]

    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]

    obs_mean = checkpoint["obs_mean"].astype(np.float32)
    obs_std = checkpoint["obs_std"].astype(np.float32)

    action_mean = checkpoint["action_mean"].astype(np.float32)
    action_std = checkpoint["action_std"].astype(np.float32)

    joint_names = [str(name) for name in checkpoint["controlled_joint_names"]]

    policy = BCWalkingPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    joint_qpos_addresses = get_joint_qpos_addresses(model, joint_names)

    root_start = root_positions[0].copy()

    print("Evaluating BC walking policy")
    print("Policy:", args.policy)
    print("Dataset:", args.dataset)
    print("MuJoCo model:", MODEL_XML_PATH)
    print("Observation dim:", obs_dim)
    print("Action dim:", action_dim)
    print("Device:", device)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Height offset:", args.height_offset)
    print("Forward scale:", args.forward_scale)
    print("Root Z range in AMASS:", float(root_positions[:, 2].min()), "to", float(root_positions[:, 2].max()))
    print()

    frame_time = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0

        while viewer.is_running():
            frame_id = frame % total_frames

            raw_obs = build_bc_observation(
                frame=frame_id,
                total_frames=total_frames,
            )

            obs_norm = (raw_obs - obs_mean.squeeze()) / obs_std.squeeze()

            obs_tensor = torch.tensor(
                obs_norm,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                action_norm = policy(obs_tensor).cpu().numpy()[0]

            joint_targets = (
                action_norm * action_std.squeeze()
                + action_mean.squeeze()
            ).astype(np.float32)

            current_root = root_positions[frame_id]

            root_x = (current_root[1] - root_start[1]) * args.forward_scale
            root_y = 0.0
            root_z = current_root[2] + args.height_offset

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0

            # Floating base position
            data.qpos[0] = root_x
            data.qpos[1] = root_y
            data.qpos[2] = root_z

            # Upright floating base quaternion
            data.qpos[3] = 1.0
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = 0.0

            # Controlled 15 lower-body + waist joints
            for i, qpos_addr in enumerate(joint_qpos_addresses):
                data.qpos[qpos_addr] = joint_targets[i]

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame += 1

            time.sleep(frame_time / args.playback_speed)


if __name__ == "__main__":
    main()
