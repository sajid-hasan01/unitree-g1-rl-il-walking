from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
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


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class G1PhaseBCResidualEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        dataset_path: str = "experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
        phase_bc_policy_path: str = "experiments/amass_b3_15dof_il_phase_bc/models/g1_amass_b3_phase_bc_15dof_best.pt",
        mujoco_model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml",
        render_mode: str | None = None,
        start_frame: int = 90,
        reverse_time: bool = False,
        gait_scale: float = 0.20,
        residual_scale: float = 0.08,
        target_smoothing: float = 0.08,
        transition_steps: int = 180,
        max_episode_steps: int = 700,
        target_velocity: float = -0.08,
        fall_height: float = 0.45,
        fall_up_z: float = 0.50,
        random_start: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.phase_bc_policy_path = Path(phase_bc_policy_path)
        self.mujoco_model_path = Path(mujoco_model_path)
        self.render_mode = render_mode

        self.start_frame = int(start_frame)
        self.reverse_time = bool(reverse_time)
        self.gait_scale = float(gait_scale)
        self.residual_scale = float(residual_scale)
        self.target_smoothing = float(target_smoothing)
        self.transition_steps = int(transition_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.target_velocity = float(target_velocity)
        self.fall_height = float(fall_height)
        self.fall_up_z = float(fall_up_z)
        self.random_start = bool(random_start)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        if not self.phase_bc_policy_path.exists():
            raise FileNotFoundError(f"Phase-BC policy not found: {self.phase_bc_policy_path}")

        if not self.mujoco_model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {self.mujoco_model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_dataset_and_policy()

        self.model = mujoco.MjModel.from_xml_path(str(self.mujoco_model_path))
        self.data = mujoco.MjData(self.model)

        if self.model.nkey > 0:
            self.stand_qpos = self.model.key_qpos[0].copy()
        else:
            self.stand_qpos = np.zeros(self.model.nq, dtype=np.float64)
            self.stand_qpos[3] = 1.0
            self.stand_qpos[2] = 0.79

        self.joint_to_actuator = self._joint_to_actuator_map()
        self.qpos_addresses = self._joint_qpos_addresses(self.joint_names)
        self.qvel_addresses = self._joint_qvel_addresses(self.joint_names)
        self.controlled_actuator_ids = [self.joint_to_actuator[name] for name in self.joint_names]

        self.stand_ctrl = self._make_ctrl_from_qpos(self.stand_qpos)
        self.stand_joint_values = np.array(
            [self.stand_qpos[qadr] for qadr in self.qpos_addresses],
            dtype=np.float64,
        )

        self.body_id = self._get_body_id()

        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = 1.0 / self.fps
        self.frame_skip = max(1, int(round(self.control_dt / self.sim_dt)))

        self.viewer = None

        self.step_count = 0
        self.phase_index = 0
        self.prev_action = np.zeros(15, dtype=np.float32)
        self.current_target = self.stand_joint_values.copy()
        self.previous_base_x = 0.0

        dummy_obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.joint_names),),
            dtype=np.float32,
        )

    def _load_dataset_and_policy(self) -> None:
        dataset = np.load(self.dataset_path, allow_pickle=True)

        self.joint_names = [str(x) for x in dataset["controlled_joint_names"]]
        self.num_reference_frames = int(np.asarray(dataset["joint_pos_15"]).shape[0])

        self.fps = 30.0
        if "fps" in dataset.files:
            fps_arr = np.asarray(dataset["fps"]).reshape(-1)
            if len(fps_arr) > 0:
                self.fps = float(fps_arr[0])

        checkpoint = load_checkpoint(self.phase_bc_policy_path, self.device)

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
        ).to(self.device)

        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        phase_features = build_phase_features(
            num_frames=self.num_reference_frames,
            num_harmonics=num_harmonics,
        )
        self.phase_features = phase_features.astype(np.float32)

        phase_norm = (phase_features - x_mean) / x_std

        with torch.no_grad():
            phase_tensor = torch.tensor(phase_norm, dtype=torch.float32, device=self.device)
            pred_norm = policy(phase_tensor).cpu().numpy()

        predicted = pred_norm * y_std + y_mean
        predicted = predicted.astype(np.float64)

        self.phase_bc_targets = self._rollout_sequence(
            predicted,
            start_frame=self.start_frame,
            reverse_time=self.reverse_time,
        )

        self.phase_features = self._rollout_sequence(
            self.phase_features,
            start_frame=self.start_frame,
            reverse_time=self.reverse_time,
        ).astype(np.float32)

    def _rollout_sequence(self, seq: np.ndarray, start_frame: int, reverse_time: bool) -> np.ndarray:
        start_frame = int(np.clip(start_frame, 0, len(seq) - 1))
        rolled = np.concatenate([seq[start_frame:], seq[:start_frame]], axis=0)

        if reverse_time:
            rolled = rolled[::-1].copy()

        return rolled

    def _joint_to_actuator_map(self) -> dict[str, int]:
        mapping: dict[str, int] = {}

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

            if joint_name is not None:
                mapping[joint_name] = actuator_id

        return mapping

    def _joint_qpos_addresses(self, joint_names: list[str]) -> list[int]:
        addresses = []

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            addresses.append(int(self.model.jnt_qposadr[joint_id]))

        return addresses

    def _joint_qvel_addresses(self, joint_names: list[str]) -> list[int]:
        addresses = []

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            addresses.append(int(self.model.jnt_dofadr[joint_id]))

        return addresses

    def _make_ctrl_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float64)

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            ctrl[actuator_id] = qpos[qadr]

        return ctrl

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        out = ctrl.copy()

        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                out[i] = np.clip(out[i], low, high)

        return out

    def _get_body_id(self) -> int:
        for name in ["pelvis", "torso_link", "waist_yaw_link", "base"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        return 1

    def _get_up_z(self) -> float:
        mat = self.data.xmat[self.body_id].reshape(3, 3)
        return float(mat[2, 2])

    def _get_joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[qadr] for qadr in self.qpos_addresses], dtype=np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[qadr] for qadr in self.qvel_addresses], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        base_z = np.array([float(self.data.qpos[2])], dtype=np.float32)
        up_z = np.array([self._get_up_z()], dtype=np.float32)

        root_vel = np.asarray(self.data.qvel[:6], dtype=np.float32)
        joint_pos = self._get_joint_pos()
        joint_vel = self._get_joint_vel()

        phase_features = self.phase_features[self.phase_index % len(self.phase_features)].astype(np.float32)
        phase_target = self.phase_bc_targets[self.phase_index % len(self.phase_bc_targets)].astype(np.float32)
        target_delta = (phase_target - self.stand_joint_values.astype(np.float32)).astype(np.float32)

        obs = np.concatenate(
            [
                base_z,
                up_z,
                root_vel,
                joint_pos,
                joint_vel,
                phase_features,
                target_delta,
                self.prev_action.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_target(self, target: np.ndarray) -> None:
        ctrl = self.stand_ctrl.copy()

        for act_id, value in zip(self.controlled_actuator_ids, target):
            ctrl[act_id] = float(value)

        self.data.ctrl[:] = self._clip_ctrl(ctrl)

    def _compute_target(self, action: np.ndarray) -> np.ndarray:
        phase_target = self.phase_bc_targets[self.phase_index % len(self.phase_bc_targets)]

        transition_alpha = smoothstep(self.step_count / float(max(self.transition_steps, 1)))

        gait_target = self.stand_joint_values + transition_alpha * self.gait_scale * (
            phase_target - self.stand_joint_values
        )

        residual = self.residual_scale * action.astype(np.float64)

        desired_target = gait_target + residual

        self.current_target = (
            (1.0 - self.target_smoothing) * self.current_target
            + self.target_smoothing * desired_target
        )

        return self.current_target.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.random_start:
            self.phase_index = int(self.np_random.integers(0, len(self.phase_bc_targets)))
        else:
            self.phase_index = 0

        self.step_count = 0
        self.prev_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.current_target = self.stand_joint_values.copy()

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._clip_ctrl(self.stand_ctrl)
        mujoco.mj_forward(self.model, self.data)

        for _ in range(80):
            self.data.ctrl[:] = self._clip_ctrl(self.stand_ctrl)
            mujoco.mj_step(self.model, self.data)

        self.previous_base_x = float(self.data.qpos[0])

        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 5.5
                self.viewer.cam.azimuth = 140.0
                self.viewer.cam.elevation = -20.0
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
            self.viewer.sync()

        obs = self._get_obs()
        info = self._get_info(action=np.zeros(len(self.joint_names), dtype=np.float32), reward=0.0)

        return obs, info

    def _get_info(self, action: np.ndarray, reward: float) -> dict[str, Any]:
        base_x = float(self.data.qpos[0])
        base_y = float(self.data.qpos[1])
        base_z = float(self.data.qpos[2])
        up_z = self._get_up_z()

        actual_q = self._get_joint_pos().astype(np.float64)
        ctrl_mae = float(np.mean(np.abs(actual_q - self.current_target)))

        return {
            "step": self.step_count,
            "phase_index": self.phase_index,
            "base_x": base_x,
            "base_y": base_y,
            "base_z": base_z,
            "up_z": up_z,
            "reward": float(reward),
            "ctrl_mae": ctrl_mae,
            "action_mean_abs": float(np.mean(np.abs(action))),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        previous_x = float(self.data.qpos[0])

        target = self._compute_target(action)
        self._apply_target(target)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

        base_x = float(self.data.qpos[0])
        base_y = float(self.data.qpos[1])
        base_z = float(self.data.qpos[2])
        up_z = self._get_up_z()

        vx = (base_x - previous_x) / max(self.frame_skip * self.sim_dt, 1e-6)

        height_target = float(self.stand_qpos[2])
        height_reward = math.exp(-((base_z - height_target) ** 2) / 0.02)
        velocity_reward = math.exp(-((vx - self.target_velocity) ** 2) / 0.08)

        lateral_penalty = 0.15 * abs(base_y)
        residual_penalty = 0.02 * float(np.mean(action * action))
        ctrl_penalty = 0.05 * float(np.mean(np.abs(self._get_joint_pos().astype(np.float64) - target)))

        reward = (
            0.50
            + 1.50 * up_z
            + 0.50 * height_reward
            + 0.30 * velocity_reward
            - lateral_penalty
            - residual_penalty
            - ctrl_penalty
        )

        terminated = bool(base_z < self.fall_height or up_z < self.fall_up_z)

        if terminated:
            reward -= 20.0

        self.step_count += 1
        self.phase_index = (self.phase_index + 1) % len(self.phase_bc_targets)
        self.prev_action = action.copy()

        truncated = bool(self.step_count >= self.max_episode_steps)

        obs = self._get_obs()
        info = self._get_info(action=action, reward=reward)
        info["vx"] = float(vx)
        info["terminated"] = terminated
        info["truncated"] = truncated

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
