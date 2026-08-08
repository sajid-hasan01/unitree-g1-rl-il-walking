from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


CONTROLLED_15_JOINTS: List[str] = [
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


@dataclass
class StageConfig:
    name: str
    max_steps: int
    target_vx: float
    target_clearance: float
    period_steps: int
    assistive_prior_scale: float
    allow_forward: bool


STAGE_CONFIGS: Dict[str, StageConfig] = {
    "balance": StageConfig("balance", 600, 0.0, 0.0, 120, 0.0, False),
    "right_lift": StageConfig("right_lift", 600, 0.0, 0.045, 120, 0.25, False),
    "left_lift": StageConfig("left_lift", 600, 0.0, 0.045, 120, 0.25, False),
    "alt_lift": StageConfig("alt_lift", 700, 0.0, 0.040, 140, 0.22, False),
    "step_in_place": StageConfig("step_in_place", 800, 0.0, 0.035, 130, 0.18, False),
    "tiny_walk": StageConfig("tiny_walk", 900, -0.035, 0.032, 120, 0.15, True),
}


class G1SolutionEnv(gym.Env):
    """
    Curriculum-based Unitree G1 lower-body control environment.

    It avoids the previous reward-patch loop by training simpler skills first:
    balance -> single-foot lift -> alternating lift -> stepping in place -> tiny walking.

    Reward has exactly five grouped terms:
    upright_height, swing_clearance, support_stability, velocity_drift, smoothness.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml",
        dataset_path: str = "datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz",
        stage: str = "balance",
        action_scale: float = 0.55,
        frame_skip: int = 5,
        randomize_reset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if stage not in STAGE_CONFIGS:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {list(STAGE_CONFIGS)}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.stage_name = stage
        self.stage_cfg = STAGE_CONFIGS[stage]
        self.action_scale = float(action_scale)
        self.frame_skip = int(frame_skip)
        self.randomize_reset = bool(randomize_reset)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.np_random = np.random.default_rng(seed)

        self.dataset = np.load(dataset_path, allow_pickle=True)
        self.reference_joint_pos = np.asarray(self.dataset["joint_pos_15"], dtype=np.float32)

        self.controlled_joint_names = list(CONTROLLED_15_JOINTS)
        self.joint_ids: List[int] = []
        self.qpos_adrs: List[int] = []
        self.qvel_adrs: List[int] = []
        self.actuator_ids: List[int] = []
        self._build_joint_and_actuator_maps()

        self.left_foot_site_id = self._required_site("left_foot")
        self.right_foot_site_id = self._required_site("right_foot")

        self.stand_qpos = self._get_stand_qpos()
        self.stand_joint_pos = np.array([self.stand_qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)

        self.control_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.control_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        self.joint_residual_scale = np.array(
            [0.75, 0.35, 0.20, 1.00, 0.65, 0.35,
             0.75, 0.35, 0.20, 1.00, 0.65, 0.35,
             0.12, 0.12, 0.12],
            dtype=np.float32,
        )

        self.prev_action = np.zeros(15, dtype=np.float32)
        self.prev_left_foot = np.zeros(3, dtype=np.float64)
        self.prev_right_foot = np.zeros(3, dtype=np.float64)
        self.episode_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)

    def _required_joint(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Required joint not found: {name}")
        return int(jid)

    def _required_site(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise RuntimeError(f"Required site not found: {name}")
        return int(sid)

    def _build_joint_and_actuator_maps(self) -> None:
        joint_to_actuator: Dict[int, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                joint_to_actuator[joint_id] = int(actuator_id)

        for joint_name in self.controlled_joint_names:
            joint_id = self._required_joint(joint_name)
            if joint_id not in joint_to_actuator:
                raise RuntimeError(f"No actuator found for controlled joint: {joint_name}")
            self.joint_ids.append(joint_id)
            self.qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_adrs.append(int(self.model.jnt_dofadr[joint_id]))
            self.actuator_ids.append(joint_to_actuator[joint_id])

    def _get_stand_qpos(self) -> np.ndarray:
        if self.model.nkey > 0:
            return np.array(self.model.key_qpos[0], dtype=np.float64).copy()
        qpos = np.array(self.model.qpos0, dtype=np.float64).copy()
        if self.model.nq >= 7:
            qpos[2] = 0.79
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return qpos

    def _phase(self) -> float:
        return (self.episode_step % self.stage_cfg.period_steps) / float(self.stage_cfg.period_steps)

    def _desired_swing(self) -> Tuple[bool, bool]:
        stage = self.stage_name
        phase = self._phase()
        if stage == "balance":
            return False, False
        if stage == "right_lift":
            return False, True
        if stage == "left_lift":
            return True, False
        if 0.12 <= phase < 0.45:
            return False, True
        if 0.57 <= phase < 0.90:
            return True, False
        return False, False

    def _assistive_prior(self) -> np.ndarray:
        left_swing, right_swing = self._desired_swing()
        prior = np.zeros(15, dtype=np.float32)
        scale = self.stage_cfg.assistive_prior_scale
        if left_swing:
            prior[0] += 0.55 * scale
            prior[1] += -0.25 * scale
            prior[3] += 0.80 * scale
            prior[4] += 0.45 * scale
            prior[5] += -0.12 * scale
            prior[9] += 0.08 * scale
            prior[10] += -0.05 * scale
        if right_swing:
            prior[6] += 0.55 * scale
            prior[7] += 0.25 * scale
            prior[9] += 0.80 * scale
            prior[10] += 0.45 * scale
            prior[11] += 0.12 * scale
            prior[3] += 0.08 * scale
            prior[4] += -0.05 * scale
        return prior

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.episode_step = 0
        self.prev_action.fill(0.0)
        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        if self.randomize_reset:
            noise = self.np_random.normal(loc=0.0, scale=0.015, size=15)
            for i, qadr in enumerate(self.qpos_adrs):
                self.data.qpos[qadr] += noise[i]
        self._set_all_actuator_targets(self.stand_joint_pos)
        mujoco.mj_forward(self.model, self.data)
        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        prior = self._assistive_prior()
        residual = self.action_scale * self.joint_residual_scale * action
        target = self.stand_joint_pos + prior + residual
        target = np.clip(target, self.control_low, self.control_high)
        self._set_all_actuator_targets(target)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        reward, reward_terms = self._compute_reward(action, info)
        info.update(reward_terms)
        terminated = self._is_terminated(info)
        truncated = self.episode_step >= self.stage_cfg.max_steps
        self.prev_action = action.copy()
        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()
        self.episode_step += 1
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _set_all_actuator_targets(self, controlled_targets: np.ndarray) -> None:
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                qadr = int(self.model.jnt_qposadr[joint_id])
                if qadr < self.model.nq:
                    self.data.ctrl[actuator_id] = self.stand_qpos[qadr]
        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = controlled_targets[i]

    def _root_up_z(self) -> float:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return float(mat[8])

    def _foot_metrics(self) -> Dict[str, float]:
        dt = max(float(self.model.opt.timestep * self.frame_skip), 1e-6)
        left_pos = self.data.site_xpos[self.left_foot_site_id].copy()
        right_pos = self.data.site_xpos[self.right_foot_site_id].copy()
        floor_z = min(float(left_pos[2]), float(right_pos[2]))
        left_clearance = max(0.0, float(left_pos[2]) - floor_z)
        right_clearance = max(0.0, float(right_pos[2]) - floor_z)
        left_vel = (left_pos - self.prev_left_foot) / dt
        right_vel = (right_pos - self.prev_right_foot) / dt
        left_slip = float(np.linalg.norm(left_vel[:2]))
        right_slip = float(np.linalg.norm(right_vel[:2]))
        contact_threshold = 0.025
        return {
            "left_foot_z": float(left_pos[2]),
            "right_foot_z": float(right_pos[2]),
            "left_foot_clearance": float(left_clearance),
            "right_foot_clearance": float(right_clearance),
            "left_foot_slip": left_slip,
            "right_foot_slip": right_slip,
            "left_contact": bool(left_clearance <= contact_threshold),
            "right_contact": bool(right_clearance <= contact_threshold),
        }

    def _get_obs(self) -> np.ndarray:
        up_z = self._root_up_z()
        root_height = float(self.data.qpos[2])
        root_vel = np.asarray(self.data.qvel[0:3], dtype=np.float32)
        root_angvel = np.asarray(self.data.qvel[3:6], dtype=np.float32)
        joint_pos = np.array([self.data.qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[vadr] for vadr in self.qvel_adrs], dtype=np.float32)
        joint_error = joint_pos - self.stand_joint_pos
        foot = self._foot_metrics()
        phase = self._phase()
        left_swing, right_swing = self._desired_swing()
        obs = np.concatenate([
            np.array([root_height, up_z, root_vel[0], root_vel[1], root_vel[2]], dtype=np.float32),
            root_angvel.astype(np.float32),
            joint_error.astype(np.float32),
            0.1 * joint_vel.astype(np.float32),
            np.array([
                foot["left_foot_clearance"], foot["right_foot_clearance"],
                foot["left_foot_slip"], foot["right_foot_slip"],
                math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase),
                float(left_swing), float(right_swing), self.stage_cfg.target_vx,
            ], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, float]:
        foot = self._foot_metrics()
        left_swing, right_swing = self._desired_swing()
        left_expected_contact = not left_swing
        right_expected_contact = not right_swing
        left_match = bool(foot["left_contact"]) == bool(left_expected_contact)
        right_match = bool(foot["right_contact"]) == bool(right_expected_contact)
        info = {
            "stage": self.stage_name,
            "episode_step": int(self.episode_step),
            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "up_z": self._root_up_z(),
            "phase": float(self._phase()),
            "left_expected_contact": bool(left_expected_contact),
            "right_expected_contact": bool(right_expected_contact),
            "contact_accuracy": 0.5 * (float(left_match) + float(right_match)),
            "target_vx": float(self.stage_cfg.target_vx),
            "target_clearance": float(self.stage_cfg.target_clearance),
        }
        info.update(foot)
        return info

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        left_swing, right_swing = self._desired_swing()
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])
        y_pos = float(info["y_position"])
        left_clear = float(info["left_foot_clearance"])
        right_clear = float(info["right_foot_clearance"])
        left_slip = float(info["left_foot_slip"])
        right_slip = float(info["right_foot_slip"])

        height_reward = math.exp(-35.0 * (height - 0.79) ** 2)
        upright_height = 2.0 * max(up_z, 0.0) + height_reward

        target_clear = self.stage_cfg.target_clearance
        if left_swing:
            swing_clearance = 3.0 * math.exp(-260.0 * (left_clear - target_clear) ** 2)
        elif right_swing:
            swing_clearance = 3.0 * math.exp(-260.0 * (right_clear - target_clear) ** 2)
        else:
            swing_clearance = math.exp(-180.0 * (left_clear**2 + right_clear**2))

        contact_accuracy = float(info["contact_accuracy"])
        support_slip = 0.0
        if not left_swing:
            support_slip += left_slip
        if not right_swing:
            support_slip += right_slip
        support_stability = 2.0 * contact_accuracy - 1.25 * min(support_slip, 2.0)

        target_vx = self.stage_cfg.target_vx
        vx_reward = math.exp(-18.0 * (x_vel - target_vx) ** 2)
        drift_penalty = 2.0 * abs(y_pos) + abs(y_vel)
        velocity_drift = 1.2 * vx_reward - drift_penalty

        action_delta = float(np.mean(np.square(action - self.prev_action)))
        action_energy = float(np.mean(np.square(action)))
        smoothness = -0.05 * action_energy - 0.10 * action_delta

        reward = upright_height + swing_clearance + support_stability + velocity_drift + smoothness
        return reward, {
            "reward_upright_height": float(upright_height),
            "reward_swing_clearance": float(swing_clearance),
            "reward_support_stability": float(support_stability),
            "reward_velocity_drift": float(velocity_drift),
            "reward_smoothness": float(smoothness),
            "reward_total": float(reward),
            "support_slip": float(support_slip),
        }

    def _is_terminated(self, info: Dict[str, float]) -> bool:
        if float(info["base_height"]) < 0.55 or float(info["base_height"]) > 1.05:
            return True
        if float(info["up_z"]) < 0.55:
            return True
        if abs(float(info["y_position"])) > 0.65:
            return True
        if not self.stage_cfg.allow_forward and abs(float(info["x_position"])) > 0.45:
            return True
        if self.stage_cfg.allow_forward and float(info["x_position"]) > 0.35:
            return True
        return False

    def close(self):
        pass
