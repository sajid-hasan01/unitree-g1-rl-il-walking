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


@dataclass(frozen=True)
class StageConfig:
    name: str
    max_steps: int
    target_x_velocity: float
    target_y_offset: float
    target_clearance: float
    phase_period: int
    teacher_scale: float
    allow_x_motion: bool


STAGE_CONFIGS: Dict[str, StageConfig] = {
    # Already solved primitive. Keep it strict and stable.
    "balance": StageConfig(
        name="balance",
        max_steps=700,
        target_x_velocity=0.0,
        target_y_offset=0.0,
        target_clearance=0.0,
        phase_period=160,
        teacher_scale=0.0,
        allow_x_motion=False,
    ),

    # New missing primitives:
    # Both feet stay down. The policy learns to tolerate / stabilize a lateral COM shift.
    "shift_left": StageConfig(
        name="shift_left",
        max_steps=700,
        target_x_velocity=0.0,
        target_y_offset=0.035,
        target_clearance=0.0,
        phase_period=180,
        teacher_scale=0.30,
        allow_x_motion=False,
    ),
    "shift_right": StageConfig(
        name="shift_right",
        max_steps=750,
        target_x_velocity=0.0,
        target_y_offset=-0.025,
        target_clearance=0.0,
        phase_period=220,
        teacher_scale=0.22,
        allow_x_motion=False,
    ),

    # Single-foot lift only after the COM is shifted toward the support foot.
    "right_lift": StageConfig(
        name="right_lift",
        max_steps=800,
        target_x_velocity=0.0,
        target_y_offset=0.030,
        target_clearance=0.030,
        phase_period=260,
        teacher_scale=0.30,
        allow_x_motion=False,
    ),
    "left_lift": StageConfig(
        name="left_lift",
        max_steps=800,
        target_x_velocity=0.0,
        target_y_offset=-0.028,
        target_clearance=0.028,
        phase_period=260,
        teacher_scale=0.26,
        allow_x_motion=False,
    ),

    # Alternating lift: slower and more conservative than the previous curriculum.
    "alt_lift": StageConfig(
        name="alt_lift",
        max_steps=800,
        target_x_velocity=0.0,
        target_y_offset=0.0,
        target_clearance=0.030,
        phase_period=240,
        teacher_scale=0.28,
        allow_x_motion=False,
    ),

    # Forward walking is delayed until alternating lift has succeeded.
    "tiny_walk": StageConfig(
        name="tiny_walk",
        max_steps=900,
        target_x_velocity=-0.025,
        target_y_offset=0.0,
        target_clearance=0.028,
        phase_period=220,
        teacher_scale=0.24,
        allow_x_motion=True,
    ),
}


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class G1ComShiftLiftEnv(gym.Env):
    """
    Unitree G1 COM-shift + foot-lift curriculum environment. V3: right/left lift use pre-shift, lift, and put-down phases.

    This environment directly attacks the failure seen in the previous curriculum:
    balance was solved, but right_lift/left_lift collapsed because the robot tried
    to lift a foot before shifting its center of mass over the support foot.

    Control:
      - 15 lower-body/waist joints.
      - Upper body/arms are held near standing keyframe.
      - The environment uses a small phase-conditioned teacher target.
      - PPO action is a residual on top of the teacher target.

    Reward has five grouped terms only:
      1. posture_reward
      2. com_reward
      3. foot_phase_reward
      4. support_reward
      5. residual_smoothness_reward
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml",
        dataset_path: str = "datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz",
        stage: str = "balance",
        action_scale: float = 0.28,
        teacher_scale_multiplier: float = 1.0,
        frame_skip: int = 5,
        randomize_reset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if stage not in STAGE_CONFIGS:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {list(STAGE_CONFIGS.keys())}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.stage_name = stage
        self.cfg = STAGE_CONFIGS[stage]
        self.action_scale = float(action_scale)
        self.teacher_scale_multiplier = float(teacher_scale_multiplier)
        self.frame_skip = int(frame_skip)
        self.randomize_reset = bool(randomize_reset)
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

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

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        # PPO residual is intentionally conservative. The teacher creates the rough skill,
        # PPO learns balance/stabilization/correction.
        self.residual_scale = np.array(
            [
                0.45, 0.25, 0.12, 0.55, 0.35, 0.20,
                0.45, 0.25, 0.12, 0.55, 0.35, 0.20,
                0.08, 0.08, 0.08,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)

        self.episode_step = 0
        self.prev_action = np.zeros(15, dtype=np.float32)
        self.prev_left_foot = np.zeros(3, dtype=np.float64)
        self.prev_right_foot = np.zeros(3, dtype=np.float64)
        self.last_teacher = np.zeros(15, dtype=np.float32)

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
            self.actuator_ids.append(int(joint_to_actuator[joint_id]))

    def _get_stand_qpos(self) -> np.ndarray:
        if self.model.nkey > 0:
            return np.array(self.model.key_qpos[0], dtype=np.float64).copy()

        qpos = np.array(self.model.qpos0, dtype=np.float64).copy()
        if self.model.nq >= 7:
            qpos[0] = 0.0
            qpos[1] = 0.0
            qpos[2] = 0.79
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return qpos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_step = 0
        self.prev_action.fill(0.0)

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0

        if self.randomize_reset:
            # Keep randomization tiny. Larger randomization broke early foot-lift stages.
            q_noise = self.rng.normal(0.0, 0.006, size=15)
            for i, qadr in enumerate(self.qpos_adrs):
                self.data.qpos[qadr] += q_noise[i]
            self.data.qvel[:6] += self.rng.normal(0.0, 0.004, size=6)

        self.last_teacher = self._teacher_offsets()
        self._set_actuator_targets(self.stand_joint_pos + self.last_teacher)

        mujoco.mj_forward(self.model, self.data)
        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        teacher = self._teacher_offsets()
        residual = self.action_scale * self.residual_scale * action
        target = self.stand_joint_pos + teacher + residual
        target = np.clip(target, self.ctrl_low, self.ctrl_high)

        self._set_actuator_targets(target)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        info = self._get_info()
        reward, reward_info = self._compute_reward(action, info)
        info.update(reward_info)

        terminated = self._terminated(info)
        truncated = self.episode_step >= self.cfg.max_steps

        obs = self._get_obs()

        self.prev_action = action.copy()
        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()
        self.last_teacher = teacher.copy()
        self.episode_step += 1

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _set_actuator_targets(self, controlled_targets: np.ndarray) -> None:
        # Hold all non-controlled actuators at stand pose.
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                qadr = int(self.model.jnt_qposadr[joint_id])
                if qadr < self.model.nq:
                    self.data.ctrl[actuator_id] = self.stand_qpos[qadr]

        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(controlled_targets[i])

    def _phase01(self) -> float:
        return (self.episode_step % self.cfg.phase_period) / float(self.cfg.phase_period)

    def _lift_envelope(self) -> float:
        phase = self._phase01()

        if self.stage_name in ("right_lift", "left_lift"):
            # No lift during pre-shift and recovery phases.
            if phase < 0.30 or phase >= 0.70:
                return 0.0
            local = (phase - 0.30) / 0.40
            if local < 0.50:
                return smoothstep(local / 0.50)
            return smoothstep((1.0 - local) / 0.50)

        # Alternating/tiny-walk stages keep the older smooth cyclic envelope.
        if phase < 0.50:
            return smoothstep(phase / 0.50)
        return smoothstep((1.0 - phase) / 0.50)

    def _alt_phase_side(self) -> Tuple[bool, bool]:
        phase = self._phase01()
        # right swing, double, left swing, double
        if 0.08 <= phase < 0.42:
            return False, True
        if 0.58 <= phase < 0.92:
            return True, False
        return False, False

    def _single_lift_active(self) -> bool:
        # Single-foot stages now contain three phases:
        #   0.00-0.30: both feet down, COM shifts over support foot
        #   0.30-0.70: swing foot lifts
        #   0.70-1.00: foot returns down while balance is recovered
        # This fixes the previous issue where right_lift expected swing contact
        # from step 0 before the COM shift had happened.
        phase = self._phase01()
        return 0.30 <= phase < 0.70

    def _desired_swing(self) -> Tuple[bool, bool]:
        if self.stage_name == "right_lift":
            return False, self._single_lift_active()
        if self.stage_name == "left_lift":
            return self._single_lift_active(), False
        if self.stage_name in ("alt_lift", "tiny_walk"):
            return self._alt_phase_side()
        return False, False

    def _target_y_offset(self) -> float:
        # The environment uses qpos[1] as a rough pelvis/COM lateral target.
        if self.stage_name in ("shift_left", "right_lift"):
            return self.cfg.target_y_offset
        if self.stage_name in ("shift_right", "left_lift"):
            return self.cfg.target_y_offset
        if self.stage_name in ("alt_lift", "tiny_walk"):
            left_swing, right_swing = self._desired_swing()
            if right_swing:
                return 0.035
            if left_swing:
                return -0.035
            return 0.0
        return 0.0

    def _teacher_offsets(self) -> np.ndarray:
        """
        A conservative analytic teacher:
        - Shift COM using hip-roll/ankle-roll/waist-roll pattern.
        - Then lift the swing foot using hip/knee/ankle pattern.
        PPO residual learns stabilization rather than discovering the whole pattern.
        """
        teacher = np.zeros(15, dtype=np.float32)
        scale = self.cfg.teacher_scale * self.teacher_scale_multiplier
        y_target = self._target_y_offset()

        # Lateral COM shift teacher.
        # Positive y target means shift toward left support, useful for right-foot lift.
        lateral = np.clip(y_target / 0.04, -1.0, 1.0) * scale

        # Roll/waist compensation. These signs are deliberately conservative.
        teacher[1] += -0.18 * lateral
        teacher[5] += 0.10 * lateral
        teacher[7] += -0.18 * lateral
        teacher[11] += 0.10 * lateral
        teacher[13] += 0.18 * lateral

        left_swing, right_swing = self._desired_swing()
        lift = self._lift_envelope()

        if left_swing:
            amp = scale * lift
            # Left swing leg lift.
            teacher[0] += 0.42 * amp
            teacher[1] += -0.18 * amp
            teacher[3] += 0.78 * amp
            teacher[4] += 0.36 * amp
            teacher[5] += -0.08 * amp
            # Right support leg soft bend and ankle.
            teacher[9] += 0.10 * amp
            teacher[10] += -0.05 * amp

        if right_swing:
            amp = scale * lift
            # Right swing leg lift.
            teacher[6] += 0.42 * amp
            teacher[7] += 0.18 * amp
            teacher[9] += 0.78 * amp
            teacher[10] += 0.36 * amp
            teacher[11] += 0.08 * amp
            # Left support leg soft bend and ankle.
            teacher[3] += 0.10 * amp
            teacher[4] += -0.05 * amp

        if self.stage_name == "tiny_walk":
            # Tiny forward walking bias in the known useful negative-X direction.
            left_swing, right_swing = self._desired_swing()
            if right_swing:
                teacher[6] += 0.04 * scale * lift
                teacher[0] += -0.02 * scale * lift
            elif left_swing:
                teacher[0] += 0.04 * scale * lift
                teacher[6] += -0.02 * scale * lift

        return teacher.astype(np.float32)

    def _root_up_z(self) -> float:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return float(mat[8])

    def _foot_metrics(self) -> Dict[str, float]:
        left = self.data.site_xpos[self.left_foot_site_id].copy()
        right = self.data.site_xpos[self.right_foot_site_id].copy()

        dt = max(float(self.model.opt.timestep * self.frame_skip), 1e-6)
        left_vel = (left - self.prev_left_foot) / dt
        right_vel = (right - self.prev_right_foot) / dt

        floor_z = min(float(left[2]), float(right[2]))
        left_clearance = max(0.0, float(left[2]) - floor_z)
        right_clearance = max(0.0, float(right[2]) - floor_z)

        contact_threshold = 0.025

        return {
            "left_foot_z": float(left[2]),
            "right_foot_z": float(right[2]),
            "left_foot_clearance": float(left_clearance),
            "right_foot_clearance": float(right_clearance),
            "left_foot_slip": float(np.linalg.norm(left_vel[:2])),
            "right_foot_slip": float(np.linalg.norm(right_vel[:2])),
            "left_contact": bool(left_clearance <= contact_threshold),
            "right_contact": bool(right_clearance <= contact_threshold),
        }

    def _get_obs(self) -> np.ndarray:
        up_z = self._root_up_z()
        qpos = self.data.qpos
        qvel = self.data.qvel

        joint_pos = np.array([qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)
        joint_vel = np.array([qvel[vadr] for vadr in self.qvel_adrs], dtype=np.float32)
        joint_error = joint_pos - self.stand_joint_pos

        foot = self._foot_metrics()
        phase = self._phase01()
        left_swing, right_swing = self._desired_swing()
        y_target = self._target_y_offset()

        obs = np.concatenate(
            [
                np.array(
                    [
                        float(qpos[2]),
                        up_z,
                        float(qpos[0]),
                        float(qpos[1]),
                        float(qvel[0]),
                        float(qvel[1]),
                        float(qvel[2]),
                        float(qvel[3]),
                        float(qvel[4]),
                        float(qvel[5]),
                    ],
                    dtype=np.float32,
                ),
                joint_error.astype(np.float32),
                (0.1 * joint_vel).astype(np.float32),
                np.array(
                    [
                        foot["left_foot_clearance"],
                        foot["right_foot_clearance"],
                        foot["left_foot_slip"],
                        foot["right_foot_slip"],
                        math.sin(2.0 * math.pi * phase),
                        math.cos(2.0 * math.pi * phase),
                        float(left_swing),
                        float(right_swing),
                        y_target,
                        self.cfg.target_clearance,
                        self.cfg.target_x_velocity,
                        self._lift_envelope(),
                        self.cfg.teacher_scale * self.teacher_scale_multiplier,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, float]:
        foot = self._foot_metrics()
        left_swing, right_swing = self._desired_swing()
        left_expected_contact = not left_swing
        right_expected_contact = not right_swing
        left_match = bool(foot["left_contact"]) == bool(left_expected_contact)
        right_match = bool(foot["right_contact"]) == bool(right_expected_contact)

        y_target = self._target_y_offset()
        info: Dict[str, float] = {
            "stage": self.stage_name,
            "episode_step": int(self.episode_step),
            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "up_z": self._root_up_z(),
            "phase": float(self._phase01()),
            "lift_envelope": float(self._lift_envelope()),
            "target_y_offset": float(y_target),
            "target_x_velocity": float(self.cfg.target_x_velocity),
            "target_clearance": float(self.cfg.target_clearance),
            "left_expected_contact": bool(left_expected_contact),
            "right_expected_contact": bool(right_expected_contact),
            "left_swing": bool(left_swing),
            "right_swing": bool(right_swing),
            "contact_accuracy": 0.5 * (float(left_match) + float(right_match)),
            "teacher_scale": float(self.cfg.teacher_scale * self.teacher_scale_multiplier),
        }
        info.update(foot)
        return info

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])
        y_target = float(info["target_y_offset"])

        left_swing = bool(info["left_swing"])
        right_swing = bool(info["right_swing"])
        left_clear = float(info["left_foot_clearance"])
        right_clear = float(info["right_foot_clearance"])
        left_slip = float(info["left_foot_slip"])
        right_slip = float(info["right_foot_slip"])

        # 1. Posture.
        posture_reward = (
            2.0 * max(up_z, 0.0)
            + 1.0 * math.exp(-30.0 * (height - 0.79) ** 2)
        )

        # 2. COM/lateral shift and x containment.
        # Stronger x anchoring is necessary. The previous version solved lateral shift
        # but drifted backward until termination during shift_right.
        com_reward = (
            2.0 * math.exp(-170.0 * (y_pos - y_target) ** 2)
            - 1.4 * abs(y_vel)
            - 2.8 * abs(x_pos)
        )
        if self.cfg.allow_x_motion:
            com_reward += 1.0 * math.exp(-30.0 * (x_vel - self.cfg.target_x_velocity) ** 2)
        else:
            com_reward += 1.6 * math.exp(-70.0 * (x_vel - 0.0) ** 2)

        # 3. Foot phase.
        target_clearance = float(self.cfg.target_clearance)
        if left_swing:
            foot_phase_reward = 3.0 * math.exp(-420.0 * (left_clear - target_clearance) ** 2)
        elif right_swing:
            foot_phase_reward = 3.0 * math.exp(-420.0 * (right_clear - target_clearance) ** 2)
        else:
            foot_phase_reward = 1.5 * math.exp(-260.0 * (left_clear ** 2 + right_clear ** 2))

        # 4. Support stability.
        support_slip = 0.0
        if not left_swing:
            support_slip += left_slip
        if not right_swing:
            support_slip += right_slip

        support_reward = (
            2.0 * float(info["contact_accuracy"])
            - 1.8 * min(support_slip, 2.0)
        )

        # 5. Residual smoothness.
        action_delta = float(np.mean(np.square(action - self.prev_action)))
        action_energy = float(np.mean(np.square(action)))
        residual_smoothness_reward = -0.06 * action_energy - 0.10 * action_delta

        reward = posture_reward + com_reward + foot_phase_reward + support_reward + residual_smoothness_reward

        return reward, {
            "reward_posture": float(posture_reward),
            "reward_com": float(com_reward),
            "reward_foot_phase": float(foot_phase_reward),
            "reward_support": float(support_reward),
            "reward_smoothness": float(residual_smoothness_reward),
            "reward_total": float(reward),
            "support_slip": float(support_slip),
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])

        if height < 0.56 or height > 1.03:
            return True
        if up_z < 0.62:
            return True
        if abs(y_pos) > 0.50:
            return True
        if not self.cfg.allow_x_motion and abs(x_pos) > 0.45:
            return True
        if self.cfg.allow_x_motion:
            if x_pos > 0.25:
                return True
            if x_pos < -0.90:
                return True
        if not self.cfg.allow_x_motion and abs(x_vel) > 1.10:
            return True
        if abs(x_vel) > 1.60 or abs(y_vel) > 1.20:
            return True
        return False

    def close(self):
        pass
