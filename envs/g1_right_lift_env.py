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
class RightLiftConfig:
    max_steps: int = 700
    phase_period: int = 420
    target_y_offset: float = 0.032
    target_right_clearance: float = 0.032
    target_x_position: float = 0.0
    target_x_velocity: float = 0.0
    teacher_scale: float = 0.30
    support_leg_scale: float = 1.0
    swing_leg_scale: float = 0.3
    waist_scale: float = 1.0


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class G1RightLiftEnv(gym.Env):
    """
    Right-foot lift environment for Unitree G1.

    Purpose:
        Continue from a passed shift_right/COM-control checkpoint and solve the remaining
        right_lift failure: backward sagittal drift during swing-foot lift.

    Key design:
        - 15 controlled lower-body/waist joints.
        - Observation shape is kept at 53 for compatibility with previous curriculum PPO checkpoints.
        - Teacher keeps the same swing-leg pattern.
        - The failed fixed sagittal offsets are removed.
        - Sagittal support stabilization is now a live feedback controller:
            * reads pelvis subtree COM x
            * reads support-foot site x
            * reads COM/root x velocity
            * applies one saturated PD correction to support hip_pitch and ankle_pitch
        - Reward explicitly tracks COM/root x position and penalizes backward drift.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml",
        dataset_path: str = "datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz",
        action_scale: float = 0.14,
        teacher_scale_multiplier: float = 1.6,
        support_leg_scale: float = 1.0,
        swing_leg_scale: float = 0.3,
        waist_scale: float = 1.0,
        sagittal_kp: float = 0.0,
        sagittal_kd: float = 0.0,
        sagittal_clip: float = 0.0,
        sagittal_hip_sign: float = 1.0,
        sagittal_ankle_sign: float = 1.0,
        enable_scripted_arms: bool = True,
        arm_swing_scale: float = 0.25,
        arm_pitch_sign: float = 1.0,
        arm_elbow_scale: float = 0.10,
        frame_skip: int = 5,
        randomize_reset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.action_scale = float(action_scale)
        self.frame_skip = int(frame_skip)
        self.randomize_reset = bool(randomize_reset)
        self.rng = np.random.default_rng(seed)

        self.cfg = RightLiftConfig(
            support_leg_scale=float(support_leg_scale),
            swing_leg_scale=float(swing_leg_scale),
            waist_scale=float(waist_scale),
        )
        self.teacher_scale_multiplier = float(teacher_scale_multiplier)
        self.sagittal_kp = float(sagittal_kp)
        self.sagittal_kd = float(sagittal_kd)
        self.sagittal_clip = float(sagittal_clip)
        self.sagittal_hip_sign = 1.0 if float(sagittal_hip_sign) >= 0.0 else -1.0
        self.sagittal_ankle_sign = 1.0 if float(sagittal_ankle_sign) >= 0.0 else -1.0
        self.enable_scripted_arms = bool(enable_scripted_arms)
        self.arm_swing_scale = float(arm_swing_scale)
        self.arm_pitch_sign = 1.0 if float(arm_pitch_sign) >= 0.0 else -1.0
        self.arm_elbow_scale = float(arm_elbow_scale)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Dataset is loaded for path compatibility and future extension.
        # This right-lift environment uses the stand keyframe plus analytic teacher.
        self.dataset = np.load(dataset_path, allow_pickle=True)

        self.joint_ids: List[int] = []
        self.qpos_adrs: List[int] = []
        self.qvel_adrs: List[int] = []
        self.actuator_ids: List[int] = []
        self._build_joint_and_actuator_maps()

        self.left_foot_site_id = self._required_site("left_foot")
        self.right_foot_site_id = self._required_site("right_foot")
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            # Fallback for XML variants. Body 1 is normally the free root body.
            self.pelvis_body_id = 1

        self.stand_qpos = self._get_stand_qpos()
        self.stand_joint_pos = np.array([self.stand_qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        # Joint-wise residual authority before support/swing/waist scaling.
        self.base_residual_scale = np.array(
            [
                0.50, 0.25, 0.12, 0.55, 0.42, 0.20,  # left/support leg
                0.38, 0.18, 0.08, 0.42, 0.30, 0.15,  # right/swing leg
                0.10, 0.10, 0.16,                    # waist
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        # Kept at 53 to load previous curriculum checkpoint safely.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)

        self.episode_step = 0
        self.prev_action = np.zeros(15, dtype=np.float32)
        self.prev_left_foot = np.zeros(3, dtype=np.float64)
        self.prev_right_foot = np.zeros(3, dtype=np.float64)

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

        for joint_name in CONTROLLED_15_JOINTS:
            joint_id = self._required_joint(joint_name)
            if joint_id not in joint_to_actuator:
                raise RuntimeError(f"No actuator found for joint: {joint_name}")
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
            # Keep reset noise mild so training stays practical on a small laptop.
            q_noise = self.rng.normal(0.0, 0.004, size=15)
            for i, qadr in enumerate(self.qpos_adrs):
                self.data.qpos[qadr] += q_noise[i]
            self.data.qvel[:6] += self.rng.normal(0.0, 0.003, size=6)

        target = self.stand_joint_pos + self._teacher_offsets()
        self._set_actuator_targets(target)
        mujoco.mj_forward(self.model, self.data)

        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        teacher = self._teacher_offsets()
        residual_scale = self._residual_authority_scale()
        residual = self.action_scale * self.base_residual_scale * residual_scale * action

        # Compute sagittal feedback every physics step, not once per policy step.
        # This is a live feedback balance controller, not an episode-timer offset.
        for _ in range(self.frame_skip):
            sagittal_command, _ = self._sagittal_feedback()
            target = self.stand_joint_pos + teacher + residual
            target[0] += self.sagittal_hip_sign * sagittal_command    # left/support hip_pitch
            target[4] += self.sagittal_ankle_sign * sagittal_command  # left/support ankle_pitch
            target = np.clip(target, self.ctrl_low, self.ctrl_high)
            self._set_actuator_targets(target)
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
        self.episode_step += 1

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _phase01(self) -> float:
        return (self.episode_step % self.cfg.phase_period) / float(self.cfg.phase_period)

    def _right_swing_active(self) -> bool:
        phase = self._phase01()
        return 0.45 <= phase < 0.82

    def _lift_envelope(self) -> float:
        phase = self._phase01()
        if phase < 0.45 or phase >= 0.82:
            return 0.0
        local = (phase - 0.45) / 0.37
        if local < 0.50:
            return smoothstep(local / 0.50)
        return smoothstep((1.0 - local) / 0.50)

    def _target_y_offset(self) -> float:
        phase = self._phase01()
        raw = self.cfg.target_y_offset
        if phase < 0.42:
            return raw * smoothstep(phase / 0.42)
        if phase < 0.82:
            return raw
        return raw * (1.0 - smoothstep((phase - 0.82) / 0.18))

    def _teacher_offsets(self) -> np.ndarray:
        """
        Teacher = lateral COM shift + right-foot lift + sagittal support compensation.

        Compensation terms intentionally target the known failure:
        right hip flexion caused backward drift. During swing, the left support leg and
        waist produce a small counteracting forward moment.
        """
        teacher = np.zeros(15, dtype=np.float32)

        scale = self.cfg.teacher_scale * self.teacher_scale_multiplier
        phase = self._phase01()
        y_target = self._target_y_offset()

        # Lateral COM shift toward left support foot.
        lateral = np.clip(y_target / max(self.cfg.target_y_offset, 1e-6), 0.0, 1.0) * scale
        teacher[1] += -0.18 * lateral      # left hip roll
        teacher[5] += 0.10 * lateral       # left ankle roll
        teacher[7] += -0.18 * lateral      # right hip roll
        teacher[11] += 0.10 * lateral      # right ankle roll
        teacher[13] += 0.18 * lateral      # waist roll

        lift = self._lift_envelope()
        if lift > 0.0:
            amp = scale * lift

            # Right swing lift: reduced hip pitch; more knee/ankle for clearance.
            teacher[6] += 0.22 * amp       # right hip pitch flexion
            teacher[7] += 0.13 * amp       # right hip roll
            teacher[9] += 0.96 * amp       # right knee
            teacher[10] += 0.52 * amp      # right ankle pitch
            teacher[11] += 0.06 * amp      # right ankle roll

            # No fixed support-leg or waist sagittal offsets here.
            # Sagittal stabilization is handled by _sagittal_feedback(), which
            # reads live COM/support-foot error every physics step.

        return teacher.astype(np.float32)

    def _support_foot_x(self) -> float:
        # In right_lift, the left foot is the stance/support foot throughout the
        # pre-shift, swing, and recovery portions.
        return float(self.data.site_xpos[self.left_foot_site_id][0])

    def _pelvis_com_x(self) -> float:
        return float(self.data.subtree_com[self.pelvis_body_id][0])

    def _sagittal_feedback(self) -> Tuple[float, Dict[str, float]]:
        """
        Live sagittal support controller.

        error > 0: COM is ahead of support foot
        error < 0: COM is behind support foot

        command = -kp * error - kd * x_velocity

        This command is a mathematical negative-feedback balance command.
        Separate empirical signs map the command to this MJCF's support
        hip_pitch and ankle_pitch joint-angle directions.
        """
        support_x = self._support_foot_x()
        com_x = self._pelvis_com_x()
        com_vx = float(self.data.qvel[0])

        com_x_error = com_x - support_x
        p_term = -self.sagittal_kp * com_x_error
        d_term = -self.sagittal_kd * com_vx
        raw = p_term + d_term
        command = float(np.clip(raw, -self.sagittal_clip, self.sagittal_clip))

        return command, {
            "pelvis_com_x": float(com_x),
            "support_foot_x": float(support_x),
            "com_x_error": float(com_x_error),
            "com_x_velocity": float(com_vx),
            "sagittal_p_term": float(p_term),
            "sagittal_d_term": float(d_term),
            "sagittal_raw": float(raw),
            "sagittal_command": float(command),
            "sagittal_correction": float(command),  # backward-compatible log name
            "sagittal_hip_offset": float(self.sagittal_hip_sign * command),
            "sagittal_ankle_offset": float(self.sagittal_ankle_sign * command),
            "sagittal_hip_sign": float(self.sagittal_hip_sign),
            "sagittal_ankle_sign": float(self.sagittal_ankle_sign),
            "sagittal_kp": float(self.sagittal_kp),
            "sagittal_kd": float(self.sagittal_kd),
            "sagittal_clip": float(self.sagittal_clip),
        }

    def _residual_authority_scale(self) -> np.ndarray:
        # Right foot is swing leg. Left leg and waist retain full authority.
        scale = np.ones(15, dtype=np.float32)
        scale[0:6] *= self.cfg.support_leg_scale
        scale[6:12] *= self.cfg.swing_leg_scale
        scale[12:15] *= self.cfg.waist_scale

        # Give the swing leg a little more control during put-down/recovery.
        if not self._right_swing_active():
            scale[6:12] = np.maximum(scale[6:12], 0.60)

        return scale

    def _scripted_arm_offset(self, joint_name: str) -> float:
        """
        Deterministic full-body assist without increasing the PPO action space.

        Right_lift swings the right leg, so the arms counter-swing:
        - left shoulder pitch and right shoulder pitch move in opposite directions
        - elbow offsets are small and mainly for natural posture

        The sign is exposed as arm_pitch_sign because MJCF shoulder pitch direction
        conventions differ across model versions.
        """
        if not self.enable_scripted_arms:
            return 0.0

        name = (joint_name or "").lower()
        if "shoulder_pitch" not in name and "elbow" not in name:
            return 0.0

        # Only use arm swing during the risky right-foot swing phase.
        lift = self._lift_envelope()
        if lift <= 0.0:
            return 0.0

        amp = self.arm_swing_scale * lift

        is_left = name.startswith("left_") or "_left_" in name or "left" in name
        is_right = name.startswith("right_") or "_right_" in name or "right" in name

        if "shoulder_pitch" in name:
            if is_left:
                return self.arm_pitch_sign * amp
            if is_right:
                return -self.arm_pitch_sign * amp

        if "elbow" in name:
            # Smaller elbow change so arms do not dominate dynamics.
            if is_left:
                return self.arm_pitch_sign * self.arm_elbow_scale * lift
            if is_right:
                return self.arm_pitch_sign * self.arm_elbow_scale * lift

        return 0.0

    def _scripted_arm_summary(self) -> Dict[str, float]:
        lift = self._lift_envelope()
        amp = self.arm_swing_scale * lift if self.enable_scripted_arms else 0.0
        return {
            "enable_scripted_arms": float(self.enable_scripted_arms),
            "arm_swing_scale": float(self.arm_swing_scale),
            "arm_pitch_sign": float(self.arm_pitch_sign),
            "arm_elbow_scale": float(self.arm_elbow_scale),
            "left_arm_pitch_offset": float(self.arm_pitch_sign * amp),
            "right_arm_pitch_offset": float(-self.arm_pitch_sign * amp),
        }

    def _set_actuator_targets(self, controlled_targets: np.ndarray) -> None:
        # Hold non-controlled actuators at stand pose, except scripted arms.
        # This gives full-body angular-momentum assist while keeping PPO action size = 15.
        controlled_set = set(self.actuator_ids)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                qadr = int(self.model.jnt_qposadr[joint_id])
                if qadr < self.model.nq:
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
                    arm_offset = 0.0 if actuator_id in controlled_set else self._scripted_arm_offset(joint_name)
                    ctrl_value = float(self.stand_qpos[qadr] + arm_offset)
                    lo, hi = self.model.actuator_ctrlrange[actuator_id]
                    self.data.ctrl[actuator_id] = float(np.clip(ctrl_value, lo, hi))

        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(controlled_targets[i])

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

    def _get_info(self) -> Dict[str, float]:
        foot = self._foot_metrics()
        right_swing = self._right_swing_active()
        left_expected_contact = True
        right_expected_contact = not right_swing

        left_match = bool(foot["left_contact"]) == left_expected_contact
        right_match = bool(foot["right_contact"]) == right_expected_contact

        residual_scale = self._residual_authority_scale()
        _, sagittal_info = self._sagittal_feedback()

        info: Dict[str, float] = {
            "stage": "right_lift",
            "episode_step": int(self.episode_step),
            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "z_velocity": float(self.data.qvel[2]),
            "up_z": self._root_up_z(),
            "phase": float(self._phase01()),
            "lift_envelope": float(self._lift_envelope()),
            "target_y_offset": float(self._target_y_offset()),
            "target_x_position": float(self.cfg.target_x_position),
            "target_x_velocity": float(self.cfg.target_x_velocity),
            "target_clearance": float(self.cfg.target_right_clearance),
            "left_expected_contact": left_expected_contact,
            "right_expected_contact": right_expected_contact,
            "left_swing": False,
            "right_swing": right_swing,
            "contact_accuracy": 0.5 * (float(left_match) + float(right_match)),
            "teacher_scale": float(self.cfg.teacher_scale * self.teacher_scale_multiplier),
            "support_leg_scale": float(self.cfg.support_leg_scale),
            "swing_leg_scale": float(self.cfg.swing_leg_scale),
            "waist_scale": float(self.cfg.waist_scale),
            "residual_scale_left_mean": float(np.mean(residual_scale[0:6])),
            "residual_scale_right_mean": float(np.mean(residual_scale[6:12])),
            "residual_scale_waist_mean": float(np.mean(residual_scale[12:15])),
        }
        info.update(sagittal_info)
        info.update(self._scripted_arm_summary())
        info.update(foot)
        return info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        up_z = self._root_up_z()

        joint_pos = np.array([qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)
        joint_vel = np.array([qvel[vadr] for vadr in self.qvel_adrs], dtype=np.float32)
        joint_error = joint_pos - self.stand_joint_pos

        foot = self._foot_metrics()
        phase = self._phase01()
        right_swing = self._right_swing_active()

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
                        0.0,                         # left_swing
                        float(right_swing),           # right_swing
                        self._target_y_offset(),
                        self.cfg.target_right_clearance,
                        self.cfg.target_x_velocity,
                        self._lift_envelope(),
                        self.cfg.teacher_scale * self.teacher_scale_multiplier,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])
        right_clearance = float(info["right_foot_clearance"])
        left_clearance = float(info["left_foot_clearance"])
        left_slip = float(info["left_foot_slip"])
        right_swing = bool(info["right_swing"])
        target_y = float(info["target_y_offset"])

        # 1. Posture stability.
        posture_reward = (
            3.2 * max(up_z, 0.0)
            + 1.5 * math.exp(-40.0 * (height - 0.79) ** 2)
        )

        # 2. COM/root tracking. This is the main sagittal drift fix.
        x_error = x_pos - self.cfg.target_x_position
        backward_excess = max(0.0, -x_pos - 0.10)
        com_tracking_reward = (
            2.4 * math.exp(-120.0 * (x_error ** 2))
            + 1.8 * math.exp(-180.0 * ((y_pos - target_y) ** 2))
            + 1.6 * math.exp(-90.0 * ((x_vel - self.cfg.target_x_velocity) ** 2))
            - 1.5 * abs(y_vel)
            - 8.0 * backward_excess
            - 2.0 * max(0.0, abs(y_pos) - 0.14)
        )

        # 3. Right-foot clearance. Penalize excessive over-lift because v4/v5 over-lifted.
        if right_swing:
            target = self.cfg.target_right_clearance
            over_lift = max(0.0, right_clearance - 0.055)
            foot_reward = (
                3.2 * math.exp(-420.0 * ((right_clearance - target) ** 2))
                + 0.8 * min(right_clearance / max(target, 1e-6), 1.0)
                - 10.0 * over_lift
            )
        else:
            foot_reward = 1.4 * math.exp(-260.0 * (right_clearance ** 2 + left_clearance ** 2))

        # 4. Support contact/slip.
        support_slip = left_slip
        support_reward = (
            2.0 * float(info["contact_accuracy"])
            - 2.0 * min(support_slip, 2.0)
        )

        # 5. Smooth residual.
        action_delta = float(np.mean(np.square(action - self.prev_action)))
        action_energy = float(np.mean(np.square(action)))
        smooth_reward = -0.05 * action_energy - 0.10 * action_delta

        reward = posture_reward + com_tracking_reward + foot_reward + support_reward + smooth_reward

        return reward, {
            "reward_posture": float(posture_reward),
            "reward_com_tracking": float(com_tracking_reward),
            "reward_foot": float(foot_reward),
            "reward_support": float(support_reward),
            "reward_smooth": float(smooth_reward),
            "reward_total": float(reward),
            "support_slip": float(support_slip),
            "backward_excess": float(backward_excess),
            "x_error": float(x_error),
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])

        if height < 0.56 or height > 1.04:
            return True
        if up_z < 0.62:
            return True
        if x_pos < -0.46 or x_pos > 0.28:
            return True
        if abs(y_pos) > 0.42:
            return True
        if abs(x_vel) > 1.25 or abs(y_vel) > 1.20:
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        if float(info["base_height"]) < 0.56:
            return "base_height_low"
        if float(info["base_height"]) > 1.04:
            return "base_height_high"
        if float(info["up_z"]) < 0.62:
            return "up_z_low"
        if float(info["x_position"]) < -0.46:
            return "backward_x_limit"
        if float(info["x_position"]) > 0.28:
            return "forward_x_limit"
        if abs(float(info["y_position"])) > 0.42:
            return "lateral_y_limit"
        if abs(float(info["x_velocity"])) > 1.25:
            return "x_velocity_limit"
        if abs(float(info["y_velocity"])) > 1.20:
            return "y_velocity_limit"
        if int(info["episode_step"]) >= self.cfg.max_steps:
            return "max_steps"
        return "not_terminated"

    def close(self):
        pass
