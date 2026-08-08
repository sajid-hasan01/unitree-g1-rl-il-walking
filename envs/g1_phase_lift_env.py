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
class PhaseLiftConfig:
    max_steps: int = 700
    cycle_duration: float = 2.0
    swing_start: float = 0.20
    swing_end: float = 0.70
    completion_window: float = 0.12
    target_clearance: float = 0.040
    target_lateral_shift: float = 0.035
    target_x_position: float = 0.0
    target_x_velocity: float = 0.0


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def raised_cosine(local_phase: float) -> float:
    local_phase = float(np.clip(local_phase, 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * local_phase))


def circular_in_window(phi: float, start: float, end: float) -> bool:
    phi = phi % 1.0
    start = start % 1.0
    end = end % 1.0
    if start <= end:
        return start <= phi < end
    return phi >= start or phi < end


class G1PhaseLiftEnv(gym.Env):
    """
    Phase-based lifting environment for Unitree G1.

    This env removes the rigid teacher pose. The policy receives only:
      - a phase clock
      - an authored contact schedule
      - smooth phase-based clearance targets
      - COM/root-x/y/support stability rewards

    There are no dataset frame lookups, no retargeted contact labels, and no analytic
    hip/knee/ankle swing offsets applied inside the environment. The base target is the
    standing keyframe, and all lift motion must come from the PPO action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml",
        stage: str = "right_lift",
        action_scale: float = 0.35,
        cycle_duration: float = 3.0,
        swing_start: float = 0.20,
        swing_end: float = 0.70,
        target_clearance: float = 0.025,
        target_lateral_shift: float = 0.025,
        max_steps: int = 700,
        frame_skip: int = 5,
        action_target_smoothing: float = 0.35,
        randomize_reset: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

        self.model_path = model_path
        self.stage = str(stage).strip().lower()
        if self.stage not in {"right_lift", "left_lift", "alt_lift"}:
            raise ValueError("stage must be one of: right_lift, left_lift, alt_lift")

        self.action_scale = float(action_scale)
        self.frame_skip = int(frame_skip)
        self.action_target_smoothing = float(np.clip(action_target_smoothing, 0.0, 0.95))
        self.randomize_reset = bool(randomize_reset)
        self.rng = np.random.default_rng(seed)

        self.cfg = PhaseLiftConfig(
            max_steps=int(max_steps),
            cycle_duration=float(cycle_duration),
            swing_start=float(swing_start),
            swing_end=float(swing_end),
            target_clearance=float(target_clearance),
            target_lateral_shift=float(target_lateral_shift),
        )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_ids: List[int] = []
        self.qpos_adrs: List[int] = []
        self.qvel_adrs: List[int] = []
        self.actuator_ids: List[int] = []
        self._build_joint_and_actuator_maps()

        self.left_foot_site_id = self._required_site("left_foot")
        self.right_foot_site_id = self._required_site("right_foot")
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            self.pelvis_body_id = 1

        self.stand_qpos = self._get_stand_qpos()
        self.stand_joint_pos = np.array([self.stand_qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)

        # Action authority is deliberately larger than the failed low-swing runs
        # because the teacher pose is gone. These are joint-offset ceilings before
        # action_scale is applied.
        self.base_action_scale = np.array(
            [
                0.70, 0.35, 0.20, 0.90, 0.65, 0.28,  # left leg
                0.70, 0.35, 0.20, 0.90, 0.65, 0.28,  # right leg
                0.25, 0.22, 0.28,                    # waist
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)
        # Same shape as previous 15-DOF curriculum envs.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)

        self.episode_step = 0
        self.prev_action = np.zeros(15, dtype=np.float32)
        self.prev_left_foot = np.zeros(3, dtype=np.float64)
        self.prev_right_foot = np.zeros(3, dtype=np.float64)
        self.current_target = self.stand_joint_pos.copy()

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
            q_noise = self.rng.normal(0.0, 0.003, size=15)
            for i, qadr in enumerate(self.qpos_adrs):
                self.data.qpos[qadr] += q_noise[i]
            self.data.qvel[:6] += self.rng.normal(0.0, 0.0025, size=6)

        self.current_target = self.stand_joint_pos.copy()
        self._set_actuator_targets(self.current_target)
        mujoco.mj_forward(self.model, self.data)

        self.prev_left_foot = self.data.site_xpos[self.left_foot_site_id].copy()
        self.prev_right_foot = self.data.site_xpos[self.right_foot_site_id].copy()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        raw_target = self.stand_joint_pos + self.action_scale * self.base_action_scale * action
        raw_target = np.clip(raw_target, self.ctrl_low, self.ctrl_high)

        # Target smoothing prevents the first no-teacher PPO attempts from injecting
        # unrealistic joint jumps.
        self.current_target = (
            self.action_target_smoothing * self.current_target
            + (1.0 - self.action_target_smoothing) * raw_target
        ).astype(np.float32)

        for _ in range(self.frame_skip):
            self._set_actuator_targets(self.current_target)
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

    @property
    def policy_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def _phase01(self) -> float:
        elapsed = self.episode_step * self.policy_dt
        return (elapsed / max(self.cfg.cycle_duration, 1e-6)) % 1.0

    def _swing_windows(self) -> List[Tuple[str, float, float]]:
        if self.stage == "right_lift":
            return [("right", self.cfg.swing_start, self.cfg.swing_end)]
        if self.stage == "left_lift":
            return [("left", self.cfg.swing_start, self.cfg.swing_end)]
        # Alternating version: two shorter single-foot windows per cycle.
        return [("right", 0.12, 0.38), ("left", 0.62, 0.88)]

    def _swing_state(self) -> Tuple[bool, bool]:
        phi = self._phase01()
        left_swing = False
        right_swing = False
        for foot, start, end in self._swing_windows():
            if circular_in_window(phi, start, end):
                if foot == "left":
                    left_swing = True
                else:
                    right_swing = True
        return left_swing, right_swing

    def _clearance_target_for(self, foot: str) -> float:
        phi = self._phase01()
        for swing_foot, start, end in self._swing_windows():
            if swing_foot != foot:
                continue
            if circular_in_window(phi, start, end):
                if start <= end:
                    local = (phi - start) / max(end - start, 1e-6)
                else:
                    local = ((phi - start) % 1.0) / max((end - start) % 1.0, 1e-6)
                return self.cfg.target_clearance * raised_cosine(local)
        return 0.0

    def _completion_active_for(self, foot: str) -> bool:
        phi = self._phase01()
        for swing_foot, _, end in self._swing_windows():
            if swing_foot != foot:
                continue
            finish = end % 1.0
            after_finish = (finish + self.cfg.completion_window) % 1.0
            if circular_in_window(phi, finish, after_finish):
                return True
        return False

    def _target_y_offset(self) -> float:
        phi = self._phase01()
        target = 0.0

        for foot, start, end in self._swing_windows():
            support_sign = +1.0 if foot == "right" else -1.0
            # right foot swing -> left support -> positive y
            # left foot swing -> right support -> negative y

            if start <= end:
                if phi < start:
                    # preload before swing
                    ramp = smoothstep(phi / max(start, 1e-6))
                    candidate = support_sign * self.cfg.target_lateral_shift * ramp
                elif phi < end:
                    candidate = support_sign * self.cfg.target_lateral_shift
                else:
                    decay_den = max(1.0 - end, 1e-6)
                    decay = 1.0 - smoothstep((phi - end) / decay_den)
                    candidate = support_sign * self.cfg.target_lateral_shift * decay
            else:
                candidate = 0.0

            # For alt_lift, choose the stronger active/preload target.
            if abs(candidate) > abs(target):
                target = candidate

        return float(target)

    def _set_actuator_targets(self, controlled_targets: np.ndarray) -> None:
        controlled_set = set(self.actuator_ids)

        # Hold non-controlled upper body at stand pose. No scripted teacher/arms here.
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                qadr = int(self.model.jnt_qposadr[joint_id])
                if qadr < self.model.nq:
                    lo, hi = self.model.actuator_ctrlrange[actuator_id]
                    self.data.ctrl[actuator_id] = float(np.clip(self.stand_qpos[qadr], lo, hi))

        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(controlled_targets[i])

    def _root_up_z(self) -> float:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return float(mat[8])

    def _pelvis_com_x(self) -> float:
        return float(self.data.subtree_com[self.pelvis_body_id][0])

    def _foot_metrics(self) -> Dict[str, float]:
        left = self.data.site_xpos[self.left_foot_site_id].copy()
        right = self.data.site_xpos[self.right_foot_site_id].copy()

        dt = max(self.policy_dt, 1e-6)
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
        left_swing, right_swing = self._swing_state()

        left_expected_contact = not left_swing
        right_expected_contact = not right_swing

        left_match = bool(foot["left_contact"]) == left_expected_contact
        right_match = bool(foot["right_contact"]) == right_expected_contact

        support_slip = 0.0
        support_count = 0
        if left_expected_contact:
            support_slip += float(foot["left_foot_slip"])
            support_count += 1
        if right_expected_contact:
            support_slip += float(foot["right_foot_slip"])
            support_count += 1
        support_slip = support_slip / max(support_count, 1)

        support_x = 0.0
        support_count_x = 0
        if left_expected_contact:
            support_x += float(self.data.site_xpos[self.left_foot_site_id][0])
            support_count_x += 1
        if right_expected_contact:
            support_x += float(self.data.site_xpos[self.right_foot_site_id][0])
            support_count_x += 1
        support_x = support_x / max(support_count_x, 1)

        pelvis_com_x = self._pelvis_com_x()

        info: Dict[str, float] = {
            "stage": self.stage,
            "episode_step": int(self.episode_step),
            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "z_velocity": float(self.data.qvel[2]),
            "up_z": self._root_up_z(),
            "phase": float(self._phase01()),
            "cycle_duration": float(self.cfg.cycle_duration),
            "target_y_offset": float(self._target_y_offset()),
            "target_x_position": float(self.cfg.target_x_position),
            "target_x_velocity": float(self.cfg.target_x_velocity),
            "left_target_clearance": float(self._clearance_target_for("left")),
            "right_target_clearance": float(self._clearance_target_for("right")),
            "left_completion_active": bool(self._completion_active_for("left")),
            "right_completion_active": bool(self._completion_active_for("right")),
            "left_expected_contact": left_expected_contact,
            "right_expected_contact": right_expected_contact,
            "left_swing": left_swing,
            "right_swing": right_swing,
            "contact_accuracy": 0.5 * (float(left_match) + float(right_match)),
            "support_slip": float(support_slip),
            "support_foot_x": float(support_x),
            "pelvis_com_x": float(pelvis_com_x),
            "com_x_error": float(pelvis_com_x - support_x),
            "action_scale": float(self.action_scale),
            "reward_version": "phase_lift_v3_no_standing_x_guard",
        }
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
        phi = self._phase01()
        left_swing, right_swing = self._swing_state()

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
                        math.sin(2.0 * math.pi * phi),
                        math.cos(2.0 * math.pi * phi),
                        float(left_swing),
                        float(right_swing),
                        self._target_y_offset(),
                        self._clearance_target_for("left"),
                        self._clearance_target_for("right"),
                        self.cfg.target_x_velocity,
                        self.action_scale,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def _clearance_reward(
        self,
        actual: float,
        target: float,
        expected_contact: bool,
        completion_active: bool,
        is_contact: bool,
    ) -> float:
        """
        Phase-clearance reward without pose tracking.

        v3 makes the anti-standing term dominant. The v2 best checkpoint learned
        to stand still because posture/root rewards outweighed the swing miss
        penalty. Here, a scheduled swing foot that remains in contact receives a
        large negative reward, so "survive while doing nothing" is no longer a
        valid optimum.
        """
        actual = float(actual)
        target = float(target)

        if target > 1e-4:
            progress = min(actual / max(target, 1e-6), 1.0)
            miss = max(0.0, target - actual)
            over_lift = max(0.0, actual - 0.075)

            reward = (
                5.0 * math.exp(-900.0 * ((actual - target) ** 2))
                + 14.0 * progress
                - 520.0 * miss
                - 14.0 * over_lift
            )

            if (not expected_contact) and is_contact and target > 0.004:
                reward -= 16.0

            return reward

        if completion_active:
            reward = 6.0 * math.exp(-750.0 * (actual ** 2))
            if not is_contact:
                reward -= 10.0
            return reward

        if expected_contact:
            reward = 0.8 * math.exp(-350.0 * (actual ** 2))
            if actual > 0.025:
                reward -= 6.0 * (actual - 0.025)
            return reward

        return 0.0

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])
        target_y = float(info["target_y_offset"])

        left_clearance = float(info["left_foot_clearance"])
        right_clearance = float(info["right_foot_clearance"])
        left_target = float(info["left_target_clearance"])
        right_target = float(info["right_target_clearance"])

        posture_reward = (
            1.7 * max(up_z, 0.0)
            + 0.7 * math.exp(-45.0 * ((height - 0.79) ** 2))
        )

        x_error = x_pos - self.cfg.target_x_position
        backward_excess = max(0.0, -x_pos - 0.12)
        forward_excess = max(0.0, x_pos - 0.12)
        abs_x_excess = max(0.0, abs(x_pos) - 0.12)
        com_tracking_reward = (
            1.2 * math.exp(-120.0 * (x_error ** 2))
            + 1.2 * math.exp(-160.0 * ((y_pos - target_y) ** 2))
            + 0.7 * math.exp(-85.0 * ((x_vel - self.cfg.target_x_velocity) ** 2))
            - 1.6 * abs(y_vel)
            - 2.0 * abs(x_vel)
            - 13.0 * backward_excess
            - 13.0 * forward_excess
            - 5.0 * abs_x_excess
            - 3.0 * max(0.0, abs(y_pos) - 0.16)
        )

        left_foot_reward = self._clearance_reward(
            actual=left_clearance,
            target=left_target,
            expected_contact=bool(info["left_expected_contact"]),
            completion_active=bool(info["left_completion_active"]),
            is_contact=bool(info["left_contact"]),
        )
        right_foot_reward = self._clearance_reward(
            actual=right_clearance,
            target=right_target,
            expected_contact=bool(info["right_expected_contact"]),
            completion_active=bool(info["right_completion_active"]),
            is_contact=bool(info["right_contact"]),
        )
        foot_reward = left_foot_reward + right_foot_reward

        contact_accuracy = float(info["contact_accuracy"])
        support_reward = (
            2.0 * contact_accuracy
            - 8.0 * (1.0 - contact_accuracy)
            - 2.2 * min(float(info["support_slip"]), 2.0)
            - 1.2 * abs(float(info["com_x_error"]))
        )

        action_delta = float(np.mean(np.square(action - self.prev_action)))
        action_energy = float(np.mean(np.square(action)))
        smooth_reward = -0.045 * action_energy - 0.12 * action_delta

        reward = posture_reward + com_tracking_reward + foot_reward + support_reward + smooth_reward

        return reward, {
            "reward_posture": float(posture_reward),
            "reward_com_tracking": float(com_tracking_reward),
            "reward_foot": float(foot_reward),
            "reward_left_foot": float(left_foot_reward),
            "reward_right_foot": float(right_foot_reward),
            "reward_support": float(support_reward),
            "reward_smooth": float(smooth_reward),
            "reward_total": float(reward),
            "backward_excess": float(backward_excess),
            "forward_excess": float(forward_excess),
            "abs_x_excess": float(abs_x_excess),
            "x_error": float(x_error),
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        height = float(info["base_height"])
        up_z = float(info["up_z"])
        x_pos = float(info["x_position"])
        y_pos = float(info["y_position"])
        x_vel = float(info["x_velocity"])
        y_vel = float(info["y_velocity"])

        if height < 0.54 or height > 1.06:
            return True
        if up_z < 0.55:
            return True
        if x_pos < -0.62 or x_pos > 0.35:
            return True
        if abs(y_pos) > 0.48:
            return True
        if abs(x_vel) > 1.45 or abs(y_vel) > 1.35:
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        if float(info["base_height"]) < 0.54:
            return "base_height_low"
        if float(info["base_height"]) > 1.06:
            return "base_height_high"
        if float(info["up_z"]) < 0.55:
            return "up_z_low"
        if float(info["x_position"]) < -0.62:
            return "backward_x_limit"
        if float(info["x_position"]) > 0.35:
            return "forward_x_limit"
        if abs(float(info["y_position"])) > 0.48:
            return "lateral_y_limit"
        if abs(float(info["x_velocity"])) > 1.45:
            return "x_velocity_limit"
        if abs(float(info["y_velocity"])) > 1.35:
            return "y_velocity_limit"
        if int(info["episode_step"]) >= self.cfg.max_steps:
            return "max_steps"
        return "not_terminated"

    def close(self):
        pass
