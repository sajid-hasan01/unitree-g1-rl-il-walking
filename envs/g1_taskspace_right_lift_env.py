from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


CONTROLLED_15_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


@dataclass
class TaskspaceRightLiftConfig:
    model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml"
    frame_skip: int = 5
    max_steps: int = 560

    # One non-repeating lift cycle.
    # Unlike the older hybrid env, phase does not wrap. This prevents a fake
    # repeated-cycle pass.
    cycle_duration: float = 5.6
    shift_start: float = 0.08
    swing_start: float = 0.46
    swing_end: float = 0.76
    land_end: float = 0.92

    # Actual visible lift target.
    target_clearance: float = 0.035
    target_lateral_shift: float = 0.018

    # Task-space IK controller.
    ik_gain: float = 1.35
    ik_damping: float = 0.030
    ik_max_delta: float = 0.22
    xy_hold_weight: float = 0.08
    z_lift_weight: float = 1.40

    # Stabilizers.
    x_hard_limit: float = 0.36
    y_hard_limit: float = 0.30
    x_velocity_hard_limit: float = 1.20
    y_velocity_hard_limit: float = 1.20
    min_up_z: float = 0.72
    min_height: float = 0.55
    max_height: float = 1.05

    randomize_reset: bool = False


class G1TaskspaceRightLiftEnv(gym.Env):
    """
    Deterministic model-based task-space right-foot lift.

    Goal:
    - Shift body weight over the left support foot.
    - Lift the right foot by commanding the right_foot site in task space.
    - Hold visible 3-5 cm clearance.
    - Land the right foot again.
    - Do this before adding PPO/RL.

    Action is optional 4D residual for later:
    action[0] = lift-height residual
    action[1] = lateral-shift residual
    action[2] = stance sagittal residual
    action[3] = lateral correction residual

    For deterministic control, use action = zeros(4).
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = TaskspaceRightLiftConfig(**kwargs)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep * self.cfg.frame_skip)

        self.joint_names = CONTROLLED_15_JOINTS
        self.qpos_adrs: List[int] = []
        self.qvel_adrs: List[int] = []
        self.actuator_ids: List[int] = []

        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Missing joint: {name}")
            self.qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            self.qvel_adrs.append(int(self.model.jnt_dofadr[jid]))

            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", ""))
            if aid < 0:
                for k in range(self.model.nu):
                    if int(self.model.actuator_trnid[k, 0]) == jid:
                        aid = k
                        break
            if aid < 0:
                raise RuntimeError(f"Missing actuator for joint: {name}")
            self.actuator_ids.append(int(aid))

        self.left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        if min(self.left_foot_site, self.right_foot_site, self.left_foot_body, self.right_foot_body) < 0:
            raise RuntimeError("Missing required foot site/body names: left_foot, right_foot, ankle_roll_link bodies.")

        self.stand_qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else self.data.qpos.copy()
        self.stand_joint_pos = np.array([self.stand_qpos[i] for i in self.qpos_adrs], dtype=np.float64)

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            if 0 <= jid < self.model.njnt:
                qadr = int(self.model.jnt_qposadr[jid])
                if qadr < len(self.stand_qpos):
                    self.default_ctrl[aid] = float(self.stand_qpos[qadr])

        # Optional residual; deterministic controller uses zeros.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.episode_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.left_foot_p0 = np.zeros(3, dtype=np.float64)
        self.right_foot_p0 = np.zeros(3, dtype=np.float64)

        self._set_state_to_stand()
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def _set_state_to_stand(self) -> None:
        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.prev_action[:] = 0.0
        self._set_state_to_stand()

        if self.cfg.randomize_reset:
            for qadr, noise in zip(self.qpos_adrs, self.np_random.normal(0.0, 0.002, len(self.qpos_adrs))):
                self.data.qpos[qadr] += float(noise)
            self.data.qvel[:] = self.np_random.normal(0.0, 0.004, self.model.nv)
            mujoco.mj_forward(self.model, self.data)

        self.left_foot_p0[:] = self.data.site_xpos[self.left_foot_site].copy()
        self.right_foot_p0[:] = self.data.site_xpos[self.right_foot_site].copy()

        info = self._get_info()
        return self._get_obs(), info

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _phase(self) -> float:
        # Non-wrapping phase.
        t = self.episode_step * self.dt
        return float(np.clip(t / max(self.cfg.cycle_duration, 1e-6), 0.0, 1.0))

    def _shift_env(self, phi: float) -> float:
        # Shift up before swing, hold during swing, return after landing.
        if phi < self.cfg.shift_start:
            return 0.0
        if phi < self.cfg.swing_start:
            u = (phi - self.cfg.shift_start) / max(self.cfg.swing_start - self.cfg.shift_start, 1e-6)
            return self._smoothstep(u)
        if phi < self.cfg.swing_end:
            return 1.0
        if phi < self.cfg.land_end:
            u = (phi - self.cfg.swing_end) / max(self.cfg.land_end - self.cfg.swing_end, 1e-6)
            return 1.0 - self._smoothstep(u)
        return 0.0

    def _swing_env(self, phi: float) -> float:
        if phi < self.cfg.swing_start or phi > self.cfg.swing_end:
            return 0.0
        u = (phi - self.cfg.swing_start) / max(self.cfg.swing_end - self.cfg.swing_start, 1e-6)
        return float(math.sin(math.pi * np.clip(u, 0.0, 1.0)))

    def _root_up_z(self) -> float:
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        return float(mat[8])

    def _foot_contact(self, body_id: int) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 == body_id or b2 == body_id:
                return True
        return False

    def _site_linear_velocity(self, site_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp @ self.data.qvel

    def _site_jacobian_for_dofs(self, site_id: int, dof_indices: List[int]) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp[:, dof_indices]

    def _ik_delta(self, site_id: int, dof_indices: List[int], task_error: np.ndarray) -> np.ndarray:
        J = self._site_jacobian_for_dofs(site_id, dof_indices)
        damping = float(self.cfg.ik_damping)
        A = J @ J.T + (damping * damping) * np.eye(3)
        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error
        delta = float(self.cfg.ik_gain) * delta
        delta = np.clip(delta, -self.cfg.ik_max_delta, self.cfg.ik_max_delta)
        return delta

    def _target_foot_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # action[0] can later adjust lift height by +/- 1.5 cm.
        lift_residual = 0.015 * float(np.clip(action[0], -1.0, 1.0))
        clearance = max(0.0, self.cfg.target_clearance + lift_residual)
        target = self.right_foot_p0.copy()
        target[2] += clearance * float(info["swing_env"])
        return target

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        phi = float(info["phase"])
        sw = float(info["swing_env"])
        sh = float(info["shift_env"])

        target = self.stand_joint_pos.copy()
        off = np.zeros(15, dtype=np.float64)

        # Residuals for later. Zero-action is deterministic controller.
        lateral_res = 0.020 * float(action[1])
        sagittal_res = 0.035 * float(action[2])
        lateral_corr_res = 0.035 * float(action[3])

        target_y = self.cfg.target_lateral_shift * sh + lateral_res * sh
        y = float(info["y_position"])
        yv = float(info["y_velocity"])
        x = float(info["x_position"])
        xv = float(info["x_velocity"])

        # Body/root feedback.
        # Positive target_y shifts COM/body over the left stance foot for right-foot lift.
        #
        # V2 fix:
        # v1 achieved real air-time, but y drift reached about +0.36 and caused
        # y_position_limit before landing. The previous lateral feedforward was
        # independent of target_lateral_shift, so even yT=0.020 still used a large
        # roll command. V2 scales roll feedforward by the requested target_y and
        # adds stronger anti-y feedback.
        y_error = y - target_y
        ycorr = float(np.clip(1.65 * y_error + 0.50 * yv, -0.16, 0.16)) + lateral_corr_res

        # V3 sagittal capture:
        # v2 reached real 3.5 cm lift, but the root fell backward:
        # x ~= -0.30, xv ~= -0.89. This stronger PD term acts through the
        # left support hip/ankle/waist to resist backward collapse during swing.
        xcorr = float(np.clip(-1.95 * x - 0.72 * xv, -0.18, 0.18)) + sagittal_res

        shift_scale = float(np.clip(abs(target_y) / 0.050, 0.0, 1.0))
        lat = sh * shift_scale

        # Left support-leg and waist roll shift.
        # Reduced and target-scaled. This keeps enough unloading for right-foot
        # air-time while avoiding uncontrolled positive-y drift.
        off[1] += -0.13 * lat + 0.20 * ycorr       # left hip roll
        off[5] += +0.075 * lat - 0.13 * ycorr      # left ankle roll
        off[7] += -0.045 * lat + 0.07 * ycorr      # right hip roll preload
        off[11] += +0.035 * lat - 0.055 * ycorr    # right ankle roll preload
        off[13] += +0.075 * lat - 0.16 * ycorr     # waist roll

        # Sagittal stabilizer through stance leg and waist pitch.
        off[0] += 1.55 * xcorr                    # left hip pitch
        off[4] += -1.05 * xcorr                   # left ankle pitch
        off[14] += -0.60 * xcorr                  # waist pitch

        # Swing-leg feedforward. V3 keeps visible lift but reduces the backward
        # pull seen in v2 by using slightly less hip pitch and a slower swing.
        off[6] += +0.010 * sw                     # right hip pitch
        off[9] += +0.40 * sw                      # right knee bend
        off[10] += +0.24 * sw                     # right ankle pitch
        off[7] += -0.030 * sw                     # right hip roll

        target += off

        # Task-space IK: directly move right_foot site upward.
        # Use right leg only: right hip pitch/roll/yaw, knee, ankle pitch/roll.
        right_leg_idx = [6, 7, 8, 9, 10, 11]
        dofs = [self.qvel_adrs[i] for i in right_leg_idx]

        current_foot = self.data.site_xpos[self.right_foot_site].copy()
        desired_foot = self._target_foot_position(action, info)
        err = desired_foot - current_foot
        # V3: do not pin swing-foot XY strongly. Strong XY holding can drag the
        # pelvis/root backward while the foot is being lifted. Keep Z dominant.
        err[0] *= float(self.cfg.xy_hold_weight)
        err[1] *= float(self.cfg.xy_hold_weight)
        err[2] *= float(self.cfg.z_lift_weight)

        if sw > 0.02:
            dq = self._ik_delta(self.right_foot_site, dofs, err)
            current_q = np.array([self.data.qpos[self.qpos_adrs[i]] for i in right_leg_idx], dtype=np.float64)
            ik_target = current_q + dq
            for local_i, joint_i in enumerate(right_leg_idx):
                # Blend IK with feedforward target. Higher IK influence during mid-swing.
                blend = 0.35 + 0.55 * sw
                target[joint_i] = (1.0 - blend) * target[joint_i] + blend * ik_target[local_i]

        # Clip to actuator ranges.
        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])

        return target

    def step(self, action=None):
        if action is None:
            action = np.zeros(4, dtype=np.float32)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        info = self._get_info()
        target = self._target_joint_position(action, info)

        self.data.ctrl[:] = self.default_ctrl
        for j, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(target[j])

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.episode_step += 1
        info = self._get_info()
        reward, rinfo = self._compute_reward(action, info)
        info.update(rinfo)

        terminated = self._terminated(info)
        truncated = self.episode_step >= self.cfg.max_steps
        self.prev_action = action.copy()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_info(self) -> Dict[str, float]:
        phi = self._phase()
        sw = self._swing_env(phi)
        sh = self._shift_env(phi)

        lpos = self.data.site_xpos[self.left_foot_site].copy()
        rpos = self.data.site_xpos[self.right_foot_site].copy()
        lclr = max(0.0, float(lpos[2] - self.left_foot_p0[2]))
        rclr = max(0.0, float(rpos[2] - self.right_foot_p0[2]))

        lc = self._foot_contact(self.left_foot_body)
        rc = self._foot_contact(self.right_foot_body)

        target_y = self.cfg.target_lateral_shift * sh
        target_clear = self.cfg.target_clearance * sw

        # During visible lift, right foot should lose contact.
        right_expected_contact = target_clear < 0.012 or sw < 0.10
        left_expected_contact = True

        contact_acc = 0.5 * (
            float(bool(lc) == bool(left_expected_contact)) +
            float(bool(rc) == bool(right_expected_contact))
        )

        lvel = self._site_linear_velocity(self.left_foot_site)
        rvel = self._site_linear_velocity(self.right_foot_site)

        return {
            "episode_step": float(self.episode_step),
            "phase": float(phi),
            "swing_env": float(sw),
            "shift_env": float(sh),
            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "z_velocity": float(self.data.qvel[2]),
            "root_ang_vel": float(np.linalg.norm(self.data.qvel[3:6])),
            "up_z": float(self._root_up_z()),
            "left_foot_x": float(lpos[0]),
            "left_foot_y": float(lpos[1]),
            "left_foot_z": float(lpos[2]),
            "right_foot_x": float(rpos[0]),
            "right_foot_y": float(rpos[1]),
            "right_foot_z": float(rpos[2]),
            "left_foot_clearance": float(lclr),
            "right_foot_clearance": float(rclr),
            "main_clearance": float(rclr),
            "main_target_clearance": float(target_clear),
            "target_y_offset": float(target_y),
            "left_contact": bool(lc),
            "right_contact": bool(rc),
            "left_expected_contact": bool(left_expected_contact),
            "right_expected_contact": bool(right_expected_contact),
            "contact_accuracy": float(contact_acc),
            "left_foot_slip": float(np.linalg.norm(lvel[:2])),
            "right_foot_slip": float(np.linalg.norm(rvel[:2])),
            "support_slip": float(np.linalg.norm(lvel[:2])),
        }

    def _get_obs(self) -> np.ndarray:
        info = self._get_info()
        jp = np.array([self.data.qpos[i] for i in self.qpos_adrs], dtype=np.float32)
        jv = np.array([self.data.qvel[i] for i in self.qvel_adrs], dtype=np.float32)

        base = np.array([
            info["phase"], info["swing_env"], info["shift_env"],
            info["base_height"], info["x_position"], info["y_position"],
            info["x_velocity"], info["y_velocity"], info["z_velocity"],
            info["root_ang_vel"], info["up_z"],
            info["main_clearance"], info["main_target_clearance"], info["target_y_offset"],
            float(info["left_contact"]), float(info["right_contact"]),
            float(info["left_expected_contact"]), float(info["right_expected_contact"]),
            info["contact_accuracy"], info["support_slip"],
        ], dtype=np.float32)

        return np.concatenate([base, jp - self.stand_joint_pos.astype(np.float32), jv, self.prev_action]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        # Reward is diagnostic only. Do not call success based on reward.
        clear = float(info["main_clearance"])
        target_clear = float(info["main_target_clearance"])
        up = float(info["up_z"])
        x = float(info["x_position"])
        y = float(info["y_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        contact = float(info["contact_accuracy"])
        slip = float(info["support_slip"])

        lift_progress = min(clear / max(target_clear, 1e-6), 1.0) if target_clear > 0.001 else 1.0
        lift_miss = max(0.0, target_clear - clear)

        reward_lift = 25.0 * lift_progress - 450.0 * lift_miss
        reward_root = (
            3.0 * max(up, 0.0)
            - 12.0 * abs(x)
            - 8.0 * abs(xv)
            - 10.0 * abs(y - float(info["target_y_offset"]))
            - 6.0 * abs(yv)
            - 0.8 * float(info["root_ang_vel"])
        )
        reward_contact = 7.0 * contact - 4.0 * min(slip, 1.0)
        effort = -0.01 * float(np.sum(np.square(action)))
        total = reward_lift + reward_root + reward_contact + effort
        return float(total), {
            "reward_lift": float(reward_lift),
            "reward_root": float(reward_root),
            "reward_contact": float(reward_contact),
            "reward_effort": float(effort),
            "reward_total": float(total),
            "reward_version": "taskspace_right_lift_v3_sagittal_capture",
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        if float(info["base_height"]) < self.cfg.min_height or float(info["base_height"]) > self.cfg.max_height:
            return True
        if float(info["up_z"]) < self.cfg.min_up_z:
            return True
        if abs(float(info["x_position"])) > self.cfg.x_hard_limit:
            return True
        if abs(float(info["y_position"])) > self.cfg.y_hard_limit:
            return True
        if abs(float(info["x_velocity"])) > self.cfg.x_velocity_hard_limit:
            return True
        if abs(float(info["y_velocity"])) > self.cfg.y_velocity_hard_limit:
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        if float(info["base_height"]) < self.cfg.min_height:
            return "base_height_low"
        if float(info["base_height"]) > self.cfg.max_height:
            return "base_height_high"
        if float(info["up_z"]) < self.cfg.min_up_z:
            return "up_z_low"
        if abs(float(info["x_position"])) > self.cfg.x_hard_limit:
            return "x_position_limit"
        if abs(float(info["y_position"])) > self.cfg.y_hard_limit:
            return "y_position_limit"
        if abs(float(info["x_velocity"])) > self.cfg.x_velocity_hard_limit:
            return "x_velocity_limit"
        if abs(float(info["y_velocity"])) > self.cfg.y_velocity_hard_limit:
            return "y_velocity_limit"
        return "not_terminated"

    def close(self):
        pass
