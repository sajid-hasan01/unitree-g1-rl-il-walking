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
class HybridLiftConfig:
    model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml"
    stage: str = "right_lift"
    frame_skip: int = 5
    max_steps: int = 700
    cycle_duration: float = 3.6
    swing_start: float = 0.28
    swing_end: float = 0.78
    target_clearance: float = 0.012
    target_lateral_shift: float = 0.032
    x_soft_limit: float = 0.08
    x_hard_limit: float = 0.20
    x_velocity_soft_limit: float = 0.22
    x_velocity_hard_limit: float = 1.00
    action_smoothing: float = 0.70
    randomize_reset: bool = True

    # V6 fixed high-level controller bias from strict one-cycle sweep.
    # Best strict one-cycle command on v5 was:
    #   lift_amp=+0.25, lateral_shift=0.00, swing_shape=+0.25
    # Because v5 already had -0.95 lift/swing biases, the equivalent
    # built-in zero-action bias is approximately -0.70.
    bias_lift_amp: float = -0.70
    bias_lateral_shift: float = -0.35
    bias_torso: float = 0.00
    bias_stance_ankle: float = -0.85
    bias_stance_hip: float = 0.70
    bias_swing_shape: float = -0.70


class G1HybridWBCLiteLiftEnv(gym.Env):
    """
    Hybrid WBC-lite + residual PPO.

    Architectural change:
    - PPO no longer controls 15 joint targets directly.
    - PPO outputs 6 high-level residuals.
    - A low-level structured balance/lift controller creates the 15 joint targets.
    - Root-x is actively locked; foot-lift reward is meaningful only inside that constraint.
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = HybridLiftConfig(**kwargs)
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
            raise RuntimeError("Missing required foot site/body names.")

        self.stand_qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else self.data.qpos.copy()
        self.stand_joint_pos = np.array([self.stand_qpos[i] for i in self.qpos_adrs], dtype=np.float32)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            if 0 <= jid < self.model.njnt:
                qadr = int(self.model.jnt_qposadr[jid])
                if qadr < len(self.stand_qpos):
                    self.default_ctrl[aid] = float(self.stand_qpos[qadr])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.episode_step = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.smoothed_action = np.zeros(6, dtype=np.float32)
        self.left_foot_z0 = 0.0
        self.right_foot_z0 = 0.0
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
        self.smoothed_action[:] = 0.0
        self._set_state_to_stand()
        if self.cfg.randomize_reset:
            for qadr, n in zip(self.qpos_adrs, self.np_random.normal(0.0, 0.003, len(self.qpos_adrs))):
                self.data.qpos[qadr] += float(n)
            self.data.qvel[:] = self.np_random.normal(0.0, 0.006, self.model.nv)
            mujoco.mj_forward(self.model, self.data)
        self.left_foot_z0 = float(self.data.site_xpos[self.left_foot_site, 2])
        self.right_foot_z0 = float(self.data.site_xpos[self.right_foot_site, 2])
        info = self._get_info()
        return self._get_obs(), info

    def _phase(self) -> float:
        return float(((self.episode_step * self.dt) / self.cfg.cycle_duration) % 1.0)

    def _swing_env(self, phi: float) -> float:
        s, e = self.cfg.swing_start, self.cfg.swing_end
        if not (s <= phi < e):
            return 0.0
        u = (phi - s) / max(e - s, 1e-6)
        return float(0.5 * (1.0 - math.cos(2.0 * math.pi * u)))

    def _shift_env(self, phi: float) -> float:
        s, e = self.cfg.swing_start, self.cfg.swing_end
        if phi < s:
            return float(0.5 * (1.0 - math.cos(math.pi * phi / max(s, 1e-6))))
        if phi < e:
            return 1.0
        u = min((phi - e) / max(1.0 - e, 1e-6), 1.0)
        return float(0.5 * (1.0 + math.cos(math.pi * u)))

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
        """
        MuJoCo-version-safe site linear velocity.

        Some mujoco Python builds do not expose data.site_xvelp. The correct
        portable method is to compute the translational Jacobian of the site
        and multiply it by qvel.
        """
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp @ self.data.qvel

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        phi = float(info["phase"])
        sw = self._swing_env(phi)
        sh = self._shift_env(phi)
        alpha = float(np.clip(self.cfg.action_smoothing, 0.0, 0.98))
        self.smoothed_action = (alpha * self.smoothed_action + (1.0 - alpha) * action).astype(np.float32)

        # V6: convert the strict one-cycle sweep result into controller structure.
        # PPO now learns around this bias instead of rediscovering the sign.
        controller_bias = np.array(
            [
                self.cfg.bias_lift_amp,
                self.cfg.bias_lateral_shift,
                self.cfg.bias_torso,
                self.cfg.bias_stance_ankle,
                self.cfg.bias_stance_hip,
                self.cfg.bias_swing_shape,
            ],
            dtype=np.float32,
        )
        a = np.clip(self.smoothed_action + controller_bias, -1.0, 1.0)

        lift_gain = float(np.clip(1.0 + 0.45 * a[0], 0.55, 1.45))
        shift_gain = float(np.clip(1.0 + 0.35 * a[1], 0.65, 1.35))
        sagittal_bias = 0.045 * float(a[2])
        stance_ankle_res = 0.050 * float(a[3])
        stance_hip_res = 0.050 * float(a[4])
        swing_shape = float(np.clip(1.0 + 0.45 * a[5], 0.60, 1.45))

        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        y = float(info["y_position"])
        yv = float(info["y_velocity"])
        target_y = float(info["target_y_offset"])

        xcorr = float(np.clip(-1.35 * x - 0.45 * xv, -0.120, 0.120))

        # V5 lateral stabilizer.
        # V4 solved sagittal drift but failed at y_position_limit. This term
        # pushes against lateral over-shift using the same hip/ankle/waist-roll
        # channels that create the planned COM shift.
        y_error = y - target_y
        ycorr = float(np.clip(1.15 * y_error + 0.30 * yv, -0.100, 0.100))

        off = np.zeros(15, dtype=np.float64)

        if self.cfg.stage == "right_lift":
            # Left leg = stance, right leg = swing.
            off[1] += -0.16 * sh * shift_gain + 0.12 * ycorr
            off[5] += +0.09 * sh * shift_gain - 0.075 * ycorr
            off[7] += -0.055 * sh * shift_gain + 0.045 * ycorr
            off[11] += +0.045 * sh * shift_gain - 0.035 * ycorr
            off[13] += +0.10 * sh * shift_gain - 0.090 * ycorr
            # Knee/ankle-dominant swing to avoid throwing the root.
            off[6] += +0.008 * sw * lift_gain * swing_shape
            off[9] += +0.22 * sw * lift_gain * swing_shape
            off[10] += +0.13 * sw * lift_gain * swing_shape
            off[7] += -0.025 * sw * lift_gain
            # Root-x stabilizer on stance leg and waist.
            off[0] += 1.15 * xcorr + stance_hip_res + sagittal_bias
            off[4] += -0.85 * xcorr + stance_ankle_res
            off[14] += -0.45 * xcorr + 0.5 * sagittal_bias
        else:
            # Mirrored left-lift version.
            off[7] += +0.16 * sh * shift_gain - 0.12 * ycorr
            off[11] += -0.09 * sh * shift_gain + 0.075 * ycorr
            off[1] += +0.055 * sh * shift_gain - 0.045 * ycorr
            off[5] += -0.045 * sh * shift_gain + 0.035 * ycorr
            off[13] += -0.10 * sh * shift_gain + 0.090 * ycorr
            off[0] += +0.008 * sw * lift_gain * swing_shape
            off[3] += +0.22 * sw * lift_gain * swing_shape
            off[4] += +0.13 * sw * lift_gain * swing_shape
            off[1] += +0.025 * sw * lift_gain
            off[6] += 1.15 * xcorr + stance_hip_res + sagittal_bias
            off[10] += -0.65 * xcorr + stance_ankle_res
            off[14] += -0.45 * xcorr + 0.5 * sagittal_bias

        target = self.stand_joint_pos.astype(np.float64) + off
        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])
        return target

    def step(self, action):
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
        lz = float(self.data.site_xpos[self.left_foot_site, 2])
        rz = float(self.data.site_xpos[self.right_foot_site, 2])
        lclr = max(0.0, lz - self.left_foot_z0)
        rclr = max(0.0, rz - self.right_foot_z0)
        lc = self._foot_contact(self.left_foot_body)
        rc = self._foot_contact(self.right_foot_body)
        if self.cfg.stage == "right_lift":
            le, re = True, sw < 0.10
            main_clear = rclr
            target_y = self.cfg.target_lateral_shift * sh
        else:
            le, re = sw < 0.10, True
            main_clear = lclr
            target_y = -self.cfg.target_lateral_shift * sh
        target_clear = self.cfg.target_clearance * sw
        contact_acc = 0.5 * (float(bool(lc) == bool(le)) + float(bool(rc) == bool(re)))
        lslip = float(np.linalg.norm(self._site_linear_velocity(self.left_foot_site)[:2]))
        rslip = float(np.linalg.norm(self._site_linear_velocity(self.right_foot_site)[:2]))
        support_slip = lslip if self.cfg.stage == "right_lift" else rslip
        return {
            "stage": self.cfg.stage,
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
            "left_foot_clearance": float(lclr),
            "right_foot_clearance": float(rclr),
            "main_clearance": float(main_clear),
            "main_target_clearance": float(target_clear),
            "target_y_offset": float(target_y),
            "left_contact": bool(lc),
            "right_contact": bool(rc),
            "left_expected_contact": bool(le),
            "right_expected_contact": bool(re),
            "contact_accuracy": float(contact_acc),
            "left_foot_slip": float(lslip),
            "right_foot_slip": float(rslip),
            "support_slip": float(support_slip),
        }

    def _get_obs(self) -> np.ndarray:
        info = self._get_info()
        jp = np.array([self.data.qpos[i] for i in self.qpos_adrs], dtype=np.float32)
        jv = np.array([self.data.qvel[i] for i in self.qvel_adrs], dtype=np.float32)
        phi = float(info["phase"])
        base = np.array([
            info["base_height"], info["x_position"], info["y_position"],
            info["x_velocity"], info["y_velocity"], info["z_velocity"],
            info["root_ang_vel"], info["up_z"], math.sin(2*math.pi*phi), math.cos(2*math.pi*phi),
            info["swing_env"], info["shift_env"], info["main_clearance"], info["main_target_clearance"],
            info["target_y_offset"], float(info["left_contact"]), float(info["right_contact"]),
            info["contact_accuracy"], info["support_slip"],
        ], dtype=np.float32)
        return np.concatenate([base, jp - self.stand_joint_pos, jv, self.prev_action]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        h = float(info["base_height"])
        up = float(info["up_z"])
        x = float(info["x_position"])
        y = float(info["y_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        clear = float(info["main_clearance"])
        target = float(info["main_target_clearance"])
        contact = float(info["contact_accuracy"])
        slip = float(info["support_slip"])
        root_ang = float(info["root_ang_vel"])

        if target > 1e-5:
            progress = min(clear / max(target, 1e-6), 1.0)
            miss = max(0.0, target - clear)
            lift_reward = 22.0 * progress - 380.0 * miss
            if clear < 0.003 and target > 0.006:
                lift_reward -= 12.0
        else:
            lift_reward = 1.0 * math.exp(-650.0 * clear * clear)

        root_reward = (
            2.0 * max(up, 0.0)
            + 0.8 * math.exp(-40.0 * ((h - 0.79) ** 2))
            - 58.0 * max(0.0, abs(x) - self.cfg.x_soft_limit)
            - 14.0 * abs(x)
            - 10.0 * abs(xv)
            - 9.0 * abs(y - float(info["target_y_offset"]))
            - 3.5 * abs(yv)
            - 0.9 * root_ang
        )
        contact_reward = 6.0 * contact - 6.0 * (1.0 - contact) - 3.0 * min(slip, 1.5)
        smooth = -0.04 * float(np.sum(np.square(action - self.prev_action)))
        effort = -0.012 * float(np.sum(np.square(action)))
        total = root_reward + contact_reward + lift_reward + smooth + effort
        return float(total), {
            "reward_root": float(root_reward),
            "reward_contact": float(contact_reward),
            "reward_lift": float(lift_reward),
            "reward_smooth": float(smooth),
            "reward_effort": float(effort),
            "reward_total": float(total),
            "reward_version": "hybrid_wbc_lite_lift_v6_strict_onecycle_lift",
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        if float(info["base_height"]) < 0.54 or float(info["base_height"]) > 1.05:
            return True
        if float(info["up_z"]) < 0.62:
            return True
        if abs(float(info["x_position"])) > self.cfg.x_hard_limit:
            return True
        if abs(float(info["x_velocity"])) > self.cfg.x_velocity_hard_limit:
            return True
        if abs(float(info["y_position"])) > 0.42:
            return True
        if float(info["main_target_clearance"]) > 0.009 and float(info["main_clearance"]) < 0.0025 and float(info["swing_env"]) > 0.55:
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        if float(info["base_height"]) < 0.54:
            return "base_height_low"
        if float(info["base_height"]) > 1.05:
            return "base_height_high"
        if float(info["up_z"]) < 0.62:
            return "up_z_low"
        if abs(float(info["x_position"])) > self.cfg.x_hard_limit:
            return "x_position_limit"
        if abs(float(info["x_velocity"])) > self.cfg.x_velocity_hard_limit:
            return "x_velocity_limit"
        if abs(float(info["y_position"])) > 0.42:
            return "y_position_limit"
        if float(info["main_target_clearance"]) > 0.009 and float(info["main_clearance"]) < 0.0025 and float(info["swing_env"]) > 0.55:
            return "no_lift_mid_swing"
        return "not_terminated"

    def close(self):
        pass
