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
class GuardedTaskspaceRightLiftConfig:
    model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml"
    frame_skip: int = 5
    max_steps: int = 520
    cycle_duration: float = 5.2

    # finite-state phase boundaries as fractions of cycle_duration
    settle_end: float = 0.14
    shift_end: float = 0.42
    guard_end: float = 0.50
    lift_end: float = 0.60
    land_end: float = 0.74
    recover_end: float = 1.00

    target_clearance: float = 0.030
    target_lateral_shift: float = 0.018

    ik_gain: float = 1.10
    ik_damping: float = 0.045
    ik_max_delta: float = 0.20
    xy_hold_weight: float = 0.05
    z_lift_weight: float = 1.20

    guard_x_abs: float = 0.080
    guard_y_abs: float = 0.060
    guard_x_velocity_abs: float = 0.280
    guard_y_velocity_abs: float = 0.350
    abort_x_velocity: float = -0.55
    abort_x_position: float = -0.18
    abort_up_z: float = 0.88

    x_hard_limit: float = 0.36
    y_hard_limit: float = 0.32
    x_velocity_hard_limit: float = 1.35
    y_velocity_hard_limit: float = 1.35
    min_up_z: float = 0.70
    min_height: float = 0.55
    max_height: float = 1.05

    randomize_reset: bool = False


class G1GuardedTaskspaceRightLiftEnv(gym.Env):
    """
    Guarded finite-state task-space right-foot lift.

    States:
    0 settle, 1 shift, 2 guard, 3 short lift, 4 forced landing, 5 recover.

    The controller will not lift unless the guard stage sees a stable root.
    During lift, it aborts and lands immediately when backward root velocity,
    backward x position, or torso tilt becomes unsafe.

    Action is optional 4D residual:
      [lift_height, lateral_shift, sagittal_capture, lateral_capture]
    Use fixed_action=0,0,0,0 for deterministic control.
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = GuardedTaskspaceRightLiftConfig(**kwargs)
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
            raise RuntimeError("Missing G1 foot sites/bodies: left_foot, right_foot, ankle_roll_link bodies.")

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.episode_step = 0
        self.abort_lift = False
        self.guard_passed_once = False
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
        self.abort_lift = False
        self.guard_passed_once = False
        self.prev_action[:] = 0.0
        self._set_state_to_stand()

        if self.cfg.randomize_reset:
            noise = self.np_random.normal(0.0, 0.002, len(self.qpos_adrs))
            for qadr, eps in zip(self.qpos_adrs, noise):
                self.data.qpos[qadr] += float(eps)
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
        t = self.episode_step * self.dt
        return float(np.clip(t / max(self.cfg.cycle_duration, 1e-6), 0.0, 1.0))

    def _state_id(self, phi: float) -> int:
        if phi < self.cfg.settle_end:
            return 0
        if phi < self.cfg.shift_end:
            return 1
        if phi < self.cfg.guard_end:
            return 2
        if phi < self.cfg.lift_end:
            return 3
        if phi < self.cfg.land_end:
            return 4
        return 5

    @staticmethod
    def _state_name(state_id: int) -> str:
        return {0: "settle", 1: "shift", 2: "guard", 3: "lift", 4: "land", 5: "recover"}.get(int(state_id), "unknown")

    def _shift_env(self, phi: float) -> float:
        if phi < self.cfg.settle_end:
            return 0.0
        if phi < self.cfg.shift_end:
            u = (phi - self.cfg.settle_end) / max(self.cfg.shift_end - self.cfg.settle_end, 1e-6)
            return self._smoothstep(u)
        if phi < self.cfg.lift_end:
            return 1.0
        if phi < self.cfg.land_end:
            u = (phi - self.cfg.lift_end) / max(self.cfg.land_end - self.cfg.lift_end, 1e-6)
            return 1.0 - 0.75 * self._smoothstep(u)
        u = (phi - self.cfg.land_end) / max(self.cfg.recover_end - self.cfg.land_end, 1e-6)
        return 0.25 * (1.0 - self._smoothstep(u))

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

    def _ik_delta(self, site_id: int, dof_indices: List[int], task_error: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = jacp[:, dof_indices]
        damping = float(self.cfg.ik_damping)
        A = J @ J.T + (damping * damping) * np.eye(3)
        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error
        return np.clip(float(self.cfg.ik_gain) * delta, -self.cfg.ik_max_delta, self.cfg.ik_max_delta)

    def _guard_ok(self, info: Dict[str, float]) -> bool:
        return (
            abs(float(info["x_position"])) <= self.cfg.guard_x_abs
            and abs(float(info["y_position"])) <= self.cfg.guard_y_abs
            and abs(float(info["x_velocity"])) <= self.cfg.guard_x_velocity_abs
            and abs(float(info["y_velocity"])) <= self.cfg.guard_y_velocity_abs
            and float(info["up_z"]) >= 0.96
            and bool(info["left_contact"])
            and bool(info["right_contact"])
        )

    def _swing_env_from_phase(self, phi: float, info: Dict[str, float]) -> float:
        state_id = self._state_id(phi)
        if self.abort_lift or state_id != 3 or not self.guard_passed_once:
            return 0.0
        if (
            float(info["x_velocity"]) < self.cfg.abort_x_velocity
            or float(info["x_position"]) < self.cfg.abort_x_position
            or float(info["up_z"]) < self.cfg.abort_up_z
        ):
            return 0.0
        u = (phi - self.cfg.guard_end) / max(self.cfg.lift_end - self.cfg.guard_end, 1e-6)
        return float(math.sin(math.pi * np.clip(u, 0.0, 1.0)))

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        sw = float(info["swing_env"])
        sh = float(info["shift_env"])
        state_id = int(info["state_id"])

        lift_res = 0.010 * float(action[0])
        lateral_res = 0.010 * float(action[1])
        sagittal_res = 0.030 * float(action[2])
        lateral_capture_res = 0.030 * float(action[3])

        x = float(info["x_position"])
        y = float(info["y_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        target_y = (self.cfg.target_lateral_shift + lateral_res) * sh

        yerr = y - target_y
        ycorr = float(np.clip(1.35 * yerr + 0.38 * yv, -0.13, 0.13)) + lateral_capture_res

        # Moderate sagittal capture. Previous v3 over-corrected and made root fall faster.
        if state_id in (3, 4):
            xcorr = float(np.clip(-1.20 * x - 0.75 * xv, -0.16, 0.16)) + sagittal_res
        else:
            xcorr = float(np.clip(-0.90 * x - 0.58 * xv, -0.13, 0.13)) + sagittal_res

        shift_scale = float(np.clip(abs(target_y) / 0.050, 0.0, 1.0))
        lat = sh * shift_scale

        target = self.stand_joint_pos.copy()
        off = np.zeros(15, dtype=np.float64)

        # Lateral left-support shift, target-scaled and feedback-stabilized.
        off[1] += -0.115 * lat + 0.16 * ycorr
        off[5] += +0.070 * lat - 0.105 * ycorr
        off[7] += -0.040 * lat + 0.060 * ycorr
        off[11] += +0.030 * lat - 0.045 * ycorr
        off[13] += +0.070 * lat - 0.130 * ycorr

        # Sagittal capture through stance leg and waist.
        off[0] += 1.10 * xcorr
        off[4] += -0.85 * xcorr
        off[14] += -0.42 * xcorr

        # Short swing feedforward. IK remains the main lift mechanism.
        off[6] += +0.006 * sw
        off[9] += +0.34 * sw
        off[10] += +0.22 * sw
        off[7] += -0.025 * sw

        target += off

        desired = self.right_foot_p0.copy()
        desired[2] += max(0.0, self.cfg.target_clearance + lift_res) * sw
        current = self.data.site_xpos[self.right_foot_site].copy()
        err = desired - current
        err[0] *= float(self.cfg.xy_hold_weight)
        err[1] *= float(self.cfg.xy_hold_weight)
        err[2] *= float(self.cfg.z_lift_weight)

        if sw > 0.02:
            right_leg_idx = [6, 7, 8, 9, 10, 11]
            dofs = [self.qvel_adrs[i] for i in right_leg_idx]
            dq = self._ik_delta(self.right_foot_site, dofs, err)
            current_q = np.array([self.data.qpos[self.qpos_adrs[i]] for i in right_leg_idx], dtype=np.float64)
            ik_target = current_q + dq
            blend = 0.55 + 0.35 * sw
            for local_i, joint_i in enumerate(right_leg_idx):
                target[joint_i] = (1.0 - blend) * target[joint_i] + blend * ik_target[local_i]

        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])
        return target

    def step(self, action=None):
        if action is None:
            action = np.zeros(4, dtype=np.float32)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        info = self._get_info()
        if int(info["state_id"]) == 2 and self._guard_ok(info):
            self.guard_passed_once = True
        if int(info["state_id"]) == 3:
            if (
                float(info["x_velocity"]) < self.cfg.abort_x_velocity
                or float(info["x_position"]) < self.cfg.abort_x_position
                or float(info["up_z"]) < self.cfg.abort_up_z
            ):
                self.abort_lift = True

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
        state_id = self._state_id(phi)
        sh = self._shift_env(phi)

        lc = self._foot_contact(self.left_foot_body)
        rc = self._foot_contact(self.right_foot_body)
        base_info = {
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "up_z": float(self._root_up_z()),
            "left_contact": bool(lc),
            "right_contact": bool(rc),
        }
        sw = self._swing_env_from_phase(phi, base_info)
        target_y = self.cfg.target_lateral_shift * sh
        target_clear = self.cfg.target_clearance * sw

        lpos = self.data.site_xpos[self.left_foot_site].copy()
        rpos = self.data.site_xpos[self.right_foot_site].copy()
        lclr = max(0.0, float(lpos[2] - self.left_foot_p0[2]))
        rclr = max(0.0, float(rpos[2] - self.right_foot_p0[2]))

        left_expected_contact = True
        right_expected_contact = not (target_clear >= 0.020 and not self.abort_lift)
        contact_accuracy = 0.5 * (
            float(bool(lc) == bool(left_expected_contact)) +
            float(bool(rc) == bool(right_expected_contact))
        )

        lvel = self._site_linear_velocity(self.left_foot_site)
        rvel = self._site_linear_velocity(self.right_foot_site)

        info = {
            "episode_step": float(self.episode_step),
            "phase": float(phi),
            "state_id": float(state_id),
            "state_name": self._state_name(state_id),
            "abort_lift": bool(self.abort_lift),
            "guard_passed_once": bool(self.guard_passed_once),
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
            "main_clearance": float(rclr),
            "main_target_clearance": float(target_clear),
            "target_y_offset": float(target_y),
            "left_contact": bool(lc),
            "right_contact": bool(rc),
            "left_expected_contact": bool(left_expected_contact),
            "right_expected_contact": bool(right_expected_contact),
            "contact_accuracy": float(contact_accuracy),
            "left_foot_slip": float(np.linalg.norm(lvel[:2])),
            "right_foot_slip": float(np.linalg.norm(rvel[:2])),
            "support_slip": float(np.linalg.norm(lvel[:2])),
        }
        info["guard_ok_now"] = bool(self._guard_ok(info))
        return info

    def _get_obs(self) -> np.ndarray:
        info = self._get_info()
        jp = np.array([self.data.qpos[i] for i in self.qpos_adrs], dtype=np.float32)
        jv = np.array([self.data.qvel[i] for i in self.qvel_adrs], dtype=np.float32)
        base = np.array([
            info["phase"], info["state_id"], float(info["abort_lift"]), float(info["guard_passed_once"]),
            info["swing_env"], info["shift_env"], info["base_height"], info["x_position"], info["y_position"],
            info["x_velocity"], info["y_velocity"], info["z_velocity"], info["root_ang_vel"], info["up_z"],
            info["main_clearance"], info["main_target_clearance"], info["target_y_offset"],
            float(info["left_contact"]), float(info["right_contact"]),
            float(info["left_expected_contact"]), float(info["right_expected_contact"]),
            info["contact_accuracy"], info["support_slip"],
        ], dtype=np.float32)
        return np.concatenate([base, jp - self.stand_joint_pos.astype(np.float32), jv, self.prev_action]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
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
        reward_lift = 24.0 * lift_progress - 500.0 * lift_miss
        reward_root = (
            4.0 * max(up, 0.0) - 14.0 * abs(x) - 10.0 * abs(xv)
            - 11.0 * abs(y - float(info["target_y_offset"])) - 8.0 * abs(yv)
            - 0.9 * float(info["root_ang_vel"])
        )
        reward_contact = 7.0 * contact - 5.0 * min(slip, 1.0)
        reward_guard = 2.5 * float(info["guard_passed_once"]) - 1.5 * float(info["abort_lift"])
        effort = -0.01 * float(np.sum(np.square(action)))
        total = reward_lift + reward_root + reward_contact + reward_guard + effort
        return float(total), {
            "reward_lift": float(reward_lift),
            "reward_root": float(reward_root),
            "reward_contact": float(reward_contact),
            "reward_guard": float(reward_guard),
            "reward_effort": float(effort),
            "reward_total": float(total),
            "reward_version": "guarded_taskspace_right_lift_v1",
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
