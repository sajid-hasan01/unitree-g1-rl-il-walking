from __future__ import annotations

from typing import Dict, List

import mujoco
import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


class G1WBCTaskspaceRightLiftEnv(G1TaskspaceRightLiftEnv):
    """
    WBC-lite v3 capture-step controller for the existing 15-DOF G1 task-space env.

    Main project fix:
    - slow right-foot lift ramp
    - reference slew limiter
    - sagittal pre-commit before lift
    - support-foot anti-hop protection
    - P+I+D root/COM-x balance servo
    - capture touchdown when backward root momentum grows
    """

    def __init__(
        self,
        support_lock_weight: float = 0.62,
        support_xy_weight: float = 0.18,
        support_z_weight: float = 1.05,
        support_ik_gain: float = 0.45,
        support_ik_damping: float = 0.075,
        support_ik_max_delta: float = 0.075,
        torso_pitch_gain: float = 0.18,
        torso_roll_gain: float = 0.08,
        angvel_pitch_gain: float = 0.105,
        angvel_roll_gain: float = 0.045,
        height_gain: float = 0.16,
        height_target: float = 0.790,
        ref_slew: float = 0.085,
        x_target_forward: float = 0.020,
        x_int_gain: float = 0.55,
        x_int_limit: float = 0.045,
        capture_x_shift: float = -0.060,
        **kwargs,
    ):
        # Must exist before parent __init__, because parent calls self._get_obs().
        self.support_lock_weight = float(support_lock_weight)
        self.support_xy_weight = float(support_xy_weight)
        self.support_z_weight = float(support_z_weight)
        self.support_ik_gain = float(support_ik_gain)
        self.support_ik_damping = float(support_ik_damping)
        self.support_ik_max_delta = float(support_ik_max_delta)

        self.torso_pitch_gain = float(torso_pitch_gain)
        self.torso_roll_gain = float(torso_roll_gain)
        self.angvel_pitch_gain = float(angvel_pitch_gain)
        self.angvel_roll_gain = float(angvel_roll_gain)
        self.height_gain = float(height_gain)
        self.height_target = float(height_target)

        self.ref_slew = float(ref_slew)
        self.x_target_forward = float(x_target_forward)
        self.x_int_gain = float(x_int_gain)
        self.x_int_limit = float(x_int_limit)
        self.capture_x_shift = float(capture_x_shift)

        self._x_int = 0.0
        self._ref_prev = None
        self._capture_active = False
        self._abort_lift = False
        self._capture_count = 0
        self._abort_count = 0
        self._wbc_state = "PRELOAD"
        self._guard_count = 0
        self._touchdown_anchor_x = 0.0
        self._touchdown_timer = 0.0
        self._touchdown_force = 0.0

        # Conservative defaults for the existing 15-DOF parent env.
        kwargs.setdefault("max_steps", 520)
        kwargs.setdefault("cycle_duration", 5.8)

        # Slow lift: previous versions committed lift too fast and created
        # backward root impulse.
        kwargs.setdefault("shift_start", 0.08)
        kwargs.setdefault("swing_start", 0.42)
        kwargs.setdefault("swing_end", 0.70)
        kwargs.setdefault("land_end", 0.88)

        # Lower target; visible but not explosive.
        kwargs.setdefault("target_clearance", 0.018)
        kwargs.setdefault("target_lateral_shift", 0.012)

        # Lower IK authority + more XY hold to prevent foot flying away.
        kwargs.setdefault("ik_gain", 0.70)
        kwargs.setdefault("ik_damping", 0.065)
        kwargs.setdefault("ik_max_delta", 0.085)
        kwargs.setdefault("xy_hold_weight", 0.18)
        kwargs.setdefault("z_lift_weight", 0.75)

        kwargs.setdefault("x_hard_limit", 0.42)
        kwargs.setdefault("y_hard_limit", 0.30)
        kwargs.setdefault("x_velocity_hard_limit", 1.35)
        kwargs.setdefault("y_velocity_hard_limit", 1.35)
        kwargs.setdefault("min_up_z", 0.70)

        super().__init__(**kwargs)

    # ---------------------------- reset / reference smoothing ----------------------------

    def reset(self, *, seed=None, options=None):
        self._x_int = 0.0
        self._ref_prev = None
        self._capture_active = False
        self._abort_lift = False
        self._capture_count = 0
        self._abort_count = 0
        self._wbc_state = "PRELOAD"
        self._guard_count = 0
        self._touchdown_anchor_x = 0.0
        self._touchdown_timer = 0.0
        self._touchdown_force = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action=None):
        if action is None:
            action = np.zeros(4, dtype=np.float32)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        info = self._get_info()

        # WBC v5 hard state-machine update.
        self._update_wbc_state(info)
        info = self._get_info()

        target = self._target_joint_position(action, info)

        # Slew limit target reference to avoid swing impulse.
        if self._ref_prev is not None:
            delta = target - self._ref_prev
            target = self._ref_prev + np.clip(delta, -self.ref_slew, self.ref_slew)
        self._ref_prev = target.copy()

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

    # ---------------------------- helpers ----------------------------

    def _root_orientation_proxies(self):
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        up_x = float(mat[2])
        up_y = float(mat[5])
        up_z = float(mat[8])
        return up_x, up_y, up_z

    def _smooth_local(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _precommit_gain(self, phi: float) -> float:
        # Build forward commitment before main lift.
        start = 0.10
        end = max(0.11, float(self.cfg.swing_start) - 0.04)
        if phi <= start:
            return 0.0
        if phi >= end:
            return 1.0
        return self._smooth_local((phi - start) / max(end - start, 1e-6))

    def _support_ik_delta(self, site_id: int, dof_indices: List[int], task_error: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = jacp[:, dof_indices]
        damping = self.support_ik_damping
        A = J @ J.T + (damping * damping) * np.eye(3)
        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error
        delta = self.support_ik_gain * delta
        return np.clip(delta, -self.support_ik_max_delta, self.support_ik_max_delta)

    # ---------------------------- hard state machine ----------------------------

    def _update_wbc_state(self, info: Dict[str, float]) -> None:
        """
        WBC v5 hard state machine.

        PRELOAD/LIFT/HOLD are nominal.
        TOUCHDOWN is a hard takeover:
        - parent lift is no longer allowed to keep the foot high
        - right knee is extended
        - right ankle is plantarflexed
        - foot target is anchored to CURRENT left support foot
        """
        phi = float(info["phase"])
        sw = float(info["swing_env"])
        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        up_z = float(info["up_z"])

        left_contact = bool(info["left_contact"])
        right_contact = bool(info["right_contact"])

        # Nominal phase label.
        if self._wbc_state not in ("TOUCHDOWN", "RECOVERY", "SETTLE"):
            if phi < float(self.cfg.swing_start):
                self._wbc_state = "PRELOAD"
            elif sw < 0.95:
                self._wbc_state = "LIFT"
            else:
                self._wbc_state = "HOLD"

        # Once touchdown starts, keep it active until right contact.
        if self._wbc_state == "TOUCHDOWN":
            self._touchdown_timer += self.dt
            self._touchdown_force = float(np.clip(self._touchdown_force + 0.16, 0.0, 1.0))

            if right_contact:
                self._wbc_state = "RECOVERY"
                self._touchdown_timer = 0.0
                self._touchdown_force = 1.0
            return

        if self._wbc_state == "RECOVERY":
            self._touchdown_timer += self.dt
            self._touchdown_force = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            if self._touchdown_timer >= 0.30:
                self._wbc_state = "SETTLE"
            return

        if self._wbc_state == "SETTLE":
            self._touchdown_force = 0.0
            return

        # Capture guard: trigger EARLY, not near xv=-0.98.
        # Use 3 consecutive control steps to avoid noise.
        guard = False

        # Momentum guard.
        if (not right_contact or sw > 0.35) and xv < -0.30:
            guard = True

        # Position guard.
        if (not right_contact or sw > 0.35) and x < -0.080:
            guard = True

        # Collapse guard.
        if up_z < 0.88 and sw > 0.25:
            guard = True

        # Support-foot unload guard: do not keep lifting if support foot hops.
        if (not left_contact) and sw > 0.15:
            guard = True

        if guard:
            self._guard_count += 1
        else:
            self._guard_count = max(0, self._guard_count - 1)

        if self._guard_count >= 3:
            self._wbc_state = "TOUCHDOWN"
            self._capture_active = True
            self._abort_lift = True
            if self._capture_count == 0:
                self._capture_count = 1

            # Anchor once, in CURRENT support-foot frame.
            # This fixes the old bug where foot returned to a stale/start frame.
            self._touchdown_anchor_x = float(info["left_foot_x"] + 0.010)
            self._touchdown_timer = 0.0
            self._touchdown_force = 0.20

    # ---------------------------- capture foot placement ----------------------------

    def _target_foot_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_foot_position(action, info)

        phi = float(info["phase"])
        xv = float(info["x_velocity"])
        x = float(info["x_position"])
        up_z = float(info["up_z"])

        # Capture condition: root is moving backward or torso is losing posture.
        capture_strength = 0.0
        if phi > 0.46 and (self._capture_active or xv < -0.32 or x < -0.090 or up_z < 0.92):
            capture_strength = float(np.clip(
                max(0.0, -xv - 0.25) * 1.25
                + max(0.0, -x - 0.070) * 2.00
                + max(0.0, 0.92 - up_z) * 2.00,
                0.0,
                1.0,
            ))

        if capture_strength > 0.0:
            # Observed fall direction is negative x in our logs.
            # Land the right foot slightly in that direction to catch the body.
            target[0] += self.capture_x_shift * capture_strength

            # Force touchdown when capture is active.
            target[2] -= 0.035 * capture_strength
            target[2] = max(float(self.right_foot_p0[2]), float(target[2]))

        # WBC v5 hard touchdown foot target.
        if self._wbc_state in ("TOUCHDOWN", "RECOVERY"):
            land = float(np.clip(self._touchdown_force, 0.0, 1.0))

            # Anchor to current support-foot frame captured at trigger.
            target[0] = (1.0 - land) * float(target[0]) + land * float(self._touchdown_anchor_x)

            # Force touchdown; do not allow parent lift to keep z high.
            ground_z = float(self.right_foot_p0[2] - 0.020)
            target[2] = (1.0 - land) * float(target[2]) + land * ground_z

        if self._abort_lift:
            target[2] = min(float(target[2]), float(self.right_foot_p0[2]))

        return target

    # ---------------------------- WBC target ----------------------------

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        phi = float(info["phase"])

        # Bake in the best residual direction found from tests:
        # reduce lift, cancel unstable sagittal bias, add lateral correction.
        action_for_parent = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        if phi >= 0.38:
            action_for_parent[0] = np.clip(action_for_parent[0] - 0.70, -1.0, 1.0)
            action_for_parent[2] = np.clip(action_for_parent[2] - 0.70, -1.0, 1.0)
            action_for_parent[3] = np.clip(action_for_parent[3] + 0.60, -1.0, 1.0)

        target = super()._target_joint_position(action_for_parent, info)

        sw = float(info["swing_env"])
        sh = float(info["shift_env"])
        x = float(info["x_position"])
        y = float(info["y_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        h = float(info["base_height"])

        up_x, up_y, up_z = self._root_orientation_proxies()
        av = np.asarray(self.data.qvel[3:6], dtype=np.float64)

        # 1) Sagittal pre-commit before lift.
        pre = self._precommit_gain(phi)
        target[0] += -0.030 * pre       # left hip extension
        target[4] += -0.040 * pre       # left ankle plantarflexion
        target[14] += +0.035 * pre      # small forward trunk lean

        # 2) P+I+D root-x servo. Positive u = push body forward.
        x_target = self.x_target_forward * max(pre, sh)
        err = x_target - x
        self._x_int = float(np.clip(self._x_int + err * self.dt, -self.x_int_limit, self.x_int_limit))
        u = float(np.clip(1.15 * err + self.x_int_gain * self._x_int - 0.42 * xv, -0.14, 0.14))

        # Do not over-push when support foot is already lifting.
        if float(info["left_foot_clearance"]) > 0.012:
            u = min(u, 0.035)

        target[0] += +0.42 * u
        target[4] += -0.34 * u
        target[14] += -0.22 * u

        # 3) Torso/base upright and pelvis height protection.
        pitch_task = np.clip(
            -self.torso_pitch_gain * up_y
            - self.angvel_pitch_gain * av[1]
            - 0.070 * xv
            - 0.025 * x,
            -0.085,
            0.085,
        )
        roll_task = np.clip(
            -self.torso_roll_gain * up_x
            - self.angvel_roll_gain * av[0]
            - 0.040 * yv
            - 0.030 * (y - float(info["target_y_offset"])),
            -0.065,
            0.065,
        )
        height_task = np.clip(self.height_gain * (self.height_target - h), -0.050, 0.065)

        target[14] += -0.45 * pitch_task
        target[0] += +0.36 * pitch_task
        target[4] += -0.30 * pitch_task

        target[13] += -0.45 * roll_task
        target[1] += +0.30 * roll_task
        target[5] += -0.22 * roll_task

        support_env = min(1.0, max(0.0, 0.35 * sh + 0.65 * sw))
        target[3] += +0.020 * support_env + 0.34 * height_task
        target[0] += +0.13 * height_task
        target[4] += -0.12 * height_task

        # 4) Support-foot task-space lock, but not too strong; previous strong lock hopped.
        if support_env > 0.02:
            left_leg_idx = [0, 1, 2, 3, 4, 5]
            left_dofs = [self.qvel_adrs[i] for i in left_leg_idx]
            current_left = self.data.site_xpos[self.left_foot_site].copy()
            desired_left = self.left_foot_p0.copy()
            err_site = desired_left - current_left
            err_site[0] *= self.support_xy_weight
            err_site[1] *= self.support_xy_weight
            err_site[2] *= self.support_z_weight
            dq = self._support_ik_delta(self.left_foot_site, left_dofs, err_site)
            current_q = np.array([self.data.qpos[self.qpos_adrs[i]] for i in left_leg_idx], dtype=np.float64)
            ik_target = current_q + dq
            blend = self.support_lock_weight * support_env
            for local_i, joint_i in enumerate(left_leg_idx):
                target[joint_i] = (1.0 - blend) * target[joint_i] + blend * ik_target[local_i]

        # 5) Anti-hop / capture touchdown.
        capture = float(np.clip(
            max(0.0, -xv - 0.25) * 1.30
            + max(0.0, -x - 0.075) * 2.00
            + max(0.0, 0.92 - up_z) * 2.00,
            0.0,
            1.0,
        ))

        if phi > 0.48 and (capture > 0.02 or self._capture_active or self._abort_lift):
            if self._abort_lift:
                capture = max(capture, 0.75)

            target[9] += -0.36 * capture       # right knee extension
            target[10] += -0.25 * capture      # right ankle down
            target[6] += -0.055 * capture      # right hip capture placement

            # Support push while touchdown happens.
            target[0] += +0.22 * capture
            target[4] += -0.18 * capture
            target[14] += -0.12 * capture

        # WBC v5 hard state-machine TOUCHDOWN takeover.
        # During TOUCHDOWN, parent lift must lose authority completely.
        if self._wbc_state in ("TOUCHDOWN", "RECOVERY"):
            land = float(np.clip(self._touchdown_force, 0.0, 1.0))
            pelvis_x = float(self.data.qpos[0])

            # Desired right-foot x in current support-foot frame.
            dx = float((self._touchdown_anchor_x - pelvis_x) / 0.36)
            hip_capture = float(np.clip(dx, -0.25, 0.35))

            # Absolute touchdown posture from the control analysis:
            # knee extension is the critical part that actually drops the foot.
            target[6] = (1.0 - land) * target[6] + land * hip_capture
            target[7] = (1.0 - land) * target[7] + land * self.stand_joint_pos[7]
            target[8] = (1.0 - land) * target[8] + land * self.stand_joint_pos[8]
            target[9] = (1.0 - land) * target[9] + land * 0.030
            target[10] = (1.0 - land) * target[10] + land * (-0.120)
            target[11] = (1.0 - land) * target[11] + land * self.stand_joint_pos[11]

            # Soft pelvis/support press using position-control joints.
            # This replaces torque-level downward force.
            target[3] += +0.06 * land      # left knee slight flexion/press
            target[0] += -0.05 * land      # left hip extension
            target[4] += -0.06 * land      # left ankle plantarflexion
            target[14] += +0.04 * land     # forward trunk during touchdown

        # WBC v5 RECOVERY: both-feet braking after right_contact.
        if self._wbc_state == "RECOVERY":
            rec = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            target[0] += -0.06 * rec
            target[4] += -0.08 * rec
            target[6] += -0.06 * rec
            target[10] += -0.08 * rec
            target[14] += +0.05 * rec

        # Anti-hop caps: keep support leg from over-pushing itself off the floor.
        target[4] = np.clip(target[4], self.stand_joint_pos[4] - 0.16, self.stand_joint_pos[4] + 0.12)
        target[14] = np.clip(target[14], self.stand_joint_pos[14] - 0.12, self.stand_joint_pos[14] + 0.12)

        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(target[i], self.ctrl_low[aid], self.ctrl_high[aid])

        return target

    # ---------------------------- info / reward ----------------------------

    def _get_info(self) -> Dict[str, float]:
        info = super()._get_info()
        up_x, up_y, up_z = self._root_orientation_proxies()
        info["torso_up_x"] = float(up_x)
        info["torso_up_y"] = float(up_y)
        info["wbc_support_lock_weight"] = float(self.support_lock_weight)
        info["wbc_height_target"] = float(self.height_target)
        info["wbc_capture_active"] = bool(self._capture_active)
        info["wbc_abort_lift"] = bool(self._abort_lift)
        info["wbc_capture_count"] = float(self._capture_count)
        info["wbc_abort_count"] = float(self._abort_count)
        info["wbc_x_integral"] = float(self._x_int)
        info["wbc_state"] = str(self._wbc_state)
        info["wbc_guard_count"] = float(self._guard_count)
        info["wbc_touchdown_anchor_x"] = float(self._touchdown_anchor_x)
        info["wbc_touchdown_timer"] = float(self._touchdown_timer)
        info["wbc_touchdown_force"] = float(self._touchdown_force)
        return info

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]):
        total, rinfo = super()._compute_reward(action, info)

        upright_bonus = 6.0 * max(float(info["up_z"]), 0.0)
        angular_penalty = -1.7 * float(info["root_ang_vel"])
        height_penalty = -9.0 * abs(float(info["base_height"]) - self.height_target)

        # COM/root-x target tracking. This is diagnostic; evaluation remains metric-based.
        x_target = self.x_target_forward * max(float(info["shift_env"]), self._precommit_gain(float(info["phase"])))
        x_err = x_target - float(info["x_position"])
        backward_penalty = -8.0 * max(0.0, x_err - 0.035)

        support_hop_penalty = -18.0 * max(0.0, float(info["left_foot_clearance"]) - 0.012)

        wbc_total = float(total + upright_bonus + angular_penalty + height_penalty + backward_penalty + support_hop_penalty)

        rinfo["reward_wbc_upright"] = float(upright_bonus)
        rinfo["reward_wbc_angvel"] = float(angular_penalty)
        rinfo["reward_wbc_height"] = float(height_penalty)
        rinfo["reward_wbc_back_x"] = float(backward_penalty)
        rinfo["reward_wbc_support_hop"] = float(support_hop_penalty)
        rinfo["reward_total"] = wbc_total
        rinfo["reward_version"] = "wbc_taskspace_right_lift_v5_hard_state_touchdown"
        return wbc_total, rinfo
