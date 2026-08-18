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
class DeterministicLeftLiftConfig:
    model_path: str = "third_party/mujoco_menagerie/unitree_g1/scene.xml"
    frame_skip: int = 5
    max_steps: int = 650
    cycle_duration: float = 5.8

    # Normalized one-shot sequence:
    # stand -> COM shift right -> lift left -> hold -> lower -> settle
    shift_start: float = 0.10
    shift_full: float = 0.38
    lift_start: float = 0.50
    lift_full: float = 0.565
    hold_end: float = 0.595
    lower_end: float = 0.685
    shift_return_end: float = 0.82

    target_clearance: float = 0.026

    # Do not ask the COM to go all the way to the right-foot center.
    # 0.0 = initial COM, 1.0 = right-foot center.
    # V7:
    # Move farther toward the RIGHT support foot before swing.
    # The old 0.58 target was sufficient for balance, but the left foot
    # could still be mechanically loaded when the release seed began.
    com_shift_fraction: float = 0.76
    support_gate_start: float = 0.26
    com_gate_tolerance: float = 0.030
    gate_min_up_z: float = 0.90

    # Actual force-based unloading.
    # Swing is not unlocked until the left foot carries <=18% of total
    # measured foot normal force.
    max_left_load_ratio: float = 0.18

    # Contact-release assistance fades according to ACTUAL load.
    release_load_ratio_full: float = 0.16
    release_load_ratio_zero: float = 0.03

    # During swing, start returning the foot if balance degrades.
    swing_guard_up_start: float = 0.965
    swing_guard_up_stop: float = 0.885
    swing_guard_com_start: float = 0.025
    swing_guard_com_stop: float = 0.070
    swing_guard_x_start: float = 0.045
    swing_guard_x_stop: float = 0.10
    swing_guard_xv_start: float = 0.08
    swing_guard_xv_stop: float = 0.20

    # Left swing-foot IK.
    swing_ik_gain: float = 0.90
    swing_ik_damping: float = 0.055
    swing_ik_max_delta: float = 0.14
    swing_xy_hold_weight: float = 0.08
    swing_z_weight: float = 1.00

    # Right support-foot lock.
    support_lock_weight: float = 0.78
    support_xy_weight: float = 0.34
    support_z_weight: float = 1.45
    support_ik_gain: float = 0.60
    support_ik_damping: float = 0.065
    support_ik_max_delta: float = 0.105

    # Conservative posture control.
    torso_pitch_gain: float = 0.16
    torso_roll_gain: float = 0.10
    angvel_pitch_gain: float = 0.075
    angvel_roll_gain: float = 0.055
    height_gain: float = 0.18
    height_target: float = 0.790

    # Hard safety termination.
    x_hard_limit: float = 0.36
    y_hard_limit: float = 0.30
    x_velocity_hard_limit: float = 1.35
    y_velocity_hard_limit: float = 1.35
    min_up_z: float = 0.70
    min_height: float = 0.55
    max_height: float = 1.05


class G1DeterministicLeftLiftEnv(gym.Env):
    """
    Deterministic Unitree G1 LEFT-foot lift diagnostic.

    There is NO RL here.

    Required sequence:
        1) stabilize standing
        2) shift COM toward RIGHT support foot
        3) verify right-foot contact + COM gate + torso gate
        4) lift LEFT foot
        5) hold
        6) lower LEFT foot
        7) return toward neutral

    The swing is deliberately gated. If the robot has not established a valid
    right-support state, the left foot is not allowed to lift. This lets the
    evaluator identify the real failure instead of hiding it inside RL reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = DeterministicLeftLiftConfig(**kwargs)

        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep * self.cfg.frame_skip)

        self.joint_names = CONTROLLED_15_JOINTS
        self.qpos_adrs: List[int] = []
        self.qvel_adrs: List[int] = []
        self.actuator_ids: List[int] = []

        for name in self.joint_names:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid < 0:
                raise RuntimeError(f"Missing joint: {name}")

            self.qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            self.qvel_adrs.append(int(self.model.jnt_dofadr[jid]))

            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if aid < 0:
                aid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name.replace("_joint", ""),
                )
            if aid < 0:
                for k in range(self.model.nu):
                    if int(self.model.actuator_trnid[k, 0]) == jid:
                        aid = k
                        break
            if aid < 0:
                raise RuntimeError(f"Missing actuator for joint: {name}")
            self.actuator_ids.append(int(aid))

        self.left_foot_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
        )
        self.right_foot_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
        )
        self.left_foot_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_foot_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        if min(
            self.left_foot_site,
            self.right_foot_site,
            self.left_foot_body,
            self.right_foot_body,
        ) < 0:
            raise RuntimeError(
                "Missing required foot site/body names: left_foot, right_foot, "
                "left_ankle_roll_link, right_ankle_roll_link."
            )

        self.stand_qpos = (
            self.model.key_qpos[0].copy()
            if self.model.nkey > 0
            else self.data.qpos.copy()
        )
        self.stand_joint_pos = np.array(
            [self.stand_qpos[i] for i in self.qpos_adrs], dtype=np.float64
        )

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            if 0 <= jid < self.model.njnt:
                qadr = int(self.model.jnt_qposadr[jid])
                if qadr < len(self.stand_qpos):
                    self.default_ctrl[aid] = float(self.stand_qpos[qadr])

        self.action_space = spaces.Box(
            low=0.0, high=0.0, shape=(1,), dtype=np.float32
        )

        self.episode_step = 0
        self.left_foot_p0 = np.zeros(3, dtype=np.float64)
        self.right_foot_p0 = np.zeros(3, dtype=np.float64)
        self.com_p0 = np.zeros(3, dtype=np.float64)

        self.target_com_y = 0.0
        self.shift_direction = -1.0

        self._support_ready_latched = False
        self._support_ready_step = -1
        self._support_ready_q = None
        self._support_ready_latched = False
        self._support_ready_step = -1
        self._support_ready_q = None
        self._lift_enabled = False
        self._lift_enable_step = -1
        self._emergency_landing = False
        self._emergency_landing_step = -1
        self._best_pre_lift_com_error = float("inf")
        self._gate_fail_reason = "not_checked"

        self._set_state_to_stand()
        self._capture_initial_geometry()

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Basic model utilities
    # ------------------------------------------------------------------

    def _set_state_to_stand(self) -> None:
        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, self.data)

    def _capture_initial_geometry(self) -> None:
        self.left_foot_p0[:] = self.data.site_xpos[self.left_foot_site].copy()
        self.right_foot_p0[:] = self.data.site_xpos[self.right_foot_site].copy()
        self.com_p0[:] = self._whole_body_com()

        dy = float(self.right_foot_p0[1] - self.com_p0[1])
        self.shift_direction = 1.0 if dy >= 0.0 else -1.0
        self.target_com_y = float(
            self.com_p0[1] + self.cfg.com_shift_fraction * dy
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self._lift_enabled = False
        self._lift_enable_step = -1
        self._emergency_landing = False
        self._emergency_landing_step = -1
        self._best_pre_lift_com_error = float("inf")
        self._gate_fail_reason = "not_checked"

        self._set_state_to_stand()
        self._capture_initial_geometry()

        info = self._get_info()
        return self._get_obs(), info

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _phase(self) -> float:
        t = self.episode_step * self.dt
        return float(
            np.clip(t / max(self.cfg.cycle_duration, 1e-6), 0.0, 1.0)
        )

    def _whole_body_com(self) -> np.ndarray:
        # data.xipos = world positions of each body's inertial frame.
        masses = np.asarray(self.model.body_mass[1:], dtype=np.float64)
        positions = np.asarray(self.data.xipos[1:], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 1e-9:
            return np.asarray(self.data.qpos[:3], dtype=np.float64).copy()
        return np.sum(positions * masses[:, None], axis=0) / total_mass

    def _root_orientation_proxies(self):
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        up_x = float(mat[2])
        up_y = float(mat[5])
        up_z = float(mat[8])
        return up_x, up_y, up_z

    def _root_up_z(self) -> float:
        return self._root_orientation_proxies()[2]

    def _foot_contact(self, body_id: int) -> bool:
        """
        Ground-contact test only.

        body id 0 is MuJoCo worldbody, which contains the floor in the
        standard G1 menagerie scene.

        Older versions counted ANY collision involving the foot body.
        That could report contact even when the contact was not the floor.
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]

            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])

            if (
                (b1 == body_id and b2 == 0)
                or
                (b2 == body_id and b1 == 0)
            ):
                return True

        return False


    def _foot_normal_force(self, body_id: int) -> float:
        """
        Sum normal contact force between one foot and the ground.

        mj_contactForce returns the contact-frame force.
        Component 0 is the normal component.
        """
        total = 0.0
        force6 = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            c = self.data.contact[i]

            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])

            ground_contact = (
                (b1 == body_id and b2 == 0)
                or
                (b2 == body_id and b1 == 0)
            )

            if ground_contact:
                force6[:] = 0.0

                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    i,
                    force6,
                )

                total += max(
                    0.0,
                    float(force6[0]),
                )

        return float(total)


    def _foot_load_ratio(self):
        """
        Return:
            left normal force,
            right normal force,
            left fraction of total foot load.
        """
        left_force = self._foot_normal_force(
            self.left_foot_body
        )

        right_force = self._foot_normal_force(
            self.right_foot_body
        )

        total = left_force + right_force

        if total <= 1e-6:
            # Fail safely: do not claim left foot is unloaded if
            # neither foot currently has measurable support force.
            return (
                float(left_force),
                float(right_force),
                1.0,
            )

        left_ratio = left_force / total

        return (
            float(left_force),
            float(right_force),
            float(left_ratio),
        )

    def _site_linear_velocity(self, site_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp @ self.data.qvel

    def _ik_delta(
        self,
        site_id: int,
        dof_indices: List[int],
        task_error: np.ndarray,
        gain: float,
        damping: float,
        max_delta: float,
    ) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = jacp[:, dof_indices]

        A = J @ J.T + (damping * damping) * np.eye(3)
        try:
            delta = J.T @ np.linalg.solve(A, task_error)
        except np.linalg.LinAlgError:
            delta = J.T @ np.linalg.pinv(A) @ task_error

        delta = gain * delta
        return np.clip(delta, -max_delta, max_delta)

    # ------------------------------------------------------------------
    # Phase envelopes
    # ------------------------------------------------------------------

    def _shift_env(self, phi: float) -> float:
        c = self.cfg
        if phi < c.shift_start:
            return 0.0
        if phi < c.shift_full:
            u = (phi - c.shift_start) / max(c.shift_full - c.shift_start, 1e-6)
            return self._smoothstep(u)
        if phi < c.lower_end:
            return 1.0
        if phi < c.shift_return_end:
            u = (phi - c.lower_end) / max(c.shift_return_end - c.lower_end, 1e-6)
            return 1.0 - self._smoothstep(u)
        return 0.0

    def _raw_swing_env(self, phi: float) -> float:
        c = self.cfg
        if phi < c.lift_start:
            return 0.0
        if phi < c.lift_full:
            u = (phi - c.lift_start) / max(c.lift_full - c.lift_start, 1e-6)
            return self._smoothstep(u)
        if phi < c.hold_end:
            return 1.0
        if phi < c.lower_end:
            u = (phi - c.hold_end) / max(c.lower_end - c.hold_end, 1e-6)
            return 1.0 - self._smoothstep(u)
        return 0.0

    def _swing_env(self, phi: float) -> float:
        if not self._lift_enabled:
            return 0.0

        raw = self._raw_swing_env(phi)
        if raw <= 0.0:
            return 0.0

        up_z = float(self._root_up_z())
        com_y = float(self._whole_body_com()[1])
        com_err = abs(com_y - float(self.target_com_y))
        x_abs = abs(float(self.data.qpos[0]))
        xv_abs = abs(float(self.data.qvel[0]))

        # V4: velocity is the earliest warning signal in the real logs.
        # Latch an early landing once sagittal momentum is becoming unsafe.
        if (
            not self._emergency_landing
            and (
                xv_abs >= self.cfg.swing_guard_xv_stop
                or x_abs >= self.cfg.swing_guard_x_stop
                or up_z <= self.cfg.swing_guard_up_stop
            )
        ):
            self._emergency_landing = True
            self._emergency_landing_step = int(self.episode_step)

        if self._emergency_landing:
            return 0.0

        def ramp_down(value, start, stop):
            if value <= start:
                return 1.0
            if value >= stop:
                return 0.0
            return float((stop - value) / max(stop - start, 1e-9))

        up_guard = 1.0
        if up_z < self.cfg.swing_guard_up_start:
            up_guard = float(np.clip(
                (up_z - self.cfg.swing_guard_up_stop)
                / max(self.cfg.swing_guard_up_start - self.cfg.swing_guard_up_stop, 1e-9),
                0.0,
                1.0,
            ))

        com_guard = ramp_down(
            com_err,
            self.cfg.swing_guard_com_start,
            self.cfg.swing_guard_com_stop,
        )
        x_guard = ramp_down(
            x_abs,
            self.cfg.swing_guard_x_start,
            self.cfg.swing_guard_x_stop,
        )
        xv_guard = ramp_down(
            xv_abs,
            self.cfg.swing_guard_xv_start,
            self.cfg.swing_guard_xv_stop,
        )

        guard = min(up_guard, com_guard, x_guard, xv_guard)
        return float(raw * guard)

    # ------------------------------------------------------------------
    # Support gate
    # ------------------------------------------------------------------

    def _maybe_enable_lift(self) -> None:
        """
        V2 gate logic.

        The V1 bug was timing: the COM reached the right-support target around
        phi ~= 0.30, but the code refused to evaluate the gate until lift_start
        (phi=0.44). The fixed lateral feedforward then kept pushing and the COM
        overshot before the gate was ever allowed to pass.

        V2 therefore:
        1) starts evaluating support readiness earlier,
        2) latches the first genuinely good single-support-ready state,
        3) stores that joint posture,
        4) waits until lift_start before allowing the left swing.
        """
        phi = self._phase()
        c = self.cfg

        # Phase A: acquire and latch a valid RIGHT-support state.
        if not self._support_ready_latched:
            if phi < c.support_gate_start:
                return
            if phi > c.hold_end:
                return

            com = self._whole_body_com()
            com_error = abs(float(com[1] - self.target_com_y))
            self._best_pre_lift_com_error = min(
                self._best_pre_lift_com_error, com_error
            )

            right_contact = self._foot_contact(
                self.right_foot_body
            )

            (
                left_force,
                right_force,
                left_load_ratio,
            ) = self._foot_load_ratio()

            up_z = self._root_up_z()

            com_ok = (
                com_error <= c.com_gate_tolerance
            )

            contact_ok = (
                bool(right_contact)
                and right_force > 1e-3
            )

            upright_ok = (
                up_z >= c.gate_min_up_z
            )

            unloaded_ok = (
                left_load_ratio
                <= c.max_left_load_ratio
            )

            if not contact_ok:
                self._gate_fail_reason = (
                    "right_support_contact_missing"
                )

            elif not upright_ok:
                self._gate_fail_reason = (
                    "upright_gate_failed"
                )

            elif not com_ok:
                self._gate_fail_reason = (
                    "com_transfer_incomplete"
                )

            elif not unloaded_ok:
                self._gate_fail_reason = (
                    "left_foot_still_loaded"
                )

            else:
                self._support_ready_latched = True
                self._support_ready_step = int(self.episode_step)
                self._support_ready_q = np.array(
                    [self.data.qpos[self.qpos_adrs[i]] for i in range(15)],
                    dtype=np.float64,
                )
                self._gate_fail_reason = "support_ready_latched"

        # Phase B: only begin swing at the planned lift time.
        if (
            self._support_ready_latched
            and not self._lift_enabled
            and phi >= c.lift_start
            and phi <= c.hold_end
        ):
            self._lift_enabled = True
            self._lift_enable_step = int(self.episode_step)
            self._gate_fail_reason = "gate_passed"

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _target_joint_position(self, info: Dict[str, float]) -> np.ndarray:
        phi = float(info["phase"])
        sh = float(info["shift_env"])
        sw = float(info["swing_env"])

        if self._support_ready_latched and self._support_ready_q is not None:
            # Hold the first valid right-support posture instead of continuing
            # to accumulate the open-loop lateral shift that caused V1 overshoot.
            target = self._support_ready_q.copy()
        else:
            target = self.stand_joint_pos.copy()
        off = np.zeros(15, dtype=np.float64)

        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        yv = float(info["y_velocity"])
        h = float(info["base_height"])

        com_y = float(info["com_y"])
        com_error_y = com_y - float(self.target_com_y)

        up_x, up_y, up_z = self._root_orientation_proxies()
        av = np.asarray(self.data.qvel[3:6], dtype=np.float64)

        # --------------------------------------------------------------
        # A) COM transfer to RIGHT support foot.
        #
        # This mirrors the old right-lift lateral pattern, but derives the
        # direction from actual geometry rather than hard-coding y sign.
        # --------------------------------------------------------------
        ycorr = float(
            np.clip(1.20 * com_error_y + 0.22 * yv, -0.12, 0.12)
        )

        d = float(self.shift_direction)

        if not self._support_ready_latched:
            # Open-loop transfer is used only to REACH the target.
            off[7] += -0.22 * d * sh + 0.13 * ycorr       # right hip roll
            off[11] += +0.13 * d * sh - 0.08 * ycorr     # right ankle roll

            # LEFT swing-leg preload
            off[1] += -0.08 * d * sh + 0.05 * ycorr      # left hip roll
            off[5] += +0.06 * d * sh - 0.04 * ycorr      # left ankle roll

            # Waist follows lateral transfer.
            off[13] += +0.13 * d * sh - 0.10 * ycorr     # waist roll
        else:
            # After latching, use only small closed-loop COM correction around
            # the saved support pose. This prevents continued lateral overshoot.
            hold_corr = float(np.clip(3.00 * com_error_y + 0.45 * yv, -0.080, 0.080))
            off[7] += +0.82 * hold_corr
            off[11] += -0.52 * hold_corr
            off[13] += -0.44 * hold_corr

        # --------------------------------------------------------------
        # B) Sagittal root stabilization through RIGHT stance leg.
        # --------------------------------------------------------------
        xcorr = float(
            np.clip(-1.35 * x - 0.48 * xv, -0.12, 0.12)
        )

        # V5:
        # Restore the experimentally stable V3 sagittal mapping.
        #
        # V4 proved that reversing these signs was incorrect because the
        # robot developed strong negative-x velocity BEFORE swing started.
        #
        # Keep this controller constant instead of increasing its gain
        # according to swing phase.
        off[6] += +1.00 * xcorr       # right hip pitch
        off[10] += -0.78 * xcorr      # right ankle pitch
        off[14] += -0.36 * xcorr      # waist pitch

        # --------------------------------------------------------------
        # C) Torso upright + height protection.
        # --------------------------------------------------------------
        pitch_task = np.clip(
            -self.cfg.torso_pitch_gain * up_y
            - self.cfg.angvel_pitch_gain * av[1]
            - 0.055 * xv
            - 0.020 * x,
            -0.080,
            0.080,
        )
        roll_task = np.clip(
            -self.cfg.torso_roll_gain * up_x
            - self.cfg.angvel_roll_gain * av[0]
            - 0.040 * yv,
            -0.065,
            0.065,
        )
        height_task = np.clip(
            self.cfg.height_gain * (self.cfg.height_target - h),
            -0.050,
            0.070,
        )

        off[14] += -0.50 * pitch_task
        off[6] += +0.42 * pitch_task
        off[10] += -0.32 * pitch_task

        off[13] += -0.50 * roll_task
        off[7] += +0.32 * roll_task
        off[11] += -0.22 * roll_task

        # V5:
        # Keep RIGHT support authority nearly constant.
        #
        # Previously:
        #
        #     support_env = 0.35*shift + 0.65*swing
        #
        # so support_env increased from ~0.35 before lifting to 1.0
        # during full swing.
        #
        # With support_lock_weight=0.78 this changed the IK blend from
        # about 0.27 to 0.78 -- a very large controller change exactly
        # when the left foot started moving.
        #
        # V5 removes that coupling.
        support_env = min(0.40, max(0.0, 0.40 * sh))

        off[9] += +0.020 * support_env + 0.34 * height_task
        off[6] += +0.14 * height_task
        off[10] += -0.12 * height_task

        # --------------------------------------------------------------
        # D) LEFT SWING = TASK-SPACE IK ONLY
        # --------------------------------------------------------------
        #
        # No knee feedforward.
        # No ankle feedforward.
        # No hip-pitch feedforward.
        #
        # The task-space IK below is now the ONLY controller allowed
        # to create left-foot clearance.
        #
        # This separates the swing-leg problem from stance balance.

        if self._emergency_landing:
            # Active touchdown: remove knee flexion and point the ankle toward
            # the latched standing pose. This is intentionally conservative.
            target_left = self.stand_joint_pos[:6]
            for j in range(6):
                off[j] += 0.45 * (target_left[j] - (target[j] + off[j]))

        target += off

        # --------------------------------------------------------------
        # E) RIGHT support-foot task-space lock.
        # --------------------------------------------------------------
        if support_env > 0.02:
            support_idx = [6, 7, 8, 9, 10, 11]
            support_dofs = [self.qvel_adrs[i] for i in support_idx]

            current = self.data.site_xpos[self.right_foot_site].copy()
            desired = self.right_foot_p0.copy()
            err = desired - current
            err[0] *= self.cfg.support_xy_weight
            err[1] *= self.cfg.support_xy_weight
            err[2] *= self.cfg.support_z_weight

            dq = self._ik_delta(
                self.right_foot_site,
                support_dofs,
                err,
                self.cfg.support_ik_gain,
                self.cfg.support_ik_damping,
                self.cfg.support_ik_max_delta,
            )
            current_q = np.array(
                [self.data.qpos[self.qpos_adrs[i]] for i in support_idx],
                dtype=np.float64,
            )
            ik_target = current_q + dq
            blend = self.cfg.support_lock_weight * support_env

            for local_i, joint_i in enumerate(support_idx):
                target[joint_i] = (
                    (1.0 - blend) * target[joint_i]
                    + blend * ik_target[local_i]
                )

        # --------------------------------------------------------------
        # F) LEFT swing-foot task-space IK.
        # --------------------------------------------------------------
        if sw > 0.001 or self._emergency_landing:
            swing_idx = [0, 1, 2, 3, 4, 5]
            swing_dofs = [
                self.qvel_adrs[i]
                for i in swing_idx
            ]

            current = self.data.site_xpos[
                self.left_foot_site
            ].copy()

            desired = self.left_foot_p0.copy()

            # Normal swing:
            #     move foot upward.
            #
            # Emergency landing:
            #     sw = 0, therefore desired z becomes original
            #     ground-height target.
            desired[2] += (
                self.cfg.target_clearance * sw
            )

            err = desired - current
            err[0] *= self.cfg.swing_xy_hold_weight
            err[1] *= self.cfg.swing_xy_hold_weight
            err[2] *= self.cfg.swing_z_weight

            dq = self._ik_delta(
                self.left_foot_site,
                swing_dofs,
                err,
                self.cfg.swing_ik_gain,
                self.cfg.swing_ik_damping,
                self.cfg.swing_ik_max_delta,
            )
            current_q = np.array(
                [self.data.qpos[self.qpos_adrs[i]] for i in swing_idx],
                dtype=np.float64,
            )
            ik_target = current_q + dq

            if self._emergency_landing:
                blend = 0.90
            else:
                blend = 0.40 + 0.50 * sw

            for local_i, joint_i in enumerate(swing_idx):
                target[joint_i] = (
                    (1.0 - blend) * target[joint_i]
                    + blend * ik_target[local_i]
                )

        # --------------------------------------------------------------
        # V7: FORCE-GATED LEFT-FOOT CONTACT RELEASE
        # --------------------------------------------------------------
        #
        # This is the important change.
        #
        # V6 kept the knee release active whenever:
        #
        #     left_contact == True
        #
        # but contact does NOT mean the foot is heavily loaded.
        #
        # V7 uses the actual left/right normal-force distribution.
        #
        # The support gate has already required:
        #
        #     left_load_ratio <= 0.18
        #
        # before swing can start.
        #
        # From there, this tiny seed smoothly disappears as
        # left-foot force approaches zero.
        if (
            sw > 0.02
            and not self._emergency_landing
        ):
            (
                left_force_now,
                right_force_now,
                left_load_ratio_now,
            ) = self._foot_load_ratio()

            denominator = max(
                self.cfg.release_load_ratio_full
                - self.cfg.release_load_ratio_zero,
                1e-6,
            )

            release_need = float(
                np.clip(
                    (
                        left_load_ratio_now
                        - self.cfg.release_load_ratio_zero
                    )
                    / denominator,
                    0.0,
                    1.0,
                )
            )

            release_ramp = float(
                np.clip(
                    sw / 0.30,
                    0.0,
                    1.0,
                )
            )

            release = (
                release_need
                * release_ramp
            )

            # Smaller than V6 because the left foot should already
            # be substantially unloaded before this is allowed.
            target[0] += (
                +0.003 * release
            )                       # left hip pitch

            target[3] += (
                +0.075 * release
            )                       # left knee flexion

            target[4] += (
                +0.025 * release
            )                       # left ankle pitch

        # Clip to actuator ranges.
        for i, aid in enumerate(self.actuator_ids):
            target[i] = np.clip(
                target[i], self.ctrl_low[aid], self.ctrl_high[aid]
            )

        return target

    # ------------------------------------------------------------------
    # Gym step / observations
    # ------------------------------------------------------------------

    def step(self, action=None):
        # No RL action is used.
        self._maybe_enable_lift()
        info = self._get_info()
        target = self._target_joint_position(info)

        self.data.ctrl[:] = self.default_ctrl
        for j, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(target[j])

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.episode_step += 1

        # Re-check the gate after physics update so it can activate as soon as
        # COM/support conditions become valid.
        self._maybe_enable_lift()

        info = self._get_info()
        reward, rinfo = self._compute_reward(info)
        info.update(rinfo)

        terminated = self._terminated(info)
        truncated = self.episode_step >= self.cfg.max_steps

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _get_info(self) -> Dict[str, float]:
        phi = self._phase()
        sh = self._shift_env(phi)
        raw_sw = self._raw_swing_env(phi)
        sw = self._swing_env(phi)

        lpos = self.data.site_xpos[self.left_foot_site].copy()
        rpos = self.data.site_xpos[self.right_foot_site].copy()

        lclr = max(0.0, float(lpos[2] - self.left_foot_p0[2]))
        rclr = max(0.0, float(rpos[2] - self.right_foot_p0[2]))

        lc = self._foot_contact(
            self.left_foot_body
        )

        rc = self._foot_contact(
            self.right_foot_body
        )

        (
            left_force,
            right_force,
            left_load_ratio,
        ) = self._foot_load_ratio()

        lvel = self._site_linear_velocity(
            self.left_foot_site
        )
        rvel = self._site_linear_velocity(self.right_foot_site)

        com = self._whole_body_com()
        com_error_y = float(com[1] - self.target_com_y)

        right_support_displacement = float(
            np.linalg.norm(rpos[:2] - self.right_foot_p0[:2])
        )

        target_clear = float(self.cfg.target_clearance * sw)

        return {
            "episode_step": float(self.episode_step),
            "phase": float(phi),
            "shift_env": float(sh),
            "raw_swing_env": float(raw_sw),
            "swing_env": float(sw),
            "support_ready_latched": bool(self._support_ready_latched),
            "support_ready_step": float(self._support_ready_step),
            "lift_enabled": bool(self._lift_enabled),
            "lift_enable_step": float(self._lift_enable_step),
            "emergency_landing": bool(self._emergency_landing),
            "emergency_landing_step": float(self._emergency_landing_step),
            "gate_fail_reason": self._gate_fail_reason,

            "base_height": float(self.data.qpos[2]),
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "z_velocity": float(self.data.qvel[2]),
            "root_ang_vel": float(np.linalg.norm(self.data.qvel[3:6])),
            "up_z": float(self._root_up_z()),

            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
            "target_com_y": float(self.target_com_y),
            "com_error_y": float(com_error_y),
            "best_pre_lift_com_error": float(self._best_pre_lift_com_error),

            "left_foot_x": float(lpos[0]),
            "left_foot_y": float(lpos[1]),
            "left_foot_z": float(lpos[2]),
            "right_foot_x": float(rpos[0]),
            "right_foot_y": float(rpos[1]),
            "right_foot_z": float(rpos[2]),

            "left_foot_clearance": float(lclr),
            "right_foot_clearance": float(rclr),
            "main_clearance": float(lclr),
            "main_target_clearance": float(target_clear),
            "swing_guard_ratio": float(sw / max(raw_sw, 1e-9)) if raw_sw > 1e-9 else 1.0,

            "left_contact": bool(lc),
            "right_contact": bool(rc),

            "left_normal_force": float(left_force),
            "right_normal_force": float(right_force),
            "left_load_ratio": float(left_load_ratio),

            "left_foot_speed_xy": float(np.linalg.norm(lvel[:2])),
            "right_foot_speed_xy": float(np.linalg.norm(rvel[:2])),
            "support_slip": float(right_support_displacement),
        }

    def _get_obs(self) -> np.ndarray:
        info = self._get_info()

        jp = np.array(
            [self.data.qpos[i] for i in self.qpos_adrs],
            dtype=np.float32,
        )
        jv = np.array(
            [self.data.qvel[i] for i in self.qvel_adrs],
            dtype=np.float32,
        )

        base = np.array(
            [
                info["phase"],
                info["shift_env"],
                info["raw_swing_env"],
                info["swing_env"],
                float(info["lift_enabled"]),
                info["base_height"],
                info["x_position"],
                info["y_position"],
                info["x_velocity"],
                info["y_velocity"],
                info["root_ang_vel"],
                info["up_z"],
                info["com_y"],
                info["target_com_y"],
                info["com_error_y"],
                info["left_foot_clearance"],
                info["right_foot_clearance"],
                float(info["left_contact"]),
                float(info["right_contact"]),
                info["support_slip"],
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                base,
                jp - self.stand_joint_pos.astype(np.float32),
                jv,
            ]
        ).astype(np.float32)

    def _compute_reward(
        self, info: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        # Diagnostic reward only. PASS/FAIL is decided by evaluator conditions.
        clear = float(info["left_foot_clearance"])
        target = float(info["main_target_clearance"])
        com_err = abs(float(info["com_error_y"]))
        support_slip = float(info["support_slip"])

        lift_progress = (
            min(clear / max(target, 1e-6), 1.0)
            if target > 0.001
            else 1.0
        )

        reward = (
            20.0 * lift_progress
            + 4.0 * max(float(info["up_z"]), 0.0)
            - 30.0 * com_err
            - 60.0 * support_slip
            + 3.0 * float(bool(info["right_contact"]))
        )

        return float(reward), {
            "reward_total": float(reward),
            "reward_version": "deterministic_left_lift_diagnostic_v7_force_gated_unload",
        }

    def _terminated(self, info: Dict[str, float]) -> bool:
        c = self.cfg

        if float(info["base_height"]) < c.min_height:
            return True
        if float(info["base_height"]) > c.max_height:
            return True
        if float(info["up_z"]) < c.min_up_z:
            return True
        if abs(float(info["x_position"])) > c.x_hard_limit:
            return True
        if abs(float(info["y_position"])) > c.y_hard_limit:
            return True
        if abs(float(info["x_velocity"])) > c.x_velocity_hard_limit:
            return True
        if abs(float(info["y_velocity"])) > c.y_velocity_hard_limit:
            return True
        return False

    def termination_reason(self, info: Dict[str, float]) -> str:
        c = self.cfg

        if float(info["base_height"]) < c.min_height:
            return "base_height_low"
        if float(info["base_height"]) > c.max_height:
            return "base_height_high"
        if float(info["up_z"]) < c.min_up_z:
            return "up_z_low"
        if abs(float(info["x_position"])) > c.x_hard_limit:
            return "x_position_limit"
        if abs(float(info["y_position"])) > c.y_hard_limit:
            return "y_position_limit"
        if abs(float(info["x_velocity"])) > c.x_velocity_hard_limit:
            return "x_velocity_limit"
        if abs(float(info["y_velocity"])) > c.y_velocity_hard_limit:
            return "y_velocity_limit"
        return "not_terminated"

    def close(self):
        pass
