from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_neutral_only_env import G1WBCNeutralOnlyEnv


CONTROLLED_15_JOINTS = [
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


class G1WBCV77LeftLiftMirrorEnv(G1WBCNeutralOnlyEnv):
    """
    V7.7 left-leg mirror test.

    Goal:
    - reproduce the right-lift milestone on the left leg
    - find correct left-side preload and ankle pitch signs
    - keep the robot stable for 520 steps

    This starts from the clean neutral-only controller, not from the old hidden WBC logic.
    """

    def __init__(
        self,
        preload_sign: float = 1.0,
        preload_amp: float = 0.040,
        lift_height: float = 0.010,
        foot_ik_gain: float = 0.55,
        foot_ik_damping: float = 0.080,
        foot_ik_max_delta: float = 0.060,
        ankle_pitch_bias: float = -0.075,
        swing_knee_bias: float = 0.004,
        swing_hip_bias: float = 0.002,
        roll_action: float = 0.0,
        residual_roll_scale: float = 0.010,
        preload_start: float = 0.08,
        preload_end: float = 0.42,
        lift_ramp_start: float = 0.46,
        lift_ramp_end: float = 0.56,
        lift_hold_end: float = 0.68,
        lift_land_end: float = 0.82,
        preload_down_start: float = 0.82,
        preload_down_end: float = 1.00,
        **kwargs,
    ):
        self.preload_sign = float(preload_sign)
        self.preload_amp = float(preload_amp)

        self.lift_height = float(lift_height)
        self.foot_ik_gain = float(foot_ik_gain)
        self.foot_ik_damping = float(foot_ik_damping)
        self.foot_ik_max_delta = float(foot_ik_max_delta)

        self.ankle_pitch_bias = float(ankle_pitch_bias)
        self.swing_knee_bias = float(swing_knee_bias)
        self.swing_hip_bias = float(swing_hip_bias)

        self.roll_action = float(roll_action)
        self.residual_roll_scale = float(residual_roll_scale)

        self.preload_start = float(preload_start)
        self.preload_end = float(preload_end)

        self.lift_ramp_start = float(lift_ramp_start)
        self.lift_ramp_end = float(lift_ramp_end)
        self.lift_hold_end = float(lift_hold_end)
        self.lift_land_end = float(lift_land_end)

        self.preload_down_start = float(preload_down_start)
        self.preload_down_end = float(preload_down_end)

        self._left_z0 = None
        self._initial_gap = 0.0
        self._dof_ids_15: List[int] = []

        super().__init__(**kwargs)

        self._dof_ids_15 = []
        for name in CONTROLLED_15_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found: {name}")
            self._dof_ids_15.append(int(self.model.jnt_dofadr[jid]))

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self._left_z0 = float(self.data.site_xpos[self.left_foot_site][2])
        self._initial_gap = self._foot_gap()

        return obs, info

    def _foot_gap(self) -> float:
        right_y = float(self.data.site_xpos[self.right_foot_site][1])
        left_y = float(self.data.site_xpos[self.left_foot_site][1])
        return abs(right_y - left_y)

    def _lift_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.lift_ramp_start:
            return 0.0

        if phi < self.lift_ramp_end:
            return self._smoothstep(
                (phi - self.lift_ramp_start)
                / max(self.lift_ramp_end - self.lift_ramp_start, 1e-6)
            )

        if phi < self.lift_hold_end:
            return 1.0

        if phi < self.lift_land_end:
            down = self._smoothstep(
                (phi - self.lift_hold_end)
                / max(self.lift_land_end - self.lift_hold_end, 1e-6)
            )
            return 1.0 - down

        return 0.0

    def _preload_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.preload_start:
            return 0.0

        if phi < self.preload_end:
            return self._smoothstep(
                (phi - self.preload_start)
                / max(self.preload_end - self.preload_start, 1e-6)
            )

        if phi < self.preload_down_start:
            return 1.0

        if phi < self.preload_down_end:
            down = self._smoothstep(
                (phi - self.preload_down_start)
                / max(self.preload_down_end - self.preload_down_start, 1e-6)
            )
            return 1.0 - down

        return 0.0

    def _phase_name(self, phi: float) -> str:
        phi = float(phi)

        if phi < self.preload_start:
            return "STAND"
        if phi < self.preload_end:
            return "PRELOAD"
        if phi < self.lift_ramp_start:
            return "PRELOAD_HOLD"
        if phi < self.lift_ramp_end:
            return "LIFT_RAMP"
        if phi < self.lift_hold_end:
            return "AIR_HOLD"
        if phi < self.lift_land_end:
            return "LAND"
        if phi < self.preload_down_end:
            return "RETURN_NEUTRAL"
        return "NEUTRAL_END"

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = self.stand_joint_pos.copy()

        phi = float(info["phase"])
        preload = self._preload_env(phi)
        lift = self._lift_env(phi)

        # ------------------------------------------------------------------
        # 1. Lateral preload
        # Swept by sign. One sign should unload the left foot.
        # ------------------------------------------------------------------
        p = self.preload_sign * self.preload_amp * preload

        # Roll-family correction.
        # Indices:
        # left_hip_roll    = 1
        # left_ankle_roll  = 5
        # right_hip_roll   = 7
        # right_ankle_roll = 11
        # waist_roll       = 13
        target[1] += +0.60 * p
        target[5] += -0.30 * p
        target[7] += +0.60 * p
        target[11] += -0.30 * p
        target[13] += -0.20 * p

        # ------------------------------------------------------------------
        # 2. Left-foot vertical IK
        # ------------------------------------------------------------------
        if lift > 1e-6:
            if self._left_z0 is None:
                self._left_z0 = float(self.data.site_xpos[self.left_foot_site][2])

            current_z = float(self.data.site_xpos[self.left_foot_site][2])
            desired_z = float(self._left_z0 + self.lift_height)
            err_z = desired_z - current_z

            dz = float(np.clip(self.foot_ik_gain * err_z, -0.030, 0.030))

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.left_foot_site)

            J = jacp[2, self._dof_ids_15].astype(np.float64)

            # Use mainly left pitch-chain joints.
            # left_hip_pitch   = 0
            # left_hip_roll    = 1
            # left_knee        = 3
            # left_ankle_pitch = 4
            # left_ankle_roll  = 5
            # waist_roll       = 13
            mask = np.zeros(15, dtype=np.float64)
            mask[0] = 0.80
            mask[1] = 0.25
            mask[3] = 1.00
            mask[4] = 0.80
            mask[5] = 0.25
            mask[13] = 0.15

            Jm = J * mask
            denom = float(Jm @ Jm + self.foot_ik_damping * self.foot_ik_damping)

            if denom > 1e-8:
                dq = Jm * (dz / denom)
                dq = np.clip(dq, -self.foot_ik_max_delta, self.foot_ik_max_delta)
                target = target + lift * dq.astype(np.float32)

            # ------------------------------------------------------------------
            # 3. Left swing shaping
            # ------------------------------------------------------------------
            # left_hip_pitch   = 0
            # left_knee        = 3
            # left_ankle_pitch = 4
            target[0] += self.swing_hip_bias * lift
            target[3] += self.swing_knee_bias * lift
            target[4] += self.ankle_pitch_bias * lift

            # ------------------------------------------------------------------
            # 4. Small roll residual mirror
            # Swept by roll_action. Used later after signs are found.
            # ------------------------------------------------------------------
            roll = self.residual_roll_scale * self.roll_action * lift

            target[1] += roll
            target[5] += -0.50 * roll
            target[13] += -0.20 * roll

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()
        phi = float(info["phase"])

        gap = self._foot_gap()

        info["v77_phase_name"] = self._phase_name(phi)
        info["v77_lift_env"] = float(self._lift_env(phi))
        info["v77_preload_env"] = float(self._preload_env(phi))
        info["v77_gap"] = float(gap)
        info["v77_gap_delta"] = float(gap - self._initial_gap)

        return info


def contact_forces(env):
    left_body = int(env.model.site_bodyid[env.left_foot_site])
    right_body = int(env.model.site_bodyid[env.right_foot_site])

    left_force = 0.0
    right_force = 0.0

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        b1 = int(env.model.geom_bodyid[int(c.geom1)])
        b2 = int(env.model.geom_bodyid[int(c.geom2)])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

        if b1 == right_body or b2 == right_body:
            right_force += normal_force

    return left_force, right_force


def run_case(
    preload_sign: float,
    preload_amp: float,
    ankle_pitch_bias: float,
    roll_action: float,
) -> dict:
    env = G1WBCV77LeftLiftMirrorEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,

        preload_sign=preload_sign,
        preload_amp=preload_amp,

        lift_height=0.010,
        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        ankle_pitch_bias=ankle_pitch_bias,
        swing_knee_bias=0.004,
        swing_hip_bias=0.002,

        roll_action=roll_action,
        residual_roll_scale=0.010,

        preload_start=0.08,
        preload_end=0.42,
        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=0.68,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset(seed=123)

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_left_clearance = 0.0
    max_right_clearance = 0.0
    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    left_air_steps = 0
    left_air_streak = 0
    max_left_air_streak = 0

    min_left_force = 1e9
    left_ratio_at_min = 1.0

    first_air_step = -1
    last_air_step = -1

    max_gap_delta = 0.0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])

        max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        max_gap_delta = max(max_gap_delta, float(info["v77_gap_delta"]))

        if not bool(info["left_contact"]):
            left_air_steps += 1
            left_air_streak += 1
            max_left_air_streak = max(max_left_air_streak, left_air_streak)

            if first_air_step < 0:
                first_air_step = steps
            last_air_step = steps
        else:
            left_air_streak = 0

        if 0.46 <= phi <= 0.82:
            lf, rf = contact_forces(env)
            if lf < min_left_force:
                min_left_force = lf
                left_ratio_at_min = lf / max(lf + rf, 1e-6)

        if steps >= 520:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "preload_sign": preload_sign,
        "preload_amp": preload_amp,
        "ankle_pitch_bias": ankle_pitch_bias,
        "roll_action": roll_action,
        "steps": steps,
        "reason": reason,
        "left_clearance": max_left_clearance,
        "right_clearance": max_right_clearance,
        "left_air_steps": left_air_steps,
        "left_air_streak": max_left_air_streak,
        "first_air_step": first_air_step,
        "last_air_step": last_air_step,
        "left_force_min": min_left_force,
        "left_force_ratio": left_ratio_at_min,
        "min_up_z": min_up_z,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "gap_delta": float(final.get("v77_gap_delta", 0.0)),
        "max_gap_delta": max_gap_delta,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v77_left_mirror_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    preload_signs = [-1.0, 1.0]
    preload_amps = [0.036, 0.040]
    ankle_values = [-0.090, -0.075, -0.060, 0.060, 0.075, 0.090]
    roll_actions = [0.0, 0.60]

    rows = []

    print("=" * 170)
    print("WBC V7.7 LEFT-MIRROR LIFT SWEEP")
    print("Goal: find stable left-foot lift equivalent to FINAL_v76b right-foot lift.")
    print("=" * 170)

    for psign in preload_signs:
        for pamp in preload_amps:
            for ankle in ankle_values:
                for roll in roll_actions:
                    row = run_case(psign, pamp, ankle, roll)
                    rows.append(row)

                    print(
                        f"psign={row['preload_sign']:+.0f} "
                        f"pamp={row['preload_amp']:.3f} "
                        f"ankle={row['ankle_pitch_bias']:+.3f} "
                        f"roll={row['roll_action']:+.2f} "
                        f"steps={row['steps']:04d} "
                        f"reason={row['reason']:<16} "
                        f"Lclear={row['left_clearance']:.4f} "
                        f"Lair={row['left_air_steps']:03d} "
                        f"Lstreak={row['left_air_streak']:03d} "
                        f"air={row['first_air_step']:03d}-{row['last_air_step']:03d} "
                        f"Lforce={row['left_force_min']:.2f} "
                        f"Lratio={row['left_force_ratio']:.3f} "
                        f"up={row['min_up_z']:.3f} "
                        f"slip={row['support_slip']:.4f} "
                        f"gapDelta={row['gap_delta']:+.4f} "
                        f"x={row['final_x']:+.3f} "
                        f"y={row['final_y']:+.3f} "
                        f"xv={row['final_x_velocity']:+.3f}"
                    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
