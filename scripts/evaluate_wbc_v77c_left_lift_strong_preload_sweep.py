from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v77a_left_preload_diag import G1WBCV77ALeftPreloadDiagEnv


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


class G1WBCV77CLeftLiftEnv(G1WBCV77ALeftPreloadDiagEnv):
    """
    V7.7C: left lift using the successful strong left-preload.

    Locked preload:
    template = right_support_c
    sign     = +1
    amp      = 0.160

    Then sweep left ankle pitch and small lift parameters.
    """

    def __init__(
        self,
        ankle_pitch_bias: float = 0.060,
        lift_height: float = 0.010,
        foot_ik_gain: float = 0.55,
        foot_ik_damping: float = 0.080,
        foot_ik_max_delta: float = 0.060,
        swing_knee_bias: float = 0.000,
        swing_hip_bias: float = 0.000,
        lift_ramp_start: float = 0.46,
        lift_ramp_end: float = 0.56,
        lift_hold_end: float = 0.68,
        lift_land_end: float = 0.82,
        preload_down_start: float = 0.82,
        preload_down_end: float = 1.00,
        **kwargs,
    ):
        self.ankle_pitch_bias = float(ankle_pitch_bias)
        self.lift_height = float(lift_height)
        self.foot_ik_gain = float(foot_ik_gain)
        self.foot_ik_damping = float(foot_ik_damping)
        self.foot_ik_max_delta = float(foot_ik_max_delta)

        self.swing_knee_bias = float(swing_knee_bias)
        self.swing_hip_bias = float(swing_hip_bias)

        self.lift_ramp_start = float(lift_ramp_start)
        self.lift_ramp_end = float(lift_ramp_end)
        self.lift_hold_end = float(lift_hold_end)
        self.lift_land_end = float(lift_land_end)

        self.preload_down_start = float(preload_down_start)
        self.preload_down_end = float(preload_down_end)

        self._left_z0 = None
        self._dof_ids_15: List[int] = []

        super().__init__(**kwargs)

        self._dof_ids_15 = []
        for name in CONTROLLED_15_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found: {name}")
            self._dof_ids_15.append(int(self.model.jnt_dofadr[jid]))

    @staticmethod
    def _smoothstep_lift(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._left_z0 = float(self.data.site_xpos[self.left_foot_site][2])
        return obs, info

    def _lift_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.lift_ramp_start:
            return 0.0

        if phi < self.lift_ramp_end:
            return self._smoothstep_lift(
                (phi - self.lift_ramp_start)
                / max(self.lift_ramp_end - self.lift_ramp_start, 1e-6)
            )

        if phi < self.lift_hold_end:
            return 1.0

        if phi < self.lift_land_end:
            down = self._smoothstep_lift(
                (phi - self.lift_hold_end)
                / max(self.lift_land_end - self.lift_hold_end, 1e-6)
            )
            return 1.0 - down

        return 0.0

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # Start from the strong preload controller.
        target = super()._target_joint_position(action, info)

        phi = float(info["phase"])
        lift = self._lift_env(phi)

        if lift <= 1e-6:
            return target

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

        # Left pitch chain.
        # left_hip_pitch   = 0
        # left_knee        = 3
        # left_ankle_pitch = 4
        mask = np.zeros(15, dtype=np.float64)
        mask[0] = 0.80
        mask[3] = 1.00
        mask[4] = 0.80

        Jm = J * mask
        denom = float(Jm @ Jm + self.foot_ik_damping * self.foot_ik_damping)

        if denom > 1e-8:
            dq = Jm * (dz / denom)
            dq = np.clip(dq, -self.foot_ik_max_delta, self.foot_ik_max_delta)
            target = target + lift * dq.astype(np.float32)

        # Left swing shaping.
        # left_hip_pitch   = 0
        # left_knee        = 3
        # left_ankle_pitch = 4
        target[0] += self.swing_hip_bias * lift
        target[3] += self.swing_knee_bias * lift
        target[4] += self.ankle_pitch_bias * lift

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target


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


def run_case(ankle: float, knee: float, hip: float, lift_height: float) -> dict:
    env = G1WBCV77CLeftLiftEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,

        template_name="right_support_c",
        preload_sign=+1.0,
        preload_amp=0.160,
        preload_start=0.08,
        preload_end=0.42,

        ankle_pitch_bias=ankle,
        lift_height=lift_height,
        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        swing_knee_bias=knee,
        swing_hip_bias=hip,

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
        "ankle": ankle,
        "knee": knee,
        "hip": hip,
        "lift_height": lift_height,
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
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v77c_left_lift_strong_preload_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ankle_values = [-0.090, -0.060, -0.030, 0.000, +0.030, +0.045, +0.060]
    knee_values = [0.000, -0.004, +0.004]
    hip_values = [0.000]
    lift_values = [0.008, 0.010]

    rows = []

    print("=" * 170)
    print("WBC V7.7C LEFT LIFT WITH STRONG PRELOAD SWEEP")
    print("Goal: add left lift after stable left unloading at Lratio≈0.247.")
    print("=" * 170)

    for lift_height in lift_values:
        for knee in knee_values:
            for hip in hip_values:
                for ankle in ankle_values:
                    row = run_case(ankle, knee, hip, lift_height)
                    rows.append(row)

                    print(
                        f"lift={row['lift_height']:.3f} "
                        f"ankle={row['ankle']:+.3f} "
                        f"knee={row['knee']:+.3f} "
                        f"hip={row['hip']:+.3f} "
                        f"steps={row['steps']:04d} "
                        f"reason={row['reason']:<16} "
                        f"Lclear={row['left_clearance']:.4f} "
                        f"Rclear={row['right_clearance']:.4f} "
                        f"Lair={row['left_air_steps']:03d} "
                        f"Lstreak={row['left_air_streak']:03d} "
                        f"air={row['first_air_step']:03d}-{row['last_air_step']:03d} "
                        f"Lforce={row['left_force_min']:.2f} "
                        f"Lratio={row['left_force_ratio']:.3f} "
                        f"up={row['min_up_z']:.3f} "
                        f"slip={row['support_slip']:.4f} "
                        f"ang={row['max_root_ang_vel']:.3f} "
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
