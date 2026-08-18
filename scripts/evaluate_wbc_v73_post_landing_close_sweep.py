from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


class G1WBCV73PostLandingCloseEnv(G1WBCV72PhaseRightLiftEnv):
    """
    V7.3: V7.2 right lift + post-landing stance-width correction.

    The lift remains the lower-gap V7.2 version:
    - preload_amp = 0.036
    - ankle_pitch_bias = -0.070
    - lift_hold_end = 0.64

    New part:
    - after landing, apply a very small roll correction
      to reduce final foot/leg gap.
    """

    def __init__(
        self,
        close_sign: float = 1.0,
        close_amp: float = 0.0,
        close_start: float = 0.82,
        close_end: float = 1.00,
        **kwargs,
    ):
        self.close_sign = float(close_sign)
        self.close_amp = float(close_amp)
        self.close_start = float(close_start)
        self.close_end = float(close_end)
        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep_close(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _close_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.close_start:
            return 0.0

        if phi < self.close_end:
            return self._smoothstep_close(
                (phi - self.close_start) / max(self.close_end - self.close_start, 1e-6)
            )

        return 1.0

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        phi = float(info["phase"])
        close = self._close_env(phi)
        c = self.close_sign * self.close_amp * close

        # Controlled 15-DOF indices:
        # left_hip_roll   = 1
        # left_ankle_roll = 5
        # right_hip_roll  = 7
        # right_ankle_roll= 11
        # waist_roll      = 13
        #
        # This is intentionally tiny and swept by sign.
        target[1] += -0.30 * c
        target[5] += +0.15 * c
        target[7] += +1.00 * c
        target[11] += -0.35 * c
        target[13] += -0.20 * c

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target


def contact_forces(env):
    right_body = int(env.model.site_bodyid[env.right_foot_site])
    left_body = int(env.model.site_bodyid[env.left_foot_site])

    right_force = 0.0
    left_force = 0.0

    for i in range(env.data.ncon):
        c = env.data.contact[i]
        b1 = int(env.model.geom_bodyid[int(c.geom1)])
        b2 = int(env.model.geom_bodyid[int(c.geom2)])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == right_body or b2 == right_body:
            right_force += normal_force

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

    return right_force, left_force


def make_env(close_sign: float, close_amp: float):
    return G1WBCV73PostLandingCloseEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,

        shift_start=0.08,
        swing_start=0.46,
        swing_end=0.62,

        target_clearance=0.0,
        target_lateral_shift=0.0,
        ik_gain=0.0,
        z_lift_weight=0.0,
        xy_hold_weight=0.0,
        support_lock_weight=0.0,
        support_xy_weight=0.0,
        support_z_weight=0.0,
        support_ik_gain=0.0,

        tiny_lift_height=0.010,
        lift_start=0.46,
        lift_peak=0.62,
        land_end=0.82,

        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        preload_sign=-1.0,
        preload_amp=0.036,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=-0.070,

        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=0.64,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,

        swing_knee_bias=0.004,
        swing_hip_bias=0.002,

        close_sign=close_sign,
        close_amp=close_amp,
        close_start=0.82,
        close_end=1.00,
    )


def run_case(close_sign: float, close_amp: float) -> dict:
    env = make_env(close_sign, close_amp)

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    initial_gap = abs(
        float(env.data.site_xpos[env.right_foot_site][1])
        - float(env.data.site_xpos[env.left_foot_site][1])
    )

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    right_air_steps = 0
    max_right_air_streak = 0
    current_air_streak = 0

    min_right_force = 1e9
    right_ratio_at_min = 1.0

    max_gap = initial_gap

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        final = info

        phi = float(info["phase"])

        current_gap = abs(
            float(env.data.site_xpos[env.right_foot_site][1])
            - float(env.data.site_xpos[env.left_foot_site][1])
        )
        max_gap = max(max_gap, current_gap)

        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            current_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, current_air_streak)
        else:
            current_air_streak = 0

        if 0.46 <= phi <= 0.82:
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= 520:
            break

    final_gap = abs(
        float(env.data.site_xpos[env.right_foot_site][1])
        - float(env.data.site_xpos[env.left_foot_site][1])
    )

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "close_sign": close_sign,
        "close_amp": close_amp,
        "steps": steps,
        "reason": reason,
        "right_clearance": max_right_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "initial_gap": initial_gap,
        "max_gap": max_gap,
        "final_gap": final_gap,
        "gap_delta": final_gap - initial_gap,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v73_post_landing_close_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    signs = [-1.0, 1.0]
    amps = [0.000, 0.004, 0.008, 0.012, 0.016, 0.020]

    rows = []

    print("=" * 150)
    print("WBC V7.3 POST-LANDING CLOSE-STANCE SWEEP")
    print("Goal: preserve right lift while reducing final foot gap after landing.")
    print("=" * 150)

    for sign in signs:
        for amp in amps:
            row = run_case(sign, amp)
            rows.append(row)

            print(
                f"sign={sign:+.0f} "
                f"amp={amp:.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['right_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"Rstreak={row['right_air_streak']:03d} "
                f"Rforce={row['right_force_min']:.2f} "
                f"up={row['min_up_z']:.3f} "
                f"slip={row['support_slip']:.4f} "
                f"gap0={row['initial_gap']:.4f} "
                f"gapFinal={row['final_gap']:.4f} "
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
