from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v77c_left_lift_strong_preload_sweep import (
    G1WBCV77CLeftLiftEnv,
    contact_forces,
)


class G1WBCV77DLeftLiftFadeEnv(G1WBCV77CLeftLiftEnv):
    """
    V7.7D: left lift with proper preload fade.

    Fixes V7.7C problem:
    - V7.7C inherited diagnostic preload that stayed ON forever.
    - V7.7D fades preload back to neutral after landing.
    """

    def _preload_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.preload_start:
            return 0.0

        if phi < self.preload_end:
            return self._smoothstep_lift(
                (phi - self.preload_start)
                / max(self.preload_end - self.preload_start, 1e-6)
            )

        if phi < self.preload_down_start:
            return 1.0

        if phi < self.preload_down_end:
            down = self._smoothstep_lift(
                (phi - self.preload_down_start)
                / max(self.preload_down_end - self.preload_down_start, 1e-6)
            )
            return 1.0 - down

        return 0.0


def run_case(preload_amp: float, ankle: float, knee: float, lift_height: float) -> dict:
    env = G1WBCV77DLeftLiftFadeEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,

        template_name="right_support_c",
        preload_sign=+1.0,
        preload_amp=preload_amp,
        preload_start=0.08,
        preload_end=0.42,

        ankle_pitch_bias=ankle,
        lift_height=lift_height,
        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        swing_knee_bias=knee,
        swing_hip_bias=0.000,

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
    left_air_swing = 0
    left_air_return = 0
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

            if 0.46 <= phi <= 0.82:
                left_air_swing += 1
            else:
                left_air_return += 1

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
        "preload_amp": preload_amp,
        "ankle": ankle,
        "knee": knee,
        "lift_height": lift_height,
        "steps": steps,
        "reason": reason,
        "left_clearance": max_left_clearance,
        "right_clearance": max_right_clearance,
        "left_air_steps": left_air_steps,
        "left_air_swing": left_air_swing,
        "left_air_return": left_air_return,
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
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v77d_left_lift_preload_fade_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    preload_values = [0.150, 0.160, 0.170]
    lift_values = [0.008, 0.010, 0.012]
    ankle_values = [-0.090, -0.060, +0.045, +0.060]
    knee_values = [0.000, -0.004]

    rows = []

    print("=" * 175)
    print("WBC V7.7D LEFT LIFT WITH PRELOAD FADE SWEEP")
    print("Goal: get left-foot air during the real swing window, not late after the controller ends.")
    print("=" * 175)

    for preload in preload_values:
        for lift_height in lift_values:
            for knee in knee_values:
                for ankle in ankle_values:
                    row = run_case(preload, ankle, knee, lift_height)
                    rows.append(row)

                    print(
                        f"pre={row['preload_amp']:.3f} "
                        f"lift={row['lift_height']:.3f} "
                        f"ankle={row['ankle']:+.3f} "
                        f"knee={row['knee']:+.3f} "
                        f"steps={row['steps']:04d} "
                        f"reason={row['reason']:<16} "
                        f"Lclear={row['left_clearance']:.4f} "
                        f"Lair={row['left_air_steps']:03d} "
                        f"Lswing={row['left_air_swing']:03d} "
                        f"Lreturn={row['left_air_return']:03d} "
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
