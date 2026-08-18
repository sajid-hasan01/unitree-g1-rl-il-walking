from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_knee_bias_sweep import G1WBCV7PreloadKneeBiasEnv


class G1WBCV7PulsePreloadEnv(G1WBCV7PreloadKneeBiasEnv):
    """
    V7 pulse preload.

    Difference from previous preload:
    - old preload ramps up and stays active until the end
    - this preload ramps up, holds through lift/landing, then ramps down
    """

    def __init__(
        self,
        preload_down_start: float = 0.78,
        preload_down_end: float = 0.98,
        **kwargs,
    ):
        self.preload_down_start = float(preload_down_start)
        self.preload_down_end = float(preload_down_end)
        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep_pulse(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _preload_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.preload_start:
            return 0.0

        if phi < self.preload_end:
            return self._smoothstep_pulse(
                (phi - self.preload_start) / max(self.preload_end - self.preload_start, 1e-6)
            )

        if phi < self.preload_down_start:
            return 1.0

        if phi < self.preload_down_end:
            down = self._smoothstep_pulse(
                (phi - self.preload_down_start) / max(self.preload_down_end - self.preload_down_start, 1e-6)
            )
            return 1.0 - down

        return 0.0


def contact_forces(env):
    right_body = int(env.model.site_bodyid[env.right_foot_site])
    left_body = int(env.model.site_bodyid[env.left_foot_site])

    right_force = 0.0
    left_force = 0.0
    right_contacts = 0
    left_contacts = 0

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        b1 = int(env.model.geom_bodyid[g1])
        b2 = int(env.model.geom_bodyid[g2])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == right_body or b2 == right_body:
            right_force += normal_force
            right_contacts += 1

        if b1 == left_body or b2 == left_body:
            left_force += normal_force
            left_contacts += 1

    return right_force, left_force, right_contacts, left_contacts


def run_case(args, preload_amp: float, knee_bias: float) -> dict:
    env = G1WBCV7PulsePreloadEnv(
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,

        shift_start=0.08,
        swing_start=args.lift_start,
        swing_end=args.lift_peak,

        target_clearance=0.0,
        target_lateral_shift=0.0,
        ik_gain=0.0,
        z_lift_weight=0.0,
        xy_hold_weight=0.0,
        support_lock_weight=0.0,
        support_xy_weight=0.0,
        support_z_weight=0.0,
        support_ik_gain=0.0,

        tiny_lift_height=args.tiny_lift_height,
        lift_start=args.lift_start,
        lift_peak=args.lift_peak,
        land_end=args.land_end,

        foot_ik_gain=args.foot_ik_gain,
        foot_ik_damping=args.foot_ik_damping,
        foot_ik_max_delta=args.foot_ik_max_delta,

        preload_sign=args.preload_sign,
        preload_amp=preload_amp,
        preload_start=args.preload_start,
        preload_end=args.preload_end,
        preload_down_start=args.preload_down_start,
        preload_down_end=args.preload_down_end,

        knee_bias=knee_bias,
        hip_ratio=args.hip_ratio,
        ankle_ratio=args.ankle_ratio,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    max_left_clearance = 0.0
    min_up_z = 1.0
    max_root_ang_vel = 0.0
    max_support_slip = 0.0

    min_right_force = 1e9
    force_at_best = None

    right_air_steps = 0
    right_air_streak = 0
    max_right_air_streak = 0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        final = info

        phi = float(info["phase"])
        rc = float(info["right_foot_clearance"])
        lc = float(info["left_foot_clearance"])

        max_right_clearance = max(max_right_clearance, rc)
        max_left_clearance = max(max_left_clearance, lc)
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_root_ang_vel = max(max_root_ang_vel, float(info["root_ang_vel"]))
        max_support_slip = max(max_support_slip, float(info["support_slip"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            right_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, right_air_streak)
        else:
            right_air_streak = 0

        if args.lift_start <= phi <= args.land_end:
            rf, lf, rcon, lcon = contact_forces(env)

            if rf < min_right_force:
                min_right_force = rf
                force_at_best = {
                    "phase_min_force": phi,
                    "right_force_min": rf,
                    "left_force_at_min": lf,
                    "right_contacts_at_min": rcon,
                    "left_contacts_at_min": lcon,
                    "right_clear_at_min": rc,
                    "left_clear_at_min": lc,
                    "up_z_at_min": float(info["up_z"]),
                    "x_at_min": float(info["x_position"]),
                    "y_at_min": float(info["y_position"]),
                }

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    if force_at_best is None:
        force_at_best = {
            "phase_min_force": -1.0,
            "right_force_min": -1.0,
            "left_force_at_min": -1.0,
            "right_contacts_at_min": -1,
            "left_contacts_at_min": -1,
            "right_clear_at_min": -1.0,
            "left_clear_at_min": -1.0,
            "up_z_at_min": -1.0,
            "x_at_min": -1.0,
            "y_at_min": -1.0,
        }

    total_force = force_at_best["right_force_min"] + force_at_best["left_force_at_min"]
    right_force_ratio = force_at_best["right_force_min"] / max(total_force, 1e-6)

    row = {
        "preload_amp": preload_amp,
        "knee_bias": knee_bias,
        "steps": steps,
        "reason": reason,
        "max_right_clearance": max_right_clearance,
        "max_left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "min_up_z": min_up_z,
        "max_root_ang_vel": max_root_ang_vel,
        "support_slip": max_support_slip,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "right_force_ratio": right_force_ratio,
        **force_at_best,
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--cycle_duration", type=float, default=5.2)

    parser.add_argument("--tiny_lift_height", type=float, default=0.010)
    parser.add_argument("--foot_ik_gain", type=float, default=0.55)
    parser.add_argument("--foot_ik_damping", type=float, default=0.080)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)

    parser.add_argument("--preload_sign", type=float, default=-1.0)
    parser.add_argument("--preload_start", type=float, default=0.08)
    parser.add_argument("--preload_end", type=float, default=0.42)
    parser.add_argument("--preload_down_start", type=float, default=0.78)
    parser.add_argument("--preload_down_end", type=float, default=0.98)

    parser.add_argument("--lift_start", type=float, default=0.46)
    parser.add_argument("--lift_peak", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument("--hip_ratio", type=float, default=0.00)
    parser.add_argument("--ankle_ratio", type=float, default=0.00)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v7_pulse_preload_force_sweep.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    preload_values = [0.038, 0.042, 0.046, 0.050, 0.054]
    knee_values = [0.000, 0.002, 0.004]

    rows = []

    print("=" * 130)
    print("WBC V7 PULSE-PRELOAD FORCE SWEEP")
    print("Preload ramps up, holds through lift, then returns to neutral.")
    print("=" * 130)

    for preload_amp in preload_values:
        for knee_bias in knee_values:
            row = run_case(args, preload_amp, knee_bias)
            rows.append(row)

            print(
                f"pre={preload_amp:.3f} "
                f"knee={knee_bias:.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['max_right_clearance']:.4f} "
                f"Lclear={row['max_left_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"RforceMin={row['right_force_min']:.2f} "
                f"LforceAtMin={row['left_force_at_min']:.2f} "
                f"Rratio={row['right_force_ratio']:.3f} "
                f"phi={row['phase_min_force']:.3f} "
                f"up={row['min_up_z']:.3f} "
                f"x={row['final_x']:+.3f} "
                f"y={row['final_y']:+.3f} "
                f"xv={row['final_x_velocity']:+.3f} "
                f"ang={row['max_root_ang_vel']:.3f} "
                f"slip={row['support_slip']:.4f}"
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
