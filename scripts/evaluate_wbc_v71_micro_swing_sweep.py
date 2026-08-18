from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_ankle_pitch_sweep import G1WBCV7AnklePitchSweepEnv


class G1WBCV71MicroSwingEnv(G1WBCV7AnklePitchSweepEnv):
    """
    V7.1 micro swing shaping.

    Starts from the successful final V7 controller:
    - preload_amp = 0.040
    - ankle_pitch_bias = -0.075
    - foot_ik_gain = 0.55
    - lift_height = 0.010

    Adds:
    - tiny right knee flexion during swing
    - optional tiny right hip pitch assist
    """

    def __init__(
        self,
        swing_knee_bias: float = 0.0,
        swing_hip_bias: float = 0.0,
        **kwargs,
    ):
        self.swing_knee_bias = float(swing_knee_bias)
        self.swing_hip_bias = float(swing_hip_bias)
        super().__init__(**kwargs)

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        lift = self._tiny_lift_env(float(info["phase"]))

        # Controlled 15-DOF index:
        # right_hip_pitch = 6
        # right_knee      = 9
        target[6] += self.swing_hip_bias * lift
        target[9] += self.swing_knee_bias * lift

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


def run_case(args, knee_bias: float, hip_bias: float) -> dict:
    env = G1WBCV71MicroSwingEnv(
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

        tiny_lift_height=args.lift_height,
        lift_start=args.lift_start,
        lift_peak=args.lift_peak,
        land_end=args.land_end,

        foot_ik_gain=args.foot_ik_gain,
        foot_ik_damping=0.080,
        foot_ik_max_delta=args.foot_ik_max_delta,

        preload_sign=-1.0,
        preload_amp=args.preload_amp,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=args.ankle_pitch_bias,

        swing_knee_bias=knee_bias,
        swing_hip_bias=hip_bias,
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

    right_air_steps = 0
    right_air_streak = 0
    max_right_air_streak = 0

    min_right_force = 1e9
    right_ratio_at_min = 1.0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])

        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
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
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "knee_bias": knee_bias,
        "hip_bias": hip_bias,
        "steps": steps,
        "reason": reason,
        "right_clearance": max_right_clearance,
        "left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
        "max_root_ang_vel": max_root_ang_vel,
        "support_slip": max_support_slip,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--cycle_duration", type=float, default=5.2)

    parser.add_argument("--preload_amp", type=float, default=0.040)
    parser.add_argument("--ankle_pitch_bias", type=float, default=-0.075)
    parser.add_argument("--lift_height", type=float, default=0.010)
    parser.add_argument("--foot_ik_gain", type=float, default=0.55)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)

    parser.add_argument("--lift_start", type=float, default=0.46)
    parser.add_argument("--lift_peak", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v71_micro_swing_sweep.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    knee_values = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
    hip_values = [0.000, 0.002]

    rows = []

    print("=" * 138)
    print("WBC V7.1 MICRO SWING-SHAPING SWEEP")
    print("Final V7 lift + tiny knee/hip swing shaping.")
    print("=" * 138)

    for hip_bias in hip_values:
        for knee_bias in knee_values:
            row = run_case(args, knee_bias, hip_bias)
            rows.append(row)

            print(
                f"hip={hip_bias:.3f} "
                f"knee={knee_bias:.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['right_clearance']:.4f} "
                f"Lclear={row['left_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"Rstreak={row['right_air_streak']:03d} "
                f"RforceMin={row['right_force_min']:.2f} "
                f"Rratio={row['right_force_ratio']:.3f} "
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
