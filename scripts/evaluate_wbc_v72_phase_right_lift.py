from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


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


def run_episode(args, ep: int) -> dict:
    env = G1WBCV72PhaseRightLiftEnv(
        frame_skip=5,
        max_steps=args.max_steps,
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

        tiny_lift_height=args.lift_height,
        lift_start=0.46,
        lift_peak=0.62,
        land_end=0.82,

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

        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=args.lift_hold_end,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,

        swing_knee_bias=args.swing_knee_bias,
        swing_hip_bias=args.swing_hip_bias,
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
    max_ang = 0.0
    max_slip = 0.0

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
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        max_slip = max(max_slip, float(info["support_slip"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            right_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, right_air_streak)
        else:
            right_air_streak = 0

        if 0.46 <= phi <= 0.82:
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "episode": ep,
        "steps": steps,
        "reason": reason,
        "right_clearance": max_right_clearance,
        "left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
        "max_root_ang_vel": max_ang,
        "support_slip": max_slip,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "final_phase": str(final.get("v72_phase_name", "unknown")),
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=520)

    parser.add_argument("--preload_amp", type=float, default=0.040)
    parser.add_argument("--ankle_pitch_bias", type=float, default=-0.075)
    parser.add_argument("--lift_height", type=float, default=0.010)
    parser.add_argument("--foot_ik_gain", type=float, default=0.55)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)

    parser.add_argument("--lift_hold_end", type=float, default=0.68)
    parser.add_argument("--swing_knee_bias", type=float, default=0.000)
    parser.add_argument("--swing_hip_bias", type=float, default=0.000)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v72_phase_right_lift_eval.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 132)
    print("WBC V7.2 PHASE RIGHT-LIFT EVALUATION")
    print("stand → preload → lift → hold → land → return neutral")
    print("=" * 132)

    for ep in range(args.episodes):
        row = run_episode(args, ep)
        rows.append(row)

        print(
            f"ep={ep:02d} "
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
            f"slip={row['support_slip']:.4f} "
            f"final_phase={row['final_phase']}"
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
