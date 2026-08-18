from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_neutral_only_env import G1WBCNeutralOnlyEnv


def parse_action(text: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--fixed_action", default="0,0,0,0")
    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v7_neutral_only.csv",
    )

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--cycle_duration", type=float, default=5.2)
    parser.add_argument("--shift_start", type=float, default=0.08)
    parser.add_argument("--swing_start", type=float, default=0.46)
    parser.add_argument("--swing_end", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    action = parse_action(args.fixed_action)

    print("=" * 118)
    print("WBC V7 NEUTRAL-ONLY DIAGNOSTIC")
    print("Fixed action:", action.tolist())
    print("This test bypasses parent swing, support push, support IK, capture, and touchdown.")
    print("=" * 118)

    rows = []

    for ep in range(args.episodes):
        env = G1WBCNeutralOnlyEnv(
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            cycle_duration=args.cycle_duration,
            shift_start=args.shift_start,
            swing_start=args.swing_start,
            swing_end=args.swing_end,
            land_end=args.land_end,
            target_clearance=0.0,
            target_lateral_shift=0.0,
            ik_gain=0.0,
            z_lift_weight=0.0,
            xy_hold_weight=0.0,
            support_lock_weight=0.0,
            support_xy_weight=0.0,
            support_z_weight=0.0,
            support_ik_gain=0.0,
        )

        obs, info = env.reset()

        total_reward = 0.0
        steps = 0

        max_right_clearance = 0.0
        max_left_clearance = 0.0
        min_up_z = 1.0
        max_root_ang_vel = 0.0
        max_support_slip = 0.0

        terminated = False
        truncated = False
        final = info

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1
            final = info

            max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
            max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
            min_up_z = min(min_up_z, float(info["up_z"]))
            max_root_ang_vel = max(max_root_ang_vel, float(info["root_ang_vel"]))
            max_support_slip = max(max_support_slip, float(info["support_slip"]))

            if steps >= args.max_steps:
                break

        reason = env.termination_reason(final) if terminated else "max_steps"

        row = {
            "episode": ep,
            "steps": steps,
            "reward": total_reward,
            "main_clearance": max_right_clearance,
            "max_left_clearance": max_left_clearance,
            "min_up_z": min_up_z,
            "max_root_ang_vel": max_root_ang_vel,
            "support_slip": max_support_slip,
            "final_x": float(final["x_position"]),
            "final_y": float(final["y_position"]),
            "final_x_velocity": float(final["x_velocity"]),
            "final_y_velocity": float(final["y_velocity"]),
            "base_height": float(final["base_height"]),
            "root_ang_vel": float(final["root_ang_vel"]),
            "left_contact": int(bool(final["left_contact"])),
            "right_contact": int(bool(final["right_contact"])),
            "reason": reason,
        }

        rows.append(row)

        print(
            f"ep={ep:02d} "
            f"steps={steps:04d} "
            f"reward={total_reward:+.1f} "
            f"Rclear={max_right_clearance:.4f} "
            f"LclearMax={max_left_clearance:.4f} "
            f"up={min_up_z:.3f} "
            f"x={row['final_x']:+.3f} "
            f"y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} "
            f"yv={row['final_y_velocity']:+.3f} "
            f"ang={row['root_ang_vel']:.3f} "
            f"L={row['left_contact']} "
            f"R={row['right_contact']} "
            f"reason={reason}"
        )

        env.close()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("SUMMARY")
    for key in [
        "steps",
        "main_clearance",
        "max_left_clearance",
        "min_up_z",
        "max_root_ang_vel",
        "support_slip",
        "final_x",
        "final_x_velocity",
        "base_height",
        "root_ang_vel",
    ]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        print(
            f"{key:<22} "
            f"mean={vals.mean():.4f} "
            f"std={vals.std():.4f} "
            f"min={vals.min():.4f} "
            f"max={vals.max():.4f}"
        )

    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
