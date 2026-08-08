from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_hybrid_wbc_lite_lift_env import G1HybridWBCLiteLiftEnv


def parse_action(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("Action must contain exactly 6 comma-separated values.")
    return np.asarray(vals, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid WBC-lite open-loop controller with fixed 6D residual action.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--fixed_action", type=str, default="0,0,0,0,0,0")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--cycle_duration", type=float, default=3.8)
    parser.add_argument("--swing_start", type=float, default=0.30)
    parser.add_argument("--swing_end", type=float, default=0.80)
    parser.add_argument("--target_clearance", type=float, default=0.008)
    parser.add_argument("--target_lateral_shift", type=float, default=0.018)
    parser.add_argument("--x_soft_limit", type=float, default=0.12)
    parser.add_argument("--x_hard_limit", type=float, default=0.35)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.28)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.20)
    parser.add_argument("--action_smoothing", type=float, default=0.75)
    args = parser.parse_args()

    action = parse_action(args.fixed_action)

    rows = []
    print("=" * 100)
    print("HYBRID WBC-LITE OPEN-LOOP EVALUATION")
    print("Fixed action:", action.tolist())
    print("=" * 100)

    for ep in range(args.episodes):
        env = G1HybridWBCLiteLiftEnv(
            model_path=args.model_path,
            stage=args.stage,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            cycle_duration=args.cycle_duration,
            swing_start=args.swing_start,
            swing_end=args.swing_end,
            target_clearance=args.target_clearance,
            target_lateral_shift=args.target_lateral_shift,
            x_soft_limit=args.x_soft_limit,
            x_hard_limit=args.x_hard_limit,
            x_velocity_soft_limit=args.x_velocity_soft_limit,
            x_velocity_hard_limit=args.x_velocity_hard_limit,
            action_smoothing=args.action_smoothing,
            randomize_reset=False,
        )

        obs, info = env.reset()
        done = False
        steps = 0
        total = 0.0
        max_clear = 0.0
        min_up = 1.0
        contacts = []
        slips = []
        final = info

        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total += float(reward)
            max_clear = max(max_clear, float(info["main_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            contacts.append(float(info["contact_accuracy"]))
            slips.append(float(info["support_slip"]))
            final = info

        reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"
        strict = int(
            steps >= args.max_steps
            and max_clear >= 0.008
            and abs(float(final["x_position"])) <= 0.20
            and min_up >= 0.85
        )
        relaxed = int(
            steps >= 600
            and max_clear >= 0.008
            and abs(float(final["x_position"])) <= args.x_hard_limit
            and min_up >= 0.80
        )

        row = {
            "episode": ep,
            "steps": steps,
            "reward": total,
            "main_clearance": max_clear,
            "min_up_z": min_up,
            "contact": float(np.mean(contacts)) if contacts else 0.0,
            "slip": float(np.mean(slips)) if slips else 0.0,
            "final_x": float(final["x_position"]),
            "final_y": float(final["y_position"]),
            "final_x_velocity": float(final["x_velocity"]),
            "reason": reason,
            "strict_pass": strict,
            "relaxed_pass": relaxed,
        }
        rows.append(row)
        print(
            f"ep={ep:02d} steps={steps:04d} reward={total:+.1f} clear={max_clear:.4f} "
            f"up={min_up:.3f} contact={row['contact']:.3f} slip={row['slip']:.3f} "
            f"x={row['final_x']:+.3f} y={row['final_y']:+.3f} xv={row['final_x_velocity']:+.3f} "
            f"strict={strict} relaxed={relaxed} reason={reason}"
        )

        env.close()

    print("\nSUMMARY")
    for key in ["steps", "reward", "main_clearance", "min_up_z", "contact", "slip", "final_x", "final_y", "final_x_velocity"]:
        vals = np.asarray([float(r[key]) for r in rows])
        print(f"{key:<22s} mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")
    print(f"\nSTRICT PASS RATE:  {100*np.mean([r['strict_pass'] for r in rows]):.1f}%")
    print(f"RELAXED PASS RATE: {100*np.mean([r['relaxed_pass'] for r in rows]):.1f}%")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV saved:", args.csv)


if __name__ == "__main__":
    main()
