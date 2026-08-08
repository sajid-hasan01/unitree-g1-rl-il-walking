from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs.g1_hybrid_wbc_lite_lift_env import G1HybridWBCLiteLiftEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid WBC-lite foot-lift controller.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--cycle_duration", type=float, default=3.6)
    parser.add_argument("--swing_start", type=float, default=0.28)
    parser.add_argument("--swing_end", type=float, default=0.78)
    parser.add_argument("--target_clearance", type=float, default=0.012)
    parser.add_argument("--target_lateral_shift", type=float, default=0.032)
    parser.add_argument("--x_soft_limit", type=float, default=0.08)
    parser.add_argument("--x_hard_limit", type=float, default=0.20)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.22)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.00)
    parser.add_argument("--action_smoothing", type=float, default=0.70)
    args = parser.parse_args()

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
    model = PPO.load(args.model, env=env)
    print("=" * 100)
    print("HYBRID WBC-LITE EVALUATION")
    print("Model:", args.model)
    print("=" * 100)

    rows = []
    for ep in range(args.episodes):
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
            action, _ = model.predict(obs, deterministic=args.deterministic)
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
            "strict_pass": int(steps >= args.max_steps and max_clear >= 0.010 and abs(float(final["x_position"])) <= 0.16 and min_up >= 0.82),
            "relaxed_pass": int(steps >= 300 and max_clear >= 0.008 and abs(float(final["x_position"])) <= args.x_hard_limit and min_up >= 0.78),
        }
        rows.append(row)
        print(
            f"ep={ep:02d} steps={steps:04d} reward={total:+.1f} clear={max_clear:.4f} "
            f"up={min_up:.3f} contact={row['contact']:.3f} slip={row['slip']:.3f} "
            f"x={row['final_x']:+.3f} xv={row['final_x_velocity']:+.3f} "
            f"strict={row['strict_pass']} relaxed={row['relaxed_pass']} reason={reason}"
        )

    print("\nSUMMARY")
    for key in ["steps", "reward", "main_clearance", "min_up_z", "contact", "slip", "final_x", "final_x_velocity"]:
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
    env.close()


if __name__ == "__main__":
    main()
