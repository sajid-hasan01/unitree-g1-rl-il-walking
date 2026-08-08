from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs.g1_com_shift_lift_env import G1ComShiftLiftEnv


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate COM-shift + lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stage", type=str, default="right_lift")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--action_scale", type=float, default=0.28)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="results/g1_com_shift_lift_eval.csv")
    args = parser.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    env = G1ComShiftLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        stage=args.stage,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        frame_skip=args.frame_skip,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=None, device="auto")

    rows = []

    print("=" * 100)
    print("COM-SHIFT + LIFT EVALUATION")
    print("Model:", args.model)
    print("Stage:", args.stage)
    print("=" * 100)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=9000 + ep)
        done = False
        total_reward = 0.0
        steps = 0
        max_left = 0.0
        max_right = 0.0
        min_up = 1.0
        contacts = []
        slips = []
        x_positions = []
        y_positions = []
        final_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            max_left = max(max_left, float(info.get("left_foot_clearance", 0.0)))
            max_right = max(max_right, float(info.get("right_foot_clearance", 0.0)))
            min_up = min(min_up, float(info.get("up_z", 1.0)))
            contacts.append(float(info.get("contact_accuracy", 0.0)))
            slips.append(float(info.get("support_slip", 0.0)))
            x_positions.append(float(info.get("x_position", 0.0)))
            y_positions.append(float(info.get("y_position", 0.0)))
            final_info = info

        row = {
            "episode": ep,
            "stage": args.stage,
            "steps": steps,
            "total_reward": total_reward,
            "max_left_clearance": max_left,
            "max_right_clearance": max_right,
            "max_clearance": max(max_left, max_right),
            "min_up_z": min_up,
            "mean_contact_accuracy": float(np.mean(contacts)) if contacts else 0.0,
            "mean_support_slip": float(np.mean(slips)) if slips else 0.0,
            "final_x": float(final_info.get("x_position", 0.0)),
            "final_y": float(final_info.get("y_position", 0.0)),
            "final_up_z": float(final_info.get("up_z", 0.0)),
            "final_height": float(final_info.get("base_height", 0.0)),
        }
        rows.append(row)

        print(
            f"ep={ep:02d} steps={steps:04d} reward={total_reward:+.2f} "
            f"Lclr={max_left:.3f} Rclr={max_right:.3f} minUp={min_up:.3f} "
            f"contact={row['mean_contact_accuracy']:.3f} slip={row['mean_support_slip']:.3f} "
            f"final_xy={row['final_x']:+.3f}/{row['final_y']:+.3f}"
        )

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("SUMMARY")
    for key in ["steps", "total_reward", "max_clearance", "min_up_z", "mean_contact_accuracy", "mean_support_slip"]:
        s = summarize([float(row[key]) for row in rows])
        print(f"{key:24s} mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")

    print("CSV saved:", args.csv)
    env.close()


if __name__ == "__main__":
    main()
