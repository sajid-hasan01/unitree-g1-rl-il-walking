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

from envs.g1_solution_env import G1SolutionEnv


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
    parser = argparse.ArgumentParser(description="Evaluate G1 solution curriculum policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stage", type=str, default="tiny_walk")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--action_scale", type=float, default=0.55)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="results/g1_solution_eval_summary.csv")
    args = parser.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    env = G1SolutionEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        stage=args.stage,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=None, device="auto")

    rows = []

    print("=" * 90)
    print("G1 SOLUTION EVALUATION")
    print("Model:", args.model)
    print("Stage:", args.stage)
    print("Episodes:", args.episodes)
    print("=" * 90)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        total_reward = 0.0
        steps = 0

        max_left_clearance = 0.0
        max_right_clearance = 0.0
        max_clearance = 0.0
        min_up_z = 1.0
        contact_scores = []
        support_slips = []
        x_positions = []
        y_positions = []
        x_velocities = []
        y_velocities = []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            lclr = float(info.get("left_foot_clearance", 0.0))
            rclr = float(info.get("right_foot_clearance", 0.0))
            max_left_clearance = max(max_left_clearance, lclr)
            max_right_clearance = max(max_right_clearance, rclr)
            max_clearance = max(max_clearance, lclr, rclr)
            min_up_z = min(min_up_z, float(info.get("up_z", 1.0)))
            contact_scores.append(float(info.get("contact_accuracy", 0.0)))
            support_slips.append(float(info.get("support_slip", 0.0)))
            x_positions.append(float(info.get("x_position", 0.0)))
            y_positions.append(float(info.get("y_position", 0.0)))
            x_velocities.append(float(info.get("x_velocity", 0.0)))
            y_velocities.append(float(info.get("y_velocity", 0.0)))

        row = {
            "episode": ep,
            "stage": args.stage,
            "steps": steps,
            "total_reward": total_reward,
            "max_left_clearance": max_left_clearance,
            "max_right_clearance": max_right_clearance,
            "max_clearance": max_clearance,
            "min_up_z": min_up_z,
            "mean_contact_accuracy": float(np.mean(contact_scores)),
            "mean_support_slip": float(np.mean(support_slips)),
            "final_x": float(x_positions[-1]) if x_positions else 0.0,
            "final_y": float(y_positions[-1]) if y_positions else 0.0,
            "mean_x_velocity": float(np.mean(x_velocities)),
            "mean_y_velocity": float(np.mean(y_velocities)),
        }
        rows.append(row)

        print(
            f"ep={ep:02d} steps={steps:04d} reward={total_reward:+.2f} "
            f"maxClr L/R={max_left_clearance:.3f}/{max_right_clearance:.3f} "
            f"minUp={min_up_z:.3f} contact={row['mean_contact_accuracy']:.3f} "
            f"slip={row['mean_support_slip']:.3f} final_xy={row['final_x']:+.3f}/{row['final_y']:+.3f}"
        )

    keys = list(rows[0].keys()) if rows else []
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("SUMMARY")
    for key in ["steps", "total_reward", "max_clearance", "min_up_z", "mean_contact_accuracy", "mean_support_slip"]:
        stats = summarize([float(r[key]) for r in rows])
        print(f"{key:24s} mean={stats['mean']:.4f} std={stats['std']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}")

    print(f"CSV saved: {args.csv}")
    env.close()


if __name__ == "__main__":
    main()
