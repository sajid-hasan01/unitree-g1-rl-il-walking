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

from envs.g1_right_lift_env import G1RightLiftEnv


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def pass_status(row: Dict[str, float], relaxed: bool = False) -> bool:
    if relaxed:
        return (
            row["steps"] >= 350
            and row["max_right_clearance"] >= 0.025
            and row["min_up_z"] >= 0.82
            and row["mean_support_slip"] <= 0.24
            and abs(row["final_x"]) <= 0.42
        )

    return (
        row["steps"] >= 450
        and row["max_right_clearance"] >= 0.025
        and row["min_up_z"] >= 0.84
        and row["mean_support_slip"] <= 0.24
        and abs(row["final_x"]) <= 0.35
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate right_lift sagittal-stability policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--action_scale", type=float, default=0.14)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.6)
    parser.add_argument("--support_leg_scale", type=float, default=1.0)
    parser.add_argument("--swing_leg_scale", type=float, default=0.3)
    parser.add_argument("--waist_scale", type=float, default=1.0)
    parser.add_argument("--sagittal_kp", type=float, default=0.70)
    parser.add_argument("--sagittal_kd", type=float, default=0.12)
    parser.add_argument("--sagittal_clip", type=float, default=0.30)
    parser.add_argument("--sagittal_hip_sign", type=float, default=1.0)
    parser.add_argument("--sagittal_ankle_sign", type=float, default=1.0)
    parser.add_argument("--disable_scripted_arms", action="store_true")
    parser.add_argument("--arm_swing_scale", type=float, default=0.25)
    parser.add_argument("--arm_pitch_sign", type=float, default=1.0)
    parser.add_argument("--arm_elbow_scale", type=float, default=0.10)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="results/right_lift_sagittal_eval.csv")
    args = parser.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    env = G1RightLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        support_leg_scale=args.support_leg_scale,
        swing_leg_scale=args.swing_leg_scale,
        waist_scale=args.waist_scale,
        sagittal_kp=args.sagittal_kp,
        sagittal_kd=args.sagittal_kd,
        sagittal_clip=args.sagittal_clip,
        sagittal_hip_sign=args.sagittal_hip_sign,
        sagittal_ankle_sign=args.sagittal_ankle_sign,
        frame_skip=args.frame_skip,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=None, device="auto")

    rows = []

    print("=" * 100)
    print("RIGHT_LIFT SAGITTAL-STABILITY EVALUATION")
    print("Model:", args.model)
    print("Episodes:", args.episodes)
    print("Sagittal feedback:", "kp=", args.sagittal_kp, "kd=", args.sagittal_kd, "clip=", args.sagittal_clip)
    print("Sagittal signs:", "hip=", args.sagittal_hip_sign, "ankle=", args.sagittal_ankle_sign)
    print("Scripted arms:", "enabled=", not args.disable_scripted_arms, "scale=", args.arm_swing_scale, "pitch_sign=", args.arm_pitch_sign, "elbow=", args.arm_elbow_scale)
    print("=" * 100)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=5000 + ep)
        done = False
        total_reward = 0.0
        steps = 0
        max_right_clearance = 0.0
        max_left_clearance = 0.0
        min_up_z = 1.0
        min_height = 10.0
        contacts = []
        slips = []
        backward_excesses = []
        x_positions = []
        y_positions = []
        x_velocities = []
        y_velocities = []
        com_x_errors = []
        sagittal_corrections = []
        final_info = info
        terminated = False
        truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
            max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
            min_up_z = min(min_up_z, float(info["up_z"]))
            min_height = min(min_height, float(info["base_height"]))
            contacts.append(float(info["contact_accuracy"]))
            slips.append(float(info.get("support_slip", 0.0)))
            backward_excesses.append(float(info.get("backward_excess", 0.0)))
            x_positions.append(float(info["x_position"]))
            y_positions.append(float(info["y_position"]))
            x_velocities.append(float(info["x_velocity"]))
            y_velocities.append(float(info["y_velocity"]))
            com_x_errors.append(float(info.get("com_x_error", 0.0)))
            sagittal_corrections.append(float(info.get("sagittal_correction", 0.0)))
            final_info = info

        row = {
            "episode": ep,
            "steps": steps,
            "total_reward": total_reward,
            "max_right_clearance": max_right_clearance,
            "max_left_clearance": max_left_clearance,
            "min_up_z": min_up_z,
            "min_height": min_height,
            "mean_contact_accuracy": float(np.mean(contacts)) if contacts else 0.0,
            "mean_support_slip": float(np.mean(slips)) if slips else 0.0,
            "max_backward_excess": float(np.max(backward_excesses)) if backward_excesses else 0.0,
            "final_x": float(final_info["x_position"]),
            "final_y": float(final_info["y_position"]),
            "mean_x_velocity": float(np.mean(x_velocities)) if x_velocities else 0.0,
            "mean_y_velocity": float(np.mean(y_velocities)) if y_velocities else 0.0,
            "mean_com_x_error": float(np.mean(com_x_errors)) if com_x_errors else 0.0,
            "max_abs_com_x_error": float(np.max(np.abs(com_x_errors))) if com_x_errors else 0.0,
            "mean_sagittal_correction": float(np.mean(sagittal_corrections)) if sagittal_corrections else 0.0,
            "max_abs_sagittal_correction": float(np.max(np.abs(sagittal_corrections))) if sagittal_corrections else 0.0,
            "termination_reason": env.termination_reason(final_info) if terminated else ("max_steps" if truncated else "unknown"),
        }
        row["strict_pass"] = int(pass_status(row, relaxed=False))
        row["relaxed_pass"] = int(pass_status(row, relaxed=True))
        rows.append(row)

        print(
            f"ep={ep:02d} steps={steps:04d} reward={total_reward:+.1f} "
            f"Rclr={max_right_clearance:.3f} up={min_up_z:.3f} slip={row['mean_support_slip']:.3f} "
            f"contact={row['mean_contact_accuracy']:.3f} final_x={row['final_x']:+.3f} "
            f"final_y={row['final_y']:+.3f} strict={row['strict_pass']} relaxed={row['relaxed_pass']} "
            f"reason={row['termination_reason']}"
        )

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("SUMMARY")
    for key in [
        "steps",
        "total_reward",
        "max_right_clearance",
        "min_up_z",
        "mean_contact_accuracy",
        "mean_support_slip",
        "max_backward_excess",
        "final_x",
        "final_y",
        "mean_com_x_error",
        "max_abs_com_x_error",
        "mean_sagittal_correction",
        "max_abs_sagittal_correction",
    ]:
        s = stats([float(r[key]) for r in rows])
        print(f"{key:24s} mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")

    strict_rate = mean([float(r["strict_pass"]) for r in rows])
    relaxed_rate = mean([float(r["relaxed_pass"]) for r in rows])
    print()
    print(f"STRICT PASS RATE:  {strict_rate * 100:.1f}%")
    print(f"RELAXED PASS RATE: {relaxed_rate * 100:.1f}%")
    print("CSV saved:", args.csv)

    env.close()


if __name__ == "__main__":
    main()
