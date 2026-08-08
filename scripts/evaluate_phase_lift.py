from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs.g1_phase_lift_env import G1PhaseLiftEnv


def make_env(args):
    return G1PhaseLiftEnv(
        model_path=args.model_path,
        stage=args.stage,
        action_scale=args.action_scale,
        cycle_duration=args.cycle_duration,
        swing_start=args.swing_start,
        swing_end=args.swing_end,
        target_clearance=args.target_clearance,
        target_lateral_shift=args.target_lateral_shift,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        action_target_smoothing=args.action_target_smoothing,
        randomize_reset=False,
    )


def summarize(rows, key):
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return vals.mean(), vals.std(), vals.min(), vals.max()


def main():
    parser = argparse.ArgumentParser(description="Evaluate phase-based G1 lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="")

    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--cycle_duration", type=float, default=3.0)
    parser.add_argument("--swing_start", type=float, default=0.35)
    parser.add_argument("--swing_end", type=float, default=0.70)
    parser.add_argument("--target_clearance", type=float, default=0.025)
    parser.add_argument("--target_lateral_shift", type=float, default=0.025)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    args = parser.parse_args()

    env = make_env(args)
    model = PPO.load(args.model, env=env)

    rows = []
    print("=" * 100)
    print("PHASE-LIFT EVALUATION")
    print("Model:", args.model)
    print("Stage:", args.stage)
    print("No pose teacher; anti-standing + x-guard phase/contact/clearance reward v3.")
    print("=" * 100)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_left_clearance = 0.0
        max_right_clearance = 0.0
        min_up_z = 1.0
        contact_acc = []
        support_slip = []
        max_abs_com_error = 0.0
        max_backward_excess = 0.0
        final_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
            max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
            min_up_z = min(min_up_z, float(info["up_z"]))
            contact_acc.append(float(info["contact_accuracy"]))
            support_slip.append(float(info["support_slip"]))
            max_abs_com_error = max(max_abs_com_error, abs(float(info["com_x_error"])))
            max_backward_excess = max(max_backward_excess, float(info.get("backward_excess", 0.0)))
            final_info = info

        reason = env.termination_reason(final_info) if steps < args.max_steps else "max_steps"

        if args.stage == "right_lift":
            main_clearance = max_right_clearance
            clearance_ok_strict = main_clearance >= max(0.018, 0.80 * args.target_clearance)
            clearance_ok_relaxed = main_clearance >= max(0.012, 0.55 * args.target_clearance)
        elif args.stage == "left_lift":
            main_clearance = max_left_clearance
            clearance_ok_strict = main_clearance >= max(0.018, 0.80 * args.target_clearance)
            clearance_ok_relaxed = main_clearance >= max(0.012, 0.55 * args.target_clearance)
        else:
            main_clearance = min(max_left_clearance, max_right_clearance)
            clearance_ok_strict = main_clearance >= max(0.015, 0.65 * args.target_clearance)
            clearance_ok_relaxed = main_clearance >= max(0.010, 0.45 * args.target_clearance)

        strict_pass = (
            steps >= args.max_steps
            and min_up_z >= 0.82
            and max_backward_excess <= 0.22
            and abs(float(final_info["x_position"])) <= 0.18
            and np.mean(contact_acc) >= 0.85
            and np.mean(support_slip) <= 0.08
            and clearance_ok_strict
        )
        relaxed_pass = (
            steps >= 350
            and min_up_z >= 0.78
            and max_backward_excess <= 0.32
            and abs(float(final_info["x_position"])) <= 0.28
            and np.mean(contact_acc) >= 0.75
            and clearance_ok_relaxed
        )

        row = {
            "episode": ep,
            "steps": steps,
            "total_reward": total_reward,
            "max_left_clearance": max_left_clearance,
            "max_right_clearance": max_right_clearance,
            "min_up_z": min_up_z,
            "mean_contact_accuracy": float(np.mean(contact_acc)) if contact_acc else 0.0,
            "mean_support_slip": float(np.mean(support_slip)) if support_slip else 0.0,
            "max_abs_com_x_error": max_abs_com_error,
            "max_backward_excess": max_backward_excess,
            "final_x": float(final_info["x_position"]),
            "final_y": float(final_info["y_position"]),
            "final_x_velocity": float(final_info["x_velocity"]),
            "final_phase": float(final_info["phase"]),
            "termination_reason": reason,
            "strict_pass": int(strict_pass),
            "relaxed_pass": int(relaxed_pass),
        }
        rows.append(row)

        print(
            f"ep={ep:02d} steps={steps:04d} reward={total_reward:+.1f} "
            f"Lclr={max_left_clearance:.3f} Rclr={max_right_clearance:.3f} "
            f"up={min_up_z:.3f} contact={row['mean_contact_accuracy']:.3f} "
            f"slip={row['mean_support_slip']:.3f} final_x={row['final_x']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} strict={row['strict_pass']} relaxed={row['relaxed_pass']} "
            f"reason={reason}"
        )

    print("\nSUMMARY")
    for key in [
        "steps",
        "total_reward",
        "max_left_clearance",
        "max_right_clearance",
        "min_up_z",
        "mean_contact_accuracy",
        "mean_support_slip",
        "max_abs_com_x_error",
        "max_backward_excess",
        "final_x",
        "final_y",
        "final_x_velocity",
    ]:
        mean, std, min_v, max_v = summarize(rows, key)
        print(f"{key:<26s} mean={mean:.4f} std={std:.4f} min={min_v:.4f} max={max_v:.4f}")

    strict_rate = 100.0 * np.mean([r["strict_pass"] for r in rows])
    relaxed_rate = 100.0 * np.mean([r["relaxed_pass"] for r in rows])
    print(f"\nSTRICT PASS RATE:  {strict_rate:.1f}%")
    print(f"RELAXED PASS RATE: {relaxed_rate:.1f}%")

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
