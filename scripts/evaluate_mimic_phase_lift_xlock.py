from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs.g1_mimic_phase_lift_xlock_env import G1MimicPhaseLiftXLockEnv


def make_env(args):
    return G1MimicPhaseLiftXLockEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
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
        mimic_weight=args.mimic_weight,
        mimic_vel_weight=args.mimic_vel_weight,
        mimic_only_during_swing=not args.mimic_all_phase,
        reference_reverse=args.reference_reverse,
        x_soft_limit=args.x_soft_limit,
        x_hard_limit=args.x_hard_limit,
        x_velocity_soft_limit=args.x_velocity_soft_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        no_lift_terminate_target=args.no_lift_terminate_target,
        no_lift_terminate_clearance=args.no_lift_terminate_clearance,
        no_lift_penalty_weight=args.no_lift_penalty_weight,
        randomize_reset=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepMimic-lite phase lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--csv", type=str, default="")

    parser.add_argument("--action_scale", type=float, default=0.30)
    parser.add_argument("--cycle_duration", type=float, default=3.4)
    parser.add_argument("--swing_start", type=float, default=0.24)
    parser.add_argument("--swing_end", type=float, default=0.76)
    parser.add_argument("--target_clearance", type=float, default=0.015)
    parser.add_argument("--target_lateral_shift", type=float, default=0.015)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.30)
    parser.add_argument("--mimic_weight", type=float, default=0.18)
    parser.add_argument("--mimic_vel_weight", type=float, default=0.04)
    parser.add_argument("--mimic_all_phase", action="store_true")
    parser.add_argument("--reference_reverse", action="store_true")
    parser.add_argument("--x_soft_limit", type=float, default=0.10)
    parser.add_argument("--x_hard_limit", type=float, default=0.24)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.25)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.05)
    parser.add_argument("--no_lift_terminate_target", type=float, default=0.012)
    parser.add_argument("--no_lift_terminate_clearance", type=float, default=0.004)
    parser.add_argument("--no_lift_penalty_weight", type=float, default=55.0)
    args = parser.parse_args()

    env = make_env(args)
    model = PPO.load(args.model, env=env)

    print("=" * 100)
    print("XLOCK-V2 MIMIC-PHASE-LIFT EVALUATION")
    print("Model:", args.model)
    print("Reference:", env.reference_note)
    print("=" * 100)

    rows = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        reward_sum = 0.0
        max_left = 0.0
        max_right = 0.0
        min_up = 1.0
        contacts = []
        slips = []
        max_abs_com = 0.0
        max_backward = 0.0
        max_reward_mimic = 0.0
        final = info

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            reward_sum += float(reward)

            max_left = max(max_left, float(info["left_foot_clearance"]))
            max_right = max(max_right, float(info["right_foot_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            contacts.append(float(info["contact_accuracy"]))
            slips.append(float(info["support_slip"]))
            max_abs_com = max(max_abs_com, abs(float(info["com_x_error"])))
            max_backward = max(max_backward, float(info.get("backward_excess", 0.0)))
            max_reward_mimic = max(max_reward_mimic, float(info.get("reward_mimic_total", 0.0)))
            final = info

        if args.stage == "right_lift":
            main_clear = max_right
        elif args.stage == "left_lift":
            main_clear = max_left
        else:
            main_clear = min(max_left, max_right)

        reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"
        strict = (
            steps >= args.max_steps
            and main_clear >= max(0.018, 0.80 * args.target_clearance)
            and min_up >= 0.82
            and abs(float(final["x_position"])) <= 0.18
            and np.mean(contacts) >= 0.85
            and np.mean(slips) <= 0.08
        )
        relaxed = (
            steps >= 350
            and main_clear >= max(0.012, 0.55 * args.target_clearance)
            and min_up >= 0.78
            and abs(float(final["x_position"])) <= 0.28
            and np.mean(contacts) >= 0.75
        )

        row = {
            "episode": ep,
            "steps": steps,
            "reward": reward_sum,
            "max_left_clearance": max_left,
            "max_right_clearance": max_right,
            "main_clearance": main_clear,
            "min_up_z": min_up,
            "mean_contact_accuracy": float(np.mean(contacts)) if contacts else 0.0,
            "mean_support_slip": float(np.mean(slips)) if slips else 0.0,
            "max_abs_com_x_error": max_abs_com,
            "max_backward_excess": max_backward,
            "final_x": float(final["x_position"]),
            "final_y": float(final["y_position"]),
            "final_x_velocity": float(final["x_velocity"]),
            "max_reward_mimic": max_reward_mimic,
            "termination_reason": reason,
            "strict_pass": int(strict),
            "relaxed_pass": int(relaxed),
        }
        rows.append(row)

        print(
            f"ep={ep:02d} steps={steps:04d} reward={reward_sum:+.1f} "
            f"mainClr={main_clear:.3f} Lclr={max_left:.3f} Rclr={max_right:.3f} "
            f"up={min_up:.3f} contact={row['mean_contact_accuracy']:.3f} "
            f"slip={row['mean_support_slip']:.3f} x={row['final_x']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} strict={row['strict_pass']} relaxed={row['relaxed_pass']} "
            f"reason={reason}"
        )

    print("\nSUMMARY")
    for key in [
        "steps", "reward", "main_clearance", "max_left_clearance", "max_right_clearance",
        "min_up_z", "mean_contact_accuracy", "mean_support_slip", "max_abs_com_x_error",
        "max_backward_excess", "final_x", "final_x_velocity", "max_reward_mimic",
    ]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        print(f"{key:<26s} mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")

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
