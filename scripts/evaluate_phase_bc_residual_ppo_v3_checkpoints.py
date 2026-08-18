from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO

from envs.g1_phase_bc_residual_env_v3 import G1PhaseBCResidualEnvV3


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match:
        return int(match.group(1))
    return -1


def evaluate_model(model_path: Path, args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    env = G1PhaseBCResidualEnvV3(
        render_mode=None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
        fall_height=args.fall_height,
        fall_up_z=args.fall_up_z,
        forward_velocity_weight=args.forward_velocity_weight,
        velocity_tracking_weight=args.velocity_tracking_weight,
        displacement_reward_weight=args.displacement_reward_weight,
        progress_deficit_penalty_weight=args.progress_deficit_penalty_weight,
        backward_velocity_penalty_weight=args.backward_velocity_penalty_weight,
        backward_position_penalty_weight=args.backward_position_penalty_weight,
        standstill_penalty_weight=args.standstill_penalty_weight,
        fall_warning_weight=args.fall_warning_weight,
        min_progress_rate=args.min_progress_rate,
        progress_grace_time=args.progress_grace_time,
        standstill_velocity_threshold=args.standstill_velocity_threshold,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()

    total_reward = 0.0
    final_info = info

    max_x = float(info["base_x"])
    min_z = float(info["base_z"])
    min_up_z = float(info["up_z"])

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        max_x = max(max_x, float(info["base_x"]))
        min_z = min(min_z, float(info["base_z"]))
        min_up_z = min(min_up_z, float(info["up_z"]))

        if terminated or truncated:
            break

    env.close()

    return {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "checkpoint_step": checkpoint_step(model_path),
        "steps_completed": int(final_info["step"]),
        "final_x": float(final_info["base_x"]),
        "final_y": float(final_info["base_y"]),
        "final_z": float(final_info["base_z"]),
        "final_up_z": float(final_info["up_z"]),
        "max_x": float(max_x),
        "min_z": float(min_z),
        "min_up_z": float(min_up_z),
        "total_reward": float(total_reward),
        "terminated": bool(final_info["terminated"]),
        "truncated": bool(final_info["truncated"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        default="experiments/amass_b3_15dof_residual_ppo_v3/models/checkpoints",
    )
    parser.add_argument(
        "--extra_model",
        action="append",
        default=[
            "experiments/amass_b3_15dof_residual_ppo_v2/models/phase_bc_residual_ppo_v2_final.zip",
            "experiments/amass_b3_15dof_residual_ppo_v3/models/phase_bc_residual_ppo_v3_final.zip",
        ],
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3/reports/v3_checkpoint_eval.csv",
    )

    parser.add_argument("--steps", type=int, default=600)

    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--gait_scale", type=float, default=0.18)
    parser.add_argument("--residual_scale", type=float, default=0.14)
    parser.add_argument("--target_smoothing", type=float, default=0.07)
    parser.add_argument("--transition_steps", type=int, default=240)
    parser.add_argument("--target_velocity", type=float, default=0.035)

    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_up_z", type=float, default=0.50)

    parser.add_argument("--forward_velocity_weight", type=float, default=1.20)
    parser.add_argument("--velocity_tracking_weight", type=float, default=0.70)
    parser.add_argument("--displacement_reward_weight", type=float, default=1.00)
    parser.add_argument("--progress_deficit_penalty_weight", type=float, default=2.00)
    parser.add_argument("--backward_velocity_penalty_weight", type=float, default=3.00)
    parser.add_argument("--backward_position_penalty_weight", type=float, default=2.00)
    parser.add_argument("--standstill_penalty_weight", type=float, default=1.20)
    parser.add_argument("--fall_warning_weight", type=float, default=5.00)
    parser.add_argument("--min_progress_rate", type=float, default=0.020)
    parser.add_argument("--progress_grace_time", type=float, default=4.00)
    parser.add_argument("--standstill_velocity_threshold", type=float, default=0.025)

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model_paths: list[Path] = []

    if checkpoint_dir.exists():
        model_paths.extend(sorted(checkpoint_dir.glob("*.zip"), key=checkpoint_step))

    for extra in args.extra_model:
        extra_path = Path(extra)
        if extra_path.exists():
            model_paths.append(extra_path)

    if not model_paths:
        raise FileNotFoundError("No model checkpoints found.")

    print("=" * 100)
    print("EVALUATING PPO V3 CHECKPOINTS")
    print("=" * 100)
    print("Models found:", len(model_paths))
    print("Output CSV:", out_csv)
    print("Steps:", args.steps)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target velocity:", args.target_velocity)
    print("=" * 100)

    rows = []

    for model_path in model_paths:
        print("Evaluating:", model_path)
        result = evaluate_model(model_path, args)
        rows.append(result)

        print(
            f"  steps={result['steps_completed']} "
            f"final_x={result['final_x']:+.3f} "
            f"max_x={result['max_x']:+.3f} "
            f"final_z={result['final_z']:+.3f} "
            f"up_z={result['final_up_z']:+.3f} "
            f"terminated={result['terminated']} "
            f"reward={result['total_reward']:+.1f}"
        )

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 100)
    print("SORTED RESULTS")
    print("=" * 100)

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            int(r["steps_completed"]),
            float(r["max_x"]),
            float(r["final_x"]),
            float(r["total_reward"]),
        ),
        reverse=True,
    )

    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:02d}. {row['model_name']} | "
            f"steps={row['steps_completed']} | "
            f"final_x={row['final_x']:+.3f} | "
            f"max_x={row['max_x']:+.3f} | "
            f"final_z={row['final_z']:+.3f} | "
            f"up_z={row['final_up_z']:+.3f} | "
            f"terminated={row['terminated']} | "
            f"reward={row['total_reward']:+.1f}"
        )

    print("=" * 100)
    print("CSV saved:", out_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
