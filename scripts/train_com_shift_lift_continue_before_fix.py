from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.g1_com_shift_lift_env import G1ComShiftLiftEnv


def make_env(args, stage: str, rank: int, eval_mode: bool):
    def _init():
        env = G1ComShiftLiftEnv(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            stage=stage,
            action_scale=args.action_scale,
            teacher_scale_multiplier=args.teacher_scale_multiplier,
            frame_skip=args.frame_skip,
            randomize_reset=not eval_mode,
            seed=args.seed + rank,
        )
        return Monitor(env)
    return _init


def evaluate_policy_metrics(model: PPO, args, stage: str, episodes: int = 5) -> Dict[str, float]:
    env = G1ComShiftLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        stage=stage,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        frame_skip=args.frame_skip,
        randomize_reset=False,
        seed=args.seed + 999,
    )

    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=args.seed + 3000 + ep)
        done = False
        total_reward = 0.0
        steps = 0
        max_l = 0.0
        max_r = 0.0
        min_up = 1.0
        contacts = []
        slips = []
        final_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            max_l = max(max_l, float(info.get("left_foot_clearance", 0.0)))
            max_r = max(max_r, float(info.get("right_foot_clearance", 0.0)))
            min_up = min(min_up, float(info.get("up_z", 1.0)))
            contacts.append(float(info.get("contact_accuracy", 0.0)))
            slips.append(float(info.get("support_slip", 0.0)))
            final_info = info

        rows.append({
            "steps": steps,
            "reward": total_reward,
            "max_left_clearance": max_l,
            "max_right_clearance": max_r,
            "max_clearance": max(max_l, max_r),
            "min_up_z": min_up,
            "contact_accuracy": float(np.mean(contacts)) if contacts else 0.0,
            "support_slip": float(np.mean(slips)) if slips else 0.0,
            "final_x": float(final_info.get("x_position", 0.0)),
            "final_y": float(final_info.get("y_position", 0.0)),
        })

    env.close()

    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0].keys()}


def pass_stage(stage: str, metrics: Dict[str, float]) -> Tuple[bool, str]:
    steps = metrics["steps"]
    up = metrics["min_up_z"]
    slip = metrics["support_slip"]
    max_l = metrics["max_left_clearance"]
    max_r = metrics["max_right_clearance"]
    contact = metrics["contact_accuracy"]
    final_x = abs(metrics["final_x"])

    if stage in ("shift_left", "shift_right"):
        ok = steps >= 600 and up >= 0.90 and slip <= 0.12 and contact >= 0.90 and final_x <= 0.25
        return ok, f"need steps>=600 up>=0.90 slip<=0.12 contact>=0.90 |final_x|<=0.25; got steps={steps:.0f}, up={up:.3f}, slip={slip:.3f}, contact={contact:.3f}, final_x={metrics['final_x']:.3f}"

    if stage == "right_lift":
        ok = steps >= 450 and up >= 0.84 and max_r >= 0.025 and slip <= 0.24 and final_x <= 0.35
        return ok, f"need steps>=450 up>=0.84 rightClr>=0.025 slip<=0.24 |final_x|<=0.35; got steps={steps:.0f}, up={up:.3f}, rightClr={max_r:.3f}, slip={slip:.3f}, final_x={metrics['final_x']:.3f}"

    if stage == "left_lift":
        ok = steps >= 450 and up >= 0.84 and max_l >= 0.022 and slip <= 0.24 and final_x <= 0.35
        return ok, f"need steps>=450 up>=0.84 leftClr>=0.022 slip<=0.24 |final_x|<=0.35; got steps={steps:.0f}, up={up:.3f}, leftClr={max_l:.3f}, slip={slip:.3f}, final_x={metrics['final_x']:.3f}"

    if stage == "alt_lift":
        ok = steps >= 500 and up >= 0.82 and max_l >= 0.018 and max_r >= 0.018 and slip <= 0.30
        return ok, f"need steps>=500 up>=0.82 bothClr>=0.018 slip<=0.30; got steps={steps:.0f}, up={up:.3f}, L/R={max_l:.3f}/{max_r:.3f}, slip={slip:.3f}"

    if stage == "tiny_walk":
        ok = steps >= 450 and up >= 0.80 and max_l >= 0.016 and max_r >= 0.016 and slip <= 0.36
        return ok, f"need steps>=450 up>=0.80 bothClr>=0.016 slip<=0.36; got steps={steps:.0f}, up={up:.3f}, L/R={max_l:.3f}/{max_r:.3f}, slip={slip:.3f}"

    ok = steps >= 600 and up >= 0.90 and slip <= 0.15
    return ok, f"default check; got steps={steps:.0f}, up={up:.3f}, slip={slip:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Continue one curriculum stage from an existing model.")
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=40000)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="models/g1_com_shift_lift_continue")
    parser.add_argument("--log_dir", type=str, default="logs/g1_com_shift_lift_continue")
    parser.add_argument("--metrics_csv", type=str, default="results/g1_com_shift_lift_continue_metrics.csv")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--action_scale", type=float, default=0.22)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.08)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    args = parser.parse_args()

    if not Path(args.init_model).exists():
        raise FileNotFoundError(f"Initial model not found: {args.init_model}")

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    metrics_csv = Path(args.metrics_csv)
    stage_dir = output_dir / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    current_model_path = args.init_model
    rows = []

    print("=" * 100)
    print("CONTINUE CURRICULUM STAGE")
    print("Initial model:", args.init_model)
    print("Stage:", args.stage)
    print("Output:", output_dir)
    print("=" * 100)

    for attempt in range(1, args.attempts + 1):
        print()
        print("-" * 100)
        print(f"Attempt {attempt}/{args.attempts}: stage={args.stage}, timesteps={args.timesteps}")
        print("-" * 100)

        env = DummyVecEnv([make_env(args, args.stage, rank=0, eval_mode=False)])
        env = VecMonitor(env)
        eval_env = DummyVecEnv([make_env(args, args.stage, rank=500, eval_mode=True)])
        eval_env = VecMonitor(eval_env)

        model = PPO.load(current_model_path, env=env, device=args.device)
        model.learning_rate = args.learning_rate
        model.clip_range = args.clip_range
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
        model.tensorboard_log = str(log_dir)

        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(stage_dir / "checkpoints"),
            name_prefix=f"continue_{args.stage}_attempt{attempt}",
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(stage_dir / "best_model"),
            log_path=str(stage_dir / "eval_logs"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=f"continue_{args.stage}",
        )

        current_model_path = str(stage_dir / f"final_{args.stage}_attempt_{attempt:02d}.zip")
        model.save(current_model_path)
        model.save(str(output_dir / "latest_model.zip"))

        metrics = evaluate_policy_metrics(model, args, args.stage, episodes=args.n_eval_episodes)
        passed, reason = pass_stage(args.stage, metrics)

        print("Stage metrics:", metrics)
        print("Pass check:", passed, reason)

        rows.append({
            "stage": args.stage,
            "attempt": attempt,
            "passed": int(passed),
            **metrics,
            "reason": reason,
            "model_path": current_model_path,
        })
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        env.close()
        eval_env.close()

        if passed:
            passed_model = output_dir / f"passed_{args.stage}.zip"
            model.save(str(passed_model))
            print("=" * 100)
            print(f"PASSED stage {args.stage}")
            print("Saved:", passed_model)
            print("Metrics CSV:", metrics_csv)
            print("=" * 100)
            return

    print("=" * 100)
    print(f"STAGE STILL FAILED: {args.stage}")
    print("Latest model:", current_model_path)
    print("Metrics CSV:", metrics_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
