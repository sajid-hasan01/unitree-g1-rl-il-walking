from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.g1_com_shift_lift_env import G1ComShiftLiftEnv


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def split_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def evaluate_policy_metrics(model: PPO, args, stage: str, episodes: int = 3) -> Dict[str, float]:
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
        obs, info = env.reset(seed=args.seed + 2000 + ep)
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

    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def pass_stage(stage: str, metrics: Dict[str, float]) -> Tuple[bool, str]:
    steps = metrics["steps"]
    up = metrics["min_up_z"]
    slip = metrics["support_slip"]
    max_l = metrics["max_left_clearance"]
    max_r = metrics["max_right_clearance"]
    contact = metrics["contact_accuracy"]

    if stage == "balance":
        ok = steps >= 650 and up >= 0.97 and slip <= 0.06
        return ok, f"need steps>=650 up>=0.97 slip<=0.06; got steps={steps:.0f}, up={up:.3f}, slip={slip:.3f}"

    if stage in ("shift_left", "shift_right"):
        ok = steps >= 600 and up >= 0.93 and slip <= 0.12 and contact >= 0.90
        return ok, f"need steps>=600 up>=0.93 slip<=0.12 contact>=0.90; got steps={steps:.0f}, up={up:.3f}, slip={slip:.3f}, contact={contact:.3f}"

    if stage == "right_lift":
        ok = steps >= 450 and up >= 0.86 and max_r >= 0.025 and slip <= 0.22
        return ok, f"need steps>=450 up>=0.86 rightClr>=0.025 slip<=0.22; got steps={steps:.0f}, up={up:.3f}, rightClr={max_r:.3f}, slip={slip:.3f}"

    if stage == "left_lift":
        ok = steps >= 450 and up >= 0.86 and max_l >= 0.025 and slip <= 0.22
        return ok, f"need steps>=450 up>=0.86 leftClr>=0.025 slip<=0.22; got steps={steps:.0f}, up={up:.3f}, leftClr={max_l:.3f}, slip={slip:.3f}"

    if stage == "alt_lift":
        ok = steps >= 500 and up >= 0.84 and max_l >= 0.020 and max_r >= 0.020 and slip <= 0.28
        return ok, f"need steps>=500 up>=0.84 bothClr>=0.020 slip<=0.28; got steps={steps:.0f}, up={up:.3f}, L/R={max_l:.3f}/{max_r:.3f}, slip={slip:.3f}"

    if stage == "tiny_walk":
        ok = steps >= 450 and up >= 0.82 and max_l >= 0.018 and max_r >= 0.018 and slip <= 0.35
        return ok, f"need steps>=450 up>=0.82 bothClr>=0.018 slip<=0.35; got steps={steps:.0f}, up={up:.3f}, L/R={max_l:.3f}/{max_r:.3f}, slip={slip:.3f}"

    return False, f"unknown stage {stage}"


def main():
    parser = argparse.ArgumentParser(description="Success-gated COM-shift + foot-lift curriculum training.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--stages", type=str, default="balance,shift_left,shift_right,right_lift,left_lift,alt_lift,tiny_walk")
    parser.add_argument("--stage_steps", type=str, default="15000,20000,20000,35000,35000,50000,60000")
    parser.add_argument("--attempts_per_stage", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="models/g1_com_shift_lift")
    parser.add_argument("--log_dir", type=str, default="logs/g1_com_shift_lift")
    parser.add_argument("--metrics_csv", type=str, default="results/g1_com_shift_lift_stage_metrics.csv")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--action_scale", type=float, default=0.28)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--frame_skip", type=int, default=5)

    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.10)
    parser.add_argument("--ent_coef", type=float, default=0.0015)
    parser.add_argument("--target_kl", type=float, default=0.025)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=3)

    args = parser.parse_args()

    stages = split_csv(args.stages)
    stage_steps = split_int_csv(args.stage_steps)
    if len(stages) != len(stage_steps):
        raise ValueError("--stages and --stage_steps must have same length")

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    metrics_csv = Path(args.metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    model = None
    previous_model_path = None

    print("=" * 100)
    print("COM-SHIFT + FOOT-LIFT SUCCESS-GATED CURRICULUM")
    print("Stages:", stages)
    print("Stage steps:", stage_steps)
    print("Attempts/stage:", args.attempts_per_stage)
    print("Output:", output_dir)
    print("=" * 100)

    for stage_idx, (stage, steps_per_attempt) in enumerate(zip(stages, stage_steps), start=1):
        stage_dir = output_dir / f"stage_{stage_idx:02d}_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_passed = False

        for attempt in range(1, args.attempts_per_stage + 1):
            run_name = f"stage_{stage_idx:02d}_{stage}_attempt_{attempt:02d}"
            print()
            print("-" * 100)
            print(f"Stage {stage_idx}/{len(stages)}: {stage}, attempt {attempt}/{args.attempts_per_stage}, steps={steps_per_attempt}")
            print("-" * 100)

            env = DummyVecEnv([make_env(args, stage=stage, rank=i, eval_mode=False) for i in range(args.n_envs)])
            env = VecMonitor(env)
            eval_env = DummyVecEnv([make_env(args, stage=stage, rank=500 + i, eval_mode=True) for i in range(1)])
            eval_env = VecMonitor(eval_env)

            if model is None:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                    target_kl=args.target_kl,
                    verbose=1,
                    tensorboard_log=str(log_dir),
                    seed=args.seed,
                    device=args.device,
                )
            else:
                model = PPO.load(previous_model_path, env=env, device=args.device)
                model.tensorboard_log = str(log_dir)

            checkpoint_callback = CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
                save_path=str(stage_dir / "checkpoints"),
                name_prefix=f"g1_com_{stage}_attempt{attempt}",
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(stage_dir / "best_model"),
                log_path=str(stage_dir / "eval_logs"),
                eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            )

            model.learn(
                total_timesteps=steps_per_attempt,
                reset_num_timesteps=False,
                callback=[checkpoint_callback, eval_callback],
                tb_log_name=run_name,
            )

            attempt_model_path = str(stage_dir / f"final_{stage}_attempt_{attempt:02d}.zip")
            model.save(attempt_model_path)
            model.save(str(output_dir / "latest_model.zip"))
            previous_model_path = attempt_model_path

            metrics = evaluate_policy_metrics(model, args, stage=stage, episodes=args.n_eval_episodes)
            passed, reason = pass_stage(stage, metrics)
            print("Stage metrics:", metrics)
            print("Pass check:", passed, reason)

            row = {
                "stage_idx": stage_idx,
                "stage": stage,
                "attempt": attempt,
                "passed": int(passed),
                **metrics,
                "reason": reason,
                "model_path": attempt_model_path,
            }
            metrics_rows.append(row)

            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metrics_rows)

            env.close()
            eval_env.close()

            if passed:
                stable_stage_path = str(stage_dir / f"passed_{stage}.zip")
                model.save(stable_stage_path)
                model.save(str(output_dir / f"passed_{stage}.zip"))
                previous_model_path = stable_stage_path
                stage_passed = True
                print(f"PASSED stage {stage}. Saved: {stable_stage_path}")
                break

        if not stage_passed:
            print()
            print("=" * 100)
            print(f"STAGE FAILED: {stage}")
            print("Stopping curriculum. Do not train later stages on an unstable primitive.")
            print(f"Metrics CSV: {metrics_csv}")
            print(f"Latest model: {previous_model_path}")
            print("=" * 100)
            return

    final_path = output_dir / "g1_com_shift_lift_final.zip"
    model.save(str(final_path))
    print("=" * 100)
    print("ALL STAGES PASSED")
    print("Final model:", final_path)
    print("Metrics CSV:", metrics_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
