from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.g1_solution_env import G1SolutionEnv


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_env(args, stage: str, rank: int = 0, eval_mode: bool = False):
    def _init():
        env = G1SolutionEnv(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            stage=stage,
            action_scale=args.action_scale,
            frame_skip=args.frame_skip,
            randomize_reset=not eval_mode,
            seed=args.seed + rank,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train curriculum solution policy for Unitree G1.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--stages", type=str, default="balance,right_lift,left_lift,alt_lift,step_in_place,tiny_walk")
    parser.add_argument("--timesteps", type=str, default="10000,15000,15000,25000,35000,50000")
    parser.add_argument("--output_dir", type=str, default="models/g1_solution_curriculum")
    parser.add_argument("--log_dir", type=str, default="logs/g1_solution_curriculum")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_scale", type=float, default=0.55)
    parser.add_argument("--frame_skip", type=int, default=5)

    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.12)
    parser.add_argument("--ent_coef", type=float, default=0.002)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=3)

    args = parser.parse_args()

    stages = parse_csv(args.stages)
    timesteps = parse_int_csv(args.timesteps)
    if len(stages) != len(timesteps):
        raise ValueError("--stages and --timesteps must have the same length")

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = None
    previous_model_path = None

    print("=" * 90)
    print("G1 SOLUTION CURRICULUM TRAINING")
    print("Stages:", stages)
    print("Timesteps:", timesteps)
    print("Output:", output_dir)
    print("=" * 90)

    for stage_idx, (stage, steps) in enumerate(zip(stages, timesteps), start=1):
        stage_dir = output_dir / f"stage_{stage_idx:02d}_{stage}"
        stage_log = log_dir / f"stage_{stage_idx:02d}_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_log.mkdir(parents=True, exist_ok=True)

        env = DummyVecEnv([make_env(args, stage, rank=i, eval_mode=False) for i in range(args.n_envs)])
        env = VecMonitor(env)
        eval_env = DummyVecEnv([make_env(args, stage, rank=1000, eval_mode=True)])
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
            name_prefix=f"g1_solution_{stage}",
            save_replay_buffer=False,
            save_vecnormalize=False,
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

        print()
        print("-" * 90)
        print(f"Training stage {stage_idx}/{len(stages)}: {stage} for {steps} steps")
        print("-" * 90)

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=f"stage_{stage_idx:02d}_{stage}",
        )

        previous_model_path = str(stage_dir / f"final_{stage}.zip")
        model.save(previous_model_path)
        model.save(str(output_dir / "latest_model.zip"))
        print(f"Saved stage model: {previous_model_path}")

        env.close()
        eval_env.close()

    final_path = output_dir / "g1_solution_final.zip"
    model.save(str(final_path))

    print("=" * 90)
    print("Training complete")
    print(f"Final model: {final_path}")
    print(f"Latest model: {output_dir / 'latest_model.zip'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
