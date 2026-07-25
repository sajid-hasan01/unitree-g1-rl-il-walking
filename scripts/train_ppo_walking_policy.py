import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


def make_env(args, monitor_log):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        random_start=args.random_start,
        enable_push=args.enable_push,
        push_window_start=args.push_window_start,
        push_window_end=args.push_window_end,
        push_interval_min=args.push_interval_min,
        push_interval_max=args.push_interval_max,
        push_force_min=args.push_force_min,
        push_force_max=args.push_force_max,
        push_duration_steps=args.push_duration_steps,
    )

    env = Monitor(env, filename=monitor_log)
    return env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(
            "datasets",
            "processed",
            "g1_amass_walking_il_15dof.npz",
        ),
        help="Processed AMASS IL dataset path.",
    )

    parser.add_argument(
        "--reference_mode",
        type=str,
        default="transition",
        choices=["transition", "cyclic"],
        help="Reference motion mode: transition for stand-to-walk, cyclic for looping walking reference.",
    )

    parser.add_argument("--target_velocity", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.15)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=250)

    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start each episode at a random reference frame.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("models", "g1_ppo_walking_policy.zip"),
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join("models", "ppo_walking_checkpoints"),
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join("logs", "ppo_walking"),
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--check_env", action="store_true")

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=25_000,
        help="Timesteps between evaluations of the current policy.",
    )

    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of episodes averaged per evaluation.",
    )

    parser.add_argument(
        "--enable_push",
        action="store_true",
        help="Enable randomized external pushes at the pelvis.",
    )

    parser.add_argument(
        "--push_window_start",
        type=int,
        default=None,
        help="Episode step after which pushes may begin. Defaults to initial_stand_steps inside the env.",
    )

    parser.add_argument("--push_window_end", type=int, default=600)
    parser.add_argument("--push_interval_min", type=int, default=100)
    parser.add_argument("--push_interval_max", type=int, default=200)
    parser.add_argument("--push_force_min", type=float, default=20.0)
    parser.add_argument("--push_force_max", type=float, default=60.0)
    parser.add_argument("--push_duration_steps", type=int, default=5)

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    monitor_log = os.path.join(args.log_dir, "monitor.csv")
    eval_monitor_log = os.path.join(args.log_dir, "eval_monitor.csv")
    best_model_dir = os.path.join(args.checkpoint_dir, "best_model")
    eval_log_dir = os.path.join(args.log_dir, "eval")

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    env = make_env(args, monitor_log)
    eval_env = make_env(args, eval_monitor_log)

    if args.check_env:
        print("Checking Gymnasium environment...")
        check_env(env.unwrapped, warn=True)
        print("Environment check complete.")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=args.checkpoint_dir,
        name_prefix="g1_ppo_walking",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume model not found: {args.resume}")

        print("Loading existing PPO model:")
        print(args.resume)

        model = PPO.load(
            args.resume,
            env=env,
            device="auto",
        )

    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=args.log_dir,
            device="auto",
        )

    print()
    print("Starting PPO walking training")
    print("Dataset path:", args.dataset_path)
    print("Reference mode:", args.reference_mode)
    print("Total timesteps:", args.total_timesteps)
    print("Target velocity:", args.target_velocity)
    print("Action scale:", args.action_scale)
    print("Reference speed:", args.reference_speed)
    print("Initial stand steps:", args.initial_stand_steps)
    print("Transition steps:", args.transition_steps)
    print("Random start:", args.random_start)
    print("Eval freq:", args.eval_freq)
    print("Best model save path:", best_model_dir)
    print("Push enabled:", args.enable_push)

    if args.enable_push:
        print("Push window:", args.push_window_start, "to", args.push_window_end)
        print("Push force range:", args.push_force_min, "to", args.push_force_max)
        print("Push interval range:", args.push_interval_min, "to", args.push_interval_max)
        print("Push duration steps:", args.push_duration_steps)

    print("Output:", args.output)
    print()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name="g1_ppo_walking",
        progress_bar=False,
    )

    model.save(args.output)

    env.close()
    eval_env.close()

    print()
    print("PPO walking training complete.")
    print("Saved final model:", args.output)
    print(
        "Best model saved at:",
        os.path.join(best_model_dir, "best_model.zip"),
    )


if __name__ == "__main__":
    main()