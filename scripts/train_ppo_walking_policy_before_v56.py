import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


def make_env(args, monitor_log, use_rsi=False):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
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
        include_contact_phase_observation=args.include_contact_phase_observation,
        use_reference_contact_mask=args.use_reference_contact_mask,
        reference_start_frame=args.reference_start_frame,
        initial_yaw_degrees=args.initial_yaw_degrees,
        reference_state_initialization=use_rsi,
        rsi_start_frame=args.rsi_start_frame,
        rsi_end_frame=args.rsi_end_frame,
    )

    env = Monitor(env, filename=monitor_log)
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO walking policy for Unitree G1 in MuJoCo."
    )

    # PPO hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.001,
        help=(
            "PPO entropy coefficient. Higher values increase exploration. "
            "This value is also explicitly overridden when resuming from a checkpoint."
        ),
    )

    # Environment parameters
    parser.add_argument("--target_velocity", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument(
        "--action_target_smoothing",
        type=float,
        default=0.55,
        help=(
            "v42 low-pass filter for residual joint targets. "
            "0.0 disables smoothing; 0.55 means 55% previous target and 45% requested target."
        ),
    )
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.15)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=250)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Path to the IL/reference dataset .npz file. "
            "If not provided, the environment default dataset is used."
        ),
    )

    parser.add_argument(
        "--reference_mode",
        type=str,
        default="transition",
        choices=["transition", "cyclic"],
        help="Reference playback mode.",
    )

    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start the reference motion at a random frame instead of frame 0.",
    )

    parser.add_argument(
        "--include_contact_phase_observation",
        action="store_true",
        help=(
            "Add expected/actual foot contact phase features to the observation. "
            "This changes observation shape from 59 to 65 and requires training "
            "a new model from scratch. Do not use this with old 59-observation checkpoints."
        ),
    )

    parser.add_argument(
        "--use_reference_contact_mask",
        action="store_true",
        help=(
            "Use dataset contact_mask as expected contact labels in reward/observation. "
            "For v51 this is disabled by default because original OpenHE contact labels "
            "did not match the actual MuJoCo G1 collision contacts."
        ),
    )

    parser.add_argument(
        "--reference_start_frame",
        type=int,
        default=0,
        help=(
            "Local reference phase offset. v55 uses 25 to skip the unstable sticky "
            "beginning of the selected OpenHE segment."
        ),
    )

    parser.add_argument(
        "--initial_yaw_degrees",
        type=float,
        default=0.0,
        help=(
            "Initial root yaw in degrees. v51 default is 0.0 because zero-residual "
            "tests showed the selected OpenHE reference naturally drives negative-X "
            "motion with yaw 0."
        ),
    )

    parser.add_argument(
        "--reference_state_initialization",
        action="store_true",
        help=(
            "Enable v39 Reference State Initialization for the training environment. "
            "Training episodes start from random walking-reference frames. "
            "Evaluation still starts from normal standing pose."
        ),
    )

    parser.add_argument(
        "--rsi_start_frame",
        type=int,
        default=5,
        help="First reference frame allowed for RSI reset sampling.",
    )

    parser.add_argument(
        "--rsi_end_frame",
        type=int,
        default=None,
        help=(
            "Last reference frame allowed for RSI reset sampling. "
            "If omitted, the final dataset frame is used."
        ),
    )

    # Output paths
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("models", "g1_ppo_walking_policy.zip"),
        help="Output path for the final PPO model.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join("models", "ppo_walking_checkpoints"),
        help="Directory for periodic checkpoints and best model.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join("logs", "ppo_walking"),
        help="Directory for Monitor logs and TensorBoard logs.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to existing PPO .zip checkpoint to resume from.",
    )

    parser.add_argument(
        "--check_env",
        action="store_true",
        help="Run Stable-Baselines3 environment checker before training.",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=25_000,
        help="Timesteps between evaluations for best-model tracking.",
    )

    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=25_000,
        help="Timesteps between periodic checkpoint saves.",
    )

    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes used by EvalCallback.",
    )

    # Push-disturbance options
    parser.add_argument(
        "--enable_push",
        action="store_true",
        help="Enable randomized external pushes at the pelvis during training.",
    )

    parser.add_argument(
        "--push_window_start",
        type=int,
        default=None,
        help="Episode step after which pushes may begin. Defaults to initial_stand_steps.",
    )

    parser.add_argument("--push_window_end", type=int, default=600)
    parser.add_argument("--push_interval_min", type=int, default=100)
    parser.add_argument("--push_interval_max", type=int, default=200)
    parser.add_argument("--push_force_min", type=float, default=20.0)
    parser.add_argument("--push_force_max", type=float, default=60.0)
    parser.add_argument("--push_duration_steps", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    monitor_log = os.path.join(args.log_dir, "monitor.csv")
    eval_monitor_log = os.path.join(args.log_dir, "eval_monitor.csv")
    best_model_dir = os.path.join(args.checkpoint_dir, "best_model")
    eval_log_dir = os.path.join(args.log_dir, "eval")

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # RSI is training-only. EvalCallback must evaluate the normal standing-start task,
    # otherwise best_model would be selected for random mid-gait resets instead of
    # the actual showcase start condition.
    env = make_env(
        args,
        monitor_log,
        use_rsi=args.reference_state_initialization,
    )
    eval_env = make_env(
        args,
        eval_monitor_log,
        use_rsi=False,
    )

    if args.check_env:
        print("Checking Gymnasium environment...")
        check_env(env.unwrapped, warn=True)
        print("Environment check complete.")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
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

        if args.include_contact_phase_observation:
            print()
            print("WARNING:")
            print(
                "You enabled --include_contact_phase_observation while also using --resume."
            )
            print(
                "This is only valid if the resumed model was originally trained with the same 65-dimensional observation."
            )
            print(
                "Old v10/v16/v20 models used 59-dimensional observations and should NOT be resumed with this flag."
            )
            print()

        print("Loading existing PPO model:")
        print(args.resume)

        model = PPO.load(
            args.resume,
            env=env,
            device="auto",
        )

        previous_ent_coef = model.ent_coef
        model.ent_coef = args.ent_coef
        print(f"Overriding ent_coef: {previous_ent_coef} -> {args.ent_coef}")

        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = args.learning_rate

        model.clip_range = get_schedule_fn(args.clip_range)
        model.vf_coef = args.vf_coef
        model.max_grad_norm = args.max_grad_norm
        model.target_kl = args.target_kl

        print("Overriding learning_rate:", args.learning_rate)
        print("Overriding clip_range:", args.clip_range)
        print("Overriding vf_coef:", args.vf_coef)
        print("Overriding max_grad_norm:", args.max_grad_norm)
        print("Overriding target_kl:", args.target_kl)

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
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            verbose=1,
            tensorboard_log=args.log_dir,
            device="auto",
        )

    print()
    print("=" * 80)
    print("Starting PPO walking training")
    print("=" * 80)
    print("Total timesteps:", args.total_timesteps)
    print("Dataset path:", args.dataset_path)
    print("Reference mode:", args.reference_mode)
    print("Target velocity:", args.target_velocity)
    print("Action scale:", args.action_scale)
    print("Action target smoothing:", args.action_target_smoothing)
    print("Reference speed:", args.reference_speed)
    print("Initial stand steps:", args.initial_stand_steps)
    print("Transition steps:", args.transition_steps)
    print("Include contact phase observation:", args.include_contact_phase_observation)
    print("Use reference contact mask:", args.use_reference_contact_mask)
    print("Reference start frame:", args.reference_start_frame)
    print("Initial yaw degrees:", args.initial_yaw_degrees)
    print("Reference State Initialization:", args.reference_state_initialization)
    print("RSI frame range:", args.rsi_start_frame, "to", args.rsi_end_frame)
    print("Ent coef:", args.ent_coef)
    print("Learning rate:", args.learning_rate)
    print("Clip range:", args.clip_range)
    print("VF coef:", args.vf_coef)
    print("Max grad norm:", args.max_grad_norm)
    print("Target KL:", args.target_kl)
    print("n_steps:", args.n_steps)
    print("Batch size:", args.batch_size)
    print("n_epochs:", args.n_epochs)
    print("Eval freq:", args.eval_freq)
    print("Checkpoint freq:", args.checkpoint_freq)
    print("n_eval_episodes:", args.n_eval_episodes)
    print("Best model save path:", best_model_dir)
    print("Push enabled:", args.enable_push)

    if args.enable_push:
        print("Push window:", args.push_window_start, "to", args.push_window_end)
        print("Push force range:", args.push_force_min, "to", args.push_force_max)
        print("Push interval range:", args.push_interval_min, "to", args.push_interval_max)
        print("Push duration steps:", args.push_duration_steps)

    print("Output:", args.output)
    print("=" * 80)
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
    print("=" * 80)
    print("PPO walking training complete.")
    print("Saved final model:", args.output)
    print("Best model saved at:", os.path.join(best_model_dir, "best_model.zip"))
    print("=" * 80)


if __name__ == "__main__":
    main()