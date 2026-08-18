from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.g1_phase_bc_residual_env_v2 import G1PhaseBCResidualEnvV2


def make_env(args: argparse.Namespace):
    def _init():
        env = G1PhaseBCResidualEnvV2(
            dataset_path=args.dataset,
            phase_bc_policy_path=args.phase_bc_policy,
            mujoco_model_path=args.mujoco_model,
            render_mode=None,
            start_frame=args.start_frame,
            reverse_time=args.reverse_time,
            gait_scale=args.gait_scale,
            residual_scale=args.residual_scale,
            target_smoothing=args.target_smoothing,
            transition_steps=args.transition_steps,
            max_episode_steps=args.max_episode_steps,
            target_velocity=args.target_velocity,
            fall_height=args.fall_height,
            fall_up_z=args.fall_up_z,
            random_start=args.random_start,
            forward_reward_weight=args.forward_reward_weight,
            backward_velocity_penalty_weight=args.backward_velocity_penalty_weight,
            backward_position_penalty_weight=args.backward_position_penalty_weight,
            fall_warning_weight=args.fall_warning_weight,
        )
        return Monitor(env)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument(
        "--phase_bc_policy",
        default="experiments/amass_b3_15dof_il_phase_bc/models/g1_amass_b3_phase_bc_15dof_best.pt",
    )
    parser.add_argument(
        "--mujoco_model",
        default="third_party/mujoco_menagerie/unitree_g1/scene.xml",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/amass_b3_15dof_residual_ppo_v2",
    )

    parser.add_argument("--total_timesteps", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)

    # Environment settings
    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--reverse_time", action="store_true")
    parser.add_argument("--random_start", action="store_true")

    parser.add_argument("--gait_scale", type=float, default=0.16)
    parser.add_argument("--residual_scale", type=float, default=0.12)
    parser.add_argument("--target_smoothing", type=float, default=0.07)
    parser.add_argument("--transition_steps", type=int, default=240)
    parser.add_argument("--max_episode_steps", type=int, default=600)

    # Corrected direction: positive X is forward.
    parser.add_argument("--target_velocity", type=float, default=0.06)

    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_up_z", type=float, default=0.50)

    # Reward settings
    parser.add_argument("--forward_reward_weight", type=float, default=0.60)
    parser.add_argument("--backward_velocity_penalty_weight", type=float, default=2.50)
    parser.add_argument("--backward_position_penalty_weight", type=float, default=1.20)
    parser.add_argument("--fall_warning_weight", type=float, default=4.00)

    # More conservative PPO settings than V1.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.10)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--vf_coef", type=float, default=0.50)
    parser.add_argument("--max_grad_norm", type=float, default=0.50)

    parser.add_argument("--eval_freq", type=int, default=15000)
    parser.add_argument("--save_freq", type=int, default=15000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = out_dir / "models"
    log_dir = out_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(args)])
    eval_env = DummyVecEnv([make_env(args)])

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="phase_bc_residual_ppo_v2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256],
        }
    }

    print("=" * 100)
    print("TRAINING PHASE-BC RESIDUAL PPO V2")
    print("=" * 100)
    print("Output dir:", out_dir)
    print("Total timesteps:", args.total_timesteps)
    print("Target velocity:", args.target_velocity)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target smoothing:", args.target_smoothing)
    print("Transition steps:", args.transition_steps)
    print("Learning rate:", args.learning_rate)
    print("Clip range:", args.clip_range)
    print("=" * 100)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=args.seed,
        device="auto",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    final_model_path = model_dir / "phase_bc_residual_ppo_v2_final.zip"
    model.save(str(final_model_path))

    print()
    print("=" * 100)
    print("TRAINING COMPLETE - PPO V2")
    print("=" * 100)
    print("Final model:", final_model_path)
    print("Best model folder:", model_dir / "best_model")
    print("Checkpoint folder:", model_dir / "checkpoints")
    print("=" * 100)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
