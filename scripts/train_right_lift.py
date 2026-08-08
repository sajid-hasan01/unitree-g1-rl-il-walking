from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.g1_right_lift_env import G1RightLiftEnv


def make_env(args, rank: int, eval_mode: bool):
    def _init():
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
            enable_scripted_arms=not args.disable_scripted_arms,
            arm_swing_scale=args.arm_swing_scale,
            arm_pitch_sign=args.arm_pitch_sign,
            arm_elbow_scale=args.arm_elbow_scale,
            frame_skip=args.frame_skip,
            randomize_reset=not eval_mode,
            seed=args.seed + rank,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train right_lift sagittal-stability fix.")
    parser.add_argument("--init_model", type=str, default="models/g1_com_shift_lift_shift_right_v2_fixed/passed_shift_right.zip")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--output", type=str, default="models/g1_right_lift_sagittal_fix.zip")
    parser.add_argument("--checkpoint_dir", type=str, default="models/g1_right_lift_sagittal_fix_checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/g1_right_lift_sagittal_fix")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--n_envs", type=int, default=1)

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

    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.05)
    parser.add_argument("--ent_coef", type=float, default=0.0006)
    parser.add_argument("--target_kl", type=float, default=0.015)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--checkpoint_freq", type=int, default=25000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    args = parser.parse_args()

    if not Path(args.init_model).exists():
        raise FileNotFoundError(f"Initial model not found: {args.init_model}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args, rank=i, eval_mode=False) for i in range(args.n_envs)])
    env = VecMonitor(env)
    eval_env = DummyVecEnv([make_env(args, rank=1000, eval_mode=True)])
    eval_env = VecMonitor(eval_env)

    print("=" * 100)
    print("RIGHT_LIFT SAGITTAL-STABILITY TRAINING")
    print("Initial model:", args.init_model)
    print("Output:", args.output)
    print("Timesteps:", args.total_timesteps)
    print("Teacher multiplier:", args.teacher_scale_multiplier)
    print("Residual scales: support=", args.support_leg_scale, "swing=", args.swing_leg_scale, "waist=", args.waist_scale)
    print("Sagittal feedback: kp=", args.sagittal_kp, "kd=", args.sagittal_kd, "clip=", args.sagittal_clip)
    print("Sagittal signs: hip=", args.sagittal_hip_sign, "ankle=", args.sagittal_ankle_sign)
    print("Scripted arms:", "enabled=", not args.disable_scripted_arms, "scale=", args.arm_swing_scale, "pitch_sign=", args.arm_pitch_sign, "elbow=", args.arm_elbow_scale)
    print("=" * 100)

    model = PPO.load(args.init_model, env=env, device=args.device)

    # Keep checkpoint-compatible policy, but update learning settings safely.
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    model.clip_range = get_schedule_fn(args.clip_range)
    model.ent_coef = args.ent_coef
    model.target_kl = args.target_kl
    model.n_epochs = args.n_epochs
    model.gamma = args.gamma
    model.gae_lambda = args.gae_lambda
    model.tensorboard_log = str(args.log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=args.checkpoint_dir,
        name_prefix="g1_right_lift",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(args.checkpoint_dir) / "best_model"),
        log_path=str(Path(args.checkpoint_dir) / "eval_logs"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="right_lift_sagittal_fix",
    )

    model.save(args.output)
    model.save(str(Path(args.checkpoint_dir) / "latest_model.zip"))

    print("=" * 100)
    print("Training complete")
    print("Saved final:", args.output)
    print("Best model folder:", str(Path(args.checkpoint_dir) / "best_model"))
    print("=" * 100)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
