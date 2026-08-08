from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.g1_mimic_phase_lift_env import G1MimicPhaseLiftEnv


def make_env(args, *, randomize_reset: bool = True):
    env = G1MimicPhaseLiftEnv(
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
        randomize_reset=randomize_reset,
    )
    return Monitor(env)


class QualityEvalCallback(BaseCallback):
    """
    Saves the best model by task quality, not raw reward.

    This prevents the old failure where standing still was saved as "best" because
    survival reward was high while clearance was zero.
    """

    def __init__(self, args, save_dir: str, eval_freq: int = 10000, n_eval_episodes: int = 3):
        super().__init__(verbose=1)
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_score = -1e18

    def _run_eval_episode(self) -> Dict[str, float]:
        env = G1MimicPhaseLiftEnv(
            model_path=self.args.model_path,
            dataset_path=self.args.dataset_path,
            stage=self.args.stage,
            action_scale=self.args.action_scale,
            cycle_duration=self.args.cycle_duration,
            swing_start=self.args.swing_start,
            swing_end=self.args.swing_end,
            target_clearance=self.args.target_clearance,
            target_lateral_shift=self.args.target_lateral_shift,
            max_steps=self.args.max_steps,
            frame_skip=self.args.frame_skip,
            action_target_smoothing=self.args.action_target_smoothing,
            mimic_weight=self.args.mimic_weight,
            mimic_vel_weight=self.args.mimic_vel_weight,
            mimic_only_during_swing=not self.args.mimic_all_phase,
            reference_reverse=self.args.reference_reverse,
            randomize_reset=False,
        )

        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        max_left = 0.0
        max_right = 0.0
        min_up = 1.0
        contacts: List[float] = []
        slips: List[float] = []
        max_backward = 0.0
        final = info

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)
            max_left = max(max_left, float(info["left_foot_clearance"]))
            max_right = max(max_right, float(info["right_foot_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            contacts.append(float(info["contact_accuracy"]))
            slips.append(float(info["support_slip"]))
            max_backward = max(max_backward, float(info.get("backward_excess", 0.0)))
            final = info

        if self.args.stage == "right_lift":
            main_clearance = max_right
        elif self.args.stage == "left_lift":
            main_clearance = max_left
        else:
            main_clearance = min(max_left, max_right)

        abs_final_x = abs(float(final["x_position"]))
        final_x_vel = abs(float(final["x_velocity"]))
        contact_mean = float(np.mean(contacts)) if contacts else 0.0
        slip_mean = float(np.mean(slips)) if slips else 0.0

        # Task-quality score. Clearance and duration both matter; zero-clearance
        # max-step policies score poorly.
        score = (
            1.0 * steps
            + 1800.0 * min(main_clearance, 0.050)
            + 160.0 * min_up
            + 120.0 * contact_mean
            - 420.0 * abs_final_x
            - 180.0 * final_x_vel
            - 240.0 * slip_mean
            - 350.0 * max_backward
        )

        row = {
            "steps": float(steps),
            "reward": float(total_reward),
            "main_clearance": float(main_clearance),
            "max_left_clearance": float(max_left),
            "max_right_clearance": float(max_right),
            "min_up_z": float(min_up),
            "contact": contact_mean,
            "slip": slip_mean,
            "final_x": float(final["x_position"]),
            "final_x_velocity": float(final["x_velocity"]),
            "score": float(score),
            "reason": env.termination_reason(final),
        }
        env.close()
        return row

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rows = [self._run_eval_episode() for _ in range(self.n_eval_episodes)]
        mean_score = float(np.mean([r["score"] for r in rows]))
        mean_steps = float(np.mean([r["steps"] for r in rows]))
        mean_clearance = float(np.mean([r["main_clearance"] for r in rows]))
        mean_up = float(np.mean([r["min_up_z"] for r in rows]))
        mean_x = float(np.mean([r["final_x"] for r in rows]))
        reason = rows[0]["reason"]

        print(
            f"[QUALITY EVAL] steps={mean_steps:.1f} clear={mean_clearance:.4f} "
            f"up={mean_up:.3f} final_x={mean_x:+.3f} score={mean_score:.2f} reason={reason}"
        )

        if mean_score > self.best_score:
            self.best_score = mean_score
            out = self.save_dir / "best_quality_model"
            self.model.save(str(out))
            print(f"[QUALITY EVAL] New best quality model saved: {out}.zip score={self.best_score:.2f}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Train DeepMimic-lite phase lift policy.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--output", type=str, default="models/g1_mimic_phase_right_lift.zip")
    parser.add_argument("--checkpoint_dir", type=str, default="models/g1_mimic_phase_right_lift_checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/g1_mimic_phase_right_lift")

    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.08)
    parser.add_argument("--ent_coef", type=float, default=0.004)
    parser.add_argument("--target_kl", type=float, default=0.02)

    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--cycle_duration", type=float, default=3.0)
    parser.add_argument("--swing_start", type=float, default=0.20)
    parser.add_argument("--swing_end", type=float, default=0.70)
    parser.add_argument("--target_clearance", type=float, default=0.025)
    parser.add_argument("--target_lateral_shift", type=float, default=0.025)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)

    parser.add_argument("--mimic_weight", type=float, default=0.18)
    parser.add_argument("--mimic_vel_weight", type=float, default=0.04)
    parser.add_argument("--mimic_all_phase", action="store_true")
    parser.add_argument("--reference_reverse", action="store_true")

    parser.add_argument("--quality_eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=3)
    parser.add_argument("--checkpoint_freq", type=int, default=25000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = make_env(args, randomize_reset=True)
    probe = env.unwrapped if hasattr(env, "unwrapped") else env

    print("=" * 100)
    print("MIMIC-PHASE-LIFT TRAINING")
    print("Reference is reward-only; no pose teacher is written to ctrl.")
    print("stage:", args.stage)
    print("dataset:", args.dataset_path)
    print("reference_note:", getattr(probe, "reference_note", "unknown"))
    print("mimic_weight:", args.mimic_weight, "mimic_vel_weight:", args.mimic_vel_weight)
    print("=" * 100)

    if args.init_model:
        print("Loading init model:", args.init_model)
        model = PPO.load(args.init_model, env=env, device=args.device)
        model.learning_rate = get_schedule_fn(args.learning_rate)
        model.clip_range = get_schedule_fn(args.clip_range)
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    callbacks = CallbackList(
        [
            CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.checkpoint_dir, name_prefix="g1_mimic_phase"),
            QualityEvalCallback(
                args=args,
                save_dir=str(Path(args.checkpoint_dir) / "quality_best"),
                eval_freq=args.quality_eval_freq,
                n_eval_episodes=args.n_eval_episodes,
            ),
        ]
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=True, tb_log_name=f"mimic_phase_{args.stage}")
    model.save(args.output)
    print("Saved:", args.output)
    env.close()


if __name__ == "__main__":
    main()
