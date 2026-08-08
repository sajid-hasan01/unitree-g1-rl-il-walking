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

from envs.g1_hybrid_wbc_lite_lift_env import G1HybridWBCLiteLiftEnv


def make_env(args, randomize_reset=True):
    return Monitor(
        G1HybridWBCLiteLiftEnv(
            model_path=args.model_path,
            stage=args.stage,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            cycle_duration=args.cycle_duration,
            swing_start=args.swing_start,
            swing_end=args.swing_end,
            target_clearance=args.target_clearance,
            target_lateral_shift=args.target_lateral_shift,
            x_soft_limit=args.x_soft_limit,
            x_hard_limit=args.x_hard_limit,
            x_velocity_soft_limit=args.x_velocity_soft_limit,
            x_velocity_hard_limit=args.x_velocity_hard_limit,
            action_smoothing=args.action_smoothing,
            randomize_reset=randomize_reset,
        )
    )


class HybridQualityCallback(BaseCallback):
    def __init__(self, args, save_dir: str, eval_freq: int, n_eval_episodes: int):
        super().__init__(verbose=1)
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_score = -1e18

    def _eval_one(self) -> Dict[str, float]:
        env = G1HybridWBCLiteLiftEnv(
            model_path=self.args.model_path,
            stage=self.args.stage,
            frame_skip=self.args.frame_skip,
            max_steps=self.args.max_steps,
            cycle_duration=self.args.cycle_duration,
            swing_start=self.args.swing_start,
            swing_end=self.args.swing_end,
            target_clearance=self.args.target_clearance,
            target_lateral_shift=self.args.target_lateral_shift,
            x_soft_limit=self.args.x_soft_limit,
            x_hard_limit=self.args.x_hard_limit,
            x_velocity_soft_limit=self.args.x_velocity_soft_limit,
            x_velocity_hard_limit=self.args.x_velocity_hard_limit,
            action_smoothing=self.args.action_smoothing,
            randomize_reset=False,
        )
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        max_clear = 0.0
        min_up = 1.0
        contacts: List[float] = []
        slips: List[float] = []
        final = info
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)
            max_clear = max(max_clear, float(info["main_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            contacts.append(float(info["contact_accuracy"]))
            slips.append(float(info["support_slip"]))
            final = info
        abs_x = abs(float(final["x_position"]))
        abs_xv = abs(float(final["x_velocity"]))
        contact = float(np.mean(contacts)) if contacts else 0.0
        slip = float(np.mean(slips)) if slips else 0.0
        score = (
            steps
            + 4200.0 * min(max_clear, 0.030)
            + 160.0 * min_up
            + 140.0 * contact
            - 900.0 * abs_x
            - 320.0 * abs_xv
            - 250.0 * slip
            - (240.0 if max_clear < 0.008 else 0.0)
        )
        row = {
            "steps": float(steps),
            "reward": float(total_reward),
            "clearance": float(max_clear),
            "min_up": float(min_up),
            "contact": float(contact),
            "slip": float(slip),
            "x": float(final["x_position"]),
            "xv": float(final["x_velocity"]),
            "score": float(score),
            "reason": env.termination_reason(final),
        }
        env.close()
        return row

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rows = [self._eval_one() for _ in range(self.n_eval_episodes)]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ["steps", "reward", "clearance", "min_up", "contact", "slip", "x", "xv", "score"]}
        reason = rows[0]["reason"]
        print(
            f"[HYBRID QUALITY] steps={mean['steps']:.1f} clear={mean['clearance']:.4f} "
            f"up={mean['min_up']:.3f} x={mean['x']:+.3f} xv={mean['xv']:+.3f} "
            f"contact={mean['contact']:.3f} score={mean['score']:.2f} reason={reason}"
        )
        eligible = (
            mean["clearance"] >= 0.008
            and abs(mean["x"]) <= self.args.x_hard_limit
            and mean["min_up"] >= 0.78
            and reason not in {"no_lift_mid_swing", "x_position_limit", "x_velocity_limit"}
        )
        if eligible and mean["score"] > self.best_score:
            self.best_score = mean["score"]
            out = self.save_dir / "best_hybrid_quality_model"
            self.model.save(str(out))
            print(f"[HYBRID QUALITY] Saved best: {out}.zip score={self.best_score:.2f}")
        elif not eligible:
            print("[HYBRID QUALITY] Not saved: failed lift/root/stability gate.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train hybrid WBC-lite + residual PPO foot-lift controller.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--output", type=str, default="models/g1_hybrid_wbc_lite_right_lift_v1.zip")
    parser.add_argument("--checkpoint_dir", type=str, default="models/g1_hybrid_wbc_lite_right_lift_v1_checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/g1_hybrid_wbc_lite_right_lift_v1")
    parser.add_argument("--total_timesteps", type=int, default=150000)
    parser.add_argument("--learning_rate", type=float, default=0.00008)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.10)
    parser.add_argument("--ent_coef", type=float, default=0.008)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--cycle_duration", type=float, default=3.6)
    parser.add_argument("--swing_start", type=float, default=0.28)
    parser.add_argument("--swing_end", type=float, default=0.78)
    parser.add_argument("--target_clearance", type=float, default=0.012)
    parser.add_argument("--target_lateral_shift", type=float, default=0.032)
    parser.add_argument("--x_soft_limit", type=float, default=0.08)
    parser.add_argument("--x_hard_limit", type=float, default=0.20)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.22)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.00)
    parser.add_argument("--action_smoothing", type=float, default=0.70)
    parser.add_argument("--quality_eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=3)
    parser.add_argument("--checkpoint_freq", type=int, default=25000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    env = make_env(args, randomize_reset=True)
    print("=" * 100)
    print("HYBRID WBC-LITE V3 + RESIDUAL PPO TRAINING")
    print("Action space: 6 high-level residuals, not 15 raw joint controls.")
    print("Optional init_model is supported when action/observation shape matches.")
    print("=" * 100)
    if args.init_model:
        print("Loading init model:", args.init_model)
        from stable_baselines3.common.utils import get_schedule_fn
        model = PPO.load(args.init_model, env=env, device=args.device)
        model.learning_rate = get_schedule_fn(args.learning_rate)
        model.clip_range = get_schedule_fn(args.clip_range)
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
    else:
        model = PPO(
            "MlpPolicy", env,
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
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        )
    callbacks = CallbackList([
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.checkpoint_dir, name_prefix="g1_hybrid_wbc_lite"),
        HybridQualityCallback(args, save_dir=str(Path(args.checkpoint_dir) / "quality_best"), eval_freq=args.quality_eval_freq, n_eval_episodes=args.n_eval_episodes),
    ])
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=True, tb_log_name=f"hybrid_wbc_lite_{args.stage}")
    model.save(args.output)
    print("Saved:", args.output)
    env.close()


if __name__ == "__main__":
    main()
