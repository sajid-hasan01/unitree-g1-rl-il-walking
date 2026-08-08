from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, obs_as_tensor

from envs.g1_phase_lift_env import G1PhaseLiftEnv


def make_env(args, *, randomize_reset: bool = True):
    env = G1PhaseLiftEnv(
        model_path=args.model_path,
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
        randomize_reset=randomize_reset,
    )
    return Monitor(env)


def manual_phase_action(stage: str, phi: float, swing_start: float, swing_end: float) -> np.ndarray:
    """
    Manual action used only for optional BC warm-start.

    It is not applied inside the environment during PPO. It only initializes policy
    weights so PPO starts near a rough "can lift a foot" behavior.
    """
    action = np.zeros(15, dtype=np.float32)

    def in_window(p, a, b):
        return a <= p < b if a <= b else (p >= a or p < b)

    def envelope(p, a, b):
        if not in_window(p, a, b):
            return 0.0
        local = (p - a) / max(b - a, 1e-6) if a <= b else ((p - a) % 1.0) / max((b - a) % 1.0, 1e-6)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * local))

    if stage == "right_lift":
        right_env = envelope(phi, swing_start, swing_end)
        left_env = 0.0
        lateral = min(1.0, phi / max(swing_start, 1e-6)) if phi < swing_start else 1.0
    elif stage == "left_lift":
        right_env = 0.0
        left_env = envelope(phi, swing_start, swing_end)
        lateral = min(1.0, phi / max(swing_start, 1e-6)) if phi < swing_start else 1.0
    else:
        right_env = envelope(phi, 0.12, 0.38)
        left_env = envelope(phi, 0.62, 0.88)
        lateral = max(right_env, left_env)

    # Lateral shift actions. Positive y target for right swing/left support.
    if right_env > 0.0 or (stage == "right_lift" and phi < swing_end):
        action[1] = -0.35 * lateral
        action[5] = +0.20 * lateral
        action[7] = -0.25 * lateral
        action[11] = +0.18 * lateral
        action[13] = +0.25 * lateral
    if left_env > 0.0 or (stage == "left_lift" and phi < swing_end):
        action[1] = +0.25 * lateral
        action[5] = -0.18 * lateral
        action[7] = +0.35 * lateral
        action[11] = -0.20 * lateral
        action[13] = -0.25 * lateral

    # Rough lift actions. PPO is expected to refine these; they are not a teacher.
    if right_env > 0.0:
        action[6] += 0.45 * right_env   # right hip pitch
        action[7] += 0.18 * right_env
        action[9] += 0.95 * right_env   # right knee
        action[10] += 0.65 * right_env  # right ankle pitch
        action[11] += 0.15 * right_env
    if left_env > 0.0:
        action[0] += 0.45 * left_env
        action[1] += -0.18 * left_env
        action[3] += 0.95 * left_env
        action[4] += 0.65 * left_env
        action[5] += -0.15 * left_env

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_bc_data(args, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    env = G1PhaseLiftEnv(
        model_path=args.model_path,
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
        randomize_reset=True,
    )

    obs_list = []
    act_list = []

    obs, info = env.reset()
    while len(obs_list) < n_samples:
        phi = float(info["phase"])
        action = manual_phase_action(args.stage, phi, args.swing_start, args.swing_end)
        obs_list.append(obs.copy())
        act_list.append(action.copy())

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def behavior_clone_warmstart(model: PPO, args) -> None:
    if args.bc_samples <= 0 or args.bc_epochs <= 0:
        return

    print("=" * 100)
    print("BC WARM-START")
    print(f"samples={args.bc_samples} epochs={args.bc_epochs} batch={args.bc_batch_size}")
    print("Teacher used only for initial policy weights; it is not used by the env during PPO.")
    print("=" * 100)

    obs_np, act_np = collect_bc_data(args, args.bc_samples)
    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    obs_tensor_all = torch.as_tensor(obs_np, device=device)
    act_tensor_all = torch.as_tensor(act_np, device=device)

    n = obs_np.shape[0]
    last_loss = None

    for epoch in range(args.bc_epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, args.bc_batch_size):
            idx = perm[start:start + args.bc_batch_size]
            obs_batch = obs_tensor_all[idx]
            act_batch = act_tensor_all[idx]

            dist = policy.get_distribution(obs_batch)
            if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                pred = dist.distribution.mean
            else:
                pred = dist.mode()

            loss = F.mse_loss(pred, act_batch)
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        last_loss = float(np.mean(epoch_losses))
        print(f"bc_epoch={epoch+1:03d}/{args.bc_epochs:03d} loss={last_loss:.6f}")

    print("BC warm-start complete. final_loss=", last_loss)


def main():
    parser = argparse.ArgumentParser(description="Train phase-based G1 lift policy without pose teacher.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--output", type=str, default="models/g1_phase_right_lift.zip")
    parser.add_argument("--checkpoint_dir", type=str, default="models/g1_phase_right_lift_checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/g1_phase_right_lift")

    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.10)
    parser.add_argument("--ent_coef", type=float, default=0.004)
    parser.add_argument("--target_kl", type=float, default=0.025)

    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--cycle_duration", type=float, default=3.0)
    parser.add_argument("--swing_start", type=float, default=0.20)
    parser.add_argument("--swing_end", type=float, default=0.70)
    parser.add_argument("--target_clearance", type=float, default=0.025)
    parser.add_argument("--target_lateral_shift", type=float, default=0.025)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)

    parser.add_argument("--bc_samples", type=int, default=0)
    parser.add_argument("--bc_epochs", type=int, default=0)
    parser.add_argument("--bc_batch_size", type=int, default=256)

    parser.add_argument("--eval_freq", type=int, default=25000)
    parser.add_argument("--n_eval_episodes", type=int, default=3)
    parser.add_argument("--checkpoint_freq", type=int, default=25000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = make_env(args, randomize_reset=True)
    eval_env = make_env(args, randomize_reset=False)

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

    behavior_clone_warmstart(model, args)

    callbacks = CallbackList(
        [
            CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.checkpoint_dir, name_prefix="g1_phase_lift"),
            EvalCallback(
                eval_env,
                best_model_save_path=str(Path(args.checkpoint_dir) / "best_model"),
                log_path=str(Path(args.checkpoint_dir) / "eval_logs"),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            ),
        ]
    )

    print("=" * 100)
    print("PHASE-LIFT PPO TRAINING")
    print("stage:", args.stage)
    print("No pose teacher. No dataset frames. Anti-standing + x-guard phase reward v3.")
    print("action_scale:", args.action_scale, "cycle_duration:", args.cycle_duration)
    print("swing window:", args.swing_start, "to", args.swing_end, "target_clearance:", args.target_clearance)
    print("=" * 100)

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=True, tb_log_name=f"phase_{args.stage}")
    model.save(args.output)
    print("Saved:", args.output)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
