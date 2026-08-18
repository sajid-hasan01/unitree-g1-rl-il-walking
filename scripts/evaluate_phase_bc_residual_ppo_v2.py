from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from stable_baselines3 import PPO

from envs.g1_phase_bc_residual_env_v2 import G1PhaseBCResidualEnvV2


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_residual_ppo_v2/models/best_model/best_model.zip",
    )

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--gait_scale", type=float, default=0.16)
    parser.add_argument("--residual_scale", type=float, default=0.12)
    parser.add_argument("--target_smoothing", type=float, default=0.07)
    parser.add_argument("--transition_steps", type=int, default=240)
    parser.add_argument("--target_velocity", type=float, default=0.06)

    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_up_z", type=float, default=0.50)

    parser.add_argument("--forward_reward_weight", type=float, default=0.60)
    parser.add_argument("--backward_velocity_penalty_weight", type=float, default=2.50)
    parser.add_argument("--backward_position_penalty_weight", type=float, default=1.20)
    parser.add_argument("--fall_warning_weight", type=float, default=4.00)

    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--sleep_time", type=float, default=0.0)

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = G1PhaseBCResidualEnvV2(
        render_mode="human" if args.render else None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
        fall_height=args.fall_height,
        fall_up_z=args.fall_up_z,
        forward_reward_weight=args.forward_reward_weight,
        backward_velocity_penalty_weight=args.backward_velocity_penalty_weight,
        backward_position_penalty_weight=args.backward_position_penalty_weight,
        fall_warning_weight=args.fall_warning_weight,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()

    print("=" * 100)
    print("EVALUATING PHASE-BC RESIDUAL PPO V2")
    print("=" * 100)
    print("Model:", model_path)
    print("Steps:", args.steps)
    print("Deterministic:", args.deterministic)
    print("Start frame:", args.start_frame)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target smoothing:", args.target_smoothing)
    print("Transition steps:", args.transition_steps)
    print("Target velocity:", args.target_velocity)
    print("=" * 100)

    total_reward = 0.0
    final_info = info

    max_x = float(info["base_x"])
    min_z = float(info["base_z"])
    min_up_z = float(info["up_z"])

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        max_x = max(max_x, float(info["base_x"]))
        min_z = min(min_z, float(info["base_z"]))
        min_up_z = min(min_up_z, float(info["up_z"]))

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"x={info['base_x']:+.3f} "
                f"y={info['base_y']:+.3f} "
                f"z={info['base_z']:+.3f} "
                f"up_z={info['up_z']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"reward={reward:+.3f} "
                f"action_abs={info['action_mean_abs']:.3f} "
                f"back_vel_pen={info['backward_velocity_penalty']:.3f} "
                f"back_pos_pen={info['backward_position_penalty']:.3f} "
                f"fall_warn={info['fall_warning_penalty']:.3f} "
                f"fallen={terminated}"
            )

        if args.sleep_time > 0:
            time.sleep(args.sleep_time)

        if terminated or truncated:
            break

    print()
    print("=" * 100)
    print("PPO V2 EVALUATION SUMMARY")
    print("=" * 100)
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Final y:", f"{final_info['base_y']:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Max x:", f"{max_x:+.3f}")
    print("Min z:", f"{min_z:+.3f}")
    print("Min up_z:", f"{min_up_z:+.3f}")
    print("Total reward:", f"{total_reward:+.3f}")
    print("Terminated:", final_info["terminated"])
    print("Truncated:", final_info["truncated"])
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
