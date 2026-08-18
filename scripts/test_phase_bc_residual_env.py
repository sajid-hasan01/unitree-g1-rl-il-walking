from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env import G1PhaseBCResidualEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gait_scale", type=float, default=0.20)
    parser.add_argument("--residual_scale", type=float, default=0.08)
    parser.add_argument("--target_smoothing", type=float, default=0.08)
    parser.add_argument("--transition_steps", type=int, default=180)
    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--target_velocity", type=float, default=-0.08)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()

    env = G1PhaseBCResidualEnv(
        render_mode="human" if args.render else None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
    )

    obs, info = env.reset()

    print("=" * 100)
    print("ZERO-RESIDUAL TEST: PHASE-BC RESIDUAL ENV")
    print("=" * 100)
    print("Observation shape:", obs.shape)
    print("Action shape:", env.action_space.shape)
    print("Start frame:", args.start_frame)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target smoothing:", args.target_smoothing)
    print("Transition steps:", args.transition_steps)
    print("=" * 100)

    action = np.zeros(env.action_space.shape, dtype=np.float32)

    final_info = info
    total_reward = 0.0

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_info = info

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"x={info['base_x']:+.3f} "
                f"y={info['base_y']:+.3f} "
                f"z={info['base_z']:+.3f} "
                f"up_z={info['up_z']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"reward={reward:+.3f} "
                f"fallen={terminated}"
            )

        if terminated or truncated:
            break

    print()
    print("=" * 100)
    print("ZERO-RESIDUAL TEST SUMMARY")
    print("=" * 100)
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Final y:", f"{final_info['base_y']:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Total reward:", f"{total_reward:+.3f}")
    print("Terminated:", final_info["terminated"])
    print("Truncated:", final_info["truncated"])
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
