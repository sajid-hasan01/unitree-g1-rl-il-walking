import argparse
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
import numpy as np

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        type=str,
        default="zero",
        choices=["zero", "random"],
    )

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--action_noise", type=float, default=0.10)
    parser.add_argument("--target_velocity", type=float, default=0.25)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--action_scale", type=float, default=0.25)
    parser.add_argument("--slowmo", type=float, default=1.0)

    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        target_forward_velocity=args.target_velocity,
        height_offset=args.height_offset,
        action_scale=args.action_scale,
        random_start=False,
    )

    observation, info = env.reset()

    print("Dynamic G1 walking environment loaded.")
    print("Observation shape:", observation.shape)
    print("Action shape:", env.action_space.shape)
    print("Initial info:", info)
    print()
    print("Viewer opening...")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.steps):
            if not viewer.is_running():
                break

            if args.policy == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action = args.action_noise * np.random.randn(*env.action_space.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)

            observation, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            if step % 25 == 0:
                print(
                    f"step={step:04d}, "
                    f"x={info['x_position']:.3f}, "
                    f"x_vel={info['x_velocity']:.3f}, "
                    f"height={info['base_height']:.3f}, "
                    f"up_z={info['up_z']:.3f}, "
                    f"reward={reward:.3f}"
                )

            if terminated or truncated:
                print("Episode ended.")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("final info:", info)
                break

            time.sleep(env.control_dt * args.slowmo)

    env.close()


if __name__ == "__main__":
    main()