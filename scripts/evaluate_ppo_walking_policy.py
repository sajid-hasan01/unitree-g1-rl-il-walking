import argparse
import sys
import time
from pathlib import Path
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/g1_ppo_walking_policy.zip",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--target_velocity", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.15)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=250)

    # Reference dataset / mode options
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the IL dataset .npz file. Defaults to the standard transition dataset if not given.",
    )
    parser.add_argument(
        "--reference_mode",
        type=str,
        default="transition",
        choices=["transition", "cyclic"],
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start the reference motion at a random frame instead of frame 0.",
    )

    # Push-disturbance (push-recovery) options
    parser.add_argument(
        "--enable_push",
        action="store_true",
        help="Enable randomized external pushes at the pelvis during evaluation.",
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
    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
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
    model = PPO.load(str(model_path), env=env, device="auto")
    observation, info = env.reset()
    print("Evaluating PPO walking policy")
    print("Model:", model_path)
    print("Observation shape:", observation.shape)
    print("Action shape:", env.action_space.shape)
    print("Reference mode:", env.reference_mode)
    print("Dataset path:", env.dataset_path)
    print("Has reference root positions:", env.has_reference_root_positions)
    print("Push enabled:", args.enable_push)
    if args.enable_push:
        print("Push window:", env.push_window_start, "to", env.push_window_end)
        print("Push force range:", args.push_force_min, "to", args.push_force_max)
    print("Initial info:", info)
    print()
    print("Viewer opening...")
    total_reward = 0.0
    push_events = []
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.steps):
            if not viewer.is_running():
                break
            action, _ = model.predict(
                observation,
                deterministic=args.deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            viewer.sync()

            if info.get("push_active"):
                if not push_events or push_events[-1]["end_step"] is not None:
                    push_events.append(
                        {
                            "start_step": step,
                            "end_step": None,
                            "force_magnitude": info["push_force_magnitude"],
                        }
                    )
            elif push_events and push_events[-1]["end_step"] is None:
                push_events[-1]["end_step"] = step

            if step % 25 == 0 or info.get("push_active"):
                push_marker = (
                    f", PUSH force={info['push_force_magnitude']:.1f}N"
                    if info.get("push_active")
                    else ""
                )
                print(
                    f"step={step:04d}, "
                    f"x={info['x_position']:.3f}, "
                    f"x_vel={info['x_velocity']:.3f}, "
                    f"height={info['base_height']:.3f}, "
                    f"up_z={info['up_z']:.3f}, "
                    f"motion_frame={info['motion_frame']:.2f}, "
                    f"reward={reward:.3f}"
                    f"{push_marker}"
                )
            if terminated or truncated:
                print()
                print("Episode ended.")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("steps survived:", step + 1)
                print("total reward:", total_reward)
                print("final info:", info)
                if args.enable_push:
                    print("push events:", push_events)
                break
            time.sleep(env.control_dt * args.slowmo)
    env.close()
if __name__ == "__main__":
    main()