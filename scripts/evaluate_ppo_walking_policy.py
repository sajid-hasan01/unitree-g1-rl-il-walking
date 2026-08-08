import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


def build_env(args):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
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
    )
    return env


def format_contact(value):
    if value is None:
        return "None"
    return str(bool(value))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO walking policy for Unitree G1 in MuJoCo."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PPO .zip model.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the IL/reference dataset .npz file.",
    )

    parser.add_argument(
        "--reference_mode",
        type=str,
        default="transition",
        choices=["transition", "cyclic"],
        help="Reference playback mode.",
    )

    parser.add_argument("--target_velocity", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.05)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.15)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=250)

    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Start the reference motion at a random frame.",
    )

    parser.add_argument(
        "--include_contact_phase_observation",
        action="store_true",
        help=(
            "Use v21 contact-phase observation features. "
            "Required for models trained with 65-dimensional observations."
        ),
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy action.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="Maximum number of evaluation steps.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Print status every N environment steps.",
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable MuJoCo viewer.",
    )

    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Sleep approximately env.control_dt each step for slower real-time viewing.",
    )

    # Push-disturbance options
    parser.add_argument(
        "--enable_push",
        action="store_true",
        help="Enable randomized external pushes at pelvis during evaluation.",
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

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    env = build_env(args)

    model_path = str(Path(args.model).resolve())

    print("Evaluating PPO walking policy")
    print("Model:", model_path)
    print("Observation shape:", env.observation_space.shape)
    print("Action shape:", env.action_space.shape)
    print("Reference mode:", args.reference_mode)
    print("Dataset path:", args.dataset_path)
    print("Has reference root positions:", env.has_reference_root_positions)
    print("Has reference contact mask:", env.has_reference_contact_mask)
    print("Include contact phase observation:", args.include_contact_phase_observation)
    print("Push enabled:", args.enable_push)

    model = PPO.load(model_path, device="auto")

    observation, info = env.reset()

    print("Initial info:", info)

    total_reward = 0.0
    steps_survived = 0

    contact_match_count = 0
    contact_check_count = 0

    viewer = None

    try:
        if not args.no_viewer:
            print()
            print("Viewer opening...")
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        for step in range(args.steps):
            action, _ = model.predict(
                observation,
                deterministic=args.deterministic,
            )

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps_survived = step + 1

            left_expected = info.get("left_expected_contact", None)
            right_expected = info.get("right_expected_contact", None)

            left_contact = bool(info.get("left_contact", False))
            right_contact = bool(info.get("right_contact", False))

            if left_expected is not None:
                contact_check_count += 1
                if left_contact == bool(left_expected):
                    contact_match_count += 1

            if right_expected is not None:
                contact_check_count += 1
                if right_contact == bool(right_expected):
                    contact_match_count += 1

            if args.print_every > 0 and step % args.print_every == 0:
                print(
                    f"step={step:04d}, "
                    f"x={info.get('x_position', 0.0):.3f}, "
                    f"x_vel={info.get('x_velocity', 0.0):.3f}, "
                    f"height={info.get('base_height', 0.0):.3f}, "
                    f"up_z={info.get('up_z', 0.0):.3f}, "
                    f"motion_frame={info.get('motion_frame', 0.0):.2f}, "
                    f"reward={float(reward):.3f}, "
                    f"L=({left_contact}/exp={format_contact(left_expected)}) "
                    f"R=({right_contact}/exp={format_contact(right_expected)}) "
                    f"slip=("
                    f"{info.get('left_foot_slip', 0.0):.3f},"
                    f"{info.get('right_foot_slip', 0.0):.3f}"
                    f") "
                    f"clr=("
                    f"{info.get('left_foot_clearance', 0.0):.3f},"
                    f"{info.get('right_foot_clearance', 0.0):.3f}"
                    f")"
                )

            if viewer is not None:
                viewer.sync()

            if args.real_time:
                time.sleep(env.control_dt)

            if terminated or truncated:
                print()
                print("Episode ended.")
                print("terminated:", terminated)
                print("truncated:", truncated)
                break

    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    print("steps survived:", steps_survived)
    print("total reward:", total_reward)
    print("final info:", info)

    if contact_check_count > 0:
        contact_match_rate = contact_match_count / contact_check_count
        print(
            "contact phase match rate:",
            f"{contact_match_rate:.3f}",
            f"({contact_match_count}/{contact_check_count} foot-checks)",
        )
    else:
        print("contact phase match rate: unavailable")


if __name__ == "__main__":
    main()