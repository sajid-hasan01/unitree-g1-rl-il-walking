import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)


def format_contact(value):
    if value is None:
        return "None"
    return str(bool(value))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic zero-residual reference-control test for Unitree G1. "
            "This does NOT use PPO. It applies zero residual action so the robot "
            "tracks only the reference joint targets through MuJoCo position actuators."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--reference_mode", type=str, default="cyclic", choices=["transition", "cyclic"])
    parser.add_argument("--target_velocity", type=float, default=-0.10)
    parser.add_argument("--action_scale", type=float, default=0.055)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.28)
    parser.add_argument("--initial_stand_steps", type=int, default=70)
    parser.add_argument("--transition_steps", type=int, default=220)
    parser.add_argument("--demo_stop_step", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--sleep_time", type=float, default=0.025)

    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
    )

    observation, info = env.reset()
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    print()
    print("=" * 80)
    print("ZERO-RESIDUAL DYNAMIC REFERENCE TEST")
    print("=" * 80)
    print("This test uses NO PPO policy.")
    print("It applies action = zeros, so the robot follows only the reference joint targets.")
    print("Dataset:", args.dataset_path)
    print("Observation shape:", env.observation_space.shape)
    print("Action shape:", env.action_space.shape)
    print("Target velocity:", args.target_velocity)
    print("Action scale:", args.action_scale)
    print("Action target smoothing:", args.action_target_smoothing)
    print("Reference speed:", args.reference_speed)
    print("Initial stand steps:", args.initial_stand_steps)
    print("Transition steps:", args.transition_steps)
    print("Initial info:", info)

    total_reward = 0.0
    contact_match_count = 0
    contact_check_count = 0
    final_info = info
    steps_survived = 0

    viewer = None

    try:
        if not args.no_viewer:
            print()
            print("Viewer opening...")
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        for step in range(args.demo_stop_step + 1):
            observation, reward, terminated, truncated, info = env.step(zero_action)
            final_info = info
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
                    f"y={info.get('y_position', 0.0):.3f}, "
                    f"x_vel={info.get('x_velocity', 0.0):.3f}, "
                    f"y_vel={info.get('y_velocity', 0.0):.3f}, "
                    f"height={info.get('base_height', 0.0):.3f}, "
                    f"up_z={info.get('up_z', 0.0):.3f}, "
                    f"motion_frame={info.get('motion_frame', 0.0):.2f}, "
                    f"reward={float(reward):.3f}, "
                    f"L=({left_contact}/exp={format_contact(left_expected)}) "
                    f"R=({right_contact}/exp={format_contact(right_expected)}) "
                    f"clr=("
                    f"{info.get('left_foot_clearance', 0.0):.3f},"
                    f"{info.get('right_foot_clearance', 0.0):.3f}"
                    f") "
                    f"residual_alpha={info.get('residual_alpha', 'NA')}"
                )

            if viewer is not None:
                viewer.sync()

            if args.real_time:
                time.sleep(args.sleep_time)

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

    print("steps shown:", steps_survived)
    print("total reward:", total_reward)
    print("final info:", final_info)

    if contact_check_count > 0:
        contact_match_rate = contact_match_count / contact_check_count
        print(
            "contact phase match rate:",
            f"{contact_match_rate:.3f}",
            f"({contact_match_count}/{contact_check_count} foot-checks)",
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
