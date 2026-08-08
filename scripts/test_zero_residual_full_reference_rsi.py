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


def initialize_from_reference(env, start_frame):
    start_frame = int(start_frame) % env.num_frames

    # Use the environment's existing RSI initializer because it sets qpos, qvel,
    # root height, joint pose, joint velocity, and actuator targets consistently.
    env.episode_step = env.initial_stand_steps + env.transition_steps
    env._apply_reference_state_initialization(start_frame)

    # Force full walking phase.
    env.episode_step = env.initial_stand_steps + env.transition_steps
    env.motion_frame = float(start_frame)
    env.rsi_active_this_episode = True
    env.rsi_frame_this_episode = int(start_frame)

    env._update_previous_foot_positions()

    left_pos = env._get_site_position(env.left_foot_site_id)
    right_pos = env._get_site_position(env.right_foot_site_id)
    env.ground_foot_height = float(min(left_pos[2], right_pos[2]))

    left_contact, right_contact = env._get_foot_contacts()

    env.last_foot_info = {
        "left_contact": bool(left_contact),
        "right_contact": bool(right_contact),
        "left_foot_slip": 0.0,
        "right_foot_slip": 0.0,
        "left_foot_clearance": float(max(left_pos[2] - env.ground_foot_height, 0.0)),
        "right_foot_clearance": float(max(right_pos[2] - env.ground_foot_height, 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-residual full-reference test. This starts directly from a walking "
            "reference frame and applies action=zeros. It bypasses the stand-to-walk "
            "blend so we can test whether the reference itself is dynamically usable."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--reference_mode", type=str, default="cyclic", choices=["transition", "cyclic"])
    parser.add_argument("--target_velocity", type=float, default=-0.10)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--action_scale", type=float, default=0.055)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.10)
    parser.add_argument("--initial_stand_steps", type=int, default=70)
    parser.add_argument("--transition_steps", type=int, default=220)
    parser.add_argument("--start_frame", type=int, default=5)
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
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    env.reset()
    initialize_from_reference(env, args.start_frame)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    left_expected, right_expected = env._get_reference_contact_for_step()

    print()
    print("=" * 90)
    print("ZERO-RESIDUAL FULL-REFERENCE RSI TEST")
    print("=" * 90)
    print("This test uses NO PPO policy.")
    print("It starts directly from a reference walking frame, then applies action=zeros.")
    print("Dataset:", args.dataset_path)
    print("Start frame:", args.start_frame)
    print("Target velocity:", args.target_velocity)
    print("Forced initial yaw degrees:", args.initial_yaw_degrees)
    print("Reference speed:", args.reference_speed)
    print("Action target smoothing:", args.action_target_smoothing)
    print("Initial stand steps:", args.initial_stand_steps)
    print("Transition steps:", args.transition_steps)
    print("Initial qpos x/y/z:", env.data.qpos[0:3].copy())
    print("Initial qvel x/y/z:", env.data.qvel[0:3].copy())
    print(
        "Initial contact:",
        f"L={env.last_foot_info['left_contact']}/exp={format_contact(left_expected)}",
        f"R={env.last_foot_info['right_contact']}/exp={format_contact(right_expected)}",
        "clearance=",
        (
            round(env.last_foot_info["left_foot_clearance"], 4),
            round(env.last_foot_info["right_foot_clearance"], 4),
        ),
    )

    total_reward = 0.0
    contact_match_count = 0
    contact_check_count = 0
    final_info = {}
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
                    f")"
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

    print("=" * 90)


if __name__ == "__main__":
    main()
