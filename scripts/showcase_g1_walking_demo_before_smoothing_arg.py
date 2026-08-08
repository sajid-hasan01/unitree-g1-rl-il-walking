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


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)

DEFAULT_CLEAN_MODEL = "models\\g1_ppo_walking_policy_final_clean_v29_best.zip"
DEFAULT_PUSH_MODEL = "models\\g1_ppo_walking_policy_v31_mild_push_100k.zip"


def format_contact(value):
    if value is None:
        return "None"
    return str(bool(value))


def yaw_to_quat_wxyz(yaw_radians):
    """
    MuJoCo free-joint quaternion order is [w, x, y, z].
    A 180 degree yaw makes the G1 body face the -X direction.
    This fixes the showcase issue where the reference root moves in -X
    while the visual body still faces +X, making the walk look backward.
    """
    half_yaw = 0.5 * float(yaw_radians)
    return np.array(
        [
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        ],
        dtype=np.float64,
    )


def build_env(args, enable_push=False):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        random_start=False,
        enable_push=enable_push,
        push_window_start=args.push_window_start,
        push_window_end=args.push_window_end,
        push_interval_min=args.push_interval_min,
        push_interval_max=args.push_interval_max,
        push_force_min=args.push_force_min,
        push_force_max=args.push_force_max,
        push_duration_steps=args.push_duration_steps,
        include_contact_phase_observation=True,
    )
    return env


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_footer():
    print("=" * 80)
    print()


def get_reference_joint_positions(env, motion_frame, demo_step, args):
    stand_joint_pos = env._get_stand_joint_positions()
    walk_joint_pos, _, _ = env._interpolate_reference(motion_frame)

    if demo_step < args.reference_initial_stand_steps:
        return stand_joint_pos

    transition_step = demo_step - args.reference_initial_stand_steps
    alpha = env._smoothstep(transition_step / max(args.reference_transition_steps, 1))

    blended_joint_pos = (1.0 - alpha) * stand_joint_pos + alpha * walk_joint_pos

    return blended_joint_pos.astype(np.float32)


def apply_reference_pose(env, motion_frame, demo_step, args):
    """
    Kinematic reference replay for showcase.

    This mode is NOT the trained RL policy.
    It visualizes the OpenHE retargeted walking reference in MuJoCo.
    It is useful for showing the imitation target used by the project.

    v2 change:
    The OpenHE segment moves mostly in the -X direction. If the root orientation
    is kept as identity, the robot can visually look like it is walking backward.
    For showcase, we yaw-rotate the base by 180 degrees by default so the robot
    visually faces the -X motion direction.
    """

    joint_pos = get_reference_joint_positions(env, motion_frame, demo_step, args)

    if env.has_reference_root_positions:
        _, _, root_pos = env._interpolate_reference(motion_frame)
        root_start = env.reference_root_positions[0]

        root_delta = root_pos - root_start

        if demo_step < args.reference_initial_stand_steps:
            blend = 0.0
        else:
            transition_step = demo_step - args.reference_initial_stand_steps
            blend = env._smoothstep(
                transition_step / max(args.reference_transition_steps, 1)
            )

        env.data.qpos[0] = float(blend * args.root_motion_scale * root_delta[0])
        env.data.qpos[1] = float(blend * args.root_motion_scale * root_delta[1])
        env.data.qpos[2] = float(
            (1.0 - blend) * env.stand_qpos[2]
            + blend * (root_pos[2] + args.height_offset)
        )
    else:
        env.data.qpos[0] = 0.0
        env.data.qpos[1] = 0.0
        env.data.qpos[2] = float(env.stand_qpos[2])

    # v2: rotate visual facing direction for reference replay.
    # Default yaw is 180 degrees, so negative-X motion looks like forward walking.
    yaw_radians = np.deg2rad(args.reference_yaw_degrees)
    quat = yaw_to_quat_wxyz(yaw_radians)

    env.data.qpos[3] = float(quat[0])
    env.data.qpos[4] = float(quat[1])
    env.data.qpos[5] = float(quat[2])
    env.data.qpos[6] = float(quat[3])

    env.data.qvel[:] = 0.0

    for i, qpos_address in enumerate(env.joint_qpos_addresses):
        env.data.qpos[qpos_address] = float(joint_pos[i])

    env.data.ctrl[:] = 0.0

    for i, actuator_id in enumerate(env.actuator_ids):
        target = env._clip_ctrl(actuator_id, joint_pos[i])
        env.data.ctrl[actuator_id] = target

    for item in env.upper_body_actuators:
        actuator_id = item["actuator_id"]
        target_qpos = item["target_qpos"]
        target = env._clip_ctrl(actuator_id, target_qpos)
        env.data.ctrl[actuator_id] = target

    mujoco.mj_forward(env.model, env.data)


def run_reference_replay(args):
    print_header("SHOWCASE MODE: REFERENCE WALKING REPLAY")

    env = build_env(args, enable_push=False)
    observation, info = env.reset()

    print("This mode shows the OpenHE retargeted walking reference in MuJoCo.")
    print("It is the imitation/reference target, not the trained PPO controller.")
    print("Dataset:", args.dataset_path)
    print("Observation shape:", env.observation_space.shape)
    print("Action shape:", env.action_space.shape)
    print("Reference frames:", env.num_frames)
    print("Reference FPS:", env.fps)
    print("Root motion scale:", args.root_motion_scale)
    print("Reference replay speed:", args.reference_replay_speed)
    print("Reference yaw degrees:", args.reference_yaw_degrees)
    print("Note: 180 degrees makes -X dataset motion look visually forward.")

    viewer = None
    motion_frame = 0.0

    contact_match_count = 0
    contact_check_count = 0

    try:
        if not args.no_viewer:
            print()
            print("Viewer opening...")
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        for step in range(args.steps):
            apply_reference_pose(env, motion_frame, step, args)

            env.motion_frame = motion_frame
            foot_info = env._get_foot_metrics()

            left_expected, right_expected = env._get_reference_contact_for_step()
            left_contact = bool(foot_info["left_contact"])
            right_contact = bool(foot_info["right_contact"])

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
                    f"x={float(env.data.qpos[0]):.3f}, "
                    f"y={float(env.data.qpos[1]):.3f}, "
                    f"height={float(env.data.qpos[2]):.3f}, "
                    f"up_z={env._get_up_z():.3f}, "
                    f"yaw_deg={args.reference_yaw_degrees:.1f}, "
                    f"motion_frame={motion_frame:.2f}, "
                    f"L=({left_contact}/exp={format_contact(left_expected)}) "
                    f"R=({right_contact}/exp={format_contact(right_expected)}) "
                    f"clr=("
                    f"{foot_info['left_foot_clearance']:.3f},"
                    f"{foot_info['right_foot_clearance']:.3f}"
                    f")"
                )

            env._update_previous_foot_positions()

            if viewer is not None:
                viewer.sync()

            if args.real_time:
                time.sleep(args.sleep_time)

            if step >= args.demo_stop_step:
                print()
                print("Reference replay stopped at showcase stop step.")
                break

            if step >= args.reference_initial_stand_steps:
                motion_frame += env.control_dt * env.fps * args.reference_replay_speed
                motion_frame = motion_frame % env.num_frames

    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    if contact_check_count > 0:
        contact_match_rate = contact_match_count / contact_check_count
        print(
            "reference contact phase match rate:",
            f"{contact_match_rate:.3f}",
            f"({contact_match_count}/{contact_check_count} foot-checks)",
        )
        print(
            "Note: this contact metric is not the main score for reference replay; "
            "reference_replay is kinematic visualization, not a trained physics policy."
        )

    print_footer()


def run_rl_policy(args, mode):
    if mode == "rl_clean":
        title = "SHOWCASE MODE: PPO CLEAN WALKING POLICY"
        model_path = args.clean_model
        enable_push = False
    elif mode == "rl_push":
        title = "SHOWCASE MODE: PPO MILD PUSH-RECOVERY POLICY"
        model_path = args.push_model
        enable_push = True
    else:
        raise ValueError(f"Unknown RL mode: {mode}")

    print_header(title)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = build_env(args, enable_push=enable_push)
    model_path = str(Path(model_path).resolve())

    print("Model:", model_path)
    print("Dataset:", args.dataset_path)
    print("Observation shape:", env.observation_space.shape)
    print("Action shape:", env.action_space.shape)
    print("Target velocity:", args.target_velocity)
    print("Action scale:", args.action_scale)
    print("Reference speed:", args.reference_speed)
    print("Initial stand steps:", args.initial_stand_steps)
    print("Transition steps:", args.transition_steps)
    print("Push enabled:", enable_push)
    print(
        "Note: RL modes use the trained policy exactly as trained. "
        "Do not yaw-rotate RL mode; that would change the policy input distribution."
    )

    model = PPO.load(model_path, device="auto")

    observation, info = env.reset()

    print("Initial info:", info)

    total_reward = 0.0
    steps_survived = 0

    contact_match_count = 0
    contact_check_count = 0

    viewer = None
    final_info = info

    try:
        if not args.no_viewer:
            print()
            print("Viewer opening...")
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        for step in range(args.steps):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(action)
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
                    f"push={info.get('push_active', False)}, "
                    f"pushN={info.get('push_force_magnitude', 0.0):.2f}, "
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

            if step >= args.demo_stop_step:
                print()
                print("Showcase stopped before late instability/fall.")
                break

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
    else:
        print("contact phase match rate: unavailable")

    print_footer()


def main():
    parser = argparse.ArgumentParser(
        description="Capstone showcase demo for Unitree G1 RL + IL walking project."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["reference_replay", "rl_clean", "rl_push"],
        help="Showcase mode.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to OpenHE processed walking dataset.",
    )

    parser.add_argument(
        "--clean_model",
        type=str,
        default=DEFAULT_CLEAN_MODEL,
        help="Path to final clean walking PPO model.",
    )

    parser.add_argument(
        "--push_model",
        type=str,
        default=DEFAULT_PUSH_MODEL,
        help="Path to final mild push-recovery PPO model.",
    )

    parser.add_argument("--target_velocity", type=float, default=-0.08)
    parser.add_argument("--action_scale", type=float, default=0.06)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.18)
    parser.add_argument("--initial_stand_steps", type=int, default=120)
    parser.add_argument("--transition_steps", type=int, default=350)

    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="Maximum number of demo steps.",
    )

    parser.add_argument(
        "--demo_stop_step",
        type=int,
        default=360,
        help=(
            "Stop demo before late instability. "
            "Use 360 for RL modes. Use 650 for reference_replay."
        ),
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Print status every N demo steps.",
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable MuJoCo viewer.",
    )

    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Run slower for recording.",
    )

    parser.add_argument(
        "--sleep_time",
        type=float,
        default=0.01,
        help="Sleep time per step when --real_time is used.",
    )

    # Reference replay options.
    parser.add_argument(
        "--reference_replay_speed",
        type=float,
        default=1.0,
        help="Reference replay speed. 1.0 means approximately normal clip speed.",
    )

    parser.add_argument(
        "--reference_initial_stand_steps",
        type=int,
        default=60,
        help="Initial standing frames for reference replay mode.",
    )

    parser.add_argument(
        "--reference_transition_steps",
        type=int,
        default=90,
        help="Blend from standing pose to reference walking pose.",
    )

    parser.add_argument(
        "--root_motion_scale",
        type=float,
        default=0.08,
        help=(
            "Scale reference root translation for visual replay. "
            "0.0 means walking in place. 0.08 gives visible forward movement."
        ),
    )

    parser.add_argument(
        "--reference_yaw_degrees",
        type=float,
        default=180.0,
        help=(
            "Yaw rotation for reference replay visual facing direction. "
            "Default 180 makes -X dataset motion look visually forward."
        ),
    )

    # Push options for rl_push mode.
    parser.add_argument("--push_window_start", type=int, default=180)
    parser.add_argument("--push_window_end", type=int, default=360)
    parser.add_argument("--push_interval_min", type=int, default=120)
    parser.add_argument("--push_interval_max", type=int, default=180)
    parser.add_argument("--push_force_min", type=float, default=5.0)
    parser.add_argument("--push_force_max", type=float, default=15.0)
    parser.add_argument("--push_duration_steps", type=int, default=3)

    args = parser.parse_args()

    if args.mode == "reference_replay":
        if args.demo_stop_step == 360:
            args.demo_stop_step = 650
        run_reference_replay(args)
    elif args.mode == "rl_clean":
        run_rl_policy(args, mode="rl_clean")
    elif args.mode == "rl_push":
        run_rl_policy(args, mode="rl_push")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
