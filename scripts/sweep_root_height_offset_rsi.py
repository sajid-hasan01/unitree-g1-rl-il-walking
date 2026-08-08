import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)


def initialize_from_reference(env, start_frame):
    start_frame = int(start_frame) % env.num_frames

    env.episode_step = env.initial_stand_steps + env.transition_steps

    # Uses env.height_offset internally.
    env._apply_reference_state_initialization(start_frame)

    env.episode_step = env.initial_stand_steps + env.transition_steps
    env.motion_frame = float(start_frame)
    env.rsi_active_this_episode = True
    env.rsi_frame_this_episode = int(start_frame)

    mujoco.mj_forward(env.model, env.data)
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


def run_case(args, height_offset, start_frame):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        height_offset=height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        use_reference_contact_mask=True,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    env.reset()
    initialize_from_reference(env, start_frame)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    left_expected, right_expected = env._get_reference_contact_for_step()
    left_contact, right_contact = env._get_foot_contacts()
    left_site = env._get_site_position(env.left_foot_site_id)
    right_site = env._get_site_position(env.right_foot_site_id)

    initial = {
        "left_contact": bool(left_contact),
        "right_contact": bool(right_contact),
        "left_expected": left_expected,
        "right_expected": right_expected,
        "left_site_z": float(left_site[2]),
        "right_site_z": float(right_site[2]),
        "site_z_diff_r_minus_l": float(right_site[2] - left_site[2]),
        "root_z": float(env.data.qpos[2]),
        "ncon": int(env.data.ncon),
    }

    total_reward = 0.0
    final_info = {}
    steps = 0

    for step in range(args.rollout_steps):
        _, reward, terminated, truncated, info = env.step(zero_action)
        total_reward += float(reward)
        final_info = info
        steps = step + 1

        if terminated or truncated:
            break

    env.close()

    return initial, steps, total_reward, final_info


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep root height_offset for zero-residual full-reference RSI. "
            "This checks whether swing-foot contact and backward/forward falling are caused "
            "by root height being too low for the MuJoCo collision geoms."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--target_velocity", type=float, default=-0.08)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--reference_speed", type=float, default=0.08)
    parser.add_argument("--action_scale", type=float, default=0.060)
    parser.add_argument("--action_target_smoothing", type=float, default=0.25)
    parser.add_argument("--initial_stand_steps", type=int, default=120)
    parser.add_argument("--transition_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--rollout_steps", type=int, default=180)
    parser.add_argument("--start_frames", type=str, default="5,25")
    parser.add_argument(
        "--height_offsets",
        type=str,
        default="0.02,0.04,0.06,0.08,0.10,0.12",
    )

    args = parser.parse_args()

    start_frames = [int(x.strip()) for x in args.start_frames.split(",") if x.strip()]
    height_offsets = [float(x.strip()) for x in args.height_offsets.split(",") if x.strip()]

    print()
    print("=" * 120)
    print("ROOT HEIGHT OFFSET SWEEP: ZERO-RESIDUAL FULL-REFERENCE RSI")
    print("=" * 120)
    print("Dataset:", args.dataset_path)
    print("Target velocity:", args.target_velocity)
    print("Yaw degrees:", args.initial_yaw_degrees)
    print("Reference speed:", args.reference_speed)
    print("Rollout steps per case:", args.rollout_steps)
    print("Start frames:", start_frames)
    print("Height offsets:", height_offsets)
    print()
    print(
        "height | frame | init L/R(exp) | init site z L/R | init ncon | "
        "survive | final x | final y | final xv | final yv | height | up_z | "
        "L/R contact | L/R expected | L/R clr | slip L/R"
    )
    print("-" * 120)

    best = None

    for height_offset in height_offsets:
        for start_frame in start_frames:
            initial, steps, total_reward, info = run_case(args, height_offset, start_frame)

            score = (
                steps
                + 150.0 * max(float(info.get("up_z", 0.0)) - 0.94, 0.0)
                - 100.0 * abs(float(info.get("y_position", 0.0)))
                - 20.0 * max(abs(float(info.get("x_velocity", 0.0))) - 0.20, 0.0)
            )

            row = (
                f"{height_offset:6.3f} | "
                f"{start_frame:5d} | "
                f"{int(initial['left_contact'])}/{int(initial['right_contact'])}"
                f"({initial['left_expected']},{initial['right_expected']}) | "
                f"{initial['left_site_z']:.3f}/{initial['right_site_z']:.3f} | "
                f"{initial['ncon']:9d} | "
                f"{steps:7d} | "
                f"{float(info.get('x_position', 0.0)):+.3f} | "
                f"{float(info.get('y_position', 0.0)):+.3f} | "
                f"{float(info.get('x_velocity', 0.0)):+.3f} | "
                f"{float(info.get('y_velocity', 0.0)):+.3f} | "
                f"{float(info.get('base_height', 0.0)):.3f} | "
                f"{float(info.get('up_z', 0.0)):.3f} | "
                f"{int(bool(info.get('left_contact', False)))}/"
                f"{int(bool(info.get('right_contact', False)))} | "
                f"{info.get('left_expected_contact', None)}/"
                f"{info.get('right_expected_contact', None)} | "
                f"{float(info.get('left_foot_clearance', 0.0)):.3f}/"
                f"{float(info.get('right_foot_clearance', 0.0)):.3f} | "
                f"{float(info.get('left_foot_slip', 0.0)):.3f}/"
                f"{float(info.get('right_foot_slip', 0.0)):.3f}"
            )
            print(row)

            candidate = {
                "score": score,
                "height_offset": height_offset,
                "start_frame": start_frame,
                "steps": steps,
                "info": info,
                "initial": initial,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    print("-" * 120)
    print("Best rough case:")
    print(
        f"  height_offset={best['height_offset']}, start_frame={best['start_frame']}, "
        f"steps={best['steps']}, score={best['score']:.2f}"
    )
    print("  final info:", best["info"])
    print()
    print("Interpretation:")
    print("  - If higher height_offset clearly removes unwanted swing-foot contacts and improves survival, use that offset in v53.")
    print("  - If all offsets still keep both feet stuck or fall early, the issue is actuator/collision dynamics, not PPO.")
    print("=" * 120)


if __name__ == "__main__":
    main()
