import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


def main():
    parser = argparse.ArgumentParser(
        description="Test v57 gait-lift prior with zero PPO action."
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--target_velocity", type=float, default=-0.04)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--reference_start_frame", type=int, default=25)
    parser.add_argument("--height_offset", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    parser.add_argument("--reference_speed", type=float, default=0.04)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=400)
    parser.add_argument("--gait_lift_prior_scale", type=float, default=0.45)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--print_every", type=int, default=40)
    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        initial_yaw_degrees=args.initial_yaw_degrees,
        reference_start_frame=args.reference_start_frame,
        height_offset=args.height_offset,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        include_contact_phase_observation=True,
        use_reference_contact_mask=True,
        use_gait_lift_prior=True,
        gait_lift_prior_scale=args.gait_lift_prior_scale,
    )

    obs, info = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    print()
    print("=" * 100)
    print("V57 ZERO-ACTION GAIT-LIFT PRIOR TEST")
    print("=" * 100)
    print("Initial info:", info)
    print()

    total_reward = 0.0
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d}, x={info.get('x_position', 0.0):+.3f}, "
                f"y={info.get('y_position', 0.0):+.3f}, "
                f"xv={info.get('x_velocity', 0.0):+.3f}, "
                f"yv={info.get('y_velocity', 0.0):+.3f}, "
                f"h={info.get('base_height', 0.0):.3f}, "
                f"upz={info.get('up_z', 0.0):.3f}, "
                f"mf={info.get('motion_frame', 0.0):.2f}, "
                f"resA={info.get('residual_alpha', 0.0):.2f}, "
                f"L={int(info.get('left_contact', False))}/"
                f"{info.get('left_expected_contact', None)}, "
                f"R={int(info.get('right_contact', False))}/"
                f"{info.get('right_expected_contact', None)}, "
                f"clr=({info.get('left_foot_clearance', 0.0):.3f},"
                f"{info.get('right_foot_clearance', 0.0):.3f}), "
                f"coll=({int(info.get('collision_left_contact', False))},"
                f"{int(info.get('collision_right_contact', False))}), "
                f"rew={float(reward):+.3f}"
            )

        if terminated or truncated:
            break

    print()
    print("Episode ended.")
    print("steps:", step + 1)
    print("total_reward:", total_reward)
    print("final_info:", info)
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
