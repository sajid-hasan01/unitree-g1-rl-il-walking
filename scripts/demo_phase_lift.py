from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO

from envs.g1_phase_lift_env import G1PhaseLiftEnv


def main():
    parser = argparse.ArgumentParser(description="Visualize phase-based G1 lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)

    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--cycle_duration", type=float, default=3.0)
    parser.add_argument("--swing_start", type=float, default=0.35)
    parser.add_argument("--swing_end", type=float, default=0.70)
    parser.add_argument("--target_clearance", type=float, default=0.025)
    parser.add_argument("--target_lateral_shift", type=float, default=0.025)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    args = parser.parse_args()

    env = G1PhaseLiftEnv(
        model_path=args.model_path,
        stage=args.stage,
        action_scale=args.action_scale,
        cycle_duration=args.cycle_duration,
        swing_start=args.swing_start,
        swing_end=args.swing_end,
        target_clearance=args.target_clearance,
        target_lateral_shift=args.target_lateral_shift,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        action_target_smoothing=args.action_target_smoothing,
        randomize_reset=False,
    )

    model = PPO.load(args.model, env=env)

    obs, info = env.reset()
    total_reward = 0.0
    max_left_clearance = 0.0
    max_right_clearance = 0.0
    min_up_z = 1.0

    print("=" * 100)
    print("PHASE-LIFT DEMO V3")
    print("Model:", args.model)
    print("Stage:", args.stage)
    print("Viewer opening...")
    print("=" * 100)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
            max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
            min_up_z = min(min_up_z, float(info["up_z"]))

            viewer.sync()

            if step % args.print_every == 0:
                print(
                    f"step={step:04d} "
                    f"phi={info['phase']:.2f} "
                    f"x={info['x_position']:+.3f} "
                    f"y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} "
                    f"h={info['base_height']:.3f} "
                    f"upz={info['up_z']:.3f} "
                    f"Lclr={info['left_foot_clearance']:.3f}/{info['left_target_clearance']:.3f} "
                    f"Rclr={info['right_foot_clearance']:.3f}/{info['right_target_clearance']:.3f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"comErr={info['com_x_error']:+.3f} "
                    f"rew={reward:+.3f}"
                )

            if terminated or truncated:
                print("\nEpisode ended")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("reason:", env.termination_reason(info))
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    print("steps:", int(info["episode_step"]))
    print("total_reward:", total_reward)
    print("max_left_clearance:", max_left_clearance)
    print("max_right_clearance:", max_right_clearance)
    print("min_up_z:", min_up_z)
    print("final_info:", info)
    env.close()


if __name__ == "__main__":
    main()
