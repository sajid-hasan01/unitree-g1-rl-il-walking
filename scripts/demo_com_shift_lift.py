from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO

from envs.g1_com_shift_lift_env import G1ComShiftLiftEnv


def main():
    parser = argparse.ArgumentParser(description="MuJoCo viewer demo for COM-shift + lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stage", type=str, default="right_lift")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--action_scale", type=float, default=0.28)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = G1ComShiftLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        stage=args.stage,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        frame_skip=args.frame_skip,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=None, device="auto")

    obs, info = env.reset()
    total_reward = 0.0

    print("=" * 100)
    print("COM-SHIFT + LIFT DEMO")
    print("Model:", args.model)
    print("Stage:", args.stage)
    print("Viewer opening...")
    print("=" * 100)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            viewer.sync()

            if step % args.print_every == 0:
                print(
                    f"step={step:04d} stage={args.stage} "
                    f"x={info['x_position']:+.3f} y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} yv={info['y_velocity']:+.3f} "
                    f"h={info['base_height']:.3f} upz={info['up_z']:.3f} "
                    f"Lclr={info['left_foot_clearance']:.3f} Rclr={info['right_foot_clearance']:.3f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"phase={info['phase']:.2f} lift={info['lift_envelope']:.2f} "
                    f"slip={info.get('support_slip', 0.0):.3f} rew={reward:+.3f}"
                )

            if terminated or truncated:
                print()
                print("Episode ended")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("steps:", step + 1)
                print("total_reward:", total_reward)
                print("final_info:", info)
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    env.close()


if __name__ == "__main__":
    main()
