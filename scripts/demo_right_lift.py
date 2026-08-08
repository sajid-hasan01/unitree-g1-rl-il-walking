from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO

from envs.g1_right_lift_env import G1RightLiftEnv


def main():
    parser = argparse.ArgumentParser(description="MuJoCo viewer demo for right_lift sagittal-stability policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--action_scale", type=float, default=0.14)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.6)
    parser.add_argument("--support_leg_scale", type=float, default=1.0)
    parser.add_argument("--swing_leg_scale", type=float, default=0.3)
    parser.add_argument("--waist_scale", type=float, default=1.0)
    parser.add_argument("--sagittal_kp", type=float, default=0.70)
    parser.add_argument("--sagittal_kd", type=float, default=0.12)
    parser.add_argument("--sagittal_clip", type=float, default=0.30)
    parser.add_argument("--sagittal_hip_sign", type=float, default=1.0)
    parser.add_argument("--sagittal_ankle_sign", type=float, default=1.0)
    parser.add_argument("--disable_scripted_arms", action="store_true")
    parser.add_argument("--arm_swing_scale", type=float, default=0.25)
    parser.add_argument("--arm_pitch_sign", type=float, default=1.0)
    parser.add_argument("--arm_elbow_scale", type=float, default=0.10)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = G1RightLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        support_leg_scale=args.support_leg_scale,
        swing_leg_scale=args.swing_leg_scale,
        waist_scale=args.waist_scale,
        sagittal_kp=args.sagittal_kp,
        sagittal_kd=args.sagittal_kd,
        sagittal_clip=args.sagittal_clip,
        sagittal_hip_sign=args.sagittal_hip_sign,
        sagittal_ankle_sign=args.sagittal_ankle_sign,
        frame_skip=args.frame_skip,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=None, device="auto")

    obs, info = env.reset()
    total_reward = 0.0
    max_right_clearance = 0.0
    min_up_z = 1.0

    print("=" * 100)
    print("RIGHT_LIFT SAGITTAL-STABILITY DEMO")
    print("Model:", args.model)
    print("Sagittal feedback:", "kp=", args.sagittal_kp, "kd=", args.sagittal_kd, "clip=", args.sagittal_clip)
    print("Sagittal signs:", "hip=", args.sagittal_hip_sign, "ankle=", args.sagittal_ankle_sign)
    print("Scripted arms:", "enabled=", not args.disable_scripted_arms, "scale=", args.arm_swing_scale, "pitch_sign=", args.arm_pitch_sign, "elbow=", args.arm_elbow_scale)
    print("Viewer opening...")
    print("=" * 100)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
            min_up_z = min(min_up_z, float(info["up_z"]))

            viewer.sync()

            if step % args.print_every == 0:
                print(
                    f"step={step:04d} "
                    f"x={info['x_position']:+.3f} y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} yv={info['y_velocity']:+.3f} "
                    f"h={info['base_height']:.3f} upz={info['up_z']:.3f} "
                    f"Rclr={info['right_foot_clearance']:.3f} maxR={max_right_clearance:.3f} "
                    f"Lslip={info.get('support_slip', 0.0):.3f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"phase={info['phase']:.2f} lift={info['lift_envelope']:.2f} "
                    f"comErr={info.get('com_x_error', 0.0):+.3f} "
                    f"sagCorr={info.get('sagittal_correction', 0.0):+.3f} "
                    f"backEx={info.get('backward_excess', 0.0):.3f} "
                    f"Larm={info.get('left_arm_pitch_offset', 0.0):+.2f} "
                    f"Rarm={info.get('right_arm_pitch_offset', 0.0):+.2f} "
                    f"rew={reward:+.3f}"
                )

            if terminated or truncated:
                reason = env.termination_reason(info) if terminated else "max_steps"
                print()
                print("Episode ended")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("reason:", reason)
                print("steps:", step + 1)
                print("total_reward:", total_reward)
                print("max_right_clearance:", max_right_clearance)
                print("min_up_z:", min_up_z)
                print("final_info:", info)
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    env.close()


if __name__ == "__main__":
    main()
