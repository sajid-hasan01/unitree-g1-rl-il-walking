from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO

from envs.g1_mimic_phase_lift_xlock_env import G1MimicPhaseLiftXLockEnv


def main():
    parser = argparse.ArgumentParser(description="Demo DeepMimic-lite phase lift policy.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)

    parser.add_argument("--action_scale", type=float, default=0.30)
    parser.add_argument("--cycle_duration", type=float, default=3.4)
    parser.add_argument("--swing_start", type=float, default=0.24)
    parser.add_argument("--swing_end", type=float, default=0.76)
    parser.add_argument("--target_clearance", type=float, default=0.015)
    parser.add_argument("--target_lateral_shift", type=float, default=0.015)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.30)
    parser.add_argument("--mimic_weight", type=float, default=0.18)
    parser.add_argument("--mimic_vel_weight", type=float, default=0.04)
    parser.add_argument("--mimic_all_phase", action="store_true")
    parser.add_argument("--reference_reverse", action="store_true")
    parser.add_argument("--x_soft_limit", type=float, default=0.10)
    parser.add_argument("--x_hard_limit", type=float, default=0.24)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.25)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.05)
    parser.add_argument("--no_lift_terminate_target", type=float, default=0.012)
    parser.add_argument("--no_lift_terminate_clearance", type=float, default=0.004)
    parser.add_argument("--no_lift_penalty_weight", type=float, default=55.0)
    args = parser.parse_args()

    env = G1MimicPhaseLiftXLockEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
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
        mimic_weight=args.mimic_weight,
        mimic_vel_weight=args.mimic_vel_weight,
        mimic_only_during_swing=not args.mimic_all_phase,
        reference_reverse=args.reference_reverse,
        x_soft_limit=args.x_soft_limit,
        x_hard_limit=args.x_hard_limit,
        x_velocity_soft_limit=args.x_velocity_soft_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        no_lift_terminate_target=args.no_lift_terminate_target,
        no_lift_terminate_clearance=args.no_lift_terminate_clearance,
        no_lift_penalty_weight=args.no_lift_penalty_weight,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=env)

    print("=" * 100)
    print("XLOCK-V2 MIMIC-PHASE-LIFT DEMO")
    print("Model:", args.model)
    print("Reference:", env.reference_note)
    print("Viewer opening...")
    print("=" * 100)

    obs, info = env.reset()
    total = 0.0
    max_l = 0.0
    max_r = 0.0
    min_up = 1.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            max_l = max(max_l, float(info["left_foot_clearance"]))
            max_r = max(max_r, float(info["right_foot_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            viewer.sync()

            if step % args.print_every == 0:
                print(
                    f"step={step:04d} phi={info['phase']:.2f} "
                    f"x={info['x_position']:+.3f} y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} h={info['base_height']:.3f} up={info['up_z']:.3f} "
                    f"Lclr={info['left_foot_clearance']:.3f}/{info['left_target_clearance']:.3f} "
                    f"Rclr={info['right_foot_clearance']:.3f}/{info['right_target_clearance']:.3f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"comErr={info['com_x_error']:+.3f} "
                    f"mimic={info.get('reward_mimic_total', 0.0):+.3f} "
                    f"dyn={info.get('reward_dynamics_penalty', 0.0):+.3f} "
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
    print("total_reward:", total)
    print("max_left_clearance:", max_l)
    print("max_right_clearance:", max_r)
    print("min_up_z:", min_up)
    print("final_info:", info)
    env.close()


if __name__ == "__main__":
    main()
