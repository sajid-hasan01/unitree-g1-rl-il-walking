from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


def parse_action(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("Action must contain exactly 4 comma-separated values.")
    return np.asarray(vals, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visual demo for deterministic task-space right-foot lift.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--fixed_action", type=str, default="0,0,0,0")
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=480)
    parser.add_argument("--cycle_duration", type=float, default=4.8)
    parser.add_argument("--shift_start", type=float, default=0.08)
    parser.add_argument("--swing_start", type=float, default=0.34)
    parser.add_argument("--swing_end", type=float, default=0.74)
    parser.add_argument("--land_end", type=float, default=0.92)

    parser.add_argument("--target_clearance", type=float, default=0.040)
    parser.add_argument("--target_lateral_shift", type=float, default=0.050)
    parser.add_argument("--ik_gain", type=float, default=0.90)
    parser.add_argument("--ik_damping", type=float, default=0.045)
    parser.add_argument("--ik_max_delta", type=float, default=0.22)
    parser.add_argument("--xy_hold_weight", type=float, default=0.30)
    parser.add_argument("--z_lift_weight", type=float, default=1.00)

    parser.add_argument("--x_hard_limit", type=float, default=0.30)
    parser.add_argument("--y_hard_limit", type=float, default=0.28)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.20)
    parser.add_argument("--y_velocity_hard_limit", type=float, default=1.20)
    args = parser.parse_args()

    action = parse_action(args.fixed_action)

    env = G1TaskspaceRightLiftEnv(
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,
        shift_start=args.shift_start,
        swing_start=args.swing_start,
        swing_end=args.swing_end,
        land_end=args.land_end,
        target_clearance=args.target_clearance,
        target_lateral_shift=args.target_lateral_shift,
        ik_gain=args.ik_gain,
        ik_damping=args.ik_damping,
        ik_max_delta=args.ik_max_delta,
        xy_hold_weight=args.xy_hold_weight,
        z_lift_weight=args.z_lift_weight,
        x_hard_limit=args.x_hard_limit,
        y_hard_limit=args.y_hard_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        y_velocity_hard_limit=args.y_velocity_hard_limit,
        randomize_reset=False,
    )

    print("=" * 112)
    print("TASK-SPACE RIGHT-FOOT LIFT DEMO")
    print("Fixed action:", action.tolist())
    print("Viewer opening...")
    print("=" * 112)

    obs, info = env.reset()
    total_reward = 0.0
    max_clear = 0.0
    min_up = 1.0
    air_steps = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break

            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()

            total_reward += float(reward)
            max_clear = max(max_clear, float(info["main_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            if float(info["main_target_clearance"]) >= 0.030 and not bool(info["right_contact"]):
                air_steps += 1

            if step % args.print_every == 0:
                print(
                    f"step={step:04d} phi={info['phase']:.2f} swing={info['swing_env']:.2f} "
                    f"shift={info['shift_env']:.2f} x={info['x_position']:+.3f} "
                    f"y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} yv={info['y_velocity']:+.3f} "
                    f"h={info['base_height']:.3f} up={info['up_z']:.3f} "
                    f"Rclr={info['right_foot_clearance']:.4f}/{info['main_target_clearance']:.4f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"contact={info['contact_accuracy']:.2f} slip={info['support_slip']:.3f} "
                    f"rew={reward:+.3f}"
                )

            if terminated or truncated:
                print("\nEpisode ended")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("reason:", env.termination_reason(info) if terminated else "max_steps")
                break

            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    print("\nRESULT")
    print("steps:", int(info["episode_step"]))
    print("total_reward:", total_reward)
    print("max_right_clearance:", max_clear)
    print("min_up_z:", min_up)
    print("air_steps_when_target_clearance_ge_3cm:", air_steps)
    print("final_info:", info)
    env.close()


if __name__ == "__main__":
    main()
