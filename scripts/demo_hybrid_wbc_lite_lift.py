from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO

from envs.g1_hybrid_wbc_lite_lift_env import G1HybridWBCLiteLiftEnv


def main():
    parser = argparse.ArgumentParser(description="Demo hybrid WBC-lite foot-lift controller.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--sleep_time", type=float, default=0.01)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--cycle_duration", type=float, default=3.6)
    parser.add_argument("--swing_start", type=float, default=0.28)
    parser.add_argument("--swing_end", type=float, default=0.78)
    parser.add_argument("--target_clearance", type=float, default=0.012)
    parser.add_argument("--target_lateral_shift", type=float, default=0.032)
    parser.add_argument("--x_soft_limit", type=float, default=0.08)
    parser.add_argument("--x_hard_limit", type=float, default=0.20)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.22)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.00)
    parser.add_argument("--action_smoothing", type=float, default=0.70)
    args = parser.parse_args()

    env = G1HybridWBCLiteLiftEnv(
        model_path=args.model_path,
        stage=args.stage,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,
        swing_start=args.swing_start,
        swing_end=args.swing_end,
        target_clearance=args.target_clearance,
        target_lateral_shift=args.target_lateral_shift,
        x_soft_limit=args.x_soft_limit,
        x_hard_limit=args.x_hard_limit,
        x_velocity_soft_limit=args.x_velocity_soft_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        action_smoothing=args.action_smoothing,
        randomize_reset=False,
    )
    model = PPO.load(args.model, env=env)
    print("=" * 100)
    print("HYBRID WBC-LITE DEMO")
    print("Model:", args.model)
    print("Viewer opening...")
    print("=" * 100)
    obs, info = env.reset()
    total = 0.0
    max_clear = 0.0
    min_up = 1.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            max_clear = max(max_clear, float(info["main_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            viewer.sync()
            if step % args.print_every == 0:
                print(
                    f"step={step:04d} phi={info['phase']:.2f} swing={info['swing_env']:.2f} "
                    f"x={info['x_position']:+.3f} y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} h={info['base_height']:.3f} up={info['up_z']:.3f} "
                    f"clear={info['main_clearance']:.4f}/{info['main_target_clearance']:.4f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"contact={info['contact_accuracy']:.2f} slip={info['support_slip']:.3f} rew={reward:+.3f}"
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
    print("max_clearance:", max_clear)
    print("min_up_z:", min_up)
    print("final_info:", info)
    env.close()


if __name__ == "__main__":
    main()
