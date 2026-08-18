from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env_v3b import G1PhaseBCResidualEnvV3B


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_residual_ppo_v3/models/phase_bc_residual_ppo_v3a_stable_drift_20k.zip",
    )

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=600)

    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--gait_scale", type=float, default=0.22)
    parser.add_argument("--residual_scale", type=float, default=0.14)
    parser.add_argument("--target_smoothing", type=float, default=0.07)
    parser.add_argument("--transition_steps", type=int, default=240)
    parser.add_argument("--target_velocity", type=float, default=0.025)

    parser.add_argument("--print_every", type=int, default=20)

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = G1PhaseBCResidualEnvV3B(
        render_mode="human" if args.render else None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()

    print("=" * 100)
    print("PPO V3-B CONTACT-AWARE ENV DRY RUN")
    print("=" * 100)
    print("Model:", model_path)
    print("Steps:", args.steps)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target velocity:", args.target_velocity)
    print("=" * 100)

    total_reward = 0.0
    final_info = info

    both_contact_steps = 0
    left_only_steps = 0
    right_only_steps = 0
    no_contact_steps = 0

    desired_left_swing_steps = 0
    desired_right_swing_steps = 0
    actual_left_swing_steps = 0
    actual_right_swing_steps = 0

    max_x = float(info["base_x"])
    max_left_clearance = 0.0
    max_right_clearance = 0.0

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        max_x = max(max_x, float(info["base_x"]))
        max_left_clearance = max(max_left_clearance, float(info["left_clearance"]))
        max_right_clearance = max(max_right_clearance, float(info["right_clearance"]))

        left_contact = bool(info["left_contact"])
        right_contact = bool(info["right_contact"])

        if left_contact and right_contact:
            both_contact_steps += 1
        elif left_contact and not right_contact:
            left_only_steps += 1
        elif right_contact and not left_contact:
            right_only_steps += 1
        else:
            no_contact_steps += 1

        if int(info["desired_left_swing"]) == 1:
            desired_left_swing_steps += 1

        if int(info["desired_right_swing"]) == 1:
            desired_right_swing_steps += 1

        if float(info["left_clearance"]) > 0.015:
            actual_left_swing_steps += 1

        if float(info["right_clearance"]) > 0.015:
            actual_right_swing_steps += 1

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"x={info['base_x']:+.3f} "
                f"z={info['base_z']:+.3f} "
                f"up_z={info['up_z']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"DL={info['desired_left_swing']} "
                f"DR={info['desired_right_swing']} "
                f"Lc={int(info['left_contact'])} "
                f"Rc={int(info['right_contact'])} "
                f"Lclear={info['left_clearance']:.3f} "
                f"Rclear={info['right_clearance']:.3f} "
                f"matchR={info['contact_match_reward']:+.3f} "
                f"swingPen={info['swing_contact_penalty']:+.3f} "
                f"doublePen={info['double_stance_penalty']:+.3f} "
                f"reward={reward:+.3f} "
                f"fallen={terminated}"
            )

        if terminated or truncated:
            break

    print()
    print("=" * 100)
    print("V3-B DRY-RUN SUMMARY")
    print("=" * 100)
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Max x:", f"{max_x:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Total reward:", f"{total_reward:+.3f}")
    print("Terminated:", final_info["terminated"])
    print("Truncated:", final_info["truncated"])
    print()
    print("Both contact steps:", both_contact_steps)
    print("Left-only contact steps:", left_only_steps)
    print("Right-only contact steps:", right_only_steps)
    print("No contact steps:", no_contact_steps)
    print("Desired left swing steps:", desired_left_swing_steps)
    print("Desired right swing steps:", desired_right_swing_steps)
    print("Actual left clearance > 0.015 steps:", actual_left_swing_steps)
    print("Actual right clearance > 0.015 steps:", actual_right_swing_steps)
    print("Max left clearance:", f"{max_left_clearance:+.4f}")
    print("Max right clearance:", f"{max_right_clearance:+.4f}")
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
