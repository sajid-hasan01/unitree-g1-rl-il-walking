from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_phase_lift_env import G1PhaseLiftEnv


def in_window(phi: float, start: float, end: float) -> bool:
    if start <= end:
        return start <= phi < end
    return phi >= start or phi < end


def envelope(phi: float, start: float, end: float) -> float:
    if not in_window(phi, start, end):
        return 0.0
    if start <= end:
        local = (phi - start) / max(end - start, 1e-6)
    else:
        local = ((phi - start) % 1.0) / max((end - start) % 1.0, 1e-6)
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * local))


def manual_phase_action(stage: str, phi: float, swing_start: float, swing_end: float, scale: float = 1.0) -> np.ndarray:
    action = np.zeros(15, dtype=np.float32)

    if stage == "right_lift":
        right_env = envelope(phi, swing_start, swing_end)
        left_env = 0.0
        preload = min(1.0, phi / max(swing_start, 1e-6)) if phi < swing_start else 1.0
    elif stage == "left_lift":
        right_env = 0.0
        left_env = envelope(phi, swing_start, swing_end)
        preload = min(1.0, phi / max(swing_start, 1e-6)) if phi < swing_start else 1.0
    else:
        right_env = envelope(phi, 0.12, 0.38)
        left_env = envelope(phi, 0.62, 0.88)
        preload = max(right_env, left_env)

    if right_env > 0.0 or (stage == "right_lift" and phi < swing_end):
        action[1] = -0.35 * preload
        action[5] = +0.20 * preload
        action[7] = -0.25 * preload
        action[11] = +0.18 * preload
        action[13] = +0.25 * preload

    if left_env > 0.0 or (stage == "left_lift" and phi < swing_end):
        action[1] = +0.25 * preload
        action[5] = -0.18 * preload
        action[7] = +0.35 * preload
        action[11] = -0.20 * preload
        action[13] = -0.25 * preload

    if right_env > 0.0:
        action[6] += 0.45 * right_env
        action[7] += 0.18 * right_env
        action[9] += 0.95 * right_env
        action[10] += 0.65 * right_env
        action[11] += 0.15 * right_env

    if left_env > 0.0:
        action[0] += 0.45 * left_env
        action[1] += -0.18 * left_env
        action[3] += 0.95 * left_env
        action[4] += 0.65 * left_env
        action[5] += -0.15 * left_env

    return np.clip(scale * action, -1.0, 1.0).astype(np.float32)


def rollout(args, mode: str, manual_scale: float = 1.0) -> Dict[str, float]:
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

    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    max_l = 0.0
    max_r = 0.0
    min_up = 1.0
    max_abs_com = 0.0
    max_back = 0.0
    contacts = []
    slips = []
    checkpoints = {}

    while not done and steps < args.max_steps:
        phi = float(info["phase"])
        if mode == "zero":
            action = np.zeros(15, dtype=np.float32)
        elif mode == "manual":
            action = manual_phase_action(args.stage, phi, args.swing_start, args.swing_end, manual_scale)
        else:
            raise ValueError(mode)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_reward += float(reward)

        max_l = max(max_l, float(info["left_foot_clearance"]))
        max_r = max(max_r, float(info["right_foot_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        max_abs_com = max(max_abs_com, abs(float(info["com_x_error"])))
        max_back = max(max_back, float(info.get("backward_excess", 0.0)))
        contacts.append(float(info["contact_accuracy"]))
        slips.append(float(info["support_slip"]))

        if steps in (25, 40, 60, 80, 100, 125, 150, 175, 200, 250, 300):
            checkpoints[steps] = dict(info)

    reason = env.termination_reason(info) if done else "running"

    print("=" * 100)
    print(f"{mode.upper()} ROLLOUT scale={manual_scale}")
    print(
        f"steps={steps} reward={total_reward:+.1f} maxL={max_l:.4f} maxR={max_r:.4f} "
        f"minUp={min_up:.4f} contact={np.mean(contacts):.3f} slip={np.mean(slips):.3f} "
        f"maxAbsCom={max_abs_com:.4f} maxBack={max_back:.4f} "
        f"final_x={float(info['x_position']):+.4f} final_y={float(info['y_position']):+.4f} "
        f"xv={float(info['x_velocity']):+.4f} phase={float(info['phase']):.3f} reason={reason}"
    )
    for s, row in checkpoints.items():
        print(
            f"  step={s:03d} phi={float(row['phase']):.3f} "
            f"x={float(row['x_position']):+.4f} y={float(row['y_position']):+.4f}/{float(row['target_y_offset']):+.4f} "
            f"xv={float(row['x_velocity']):+.4f} up={float(row['up_z']):.4f} "
            f"Lclr={float(row['left_foot_clearance']):.4f}/{float(row['left_target_clearance']):.4f} "
            f"Rclr={float(row['right_foot_clearance']):.4f}/{float(row['right_target_clearance']):.4f} "
            f"L={int(row['left_contact'])}/exp={int(row['left_expected_contact'])} "
            f"R={int(row['right_contact'])}/exp={int(row['right_expected_contact'])} "
            f"comErr={float(row['com_x_error']):+.4f}"
        )

    env.close()
    return {
        "steps": float(steps),
        "max_r": float(max_r),
        "min_up": float(min_up),
        "final_x": float(info["x_position"]),
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose phase-lift env before more PPO training.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift", "alt_lift"])
    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--cycle_duration", type=float, default=3.0)
    parser.add_argument("--swing_start", type=float, default=0.35)
    parser.add_argument("--swing_end", type=float, default=0.70)
    parser.add_argument("--target_clearance", type=float, default=0.025)
    parser.add_argument("--target_lateral_shift", type=float, default=0.025)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    args = parser.parse_args()

    print("PHASE-LIFT DIAGNOSTIC")
    print("This checks whether zero-action standing and manual BC action are stable before PPO.")

    rollout(args, "zero", 0.0)
    rollout(args, "manual", 0.25)
    rollout(args, "manual", 0.50)
    rollout(args, "manual", 1.00)


if __name__ == "__main__":
    main()
