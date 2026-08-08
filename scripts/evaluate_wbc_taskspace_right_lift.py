from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_wbc_taskspace_right_lift_env import G1WBCTaskspaceRightLiftEnv


def parse_action(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("Action must contain exactly 4 comma-separated values.")
    return np.asarray(vals, dtype=np.float32)


def max_consecutive_true(flags):
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def make_env(args):
    return G1WBCTaskspaceRightLiftEnv(
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
        support_lock_weight=args.support_lock_weight,
        support_xy_weight=args.support_xy_weight,
        support_z_weight=args.support_z_weight,
        support_ik_gain=args.support_ik_gain,
        support_ik_damping=args.support_ik_damping,
        support_ik_max_delta=args.support_ik_max_delta,
        torso_pitch_gain=args.torso_pitch_gain,
        torso_roll_gain=args.torso_roll_gain,
        angvel_pitch_gain=args.angvel_pitch_gain,
        angvel_roll_gain=args.angvel_roll_gain,
        height_gain=args.height_gain,
        height_target=args.height_target,
        x_hard_limit=args.x_hard_limit,
        y_hard_limit=args.y_hard_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        y_velocity_hard_limit=args.y_velocity_hard_limit,
        min_up_z=args.min_up_z,
        randomize_reset=False,
    )


def run_episode(args, action):
    env = make_env(args)
    obs, info = env.reset()
    done = False
    steps = 0
    total = 0.0
    max_clear = 0.0
    max_left_clear = 0.0
    min_up = 1.0
    max_ang = 0.0
    contacts = []
    slips = []
    air_flags = []
    support_flags = []
    final = info

    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total += float(reward)
        max_clear = max(max_clear, float(info["main_clearance"]))
        max_left_clear = max(max_left_clear, float(info["left_foot_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        contacts.append(float(info["contact_accuracy"]))
        slips.append(float(info["support_slip"]))

        if float(info["main_target_clearance"]) >= args.strict_clearance:
            air_flags.append(not bool(info["right_contact"]))
            support_flags.append(bool(info["left_contact"]))

        final = info

    reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"
    air_steps = int(sum(air_flags))
    max_air_streak = int(max_consecutive_true(air_flags))
    support_ok_ratio = float(np.mean(support_flags)) if support_flags else 0.0
    landing_ok = int(bool(final["left_contact"]) and bool(final["right_contact"]))

    strict = int(
        steps >= args.max_steps
        and reason == "max_steps"
        and max_clear >= args.strict_clearance
        and air_steps >= args.min_air_steps
        and max_air_streak >= args.min_air_streak
        and support_ok_ratio >= args.min_support_ratio
        and min_up >= args.strict_min_up
        and abs(float(final["x_position"])) <= args.strict_x_abs
        and abs(float(final["y_position"])) <= args.strict_y_abs
        and abs(float(final["x_velocity"])) <= args.strict_x_vel_abs
        and abs(float(final["y_velocity"])) <= args.strict_y_vel_abs
        and landing_ok == 1
    )

    env.close()
    return {
        "steps": steps,
        "reward": total,
        "main_clearance": max_clear,
        "max_left_clearance": max_left_clear,
        "min_up_z": min_up,
        "max_root_ang_vel": max_ang,
        "contact": float(np.mean(contacts)) if contacts else 0.0,
        "support_slip": float(np.mean(slips)) if slips else 0.0,
        "air_steps": air_steps,
        "max_air_streak": max_air_streak,
        "support_ok_ratio": support_ok_ratio,
        "landing_ok": landing_ok,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "base_height": float(final["base_height"]),
        "root_ang_vel": float(final["root_ang_vel"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
        "reason": reason,
        "strict_pass": strict,
    }


def add_args(parser):
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--csv", type=str, default="results/wbc_taskspace_right_lift_eval.csv")
    parser.add_argument("--fixed_action", type=str, default="0,0,0,0")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--cycle_duration", type=float, default=5.2)
    parser.add_argument("--shift_start", type=float, default=0.08)
    parser.add_argument("--swing_start", type=float, default=0.42)
    parser.add_argument("--swing_end", type=float, default=0.60)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument("--target_clearance", type=float, default=0.024)
    parser.add_argument("--target_lateral_shift", type=float, default=0.016)
    parser.add_argument("--ik_gain", type=float, default=0.92)
    parser.add_argument("--ik_damping", type=float, default=0.055)
    parser.add_argument("--ik_max_delta", type=float, default=0.14)
    parser.add_argument("--xy_hold_weight", type=float, default=0.04)
    parser.add_argument("--z_lift_weight", type=float, default=1.05)

    parser.add_argument("--support_lock_weight", type=float, default=0.75)
    parser.add_argument("--support_xy_weight", type=float, default=0.28)
    parser.add_argument("--support_z_weight", type=float, default=1.35)
    parser.add_argument("--support_ik_gain", type=float, default=0.60)
    parser.add_argument("--support_ik_damping", type=float, default=0.065)
    parser.add_argument("--support_ik_max_delta", type=float, default=0.105)

    parser.add_argument("--torso_pitch_gain", type=float, default=0.16)
    parser.add_argument("--torso_roll_gain", type=float, default=0.10)
    parser.add_argument("--angvel_pitch_gain", type=float, default=0.075)
    parser.add_argument("--angvel_roll_gain", type=float, default=0.055)
    parser.add_argument("--height_gain", type=float, default=0.18)
    parser.add_argument("--height_target", type=float, default=0.790)

    parser.add_argument("--x_hard_limit", type=float, default=0.36)
    parser.add_argument("--y_hard_limit", type=float, default=0.30)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.35)
    parser.add_argument("--y_velocity_hard_limit", type=float, default=1.35)
    parser.add_argument("--min_up_z", type=float, default=0.70)

    parser.add_argument("--strict_clearance", type=float, default=0.020)
    parser.add_argument("--min_air_steps", type=int, default=3)
    parser.add_argument("--min_air_streak", type=int, default=2)
    parser.add_argument("--min_support_ratio", type=float, default=0.90)
    parser.add_argument("--strict_min_up", type=float, default=0.86)
    parser.add_argument("--strict_x_abs", type=float, default=0.24)
    parser.add_argument("--strict_y_abs", type=float, default=0.20)
    parser.add_argument("--strict_x_vel_abs", type=float, default=0.90)
    parser.add_argument("--strict_y_vel_abs", type=float, default=0.90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WBC-lite right-foot lift with torso + support + swing tasks.")
    add_args(parser)
    args = parser.parse_args()
    action = parse_action(args.fixed_action)

    print("=" * 118)
    print("WBC-LITE TASK-SPACE RIGHT-FOOT LIFT EVALUATION")
    print("Fixed action:", action.tolist())
    print("Tasks: torso upright + pelvis height, left support lock, right swing lift, root damping.")
    print("=" * 118)

    rows = []
    for ep in range(args.episodes):
        row = run_episode(args, action)
        row["episode"] = ep
        rows.append(row)
        print(
            f"ep={ep:02d} steps={row['steps']:04d} reward={row['reward']:+.1f} "
            f"Rclear={row['main_clearance']:.4f} LclearMax={row['max_left_clearance']:.4f} "
            f"up={row['min_up_z']:.3f} air={row['air_steps']:03d}/{row['max_air_streak']:03d} "
            f"support={row['support_ok_ratio']:.2f} land={row['landing_ok']} "
            f"x={row['final_x']:+.3f} y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} yv={row['final_y_velocity']:+.3f} "
            f"ang={row['root_ang_vel']:.3f} L={row['left_contact']} R={row['right_contact']} "
            f"strict={row['strict_pass']} reason={row['reason']}"
        )

    print("\nSUMMARY")
    for key in [
        "steps", "reward", "main_clearance", "max_left_clearance", "min_up_z", "max_root_ang_vel",
        "contact", "support_slip", "air_steps", "max_air_streak", "support_ok_ratio", "landing_ok",
        "final_x", "final_y", "final_x_velocity", "final_y_velocity", "base_height", "root_ang_vel",
    ]:
        vals = np.asarray([float(r[key]) for r in rows])
        print(f"{key:<22s} mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")

    print(f"\nSTRICT PASS RATE: {100*np.mean([r['strict_pass'] for r in rows]):.1f}%")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV saved:", args.csv)


if __name__ == "__main__":
    main()
