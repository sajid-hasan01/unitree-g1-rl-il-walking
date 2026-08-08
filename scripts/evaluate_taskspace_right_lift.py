from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


def parse_action(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("Action must contain exactly 4 comma-separated values.")
    return np.asarray(vals, dtype=np.float32)


def max_consecutive_true(flags):
    best = 0
    cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_episode(args, action):
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

    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    max_clear = 0.0
    min_up = 1.0
    contacts = []
    slips = []
    swing_right_air = []
    swing_left_support = []
    final = info

    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total_reward += float(reward)
        max_clear = max(max_clear, float(info["main_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        contacts.append(float(info["contact_accuracy"]))
        slips.append(float(info["support_slip"]))

        in_required_swing = float(info["main_target_clearance"]) >= args.strict_clearance
        if in_required_swing:
            swing_right_air.append(not bool(info["right_contact"]))
            swing_left_support.append(bool(info["left_contact"]))
        final = info

    reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"

    air_steps = int(sum(swing_right_air))
    max_air_streak = int(max_consecutive_true(swing_right_air))
    support_ok_ratio = float(np.mean(swing_left_support)) if swing_left_support else 0.0

    landing_ok = (
        bool(final["left_contact"])
        and bool(final["right_contact"])
        and bool(final["left_expected_contact"])
        and bool(final["right_expected_contact"])
    )

    strict_pass = int(
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
        and landing_ok
    )

    row = {
        "steps": steps,
        "reward": float(total_reward),
        "main_clearance": float(max_clear),
        "min_up_z": float(min_up),
        "contact": float(np.mean(contacts)) if contacts else 0.0,
        "support_slip": float(np.mean(slips)) if slips else 0.0,
        "air_steps": air_steps,
        "max_air_streak": max_air_streak,
        "support_ok_ratio": support_ok_ratio,
        "landing_ok": int(landing_ok),
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "base_height": float(final["base_height"]),
        "root_ang_vel": float(final["root_ang_vel"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
        "reason": reason,
        "strict_pass": strict_pass,
    }
    env.close()
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate deterministic task-space right-foot lift.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--csv", type=str, default="results/taskspace_right_lift_eval.csv")
    parser.add_argument("--fixed_action", type=str, default="0,0,0,0")

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

    parser.add_argument("--strict_clearance", type=float, default=0.030)
    parser.add_argument("--min_air_steps", type=int, default=20)
    parser.add_argument("--min_air_streak", type=int, default=12)
    parser.add_argument("--min_support_ratio", type=float, default=0.90)
    parser.add_argument("--strict_min_up", type=float, default=0.90)
    parser.add_argument("--strict_x_abs", type=float, default=0.18)
    parser.add_argument("--strict_y_abs", type=float, default=0.18)
    parser.add_argument("--strict_x_vel_abs", type=float, default=0.80)
    parser.add_argument("--strict_y_vel_abs", type=float, default=0.80)
    args = parser.parse_args()

    action = parse_action(args.fixed_action)
    print("=" * 112)
    print("TASK-SPACE RIGHT-FOOT LIFT EVALUATION")
    print("Fixed action:", action.tolist())
    print("Strict success requires visible clearance + right foot air-time + landing.")
    print("=" * 112)

    rows = []
    for ep in range(args.episodes):
        row = run_episode(args, action)
        row["episode"] = ep
        rows.append(row)
        print(
            f"ep={ep:02d} steps={row['steps']:04d} reward={row['reward']:+.1f} "
            f"clear={row['main_clearance']:.4f} up={row['min_up_z']:.3f} "
            f"air={row['air_steps']:03d} streak={row['max_air_streak']:03d} "
            f"support={row['support_ok_ratio']:.2f} land={row['landing_ok']} "
            f"x={row['final_x']:+.3f} y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} yv={row['final_y_velocity']:+.3f} "
            f"L={row['left_contact']} R={row['right_contact']} "
            f"strict={row['strict_pass']} reason={row['reason']}"
        )

    print("\nSUMMARY")
    for key in [
        "steps", "reward", "main_clearance", "min_up_z", "contact", "support_slip",
        "air_steps", "max_air_streak", "support_ok_ratio", "landing_ok",
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
