from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_hybrid_wbc_lite_lift_env import G1HybridWBCLiteLiftEnv


def rollout(args, action: np.ndarray, label: str) -> Dict[str, float | str]:
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

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_clear = 0.0
    min_up = 1.0
    contacts = []
    slips = []
    final = info

    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1
        max_clear = max(max_clear, float(info["main_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        contacts.append(float(info["contact_accuracy"]))
        slips.append(float(info["support_slip"]))
        final = info

    reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"

    expected_landing_ok = True
    if bool(final["left_expected_contact"]):
        expected_landing_ok = expected_landing_ok and bool(final["left_contact"])
    if bool(final["right_expected_contact"]):
        expected_landing_ok = expected_landing_ok and bool(final["right_contact"])

    strict = int(
        steps >= args.max_steps
        and reason == "max_steps"
        and max_clear >= args.strict_clearance
        and min_up >= args.strict_min_up
        and abs(float(final["x_position"])) <= args.strict_x_abs
        and abs(float(final["y_position"])) <= args.strict_y_abs
        and abs(float(final["y_velocity"])) <= args.strict_y_velocity_abs
        and expected_landing_ok
    )

    # Score prefers clean one-cycle lift, landing, and bounded lateral state.
    score = (
        500.0 * strict
        + float(steps)
        + 9000.0 * min(max_clear, 0.020)
        + 120.0 * min_up
        + 90.0 * (float(np.mean(contacts)) if contacts else 0.0)
        - 500.0 * abs(float(final["x_position"]))
        - 650.0 * abs(float(final["y_position"]))
        - 220.0 * abs(float(final["y_velocity"]))
        - 250.0 * (float(np.mean(slips)) if slips else 0.0)
        - (250.0 if not expected_landing_ok else 0.0)
        - (300.0 if max_clear < args.strict_clearance else 0.0)
    )

    row = {
        "label": label,
        "a0_lift_amp": float(action[0]),
        "a1_lateral": float(action[1]),
        "a2_torso": float(action[2]),
        "a3_stance_ankle": float(action[3]),
        "a4_stance_hip": float(action[4]),
        "a5_swing_shape": float(action[5]),
        "steps": float(steps),
        "reward": float(total_reward),
        "clearance": float(max_clear),
        "min_up": float(min_up),
        "contact": float(np.mean(contacts)) if contacts else 0.0,
        "slip": float(np.mean(slips)) if slips else 0.0,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "target_y": float(final["target_y_offset"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
        "left_expected_contact": int(bool(final["left_expected_contact"])),
        "right_expected_contact": int(bool(final["right_expected_contact"])),
        "expected_landing_ok": int(expected_landing_ok),
        "strict_pass": strict,
        "score": float(score),
        "reason": reason,
    }
    env.close()
    return row


def print_row(row: Dict[str, float | str]) -> None:
    print(
        f"{row['label']:<26s} "
        f"steps={row['steps']:>4.0f} clear={row['clearance']:.4f} up={row['min_up']:.3f} "
        f"x={row['final_x']:+.3f} y={row['final_y']:+.3f} yv={row['final_y_velocity']:+.3f} "
        f"L={row['left_contact']}/exp={row['left_expected_contact']} "
        f"R={row['right_contact']}/exp={row['right_expected_contact']} "
        f"strict={row['strict_pass']} score={row['score']:+.1f} reason={row['reason']}"
    )


def main():
    parser = argparse.ArgumentParser(description="One-cycle positive lift/swing-shape sweep for v5 hybrid WBC-lite controller.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])
    parser.add_argument("--csv", type=str, default="results/v5_onecycle_positive_lift_sweep.csv")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=380)
    parser.add_argument("--cycle_duration", type=float, default=3.8)
    parser.add_argument("--swing_start", type=float, default=0.30)
    parser.add_argument("--swing_end", type=float, default=0.80)
    parser.add_argument("--target_clearance", type=float, default=0.008)
    parser.add_argument("--target_lateral_shift", type=float, default=0.018)
    parser.add_argument("--x_soft_limit", type=float, default=0.12)
    parser.add_argument("--x_hard_limit", type=float, default=0.35)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.28)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.20)
    parser.add_argument("--action_smoothing", type=float, default=0.75)

    parser.add_argument("--strict_x_abs", type=float, default=0.20)
    parser.add_argument("--strict_y_abs", type=float, default=0.12)
    parser.add_argument("--strict_y_velocity_abs", type=float, default=0.35)
    parser.add_argument("--strict_min_up", type=float, default=0.85)
    parser.add_argument("--strict_clearance", type=float, default=0.008)
    args = parser.parse_args()

    print("=" * 112)
    print("V5 ONE-CYCLE POSITIVE LIFT SWEEP")
    print("Reason: negative lift_amp reduces lift in this controller; this sweep tests positive lift/swing commands.")
    print("Action order: [lift_amp, lateral_shift, torso_bias, stance_ankle, stance_hip, swing_shape]")
    print("=" * 112)

    rows: List[Dict[str, float | str]] = []

    # Baseline and known failed negative commands.
    fixed_tests = [
        ("zero_action", [0, 0, 0, 0, 0, 0]),
        ("lift_amp_-1", [-1, 0, 0, 0, 0, 0]),
        ("lift_amp_+1", [1, 0, 0, 0, 0, 0]),
        ("swing_shape_+1", [0, 0, 0, 0, 0, 1]),
        ("lift+shape_+1_+1", [1, 0, 0, 0, 0, 1]),
    ]

    for label, vals in fixed_tests:
        rows.append(rollout(args, np.asarray(vals, dtype=np.float32), label))

    lift_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    shape_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    lateral_vals = [-0.25, 0.0, 0.25]
    ankle_vals = [0.0, -0.25, -0.5]
    hip_vals = [0.0, 0.25, 0.5]

    # Main grid: lift, swing shape, and small lateral trim.
    for lift, shape, lateral in product(lift_vals, shape_vals, lateral_vals):
        a = np.zeros(6, dtype=np.float32)
        a[0] = lift
        a[1] = lateral
        a[5] = shape
        rows.append(rollout(args, a, f"L{lift:+.2f}_Y{lateral:+.2f}_S{shape:+.2f}"))

    # Small sagittal trim around promising high-lift commands.
    for lift, shape, ankle, hip in product([0.5, 0.75, 1.0], [0.5, 1.0], ankle_vals, hip_vals):
        a = np.zeros(6, dtype=np.float32)
        a[0] = lift
        a[3] = ankle
        a[4] = hip
        a[5] = shape
        rows.append(rollout(args, a, f"L{lift:+.2f}_A{ankle:+.2f}_H{hip:+.2f}_S{shape:+.2f}"))

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)

    print("\nTOP 15 BY SCORE")
    for r in rows_sorted[:15]:
        print_row(r)

    strict_rows = [r for r in rows_sorted if int(r["strict_pass"]) == 1]
    print("\nSTRICT PASSING ROWS")
    if strict_rows:
        for r in strict_rows[:15]:
            print_row(r)
    else:
        print("None")

    print("\nBASELINE")
    for r in rows:
        if r["label"] in {"zero_action", "lift_amp_-1", "lift_amp_+1", "swing_shape_+1", "lift+shape_+1_+1"}:
            print_row(r)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV saved:", args.csv)


if __name__ == "__main__":
    main()
