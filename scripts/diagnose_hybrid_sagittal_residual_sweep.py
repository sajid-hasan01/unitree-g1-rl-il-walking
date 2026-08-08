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


def make_env(args) -> G1HybridWBCLiteLiftEnv:
    return G1HybridWBCLiteLiftEnv(
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


def rollout(args, action: np.ndarray, label: str) -> Dict[str, float | str]:
    env = make_env(args)
    obs, info = env.reset()

    done = False
    steps = 0
    total_reward = 0.0
    max_clear = 0.0
    min_up = 1.0
    contacts: List[float] = []
    slips: List[float] = []
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
        final = info

    reason = env.termination_reason(final) if steps < args.max_steps else "max_steps"

    # Prefer: long survival + some clearance + small |x| and |xv|.
    score = (
        float(steps)
        + 5000.0 * min(max_clear, 0.025)
        + 120.0 * min_up
        + 100.0 * (float(np.mean(contacts)) if contacts else 0.0)
        - 500.0 * abs(float(final["x_position"]))
        - 250.0 * abs(float(final["x_velocity"]))
        - 250.0 * (float(np.mean(slips)) if slips else 0.0)
        - (200.0 if max_clear < 0.006 else 0.0)
    )

    row = {
        "label": label,
        "a0_lift": float(action[0]),
        "a1_shift": float(action[1]),
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
        "final_x_velocity": float(final["x_velocity"]),
        "score": float(score),
        "reason": reason,
    }
    env.close()
    return row


def print_row(row: Dict[str, float | str]) -> None:
    print(
        f"{row['label']:<24s} "
        f"steps={row['steps']:>5.0f} clear={row['clearance']:.4f} "
        f"up={row['min_up']:.3f} x={row['final_x']:+.3f} xv={row['final_x_velocity']:+.3f} "
        f"contact={row['contact']:.3f} score={row['score']:+.1f} reason={row['reason']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Empirical sagittal-residual sweep for the hybrid WBC-lite lift controller. "
            "This diagnoses whether high-level residual action signs can stop backward drift."
        )
    )
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--stage", type=str, default="right_lift", choices=["right_lift", "left_lift"])

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--cycle_duration", type=float, default=3.8)
    parser.add_argument("--swing_start", type=float, default=0.30)
    parser.add_argument("--swing_end", type=float, default=0.80)
    parser.add_argument("--target_clearance", type=float, default=0.010)
    parser.add_argument("--target_lateral_shift", type=float, default=0.030)
    parser.add_argument("--x_soft_limit", type=float, default=0.12)
    parser.add_argument("--x_hard_limit", type=float, default=0.35)
    parser.add_argument("--x_velocity_soft_limit", type=float, default=0.28)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.20)
    parser.add_argument("--action_smoothing", type=float, default=0.75)

    parser.add_argument("--csv", type=str, default="results/hybrid_sagittal_residual_sweep.csv")
    args = parser.parse_args()

    print("=" * 110)
    print("HYBRID WBC-LITE SAGITTAL RESIDUAL SWEEP")
    print("Action order: [lift_amp, lateral_shift, torso_bias, stance_ankle, stance_hip, swing_shape]")
    print("Goal: find residual signs that reduce backward x drift without killing clearance.")
    print("=" * 110)

    rows: List[Dict[str, float | str]] = []

    # Baseline high-level action: all residuals zero.
    zero = np.zeros(6, dtype=np.float32)
    rows.append(rollout(args, zero, "zero_action"))

    # Single-axis sign tests.
    values = [-1.0, -0.5, 0.5, 1.0]
    axes = [
        (0, "lift_amp"),
        (1, "lateral_shift"),
        (2, "torso_bias"),
        (3, "stance_ankle"),
        (4, "stance_hip"),
        (5, "swing_shape"),
    ]

    for axis, name in axes:
        for v in values:
            a = np.zeros(6, dtype=np.float32)
            a[axis] = v
            rows.append(rollout(args, a, f"{name}_{v:+.1f}"))

    # Focused sagittal combinations: torso + ankle + hip.
    combo_vals = [-1.0, 0.0, 1.0]
    for torso, ankle, hip in product(combo_vals, combo_vals, combo_vals):
        if torso == 0.0 and ankle == 0.0 and hip == 0.0:
            continue
        a = np.zeros(6, dtype=np.float32)
        a[2] = torso
        a[3] = ankle
        a[4] = hip
        rows.append(rollout(args, a, f"combo_T{torso:+.0f}_A{ankle:+.0f}_H{hip:+.0f}"))

    # A few gentler combined lift-shape tests.
    for lift_amp, swing_shape in [(-0.5, -0.5), (-0.5, 0.0), (0.0, -0.5), (0.5, -0.5)]:
        a = np.zeros(6, dtype=np.float32)
        a[0] = lift_amp
        a[5] = swing_shape
        rows.append(rollout(args, a, f"liftshape_L{lift_amp:+.1f}_S{swing_shape:+.1f}"))

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)

    print("\nTOP 12 BY SCORE")
    for r in rows_sorted[:12]:
        print_row(r)

    print("\nBASELINE AND IMPORTANT FAILURES")
    for r in rows:
        if r["label"] == "zero_action":
            print_row(r)

    # Save CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV saved:", args.csv)
    print("\nInterpretation:")
    print("- If a row has clearance >= 0.008, |x| lower than baseline, and longer steps, its residual signs are useful.")
    print("- If every row keeps x <= -0.35 or xv below -1.0, the low-level sagittal controller signs/structure are insufficient.")
    print("- If one combo improves x strongly, hard-code that bias into the hybrid controller or initialize PPO around it.")


if __name__ == "__main__":
    main()
