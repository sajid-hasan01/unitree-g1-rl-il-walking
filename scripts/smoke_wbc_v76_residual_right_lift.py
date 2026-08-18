from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v76_residual_right_lift_env import G1WBCV76ResidualRightLiftEnv


def make_env():
    return G1WBCV76ResidualRightLiftEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,

        shift_start=0.08,
        swing_start=0.46,
        swing_end=0.62,

        target_clearance=0.0,
        target_lateral_shift=0.0,
        ik_gain=0.0,
        z_lift_weight=0.0,
        xy_hold_weight=0.0,
        support_lock_weight=0.0,
        support_xy_weight=0.0,
        support_z_weight=0.0,
        support_ik_gain=0.0,

        tiny_lift_height=0.010,
        lift_start=0.46,
        lift_peak=0.62,
        land_end=0.82,

        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        preload_sign=-1.0,
        preload_amp=0.040,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=-0.075,

        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=0.68,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,

        swing_knee_bias=0.004,
        swing_hip_bias=0.002,
    )


def run_case(label: str, mode: str, random_scale: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = make_env()

    obs, info = env.reset(seed=seed)

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_rclear = 0.0
    min_up = 1.0
    max_slip = 0.0
    max_ang = 0.0
    max_gap_delta = 0.0
    right_air = 0
    right_air_streak = 0
    max_air_streak = 0
    total_reward = 0.0

    while not terminated and not truncated:
        if mode == "zero":
            action = np.zeros(4, dtype=np.float32)
        elif mode == "random":
            action = rng.uniform(-random_scale, random_scale, size=(4,)).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        final = info

        max_rclear = max(max_rclear, float(info["right_foot_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        max_gap_delta = max(max_gap_delta, float(info["v76_gap_delta"]))

        if not bool(info["right_contact"]):
            right_air += 1
            right_air_streak += 1
            max_air_streak = max(max_air_streak, right_air_streak)
        else:
            right_air_streak = 0

        if steps >= 520:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "label": label,
        "mode": mode,
        "steps": steps,
        "reason": reason,
        "return": total_reward,
        "right_clearance": max_rclear,
        "right_air": right_air,
        "right_air_streak": max_air_streak,
        "min_up_z": min_up,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "gap_delta": float(final.get("v76_gap_delta", 0.0)),
        "max_gap_delta": max_gap_delta,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_scale", type=float, default=0.10)
    args = parser.parse_args()

    cases = [
        ("baseline_zero", "zero", 0),
        ("random_small_0", "random", 1),
        ("random_small_1", "random", 2),
        ("random_small_2", "random", 3),
    ]

    print("=" * 132)
    print("WBC V7.6 RESIDUAL RIGHT-LIFT SMOKE TEST")
    print("Zero residual must reproduce FINAL_v72_phase_right_lift_shaped.")
    print("=" * 132)

    for label, mode, seed in cases:
        row = run_case(label, mode, args.random_scale, seed)

        print(
            f"{row['label']:<16} "
            f"mode={row['mode']:<6} "
            f"steps={row['steps']:04d} "
            f"reason={row['reason']:<16} "
            f"return={row['return']:+.2f} "
            f"Rclear={row['right_clearance']:.4f} "
            f"Rair={row['right_air']:03d} "
            f"Rstreak={row['right_air_streak']:03d} "
            f"up={row['min_up_z']:.3f} "
            f"slip={row['support_slip']:.4f} "
            f"gapDelta={row['gap_delta']:+.4f} "
            f"maxGapDelta={row['max_gap_delta']:+.4f} "
            f"x={row['final_x']:+.3f} "
            f"y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f}"
        )


if __name__ == "__main__":
    main()
