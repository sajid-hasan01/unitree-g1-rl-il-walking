from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v7_tiny_right_lift_env import G1WBCV7TinyRightLiftEnv


class G1WBCV7PreloadKneeBiasEnv(G1WBCV7TinyRightLiftEnv):
    """
    Clean V7 preload + tiny right-leg knee-bias lift.

    Keeps:
    - no parent swing target
    - no capture touchdown
    - no support IK lock
    - no old support push

    Adds:
    - safe lateral preload, sign=-1 by default
    - small right-leg pitch/knee/ankle swing bias
    """

    def __init__(
        self,
        preload_sign: float = -1.0,
        preload_amp: float = 0.015,
        preload_start: float = 0.08,
        preload_end: float = 0.42,
        knee_bias: float = 0.06,
        hip_ratio: float = 0.15,
        ankle_ratio: float = 0.35,
        **kwargs,
    ):
        self.preload_sign = float(preload_sign)
        self.preload_amp = float(preload_amp)
        self.preload_start = float(preload_start)
        self.preload_end = float(preload_end)

        self.knee_bias = float(knee_bias)
        self.hip_ratio = float(hip_ratio)
        self.ankle_ratio = float(ankle_ratio)

        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep_local(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _preload_env(self, phi: float) -> float:
        if phi < self.preload_start:
            return 0.0
        if phi >= self.preload_end:
            return 1.0
        return self._smoothstep_local((phi - self.preload_start) / max(self.preload_end - self.preload_start, 1e-6))

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        phi = float(info["phase"])
        preload = self._preload_env(phi)
        lift = self._tiny_lift_env(phi)

        s = self.preload_sign
        a = self.preload_amp

        # Safe lateral preload discovered by sweep.
        target[1] += s * a * preload             # left hip roll
        target[7] += s * a * preload             # right hip roll
        target[5] += -s * 0.60 * a * preload     # left ankle roll
        target[11] += -s * 0.60 * a * preload    # right ankle roll
        target[13] += s * 0.40 * a * preload     # waist roll

        # Right-leg swing shortening bias.
        # Based on the AMASS reference direction: hip pitch +, knee +, ankle pitch +.
        kb = self.knee_bias * lift
        target[6] += self.hip_ratio * kb          # right hip pitch
        target[9] += kb                           # right knee
        target[10] += self.ankle_ratio * kb       # right ankle pitch

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()
        phi = float(info["phase"])
        info["wbc_state"] = "V7_PRELOAD_KNEE_BIAS"
        info["v7_preload"] = float(self._preload_env(phi))
        info["v7_knee_bias"] = float(self.knee_bias * self._tiny_lift_env(phi))
        return info


def run_episode(args, knee_bias: float) -> dict:
    env = G1WBCV7PreloadKneeBiasEnv(
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,
        shift_start=0.08,
        swing_start=args.lift_start,
        swing_end=args.lift_peak,
        target_clearance=0.0,
        target_lateral_shift=0.0,
        ik_gain=0.0,
        z_lift_weight=0.0,
        xy_hold_weight=0.0,
        support_lock_weight=0.0,
        support_xy_weight=0.0,
        support_z_weight=0.0,
        support_ik_gain=0.0,
        tiny_lift_height=args.tiny_lift_height,
        lift_start=args.lift_start,
        lift_peak=args.lift_peak,
        land_end=args.land_end,
        foot_ik_gain=args.foot_ik_gain,
        foot_ik_damping=args.foot_ik_damping,
        foot_ik_max_delta=args.foot_ik_max_delta,
        preload_sign=args.preload_sign,
        preload_amp=args.preload_amp,
        preload_start=args.preload_start,
        preload_end=args.preload_end,
        knee_bias=knee_bias,
        hip_ratio=args.hip_ratio,
        ankle_ratio=args.ankle_ratio,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    max_left_clearance = 0.0
    min_up_z = 1.0
    max_root_ang_vel = 0.0
    max_support_slip = 0.0
    right_air_steps = 0
    right_air_streak = 0
    max_right_air_streak = 0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        final = info

        rc = float(info["right_foot_clearance"])
        lc = float(info["left_foot_clearance"])

        max_right_clearance = max(max_right_clearance, rc)
        max_left_clearance = max(max_left_clearance, lc)
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_root_ang_vel = max(max_root_ang_vel, float(info["root_ang_vel"]))
        max_support_slip = max(max_support_slip, float(info["support_slip"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            right_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, right_air_streak)
        else:
            right_air_streak = 0

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "knee_bias": knee_bias,
        "steps": steps,
        "reason": reason,
        "reward": total_reward,
        "right_clearance": max_right_clearance,
        "left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "min_up_z": min_up_z,
        "max_root_ang_vel": max_root_ang_vel,
        "support_slip": max_support_slip,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--cycle_duration", type=float, default=5.2)

    parser.add_argument("--tiny_lift_height", type=float, default=0.010)
    parser.add_argument("--foot_ik_gain", type=float, default=0.55)
    parser.add_argument("--foot_ik_damping", type=float, default=0.080)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)

    parser.add_argument("--preload_sign", type=float, default=-1.0)
    parser.add_argument("--preload_amp", type=float, default=0.015)
    parser.add_argument("--preload_start", type=float, default=0.08)
    parser.add_argument("--preload_end", type=float, default=0.42)

    parser.add_argument("--lift_start", type=float, default=0.46)
    parser.add_argument("--lift_peak", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument("--hip_ratio", type=float, default=0.15)
    parser.add_argument("--ankle_ratio", type=float, default=0.35)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v7_knee_bias_sweep.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    knee_values = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

    rows = []

    print("=" * 118)
    print("WBC V7 PRELOAD + RIGHT-KNEE-BIAS SWEEP")
    print("Best preload fixed: sign=-1 amp=0.015. Sweeping right-leg shortening bias.")
    print("=" * 118)

    for knee_bias in knee_values:
        row = run_episode(args, knee_bias)
        rows.append(row)

        print(
            f"knee={knee_bias:.3f} "
            f"steps={row['steps']:04d} reason={row['reason']:<16} "
            f"Rclear={row['right_clearance']:.4f} "
            f"Lclear={row['left_clearance']:.4f} "
            f"Rair={row['right_air_steps']:03d} "
            f"Rstreak={row['right_air_streak']:03d} "
            f"up={row['min_up_z']:.3f} "
            f"x={row['final_x']:+.3f} "
            f"y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f} "
            f"ang={row['max_root_ang_vel']:.3f} "
            f"slip={row['support_slip']:.4f}"
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
