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


class G1WBCV7PreloadRightLiftEnv(G1WBCV7TinyRightLiftEnv):
    """
    Clean V7 lateral preload + tiny right-foot lift.

    It keeps the V7 clean baseline:
    - no parent swing target
    - no capture touchdown
    - no support push
    - no support IK lock

    It only adds small roll-joint preload to test which direction unloads the right foot.
    """

    def __init__(
        self,
        preload_sign: float = 1.0,
        preload_amp: float = 0.02,
        preload_start: float = 0.08,
        preload_end: float = 0.42,
        **kwargs,
    ):
        self.preload_sign = float(preload_sign)
        self.preload_amp = float(preload_amp)
        self.preload_start = float(preload_start)
        self.preload_end = float(preload_end)
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
        p = self._preload_env(phi)
        s = self.preload_sign
        a = self.preload_amp

        # Roll-joint preload sweep.
        # Sign is unknown for this model, so the evaluator tests both signs.
        target[1] += s * a * p             # left hip roll
        target[7] += s * a * p             # right hip roll
        target[5] += -s * 0.60 * a * p     # left ankle roll
        target[11] += -s * 0.60 * a * p    # right ankle roll
        target[13] += s * 0.40 * a * p     # waist roll

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()
        info["wbc_state"] = "V7_PRELOAD_RIGHT_LIFT"
        info["v7_preload"] = float(self._preload_env(float(info["phase"])))
        info["v7_preload_sign"] = float(self.preload_sign)
        info["v7_preload_amp"] = float(self.preload_amp)
        return info


def run_episode(args, sign: float, amp: float) -> dict:
    env = G1WBCV7PreloadRightLiftEnv(
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
        preload_sign=sign,
        preload_amp=amp,
        preload_start=args.preload_start,
        preload_end=args.preload_end,
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

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "sign": sign,
        "amp": amp,
        "steps": steps,
        "reason": reason,
        "reward": total_reward,
        "right_clearance": max_right_clearance,
        "left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
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

    parser.add_argument("--tiny_lift_height", type=float, default=0.005)
    parser.add_argument("--foot_ik_gain", type=float, default=0.55)
    parser.add_argument("--foot_ik_damping", type=float, default=0.080)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)

    parser.add_argument("--preload_start", type=float, default=0.08)
    parser.add_argument("--preload_end", type=float, default=0.42)

    parser.add_argument("--lift_start", type=float, default=0.46)
    parser.add_argument("--lift_peak", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v7_preload_sweep.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    signs = [-1.0, 1.0]
    amps = [0.00, 0.015, 0.030, 0.045, 0.060]

    rows = []

    print("=" * 118)
    print("WBC V7 PRELOAD SWEEP")
    print("Clean neutral baseline + lateral roll preload + 5mm right-foot vertical IK.")
    print("=" * 118)

    for sign in signs:
        for amp in amps:
            row = run_episode(args, sign, amp)
            rows.append(row)

            print(
                f"sign={sign:+.0f} amp={amp:.3f} "
                f"steps={row['steps']:04d} reason={row['reason']:<16} "
                f"Rclear={row['right_clearance']:.4f} "
                f"Lclear={row['left_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
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
