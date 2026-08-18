from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_knee_bias_sweep import G1WBCV7PreloadKneeBiasEnv


class G1WBCV7AnklePitchSweepEnv(G1WBCV7PreloadKneeBiasEnv):
    """
    V7 targeted ankle-pitch sweep.

    Fixed:
    - clean V7 controller
    - static preload for right-foot unloading
    - no knee bias
    - no hip bias

    Swept:
    - right ankle pitch during lift window

    Purpose:
    reduce toe/heel floor contact after force unloading.
    """

    def __init__(self, ankle_pitch_bias: float = 0.0, **kwargs):
        self.ankle_pitch_bias = float(ankle_pitch_bias)
        super().__init__(**kwargs)

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        lift = self._tiny_lift_env(float(info["phase"]))

        # right ankle pitch joint index = 10 in CONTROLLED_15_JOINTS
        target[10] += self.ankle_pitch_bias * lift

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target


def contact_forces(env):
    right_body = int(env.model.site_bodyid[env.right_foot_site])
    left_body = int(env.model.site_bodyid[env.left_foot_site])

    right_force = 0.0
    left_force = 0.0
    right_geom_forces = {}

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        b1 = int(env.model.geom_bodyid[g1])
        b2 = int(env.model.geom_bodyid[g2])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)
        normal_force = float(force[0])

        if b1 == right_body:
            right_force += normal_force
            right_geom_forces[g1] = right_geom_forces.get(g1, 0.0) + normal_force

        if b2 == right_body:
            right_force += normal_force
            right_geom_forces[g2] = right_geom_forces.get(g2, 0.0) + normal_force

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

    return right_force, left_force, right_geom_forces


def geom_force_text(geom_forces: dict) -> str:
    if not geom_forces:
        return "none"
    return "|".join([f"{gid}:{force:.1f}" for gid, force in sorted(geom_forces.items())])


def run_case(args, ankle_bias: float, ik_gain: float) -> dict:
    env = G1WBCV7AnklePitchSweepEnv(
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

        tiny_lift_height=args.lift_height,
        lift_start=args.lift_start,
        lift_peak=args.lift_peak,
        land_end=args.land_end,

        foot_ik_gain=ik_gain,
        foot_ik_damping=args.foot_ik_damping,
        foot_ik_max_delta=args.foot_ik_max_delta,

        preload_sign=-1.0,
        preload_amp=args.preload_amp,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=ankle_bias,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    max_left_clearance = 0.0
    min_up_z = 1.0
    max_root_ang_vel = 0.0
    max_support_slip = 0.0

    min_right_force = 1e9
    best = None

    right_air_steps = 0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])
        rc = float(info["right_foot_clearance"])
        lc = float(info["left_foot_clearance"])

        max_right_clearance = max(max_right_clearance, rc)
        max_left_clearance = max(max_left_clearance, lc)
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_root_ang_vel = max(max_root_ang_vel, float(info["root_ang_vel"]))
        max_support_slip = max(max_support_slip, float(info["support_slip"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1

        if args.lift_start <= phi <= args.land_end:
            rf, lf, rgf = contact_forces(env)

            if rf < min_right_force:
                min_right_force = rf
                total = rf + lf
                best = {
                    "phase_min_force": phi,
                    "right_force_min": rf,
                    "left_force_at_min": lf,
                    "right_force_ratio": rf / max(total, 1e-6),
                    "right_geoms": geom_force_text(rgf),
                    "right_clear_at_min": rc,
                    "left_clear_at_min": lc,
                    "up_z_at_min": float(info["up_z"]),
                    "x_at_min": float(info["x_position"]),
                    "y_at_min": float(info["y_position"]),
                }

        if steps >= args.max_steps:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    if best is None:
        best = {
            "phase_min_force": -1.0,
            "right_force_min": -1.0,
            "left_force_at_min": -1.0,
            "right_force_ratio": -1.0,
            "right_geoms": "none",
            "right_clear_at_min": -1.0,
            "left_clear_at_min": -1.0,
            "up_z_at_min": -1.0,
            "x_at_min": -1.0,
            "y_at_min": -1.0,
        }

    row = {
        "ankle_bias": ankle_bias,
        "ik_gain": ik_gain,
        "steps": steps,
        "reason": reason,
        "max_right_clearance": max_right_clearance,
        "max_left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "min_up_z": min_up_z,
        "max_root_ang_vel": max_root_ang_vel,
        "support_slip": max_support_slip,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        **best,
    }

    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--cycle_duration", type=float, default=5.2)

    parser.add_argument("--preload_amp", type=float, default=0.042)
    parser.add_argument("--lift_height", type=float, default=0.010)
    parser.add_argument("--foot_ik_max_delta", type=float, default=0.060)
    parser.add_argument("--foot_ik_damping", type=float, default=0.080)

    parser.add_argument("--lift_start", type=float, default=0.46)
    parser.add_argument("--lift_peak", type=float, default=0.62)
    parser.add_argument("--land_end", type=float, default=0.78)

    parser.add_argument(
        "--csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v7_ankle_pitch_sweep.csv",
    )

    args = parser.parse_args()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ankle_values = [-0.090, -0.085, -0.080, -0.075, -0.070, -0.065, -0.060]
    ik_gains = [0.55]

    rows = []

    print("=" * 150)
    print("WBC V7 RIGHT ANKLE-PITCH SWEEP")
    print("Fixed static preload. Sweeping ankle pitch sign to reduce toe/heel floor contact.")
    print("=" * 150)

    for ik_gain in ik_gains:
        for ankle_bias in ankle_values:
            row = run_case(args, ankle_bias, ik_gain)
            rows.append(row)

            print(
                f"gain={ik_gain:.2f} "
                f"ankle={ankle_bias:+.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['max_right_clearance']:.4f} "
                f"Lclear={row['max_left_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"RforceMin={row['right_force_min']:.2f} "
                f"Rratio={row['right_force_ratio']:.3f} "
                f"Rgeoms={row['right_geoms']:<18} "
                f"phi={row['phase_min_force']:.3f} "
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
