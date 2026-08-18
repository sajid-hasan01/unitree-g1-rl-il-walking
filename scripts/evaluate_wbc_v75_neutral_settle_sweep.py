from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


OLD_DURATION = 5.2


def old_phi_to_new_phi(old_phi: float, new_duration: float) -> float:
    return float(old_phi * OLD_DURATION / new_duration)


def contact_forces(env):
    right_body = int(env.model.site_bodyid[env.right_foot_site])
    left_body = int(env.model.site_bodyid[env.left_foot_site])

    right_force = 0.0
    left_force = 0.0

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        b1 = int(env.model.geom_bodyid[int(c.geom1)])
        b2 = int(env.model.geom_bodyid[int(c.geom2)])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == right_body or b2 == right_body:
            right_force += normal_force

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

    return right_force, left_force


def make_env(
    total_duration: float,
    max_steps: int,
    preload_amp: float,
    ankle_pitch_bias: float,
    lift_hold_old_phi: float,
):
    p = lambda old_phi: old_phi_to_new_phi(old_phi, total_duration)

    return G1WBCV72PhaseRightLiftEnv(
        frame_skip=5,
        max_steps=max_steps,
        cycle_duration=total_duration,

        shift_start=p(0.08),
        swing_start=p(0.46),
        swing_end=p(0.62),

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
        lift_start=p(0.46),
        lift_peak=p(0.62),
        land_end=p(0.82),

        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,

        preload_sign=-1.0,
        preload_amp=preload_amp,
        preload_start=p(0.08),
        preload_end=p(0.42),

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=ankle_pitch_bias,

        lift_ramp_start=p(0.46),
        lift_ramp_end=p(0.56),
        lift_hold_end=p(lift_hold_old_phi),
        lift_land_end=p(0.82),
        preload_down_start=p(0.82),
        preload_down_end=p(1.00),

        swing_knee_bias=0.004,
        swing_hip_bias=0.002,
    )


def run_case(
    label: str,
    total_duration: float,
    preload_amp: float,
    ankle_pitch_bias: float,
    lift_hold_old_phi: float,
) -> dict:
    max_steps = int(round(total_duration * 100))

    env = make_env(
        total_duration=total_duration,
        max_steps=max_steps,
        preload_amp=preload_amp,
        ankle_pitch_bias=ankle_pitch_bias,
        lift_hold_old_phi=lift_hold_old_phi,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    initial_gap = abs(
        float(env.data.site_xpos[env.right_foot_site][1])
        - float(env.data.site_xpos[env.left_foot_site][1])
    )

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    right_air_steps = 0
    right_air_streak = 0
    max_right_air_streak = 0

    min_right_force = 1e9
    right_ratio_at_min = 1.0

    gap_at_old_end = initial_gap

    old_end_new_phi = old_phi_to_new_phi(1.00, total_duration)

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])

        current_gap = abs(
            float(env.data.site_xpos[env.right_foot_site][1])
            - float(env.data.site_xpos[env.left_foot_site][1])
        )

        if abs(phi - old_end_new_phi) < 0.004:
            gap_at_old_end = current_gap

        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            right_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, right_air_streak)
        else:
            right_air_streak = 0

        if old_phi_to_new_phi(0.46, total_duration) <= phi <= old_phi_to_new_phi(0.82, total_duration):
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= max_steps:
            break

    final_gap = abs(
        float(env.data.site_xpos[env.right_foot_site][1])
        - float(env.data.site_xpos[env.left_foot_site][1])
    )

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "label": label,
        "total_duration": total_duration,
        "max_steps": max_steps,
        "preload_amp": preload_amp,
        "ankle_pitch_bias": ankle_pitch_bias,
        "lift_hold_old_phi": lift_hold_old_phi,
        "steps": steps,
        "reason": reason,
        "right_clearance": max_right_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "initial_gap": initial_gap,
        "gap_at_old_end": gap_at_old_end,
        "final_gap": final_gap,
        "gap_delta": final_gap - initial_gap,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "final_phase": str(final.get("v72_phase_name", "unknown")),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v75_neutral_settle_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "label": "lower_gap",
            "preload_amp": 0.036,
            "ankle_pitch_bias": -0.070,
            "lift_hold_old_phi": 0.64,
        },
        {
            "label": "higher_air",
            "preload_amp": 0.040,
            "ankle_pitch_bias": -0.075,
            "lift_hold_old_phi": 0.68,
        },
    ]

    total_durations = [5.2, 6.0, 6.5, 7.0]

    rows = []

    print("=" * 160)
    print("WBC V7.5 NEUTRAL-SETTLE SWEEP")
    print("Goal: add settle time after return-neutral and check whether final leg gap reduces naturally.")
    print("=" * 160)

    for cfg in configs:
        for total_duration in total_durations:
            row = run_case(
                label=cfg["label"],
                total_duration=total_duration,
                preload_amp=cfg["preload_amp"],
                ankle_pitch_bias=cfg["ankle_pitch_bias"],
                lift_hold_old_phi=cfg["lift_hold_old_phi"],
            )

            rows.append(row)

            print(
                f"{row['label']:<10} "
                f"T={row['total_duration']:.1f}s "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['right_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"Rstreak={row['right_air_streak']:03d} "
                f"Rforce={row['right_force_min']:.2f} "
                f"up={row['min_up_z']:.3f} "
                f"slip={row['support_slip']:.4f} "
                f"gapOldEnd={row['gap_at_old_end']:.4f} "
                f"gapFinal={row['final_gap']:.4f} "
                f"gapDelta={row['gap_delta']:+.4f} "
                f"x={row['final_x']:+.3f} "
                f"y={row['final_y']:+.3f} "
                f"xv={row['final_x_velocity']:+.3f} "
                f"phase={row['final_phase']}"
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
