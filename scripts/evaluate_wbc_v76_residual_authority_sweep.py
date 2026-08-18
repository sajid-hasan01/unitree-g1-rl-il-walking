from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.smoke_wbc_v76_residual_right_lift import make_env


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


def run_case(label: str, action_vec: np.ndarray) -> dict:
    env = make_env()

    obs, info = env.reset(seed=123)

    steps = 0
    terminated = False
    truncated = False
    final = info

    total_reward = 0.0

    max_right_clearance = 0.0
    max_left_clearance = 0.0
    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    right_air_steps = 0
    right_air_streak = 0
    max_right_air_streak = 0

    min_right_force = 1e9
    right_ratio_at_min = 1.0

    max_gap_delta = 0.0

    first_air_step = -1
    last_air_step = -1

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action_vec)

        total_reward += float(reward)
        steps += 1
        final = info

        phi = float(info["phase"])

        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        max_gap_delta = max(max_gap_delta, float(info["v76_gap_delta"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            right_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, right_air_streak)

            if first_air_step < 0:
                first_air_step = steps
            last_air_step = steps
        else:
            right_air_streak = 0

        if 0.46 <= phi <= 0.82:
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= 520:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "label": label,
        "a_ankle": float(action_vec[0]),
        "a_knee": float(action_vec[1]),
        "a_hip": float(action_vec[2]),
        "a_roll": float(action_vec[3]),
        "steps": steps,
        "reason": reason,
        "return": total_reward,
        "right_clearance": max_right_clearance,
        "left_clearance": max_left_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "first_air_step": first_air_step,
        "last_air_step": last_air_step,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
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
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v76_residual_authority_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        ("zero",                 [ 0.0,  0.0,  0.0,  0.0]),

        ("ankle_neg",            [-1.0,  0.0,  0.0,  0.0]),
        ("ankle_pos",            [ 1.0,  0.0,  0.0,  0.0]),

        ("knee_neg",             [ 0.0, -1.0,  0.0,  0.0]),
        ("knee_pos",             [ 0.0,  1.0,  0.0,  0.0]),

        ("hip_neg",              [ 0.0,  0.0, -1.0,  0.0]),
        ("hip_pos",              [ 0.0,  0.0,  1.0,  0.0]),

        ("roll_neg",             [ 0.0,  0.0,  0.0, -1.0]),
        ("roll_pos",             [ 0.0,  0.0,  0.0,  1.0]),

        ("ankle_neg_knee_pos",   [-1.0,  1.0,  0.0,  0.0]),
        ("ankle_neg_hip_pos",    [-1.0,  0.0,  1.0,  0.0]),
        ("ankle_neg_roll_neg",   [-1.0,  0.0,  0.0, -1.0]),
        ("ankle_neg_roll_pos",   [-1.0,  0.0,  0.0,  1.0]),

        ("knee_pos_hip_pos",     [ 0.0,  1.0,  1.0,  0.0]),
        ("knee_pos_roll_neg",    [ 0.0,  1.0,  0.0, -1.0]),
        ("knee_pos_roll_pos",    [ 0.0,  1.0,  0.0,  1.0]),
    ]

    rows = []

    print("=" * 160)
    print("WBC V7.6 RESIDUAL ACTION AUTHORITY SWEEP")
    print("Goal: find which residual channels can improve lift/gap/slip before PPO training.")
    print("=" * 160)

    for label, action in cases:
        action_vec = np.asarray(action, dtype=np.float32)
        row = run_case(label, action_vec)
        rows.append(row)

        print(
            f"{row['label']:<22} "
            f"a=[{row['a_ankle']:+.1f},{row['a_knee']:+.1f},{row['a_hip']:+.1f},{row['a_roll']:+.1f}] "
            f"steps={row['steps']:04d} "
            f"reason={row['reason']:<16} "
            f"return={row['return']:+.2f} "
            f"Rclear={row['right_clearance']:.4f} "
            f"Rair={row['right_air_steps']:03d} "
            f"Rstreak={row['right_air_streak']:03d} "
            f"air={row['first_air_step']:03d}-{row['last_air_step']:03d} "
            f"Rforce={row['right_force_min']:.2f} "
            f"up={row['min_up_z']:.3f} "
            f"slip={row['support_slip']:.4f} "
            f"gapDelta={row['gap_delta']:+.4f} "
            f"maxGap={row['max_gap_delta']:+.4f} "
            f"x={row['final_x']:+.3f} "
            f"y={row['final_y']:+.3f} "
            f"xv={row['final_x_velocity']:+.3f}"
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
