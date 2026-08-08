from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_taskspace_right_lift_env import G1TaskspaceRightLiftEnv


ENV_KEYS = {
    "model_path",
    "frame_skip",
    "max_steps",
    "cycle_duration",
    "shift_start",
    "swing_start",
    "swing_end",
    "land_end",
    "target_clearance",
    "target_lateral_shift",
    "ik_gain",
    "ik_damping",
    "ik_max_delta",
    "xy_hold_weight",
    "z_lift_weight",
    "x_hard_limit",
    "y_hard_limit",
    "x_velocity_hard_limit",
    "y_velocity_hard_limit",
    "min_up_z",
    "min_height",
    "max_height",
    "randomize_reset",
}


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


def env_kwargs_from_cfg(cfg):
    return {k: v for k, v in cfg.items() if k in ENV_KEYS}


def rollout(cfg):
    env = G1TaskspaceRightLiftEnv(**env_kwargs_from_cfg(cfg))
    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    steps = 0
    total_reward = 0.0
    max_clear = 0.0
    min_up = 1.0
    contacts = []
    slips = []
    swing_air = []
    swing_support = []
    final = info

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total_reward += float(reward)
        max_clear = max(max_clear, float(info["main_clearance"]))
        min_up = min(min_up, float(info["up_z"]))
        contacts.append(float(info["contact_accuracy"]))
        slips.append(float(info["support_slip"]))

        # Count right-foot air only when the commanded target is visibly above ground.
        if float(info["main_target_clearance"]) >= cfg["strict_clearance"]:
            swing_air.append(not bool(info["right_contact"]))
            swing_support.append(bool(info["left_contact"]))

        final = info

    reason = env.termination_reason(final) if steps < cfg["max_steps"] else "max_steps"
    air_steps = int(sum(swing_air))
    max_air_streak = int(max_consecutive_true(swing_air))
    support_ok_ratio = float(np.mean(swing_support)) if swing_support else 0.0
    landing_ok = int(bool(final["left_contact"]) and bool(final["right_contact"]))

    strict = int(
        steps >= cfg["max_steps"]
        and reason == "max_steps"
        and max_clear >= cfg["strict_clearance"]
        and air_steps >= cfg["min_air_steps"]
        and max_air_streak >= cfg["min_air_streak"]
        and support_ok_ratio >= cfg["min_support_ratio"]
        and min_up >= cfg["strict_min_up"]
        and abs(float(final["x_position"])) <= cfg["strict_x_abs"]
        and abs(float(final["y_position"])) <= cfg["strict_y_abs"]
        and abs(float(final["x_velocity"])) <= cfg["strict_x_vel_abs"]
        and abs(float(final["y_velocity"])) <= cfg["strict_y_vel_abs"]
        and landing_ok == 1
    )

    # Score prefers real air-time. Rows with clear foot lift/contact break should rise above
    # rows that merely press upward while still in contact.
    score = (
        1400.0 * strict
        + steps
        + 12000.0 * min(max_clear, 0.050)
        + 7.0 * air_steps
        + 6.0 * max_air_streak
        + 150.0 * min_up
        + 110.0 * support_ok_ratio
        + 140.0 * landing_ok
        - 480.0 * abs(float(final["x_position"]))
        - 620.0 * abs(float(final["y_position"]))
        - 190.0 * abs(float(final["x_velocity"]))
        - 190.0 * abs(float(final["y_velocity"]))
        - 320.0 * float(np.mean(slips) if slips else 0.0)
        - (600.0 if air_steps == 0 else 0.0)
    )

    row = {
        "target_clearance": cfg["target_clearance"],
        "target_lateral_shift": cfg["target_lateral_shift"],
        "cycle_duration": cfg["cycle_duration"],
        "shift_start": cfg["shift_start"],
        "swing_start": cfg["swing_start"],
        "swing_end": cfg["swing_end"],
        "land_end": cfg["land_end"],
        "ik_gain": cfg["ik_gain"],
        "ik_damping": cfg["ik_damping"],
        "ik_max_delta": cfg["ik_max_delta"],
        "xy_hold_weight": cfg["xy_hold_weight"],
        "z_lift_weight": cfg["z_lift_weight"],
        "steps": steps,
        "reward": total_reward,
        "main_clearance": max_clear,
        "min_up_z": min_up,
        "air_steps": air_steps,
        "max_air_streak": max_air_streak,
        "support_ok_ratio": support_ok_ratio,
        "landing_ok": landing_ok,
        "contact": float(np.mean(contacts)) if contacts else 0.0,
        "support_slip": float(np.mean(slips)) if slips else 0.0,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "base_height": float(final["base_height"]),
        "root_ang_vel": float(final["root_ang_vel"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
        "strict_pass": strict,
        "score": score,
        "reason": reason,
    }
    env.close()
    return row


def print_row(r):
    print(
        f"clearT={r['target_clearance']:.3f} yT={r['target_lateral_shift']:.3f} "
        f"dur={r['cycle_duration']:.1f} sw={r['swing_start']:.2f}-{r['swing_end']:.2f} "
        f"ik={r['ik_gain']:.2f} damp={r['ik_damping']:.3f} xy={r['xy_hold_weight']:.2f} "
        f"zW={r['z_lift_weight']:.1f} | steps={r['steps']:04d} "
        f"clear={r['main_clearance']:.4f} air={r['air_steps']:03d}/{r['max_air_streak']:03d} "
        f"up={r['min_up_z']:.3f} x={r['final_x']:+.3f} y={r['final_y']:+.3f} "
        f"yv={r['final_y_velocity']:+.3f} land={r['landing_ok']} "
        f"L={r['left_contact']} R={r['right_contact']} strict={r['strict_pass']} "
        f"score={r['score']:+.1f} reason={r['reason']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fixed v2 sweep for deterministic task-space right-foot lift parameters.")
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--csv", type=str, default="results/taskspace_right_lift_sweep_v2.csv")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--shift_start", type=float, default=0.08)
    parser.add_argument("--x_hard_limit", type=float, default=0.32)
    parser.add_argument("--y_hard_limit", type=float, default=0.36)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.30)
    parser.add_argument("--y_velocity_hard_limit", type=float, default=1.30)

    parser.add_argument("--strict_clearance", type=float, default=0.030)
    parser.add_argument("--min_air_steps", type=int, default=15)
    parser.add_argument("--min_air_streak", type=int, default=8)
    parser.add_argument("--min_support_ratio", type=float, default=0.90)
    parser.add_argument("--strict_min_up", type=float, default=0.88)
    parser.add_argument("--strict_x_abs", type=float, default=0.20)
    parser.add_argument("--strict_y_abs", type=float, default=0.22)
    parser.add_argument("--strict_x_vel_abs", type=float, default=0.85)
    parser.add_argument("--strict_y_vel_abs", type=float, default=0.85)
    args = parser.parse_args()

    print("=" * 128)
    print("TASK-SPACE RIGHT-LIFT PARAMETER SWEEP V2")
    print("Fixed bug: only environment kwargs are passed to G1TaskspaceRightLiftEnv.")
    print("Goal: visible 3 cm+ right foot lift, contact break, left support, bounded y, clean landing.")
    print("=" * 128)

    rows = []

    # More conservative y shift because default yT=0.050 caused y_position_limit.
    target_clearances = [0.030, 0.035, 0.040, 0.045]
    lateral_shifts = [0.020, 0.030, 0.040]
    cycle_durations = [4.8, 5.4]
    swing_windows = [(0.38, 0.72), (0.42, 0.74)]
    ik_gains = [0.85, 1.10, 1.35]
    dampings = [0.030, 0.050]
    xy_weights = [0.05, 0.15, 0.30]
    z_weights = [1.0, 1.4]

    for clear, lat, dur, sw_win, ik, damp, xy, zw in product(
        target_clearances,
        lateral_shifts,
        cycle_durations,
        swing_windows,
        ik_gains,
        dampings,
        xy_weights,
        z_weights,
    ):
        cfg = vars(args).copy()
        cfg.update(
            {
                "target_clearance": clear,
                "target_lateral_shift": lat,
                "cycle_duration": dur,
                "swing_start": sw_win[0],
                "swing_end": sw_win[1],
                "land_end": 0.92,
                "ik_gain": ik,
                "ik_damping": damp,
                "ik_max_delta": 0.26,
                "xy_hold_weight": xy,
                "z_lift_weight": zw,
                "randomize_reset": False,
            }
        )
        row = rollout(cfg)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)

    print("\nTOP 20 BY SCORE")
    for r in rows_sorted[:20]:
        print_row(r)

    passing = [r for r in rows_sorted if int(r["strict_pass"]) == 1]
    print("\nSTRICT PASSING ROWS")
    if passing:
        for r in passing[:20]:
            print_row(r)
    else:
        print("None")

    air_rows = [r for r in rows_sorted if int(r["air_steps"]) > 0]
    print("\nBEST ROWS WITH RIGHT FOOT AIR-TIME")
    if air_rows:
        for r in air_rows[:20]:
            print_row(r)
    else:
        print("None")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV saved:", args.csv)


if __name__ == "__main__":
    main()
