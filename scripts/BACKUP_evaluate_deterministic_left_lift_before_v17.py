from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_deterministic_left_lift_env import G1DeterministicLeftLiftEnv


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


def make_env(args):
    return G1DeterministicLeftLiftEnv(
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,
        target_clearance=args.target_clearance,
        com_shift_fraction=args.com_shift_fraction,
        support_gate_start=args.support_gate_start,
        com_gate_tolerance=args.com_gate_tolerance,
        gate_min_up_z=args.gate_min_up_z,
        swing_guard_x_start=args.swing_guard_x_start,
        swing_guard_x_stop=args.swing_guard_x_stop,
        swing_guard_xv_start=args.swing_guard_xv_start,
        swing_guard_xv_stop=args.swing_guard_xv_stop,
        support_lock_weight=args.support_lock_weight,
        support_xy_weight=args.support_xy_weight,
        support_z_weight=args.support_z_weight,
        support_ik_gain=args.support_ik_gain,
        support_ik_damping=args.support_ik_damping,
        support_ik_max_delta=args.support_ik_max_delta,
        swing_ik_gain=args.swing_ik_gain,
        swing_ik_damping=args.swing_ik_damping,
        swing_ik_max_delta=args.swing_ik_max_delta,
        swing_xy_hold_weight=args.swing_xy_hold_weight,
        swing_z_weight=args.swing_z_weight,
        torso_pitch_gain=args.torso_pitch_gain,
        torso_roll_gain=args.torso_roll_gain,
        angvel_pitch_gain=args.angvel_pitch_gain,
        angvel_roll_gain=args.angvel_roll_gain,
        height_gain=args.height_gain,
        height_target=args.height_target,
        x_hard_limit=args.x_hard_limit,
        y_hard_limit=args.y_hard_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        y_velocity_hard_limit=args.y_velocity_hard_limit,
        min_up_z=args.min_up_z,
    )


def diagnose(row, args):
    if row["reason"] != "max_steps":
        return f"BALANCE/TERMINATION_FAILED:{row['reason']}"

    if not row["lift_enabled"]:
        if row["right_gate_contact_ratio"] < args.min_gate_contact_ratio:
            return "RIGHT_SUPPORT_CONTACT_FAILED"
        if row["min_gate_up_z"] < args.gate_min_up_z:
            return "UPRIGHT_GATE_FAILED"
        return "COM_TRANSFER_FAILED"

    if row["main_clearance"] < args.strict_clearance:
        return "LEFT_SWING_IK_FAILED"

    if row["support_ok_ratio"] < args.min_support_ratio:
        return "RIGHT_SUPPORT_LOST_DURING_SWING"

    if row["max_support_slip"] > args.max_support_slip:
        return "RIGHT_SUPPORT_SLIP_TOO_HIGH"

    if row["min_up_z"] < args.strict_min_up:
        return "TORSO_BALANCE_FAILED"

    if row["air_steps"] < args.min_air_steps:
        return "LEFT_FOOT_DID_NOT_UNLOAD"

    if row["max_air_streak"] < args.min_air_streak:
        return "LEFT_AIR_TIME_TOO_SHORT"

    if not row["landing_ok"]:
        return "LEFT_LANDING_FAILED"

    return "PASS"


def run_episode(args, viewer=None):
    env = make_env(args)
    obs, info = env.reset()

    done = False
    steps = 0
    total = 0.0

    max_clear = 0.0
    max_right_clear = 0.0
    min_up = 1.0
    max_ang = 0.0
    max_support_slip = 0.0

    air_flags = []
    support_flags = []

    gate_contact_flags = []
    gate_up_values = []
    gate_com_errors = []

    lift_enabled_ever = False
    final = info

    while not done:
        obs, reward, terminated, truncated, info = env.step()
        done = bool(terminated or truncated)
        steps += 1
        total += float(reward)

        max_clear = max(max_clear, float(info["left_foot_clearance"]))
        max_right_clear = max(
            max_right_clear, float(info["right_foot_clearance"])
        )
        min_up = min(min_up, float(info["up_z"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))
        max_support_slip = max(
            max_support_slip, float(info["support_slip"])
        )

        lift_enabled_ever = lift_enabled_ever or bool(info["lift_enabled"])

        phi = float(info["phase"])

        # Gate inspection window: once shift should be established and before
        # the hold window ends.
        if 0.38 <= phi <= 0.68:
            gate_contact_flags.append(bool(info["right_contact"]))
            gate_up_values.append(float(info["up_z"]))
            gate_com_errors.append(abs(float(info["com_error_y"])))

        # Strict swing window: only evaluate when the controller is actually
        # requesting visible left-foot clearance.
        if float(info["main_target_clearance"]) >= args.strict_clearance:
            air_flags.append(not bool(info["left_contact"]))
            support_flags.append(bool(info["right_contact"]))

        final = info

        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, env.dt))

    reason = (
        env.termination_reason(final)
        if steps < args.max_steps
        else "max_steps"
    )

    air_steps = int(sum(air_flags))
    max_air_streak = int(max_consecutive_true(air_flags))
    support_ok_ratio = (
        float(np.mean(support_flags)) if support_flags else 0.0
    )

    right_gate_contact_ratio = (
        float(np.mean(gate_contact_flags))
        if gate_contact_flags
        else 0.0
    )
    min_gate_up_z = (
        float(np.min(gate_up_values))
        if gate_up_values
        else float(final["up_z"])
    )
    best_gate_com_error = (
        float(np.min(gate_com_errors))
        if gate_com_errors
        else float("inf")
    )

    landing_ok = int(
        bool(final["left_contact"]) and bool(final["right_contact"])
    )

    row = {
        "steps": steps,
        "reward": total,
        "main_clearance": max_clear,
        "max_right_clearance": max_right_clear,
        "min_up_z": min_up,
        "max_root_ang_vel": max_ang,
        "max_support_slip": max_support_slip,

        "air_steps": air_steps,
        "max_air_streak": max_air_streak,
        "support_ok_ratio": support_ok_ratio,

        "lift_enabled": int(lift_enabled_ever),
        "lift_enable_step": int(final["lift_enable_step"]),
        "gate_fail_reason": str(final["gate_fail_reason"]),
        "right_gate_contact_ratio": right_gate_contact_ratio,
        "min_gate_up_z": min_gate_up_z,
        "best_gate_com_error": best_gate_com_error,

        "target_com_y": float(final["target_com_y"]),
        "final_com_y": float(final["com_y"]),
        "final_com_error_y": float(final["com_error_y"]),

        "landing_ok": landing_ok,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
        "base_height": float(final["base_height"]),
        "left_contact": int(bool(final["left_contact"])),
        "right_contact": int(bool(final["right_contact"])),
        "reason": reason,
    }

    row["diagnosis"] = diagnose(row, args)
    row["strict_pass"] = int(row["diagnosis"] == "PASS")

    env.close()
    return row


def add_args(parser):
    parser.add_argument(
        "--model_path",
        type=str,
        default="third_party/mujoco_menagerie/unitree_g1/scene.xml",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--csv",
        type=str,
        default="results/deterministic_left_lift_eval.csv",
    )
    parser.add_argument("--viewer", action="store_true")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=650)
    parser.add_argument("--cycle_duration", type=float, default=5.8)

    parser.add_argument("--target_clearance", type=float, default=0.026)

    parser.add_argument("--com_shift_fraction", type=float, default=0.70)
    parser.add_argument("--support_gate_start", type=float, default=0.26)
    parser.add_argument("--com_gate_tolerance", type=float, default=0.015)
    parser.add_argument("--gate_min_up_z", type=float, default=0.90)
    parser.add_argument("--swing_guard_x_start", type=float, default=0.045)
    parser.add_argument("--swing_guard_x_stop", type=float, default=0.10)
    parser.add_argument("--swing_guard_xv_start", type=float, default=0.08)
    parser.add_argument("--swing_guard_xv_stop", type=float, default=0.20)

    parser.add_argument("--support_lock_weight", type=float, default=0.78)
    parser.add_argument("--support_xy_weight", type=float, default=0.34)
    parser.add_argument("--support_z_weight", type=float, default=1.45)
    parser.add_argument("--support_ik_gain", type=float, default=0.60)
    parser.add_argument("--support_ik_damping", type=float, default=0.065)
    parser.add_argument("--support_ik_max_delta", type=float, default=0.105)

    parser.add_argument("--swing_ik_gain", type=float, default=0.90)
    parser.add_argument("--swing_ik_damping", type=float, default=0.055)
    parser.add_argument("--swing_ik_max_delta", type=float, default=0.14)
    parser.add_argument("--swing_xy_hold_weight", type=float, default=0.08)
    parser.add_argument("--swing_z_weight", type=float, default=1.00)

    parser.add_argument("--torso_pitch_gain", type=float, default=0.16)
    parser.add_argument("--torso_roll_gain", type=float, default=0.10)
    parser.add_argument("--angvel_pitch_gain", type=float, default=0.075)
    parser.add_argument("--angvel_roll_gain", type=float, default=0.055)
    parser.add_argument("--height_gain", type=float, default=0.18)
    parser.add_argument("--height_target", type=float, default=0.790)

    parser.add_argument("--x_hard_limit", type=float, default=0.36)
    parser.add_argument("--y_hard_limit", type=float, default=0.30)
    parser.add_argument("--x_velocity_hard_limit", type=float, default=1.35)
    parser.add_argument("--y_velocity_hard_limit", type=float, default=1.35)
    parser.add_argument("--min_up_z", type=float, default=0.70)

    # Strict diagnosis thresholds.
    parser.add_argument("--strict_clearance", type=float, default=0.025)
    parser.add_argument("--min_air_steps", type=int, default=4)
    parser.add_argument("--min_air_streak", type=int, default=3)
    parser.add_argument("--min_support_ratio", type=float, default=0.95)
    parser.add_argument(
        "--min_gate_contact_ratio", type=float, default=0.95
    )
    parser.add_argument("--max_support_slip", type=float, default=0.025)
    parser.add_argument("--strict_min_up", type=float, default=0.86)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic Unitree G1 LEFT-foot lift diagnostic. "
            "No PPO / no BC / no exploration."
        )
    )
    add_args(parser)
    args = parser.parse_args()

    print("=" * 126)
    print("DETERMINISTIC G1 LEFT-FOOT LIFT DIAGNOSTIC")
    print(
        "Sequence: stand -> COM to RIGHT -> verify RIGHT support -> "
        "lift LEFT -> hold -> lower -> settle"
    )
    print(
        f"Target clearance={args.target_clearance:.3f} m | "
        f"COM fraction={args.com_shift_fraction:.2f} | "
        f"COM gate tol={args.com_gate_tolerance:.3f} m"
    )
    print("=" * 126)

    rows = []

    if args.viewer:
        import mujoco.viewer

        env = make_env(args)
        # We only need the viewer object; run_episode creates its own env, so
        # for a visual run execute the loop here against this env.
        obs, info = env.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            done = False
            max_clear = 0.0
            while not done and viewer.is_running():
                obs, reward, terminated, truncated, info = env.step()
                done = bool(terminated or truncated)
                max_clear = max(
                    max_clear, float(info["left_foot_clearance"])
                )
                viewer.sync()
                time.sleep(max(0.0, env.dt))
                if env.episode_step % 25 == 0:
                    print(
                        f"step={env.episode_step:04d} "
                        f"phi={info['phase']:.3f} "
                        f"ready={int(info['support_ready_latched'])} "
                        f"gate={int(info['lift_enabled'])} "
                        f"COMy={info['com_y']:+.4f} "
                        f"targetCOMy={info['target_com_y']:+.4f} "
                        f"COMerr={info['com_error_y']:+.4f} "
                        f"Lclear={info['left_foot_clearance']:.4f} "
                        f"Lcontact={int(info['left_contact'])} "
                        f"Lload={info['left_load_ratio']:.2f} "
                        f"LF={info['left_normal_force']:.1f} "
                        f"RF={info['right_normal_force']:.1f} "
                        f"sw={info['swing_env']:.2f} "
                        f"airOK={int(info['swing_confirmed'])} "
                        f"air={int(info['left_air_streak'])} "
                        f"pk={info['left_peak_clearance']:.3f} "
                        f"td={int(info['touchdown_latched'])} "
                        f"loadOK={int(info['recovery_load_ready'])} "
                        f"rst={int(info['recovery_load_streak'])} "
                        f"share={info['shared_support_left_load']:.2f} "
                        f"Ldes={info['recovery_desired_left_load']:.2f} "
                        f"Lcmd={info['recovery_load_cmd']:+.2f} "
                        f"rec={info['recovery_progress']:.2f} "
                        f"guard={info['swing_guard_ratio']:.2f} "
                        f"eland={int(info['emergency_landing'])} "
                        f"Rcontact={int(info['right_contact'])} "
                        f"up={info['up_z']:.3f} "
                        f"x={info['x_position']:+.4f} "
                        f"xv={info['x_velocity']:+.4f} "
                        f"Rslip={info['support_slip']:.4f}"
                    )
            print(
                "\nVisual run finished. "
                f"Max LEFT clearance={max_clear:.4f} m | "
                f"ready={int(info['support_ready_latched'])} | "
                f"gate={int(info['lift_enabled'])} | "
                f"eland={int(info['emergency_landing'])} | "
                f"gate_reason={info['gate_fail_reason']} | "
                f"termination={env.termination_reason(info)}"
            )
        env.close()
        return

    for ep in range(args.episodes):
        row = run_episode(args)
        row["episode"] = ep
        rows.append(row)

        print(
            f"ep={ep:02d} "
            f"steps={row['steps']:04d} "
            f"Lclear={row['main_clearance']:.4f} "
            f"Rclear={row['max_right_clearance']:.4f} "
            f"gate={row['lift_enabled']} "
            f"gateCOM={row['best_gate_com_error']:.4f} "
            f"gateR={row['right_gate_contact_ratio']:.2f} "
            f"up={row['min_up_z']:.3f} "
            f"air={row['air_steps']:03d}/{row['max_air_streak']:03d} "
            f"Rsupport={row['support_ok_ratio']:.2f} "
            f"Rslip={row['max_support_slip']:.4f} "
            f"land={row['landing_ok']} "
            f"x={row['final_x']:+.3f} "
            f"y={row['final_y']:+.3f} "
            f"L={row['left_contact']} "
            f"R={row['right_contact']} "
            f"PASS={row['strict_pass']} "
            f"DIAG={row['diagnosis']}"
        )

    print("\nSUMMARY")
    numeric_keys = [
        "steps",
        "reward",
        "main_clearance",
        "max_right_clearance",
        "min_up_z",
        "max_root_ang_vel",
        "max_support_slip",
        "air_steps",
        "max_air_streak",
        "support_ok_ratio",
        "right_gate_contact_ratio",
        "min_gate_up_z",
        "best_gate_com_error",
        "landing_ok",
        "final_x",
        "final_y",
        "final_x_velocity",
        "final_y_velocity",
        "base_height",
    ]

    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            print(
                f"{key:<25s} "
                f"mean={finite.mean():.4f} "
                f"std={finite.std():.4f} "
                f"min={finite.min():.4f} "
                f"max={finite.max():.4f}"
            )

    pass_rate = 100.0 * np.mean([r["strict_pass"] for r in rows])
    print(f"\nSTRICT PASS RATE: {pass_rate:.1f}%")

    diagnoses = {}
    for row in rows:
        diagnoses[row["diagnosis"]] = diagnoses.get(row["diagnosis"], 0) + 1

    print("DIAGNOSIS COUNTS:")
    for key, count in diagnoses.items():
        print(f"  {key}: {count}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)
        print("CSV saved:", args.csv)


if __name__ == "__main__":
    main()
