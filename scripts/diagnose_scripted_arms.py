from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco
import numpy as np

from envs.g1_right_lift_env import G1RightLiftEnv


ARM_KEYWORDS = ("shoulder", "elbow", "wrist")


def name(model, obj_type, idx: int) -> str:
    value = mujoco.mj_id2name(model, obj_type, idx)
    return value if value is not None else ""


def find_arm_actuators(env: G1RightLiftEnv) -> List[Dict[str, object]]:
    rows = []
    for actuator_id in range(env.model.nu):
        joint_id = int(env.model.actuator_trnid[actuator_id, 0])
        actuator_name = name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""

        combined = f"{actuator_name} {joint_name}".lower()
        if any(k in combined for k in ARM_KEYWORDS):
            qadr = int(env.model.jnt_qposadr[joint_id]) if joint_id >= 0 else -1
            rows.append(
                {
                    "actuator_id": actuator_id,
                    "actuator_name": actuator_name,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "qpos_adr": qadr,
                    "ctrlrange": tuple(float(x) for x in env.model.actuator_ctrlrange[actuator_id]),
                }
            )
    return rows


def print_arm_table(rows: List[Dict[str, object]]) -> None:
    print("=" * 100)
    print("ARM ACTUATOR / JOINT MAP")
    print("=" * 100)
    if not rows:
        print("No arm actuators found using keywords:", ARM_KEYWORDS)
        return

    for r in rows:
        print(
            f"act={r['actuator_id']:02d} "
            f"act_name={r['actuator_name']:<32s} "
            f"joint={r['joint_name']:<32s} "
            f"qpos={r['qpos_adr']:<3d} "
            f"ctrlrange={r['ctrlrange']}"
        )


def direct_arm_motion_test(env: G1RightLiftEnv, rows: List[Dict[str, object]], offset: float, hold_steps: int) -> None:
    print()
    print("=" * 100)
    print("DIRECT ARM ACTUATOR MOTION TEST")
    print("Applies +offset then -offset to each arm actuator from stand.")
    print("If qpos_delta is near zero, that actuator/control mapping is not moving.")
    print("=" * 100)

    for r in rows:
        actuator_id = int(r["actuator_id"])
        qadr = int(r["qpos_adr"])
        joint_name = str(r["joint_name"])

        if qadr < 0:
            continue

        def run_one(sign: float):
            env.reset()
            mujoco.mj_forward(env.model, env.data)

            q0 = float(env.data.qpos[qadr])
            x0 = float(env.data.qpos[0])
            up0 = float(env._root_up_z())
            ctrl0 = float(env.data.ctrl[actuator_id])

            lo, hi = env.model.actuator_ctrlrange[actuator_id]
            target = float(np.clip(q0 + sign * offset, lo, hi))

            for _ in range(hold_steps):
                # Hold all actuators at their current stand/default values.
                for aid in range(env.model.nu):
                    jid = int(env.model.actuator_trnid[aid, 0])
                    if jid >= 0:
                        qa = int(env.model.jnt_qposadr[jid])
                        env.data.ctrl[aid] = env.stand_qpos[qa]

                env.data.ctrl[actuator_id] = target
                mujoco.mj_step(env.model, env.data)

            q1 = float(env.data.qpos[qadr])
            x1 = float(env.data.qpos[0])
            up1 = float(env._root_up_z())

            return {
                "sign": sign,
                "ctrl0": ctrl0,
                "target": target,
                "q0": q0,
                "q1": q1,
                "q_delta": q1 - q0,
                "x_delta": x1 - x0,
                "up_delta": up1 - up0,
            }

        plus = run_one(+1.0)
        minus = run_one(-1.0)

        print(f"\n{joint_name}")
        for label, result in [("+offset", plus), ("-offset", minus)]:
            print(
                f"{label:8s} target={result['target']:+.4f} "
                f"q {result['q0']:+.4f}->{result['q1']:+.4f} "
                f"q_delta={result['q_delta']:+.6f} "
                f"x_delta={result['x_delta']:+.6f} "
                f"up_delta={result['up_delta']:+.6f}"
            )


def scripted_arm_rollout_compare(args) -> None:
    print()
    print("=" * 100)
    print("SCRIPTED ARM ROLLOUT COMPARISON")
    print("Runs zero-action teacher rollouts with arms disabled/enabled.")
    print("If rows are identical, scripted arms are not being applied or are dynamically negligible.")
    print("=" * 100)

    configs = [
        ("arms_off", True, 0.0, 1.0),
        ("arms_pos_025", False, 0.25, 1.0),
        ("arms_neg_025", False, 0.25, -1.0),
        ("arms_pos_040", False, 0.40, 1.0),
        ("arms_neg_040", False, 0.40, -1.0),
    ]

    for label, disable, scale, sign in configs:
        env = G1RightLiftEnv(
            action_scale=args.action_scale,
            teacher_scale_multiplier=args.teacher_scale_multiplier,
            support_leg_scale=1.0,
            swing_leg_scale=args.swing_leg_scale,
            waist_scale=args.waist_scale,
            sagittal_kp=0.0,
            sagittal_kd=0.0,
            sagittal_clip=0.0,
            enable_scripted_arms=not disable,
            arm_swing_scale=scale,
            arm_pitch_sign=sign,
            arm_elbow_scale=args.arm_elbow_scale,
            randomize_reset=False,
        )

        obs, info = env.reset()
        done = False
        steps = 0
        max_r = 0.0
        min_up = 1.0
        final = info
        checkpoints = {}

        while not done and steps < args.max_steps:
            obs, reward, terminated, truncated, info = env.step(np.zeros(15, dtype=np.float32))
            done = terminated or truncated
            steps += 1
            max_r = max(max_r, float(info["right_foot_clearance"]))
            min_up = min(min_up, float(info["up_z"]))
            final = info

            if steps in (200, 225, 250, 275):
                checkpoints[steps] = {
                    "x": float(info["x_position"]),
                    "vx": float(info["x_velocity"]),
                    "up_z": float(info["up_z"]),
                    "rclr": float(info["right_foot_clearance"]),
                    "larm": float(info.get("left_arm_pitch_offset", 0.0)),
                    "rarm": float(info.get("right_arm_pitch_offset", 0.0)),
                }

        print(
            f"\n{label}: steps={steps} maxR={max_r:.4f} minUp={min_up:.4f} "
            f"final_x={float(final['x_position']):+.4f} final_y={float(final['y_position']):+.4f} "
            f"reason={env.termination_reason(final) if done else 'running'}"
        )
        for step, row in checkpoints.items():
            print(
                f"  step={step:03d} x={row['x']:+.4f} vx={row['vx']:+.4f} "
                f"up={row['up_z']:.4f} Rclr={row['rclr']:.4f} "
                f"Larm={row['larm']:+.4f} Rarm={row['rarm']:+.4f}"
            )

        env.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether scripted arms are actually moving and affecting G1.")
    parser.add_argument("--offset", type=float, default=0.30)
    parser.add_argument("--hold_steps", type=int, default=80)
    parser.add_argument("--max_steps", type=int, default=350)
    parser.add_argument("--action_scale", type=float, default=0.08)
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.35)
    parser.add_argument("--swing_leg_scale", type=float, default=0.1)
    parser.add_argument("--waist_scale", type=float, default=0.8)
    parser.add_argument("--arm_elbow_scale", type=float, default=0.10)
    args = parser.parse_args()

    env = G1RightLiftEnv(
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        support_leg_scale=1.0,
        swing_leg_scale=args.swing_leg_scale,
        waist_scale=args.waist_scale,
        sagittal_kp=0.0,
        sagittal_kd=0.0,
        sagittal_clip=0.0,
        enable_scripted_arms=True,
        arm_swing_scale=0.25,
        arm_pitch_sign=1.0,
        arm_elbow_scale=args.arm_elbow_scale,
        randomize_reset=False,
    )

    try:
        rows = find_arm_actuators(env)
        print_arm_table(rows)
        direct_arm_motion_test(env, rows, args.offset, args.hold_steps)
    finally:
        env.close()

    scripted_arm_rollout_compare(args)


if __name__ == "__main__":
    main()
