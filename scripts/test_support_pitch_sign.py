from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco
import numpy as np

from envs.g1_right_lift_env import CONTROLLED_15_JOINTS, G1RightLiftEnv


SUPPORT_JOINTS = [
    "left_hip_pitch_joint",
    "left_ankle_pitch_joint",
]


def get_info_direct(env: G1RightLiftEnv) -> Dict[str, float]:
    _, info = env._sagittal_feedback()
    foot = env._foot_metrics()
    info.update(
        {
            "x_position": float(env.data.qpos[0]),
            "x_velocity": float(env.data.qvel[0]),
            "base_height": float(env.data.qpos[2]),
            "up_z": float(env._root_up_z()),
            "left_foot_x": float(env.data.site_xpos[env.left_foot_site_id][0]),
            "right_foot_x": float(env.data.site_xpos[env.right_foot_site_id][0]),
        }
    )
    info.update(foot)
    return info


def set_teacher_target_with_extra(env: G1RightLiftEnv, extra_offsets: Dict[str, float]) -> None:
    target = env.stand_joint_pos + env._teacher_offsets()
    for joint_name, offset in extra_offsets.items():
        idx = CONTROLLED_15_JOINTS.index(joint_name)
        target[idx] += float(offset)
    target = np.clip(target, env.ctrl_low, env.ctrl_high)
    env._set_actuator_targets(target)


def run_to_test_phase(env: G1RightLiftEnv, start_step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    obs, info = env.reset()
    action = np.zeros(15, dtype=np.float32)

    for _ in range(start_step):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "The teacher reached termination before the sign-test phase. "
                f"Ended at step={info.get('episode_step')} reason={env.termination_reason(info)} info={info}"
            )

    return env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), get_info_direct(env)


def test_joint(
    env: G1RightLiftEnv,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl0: np.ndarray,
    test_step: int,
    joint_name: str,
    offset: float,
    hold_steps: int,
) -> Dict[str, float]:
    env.data.qpos[:] = qpos0
    env.data.qvel[:] = qvel0
    env.data.ctrl[:] = ctrl0
    env.episode_step = test_step
    mujoco.mj_forward(env.model, env.data)

    before = get_info_direct(env)

    for _ in range(hold_steps):
        env.episode_step = test_step
        set_teacher_target_with_extra(env, {joint_name: offset})
        mujoco.mj_step(env.model, env.data)

    after = get_info_direct(env)

    return {
        "before_error": float(before["com_x_error"]),
        "after_error": float(after["com_x_error"]),
        "before_com_x": float(before["pelvis_com_x"]),
        "after_com_x": float(after["pelvis_com_x"]),
        "before_support_x": float(before["support_foot_x"]),
        "after_support_x": float(after["support_foot_x"]),
        "before_x": float(before["x_position"]),
        "after_x": float(after["x_position"]),
        "before_vx": float(before["x_velocity"]),
        "after_vx": float(after["x_velocity"]),
        "before_up_z": float(before["up_z"]),
        "after_up_z": float(after["up_z"]),
        "right_clearance": float(after["right_foot_clearance"]),
        "left_slip": float(after["left_foot_slip"]),
        "offset": float(offset),
        "abs_error_improvement": float(abs(before["com_x_error"]) - abs(after["com_x_error"])),
        "signed_error_change": float(after["com_x_error"] - before["com_x_error"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Empirically determine support hip/ankle pitch signs for live sagittal COM feedback."
    )
    parser.add_argument("--model_path", type=str, default="third_party/mujoco_menagerie/unitree_g1/scene.xml")
    parser.add_argument("--dataset_path", type=str, default="datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz")
    parser.add_argument("--start_step", type=int, default=250, help="Right-lift phase step near active single-support.")
    parser.add_argument("--hold_steps", type=int, default=20, help="How many raw MuJoCo steps to hold each joint offset.")
    parser.add_argument("--delta", type=float, default=0.05, help="Joint offset in radians for + and - tests.")
    parser.add_argument("--teacher_scale_multiplier", type=float, default=1.35)
    parser.add_argument("--action_scale", type=float, default=0.08)
    parser.add_argument("--swing_leg_scale", type=float, default=0.1)
    parser.add_argument("--waist_scale", type=float, default=0.8)
    args = parser.parse_args()

    env = G1RightLiftEnv(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        action_scale=args.action_scale,
        teacher_scale_multiplier=args.teacher_scale_multiplier,
        support_leg_scale=1.0,
        swing_leg_scale=args.swing_leg_scale,
        waist_scale=args.waist_scale,
        sagittal_kp=0.0,
        sagittal_kd=0.0,
        sagittal_clip=0.0,
        randomize_reset=False,
    )

    try:
        qpos0, qvel0, ctrl0, base = run_to_test_phase(env, args.start_step)

        print("=" * 100)
        print("SUPPORT PITCH SIGN TEST")
        print("This test disables sagittal feedback and applies tiny isolated joint offsets.")
        print("Goal: choose the sign that reduces |com_x - support_foot_x| from the same snapshot.")
        print("=" * 100)
        print(
            f"Snapshot step={args.start_step} "
            f"phase={env._phase01():.3f} "
            f"com_x_error={base['com_x_error']:+.6f} "
            f"com_x={base['pelvis_com_x']:+.6f} "
            f"support_x={base['support_foot_x']:+.6f} "
            f"x={base['x_position']:+.6f} "
            f"vx={base['x_velocity']:+.6f} "
            f"up_z={base['up_z']:.6f} "
            f"Rclr={base['right_foot_clearance']:.6f}"
        )
        print()

        if abs(base["com_x_error"]) < 1e-4:
            print("WARNING: baseline COM error is very small. Increase --start_step or --hold_steps if results look tied.")

        recommended = {}

        for joint_name in SUPPORT_JOINTS:
            plus = test_joint(env, qpos0, qvel0, ctrl0, args.start_step, joint_name, +args.delta, args.hold_steps)
            minus = test_joint(env, qpos0, qvel0, ctrl0, args.start_step, joint_name, -args.delta, args.hold_steps)

            print("-" * 100)
            print(joint_name)
            for label, result in [("+delta", plus), ("-delta", minus)]:
                print(
                    f"{label:7s} offset={result['offset']:+.3f} "
                    f"err {result['before_error']:+.6f}->{result['after_error']:+.6f} "
                    f"abs_improve={result['abs_error_improvement']:+.6f} "
                    f"com_x {result['before_com_x']:+.6f}->{result['after_com_x']:+.6f} "
                    f"support_x {result['before_support_x']:+.6f}->{result['after_support_x']:+.6f} "
                    f"x {result['before_x']:+.6f}->{result['after_x']:+.6f} "
                    f"vx {result['before_vx']:+.6f}->{result['after_vx']:+.6f} "
                    f"up {result['before_up_z']:.4f}->{result['after_up_z']:.4f}"
                )

            # Which physical offset reduced absolute COM-support error more?
            best_offset_sign = +1.0 if plus["abs_error_improvement"] >= minus["abs_error_improvement"] else -1.0

            # The controller command is defined as: command = -kp*error - kd*velocity.
            # For sign identification, use the P direction from the baseline error.
            command_for_current_error = -math.copysign(1.0, base["com_x_error"]) if abs(base["com_x_error"]) > 1e-8 else 1.0
            joint_sign_for_positive_command = best_offset_sign / command_for_current_error

            recommended[joint_name] = int(np.sign(joint_sign_for_positive_command))

            print(
                f"Best physical offset sign at this snapshot: {best_offset_sign:+.0f}. "
                f"Recommended controller sign for positive command: {recommended[joint_name]:+d}"
            )

        print("=" * 100)
        print("RECOMMENDED ARGS FOR FEEDBACK ENV")
        print(
            f"--sagittal_hip_sign {recommended['left_hip_pitch_joint']} "
            f"--sagittal_ankle_sign {recommended['left_ankle_pitch_joint']}"
        )
        print("=" * 100)
        print("Next: run evaluation first with these signs before training.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
