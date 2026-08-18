from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env_v3b import G1PhaseBCResidualEnvV3B, geom_min_z


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def apply_joint_positions(qpos: np.ndarray, qpos_addresses: list[int], values: np.ndarray) -> None:
    for value, qadr in zip(values, qpos_addresses):
        qpos[qadr] = float(value)


def find_reference_peak_frame(
    env: G1PhaseBCResidualEnvV3B,
    side: str,
    joint_pos_15: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    temp_data = mujoco.MjData(env.model)

    if side == "left":
        swing_geoms = env.left_foot_geom_ids
    else:
        swing_geoms = env.right_foot_geom_ids

    z_values = []

    for frame in range(len(joint_pos_15)):
        qpos = env.stand_qpos.copy()
        apply_joint_positions(qpos, env.qpos_addresses, joint_pos_15[frame])

        qpos[0] = env.stand_qpos[0]
        qpos[1] = env.stand_qpos[1]
        qpos[2] = env.stand_qpos[2]

        temp_data.qpos[:] = qpos
        temp_data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, temp_data)

        z_values.append(geom_min_z(temp_data, swing_geoms))

    z_arr = np.asarray(z_values, dtype=np.float64)
    floor_z = float(np.percentile(z_arr, 5))
    clearance = z_arr - floor_z

    peak_frame = int(np.argmax(clearance))
    peak_clearance = float(clearance[peak_frame])

    return peak_frame, peak_clearance, z_arr


def joint_value(env: G1PhaseBCResidualEnvV3B, joint_name: str) -> float:
    if joint_name not in env.joint_names:
        return 0.0

    idx = env.joint_names.index(joint_name)
    return float(env._get_joint_pos()[idx])


def target_value(env: G1PhaseBCResidualEnvV3B, target: np.ndarray, joint_name: str) -> float:
    if joint_name not in env.joint_names:
        return 0.0

    idx = env.joint_names.index(joint_name)
    return float(target[idx])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=500)

    parser.add_argument(
        "--dataset",
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )

    parser.add_argument("--pose_gain", type=float, default=1.0)
    parser.add_argument("--lift_start", type=int, default=80)
    parser.add_argument("--ramp_steps", type=int, default=180)
    parser.add_argument("--target_smoothing", type=float, default=0.06)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    env = G1PhaseBCResidualEnvV3B(
        render_mode="human" if args.render else None,
        dataset_path=str(dataset_path),
        start_frame=90,
        gait_scale=0.0,
        residual_scale=0.0,
        target_smoothing=args.target_smoothing,
        transition_steps=1,
        target_velocity=0.0,
        max_episode_steps=args.steps,
    )

    dataset = np.load(dataset_path, allow_pickle=True)
    joint_pos_15 = np.asarray(dataset["joint_pos_15"], dtype=np.float64)

    peak_frame, peak_clearance, _ = find_reference_peak_frame(
        env=env,
        side=args.side,
        joint_pos_15=joint_pos_15,
    )

    reference_pose = joint_pos_15[peak_frame].astype(np.float64)
    stand_pose = env.stand_joint_values.astype(np.float64)

    target_pose = stand_pose + args.pose_gain * (reference_pose - stand_pose)

    if args.side == "left":
        swing_geoms = env.left_foot_geom_ids
        key_joints = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ]
    else:
        swing_geoms = env.right_foot_geom_ids
        key_joints = [
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]

    obs, info = env.reset()

    initial_swing_z = geom_min_z(env.data, swing_geoms)
    current_target = stand_pose.copy()

    print("=" * 100)
    print("REFERENCE-POSE SINGLE-FOOT LIFT DIAGNOSTIC")
    print("=" * 100)
    print("Side:", args.side)
    print("Dataset:", dataset_path)
    print("Peak reference frame:", peak_frame)
    print("Kinematic peak reference clearance:", f"{peak_clearance:+.4f}")
    print("Pose gain:", args.pose_gain)
    print("Target smoothing:", args.target_smoothing)
    print("=" * 100)
    print("Key joint target deltas from stand:")

    for name in key_joints:
        if name in env.joint_names:
            idx = env.joint_names.index(name)
            delta = target_pose[idx] - stand_pose[idx]
            print(f"  {name}: {delta:+.4f} rad")

    print("=" * 100)

    max_swing_clearance = 0.0
    min_z = float(info["base_z"])
    min_up_z = float(info["up_z"])
    max_x = float(info["base_x"])
    final_info = info

    for step in range(args.steps):
        if step < args.lift_start:
            alpha = 0.0
        else:
            alpha = smoothstep((step - args.lift_start) / max(args.ramp_steps, 1))

        desired_target = stand_pose + alpha * (target_pose - stand_pose)

        current_target = (
            (1.0 - args.target_smoothing) * current_target
            + args.target_smoothing * desired_target
        )

        env._apply_target(current_target)

        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)

        if env.viewer is not None:
            if env.viewer.is_running():
                env.viewer.sync()

        base_x = float(env.data.qpos[0])
        base_z = float(env.data.qpos[2])
        up_z = env._get_up_z()

        swing_z = geom_min_z(env.data, swing_geoms)
        swing_clearance = max(0.0, swing_z - initial_swing_z)

        actual_joint_pos = env._get_joint_pos().astype(np.float64)
        target_error = float(np.mean(np.abs(actual_joint_pos - current_target)))

        max_swing_clearance = max(max_swing_clearance, swing_clearance)
        min_z = min(min_z, base_z)
        min_up_z = min(min_up_z, up_z)
        max_x = max(max_x, base_x)

        terminated = bool(base_z < env.fall_height or up_z < env.fall_up_z)

        final_info = {
            "step": step + 1,
            "base_x": base_x,
            "base_z": base_z,
            "up_z": up_z,
            "terminated": terminated,
        }

        if step % args.print_every == 0 or terminated:
            joint_report = []

            for name in key_joints[:3]:
                joint_report.append(
                    f"{name.replace('_joint', '')}:"
                    f"{joint_value(env, name):+.2f}/{target_value(env, current_target, name):+.2f}"
                )

            print(
                f"step={step:04d} "
                f"x={base_x:+.3f} "
                f"z={base_z:+.3f} "
                f"up_z={up_z:+.3f} "
                f"alpha={alpha:.2f} "
                f"swing_z={swing_z:+.4f} "
                f"swing_clear={swing_clearance:+.4f} "
                f"target_err={target_error:.4f} "
                f"{' | '.join(joint_report)} "
                f"fallen={terminated}"
            )

        if terminated:
            break

    print()
    print("=" * 100)
    print("REFERENCE-POSE SINGLE-FOOT LIFT SUMMARY")
    print("=" * 100)
    print("Side:", args.side)
    print("Peak reference frame:", peak_frame)
    print("Kinematic peak reference clearance:", f"{peak_clearance:+.4f}")
    print("Steps completed:", final_info["step"])
    print("Final x:", f"{final_info['base_x']:+.3f}")
    print("Max x:", f"{max_x:+.3f}")
    print("Final z:", f"{final_info['base_z']:+.3f}")
    print("Final up_z:", f"{final_info['up_z']:+.3f}")
    print("Min z:", f"{min_z:+.3f}")
    print("Min up_z:", f"{min_up_z:+.3f}")
    print("Max dynamic swing foot clearance:", f"{max_swing_clearance:+.4f}")
    print("Terminated:", final_info["terminated"])
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
