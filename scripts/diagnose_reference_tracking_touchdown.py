import argparse
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\"
    "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)


def load_existing_initializer():
    """
    Load initialize_from_reference() directly from the existing
    zero-residual test script.

    This avoids duplicating or guessing the initialization procedure.
    """
    test_script_path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_residual_test_module",
        test_script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load existing test script: {test_script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "initialize_from_reference"):
        raise RuntimeError(
            "initialize_from_reference() was not found in "
            "test_zero_residual_full_reference_rsi.py"
        )

    return module.initialize_from_reference


def get_base_yaw_degrees(env):
    """
    Convert the MuJoCo free-joint quaternion to a rotation matrix using
    MuJoCo itself, then extract world-frame yaw.
    """
    quat = np.asarray(env.data.qpos[3:7], dtype=np.float64)

    rotation_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, quat)

    rotation = rotation_flat.reshape(3, 3)

    yaw = np.arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )

    return float(np.degrees(yaw))


def get_control_targets(env):
    """
    Current final smoothed/clipped targets used by the controlled actuators.
    """
    return np.asarray(
        [env.data.ctrl[actuator_id] for actuator_id in env.actuator_ids],
        dtype=np.float64,
    )


def make_joint_index_map(env):
    return {
        str(name): i
        for i, name in enumerate(env.controlled_joint_names)
    }


def capture_record(
    env,
    step,
    control_frame,
    reference,
    target,
):
    actual = env._get_joint_positions().astype(np.float64)

    collision_left, collision_right = env._get_foot_contacts()

    left_site = env._get_site_position(
        env.left_foot_site_id
    ).astype(np.float64)

    right_site = env._get_site_position(
        env.right_foot_site_id
    ).astype(np.float64)

    return {
        "step": int(step),
        "control_frame": float(control_frame),
        "post_motion_frame": float(env.motion_frame),

        "reference": np.asarray(reference, dtype=np.float64).copy(),
        "target": np.asarray(target, dtype=np.float64).copy(),
        "actual": actual.copy(),

        "collision_left": bool(collision_left),
        "collision_right": bool(collision_right),

        "left_site": left_site.copy(),
        "right_site": right_site.copy(),

        "root_xyz": np.asarray(
            env.data.qpos[0:3],
            dtype=np.float64,
        ).copy(),

        "root_linear_velocity": np.asarray(
            env.data.qvel[0:3],
            dtype=np.float64,
        ).copy(),

        "yaw_degrees": get_base_yaw_degrees(env),

        "up_z": float(env._get_up_z()),
    }


def print_joint_comparison(title, record, joint_indices):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)

    print(
        f"step={record['step']}  "
        f"control_frame={record['control_frame']:.3f}  "
        f"post_frame={record['post_motion_frame']:.3f}"
    )

    print(
        f"collision: "
        f"L={record['collision_left']} "
        f"R={record['collision_right']}"
    )

    print(
        f"left_site_xyz="
        f"{np.round(record['left_site'], 5)}"
    )

    print(
        f"right_site_xyz="
        f"{np.round(record['right_site'], 5)}"
    )

    print(
        f"root_xyz="
        f"{np.round(record['root_xyz'], 5)}"
    )

    print(
        f"root_velocity="
        f"{np.round(record['root_linear_velocity'], 5)}"
    )

    print(
        f"yaw={record['yaw_degrees']:+.2f} deg  "
        f"up_z={record['up_z']:.4f}"
    )

    print()
    print(
        "JOINT                    "
        "REFERENCE      TARGET        ACTUAL       "
        "ACTUAL-REF    ACTUAL-TARGET"
    )
    print("-" * 108)

    selected_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    for name in selected_names:
        if name not in joint_indices:
            continue

        idx = joint_indices[name]

        ref = record["reference"][idx]
        target = record["target"][idx]
        actual = record["actual"][idx]

        print(
            f"{name:26s} "
            f"{ref:+11.5f} "
            f"{target:+11.5f} "
            f"{actual:+11.5f} "
            f"{actual - ref:+11.5f} "
            f"{actual - target:+13.5f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose OpenHE reference tracking during the successful "
            "right-foot touchdown and the failed left-foot touchdown."
        )
    )

    # Defaults copied from the existing zero-residual test.
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--reference_mode",
        type=str,
        default="cyclic",
        choices=["transition", "cyclic"],
    )

    parser.add_argument(
        "--target_velocity",
        type=float,
        default=-0.10,
    )

    parser.add_argument(
        "--initial_yaw_degrees",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.055,
    )

    parser.add_argument(
        "--action_target_smoothing",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--reference_speed",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--initial_stand_steps",
        type=int,
        default=70,
    )

    parser.add_argument(
        "--transition_steps",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=110,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    initialize_from_reference = load_existing_initializer()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode=args.reference_mode,
        target_forward_velocity=args.target_velocity,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    try:
        env.reset()

        initialize_from_reference(
            env,
            args.start_frame,
        )

        joint_indices = make_joint_index_map(env)

        zero_action = np.zeros(
            env.num_actions,
            dtype=np.float32,
        )

        initial_left, initial_right = env._get_foot_contacts()

        previous_left = bool(initial_left)
        previous_right = bool(initial_right)

        right_touchdown_record = None
        left_liftoff_record = None
        left_touchdown_record = None

        right_touchdown_seen = previous_right

        closest_left_record = None
        closest_left_site_z = float("inf")

        all_records = []

        print()
        print("=" * 108)
        print("OPENHE TOUCHDOWN / TRACKING DIAGNOSTIC")
        print("=" * 108)

        print("Dataset:", args.dataset_path)
        print("Start frame:", args.start_frame)
        print("Target velocity:", args.target_velocity)
        print("Reference speed:", args.reference_speed)
        print("Height offset:", args.height_offset)
        print(
            "Action target smoothing:",
            args.action_target_smoothing,
        )
        print("Frame skip:", args.frame_skip)

        print()
        print(
            "Initial physical collision:"
            f" L={previous_left}"
            f" R={previous_right}"
        )

        print()
        print(
            "step | ctrlFrm | postFrm | "
            "L/R physical | Lz      Rz      | "
            "x       y       | yaw(deg) | up_z"
        )
        print("-" * 108)

        for step in range(args.steps):
            # IMPORTANT:
            # This reference is captured BEFORE env.step().
            #
            # The environment applies control using the current motion_frame,
            # performs physics, and only then advances motion_frame.
            control_frame = float(env.motion_frame)

            reference = (
                env._get_reference_joint_positions_for_step()
                .astype(np.float64)
                .copy()
            )

            observation, reward, terminated, truncated, info = (
                env.step(zero_action)
            )

            # After env.step(), this is the final smoothed/clipped
            # actuator target that was actually applied during the step.
            target = get_control_targets(env)

            record = capture_record(
                env=env,
                step=step,
                control_frame=control_frame,
                reference=reference,
                target=target,
            )

            all_records.append(record)

            current_left = record["collision_left"]
            current_right = record["collision_right"]

            # Successful first right-foot touchdown.
            if (
                not previous_right
                and current_right
                and right_touchdown_record is None
            ):
                right_touchdown_record = record
                right_touchdown_seen = True

                print()
                print(
                    f">>> EVENT: RIGHT FOOT TOUCHDOWN "
                    f"at step {step}, "
                    f"control_frame {control_frame:.3f}"
                )

            # Left foot leaves the floor after the right foot has landed.
            if (
                right_touchdown_seen
                and previous_left
                and not current_left
                and left_liftoff_record is None
            ):
                left_liftoff_record = record

                print()
                print(
                    f">>> EVENT: LEFT FOOT LIFTOFF "
                    f"at step {step}, "
                    f"control_frame {control_frame:.3f}"
                )

            # Track closest left-foot site after left liftoff.
            if left_liftoff_record is not None:
                left_z = float(record["left_site"][2])

                if left_z < closest_left_site_z:
                    closest_left_site_z = left_z
                    closest_left_record = record

            # Detect actual left re-touchdown.
            if (
                left_liftoff_record is not None
                and not previous_left
                and current_left
                and left_touchdown_record is None
            ):
                left_touchdown_record = record

                print()
                print(
                    f">>> EVENT: LEFT FOOT PHYSICAL TOUCHDOWN "
                    f"at step {step}, "
                    f"control_frame {control_frame:.3f}"
                )

            if (
                args.print_every > 0
                and step % args.print_every == 0
            ):
                print(
                    f"{step:4d} | "
                    f"{control_frame:7.2f} | "
                    f"{record['post_motion_frame']:7.2f} | "
                    f"{int(current_left)}/{int(current_right)}          | "
                    f"{record['left_site'][2]:+.4f} "
                    f"{record['right_site'][2]:+.4f} | "
                    f"{record['root_xyz'][0]:+.3f} "
                    f"{record['root_xyz'][1]:+.3f} | "
                    f"{record['yaw_degrees']:+8.2f} | "
                    f"{record['up_z']:.3f}"
                )

            previous_left = current_left
            previous_right = current_right

            if terminated or truncated:
                print()
                print(
                    "Episode ended:",
                    f"terminated={terminated}",
                    f"truncated={truncated}",
                    f"step={step}",
                )
                break

        print()
        print("=" * 108)
        print("EVENT SUMMARY")
        print("=" * 108)

        if right_touchdown_record is None:
            print("Right-foot touchdown: NOT DETECTED")
        else:
            print(
                "Right-foot touchdown:",
                f"step={right_touchdown_record['step']}",
                f"frame={right_touchdown_record['control_frame']:.3f}",
            )

        if left_liftoff_record is None:
            print("Left-foot liftoff: NOT DETECTED")
        else:
            print(
                "Left-foot liftoff:",
                f"step={left_liftoff_record['step']}",
                f"frame={left_liftoff_record['control_frame']:.3f}",
            )

        if left_touchdown_record is None:
            print(
                "Left-foot re-touchdown: NOT DETECTED"
            )
        else:
            print(
                "Left-foot re-touchdown:",
                f"step={left_touchdown_record['step']}",
                f"frame={left_touchdown_record['control_frame']:.3f}",
            )

        if closest_left_record is not None:
            print(
                "Closest left-foot site after liftoff:",
                f"step={closest_left_record['step']}",
                f"frame={closest_left_record['control_frame']:.3f}",
                f"site_z={closest_left_record['left_site'][2]:.6f}",
                f"yaw={closest_left_record['yaw_degrees']:+.2f} deg",
            )

        if right_touchdown_record is not None:
            print_joint_comparison(
                "SUCCESSFUL RIGHT-FOOT TOUCHDOWN",
                right_touchdown_record,
                joint_indices,
            )

        if left_touchdown_record is not None:
            print_joint_comparison(
                "ACTUAL LEFT-FOOT TOUCHDOWN",
                left_touchdown_record,
                joint_indices,
            )
        elif closest_left_record is not None:
            print_joint_comparison(
                "FAILED LEFT-FOOT LANDING: CLOSEST PHYSICAL APPROACH",
                closest_left_record,
                joint_indices,
            )

        print()
        print("=" * 108)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 108)

    finally:
        env.close()


if __name__ == "__main__":
    main()