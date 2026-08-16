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


LEFT_FOOT_SPHERES = [15, 16, 17, 18]
RIGHT_FOOT_SPHERES = [30, 31, 32, 33]


def load_initializer():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_test",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def yaw_degrees(env):
    quat = np.asarray(
        env.data.qpos[3:7],
        dtype=np.float64,
    )

    matrix_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix_flat, quat)

    matrix = matrix_flat.reshape(3, 3)

    return float(
        np.degrees(
            np.arctan2(
                matrix[1, 0],
                matrix[0, 0],
            )
        )
    )


def sphere_bottom_z(env, geom_id):
    center_z = float(
        env.data.geom_xpos[geom_id][2]
    )

    radius = float(
        env.model.geom_size[geom_id][0]
    )

    return center_z - radius


def active_floor_contacts(env, geom_ids, floor_id):
    active = []

    geom_ids = set(geom_ids)

    for contact_id in range(env.data.ncon):
        contact = env.data.contact[contact_id]

        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

        if (
            g1 == floor_id
            and g2 in geom_ids
        ):
            active.append(g2)

        elif (
            g2 == floor_id
            and g1 in geom_ids
        ):
            active.append(g1)

    return sorted(set(active))


def print_foot_line(env, geom_ids):
    values = []

    for geom_id in geom_ids:
        z = sphere_bottom_z(
            env,
            geom_id,
        )

        values.append(
            f"{geom_id}:{z:+.4f}"
        )

    return " ".join(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--target_velocity",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--reference_speed",
        type=float,
        default=1.0,
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
        default=0.25,
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=5,
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
        "--steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    initialize_from_reference = (
        load_initializer()
    )

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=(
            args.target_velocity
        ),
        action_scale=args.action_scale,
        action_target_smoothing=(
            args.action_target_smoothing
        ),
        frame_skip=args.frame_skip,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=(
            args.initial_stand_steps
        ),
        transition_steps=(
            args.transition_steps
        ),
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=(
            args.initial_yaw_degrees
        ),
    )

    try:
        env.reset()

        initialize_from_reference(
            env,
            args.start_frame,
        )

        floor_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )

        zero_action = np.zeros(
            env.num_actions,
            dtype=np.float32,
        )

        print()
        print("=" * 130)
        print(
            "FOOT CONTACT-SPHERE HEIGHT DIAGNOSTIC"
        )
        print("=" * 130)

        print(
            "Bottom height = sphere center Z - sphere radius."
        )

        print(
            "A value near 0 means that contact point "
            "is at floor level."
        )

        print()
        print(
            "step frame yaw    yVel   "
            "| LEFT bottom-z (15 16 17 18) "
            "| RIGHT bottom-z (30 31 32 33) "
            "| contacts"
        )

        print("-" * 130)

        for step in range(args.steps):
            control_frame = float(
                env.motion_frame
            )

            env.step(zero_action)

            if (
                args.print_every > 0
                and step % args.print_every != 0
            ):
                continue

            left_contacts = (
                active_floor_contacts(
                    env,
                    LEFT_FOOT_SPHERES,
                    floor_id,
                )
            )

            right_contacts = (
                active_floor_contacts(
                    env,
                    RIGHT_FOOT_SPHERES,
                    floor_id,
                )
            )

            left_values = print_foot_line(
                env,
                LEFT_FOOT_SPHERES,
            )

            right_values = print_foot_line(
                env,
                RIGHT_FOOT_SPHERES,
            )

            print(
                f"{step:4d} "
                f"{control_frame:5.1f} "
                f"{yaw_degrees(env):+7.2f} "
                f"{float(env.data.qvel[1]):+7.3f} "
                f"| {left_values} "
                f"| {right_values} "
                f"| L={left_contacts} "
                f"R={right_contacts}"
            )

        print()
        print("=" * 130)

    finally:
        env.close()


if __name__ == "__main__":
    main()