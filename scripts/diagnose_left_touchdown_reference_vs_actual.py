import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_rawlegs_15dof.npz"
)

LEFT_SPHERES = [15, 16, 17, 18]

LEFT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]


def load_initializer():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "rsi_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def sphere_bottoms(env):
    values = []

    for geom_id in LEFT_SPHERES:
        center_z = float(
            env.data.geom_xpos[geom_id, 2]
        )

        radius = float(
            env.model.geom_size[geom_id, 0]
        )

        values.append(
            center_z - radius
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def heel_to_toe(values):
    heel = float(
        np.mean(values[0:2])
    )

    toe = float(
        np.mean(values[2:4])
    )

    return heel - toe


def get_yaw_degrees(env):
    flat = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        flat,
        np.asarray(
            env.data.qpos[3:7],
            dtype=np.float64,
        ),
    )

    R = flat.reshape(3, 3)

    return float(
        np.degrees(
            np.arctan2(
                R[1, 0],
                R[0, 0],
            )
        )
    )


def get_left_contact_geoms(env):
    active = []

    for contact_id in range(
        env.data.ncon
    ):
        contact = env.data.contact[
            contact_id
        ]

        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

        if g1 in LEFT_SPHERES:
            active.append(g1)

        if g2 in LEFT_SPHERES:
            active.append(g2)

    return sorted(set(active))


def ideal_reference_geometry(
    env,
    reference_joint_pos,
):
    """
    Keep the ACTUAL floating-base position/orientation
    exactly as it currently is, but temporarily replace
    the controlled joints with the reference values.

    This tells us what foot geometry the reference itself
    is requesting at the current physical base pose.
    """

    saved_qpos = (
        env.data.qpos.copy()
    )

    try:
        for index, address in enumerate(
            env.joint_qpos_addresses
        ):
            env.data.qpos[address] = float(
                reference_joint_pos[index]
            )

        mujoco.mj_forward(
            env.model,
            env.data,
        )

        bottoms = sphere_bottoms(
            env
        )

        return bottoms.copy()

    finally:
        env.data.qpos[:] = (
            saved_qpos
        )

        mujoco.mj_forward(
            env.model,
            env.data,
        )


def print_joint_table(
    names,
    reference,
    target,
    actual,
    joint_indices,
):
    print(
        "JOINT                    "
        "REFERENCE      TARGET        ACTUAL       "
        "ACTUAL-REF    ACTUAL-TARGET"
    )

    print("-" * 100)

    for name in names:
        index = joint_indices[name]

        ref_value = float(
            reference[index]
        )

        target_value = float(
            target[index]
        )

        actual_value = float(
            actual[index]
        )

        print(
            f"{name:27s} "
            f"{ref_value:+11.5f} "
            f"{target_value:+11.5f} "
            f"{actual_value:+11.5f} "
            f"{actual_value-ref_value:+11.5f} "
            f"{actual_value-target_value:+13.5f}"
        )


def main():
    initialize_from_reference = (
        load_initializer()
    )

    env = G1DynamicWalkingEnv(
        dataset_path=str(DATASET),
        reference_mode="cyclic",
        target_forward_velocity=0.10,
        action_scale=0.055,
        action_target_smoothing=0.25,
        frame_skip=5,
        height_offset=-0.01348,
        reference_speed=1.0,
        initial_stand_steps=70,
        transition_steps=220,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=0.0,
    )

    try:
        env.reset()

        initialize_from_reference(
            env,
            28,
        )

        zero_action = np.zeros(
            env.num_actions,
            dtype=np.float32,
        )

        joint_names = [
            str(name)
            for name in env.controlled_joint_names
        ]

        joint_indices = {
            name: index
            for index, name in enumerate(
                joint_names
            )
        }

        print("=" * 120)
        print(
            "LEFT TOUCHDOWN: REFERENCE vs TARGET vs ACTUAL"
        )
        print("=" * 120)

        previous_left_contact = False

        for step in range(30):

            control_frame = float(
                env.motion_frame
            )

            reference = np.asarray(
                env._get_reference_joint_positions_for_step(),
                dtype=np.float64,
            ).copy()

            ideal_bottoms = (
                ideal_reference_geometry(
                    env,
                    reference,
                )
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                zero_action
            )

            actual = np.asarray(
                env._get_joint_positions(),
                dtype=np.float64,
            ).copy()

            target = np.asarray(
                env.last_targets,
                dtype=np.float64,
            ).copy()

            actual_bottoms = (
                sphere_bottoms(
                    env
                )
            )

            left_contact, right_contact = (
                env._get_foot_contacts()
            )

            left_geoms = (
                get_left_contact_geoms(
                    env
                )
            )

            first_touchdown = (
                left_contact
                and not previous_left_contact
            )

            interesting = (
                step >= 14
                or first_touchdown
            )

            if interesting:
                print()
                print("=" * 120)

                print(
                    f"step={step} "
                    f"control_frame={control_frame:.2f} "
                    f"post_frame={env.motion_frame:.2f}"
                )

                print(
                    f"physical contact: "
                    f"L={left_contact} "
                    f"R={right_contact} "
                    f"left_geoms={left_geoms}"
                )

                print(
                    f"yaw={get_yaw_degrees(env):+.2f} deg "
                    f"up_z={float(info['up_z']):.4f} "
                    f"y_vel={float(info['y_velocity']):+.4f}"
                )

                print()

                print(
                    "REFERENCE left sphere bottoms mm:",
                    " ".join(
                        f"{1000*z:+8.2f}"
                        for z in ideal_bottoms
                    ),
                )

                print(
                    "ACTUAL    left sphere bottoms mm:",
                    " ".join(
                        f"{1000*z:+8.2f}"
                        for z in actual_bottoms
                    ),
                )

                print(
                    "REFERENCE heel-to-toe:",
                    f"{1000*heel_to_toe(ideal_bottoms):+.2f} mm",
                )

                print(
                    "ACTUAL    heel-to-toe:",
                    f"{1000*heel_to_toe(actual_bottoms):+.2f} mm",
                )

                print()

                print_joint_table(
                    LEFT_JOINT_NAMES,
                    reference,
                    target,
                    actual,
                    joint_indices,
                )

            previous_left_contact = (
                left_contact
            )

            if terminated or truncated:
                break

        print()
        print("=" * 120)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 120)

    finally:
        env.close()


if __name__ == "__main__":
    main()