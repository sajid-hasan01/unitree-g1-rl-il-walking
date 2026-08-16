import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)

RIGHT_FOOT_SPHERES = [30, 31, 32, 33]
LEFT_FOOT_SPHERES = [15, 16, 17, 18]

TEST_FRAMES = [
    16,
    19,
    20,
    22,
    25,
    28,
]


def load_initializer():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_residual_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load initializer from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def xyzw_to_mujoco(raw_xyzw):
    raw_xyzw = np.asarray(
        raw_xyzw,
        dtype=np.float64,
    )

    q = np.array(
        [
            raw_xyzw[3],
            raw_xyzw[0],
            raw_xyzw[1],
            raw_xyzw[2],
        ],
        dtype=np.float64,
    )

    q /= np.linalg.norm(q)

    return q


def quaternion_rpy_degrees(q):
    flat = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        flat,
        q,
    )

    R = flat.reshape(3, 3)

    roll = np.arctan2(
        R[2, 1],
        R[2, 2],
    )

    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(
            R[2, 1] ** 2
            + R[2, 2] ** 2
        ),
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0],
    )

    return np.degrees(
        [roll, pitch, yaw]
    )


def remove_world_yaw(q):
    """
    Remove only the global heading component.

    The remaining orientation keeps the OpenHE
    root/pelvis tilt while aligning walking heading
    with yaw = 0.
    """

    rpy = quaternion_rpy_degrees(q)

    yaw_rad = np.radians(
        float(rpy[2])
    )

    inverse_yaw = np.zeros(
        4,
        dtype=np.float64,
    )

    mujoco.mju_axisAngle2Quat(
        inverse_yaw,
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        -yaw_rad,
    )

    aligned = np.zeros(
        4,
        dtype=np.float64,
    )

    # World-yaw alignment is pre-multiplied.
    mujoco.mju_mulQuat(
        aligned,
        inverse_yaw,
        q,
    )

    aligned /= np.linalg.norm(
        aligned
    )

    return aligned


def sphere_bottom_z(env, geom_id):
    center_z = float(
        env.data.geom_xpos[geom_id, 2]
    )

    radius = float(
        env.model.geom_size[geom_id, 0]
    )

    return center_z - radius


def foot_bottom_values(env, geom_ids):
    return np.array(
        [
            sphere_bottom_z(
                env,
                geom_id,
            )
            for geom_id in geom_ids
        ],
        dtype=np.float64,
    )


def right_heel_to_toe_delta(z):
    """
    Geoms:
      30,31 = rear / heel
      32,33 = front / toe

    Positive value:
        heel is higher than toe
        => toe-down orientation
    """

    heel = float(
        np.mean(z[0:2])
    )

    toe = float(
        np.mean(z[2:4])
    )

    return heel - toe


def left_heel_to_toe_delta(z):
    """
    Geoms:
      15,16 = rear / heel
      17,18 = front / toe
    """

    heel = float(
        np.mean(z[0:2])
    )

    toe = float(
        np.mean(z[2:4])
    )

    return heel - toe


def format_mm(values):
    return " ".join(
        f"{1000.0 * value:+7.2f}"
        for value in values
    )


def main():
    data = np.load(
        DATASET_PATH,
        allow_pickle=True,
    )

    root_rot = np.asarray(
        data["root_rot"],
        dtype=np.float64,
    )

    initialize_from_reference = (
        load_initializer()
    )

    env = G1DynamicWalkingEnv(
        dataset_path=str(DATASET_PATH),
        reference_mode="cyclic",
        target_forward_velocity=0.10,
        action_scale=0.055,
        action_target_smoothing=0.25,
        frame_skip=5,
        height_offset=0.02,
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

        print()
        print("=" * 132)
        print(
            "OPENHE ROOT-TILT / FOOT-GEOMETRY DIAGNOSTIC"
        )
        print("=" * 132)

        print(
            "All foot heights below are sphere-bottom "
            "heights in millimetres."
        )

        print(
            "Positive heel-to-toe delta means "
            "HEEL HIGHER THAN TOE (toe-down foot)."
        )

        print()

        for frame in TEST_FRAMES:

            # -------------------------------------------------
            # A) CURRENT RSI / FLAT BASE
            # -------------------------------------------------

            initialize_from_reference(
                env,
                frame,
            )

            flat_quat = np.asarray(
                env.data.qpos[3:7],
                dtype=np.float64,
            ).copy()

            mujoco.mj_forward(
                env.model,
                env.data,
            )

            right_flat = (
                foot_bottom_values(
                    env,
                    RIGHT_FOOT_SPHERES,
                )
            )

            left_flat = (
                foot_bottom_values(
                    env,
                    LEFT_FOOT_SPHERES,
                )
            )

            right_flat_delta = (
                right_heel_to_toe_delta(
                    right_flat
                )
            )

            left_flat_delta = (
                left_heel_to_toe_delta(
                    left_flat
                )
            )

            # -------------------------------------------------
            # B) SAME JOINT POSE + VERIFIED OPENHE ROOT TILT
            # -------------------------------------------------

            q_openhe = xyzw_to_mujoco(
                root_rot[frame]
            )

            q_openhe_aligned = (
                remove_world_yaw(
                    q_openhe
                )
            )

            # Only change base orientation.
            # Joint positions and base XYZ remain identical.
            env.data.qpos[3:7] = (
                q_openhe_aligned
            )

            mujoco.mj_forward(
                env.model,
                env.data,
            )

            right_openhe = (
                foot_bottom_values(
                    env,
                    RIGHT_FOOT_SPHERES,
                )
            )

            left_openhe = (
                foot_bottom_values(
                    env,
                    LEFT_FOOT_SPHERES,
                )
            )

            right_openhe_delta = (
                right_heel_to_toe_delta(
                    right_openhe
                )
            )

            left_openhe_delta = (
                left_heel_to_toe_delta(
                    left_openhe
                )
            )

            source_rpy = (
                quaternion_rpy_degrees(
                    q_openhe
                )
            )

            aligned_rpy = (
                quaternion_rpy_degrees(
                    q_openhe_aligned
                )
            )

            flat_rpy = (
                quaternion_rpy_degrees(
                    flat_quat
                )
            )

            print("-" * 132)

            print(
                f"FRAME {frame}"
            )

            print(
                "  Current base RPY : "
                f"{flat_rpy[0]:+7.2f} "
                f"{flat_rpy[1]:+7.2f} "
                f"{flat_rpy[2]:+7.2f} deg"
            )

            print(
                "  OpenHE raw RPY   : "
                f"{source_rpy[0]:+7.2f} "
                f"{source_rpy[1]:+7.2f} "
                f"{source_rpy[2]:+7.2f} deg"
            )

            print(
                "  Yaw-aligned RPY  : "
                f"{aligned_rpy[0]:+7.2f} "
                f"{aligned_rpy[1]:+7.2f} "
                f"{aligned_rpy[2]:+7.2f} deg"
            )

            print()

            print(
                "  RIGHT FLAT       "
                f"[30 31 32 33] mm: "
                f"{format_mm(right_flat)}"
            )

            print(
                "  RIGHT OPENHE     "
                f"[30 31 32 33] mm: "
                f"{format_mm(right_openhe)}"
            )

            print(
                "  RIGHT heel-to-toe:"
                f" FLAT={1000.0 * right_flat_delta:+7.2f} mm"
                f"  OPENHE={1000.0 * right_openhe_delta:+7.2f} mm"
                f"  CHANGE={1000.0 * (right_openhe_delta - right_flat_delta):+7.2f} mm"
            )

            print()

            print(
                "  LEFT  FLAT       "
                f"[15 16 17 18] mm: "
                f"{format_mm(left_flat)}"
            )

            print(
                "  LEFT  OPENHE     "
                f"[15 16 17 18] mm: "
                f"{format_mm(left_openhe)}"
            )

            print(
                "  LEFT heel-to-toe :"
                f" FLAT={1000.0 * left_flat_delta:+7.2f} mm"
                f"  OPENHE={1000.0 * left_openhe_delta:+7.2f} mm"
                f"  CHANGE={1000.0 * (left_openhe_delta - left_flat_delta):+7.2f} mm"
            )

        print()
        print("=" * 132)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 132)

    finally:
        env.close()


if __name__ == "__main__":
    main()