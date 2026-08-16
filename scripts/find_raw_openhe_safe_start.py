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
RIGHT_SPHERES = [30, 31, 32, 33]

BASE_HEIGHT_OFFSET = 0.02
TARGET_PENETRATION = -0.001  # 1 mm


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


def sphere_bottoms(env, geom_ids):
    values = []

    for geom_id in geom_ids:
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


def floor_contact_geoms(env, floor_id):
    geom_ids = []

    for contact_id in range(env.data.ncon):
        contact = env.data.contact[contact_id]

        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

        if g1 == floor_id:
            geom_ids.append(g2)

        elif g2 == floor_id:
            geom_ids.append(g1)

    return geom_ids


def heel_to_toe_delta(values):
    heel = float(
        np.mean(values[0:2])
    )

    toe = float(
        np.mean(values[2:4])
    )

    return heel - toe


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
        height_offset=BASE_HEIGHT_OFFSET,
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

        floor_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )

        candidates = []

        for frame in range(env.num_frames):

            # ---------------------------------------
            # Initialize using current known RSI.
            # ---------------------------------------

            initialize_from_reference(
                env,
                frame,
            )

            mujoco.mj_forward(
                env.model,
                env.data,
            )

            left_before = sphere_bottoms(
                env,
                LEFT_SPHERES,
            )

            right_before = sphere_bottoms(
                env,
                RIGHT_SPHERES,
            )

            left_min_before = float(
                np.min(left_before)
            )

            right_min_before = float(
                np.min(right_before)
            )

            lowest_before = min(
                left_min_before,
                right_min_before,
            )

            # ---------------------------------------
            # Move root vertically so the lowest
            # foot sphere penetrates ~1 mm.
            # ---------------------------------------

            vertical_shift = (
                TARGET_PENETRATION
                - lowest_before
            )

            original_root_z = float(
                env.data.qpos[2]
            )

            new_root_z = (
                original_root_z
                + vertical_shift
            )

            # Do not use a pose that would hit the
            # RSI's existing minimum-root-height area.
            if new_root_z <= 0.72:
                continue

            env.data.qpos[2] = new_root_z

            mujoco.mj_forward(
                env.model,
                env.data,
            )

            left = sphere_bottoms(
                env,
                LEFT_SPHERES,
            )

            right = sphere_bottoms(
                env,
                RIGHT_SPHERES,
            )

            left_contact, right_contact = (
                env._get_foot_contacts()
            )

            # Require exactly one support foot.
            if left_contact == right_contact:
                continue

            if left_contact:
                support = "LEFT"
                support_values = left
                swing_values = right
                allowed_foot_geoms = set(
                    LEFT_SPHERES
                )
            else:
                support = "RIGHT"
                support_values = right
                swing_values = left
                allowed_foot_geoms = set(
                    RIGHT_SPHERES
                )

            support_min = float(
                np.min(support_values)
            )

            swing_min = float(
                np.min(swing_values)
            )

            # ---------------------------------------
            # Inspect ALL floor collisions.
            # ---------------------------------------

            floor_geoms = (
                floor_contact_geoms(
                    env,
                    floor_id,
                )
            )

            non_support_floor_geoms = [
                geom_id
                for geom_id in floor_geoms
                if geom_id
                not in allowed_foot_geoms
            ]

            if non_support_floor_geoms:
                continue

            # Swing foot must actually be clear.
            if swing_min < 0.010:
                continue

            # Reject unexpectedly deep contacts.
            if support_min < -0.005:
                continue

            support_pitch = (
                heel_to_toe_delta(
                    support_values
                )
            )

            required_height_offset = (
                BASE_HEIGHT_OFFSET
                + vertical_shift
            )

            candidates.append(
                {
                    "frame": frame,
                    "support": support,
                    "height_offset": (
                        required_height_offset
                    ),
                    "root_z": new_root_z,
                    "support_min": support_min,
                    "swing_min": swing_min,
                    "support_pitch": (
                        support_pitch
                    ),
                    "left": left.copy(),
                    "right": right.copy(),
                    "floor_geoms": (
                        floor_geoms.copy()
                    ),
                }
            )

        # -------------------------------------------
        # Rank:
        # 1. flatter support foot
        # 2. greater swing-foot clearance
        # -------------------------------------------

        candidates.sort(
            key=lambda item: (
                abs(
                    item[
                        "support_pitch"
                    ]
                ),
                -item["swing_min"],
            )
        )

        print("=" * 130)
        print(
            "RAW OPENHE AUTOMATIC SAFE-START SEARCH"
        )
        print("=" * 130)

        print(
            "Valid candidates:",
            len(candidates),
        )

        print()

        print(
            "rank frame support heightOff rootZ   "
            "supportMin swingMin pitchDelta "
            "contacts | LEFT bottoms mm "
            "| RIGHT bottoms mm"
        )

        print("-" * 130)

        for rank, item in enumerate(
            candidates[:20],
            start=1,
        ):

            left_text = " ".join(
                f"{1000*z:+7.2f}"
                for z in item["left"]
            )

            right_text = " ".join(
                f"{1000*z:+7.2f}"
                for z in item["right"]
            )

            contacts_text = ",".join(
                str(x)
                for x in item[
                    "floor_geoms"
                ]
            )

            print(
                f"{rank:4d} "
                f"{item['frame']:5d} "
                f"{item['support']:7s} "
                f"{item['height_offset']:+9.5f} "
                f"{item['root_z']:+7.4f} "
                f"{1000*item['support_min']:+10.2f} "
                f"{1000*item['swing_min']:+8.2f} "
                f"{1000*item['support_pitch']:+10.2f} "
                f"{contacts_text:8s} "
                f"| {left_text} "
                f"| {right_text}"
            )

        print("=" * 130)

    finally:
        env.close()


if __name__ == "__main__":
    main()
