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
    test_path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_residual_test_module",
        test_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load: {test_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def get_yaw_degrees(env):
    quat = np.asarray(
        env.data.qpos[3:7],
        dtype=np.float64,
    )

    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, quat)

    rotation = flat.reshape(3, 3)

    yaw = np.arctan2(
        rotation[1, 0],
        rotation[0, 0],
    )

    return float(np.degrees(yaw))


def angle_difference_degrees(current, previous):
    difference = current - previous

    while difference > 180.0:
        difference -= 360.0

    while difference < -180.0:
        difference += 360.0

    return difference


def geom_name(model, geom_id):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        int(geom_id),
    )

    if name is None:
        return f"geom{int(geom_id)}"

    return str(name)


def get_foot_geom_sets(env):
    left = set()
    right = set()

    for geom_id in range(env.model.ngeom):
        body_id = int(
            env.model.geom_bodyid[geom_id]
        )

        if body_id in env.left_foot_body_ids:
            left.add(int(geom_id))

        if body_id in env.right_foot_body_ids:
            right.add(int(geom_id))

    return left, right


def contact_force_on_foot_world(
    env,
    contact_id,
    foot_geom_ids,
    floor_geom_id,
):
    """
    Return the ground-reaction force acting on the foot.

    MuJoCo's contact frame axes are stored as rows:
        row 0 = normal
        row 1 = tangent 1
        row 2 = tangent 2

    mj_contactForce() returns force components in that
    contact frame.

    The contact normal points geom1 -> geom2.

    Therefore:
      floor=geom1, foot=geom2:
          transformed force already points as the force
          acting on the foot.

      foot=geom1, floor=geom2:
          invert the transformed force to express the
          ground-reaction force acting on the foot.
    """
    contact = env.data.contact[contact_id]

    g1 = int(contact.geom1)
    g2 = int(contact.geom2)

    pair_is_relevant = (
        (
            g1 == floor_geom_id
            and g2 in foot_geom_ids
        )
        or
        (
            g2 == floor_geom_id
            and g1 in foot_geom_ids
        )
    )

    if not pair_is_relevant:
        return None

    force6 = np.zeros(
        6,
        dtype=np.float64,
    )

    mujoco.mj_contactForce(
        env.model,
        env.data,
        contact_id,
        force6,
    )

    # Contact axes are stored in rows.
    contact_frame = np.asarray(
        contact.frame,
        dtype=np.float64,
    ).reshape(3, 3)

    force_world_geom2 = (
        contact_frame.T @ force6[:3]
    )

    if g2 in foot_geom_ids:
        foot_force_world = force_world_geom2
        foot_geom = g2
    else:
        foot_force_world = -force_world_geom2
        foot_geom = g1

    return {
        "contact_id": int(contact_id),
        "foot_geom": int(foot_geom),
        "foot_geom_name": geom_name(
            env.model,
            foot_geom,
        ),
        "contact_position": np.asarray(
            contact.pos,
            dtype=np.float64,
        ).copy(),
        "distance": float(contact.dist),
        "local_force": force6[:3].copy(),
        "world_force": (
            foot_force_world.copy()
        ),
    }


def collect_foot_wrench(
    env,
    foot_geom_ids,
    floor_geom_id,
):
    contacts = []

    total_force = np.zeros(
        3,
        dtype=np.float64,
    )

    total_moment_about_root = np.zeros(
        3,
        dtype=np.float64,
    )

    root_position = np.asarray(
        env.data.qpos[0:3],
        dtype=np.float64,
    )

    for contact_id in range(env.data.ncon):
        result = contact_force_on_foot_world(
            env,
            contact_id,
            foot_geom_ids,
            floor_geom_id,
        )

        if result is None:
            continue

        force = result["world_force"]
        position = result["contact_position"]

        lever_arm = position - root_position

        moment = np.cross(
            lever_arm,
            force,
        )

        result["moment_about_root"] = (
            moment.copy()
        )

        total_force += force
        total_moment_about_root += moment

        contacts.append(result)

    horizontal_force = float(
        np.linalg.norm(total_force[:2])
    )

    vertical_force = float(
        total_force[2]
    )

    if abs(vertical_force) > 1e-8:
        horizontal_vertical_ratio = (
            horizontal_force
            / abs(vertical_force)
        )
    else:
        horizontal_vertical_ratio = 0.0

    return {
        "contacts": contacts,
        "total_force": total_force,
        "total_moment": (
            total_moment_about_root
        ),
        "horizontal_force": (
            horizontal_force
        ),
        "vertical_force": (
            vertical_force
        ),
        "friction_ratio": (
            horizontal_vertical_ratio
        ),
    }


def contact_geom_string(wrench):
    if not wrench["contacts"]:
        return "-"

    return ",".join(
        str(item["foot_geom"])
        for item in wrench["contacts"]
    )


def print_detailed_contacts(
    side,
    wrench,
):
    if not wrench["contacts"]:
        print(
            f"    {side}: no floor contact"
        )
        return

    for item in wrench["contacts"]:
        force = item["world_force"]
        moment = item["moment_about_root"]
        position = item["contact_position"]

        print(
            f"    {side} geom={item['foot_geom']:2d} "
            f"dist={item['distance']:+.6f} "
            f"pos=({position[0]:+.3f},"
            f"{position[1]:+.3f},"
            f"{position[2]:+.3f}) "
            f"Fworld=({force[0]:+.1f},"
            f"{force[1]:+.1f},"
            f"{force[2]:+.1f}) N "
            f"Mz(root)={moment[2]:+.3f} Nm"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure G1 foot-ground contact forces "
            "and yaw moments during the OpenHE "
            "zero-residual rollout."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--target_velocity",
        type=float,
        default=0.10,
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
        default=1.0,
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
        default=16,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    initialize_from_reference = (
        load_existing_initializer()
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
        max_episode_steps=(
            args.max_episode_steps
        ),
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

        floor_geom_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )

        if floor_geom_id < 0:
            raise RuntimeError(
                "Could not find floor geom."
            )

        left_geoms, right_geoms = (
            get_foot_geom_sets(env)
        )

        zero_action = np.zeros(
            env.num_actions,
            dtype=np.float32,
        )

        previous_yaw = get_yaw_degrees(
            env
        )

        previous_contact_signature = None

        print()
        print("=" * 132)
        print(
            "OPENHE CONTACT FORCE / "
            "YAW DIAGNOSTIC"
        )
        print("=" * 132)

        print(
            "Left foot geom IDs :",
            sorted(left_geoms),
        )

        print(
            "Right foot geom IDs:",
            sorted(right_geoms),
        )

        print(
            "Floor geom ID      :",
            floor_geom_id,
        )

        print()
        print(
            "step frm   yaw    yawRate | "
            "Lgeom  LFz    LFxy   LMz     | "
            "Rgeom  RFz    RFxy   RMz     | "
            "netMz    yVel"
        )
        print("-" * 132)

        for step in range(args.steps):
            control_frame = float(
                env.motion_frame
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(zero_action)

            yaw = get_yaw_degrees(env)

            yaw_delta = (
                angle_difference_degrees(
                    yaw,
                    previous_yaw,
                )
            )

            yaw_rate = (
                yaw_delta
                / max(env.control_dt, 1e-8)
            )

            previous_yaw = yaw

            left = collect_foot_wrench(
                env,
                left_geoms,
                floor_geom_id,
            )

            right = collect_foot_wrench(
                env,
                right_geoms,
                floor_geom_id,
            )

            left_ids = contact_geom_string(
                left
            )

            right_ids = contact_geom_string(
                right
            )

            signature = (
                left_ids,
                right_ids,
            )

            contact_changed = (
                signature
                != previous_contact_signature
            )

            should_print = (
                step % args.print_every == 0
                or contact_changed
            )

            if should_print:
                net_mz = (
                    left["total_moment"][2]
                    + right["total_moment"][2]
                )

                print(
                    f"{step:4d} "
                    f"{control_frame:5.1f} "
                    f"{yaw:+7.2f} "
                    f"{yaw_rate:+8.1f} | "
                    f"{left_ids:5s} "
                    f"{left['vertical_force']:+6.1f} "
                    f"{left['horizontal_force']:6.1f} "
                    f"{left['total_moment'][2]:+7.2f} | "
                    f"{right_ids:5s} "
                    f"{right['vertical_force']:+6.1f} "
                    f"{right['horizontal_force']:6.1f} "
                    f"{right['total_moment'][2]:+7.2f} | "
                    f"{net_mz:+7.2f} "
                    f"{float(env.data.qvel[1]):+7.3f}"
                )

            if contact_changed:
                print(
                    "  CONTACT SET CHANGED:"
                )

                print_detailed_contacts(
                    "LEFT ",
                    left,
                )

                print_detailed_contacts(
                    "RIGHT",
                    right,
                )

            previous_contact_signature = (
                signature
            )

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
        print("=" * 132)
        print(
            "Interpretation fields:"
        )
        print(
            "  Fz   = vertical ground-reaction "
            "force on that foot"
        )
        print(
            "  Fxy  = horizontal ground-reaction "
            "force magnitude"
        )
        print(
            "  Mz   = yaw moment produced by the "
            "ground force about the robot root"
        )
        print(
            "  yawRate = finite-difference base yaw "
            "rate in deg/s"
        )
        print("=" * 132)

    finally:
        env.close()


if __name__ == "__main__":
    main()