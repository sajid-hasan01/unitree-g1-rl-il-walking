import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)


def obj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    if name is None:
        return f"<unnamed:{int(obj_id)}>"
    return str(name)


def geom_info(model, geom_id):
    geom_id = int(geom_id)
    body_id = int(model.geom_bodyid[geom_id])

    return {
        "geom_id": geom_id,
        "geom_name": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body_id": body_id,
        "body_name": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
    }


def initialize_from_reference(env, start_frame):
    start_frame = int(start_frame) % env.num_frames

    env.episode_step = env.initial_stand_steps + env.transition_steps
    env._apply_reference_state_initialization(start_frame)

    env.episode_step = env.initial_stand_steps + env.transition_steps
    env.motion_frame = float(start_frame)
    env.rsi_active_this_episode = True
    env.rsi_frame_this_episode = int(start_frame)

    env._update_previous_foot_positions()

    left_pos = env._get_site_position(env.left_foot_site_id)
    right_pos = env._get_site_position(env.right_foot_site_id)

    env.ground_foot_height = float(min(left_pos[2], right_pos[2]))

    left_contact, right_contact = env._get_foot_contacts()

    env.last_foot_info = {
        "left_contact": bool(left_contact),
        "right_contact": bool(right_contact),
        "left_foot_slip": 0.0,
        "right_foot_slip": 0.0,
        "left_foot_clearance": float(max(left_pos[2] - env.ground_foot_height, 0.0)),
        "right_foot_clearance": float(max(right_pos[2] - env.ground_foot_height, 0.0)),
    }


def print_contact_report(env, label):
    left_site = env._get_site_position(env.left_foot_site_id)
    right_site = env._get_site_position(env.right_foot_site_id)

    left_expected, right_expected = env._get_reference_contact_for_step()
    left_contact, right_contact = env._get_foot_contacts()

    print()
    print("-" * 100)
    print(label)
    print("-" * 100)
    print("episode_step:", env.episode_step)
    print("motion_frame:", env.motion_frame)
    print("root xyz:", env.data.qpos[0:3].copy())
    print("root qvel xyz:", env.data.qvel[0:3].copy())
    print("up_z:", env._get_up_z())
    print("left site xyz:", left_site)
    print("right site xyz:", right_site)
    print("site z difference R-L:", float(right_site[2] - left_site[2]))
    print(
        "env foot contact:",
        f"L={left_contact}/expected={left_expected}",
        f"R={right_contact}/expected={right_expected}",
    )
    print("number of MuJoCo contacts:", env.data.ncon)
    print("left foot body ids:", sorted(list(env.left_foot_body_ids)))
    print("right foot body ids:", sorted(list(env.right_foot_body_ids)))
    print()

    if env.data.ncon == 0:
        print("No contacts.")
        return

    ground_candidates = {
        "floor",
        "ground",
        "plane",
        "world",
    }

    for contact_id in range(env.data.ncon):
        contact = env.data.contact[contact_id]

        g1 = geom_info(env.model, contact.geom1)
        g2 = geom_info(env.model, contact.geom2)

        bodies = {g1["body_id"], g2["body_id"]}

        touches_left = bool(bodies & env.left_foot_body_ids)
        touches_right = bool(bodies & env.right_foot_body_ids)

        g1_is_ground = g1["geom_name"].lower() in ground_candidates
        g2_is_ground = g2["geom_name"].lower() in ground_candidates
        touches_ground_named = g1_is_ground or g2_is_ground

        print(
            f"contact[{contact_id:02d}] "
            f"dist={float(contact.dist):+.6f} "
            f"pos=({contact.pos[0]:+.4f}, {contact.pos[1]:+.4f}, {contact.pos[2]:+.4f}) "
            f"normal=({contact.frame[0]:+.3f}, {contact.frame[1]:+.3f}, {contact.frame[2]:+.3f})"
        )
        print(
            "    geom1:",
            f"{g1['geom_name']} / body={g1['body_name']}({g1['body_id']})",
        )
        print(
            "    geom2:",
            f"{g2['geom_name']} / body={g2['body_name']}({g2['body_id']})",
        )
        print(
            "    flags:",
            f"touches_left={touches_left}",
            f"touches_right={touches_right}",
            f"touches_named_ground={touches_ground_named}",
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect exact MuJoCo contact pairs for Unitree G1 reference frames. "
            "Use this to debug false foot-contact detection and unexpected swing-foot contact."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--target_velocity", type=float, default=-0.10)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--reference_speed", type=float, default=0.10)
    parser.add_argument("--start_frame", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    parser.add_argument("--height_offset", type=float, default=0.02)

    args = parser.parse_args()

    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        action_scale=0.055,
        action_target_smoothing=args.action_target_smoothing,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=70,
        transition_steps=220,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    env.reset()
    initialize_from_reference(env, args.start_frame)

    print()
    print("=" * 100)
    print("DYNAMIC CONTACT PAIR INSPECTION")
    print("=" * 100)
    print("Dataset:", args.dataset_path)
    print("start_frame:", args.start_frame)
    print("target_velocity:", args.target_velocity)
    print("initial_yaw_degrees:", args.initial_yaw_degrees)
    print("reference_speed:", args.reference_speed)
    print("height_offset:", args.height_offset)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    print_contact_report(env, "INITIAL REFERENCE STATE BEFORE STEP")

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % max(args.print_every, 1) == 0:
            print_contact_report(
                env,
                (
                    f"AFTER STEP {step + 1} "
                    f"reward={reward:.4f} "
                    f"terminated={terminated} truncated={truncated}"
                ),
            )

        if terminated or truncated:
            break

    env.close()
    print("=" * 100)


if __name__ == "__main__":
    main()
