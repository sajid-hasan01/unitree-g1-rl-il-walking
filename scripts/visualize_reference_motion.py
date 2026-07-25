import os
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_XML_PATH = os.path.join(
    "third_party",
    "mujoco_menagerie",
    "unitree_g1",
    "scene.xml",
)


def get_joint_qpos_addresses(model, joint_names):
    qpos_addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")

        qpos_addresses.append(model.jnt_qposadr[joint_id])

    return qpos_addresses


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(
            "datasets",
            "processed",
            "g1_lafan1_walk1_subject1_il_15dof.npz",
        ),
    )

    parser.add_argument("--playback_speed", type=float, default=0.45)

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Positive value lifts the robot, negative value lowers it.",
    )

    parser.add_argument(
        "--forward_axis",
        type=str,
        default="x",
        choices=["x", "y"],
        help="Which column of root_positions to use as MuJoCo forward "
             "(X) motion. Use 'x' for heading-canonicalized LAFAN1 data. "
             "Use 'y' if replaying an older, non-canonicalized AMASS dataset.",
    )

    parser.add_argument(
        "--keep_lateral_drift",
        action="store_true",
        help="If set, keep the original lateral (sideways) drift instead "
             "of zeroing it.",
    )

    parser.add_argument(
        "--use_stand_keyframe",
        action="store_true",
        default=True,
        help="Initialize non-controlled joints (arms, hands) from the "
             "model's stand keyframe each frame, instead of leaving them "
             "at zero.",
    )

    parser.add_argument(
        "--no_stand_keyframe",
        action="store_true",
        help="Disable --use_stand_keyframe.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Processed IL dataset not found: {args.dataset}")

    if not os.path.exists(MODEL_XML_PATH):
        raise FileNotFoundError(f"MuJoCo model not found: {MODEL_XML_PATH}")

    use_stand_keyframe = args.use_stand_keyframe and not args.no_stand_keyframe

    dataset = np.load(args.dataset, allow_pickle=True)

    joint_pos_15 = dataset["joint_pos_15"].astype(np.float32)
    root_positions = dataset["root_positions"].astype(np.float32)
    fps = float(dataset["fps"][0])
    joint_names = [str(name) for name in dataset["controlled_joint_names"]]
    total_frames = joint_pos_15.shape[0]

    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    joint_qpos_addresses = get_joint_qpos_addresses(model, joint_names)

    forward_col = 0 if args.forward_axis == "x" else 1
    lateral_col = 1 if args.forward_axis == "x" else 0

    root_start = root_positions[0].copy()

    print("Visualizing raw IL reference motion (no trained policy)")
    print("Dataset:", args.dataset)
    print("MuJoCo model:", MODEL_XML_PATH)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Forward axis column:", forward_col, f"('{args.forward_axis}')")
    print("Height offset:", args.height_offset)
    print("Keep lateral drift:", args.keep_lateral_drift)
    print("Use stand keyframe for arms:", use_stand_keyframe)
    print(
        "Root height range:",
        float(root_positions[:, 2].min()),
        "to",
        float(root_positions[:, 2].max()),
    )
    print()

    frame_time = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0

        while viewer.is_running():
            frame_id = frame % total_frames

            current_root = root_positions[frame_id]

            if use_stand_keyframe and model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                data.qpos[:] = 0.0
                data.qpos[3] = 1.0  # identity quaternion (w, x, y, z)

            data.qvel[:] = 0.0

            root_forward = float(current_root[forward_col] - root_start[forward_col])

            if args.keep_lateral_drift:
                root_lateral = float(current_root[lateral_col] - root_start[lateral_col])
            else:
                root_lateral = 0.0

            root_z = float(current_root[2]) + args.height_offset

            data.qpos[0] = root_forward
            data.qpos[1] = root_lateral
            data.qpos[2] = root_z

            # Keep the floating base upright; only the 15 controlled
            # joints below are driven by the reference motion.
            data.qpos[3] = 1.0
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = 0.0

            for i, qpos_addr in enumerate(joint_qpos_addresses):
                data.qpos[qpos_addr] = joint_pos_15[frame_id, i]

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame += 1

            time.sleep(frame_time / args.playback_speed)


if __name__ == "__main__":
    main()
