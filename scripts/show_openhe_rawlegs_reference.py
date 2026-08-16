from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_rawlegs_15dof.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)


def main():
    print("=" * 78)
    print("OPENHE -> UNITREE G1 EXACT RAW-LEGS REFERENCE")
    print("KINEMATIC REPLAY - NO PHYSICS")
    print("=" * 78)

    # ============================================================
    # LOAD DATASET
    # ============================================================

    data_npz = np.load(
        DATASET_PATH,
        allow_pickle=True,
    )

    joint_pos = np.asarray(
        data_npz["joint_pos_15"],
        dtype=np.float64,
    )

    root_positions = np.asarray(
        data_npz["root_positions"],
        dtype=np.float64,
    )

    root_rot = np.asarray(
        data_npz["root_rot"],
        dtype=np.float64,
    )

    joint_names = [
        str(name)
        for name in data_npz["controlled_joint_names"]
    ]

    fps = float(
        np.asarray(data_npz["fps"]).reshape(-1)[0]
    )

    num_frames = len(joint_pos)

    print()
    print("Dataset:")
    print(DATASET_PATH)

    print()
    print("Frames:", num_frames)
    print("FPS:", fps)
    print("Duration:", num_frames / fps, "seconds")
    print("Joint pose shape:", joint_pos.shape)
    print("Root position shape:", root_positions.shape)
    print("Root rotation shape:", root_rot.shape)

    # ============================================================
    # LOAD UNITREE G1
    # ============================================================

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            0,
        )
    else:
        mujoco.mj_resetData(
            model,
            data,
        )

    # ============================================================
    # GET QPOS ADDRESSES FOR OUR 15 CONTROLLED JOINTS
    # ============================================================

    qpos_addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"Joint not found in G1 model: {joint_name}"
            )

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        qpos_addresses.append(
            qpos_address
        )

    print()
    print("All 15 controlled G1 joints found.")

    # ============================================================
    # ROOT TRAJECTORY
    # ============================================================

    # Remove only the initial horizontal offset.
    # The relative OpenHE root motion remains unchanged.
    root_xy_start = root_positions[0, :2].copy()

    frame_time = 1.0 / fps

    print()
    print("Playback mode:")
    print("  Expert joint poses : EXACT dataset values")
    print("  Root trajectory    : dataset root_positions")
    print("  Root orientation   : dataset root_rot")
    print("  Physics            : OFF")
    print("  mj_step()          : NOT USED")

    print()
    print("This is the expert motion BC will learn.")
    print("=" * 78)

    # ============================================================
    # VIEWER
    # ============================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        while viewer.is_running():

            for frame in range(num_frames):

                if not viewer.is_running():
                    break

                loop_start = time.perf_counter()

                # =================================================
                # EXACT 15-DOF EXPERT JOINT POSE
                # =================================================

                for i, qpos_address in enumerate(
                    qpos_addresses
                ):
                    data.qpos[qpos_address] = (
                        joint_pos[frame, i]
                    )

                # =================================================
                # EXACT ROOT POSITION
                #
                # Only remove initial XY translation so the
                # demonstration starts around world origin.
                # =================================================

                data.qpos[0] = (
                    root_positions[frame, 0]
                    - root_xy_start[0]
                )

                data.qpos[1] = (
                    root_positions[frame, 1]
                    - root_xy_start[1]
                )

                data.qpos[2] = (
                    root_positions[frame, 2]
                )

                # =================================================
                # ROOT ROTATION
                #
                # OpenHE stores quaternion as:
                # [x, y, z, w]
                #
                # MuJoCo free-joint qpos expects:
                # [w, x, y, z]
                # =================================================

                x = root_rot[frame, 0]
                y = root_rot[frame, 1]
                z = root_rot[frame, 2]
                w = root_rot[frame, 3]

                data.qpos[3:7] = [
                    w,
                    x,
                    y,
                    z,
                ]

                # No dynamic velocity integration.
                data.qvel[:] = 0.0

                # Forward kinematics only.
                mujoco.mj_forward(
                    model,
                    data,
                )

                # =================================================
                # CAMERA FOLLOWS ROOT
                # =================================================

                viewer.cam.lookat[:] = [
                    data.qpos[0],
                    data.qpos[1],
                    0.55,
                ]

                viewer.sync()

                # =================================================
                # REAL-TIME 30 FPS PLAYBACK
                # =================================================

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                remaining = (
                    frame_time
                    - elapsed
                )

                if remaining > 0:
                    time.sleep(
                        remaining
                    )


if __name__ == "__main__":
    main()