from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IL_ROOT = (
    PROJECT_ROOT.parent
    / "unitree-g1-rl-il-walking(OpenHE IL - Dynamic Physics)"
)

MODEL_DIR = (
    IL_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
)

DATASET = (
    IL_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_rawlegs_15dof.npz"
)


JOINT_NAMES = [
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

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


# ============================================================
# TERRAIN
# ============================================================

TERRAIN = [
    # x_start, x_end, top_height
    (-0.45, 0.20, 0.020),
    (0.20, 0.60, 0.045),
    (0.60, 1.00, 0.075),
    (1.00, 1.40, 0.040),
    (1.40, 1.80, 0.095),
    (1.80, 2.20, 0.060),
    (2.20, 2.60, 0.110),
    (2.60, 3.00, 0.050),
]


def terrain_height(x):
    for start, end, height in TERRAIN:
        if start <= x < end:
            return height

    return 0.02


def make_scene():
    source = MODEL_DIR / "scene.xml"

    if not source.exists():
        raise FileNotFoundError(
            f"G1 scene not found:\n{source}"
        )

    tree = ET.parse(source)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "No <worldbody> found."
        )

    for i, (start, end, height) in enumerate(TERRAIN):
        length = end - start
        center_x = (start + end) / 2.0

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"uneven_block_{i}",
                "type": "box",
                "pos": f"{center_x} 0 {height / 2.0}",
                "size": f"{length / 2.0} 0.48 {height / 2.0}",
                "rgba": "0.43 0.31 0.18 1",
                "friction": "1.0 0.005 0.0001",
                "condim": "3",
            },
        )

    # Extra blocks beside the path to make the terrain
    # visibly uneven from the camera.
    extras = [
        (0.40, 0.65, 0.10),
        (1.15, -0.65, 0.14),
        (1.75, 0.65, 0.08),
        (2.35, -0.65, 0.16),
    ]

    for i, (x, y, h) in enumerate(extras):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"side_terrain_{i}",
                "type": "box",
                "pos": f"{x} {y} {h / 2.0}",
                "size": f"0.28 0.16 {h / 2.0}",
                "rgba": "0.34 0.27 0.16 1",
            },
        )

    output = MODEL_DIR / "_faculty_openhe_uneven.xml"

    tree.write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )

    return output


# ============================================================
# JOINT LOOKUP
# ============================================================

def get_joint_addresses(model):
    addresses = []

    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"Joint not found: {name}"
            )

        addresses.append(
            int(
                model.jnt_qposadr[joint_id]
            )
        )

    return addresses


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Processed OpenHE dataset not found:\n{DATASET}"
        )

    scene = make_scene()

    print("=" * 78)
    print("UNITREE G1 + OPENHE REFERENCE + UNEVEN TERRAIN")
    print("=" * 78)

    print("Dataset:")
    print(DATASET)

    print()
    print("Scene:")
    print(scene)

    motion = np.load(
        DATASET,
        allow_pickle=True,
    )

    joint_motion = np.asarray(
        motion["joint_pos_15"],
        dtype=np.float64,
    )

    print()
    print(
        f"Loaded OpenHE frames: {len(joint_motion)}"
    )

    model = mujoco.MjModel.from_xml_path(
        str(scene)
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

    joint_addresses = get_joint_addresses(
        model
    )

    # Preserve the normal G1 standing/root height.
    base_root_z = float(
        data.qpos[2]
    )

    if base_root_z < 0.70:
        base_root_z = 0.79

    # --------------------------------------------------------
    # IMPORTANT:
    # Robot starts RIGHT ON the first terrain section.
    # --------------------------------------------------------

    START_X = -0.25

    # Forward display speed.
    WALK_SPEED = 0.32

    FPS = 30.0

    print()
    print(
        "Robot starts directly at the terrain entrance."
    )

    print(
        "Using real OpenHE reference leg motion."
    )

    print(
        "This is KINEMATIC REFERENCE REPLAY, not trained PPO."
    )

    print("=" * 78)

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.lookat[:] = [
            1.15,
            0.0,
            0.55,
        ]

        viewer.cam.distance = 3.8
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -16

        frame = 0

        while viewer.is_running():

            idx = frame % len(
                joint_motion
            )

            # ------------------------------------------------
            # Apply actual OpenHE/G1 reference joint pose
            # ------------------------------------------------

            pose = joint_motion[idx]

            for j, qpos_address in enumerate(
                joint_addresses
            ):
                data.qpos[
                    qpos_address
                ] = pose[j]

            # ------------------------------------------------
            # Forward movement
            # ------------------------------------------------

            elapsed = (
                frame / FPS
            )

            x = (
                START_X
                + WALK_SPEED
                * elapsed
            )

            # Loop after reaching end of demonstration terrain.
            if x > 2.75:
                frame = 0
                x = START_X

            data.qpos[0] = x
            data.qpos[1] = 0.0

            # ------------------------------------------------
            # Move root vertically with terrain.
            #
            # This is for PRESENTATION ONLY.
            # It is not terrain adaptation learned by RL.
            # ------------------------------------------------

            h = terrain_height(
                x
            )

            data.qpos[2] = (
                base_root_z
                + h
            )

            # Keep body facing straight down terrain.
            data.qpos[3:7] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]

            data.qvel[:] = 0.0

            mujoco.mj_forward(
                model,
                data,
            )

            # Follow robot slightly with camera.
            viewer.cam.lookat[0] = (
                x + 0.8
            )

            viewer.sync()

            time.sleep(
                1.0 / FPS
            )

            frame += 1


if __name__ == "__main__":
    main()
