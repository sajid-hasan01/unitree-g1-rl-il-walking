from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FIND UNITREE G1 MODEL
# ============================================================

def find_g1_model_dir():
    candidates = [
        PROJECT_ROOT
        / "third_party"
        / "mujoco_menagerie"
        / "unitree_g1",

        PROJECT_ROOT.parent
        / "unitree-g1-rl-il-walking(OpenHE IL)"
        / "third_party"
        / "mujoco_menagerie"
        / "unitree_g1",
    ]

    for path in candidates:
        if (path / "scene.xml").exists():
            return path

    raise FileNotFoundError(
        "Could not find Unitree G1 scene.xml."
    )


# ============================================================
# CREATE UNEVEN TERRAIN
# ============================================================

def create_scene(model_dir):
    source_scene = model_dir / "scene.xml"
    demo_scene = model_dir / "_faculty_uneven_walk_demo.xml"

    tree = ET.parse(source_scene)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "Could not find <worldbody>."
        )

    # Robot starts around x = -0.7 on the original flat floor.
    # Uneven terrain begins ahead of it.

    terrain = [
        # name, x, y, height, half_x, half_y

        ("step_1", 0.55, 0.0, 0.025, 0.20, 0.42),
        ("step_2", 0.98, 0.0, 0.050, 0.20, 0.42),
        ("step_3", 1.41, 0.0, 0.020, 0.20, 0.42),
        ("step_4", 1.84, 0.0, 0.075, 0.20, 0.42),
        ("step_5", 2.27, 0.0, 0.040, 0.20, 0.42),
        ("step_6", 2.70, 0.0, 0.090, 0.20, 0.42),
    ]

    for (
        name,
        x,
        y,
        height,
        sx,
        sy,
    ) in terrain:

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",

                # Box center is half its height
                "pos": f"{x} {y} {height / 2.0}",

                "size": f"{sx} {sy} {height / 2.0}",

                "rgba": "0.42 0.32 0.20 1",

                "friction": "1.0 0.005 0.0001",

                "condim": "3",
            },
        )

    # A few irregular side pieces for appearance

    side_blocks = [
        (
            "side_left_1",
            1.15,
            0.62,
            0.10,
            0.30,
            0.16,
            0.10,
        ),
        (
            "side_left_2",
            2.00,
            0.62,
            0.16,
            0.30,
            0.16,
            0.16,
        ),
        (
            "side_right_1",
            1.50,
            -0.62,
            0.07,
            0.30,
            0.16,
            0.07,
        ),
        (
            "side_right_2",
            2.40,
            -0.62,
            0.13,
            0.30,
            0.16,
            0.13,
        ),
    ]

    for (
        name,
        x,
        y,
        height,
        sx,
        sy,
        sz,
    ) in side_blocks:

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{x} {y} {height / 2.0}",
                "size": f"{sx} {sy} {sz / 2.0}",
                "rgba": "0.35 0.28 0.18 1",
            },
        )

    tree.write(
        demo_scene,
        encoding="utf-8",
        xml_declaration=True,
    )

    return demo_scene


# ============================================================
# ACTUATOR LOOKUP
# ============================================================

def actuator_for_joint(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    for actuator_id in range(model.nu):

        if int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        ) == joint_id:

            return actuator_id

    raise RuntimeError(
        f"No actuator found for {joint_name}"
    )


# ============================================================
# INITIAL STANDING TARGET
# ============================================================

def get_stand_targets(model, data):
    targets = np.zeros(
        model.nu,
        dtype=np.float64,
    )

    for actuator_id in range(model.nu):

        joint_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if joint_id < 0:
            targets[actuator_id] = 0.0
            continue

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        targets[actuator_id] = float(
            data.qpos[
                qpos_address
            ]
        )

    return targets


# ============================================================
# MAIN
# ============================================================

def main():

    model_dir = find_g1_model_dir()

    scene_path = create_scene(
        model_dir
    )

    print("=" * 76)
    print("UNITREE G1 - UNEVEN TERRAIN WALKING ATTEMPT")
    print("=" * 76)

    print("Scene:")
    print(scene_path)

    model = mujoco.MjModel.from_xml_path(
        str(scene_path)
    )

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # RESET TO G1 STANDING KEYFRAME
    # --------------------------------------------------------

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

    # Start before uneven terrain
    if model.nq >= 7:

        data.qpos[0] = -0.70
        data.qpos[1] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    # --------------------------------------------------------
    # FIND LEG ACTUATORS
    # --------------------------------------------------------

    joints = {

        "L_hip_pitch":
            "left_hip_pitch_joint",

        "L_hip_roll":
            "left_hip_roll_joint",

        "L_knee":
            "left_knee_joint",

        "L_ankle_pitch":
            "left_ankle_pitch_joint",

        "R_hip_pitch":
            "right_hip_pitch_joint",

        "R_hip_roll":
            "right_hip_roll_joint",

        "R_knee":
            "right_knee_joint",

        "R_ankle_pitch":
            "right_ankle_pitch_joint",
    }

    actuators = {}

    for short_name, joint_name in joints.items():

        actuators[short_name] = (
            actuator_for_joint(
                model,
                joint_name,
            )
        )

    print()
    print("Leg actuators found successfully.")

    # --------------------------------------------------------
    # STANDING POSITION TARGETS
    # --------------------------------------------------------

    stand = get_stand_targets(
        model,
        data,
    )

    data.ctrl[:] = stand

    mujoco.mj_forward(
        model,
        data,
    )

    timestep = float(
        model.opt.timestep
    )

    warmup_time = 1.5

    walking_frequency = 0.65

    print()
    print(
        "Robot will stand briefly, then attempt to walk."
    )

    print(
        "This is a simple open-loop gait, NOT trained PPO."
    )

    print(
        "If it falls, the demo automatically resets."
    )

    print("=" * 76)

    # --------------------------------------------------------
    # VIEWER
    # --------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.lookat[:] = [
            0.9,
            0.0,
            0.55,
        ]

        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        start_time = time.time()

        last_reset = start_time

        while viewer.is_running():

            simulation_time = (
                time.time()
                - last_reset
            )

            # Start from standing targets
            target = stand.copy()

            # =================================================
            # STAND FIRST
            # =================================================

            if simulation_time < warmup_time:

                ramp = 0.0

            else:

                gait_time = (
                    simulation_time
                    - warmup_time
                )

                # Smoothly increase gait amplitude
                ramp = min(
                    1.0,
                    gait_time / 1.0,
                )

                phase = (
                    2.0
                    * math.pi
                    * walking_frequency
                    * gait_time
                )

                s = math.sin(
                    phase
                )

                # ---------------------------------------------
                # Alternating swing
                # ---------------------------------------------

                left_swing = max(
                    0.0,
                    s,
                )

                right_swing = max(
                    0.0,
                    -s,
                )

                # ---------------------------------------------
                # HIP PITCH
                #
                # Negative = leg moves forward for this G1
                # convention used by our OpenHE reference.
                # ---------------------------------------------

                hip_amplitude = (
                    0.18
                    * ramp
                )

                target[
                    actuators[
                        "L_hip_pitch"
                    ]
                ] += (
                    -hip_amplitude
                    * s
                )

                target[
                    actuators[
                        "R_hip_pitch"
                    ]
                ] += (
                    hip_amplitude
                    * s
                )

                # ---------------------------------------------
                # KNEE FLEXION DURING SWING
                # ---------------------------------------------

                knee_lift = (
                    0.28
                    * ramp
                )

                target[
                    actuators[
                        "L_knee"
                    ]
                ] += (
                    knee_lift
                    * left_swing
                )

                target[
                    actuators[
                        "R_knee"
                    ]
                ] += (
                    knee_lift
                    * right_swing
                )

                # ---------------------------------------------
                # ANKLE COMPENSATION
                # ---------------------------------------------

                ankle_amplitude = (
                    0.10
                    * ramp
                )

                target[
                    actuators[
                        "L_ankle_pitch"
                    ]
                ] += (
                    -ankle_amplitude
                    * s
                )

                target[
                    actuators[
                        "R_ankle_pitch"
                    ]
                ] += (
                    ankle_amplitude
                    * s
                )

                # ---------------------------------------------
                # SMALL SIDE-TO-SIDE WEIGHT SHIFT
                # ---------------------------------------------

                roll_amplitude = (
                    0.025
                    * ramp
                )

                target[
                    actuators[
                        "L_hip_roll"
                    ]
                ] += (
                    roll_amplitude
                    * s
                )

                target[
                    actuators[
                        "R_hip_roll"
                    ]
                ] += (
                    roll_amplitude
                    * s
                )

            # -------------------------------------------------
            # SEND TARGETS
            # -------------------------------------------------

            data.ctrl[:] = target

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            # -------------------------------------------------
            # FALL DETECTOR
            # -------------------------------------------------

            if (
                model.nq >= 7
                and data.qpos[2] < 0.50
            ):

                print(
                    "Robot fell -> resetting walking attempt."
                )

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

                data.qpos[0] = -0.70
                data.qpos[1] = 0.0

                mujoco.mj_forward(
                    model,
                    data,
                )

                stand = get_stand_targets(
                    model,
                    data,
                )

                data.ctrl[:] = stand

                last_reset = time.time()

            # Roughly real-time
            time.sleep(
                max(
                    timestep * 0.8,
                    0.001,
                )
            )


if __name__ == "__main__":
    main()