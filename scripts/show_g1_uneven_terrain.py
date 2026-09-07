from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "Could not find Unitree G1 scene.xml in either "
        "the OpenHE RL or OpenHE IL workspace."
    )


def create_uneven_scene(model_dir):
    source_scene = model_dir / "scene.xml"
    demo_scene = model_dir / "_faculty_uneven_demo.xml"

    tree = ET.parse(source_scene)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "Could not find <worldbody> in scene.xml"
        )

    terrain_blocks = [
        # name, x, y, z, sx, sy, sz
        ("block_0", 0.00,  0.00, 0.025, 0.45, 0.45, 0.025),
        ("block_1", 0.65,  0.00, 0.060, 0.20, 0.45, 0.060),
        ("block_2", 1.08,  0.00, 0.110, 0.20, 0.45, 0.110),
        ("block_3", 1.51,  0.00, 0.045, 0.20, 0.45, 0.045),
        ("block_4", 1.94,  0.00, 0.145, 0.20, 0.45, 0.145),
        ("block_5", 2.37,  0.00, 0.075, 0.20, 0.45, 0.075),

        # Extra irregular side blocks
        ("side_L1", 0.85,  0.58, 0.090, 0.28, 0.18, 0.090),
        ("side_L2", 1.55,  0.58, 0.150, 0.28, 0.18, 0.150),

        ("side_R1", 1.15, -0.58, 0.055, 0.28, 0.18, 0.055),
        ("side_R2", 1.90, -0.58, 0.105, 0.28, 0.18, 0.105),
    ]

    for (
        name,
        x,
        y,
        z,
        sx,
        sy,
        sz,
    ) in terrain_blocks:

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{x} {y} {z}",
                "size": f"{sx} {sy} {sz}",
                "rgba": "0.45 0.32 0.18 1",
                "friction": "1.0 0.005 0.0001",
                "condim": "3",
            },
        )

    tree.write(
        demo_scene,
        encoding="utf-8",
        xml_declaration=True,
    )

    return demo_scene


def main():
    model_dir = find_g1_model_dir()

    print("=" * 72)
    print("UNITREE G1 - UNEVEN TERRAIN FACULTY DEMO")
    print("=" * 72)
    print("G1 model directory:")
    print(model_dir)

    demo_scene = create_uneven_scene(
        model_dir
    )

    print()
    print("Demo scene:")
    print(demo_scene)

    model = mujoco.MjModel.from_xml_path(
        str(demo_scene)
    )

    data = mujoco.MjData(model)

    # Use the G1 default keyframe/standing pose.
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

    # Place robot near the beginning of terrain.
    if model.nq >= 7:
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0

        # Keep it clearly visible above the first block.
        if data.qpos[2] < 0.75:
            data.qpos[2] = 0.80

    mujoco.mj_forward(
        model,
        data,
    )

    print()
    print("G1 loaded successfully.")
    print("Uneven terrain loaded successfully.")
    print()
    print(
        "STATIC DEMO MODE: physics stepping is intentionally disabled."
    )
    print(
        "The robot will remain visible for faculty presentation."
    )
    print("=" * 72)

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Camera for a clear side/angled view.
        viewer.cam.lookat[:] = [
            1.0,
            0.0,
            0.55,
        ]

        viewer.cam.distance = 4.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18

        while viewer.is_running():

            # No mj_step().
            # This deliberately keeps the robot static.
            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()

            time.sleep(0.02)


if __name__ == "__main__":
    main()