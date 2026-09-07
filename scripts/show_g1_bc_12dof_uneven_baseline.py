import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IL_ROOT = (
    PROJECT_ROOT.parent
    / "unitree-g1-rl-il-walking(OpenHE IL - Kinematic BC)"
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
    / "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)

CHECKPOINT = (
    IL_ROOT
    / "models"
    / "g1_bc_openhe_kinematic_12dof.pt"
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
# LARGER UNEVEN TERRAIN
#
# Robot travels mainly toward NEGATIVE X.
#
# x_start, x_end, top_height
# ============================================================

TERRAIN = [
    (-3.60, -3.20, 0.14),
    (-3.20, -2.80, 0.07),
    (-2.80, -2.40, 0.18),
    (-2.40, -2.00, 0.09),
    (-2.00, -1.60, 0.16),
    (-1.60, -1.20, 0.05),
    (-1.20, -0.80, 0.12),
    (-0.80, -0.40, 0.06),
    (-0.40, 0.40, 0.02),
]

BASE_TERRAIN_HEIGHT = 0.02


# ============================================================
# SAME IL MOTION SETTINGS
# ============================================================

ROOT_MOTION_SCALE = 0.45

MOTION_FRAME_INCREMENT = 0.42

INITIAL_STAND_STEPS = 60

TRANSITION_STEPS = 90

DEMO_STOP_STEP = 750

REAL_TIME_SLEEP = 0.007


# ============================================================
# 12-DOF BC NETWORK
# ============================================================

class BCPolicy12DOF(nn.Module):

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        output_dim=12,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Trained 12-DOF BC baseline "
            "on larger uneven terrain. "
            "Kinematic only."
        )
    )

    parser.add_argument(
        "--azimuth",
        type=float,
        default=135.0,
        help=(
            "Camera azimuth in degrees. "
            "Examples: 0, 90, 135, 180, 270."
        ),
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=5.0,
        help="Camera distance.",
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=-18.0,
        help="Camera elevation.",
    )

    parser.add_argument(
        "--lookat_z_offset",
        type=float,
        default=-0.10,
        help="Camera vertical look-at offset.",
    )

    parser.add_argument(
        "--no_camera_follow",
        action="store_true",
        help=(
            "Keep the camera look-at fixed "
            "instead of following the robot."
        ),
    )

    parser.add_argument(
        "--sleep_time",
        type=float,
        default=REAL_TIME_SLEEP,
        help="Viewer delay per frame.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def smoothstep(
    value,
):

    value = float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )

    return (
        value
        *
        value
        *
        (
            3.0
            -
            2.0
            *
            value
        )
    )


def transition_blend(
    step,
):

    if (
        step
        <
        INITIAL_STAND_STEPS
    ):
        return 0.0

    transition_step = (
        step
        -
        INITIAL_STAND_STEPS
    )

    return smoothstep(
        transition_step
        /
        max(
            TRANSITION_STEPS,
            1,
        )
    )


def terrain_height(
    x,
):

    x = float(
        x
    )

    for (
        start,
        end,
        height,
    ) in TERRAIN:

        if (
            start
            <=
            x
            <
            end
        ):
            return float(
                height
            )

    return BASE_TERRAIN_HEIGHT


def smooth_root_terrain_height(
    x,
    half_window=0.16,
    samples=9,
):
    """
    Presentation-only root-height rule.

    Because there is no physics, the robot's root
    will not automatically rise when it reaches a
    higher terrain block.

    This is NOT learned by BC or PPO.
    """

    offsets = np.linspace(
        -half_window,
        half_window,
        samples,
    )

    heights = [
        terrain_height(
            float(
                x
            )
            +
            float(
                offset
            )
        )
        for offset in offsets
    ]

    return float(
        np.mean(
            heights
        )
    )


def to_numpy(
    value,
):

    if isinstance(
        value,
        torch.Tensor,
    ):

        return (
            value
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    return np.asarray(
        value,
        dtype=np.float32,
    )


def yaw_to_quat_wxyz(
    yaw_degrees,
):

    yaw_radians = np.deg2rad(
        float(
            yaw_degrees
        )
    )

    half = (
        0.5
        *
        yaw_radians
    )

    return np.array(
        [
            np.cos(
                half
            ),
            0.0,
            0.0,
            np.sin(
                half
            ),
        ],
        dtype=np.float64,
    )


# ============================================================
# TERRAIN SCENE
# ============================================================

def make_scene():

    source = (
        MODEL_DIR
        /
        "scene.xml"
    )

    if not source.exists():

        raise FileNotFoundError(
            f"G1 scene not found:\n"
            f"{source}"
        )

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:

        raise RuntimeError(
            "No <worldbody> found "
            "in G1 scene.xml."
        )

    # --------------------------------------------------------
    # MAIN TERRAIN
    # --------------------------------------------------------

    for (
        i,
        (
            start,
            end,
            height,
        ),
    ) in enumerate(
        TERRAIN
    ):

        length = (
            end
            -
            start
        )

        center_x = (
            start
            +
            end
        ) / 2.0

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name":
                    f"rl_terrain_block_{i}",

                "type":
                    "box",

                "pos":
                    f"{center_x} "
                    f"0 "
                    f"{height / 2.0}",

                "size":
                    f"{length / 2.0} "
                    f"0.62 "
                    f"{height / 2.0}",

                "rgba":
                    "0.43 0.31 0.18 1",

                "friction":
                    "1.0 0.005 0.0001",

                "condim":
                    "3",
            },
        )

    # --------------------------------------------------------
    # SIDE TERRAIN FOR VISUAL DEPTH
    # --------------------------------------------------------

    extras = [
        (-0.95, 0.80, 0.16),
        (-1.45, -0.80, 0.11),
        (-2.05, 0.80, 0.21),
        (-2.55, -0.80, 0.13),
        (-3.05, 0.80, 0.19),
    ]

    for (
        i,
        (
            x,
            y,
            height,
        ),
    ) in enumerate(
        extras
    ):

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name":
                    f"rl_side_terrain_{i}",

                "type":
                    "box",

                "pos":
                    f"{x} "
                    f"{y} "
                    f"{height / 2.0}",

                "size":
                    f"0.27 "
                    f"0.18 "
                    f"{height / 2.0}",

                "rgba":
                    "0.34 0.27 0.16 1",
            },
        )

    output = (
        MODEL_DIR
        /
        "_kinematic_rl_uneven_baseline.xml"
    )

    tree.write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )

    return output


# ============================================================
# JOINT LOOKUP
# ============================================================

def get_joint_addresses(
    model,
):

    addresses = []

    for name in JOINT_NAMES:

        joint_id = (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
        )

        if (
            joint_id
            <
            0
        ):

            raise RuntimeError(
                f"Joint not found: "
                f"{name}"
            )

        addresses.append(
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
        )

    return addresses


# ============================================================
# LOAD TRAINED BC
# ============================================================

def load_bc_policy(
    device,
):

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "12-DOF BC checkpoint "
            f"not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = BCPolicy12DOF(
        input_dim=
            int(
                checkpoint[
                    "input_dim"
                ]
            ),

        hidden_dim=
            int(
                checkpoint[
                    "hidden_dim"
                ]
            ),

        output_dim=
            int(
                checkpoint[
                    "output_dim"
                ]
            ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    normalization = {

        "x_mean":
            to_numpy(
                checkpoint[
                    "x_mean"
                ]
            ),

        "x_std":
            to_numpy(
                checkpoint[
                    "x_std"
                ]
            ),

        "y_mean":
            to_numpy(
                checkpoint[
                    "y_mean"
                ]
            ),

        "y_std":
            to_numpy(
                checkpoint[
                    "y_std"
                ]
            ),
    }

    return (
        model,
        checkpoint,
        normalization,
    )


# ============================================================
# BC PREDICTION
# ============================================================

def predict_all_leg_frames(
    model,
    observations,
    normalization,
    device,
):

    x = (
        (
            observations
            -
            normalization[
                "x_mean"
            ]
        )
        /
        normalization[
            "x_std"
        ]
    ).astype(
        np.float32
    )

    x_tensor = (
        torch
        .from_numpy(
            x
        )
        .to(
            device
        )
    )

    with torch.no_grad():

        normalized_output = (
            model(
                x_tensor
            )
            .cpu()
            .numpy()
        )

    output = (
        normalized_output
        *
        normalization[
            "y_std"
        ]
        +
        normalization[
            "y_mean"
        ]
    )

    return output.astype(
        np.float32
    )


def interpolate_array(
    frames,
    motion_frame,
):

    count = (
        frames.shape[
            0
        ]
    )

    frame = float(
        np.clip(
            motion_frame,
            0.0,
            float(
                count
                -
                1
            ),
        )
    )

    frame_0 = int(
        np.floor(
            frame
        )
    )

    frame_1 = min(
        frame_0
        +
        1,
        count
        -
        1,
    )

    alpha = (
        frame
        -
        frame_0
    )

    return (
        (
            1.0
            -
            alpha
        )
        *
        frames[
            frame_0
        ]
        +
        alpha
        *
        frames[
            frame_1
        ]
    ).astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not DATASET.exists():

        raise FileNotFoundError(
            "Approved IL dataset "
            f"not found:\n{DATASET}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    motion = np.load(
        DATASET,
        allow_pickle=True,
    )

    observations = np.asarray(
        motion[
            "il_observations"
        ],
        dtype=np.float32,
    )

    expert_actions_15 = np.asarray(
        motion[
            "il_actions"
        ],
        dtype=np.float32,
    )

    root_positions = np.asarray(
        motion[
            "root_positions"
        ],
        dtype=np.float32,
    )

    dataset_joint_names = [
        str(
            name
        )
        for name in
        motion[
            "controlled_joint_names"
        ].tolist()
    ]

    if (
        observations.shape
        !=
        (300, 3)
    ):

        raise ValueError(
            "Expected observations "
            "(300, 3), got "
            f"{observations.shape}"
        )

    if (
        expert_actions_15.shape
        !=
        (300, 15)
    ):

        raise ValueError(
            "Expected actions "
            "(300, 15), got "
            f"{expert_actions_15.shape}"
        )

    if (
        dataset_joint_names
        !=
        JOINT_NAMES
    ):

        raise ValueError(
            "Dataset joint order does "
            "not match this script."
        )

    (
        bc_model,
        checkpoint,
        normalization,
    ) = load_bc_policy(
        device
    )

    checkpoint_joint_names = [
        str(
            name
        )
        for name in
        checkpoint[
            "output_features"
        ]
    ]

    if (
        checkpoint_joint_names
        !=
        JOINT_NAMES[
            :12
        ]
    ):

        raise ValueError(
            "Checkpoint learned-joint "
            "order does not match "
            "the first 12 dataset joints."
        )

    fixed_waist_values = np.asarray(
        checkpoint[
            "fixed_waist_values"
        ],
        dtype=np.float32,
    )

    if (
        fixed_waist_values.shape
        !=
        (3,)
    ):

        raise ValueError(
            "Expected exactly three "
            "fixed waist values."
        )

    # ========================================================
    # TRAINED BC OUTPUT FOR ALL ORIGINAL FRAMES
    # ========================================================

    predicted_leg_frames = (
        predict_all_leg_frames(
            bc_model,
            observations,
            normalization,
            device,
        )
    )

    integer_error = (
        predicted_leg_frames
        -
        expert_actions_15[
            :,
            :12
        ]
    )

    integer_rmse = float(
        np.sqrt(
            np.mean(
                integer_error
                .astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    # ========================================================
    # CREATE TERRAIN
    # ========================================================

    scene = make_scene()

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                scene
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    if (
        model.nkey
        >
        0
    ):

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

    joint_addresses = (
        get_joint_addresses(
            model
        )
    )

    stand_joint_positions = np.array(
        [
            data.qpos[
                address
            ]
            for address
            in joint_addresses
        ],
        dtype=np.float32,
    )

    stand_root_z = float(
        data.qpos[
            2
        ]
    )

    if (
        stand_root_z
        <
        0.70
    ):

        stand_root_z = (
            0.79
        )

    root_start = (
        root_positions[
            0
        ]
        .copy()
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        "=" * 90
    )

    print(
        "UNITREE G1 + TRAINED 12-DOF BC BASELINE "
        "+ LARGER UNEVEN TERRAIN"
    )

    print(
        "=" * 90
    )

    print(
        "Policy:"
    )

    print(
        CHECKPOINT
    )

    print()

    print(
        "Dataset:"
    )

    print(
        DATASET
    )

    print()

    print(
        "Integer-frame BC leg RMSE:",
        f"{integer_rmse:.8f} rad",
    )

    print(
        "Terrain top-height range: "
        "0.02 m to 0.18 m"
    )

    print(
        "BC learned outputs: "
        "12 leg joint positions"
    )

    print(
        "Waist: fixed to [0, 0, 0]"
    )

    print(
        "PPO correction: NONE"
    )

    print(
        "MuJoCo physics: NONE"
    )

    print(
        "mj_step(): NEVER CALLED"
    )

    print(
        "Root Z terrain rule: "
        "PRESENTATION ONLY"
    )

    print()

    print(
        "Camera azimuth:",
        args.azimuth,
    )

    print(
        "Camera distance:",
        args.distance,
    )

    print(
        "Camera elevation:",
        args.elevation,
    )

    print(
        "Camera follow:",
        not args.no_camera_follow,
    )

    print(
        "=" * 90
    )

    # ========================================================
    # SHOWCASE
    # ========================================================

    motion_frame = (
        0.0
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.distance = (
            args.distance
        )

        viewer.cam.azimuth = (
            args.azimuth
        )

        viewer.cam.elevation = (
            args.elevation
        )

        viewer.cam.lookat[:] = [
            float(
                data.qpos[
                    0
                ]
            ),

            float(
                data.qpos[
                    1
                ]
            ),

            float(
                data.qpos[
                    2
                ]
                +
                args.lookat_z_offset
            ),
        ]

        for step in range(
            DEMO_STOP_STEP
            +
            1
        ):

            if not viewer.is_running():
                break

            blend = (
                transition_blend(
                    step
                )
            )

            # ------------------------------------------------
            # TRAINED BC LEGS
            # ------------------------------------------------

            predicted_legs = (
                interpolate_array(
                    predicted_leg_frames,
                    motion_frame,
                )
            )

            predicted_walk_15 = np.zeros(
                15,
                dtype=np.float32,
            )

            predicted_walk_15[
                :12
            ] = (
                predicted_legs
            )

            predicted_walk_15[
                12:15
            ] = (
                fixed_waist_values
            )

            applied_joints = (
                (
                    1.0
                    -
                    blend
                )
                *
                stand_joint_positions
                +
                blend
                *
                predicted_walk_15
            ).astype(
                np.float32
            )

            # ------------------------------------------------
            # REFERENCE ROOT TRAJECTORY
            # ------------------------------------------------

            root_pos = (
                interpolate_array(
                    root_positions,
                    motion_frame,
                )
            )

            root_delta = (
                root_pos
                -
                root_start
            )

            world_x = float(
                blend
                *
                ROOT_MOTION_SCALE
                *
                root_delta[
                    0
                ]
            )

            world_y = float(
                blend
                *
                ROOT_MOTION_SCALE
                *
                root_delta[
                    1
                ]
            )

            # ------------------------------------------------
            # SCRIPTED ROOT HEIGHT FOR KINEMATIC TERRAIN
            # ------------------------------------------------

            root_terrain_z = (
                smooth_root_terrain_height(
                    world_x
                )
            )

            data.qpos[
                0
            ] = (
                world_x
            )

            data.qpos[
                1
            ] = (
                world_y
            )

            data.qpos[
                2
            ] = float(
                (
                    1.0
                    -
                    blend
                )
                *
                stand_root_z
                +
                blend
                *
                (
                    root_pos[
                        2
                    ]
                    +
                    root_terrain_z
                )
            )

            # Same walking presentation yaw
            # as our final IL showcase.

            data.qpos[
                3:7
            ] = (
                yaw_to_quat_wxyz(
                    180.0
                )
            )

            # ------------------------------------------------
            # APPLY 12 BC JOINTS + FIXED WAIST
            # ------------------------------------------------

            for (
                i,
                qpos_address,
            ) in enumerate(
                joint_addresses
            ):

                data.qpos[
                    qpos_address
                ] = float(
                    applied_joints[
                        i
                    ]
                )

            data.qvel[
                :
            ] = 0.0

            # ------------------------------------------------
            # KINEMATICS ONLY
            # ------------------------------------------------

            mujoco.mj_forward(
                model,
                data,
            )

            # ------------------------------------------------
            # OPTIONAL FOLLOW CAMERA
            #
            # Azimuth is NEVER overwritten.
            # Therefore --azimuth always controls viewing angle.
            # ------------------------------------------------

            if not args.no_camera_follow:

                viewer.cam.lookat[
                    0
                ] = float(
                    data.qpos[
                        0
                    ]
                    -
                    0.30
                )

                viewer.cam.lookat[
                    1
                ] = float(
                    data.qpos[
                        1
                    ]
                )

                viewer.cam.lookat[
                    2
                ] = float(
                    data.qpos[
                        2
                    ]
                    +
                    args.lookat_z_offset
                )

            if (
                step
                %
                100
                ==
                0
            ):

                print(
                    f"step={step:04d}, "
                    f"motion_frame="
                    f"{motion_frame:7.2f}, "
                    f"x="
                    f"{world_x:+.3f}, "
                    f"y="
                    f"{world_y:+.3f}, "
                    f"terrain_h="
                    f"{terrain_height(world_x):.3f}, "
                    f"root_h_smooth="
                    f"{root_terrain_z:.3f}, "
                    f"blend="
                    f"{blend:.3f}"
                )

            viewer.sync()

            time.sleep(
                args.sleep_time
            )

            # ------------------------------------------------
            # SAME MOTION PROGRESSION
            # ------------------------------------------------

            if (
                step
                >=
                INITIAL_STAND_STEPS
            ):

                motion_frame += (
                    MOTION_FRAME_INCREMENT
                )

                if (
                    motion_frame
                    >=
                    len(
                        predicted_leg_frames
                    )
                    -
                    1
                ):

                    print()

                    print(
                        "Reached the end of "
                        "the approved 300-frame "
                        "IL demonstration."
                    )

                    break

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "BASELINE FINISHED"
    )

    print(
        "=" * 90
    )

    print(
        "Policy used: "
        "trained 12-DOF kinematic BC"
    )

    print(
        "PPO used: NO"
    )

    print(
        "Physics used: NO"
    )

    print(
        "Terrain adaptation learned: NO"
    )

    print(
        "This is the pre-RL "
        "comparison baseline."
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()