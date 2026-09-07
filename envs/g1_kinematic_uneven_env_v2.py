import importlib.util
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IL_ROOT = (
    PROJECT_ROOT.parent
    / "unitree-g1-rl-il-walking(OpenHE IL - Kinematic BC)"
)

BASELINE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "show_g1_bc_12dof_uneven_baseline.py"
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

FLAT_SCENE = (
    MODEL_DIR
    / "scene.xml"
)

# IMPORTANT:
# This XML already exists and was verified by the
# pre-training audit to match the current baseline terrain
# exactly.
#
# The RL environment READS it only.
#
# It does NOT call baseline.make_scene() and therefore does
# NOT write into the completed IL workspace.
UNEVEN_SCENE = (
    MODEL_DIR
    / "_kinematic_rl_uneven_baseline.xml"
)


# ============================================================
# TRAINING RANGE
# ============================================================

TRAIN_START_STEP = 150
TRAIN_END_STEP = 750

DEFAULT_EPISODE_LENGTH = 120


# ============================================================
# VERIFIED / TESTED PPO RESIDUAL SCALES
#
# Order per leg:
#
# hip pitch
# hip roll
# hip yaw
# knee
# ankle pitch
# ankle roll
#
# Offline bounded optimization confirmed that these bounds
# contain useful solutions for the distinct hardest terrain
# events while remaining inside the G1 mechanical limits.
# ============================================================

RESIDUAL_SCALES = np.array(
    [
        0.45,
        0.15,
        0.15,
        0.30,
        0.35,
        0.08,

        0.45,
        0.15,
        0.15,
        0.30,
        0.35,
        0.08,
    ],
    dtype=np.float32,
)


# ============================================================
# TERRAIN PREVIEW
#
# Robot travels mainly toward negative world X.
# ============================================================

TERRAIN_PREVIEW_OFFSETS = np.array(
    [
        0.00,
        -0.10,
        -0.20,
        -0.35,
        -0.50,
        -0.70,
    ],
    dtype=np.float32,
)


# ============================================================
# REWARD HYPERPARAMETERS
#
# These are training hyperparameters, not measured robot
# constants.
# ============================================================

FLAT_DISTANCE_TOLERANCE = 0.005

DEFICIT_SCALE = 0.050

PENETRATION_SCALE = 0.050

EXCESS_ALLOWANCE = 0.080
EXCESS_SCALE = 0.080

DEFICIT_WEIGHT = 1.80
IMPROVEMENT_WEIGHT = 0.00
PENETRATION_WEIGHT = 8.00
EXCESS_WEIGHT = 0.05
RESIDUAL_WEIGHT = 0.03
SMOOTHNESS_WEIGHT = 0.02
ACTION_WEIGHT = 0.005
JOINT_CLIP_WEIGHT = 0.50


# ============================================================
# HELPERS
# ============================================================

def load_module(
    module_name,
    file_path,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load module:\n{file_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def reset_mujoco_data(
    model,
):
    data = mujoco.MjData(
        model
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

    return data


def get_joint_id(
    model,
    name,
):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        joint_id
    )


def get_foot_spheres(
    model,
):
    spheres = []

    for geom_id in range(
        model.ngeom
    ):
        if (
            int(
                model.geom_type[
                    geom_id
                ]
            )
            !=
            int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            )
        ):
            continue

        body_id = int(
            model.geom_bodyid[
                geom_id
            ]
        )

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if (
            body_name
            not in
            [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ]
        ):
            continue

        local_pos = (
            model.geom_pos[
                geom_id
            ]
            .copy()
        )

        side = (
            "left"
            if
            str(
                body_name
            ).startswith(
                "left_"
            )
            else
            "right"
        )

        region = (
            "heel"
            if
            float(
                local_pos[
                    0
                ]
            )
            <
            0.0
            else
            "toe"
        )

        spheres.append(
            {
                "geom_id":
                    int(
                        geom_id
                    ),

                "side":
                    side,

                "region":
                    region,

                "local_pos":
                    local_pos,
            }
        )

    spheres.sort(
        key=
            lambda item:
                (
                    item[
                        "side"
                    ],

                    item[
                        "region"
                    ],

                    float(
                        item[
                            "local_pos"
                        ][
                            1
                        ]
                    ),
                )
    )

    if len(spheres) != 8:
        raise RuntimeError(
            "Expected exactly 8 foot "
            "collision spheres, found "
            f"{len(spheres)}"
        )

    return spheres


def get_main_terrain_geom_ids(
    model,
):
    result = []

    for geom_id in range(
        model.ngeom
    ):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        if (
            name is not None
            and
            str(
                name
            ).startswith(
                "rl_terrain_block_"
            )
        ):
            result.append(
                int(
                    geom_id
                )
            )

    if len(result) != 9:
        raise RuntimeError(
            "Expected exactly 9 main "
            "terrain blocks, found "
            f"{len(result)}"
        )

    return result


def signed_geom_distance(
    model,
    data,
    geom_1,
    geom_2,
):
    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(
                geom_1
            ),
            int(
                geom_2
            ),
            1.0,
            None,
        )
    )


# ============================================================
# ENVIRONMENT
# ============================================================

class G1KinematicUnevenEnv(
    gym.Env
):

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        episode_length=
            DEFAULT_EPISODE_LENGTH,
        random_start=True,
        residual_smoothing=0.35,
    ):
        super().__init__()

        # ----------------------------------------------------
        # REQUIRED FILES
        # ----------------------------------------------------

        required_paths = [
            BASELINE_SCRIPT,
            DATASET,
            CHECKPOINT,
            FLAT_SCENE,
            UNEVEN_SCENE,
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    "Required file missing:\n"
                    f"{path}"
                )

        # ----------------------------------------------------
        # EXACT CURRENT BASELINE CODE
        # ----------------------------------------------------

        self.baseline = load_module(
            "g1_kinematic_ppo_baseline",
            BASELINE_SCRIPT,
        )

        # IMPORTANT:
        #
        # Do NOT call:
        #
        #     self.baseline.make_scene()
        #
        # That would rewrite an XML inside the completed
        # IL workspace.
        #
        # Use the already-audited scene read-only.
        self.uneven_scene = (
            UNEVEN_SCENE
        )

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        self.dataset = np.load(
            DATASET,
            allow_pickle=True,
        )

        self.observations = np.asarray(
            self.dataset[
                "il_observations"
            ],
            dtype=np.float32,
        )

        self.root_positions = np.asarray(
            self.dataset[
                "root_positions"
            ],
            dtype=np.float32,
        )

        self.expert_actions = np.asarray(
            self.dataset[
                "il_actions"
            ],
            dtype=np.float32,
        )

        if (
            self.observations.shape
            !=
            (300, 3)
        ):
            raise ValueError(
                "Expected observations "
                "(300,3), got "
                f"{self.observations.shape}"
            )

        if (
            self.expert_actions.shape
            !=
            (300, 15)
        ):
            raise ValueError(
                "Expected actions "
                "(300,15), got "
                f"{self.expert_actions.shape}"
            )

        # ----------------------------------------------------
        # FROZEN TRAINED BC
        # ----------------------------------------------------

        self.device = torch.device(
            "cpu"
        )

        (
            self.bc_model,
            self.checkpoint,
            self.normalization,
        ) = self.baseline.load_bc_policy(
            self.device
        )

        for parameter in (
            self.bc_model.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self.bc_model.eval()

        self.predicted_leg_frames = (
            self.baseline
            .predict_all_leg_frames(
                self.bc_model,
                self.observations,
                self.normalization,
                self.device,
            )
        )

        self.fixed_waist_values = np.asarray(
            self.checkpoint[
                "fixed_waist_values"
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # MUJOCO MODELS
        # ----------------------------------------------------

        self.model = (
            mujoco.MjModel
            .from_xml_path(
                str(
                    self.uneven_scene
                )
            )
        )

        # Verify that the existing read-only XML still
        # matches the exact current baseline terrain.
        self._validate_compiled_terrain()

        self.data = reset_mujoco_data(
            self.model
        )

        self.flat_model = (
            mujoco.MjModel
            .from_xml_path(
                str(
                    FLAT_SCENE
                )
            )
        )

        self.flat_data = reset_mujoco_data(
            self.flat_model
        )

        # ----------------------------------------------------
        # JOINT ADDRESSES
        # ----------------------------------------------------

        self.joint_addresses = (
            self.baseline
            .get_joint_addresses(
                self.model
            )
        )

        self.flat_joint_addresses = (
            self.baseline
            .get_joint_addresses(
                self.flat_model
            )
        )

        self.stand_joint_positions = (
            np.asarray(
                [
                    self.data.qpos[
                        address
                    ]

                    for address
                    in self.joint_addresses
                ],
                dtype=np.float32,
            )
        )

        self.flat_stand_joint_positions = (
            np.asarray(
                [
                    self.flat_data.qpos[
                        address
                    ]

                    for address
                    in self.flat_joint_addresses
                ],
                dtype=np.float32,
            )
        )

        # Earlier audit proved these are exactly equal.
        stand_difference = float(
            np.max(
                np.abs(
                    self.stand_joint_positions
                    -
                    self.flat_stand_joint_positions
                )
            )
        )

        if stand_difference > 1e-12:
            raise RuntimeError(
                "Flat and uneven stand joint "
                "poses do not match; max diff="
                f"{stand_difference}"
            )

        self.stand_root_z = float(
            self.data.qpos[
                2
            ]
        )

        self.flat_stand_root_z = float(
            self.flat_data.qpos[
                2
            ]
        )

        if self.stand_root_z < 0.70:
            self.stand_root_z = 0.79

        if self.flat_stand_root_z < 0.70:
            self.flat_stand_root_z = 0.79

        if (
            abs(
                self.stand_root_z
                -
                self.flat_stand_root_z
            )
            >
            1e-12
        ):
            raise RuntimeError(
                "Flat and uneven stand root "
                "heights do not match."
            )

        # ----------------------------------------------------
        # EXACT 12-DOF JOINT LIMITS
        # ----------------------------------------------------

        self.joint_lower = np.zeros(
            12,
            dtype=np.float32,
        )

        self.joint_upper = np.zeros(
            12,
            dtype=np.float32,
        )

        for (
            index,
            name,
        ) in enumerate(
            self.baseline
            .JOINT_NAMES[
                :12
            ]
        ):
            joint_id = get_joint_id(
                self.model,
                name,
            )

            if not bool(
                self.model.jnt_limited[
                    joint_id
                ]
            ):
                raise RuntimeError(
                    "Expected mechanically "
                    f"limited joint: {name}"
                )

            self.joint_lower[
                index
            ] = float(
                self.model.jnt_range[
                    joint_id,
                    0,
                ]
            )

            self.joint_upper[
                index
            ] = float(
                self.model.jnt_range[
                    joint_id,
                    1,
                ]
            )

        # ----------------------------------------------------
        # RESIDUAL SCALES
        # ----------------------------------------------------

        self.residual_scales = (
            RESIDUAL_SCALES.copy()
        )

        if (
            self.residual_scales.shape
            !=
            (12,)
        ):
            raise RuntimeError(
                "Residual scales must "
                "have shape (12,)."
            )

        # Verify once more that the selected full residual
        # bounds remain legal for every BC frame.
        bc_min = np.min(
            self.predicted_leg_frames,
            axis=0,
        )

        bc_max = np.max(
            self.predicted_leg_frames,
            axis=0,
        )

        lower_margin = (
            bc_min
            -
            self.joint_lower
        )

        upper_margin = (
            self.joint_upper
            -
            bc_max
        )

        symmetric_margin = np.minimum(
            lower_margin,
            upper_margin,
        )

        if np.any(
            self.residual_scales
            >
            symmetric_margin
            +
            1e-7
        ):
            bad = np.where(
                self.residual_scales
                >
                symmetric_margin
                +
                1e-7
            )[
                0
            ]

            raise RuntimeError(
                "Residual scale exceeds "
                "verified BC joint margin "
                "for indices: "
                f"{bad.tolist()}"
            )

        # ----------------------------------------------------
        # FOOT / TERRAIN GEOMETRY
        # ----------------------------------------------------

        self.foot_spheres = (
            get_foot_spheres(
                self.model
            )
        )

        self.flat_foot_spheres = (
            get_foot_spheres(
                self.flat_model
            )
        )

        self.terrain_geom_ids = (
            get_main_terrain_geom_ids(
                self.model
            )
        )

        self.floor_geom_id = (
            mujoco.mj_name2id(
                self.flat_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
        )

        if self.floor_geom_id < 0:
            raise RuntimeError(
                "Flat floor geom "
                "was not found."
            )

        # ----------------------------------------------------
        # EPISODE SETTINGS
        # ----------------------------------------------------

        self.episode_length = int(
            episode_length
        )

        if self.episode_length <= 0:
            raise ValueError(
                "episode_length must "
                "be positive."
            )

        self.random_start = bool(
            random_start
        )

        self.residual_smoothing = float(
            residual_smoothing
        )

        if not (
            0.0
            <
            self.residual_smoothing
            <=
            1.0
        ):
            raise ValueError(
                "residual_smoothing "
                "must be in (0,1]."
            )

        # ----------------------------------------------------
        # PRECOMPUTED REFERENCE DISTANCES
        # ----------------------------------------------------

        self.flat_reference_distances = (
            np.zeros(
                (
                    TRAIN_END_STEP
                    +
                    1,
                    8,
                ),
                dtype=np.float32,
            )
        )

        self.bc_uneven_distances = (
            np.zeros(
                (
                    TRAIN_END_STEP
                    +
                    1,
                    8,
                ),
                dtype=np.float32,
            )
        )

        self._precompute_reference_distances()

        # ----------------------------------------------------
        # GYMNASIUM SPACES
        #
        # Observation:
        #
        # 3   BC observation
        # 12  BC joint pose
        # 12  next BC joint delta
        # 12  previous PPO residual
        # 8   current signed terrain distances
        # 8   distance deficit vs flat BC
        # 6   future terrain preview
        # 1   path progress
        #
        # Total = 62
        # ----------------------------------------------------

        self.observation_size = 62

        self.observation_space = (
            spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(
                    self.observation_size,
                ),
                dtype=np.float32,
            )
        )

        # PPO action remains normalized.
        self.action_space = (
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32,
            )
        )

        # ----------------------------------------------------
        # RUNTIME STATE
        # ----------------------------------------------------

        self.current_step = (
            TRAIN_START_STEP
        )

        self.episode_steps = 0

        self.previous_residual = (
            np.zeros(
                12,
                dtype=np.float32,
            )
        )

        self.current_metadata = None
        self.current_distances = None
        self.current_clip_amount = 0.0

        self._apply_uneven_state(
            self.current_step,
            self.previous_residual,
        )


    # ========================================================
    # TERRAIN VALIDATION
    # ========================================================

    def _validate_compiled_terrain(
        self,
    ):
        expected_terrain = list(
            self.baseline.TERRAIN
        )

        found = []

        for geom_id in range(
            self.model.ngeom
        ):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )

            if (
                name is not None
                and
                str(
                    name
                ).startswith(
                    "rl_terrain_block_"
                )
            ):
                found.append(
                    int(
                        geom_id
                    )
                )

        if (
            len(
                found
            )
            !=
            len(
                expected_terrain
            )
        ):
            raise RuntimeError(
                "Existing uneven terrain XML "
                "does not contain the expected "
                "number of main terrain blocks. "
                f"expected={len(expected_terrain)}, "
                f"found={len(found)}"
            )

        for (
            index,
            (
                start,
                end,
                height,
            ),
        ) in enumerate(
            expected_terrain
        ):
            name = (
                f"rl_terrain_block_"
                f"{index}"
            )

            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )

            if geom_id < 0:
                raise RuntimeError(
                    "Existing uneven terrain "
                    f"XML is missing {name}."
                )

            expected_pos = np.array(
                [
                    (
                        start
                        +
                        end
                    )
                    /
                    2.0,

                    0.0,

                    height
                    /
                    2.0,
                ],
                dtype=np.float64,
            )

            expected_size = np.array(
                [
                    (
                        end
                        -
                        start
                    )
                    /
                    2.0,

                    0.62,

                    height
                    /
                    2.0,
                ],
                dtype=np.float64,
            )

            position_error = float(
                np.max(
                    np.abs(
                        self.model.geom_pos[
                            geom_id
                        ]
                        -
                        expected_pos
                    )
                )
            )

            size_error = float(
                np.max(
                    np.abs(
                        self.model.geom_size[
                            geom_id
                        ]
                        -
                        expected_size
                    )
                )
            )

            if (
                position_error
                >
                1e-12
                or
                size_error
                >
                1e-12
            ):
                raise RuntimeError(
                    "Existing uneven terrain XML "
                    "does not match the current "
                    f"baseline definition for {name}. "
                    f"position_error={position_error}, "
                    f"size_error={size_error}"
                )


    # ========================================================
    # EXACT APPROVED SHOWCASE TIMING
    # ========================================================

    def _motion_frame_for_step(
        self,
        step,
    ):
        step = int(
            step
        )

        if (
            step
            <=
            self.baseline
            .INITIAL_STAND_STEPS
        ):
            return 0.0

        frame = (
            float(
                step
                -
                self.baseline
                .INITIAL_STAND_STEPS
            )
            *
            float(
                self.baseline
                .MOTION_FRAME_INCREMENT
            )
        )

        return float(
            np.clip(
                frame,
                0.0,
                len(
                    self.predicted_leg_frames
                )
                -
                1,
            )
        )


    # ========================================================
    # ZERO-RESIDUAL BC STATE
    # ========================================================

    def _base_state(
        self,
        step,
        terrain_mode,
    ):
        motion_frame = (
            self._motion_frame_for_step(
                step
            )
        )

        blend = (
            self.baseline
            .transition_blend(
                step
            )
        )

        predicted_legs = (
            self.baseline
            .interpolate_array(
                self.predicted_leg_frames,
                motion_frame,
            )
        )

        predicted_walk_15 = np.zeros(
            15,
            dtype=np.float32,
        )

        predicted_walk_15[
            :12
        ] = predicted_legs

        predicted_walk_15[
            12:15
        ] = (
            self.fixed_waist_values
        )

        base_joints = (
            (
                1.0
                -
                blend
            )
            *
            self.stand_joint_positions
            +
            blend
            *
            predicted_walk_15
        ).astype(
            np.float32
        )

        root_pos = (
            self.baseline
            .interpolate_array(
                self.root_positions,
                motion_frame,
            )
        )

        root_delta = (
            root_pos
            -
            self.root_positions[
                0
            ]
        )

        world_x = float(
            blend
            *
            self.baseline
            .ROOT_MOTION_SCALE
            *
            root_delta[
                0
            ]
        )

        world_y = float(
            blend
            *
            self.baseline
            .ROOT_MOTION_SCALE
            *
            root_delta[
                1
            ]
        )

        if terrain_mode == "uneven":
            terrain_root_height = (
                self.baseline
                .smooth_root_terrain_height(
                    world_x
                )
            )

        elif terrain_mode == "flat":
            terrain_root_height = (
                0.02
            )

        else:
            raise ValueError(
                "Unknown terrain mode: "
                f"{terrain_mode}"
            )

        root_z = float(
            (
                1.0
                -
                blend
            )
            *
            self.stand_root_z
            +
            blend
            *
            (
                root_pos[
                    2
                ]
                +
                terrain_root_height
            )
        )

        return {
            "step":
                int(
                    step
                ),

            "motion_frame":
                float(
                    motion_frame
                ),

            "blend":
                float(
                    blend
                ),

            "base_joints":
                base_joints,

            "world_x":
                world_x,

            "world_y":
                world_y,

            "root_z":
                root_z,
        }


    # ========================================================
    # APPLY KINEMATIC POSE
    # ========================================================

    def _apply_pose_to_model(
        self,
        model,
        data,
        joint_addresses,
        step,
        residual,
        terrain_mode,
    ):
        metadata = (
            self._base_state(
                step,
                terrain_mode,
            )
        )

        requested_leg_joints = (
            metadata[
                "base_joints"
            ][
                :12
            ]
            +
            residual
        )

        applied_leg_joints = np.clip(
            requested_leg_joints,
            self.joint_lower,
            self.joint_upper,
        )

        clip_amount = float(
            np.mean(
                np.abs(
                    requested_leg_joints
                    -
                    applied_leg_joints
                )
                /
                np.maximum(
                    self.residual_scales,
                    1e-6,
                )
            )
        )

        applied_joints = (
            metadata[
                "base_joints"
            ]
            .copy()
        )

        applied_joints[
            :12
        ] = applied_leg_joints

        data.qpos[
            0
        ] = (
            metadata[
                "world_x"
            ]
        )

        data.qpos[
            1
        ] = (
            metadata[
                "world_y"
            ]
        )

        data.qpos[
            2
        ] = (
            metadata[
                "root_z"
            ]
        )

        data.qpos[
            3:7
        ] = (
            self.baseline
            .yaw_to_quat_wxyz(
                180.0
            )
        )

        for (
            index,
            address,
        ) in enumerate(
            joint_addresses
        ):
            data.qpos[
                address
            ] = float(
                applied_joints[
                    index
                ]
            )

        data.qvel[
            :
        ] = 0.0

        # KINEMATICS / COLLISION UPDATE ONLY.
        #
        # NO mujoco.mj_step().
        mujoco.mj_forward(
            model,
            data,
        )

        metadata[
            "applied_joints"
        ] = applied_joints

        metadata[
            "clip_amount"
        ] = clip_amount

        return metadata


    # ========================================================
    # SIGNED GEOMETRY DISTANCE VECTOR
    # ========================================================

    def _distance_vector(
        self,
        model,
        data,
        foot_spheres,
        target_geom_ids,
    ):
        distances = np.zeros(
            8,
            dtype=np.float32,
        )

        for (
            sphere_index,
            sphere,
        ) in enumerate(
            foot_spheres
        ):
            best_distance = np.inf

            for target_geom_id in (
                target_geom_ids
            ):
                distance = (
                    signed_geom_distance(
                        model,
                        data,
                        sphere[
                            "geom_id"
                        ],
                        target_geom_id,
                    )
                )

                if (
                    distance
                    <
                    best_distance
                ):
                    best_distance = (
                        distance
                    )

            distances[
                sphere_index
            ] = float(
                best_distance
            )

        return distances


    # ========================================================
    # PRECOMPUTE FLAT + BC-ONLY UNEVEN REFERENCES
    # ========================================================

    def _precompute_reference_distances(
        self,
    ):
        zero_residual = np.zeros(
            12,
            dtype=np.float32,
        )

        for step in range(
            TRAIN_END_STEP
            +
            1
        ):
            self._apply_pose_to_model(
                self.flat_model,
                self.flat_data,
                self.flat_joint_addresses,
                step,
                zero_residual,
                "flat",
            )

            self.flat_reference_distances[
                step
            ] = (
                self._distance_vector(
                    self.flat_model,
                    self.flat_data,
                    self.flat_foot_spheres,
                    [
                        int(
                            self.floor_geom_id
                        )
                    ],
                )
            )

            self._apply_pose_to_model(
                self.model,
                self.data,
                self.joint_addresses,
                step,
                zero_residual,
                "uneven",
            )

            self.bc_uneven_distances[
                step
            ] = (
                self._distance_vector(
                    self.model,
                    self.data,
                    self.foot_spheres,
                    self.terrain_geom_ids,
                )
            )


    # ========================================================
    # CURRENT UNEVEN STATE
    # ========================================================

    def _apply_uneven_state(
        self,
        step,
        residual,
    ):
        self.current_metadata = (
            self._apply_pose_to_model(
                self.model,
                self.data,
                self.joint_addresses,
                step,
                residual,
                "uneven",
            )
        )

        self.current_clip_amount = float(
            self.current_metadata[
                "clip_amount"
            ]
        )

        self.current_distances = (
            self._distance_vector(
                self.model,
                self.data,
                self.foot_spheres,
                self.terrain_geom_ids,
            )
        )


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _observation(
        self,
    ):
        step = int(
            self.current_step
        )

        motion_frame = float(
            self.current_metadata[
                "motion_frame"
            ]
        )

        bc_observation = (
            self.baseline
            .interpolate_array(
                self.observations,
                motion_frame,
            )
        )

        current_bc_joints = (
            self.current_metadata[
                "base_joints"
            ][
                :12
            ]
        )

        next_step = min(
            step
            +
            1,
            TRAIN_END_STEP,
        )

        next_bc_state = (
            self._base_state(
                next_step,
                "uneven",
            )
        )

        bc_delta = (
            next_bc_state[
                "base_joints"
            ][
                :12
            ]
            -
            current_bc_joints
        )

        residual_normalized = (
            self.previous_residual
            /
            self.residual_scales
        )

        distance_normalized = (
            self.current_distances
            /
            0.15
        )

        flat_reference = (
            self.flat_reference_distances[
                step
            ]
        )

        # Positive = current terrain pose is
        # below its same-phase flat BC reference.
        distance_deficit = (
            flat_reference
            -
            self.current_distances
        )

        deficit_normalized = (
            distance_deficit
            /
            0.10
        )

        world_x = float(
            self.current_metadata[
                "world_x"
            ]
        )

        current_height = float(
            self.baseline
            .terrain_height(
                world_x
            )
        )

        terrain_preview = np.asarray(
            [
                (
                    self.baseline
                    .terrain_height(
                        world_x
                        +
                        float(
                            offset
                        )
                    )
                    -
                    current_height
                )
                /
                0.18

                for offset
                in
                TERRAIN_PREVIEW_OFFSETS
            ],
            dtype=np.float32,
        )

        path_progress = np.array(
            [
                np.clip(
                    (
                        -world_x
                    )
                    /
                    3.30,
                    -0.25,
                    1.25,
                )
            ],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                # 3
                bc_observation,

                # 12
                current_bc_joints
                /
                1.20,

                # 12
                bc_delta
                /
                0.10,

                # 12
                residual_normalized,

                # 8
                distance_normalized,

                # 8
                deficit_normalized,

                # 6
                terrain_preview,

                # 1
                path_progress,
            ]
        ).astype(
            np.float32
        )

        if (
            observation.shape
            !=
            (
                self.observation_size,
            )
        ):
            raise RuntimeError(
                "Observation shape mismatch: "
                f"{observation.shape}"
            )

        observation = np.clip(
            observation,
            -5.0,
            5.0,
        ).astype(
            np.float32
        )

        return observation


    # ========================================================
    # REWARD
    # ========================================================

    def _reward(
        self,
        action,
        previous_residual_before_action,
    ):
        step = int(
            self.current_step
        )

        current = (
            self.current_distances
            .astype(
                np.float64
            )
        )

        target = (
            self.flat_reference_distances[
                step
            ]
            .astype(
                np.float64
            )
        )

        baseline = (
            self.bc_uneven_distances[
                step
            ]
            .astype(
                np.float64
            )
        )

        # ----------------------------------------------------
        # DEFICIT RELATIVE TO SAME-PHASE FLAT BC
        # ----------------------------------------------------

        deficit = np.maximum(
            target
            -
            current
            -
            FLAT_DISTANCE_TOLERANCE,
            0.0,
        )

        baseline_deficit = np.maximum(
            target
            -
            baseline
            -
            FLAT_DISTANCE_TOLERANCE,
            0.0,
        )

        deficit_cost = float(
            np.mean(
                (
                    deficit
                    /
                    DEFICIT_SCALE
                )
                **
                2
            )
        )

        baseline_deficit_cost = float(
            np.mean(
                (
                    baseline_deficit
                    /
                    DEFICIT_SCALE
                )
                **
                2
            )
        )

        improvement = (
            baseline_deficit_cost
            -
            deficit_cost
        )

        # ----------------------------------------------------
        # PHASE-AWARE EXCESS PENETRATION
        #
        # Preserve the same-phase flat-BC signed geometry.
        #
        # If the flat reference is above the terrain, the
        # allowed floor is 0.0. If the flat reference itself
        # is slightly negative, preserve that negative
        # same-phase geometry instead of forcing every sphere
        # to be positive.
        #
        # A 5 mm tolerance is retained through
        # FLAT_DISTANCE_TOLERANCE.
        # ----------------------------------------------------

        allowed_floor = np.minimum(
            target,
            0.0,
        )

        penetration_excess = np.maximum(
            allowed_floor
            -
            current
            -
            FLAT_DISTANCE_TOLERANCE,
            0.0,
        )

        penetration_cost = float(
            np.mean(
                (
                    penetration_excess
                    /
                    PENETRATION_SCALE
                )
                **
                2
            )
        )

        # ----------------------------------------------------
        # EXCESS CLEARANCE
        # ----------------------------------------------------

        excess = np.maximum(
            current
            -
            target
            -
            EXCESS_ALLOWANCE,
            0.0,
        )

        excess_cost = float(
            np.mean(
                (
                    excess
                    /
                    EXCESS_SCALE
                )
                **
                2
            )
        )

        # ----------------------------------------------------
        # BC / RESIDUAL REGULARIZATION
        # ----------------------------------------------------

        residual_cost = float(
            np.mean(
                (
                    self.previous_residual
                    /
                    self.residual_scales
                )
                **
                2
            )
        )

        residual_change = (
            self.previous_residual
            -
            previous_residual_before_action
        )

        smoothness_cost = float(
            np.mean(
                (
                    residual_change
                    /
                    self.residual_scales
                )
                **
                2
            )
        )

        action_cost = float(
            np.mean(
                np.asarray(
                    action,
                    dtype=np.float64,
                )
                **
                2
            )
        )

        clip_cost = float(
            self.current_clip_amount
        )

        reward = (
            1.0

            -
            DEFICIT_WEIGHT
            *
            deficit_cost

            -
            PENETRATION_WEIGHT
            *
            penetration_cost

            -
            EXCESS_WEIGHT
            *
            excess_cost

            -
            RESIDUAL_WEIGHT
            *
            residual_cost

            -
            SMOOTHNESS_WEIGHT
            *
            smoothness_cost

            -
            ACTION_WEIGHT
            *
            action_cost

            -
            JOINT_CLIP_WEIGHT
            *
            clip_cost
        )

        info = {
            "reward":
                float(
                    reward
                ),

            "deficit_cost":
                deficit_cost,

            "baseline_deficit_cost":
                baseline_deficit_cost,

            "improvement":
                float(
                    improvement
                ),

            "penetration_cost":
                penetration_cost,

            "max_penetration_excess_m":
                float(
                    np.max(
                        penetration_excess
                    )
                ),

            "mean_penetration_excess_m":
                float(
                    np.mean(
                        penetration_excess
                    )
                ),

            "excess_cost":
                excess_cost,

            "residual_cost":
                residual_cost,

            "smoothness_cost":
                smoothness_cost,

            "action_cost":
                action_cost,

            "joint_clip_cost":
                clip_cost,

            "min_signed_distance_m":
                float(
                    np.min(
                        current
                    )
                ),

            "mean_signed_distance_m":
                float(
                    np.mean(
                        current
                    )
                ),

            "max_residual_rad":
                float(
                    np.max(
                        np.abs(
                            self.previous_residual
                        )
                    )
                ),

            "step":
                step,

            "motion_frame":
                float(
                    self.current_metadata[
                        "motion_frame"
                    ]
                ),

            "world_x":
                float(
                    self.current_metadata[
                        "world_x"
                    ]
                ),
        }

        return (
            float(
                reward
            ),
            info,
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.episode_steps = 0

        self.previous_residual = (
            np.zeros(
                12,
                dtype=np.float32,
            )
        )

        if (
            options is not None
            and
            "start_step"
            in options
        ):
            start_step = int(
                options[
                    "start_step"
                ]
            )

            if not (
                TRAIN_START_STEP
                <=
                start_step
                <=
                TRAIN_END_STEP
            ):
                raise ValueError(
                    "start_step must be "
                    f"between {TRAIN_START_STEP} "
                    f"and {TRAIN_END_STEP}."
                )

            self.current_step = (
                start_step
            )

        elif self.random_start:
            latest_start = max(
                TRAIN_START_STEP,
                TRAIN_END_STEP
                -
                self.episode_length,
            )

            self.current_step = int(
                self.np_random.integers(
                    TRAIN_START_STEP,
                    latest_start
                    +
                    1,
                )
            )

        else:
            self.current_step = (
                TRAIN_START_STEP
            )

        self._apply_uneven_state(
            self.current_step,
            self.previous_residual,
        )

        observation = (
            self._observation()
        )

        info = {
            "start_step":
                int(
                    self.current_step
                ),

            "motion_frame":
                float(
                    self.current_metadata[
                        "motion_frame"
                    ]
                ),

            "world_x":
                float(
                    self.current_metadata[
                        "world_x"
                    ]
                ),

            "min_signed_distance_m":
                float(
                    np.min(
                        self.current_distances
                    )
                ),
        }

        return (
            observation,
            info,
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if (
            action.shape
            !=
            (12,)
        ):
            raise ValueError(
                "Expected PPO action "
                "(12,), got "
                f"{action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        previous_residual_before_action = (
            self.previous_residual
            .copy()
        )

        desired_residual = (
            action
            *
            self.residual_scales
        )

        # ----------------------------------------------------
        # SEQUENTIAL KINEMATIC TRANSITION
        #
        # r_t =
        #
        # (1-alpha) r_(t-1)
        # +
        # alpha desired_r_t
        # ----------------------------------------------------

        alpha = (
            self.residual_smoothing
        )

        self.previous_residual = (
            (
                1.0
                -
                alpha
            )
            *
            self.previous_residual
            +
            alpha
            *
            desired_residual
        ).astype(
            np.float32
        )

        # Advance one reference step.
        self.current_step = min(
            self.current_step
            +
            1,
            TRAIN_END_STEP,
        )

        self.episode_steps += 1

        # Apply frozen BC + PPO residual.
        self._apply_uneven_state(
            self.current_step,
            self.previous_residual,
        )

        (
            reward,
            reward_info,
        ) = (
            self._reward(
                action,
                previous_residual_before_action,
            )
        )

        observation = (
            self._observation()
        )

        # Kinematic stage:
        # no physical fall termination.
        terminated = False

        truncated = bool(
            self.episode_steps
            >=
            self.episode_length

            or

            self.current_step
            >=
            TRAIN_END_STEP
        )

        info = {
            **reward_info,

            "episode_steps":
                int(
                    self.episode_steps
                ),

            "residual_smoothing":
                float(
                    self.residual_smoothing
                ),

            "residual":
                self.previous_residual
                .copy(),

            "signed_distances":
                self.current_distances
                .copy(),

            "flat_reference_distances":
                self.flat_reference_distances[
                    self.current_step
                ]
                .copy(),

            "bc_only_distances":
                self.bc_uneven_distances[
                    self.current_step
                ]
                .copy(),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    # ========================================================
    # RENDER / CLOSE
    # ========================================================

    def render(
        self,
    ):
        return None


    def close(
        self,
    ):
        return None