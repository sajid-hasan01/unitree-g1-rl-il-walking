import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.optimize import differential_evolution


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

ENV_SCRIPT = (
    PROJECT_ROOT
    / "envs"
    / "g1_kinematic_uneven_env.py"
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

MODEL_DIR = (
    IL_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
)

FLAT_SCENE = (
    MODEL_DIR
    / "scene.xml"
)

UNEVEN_SCENE = (
    MODEL_DIR
    / "_kinematic_rl_uneven_baseline.xml"
)


# ============================================================
# DIAGNOSTIC SETTINGS
# ============================================================

TRAIN_START_STEP = 150
TRAIN_END_STEP = 750

# Same tolerance currently used by the PPO environment.
FLAT_DISTANCE_TOLERANCE = 0.005

# Same scale used by the PPO terrain-deficit reward.
DEFICIT_SCALE = 0.050

# Same excessive-clearance allowance/scale used
# by the PPO reward design.
EXCESS_ALLOWANCE = 0.080
EXCESS_SCALE = 0.080

# Analyze distinct hard terrain events rather than
# five neighboring frames from the same obstacle edge.
SELECTED_EVENTS = 5
MIN_EVENT_SEPARATION_STEPS = 25

# Differential-evolution settings.
DE_MAXITER = 80
DE_POPSIZE = 10
DE_TOL = 1e-7
DE_SEED = 425

DISTMAX = 1.0


# ============================================================
# HELPERS
# ============================================================

def load_module(
    module_name,
    file_path,
):
    spec = (
        importlib
        .util
        .spec_from_file_location(
            module_name,
            str(file_path),
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load module:\n{file_path}"
        )

    module = (
        importlib
        .util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def reset_data(
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
    joint_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
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

        body_name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
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
            "Expected exactly 8 "
            "foot spheres, found "
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
        name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
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
            "Expected exactly 9 "
            "main terrain geoms, found "
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
            DISTMAX,
            None,
        )
    )


def distance_vector(
    model,
    data,
    foot_spheres,
    target_geom_ids,
):
    result = np.zeros(
        len(
            foot_spheres
        ),
        dtype=np.float64,
    )

    for (
        sphere_index,
        sphere,
    ) in enumerate(
        foot_spheres
    ):
        best = np.inf

        for target_geom_id in (
            target_geom_ids
        ):
            value = (
                signed_geom_distance(
                    model,
                    data,
                    sphere[
                        "geom_id"
                    ],
                    target_geom_id,
                )
            )

            if value < best:
                best = value

        result[
            sphere_index
        ] = best

    return result


def motion_frame_for_step(
    baseline,
    step,
    num_frames,
):
    if (
        step
        <=
        baseline.INITIAL_STAND_STEPS
    ):
        return 0.0

    frame = (
        float(
            step
            -
            baseline.INITIAL_STAND_STEPS
        )
        *
        float(
            baseline.MOTION_FRAME_INCREMENT
        )
    )

    return float(
        np.clip(
            frame,
            0.0,
            float(
                num_frames
                -
                1
            ),
        )
    )


def setup_model(
    baseline,
    model,
):
    data = reset_data(
        model
    )

    joint_addresses = (
        baseline
        .get_joint_addresses(
            model
        )
    )

    stand_joint_positions = np.asarray(
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

    if stand_root_z < 0.70:
        stand_root_z = 0.79

    return (
        data,
        joint_addresses,
        stand_joint_positions,
        stand_root_z,
    )


def apply_zero_residual_pose(
    baseline,
    model,
    data,
    joint_addresses,
    stand_joint_positions,
    stand_root_z,
    root_positions,
    predicted_leg_frames,
    fixed_waist_values,
    step,
    terrain_mode,
):
    motion_frame = (
        motion_frame_for_step(
            baseline,
            step,
            len(
                predicted_leg_frames
            ),
        )
    )

    blend = (
        baseline
        .transition_blend(
            step
        )
    )

    predicted_legs = (
        baseline
        .interpolate_array(
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
    ] = predicted_legs

    predicted_walk_15[
        12:15
    ] = fixed_waist_values

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

    root_pos = (
        baseline
        .interpolate_array(
            root_positions,
            motion_frame,
        )
    )

    root_delta = (
        root_pos
        -
        root_positions[
            0
        ]
    )

    world_x = float(
        blend
        *
        baseline.ROOT_MOTION_SCALE
        *
        root_delta[
            0
        ]
    )

    world_y = float(
        blend
        *
        baseline.ROOT_MOTION_SCALE
        *
        root_delta[
            1
        ]
    )

    if terrain_mode == "uneven":
        root_height_offset = (
            baseline
            .smooth_root_terrain_height(
                world_x
            )
        )

    elif terrain_mode == "flat":
        root_height_offset = 0.02

    else:
        raise ValueError(
            "Unknown terrain_mode: "
            f"{terrain_mode}"
        )

    root_z = float(
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
            root_height_offset
        )
    )

    data.qpos[
        0
    ] = world_x

    data.qpos[
        1
    ] = world_y

    data.qpos[
        2
    ] = root_z

    data.qpos[
        3:7
    ] = (
        baseline
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

    # Kinematic/collision update only.
    # NO mj_step().
    mujoco.mj_forward(
        model,
        data,
    )

    return {
        "motion_frame":
            float(
                motion_frame
            ),

        "world_x":
            world_x,

        "world_y":
            world_y,

        "base_joints":
            applied_joints
            .copy(),
    }


def select_distinct_worst_events(
    records,
):
    selected = []

    for record in records:
        if all(
            abs(
                record[
                    "step"
                ]
                -
                existing[
                    "step"
                ]
            )
            >=
            MIN_EVENT_SEPARATION_STEPS

            for existing
            in selected
        ):
            selected.append(
                record
            )

        if (
            len(
                selected
            )
            >=
            SELECTED_EVENTS
        ):
            break

    return selected


def side_indices(
    foot_spheres,
    side,
):
    return np.asarray(
        [
            index

            for (
                index,
                item,
            )
            in enumerate(
                foot_spheres
            )

            if
            item[
                "side"
            ]
            ==
            side
        ],
        dtype=np.int32,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=" * 100
    )

    print(
        "UNITREE G1 KINEMATIC PPO "
        "- BOUNDED ACTION FEASIBILITY AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "This performs NO PPO training."
    )

    print(
        "mujoco.mj_step() is NEVER called."
    )

    print(
        "SciPy differential_evolution "
        "is used only as an offline diagnostic."
    )

    required_paths = [
        BASELINE_SCRIPT,
        ENV_SCRIPT,
        DATASET,
        CHECKPOINT,
        FLAT_SCENE,
        UNEVEN_SCENE,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Required path missing:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # LOAD CURRENT EXACT PROJECT FILES
    # --------------------------------------------------------

    baseline = load_module(
        "ppo_bounds_audit_baseline",
        BASELINE_SCRIPT,
    )

    # Import the environment MODULE only.
    #
    # We do NOT instantiate the environment here,
    # therefore its constructor and make_scene()
    # are not executed.
    env_module = load_module(
        "ppo_bounds_audit_env_constants",
        ENV_SCRIPT,
    )

    residual_scales = np.asarray(
        env_module.RESIDUAL_SCALES,
        dtype=np.float64,
    )

    if (
        residual_scales.shape
        !=
        (12,)
    ):
        raise RuntimeError(
            "Expected RESIDUAL_SCALES "
            "shape (12,), got "
            f"{residual_scales.shape}"
        )

    print()

    print(
        "Current PPO residual "
        "scales [rad]:"
    )

    for (
        index,
        name,
    ) in enumerate(
        baseline.JOINT_NAMES[
            :12
        ]
    ):
        print(
            f"  {index:02d} "
            f"{name:29s} "
            f"+/- "
            f"{residual_scales[index]:.4f}"
        )

    # --------------------------------------------------------
    # DATA + FROZEN BC
    # --------------------------------------------------------

    dataset = np.load(
        DATASET,
        allow_pickle=True,
    )

    observations = np.asarray(
        dataset[
            "il_observations"
        ],
        dtype=np.float32,
    )

    root_positions = np.asarray(
        dataset[
            "root_positions"
        ],
        dtype=np.float32,
    )

    device = torch.device(
        "cpu"
    )

    (
        bc_model,
        checkpoint,
        normalization,
    ) = (
        baseline
        .load_bc_policy(
            device
        )
    )

    predicted_leg_frames = (
        baseline
        .predict_all_leg_frames(
            bc_model,
            observations,
            normalization,
            device,
        )
    )

    fixed_waist_values = np.asarray(
        checkpoint[
            "fixed_waist_values"
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    uneven_model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                UNEVEN_SCENE
            )
        )
    )

    flat_model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                FLAT_SCENE
            )
        )
    )

    (
        uneven_data,
        uneven_joint_addresses,
        uneven_stand_joints,
        uneven_stand_root_z,
    ) = (
        setup_model(
            baseline,
            uneven_model,
        )
    )

    (
        flat_data,
        flat_joint_addresses,
        flat_stand_joints,
        flat_stand_root_z,
    ) = (
        setup_model(
            baseline,
            flat_model,
        )
    )

    uneven_spheres = (
        get_foot_spheres(
            uneven_model
        )
    )

    flat_spheres = (
        get_foot_spheres(
            flat_model
        )
    )

    terrain_geom_ids = (
        get_main_terrain_geom_ids(
            uneven_model
        )
    )

    floor_geom_id = (
        mujoco.mj_name2id(
            flat_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
    )

    if floor_geom_id < 0:
        raise RuntimeError(
            "Flat floor geom not found."
        )

    # --------------------------------------------------------
    # EXACT JOINT LIMITS
    # --------------------------------------------------------

    joint_lower = np.zeros(
        12,
        dtype=np.float64,
    )

    joint_upper = np.zeros(
        12,
        dtype=np.float64,
    )

    for (
        index,
        name,
    ) in enumerate(
        baseline.JOINT_NAMES[
            :12
        ]
    ):
        joint_id = (
            get_joint_id(
                uneven_model,
                name,
            )
        )

        joint_lower[
            index
        ] = float(
            uneven_model
            .jnt_range[
                joint_id,
                0,
            ]
        )

        joint_upper[
            index
        ] = float(
            uneven_model
            .jnt_range[
                joint_id,
                1,
            ]
        )

    # ========================================================
    # COLLECT ALL WALKING FRAMES
    # ========================================================

    records = []

    for step in range(
        TRAIN_START_STEP,
        TRAIN_END_STEP
        +
        1,
    ):
        uneven_meta = (
            apply_zero_residual_pose(
                baseline,
                uneven_model,
                uneven_data,
                uneven_joint_addresses,
                uneven_stand_joints,
                uneven_stand_root_z,
                root_positions,
                predicted_leg_frames,
                fixed_waist_values,
                step,
                "uneven",
            )
        )

        uneven_distances = (
            distance_vector(
                uneven_model,
                uneven_data,
                uneven_spheres,
                terrain_geom_ids,
            )
        )

        flat_meta = (
            apply_zero_residual_pose(
                baseline,
                flat_model,
                flat_data,
                flat_joint_addresses,
                flat_stand_joints,
                flat_stand_root_z,
                root_positions,
                predicted_leg_frames,
                fixed_waist_values,
                step,
                "flat",
            )
        )

        flat_distances = (
            distance_vector(
                flat_model,
                flat_data,
                flat_spheres,
                [
                    int(
                        floor_geom_id
                    )
                ],
            )
        )

        if (
            abs(
                uneven_meta[
                    "motion_frame"
                ]
                -
                flat_meta[
                    "motion_frame"
                ]
            )
            >
            1e-12
        ):
            raise RuntimeError(
                "Flat/uneven motion-frame "
                "mismatch."
            )

        # Same target philosophy as PPO:
        #
        # preserve the same-phase flat BC
        # geometry with a 5 mm tolerance.
        target = (
            flat_distances
            -
            FLAT_DISTANCE_TOLERANCE
        )

        deficit = np.maximum(
            target
            -
            uneven_distances,
            0.0,
        )

        worst_sphere_index = int(
            np.argmax(
                deficit
            )
        )

        worst_side = (
            uneven_spheres[
                worst_sphere_index
            ][
                "side"
            ]
        )

        records.append(
            {
                "step":
                    int(
                        step
                    ),

                "motion_frame":
                    float(
                        uneven_meta[
                            "motion_frame"
                        ]
                    ),

                "world_x":
                    float(
                        uneven_meta[
                            "world_x"
                        ]
                    ),

                "base_joints":
                    uneven_meta[
                        "base_joints"
                    ]
                    .copy(),

                "uneven_distances":
                    uneven_distances
                    .copy(),

                "flat_distances":
                    flat_distances
                    .copy(),

                "target":
                    target
                    .copy(),

                "deficit":
                    deficit
                    .copy(),

                "max_deficit":
                    float(
                        np.max(
                            deficit
                        )
                    ),

                "mean_deficit":
                    float(
                        np.mean(
                            deficit
                        )
                    ),

                "worst_sphere_index":
                    worst_sphere_index,

                "worst_side":
                    worst_side,
            }
        )

    records.sort(
        key=
            lambda item:
                item[
                    "max_deficit"
                ],
        reverse=True,
    )

    selected = (
        select_distinct_worst_events(
            records
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "DISTINCT WORST EVENTS"
    )

    print(
        "=" * 100
    )

    for (
        rank,
        record,
    ) in enumerate(
        selected,
        start=1,
    ):
        sphere = (
            uneven_spheres[
                record[
                    "worst_sphere_index"
                ]
            ]
        )

        print(
            f"#{rank} "
            f"step="
            f"{record['step']:04d} "
            f"frame="
            f"{record['motion_frame']:7.2f} "
            f"x="
            f"{record['world_x']:+.3f} "
            f"side="
            f"{record['worst_side']:5s} "
            f"region="
            f"{sphere['region']:4s} "
            f"max_deficit="
            f"{record['max_deficit'] * 1000:8.3f} mm"
        )

    # ========================================================
    # OPTIMIZE EACH DISTINCT HARD EVENT
    # ========================================================

    final_results = []

    for (
        rank,
        record,
    ) in enumerate(
        selected,
        start=1,
    ):
        step = (
            record[
                "step"
            ]
        )

        affected_side = (
            record[
                "worst_side"
            ]
        )

        if affected_side == "left":
            leg_joint_indices = np.arange(
                0,
                6,
                dtype=np.int32,
            )

        else:
            leg_joint_indices = np.arange(
                6,
                12,
                dtype=np.int32,
            )

        sphere_indices = (
            side_indices(
                uneven_spheres,
                affected_side,
            )
        )

        # ----------------------------------------------------
        # RESTORE EXACT ZERO-RESIDUAL BASE STATE
        # ----------------------------------------------------

        meta = (
            apply_zero_residual_pose(
                baseline,
                uneven_model,
                uneven_data,
                uneven_joint_addresses,
                uneven_stand_joints,
                uneven_stand_root_z,
                root_positions,
                predicted_leg_frames,
                fixed_waist_values,
                step,
                "uneven",
            )
        )

        base_qpos = (
            uneven_data
            .qpos
            .copy()
        )

        base_leg_joints = (
            meta[
                "base_joints"
            ][
                :12
            ]
            .astype(
                np.float64
            )
        )

        flat_target = (
            record[
                "target"
            ]
            .astype(
                np.float64
            )
        )

        # ----------------------------------------------------
        # CURRENT PPO BOUNDS + EXACT MECHANICAL BOUNDS
        # ----------------------------------------------------

        optimizer_bounds = []

        for joint_index in (
            leg_joint_indices
        ):
            mechanical_lower = (
                joint_lower[
                    joint_index
                ]
                -
                base_leg_joints[
                    joint_index
                ]
            )

            mechanical_upper = (
                joint_upper[
                    joint_index
                ]
                -
                base_leg_joints[
                    joint_index
                ]
            )

            lower = max(
                -float(
                    residual_scales[
                        joint_index
                    ]
                ),
                float(
                    mechanical_lower
                ),
            )

            upper = min(
                float(
                    residual_scales[
                        joint_index
                    ]
                ),
                float(
                    mechanical_upper
                ),
            )

            if lower > upper:
                raise RuntimeError(
                    "Invalid optimizer bound "
                    f"for joint "
                    f"{joint_index}: "
                    f"[{lower}, {upper}]"
                )

            optimizer_bounds.append(
                (
                    lower,
                    upper,
                )
            )

        # ----------------------------------------------------
        # EXACT MUJOCO OBJECTIVE
        # ----------------------------------------------------

        def evaluate_leg_residual(
            leg_residual,
        ):
            uneven_data.qpos[
                :
            ] = base_qpos

            full_residual = np.zeros(
                12,
                dtype=np.float64,
            )

            full_residual[
                leg_joint_indices
            ] = np.asarray(
                leg_residual,
                dtype=np.float64,
            )

            for joint_index in (
                leg_joint_indices
            ):
                address = (
                    uneven_joint_addresses[
                        int(
                            joint_index
                        )
                    ]
                )

                uneven_data.qpos[
                    address
                ] = float(
                    base_qpos[
                        address
                    ]
                    +
                    full_residual[
                        joint_index
                    ]
                )

            mujoco.mj_forward(
                uneven_model,
                uneven_data,
            )

            distances = (
                distance_vector(
                    uneven_model,
                    uneven_data,
                    uneven_spheres,
                    terrain_geom_ids,
                )
            )

            side_deficit = np.maximum(
                flat_target[
                    sphere_indices
                ]
                -
                distances[
                    sphere_indices
                ],
                0.0,
            )

            normalized_deficit = (
                side_deficit
                /
                DEFICIT_SCALE
            )

            normalized_residual = (
                np.asarray(
                    leg_residual,
                    dtype=np.float64,
                )
                /
                residual_scales[
                    leg_joint_indices
                ]
            )

            # Do not allow the offline optimizer
            # to "solve" the task by lifting
            # the foot arbitrarily far above
            # the original BC gait.
            excess = np.maximum(
                distances[
                    sphere_indices
                ]
                -
                flat_target[
                    sphere_indices
                ]
                -
                EXCESS_ALLOWANCE,
                0.0,
            )

            normalized_excess = (
                excess
                /
                EXCESS_SCALE
            )

            # Primary:
            # eliminate worst terrain deficit.
            #
            # Secondary:
            # improve the other spheres.
            #
            # Small excessive-clearance penalty:
            # discourage absurd foot lifting.
            #
            # Tiny residual regularizer:
            # among similarly good solutions,
            # prefer smaller corrections.
            loss = (
                2.0
                *
                float(
                    np.max(
                        normalized_deficit
                        **
                        2
                    )
                )

                +
                float(
                    np.mean(
                        normalized_deficit
                        **
                        2
                    )
                )

                +
                0.05
                *
                float(
                    np.mean(
                        normalized_excess
                        **
                        2
                    )
                )

                +
                1e-4
                *
                float(
                    np.mean(
                        normalized_residual
                        **
                        2
                    )
                )
            )

            return loss

        print()

        print(
            "=" * 100
        )

        print(
            f"OPTIMIZING EVENT #{rank}: "
            f"step={step}, "
            f"side={affected_side}"
        )

        print(
            "=" * 100
        )

        print(
            "Residual bounds used [rad]:"
        )

        for (
            local_index,
            joint_index,
        ) in enumerate(
            leg_joint_indices
        ):
            joint_name = (
                baseline.JOINT_NAMES[
                    int(
                        joint_index
                    )
                ]
            )

            (
                lower,
                upper,
            ) = (
                optimizer_bounds[
                    local_index
                ]
            )

            print(
                f"  "
                f"{int(joint_index):02d} "
                f"{joint_name:29s} "
                f"[{lower:+.5f}, "
                f"{upper:+.5f}]"
            )

        # ----------------------------------------------------
        # BOUNDED GLOBAL OPTIMIZATION
        # ----------------------------------------------------

        result = (
            differential_evolution(
                evaluate_leg_residual,
                bounds=
                    optimizer_bounds,

                strategy=
                    "best1bin",

                maxiter=
                    DE_MAXITER,

                popsize=
                    DE_POPSIZE,

                tol=
                    DE_TOL,

                mutation=
                    (
                        0.5,
                        1.0,
                    ),

                recombination=
                    0.7,

                seed=
                    DE_SEED
                    +
                    rank,

                polish=
                    True,

                updating=
                    "immediate",

                workers=
                    1,

                disp=
                    False,
            )
        )

        optimized_leg_residual = np.asarray(
            result.x,
            dtype=np.float64,
        )

        full_residual = np.zeros(
            12,
            dtype=np.float64,
        )

        full_residual[
            leg_joint_indices
        ] = (
            optimized_leg_residual
        )

        # ----------------------------------------------------
        # APPLY OPTIMIZED RESULT ONCE MORE
        # ----------------------------------------------------

        uneven_data.qpos[
            :
        ] = base_qpos

        for joint_index in (
            leg_joint_indices
        ):
            address = (
                uneven_joint_addresses[
                    int(
                        joint_index
                    )
                ]
            )

            uneven_data.qpos[
                address
            ] = float(
                base_qpos[
                    address
                ]
                +
                full_residual[
                    joint_index
                ]
            )

        mujoco.mj_forward(
            uneven_model,
            uneven_data,
        )

        optimized_distances = (
            distance_vector(
                uneven_model,
                uneven_data,
                uneven_spheres,
                terrain_geom_ids,
            )
        )

        optimized_deficit = np.maximum(
            flat_target
            -
            optimized_distances,
            0.0,
        )

        base_deficit = (
            record[
                "deficit"
            ]
        )

        before_max_deficit = float(
            np.max(
                base_deficit
            )
        )

        after_max_deficit = float(
            np.max(
                optimized_deficit
            )
        )

        if before_max_deficit > 1e-12:
            deficit_reduction = (
                (
                    before_max_deficit
                    -
                    after_max_deficit
                )
                /
                before_max_deficit
            )

        else:
            deficit_reduction = 0.0

        normalized_action = (
            full_residual
            /
            residual_scales
        )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print()

        print(
            "Optimization result:"
        )

        print(
            "  success:",
            bool(
                result.success
            ),
        )

        print(
            "  message:",
            str(
                result.message
            ),
        )

        print(
            "  objective:",
            f"{float(result.fun):.9f}",
        )

        print(
            "  evaluations:",
            int(
                result.nfev
            ),
        )

        print()

        print(
            "  baseline minimum "
            "signed distance:",
            (
                f"{float(np.min(record['uneven_distances'])) * 1000:+.3f} mm"
            ),
        )

        print(
            "  optimized minimum "
            "signed distance:",
            (
                f"{float(np.min(optimized_distances)) * 1000:+.3f} mm"
            ),
        )

        print(
            "  baseline maximum "
            "flat-target deficit:",
            (
                f"{before_max_deficit * 1000:.3f} mm"
            ),
        )

        print(
            "  optimized maximum "
            "flat-target deficit:",
            (
                f"{after_max_deficit * 1000:.3f} mm"
            ),
        )

        print(
            "  maximum-deficit reduction:",
            (
                f"{100.0 * deficit_reduction:.2f}%"
            ),
        )

        print(
            "  max |residual|:",
            (
                f"{float(np.max(np.abs(full_residual))):.6f} rad"
            ),
        )

        print(
            "  max |normalized action|:",
            (
                f"{float(np.max(np.abs(normalized_action))):.6f}"
            ),
        )

        print()

        print(
            "Optimized residual / "
            "normalized action:"
        )

        for joint_index in (
            leg_joint_indices
        ):
            joint_name = (
                baseline.JOINT_NAMES[
                    int(
                        joint_index
                    )
                ]
            )

            print(
                f"  "
                f"{int(joint_index):02d} "
                f"{joint_name:29s} "
                f"dq="
                f"{full_residual[joint_index]:+9.6f} rad  "
                f"a="
                f"{normalized_action[joint_index]:+8.5f}"
            )

        print()

        print(
            "Affected-foot sphere distances:"
        )

        print(
            "sphere | region | "
            "flat_target_mm | "
            "baseline_mm | "
            "optimized_mm | "
            "deficit_after_mm"
        )

        print(
            "-" * 90
        )

        for sphere_index in (
            sphere_indices
        ):
            sphere = (
                uneven_spheres[
                    int(
                        sphere_index
                    )
                ]
            )

            print(
                f"{int(sphere_index):6d} | "
                f"{sphere['region']:6s} | "
                f"{flat_target[sphere_index] * 1000:+14.3f} | "
                f"{record['uneven_distances'][sphere_index] * 1000:+11.3f} | "
                f"{optimized_distances[sphere_index] * 1000:+12.3f} | "
                f"{optimized_deficit[sphere_index] * 1000:16.3f}"
            )

        final_results.append(
            {
                "rank":
                    rank,

                "step":
                    step,

                "side":
                    affected_side,

                "before_max_deficit":
                    before_max_deficit,

                "after_max_deficit":
                    after_max_deficit,

                "reduction":
                    float(
                        deficit_reduction
                    ),

                "max_normalized_action":
                    float(
                        np.max(
                            np.abs(
                                normalized_action
                            )
                        )
                    ),

                "success":
                    bool(
                        result.success
                    ),
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "BOUNDED ACTION FEASIBILITY SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "event | step | side  | "
        "before_deficit_mm | "
        "after_deficit_mm | "
        "reduction | max|a|"
    )

    print(
        "-" * 92
    )

    for item in (
        final_results
    ):
        print(
            f"{item['rank']:5d} | "
            f"{item['step']:4d} | "
            f"{item['side']:5s} | "
            f"{item['before_max_deficit'] * 1000:17.3f} | "
            f"{item['after_max_deficit'] * 1000:16.3f} | "
            f"{100.0 * item['reduction']:8.2f}% | "
            f"{item['max_normalized_action']:6.3f}"
        )

    if final_results:
        worst_remaining = max(
            item[
                "after_max_deficit"
            ]
            for item
            in final_results
        )

        minimum_reduction = min(
            item[
                "reduction"
            ]
            for item
            in final_results
        )

        print()

        print(
            "Worst remaining "
            "flat-target deficit "
            "across selected events:",
            (
                f"{worst_remaining * 1000:.3f} mm"
            ),
        )

        print(
            "Smallest maximum-deficit "
            "reduction across selected events:",
            (
                f"{100.0 * minimum_reduction:.2f}%"
            ),
        )

    print()

    print(
        "Interpretation rule:"
    )

    print(
        "  This audit does NOT prove "
        "PPO will learn the optimized actions."
    )

    print(
        "  It only tests whether the CURRENT "
        "bounded residual action space contains "
        "useful kinematic corrections at "
        "distinct hard frames."
    )

    print(
        "  If large deficits remain even after "
        "optimization, the action bounds or "
        "task formulation must be reconsidered "
        "before PPO."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()