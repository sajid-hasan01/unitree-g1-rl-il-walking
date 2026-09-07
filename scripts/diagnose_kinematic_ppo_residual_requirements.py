import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

RL_ROOT = Path(__file__).resolve().parents[1]

IL_ROOT = (
    RL_ROOT.parent
    / "unitree-g1-rl-il-walking(OpenHE IL - Kinematic BC)"
)

BASELINE_SCRIPT = (
    RL_ROOT
    / "scripts"
    / "show_g1_bc_12dof_uneven_baseline.py"
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


# ============================================================
# VERIFIED SETTINGS
# ============================================================

# Final approved flat IL showcase uses +0.02 m.
FLAT_HEIGHT_OFFSET = 0.02

# mj_geomDistance search distance.
DISTMAX = 1.0

# Small numerical perturbation used only for
# local kinematic sensitivity measurement.
JACOBIAN_EPS = 0.01

TOP_WORST_FRAMES = 5

# Perform detailed Jacobian / least-squares analysis
# on the three worst frames.
LS_ANALYZE_FRAMES = 3


# ============================================================
# MODULE LOADING
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
            "Could not load module:\n"
            f"{file_path}"
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


# ============================================================
# MUJOCO HELPERS
# ============================================================

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


def get_floor_geom(
    model,
):
    geom_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
    )

    if geom_id < 0:

        raise RuntimeError(
            "Flat-scene floor geom not found."
        )

    return int(
        geom_id
    )


def get_foot_spheres(
    model,
):
    result = []

    for geom_id in range(
        model.ngeom
    ):

        geom_type = int(
            model.geom_type[
                geom_id
            ]
        )

        if (
            geom_type
            !=
            int(
                mujoco
                .mjtGeom
                .mjGEOM_SPHERE
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

        # Verified from G1 XML:
        #
        # local X = -0.05 -> rear / heel
        # local X = +0.12 -> front / toe
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

        result.append(
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

                "radius":
                    float(
                        model.geom_size[
                            geom_id,
                            0,
                        ]
                    ),
            }
        )

    # Keep deterministic ordering.
    result.sort(
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

    if len(result) != 8:

        raise RuntimeError(
            "Expected exactly 8 "
            "foot collision spheres, "
            f"found {len(result)}"
        )

    return result


def get_main_terrain_geoms(
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

    if not result:

        raise RuntimeError(
            "No rl_terrain_block_* "
            "geoms found."
        )

    return result


def signed_geom_distance(
    model,
    data,
    geom_1,
    geom_2,
):
    """
    Exact MuJoCo signed geometry distance.

    Negative -> penetration
    Zero     -> touching
    Positive -> separated
    """

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


def nearest_distances(
    model,
    data,
    foot_spheres,
    target_geom_ids,
):
    distances = np.zeros(
        len(
            foot_spheres
        ),
        dtype=np.float64,
    )

    nearest_ids = np.zeros(
        len(
            foot_spheres
        ),
        dtype=np.int32,
    )

    for (
        index,
        item,
    ) in enumerate(
        foot_spheres
    ):

        best_distance = np.inf
        best_geom_id = -1

        for target_geom_id in (
            target_geom_ids
        ):

            distance = (
                signed_geom_distance(
                    model,
                    data,
                    item[
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

                best_geom_id = int(
                    target_geom_id
                )

        distances[
            index
        ] = best_distance

        nearest_ids[
            index
        ] = best_geom_id

    return (
        distances,
        nearest_ids,
    )


# ============================================================
# EXACT SHOWCASE TIMING
# ============================================================

def motion_frame_for_step(
    baseline,
    step,
):
    """
    Matches the approved showcase timing.

    Step 60 still displays frame 0.
    After step 60, motion_frame advances by 0.42.
    """

    if (
        step
        <=
        baseline.INITIAL_STAND_STEPS
    ):

        return 0.0

    return (
        float(
            step
            -
            baseline.INITIAL_STAND_STEPS
        )
        *
        float(
            baseline
            .MOTION_FRAME_INCREMENT
        )
    )


# ============================================================
# LOAD BC + DATA
# ============================================================

def prepare_common(
    baseline,
):
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
        baseline.load_bc_policy(
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

    return (
        dataset,
        root_positions,
        predicted_leg_frames,
        fixed_waist_values,
        checkpoint,
    )


# ============================================================
# MODEL SETUP
# ============================================================

def setup_model_for_baseline(
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
                qpos_address
            ]
            for qpos_address
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

    return (
        data,
        joint_addresses,
        stand_joint_positions,
        stand_root_z,
    )


# ============================================================
# APPLY EXACT BC POSE
# ============================================================

def pose_model_at_step(
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
        )
    )

    if (
        motion_frame
        >=
        len(
            predicted_leg_frames
        )
    ):

        motion_frame = float(
            len(
                predicted_leg_frames
            )
            -
            1
        )

    blend = (
        baseline.transition_blend(
            step
        )
    )

    predicted_legs = (
        baseline.interpolate_array(
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

    root_pos = (
        baseline.interpolate_array(
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

    if (
        terrain_mode
        ==
        "flat"
    ):

        root_height_offset = (
            FLAT_HEIGHT_OFFSET
        )

    elif (
        terrain_mode
        ==
        "uneven"
    ):

        root_height_offset = (
            baseline
            .smooth_root_terrain_height(
                world_x
            )
        )

    else:

        raise ValueError(
            "Unknown terrain_mode: "
            f"{terrain_mode}"
        )

    data.qpos[
        0
    ] = world_x

    data.qpos[
        1
    ] = world_y

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
            root_height_offset
        )
    )

    # Same approved walking presentation yaw.
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
        qpos_address,
    ) in enumerate(
        joint_addresses
    ):

        data.qpos[
            qpos_address
        ] = float(
            applied_joints[
                index
            ]
        )

    data.qvel[
        :
    ] = 0.0

    # Kinematics / collision update only.
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

        "blend":
            float(
                blend
            ),

        "world_x":
            world_x,

        "world_y":
            world_y,

        "applied_joints":
            applied_joints.copy(),
    }


# ============================================================
# STATISTIC PRINTING
# ============================================================

def percentile_text(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.size
        ==
        0
    ):

        return "NO DATA"

    percentiles = np.percentile(
        values,
        [
            0,
            1,
            5,
            25,
            50,
            75,
            95,
            99,
            100,
        ],
    )

    return (
        f"min={percentiles[0] * 1000:+8.3f}mm  "
        f"p01={percentiles[1] * 1000:+8.3f}mm  "
        f"p05={percentiles[2] * 1000:+8.3f}mm  "
        f"p25={percentiles[3] * 1000:+8.3f}mm  "
        f"p50={percentiles[4] * 1000:+8.3f}mm  "
        f"p75={percentiles[5] * 1000:+8.3f}mm  "
        f"p95={percentiles[6] * 1000:+8.3f}mm  "
        f"p99={percentiles[7] * 1000:+8.3f}mm  "
        f"max={percentiles[8] * 1000:+8.3f}mm"
    )


# ============================================================
# 1. FLAT-GROUND REFERENCE
# ============================================================

def audit_flat_reference(
    baseline,
    root_positions,
    predicted_leg_frames,
    fixed_waist_values,
):
    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                FLAT_SCENE
            )
        )
    )

    (
        data,
        joint_addresses,
        stand_joint_positions,
        stand_root_z,
    ) = (
        setup_model_for_baseline(
            baseline,
            model,
        )
    )

    foot_spheres = (
        get_foot_spheres(
            model
        )
    )

    floor_id = (
        get_floor_geom(
            model
        )
    )

    all_distances = []

    walking_all_distances = []

    lower_foot_distances = []

    higher_foot_distances = []

    per_sphere = [
        []
        for _
        in foot_spheres
    ]

    negative_depths = []

    start_walking = (
        baseline
        .INITIAL_STAND_STEPS
        +
        baseline
        .TRANSITION_STEPS
    )

    last_step = (
        baseline
        .DEMO_STOP_STEP
    )

    for step in range(
        last_step
        +
        1
    ):

        pose_model_at_step(
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
            terrain_mode=
                "flat",
        )

        (
            distances,
            _,
        ) = (
            nearest_distances(
                model,
                data,
                foot_spheres,
                [
                    floor_id
                ],
            )
        )

        all_distances.extend(
            distances.tolist()
        )

        for (
            index,
            distance,
        ) in enumerate(
            distances
        ):

            per_sphere[
                index
            ].append(
                float(
                    distance
                )
            )

            if (
                distance
                <
                0.0
            ):

                negative_depths.append(
                    -float(
                        distance
                    )
                )

        if (
            step
            >=
            start_walking
        ):

            walking_all_distances.extend(
                distances.tolist()
            )

            left_values = [
                distances[
                    index
                ]
                for (
                    index,
                    sphere,
                ) in enumerate(
                    foot_spheres
                )
                if
                sphere[
                    "side"
                ]
                ==
                "left"
            ]

            right_values = [
                distances[
                    index
                ]
                for (
                    index,
                    sphere,
                ) in enumerate(
                    foot_spheres
                )
                if
                sphere[
                    "side"
                ]
                ==
                "right"
            ]

            left_min = float(
                np.min(
                    left_values
                )
            )

            right_min = float(
                np.min(
                    right_values
                )
            )

            # Lower foot behaves like the
            # support/stance foot.
            lower_foot_distances.append(
                min(
                    left_min,
                    right_min,
                )
            )

            # Higher foot behaves like the
            # swing-foot proxy.
            higher_foot_distances.append(
                max(
                    left_min,
                    right_min,
                )
            )

    print()

    print(
        "=" * 100
    )

    print(
        "1. FLAT-GROUND BC SIGNED-DISTANCE REFERENCE"
    )

    print(
        "=" * 100
    )

    print(
        "Signed distance convention:"
    )

    print(
        "  negative = penetration"
    )

    print(
        "  zero     = touching"
    )

    print(
        "  positive = separated"
    )

    print()

    print(
        "Frames evaluated:",
        last_step
        +
        1,
    )

    print(
        "Walking-only starts at step:",
        start_walking,
    )

    print()

    print(
        "All 8 sole spheres, all frames:"
    )

    print(
        percentile_text(
            all_distances
        )
    )

    print()

    print(
        "All 8 sole spheres, walking-only:"
    )

    print(
        percentile_text(
            walking_all_distances
        )
    )

    print()

    print(
        "Lower-foot minimum signed distance "
        "per walking frame "
        "(stance-like proxy):"
    )

    print(
        percentile_text(
            lower_foot_distances
        )
    )

    print()

    print(
        "Higher-foot minimum signed distance "
        "per walking frame "
        "(swing-like proxy):"
    )

    print(
        percentile_text(
            higher_foot_distances
        )
    )

    negative_depths = np.asarray(
        negative_depths,
        dtype=np.float64,
    )

    print()

    if (
        negative_depths.size
        >
        0
    ):

        print(
            "Negative flat-ground "
            "distances only:"
        )

        print(
            "  count:",
            len(
                negative_depths
            )
        )

        print(
            "  mean:",
            f"{np.mean(negative_depths) * 1000:.3f} mm",
        )

        print(
            "  median:",
            f"{np.median(negative_depths) * 1000:.3f} mm",
        )

        print(
            "  p95:",
            f"{np.percentile(negative_depths, 95) * 1000:.3f} mm",
        )

        print(
            "  max:",
            f"{np.max(negative_depths) * 1000:.3f} mm",
        )

    else:

        print(
            "No negative sole-to-floor "
            "distances were found "
            "on the flat reference."
        )

    print()

    print(
        "Per-sphere flat-ground "
        "signed-distance ranges:"
    )

    for (
        index,
        item,
    ) in enumerate(
        foot_spheres
    ):

        label = (
            f"{item['side']}-"
            f"{item['region']}-"
            f"{index}"
        )

        print(
            f"  {label:18s}: "
            f"{percentile_text(per_sphere[index])}"
        )

    return {
        "stance_like_median":
            float(
                np.median(
                    lower_foot_distances
                )
            ),

        "swing_like_median":
            float(
                np.median(
                    higher_foot_distances
                )
            ),
    }


# ============================================================
# UNEVEN WORST-FRAME COLLECTION
# ============================================================

def collect_uneven_worst_frames(
    baseline,
    root_positions,
    predicted_leg_frames,
    fixed_waist_values,
):
    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                UNEVEN_SCENE
            )
        )
    )

    (
        data,
        joint_addresses,
        stand_joint_positions,
        stand_root_z,
    ) = (
        setup_model_for_baseline(
            baseline,
            model,
        )
    )

    foot_spheres = (
        get_foot_spheres(
            model
        )
    )

    terrain_ids = (
        get_main_terrain_geoms(
            model
        )
    )

    start_walking = (
        baseline
        .INITIAL_STAND_STEPS
        +
        baseline
        .TRANSITION_STEPS
    )

    records = []

    for step in range(
        baseline.DEMO_STOP_STEP
        +
        1
    ):

        metadata = (
            pose_model_at_step(
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
                terrain_mode=
                    "uneven",
            )
        )

        (
            distances,
            nearest_ids,
        ) = (
            nearest_distances(
                model,
                data,
                foot_spheres,
                terrain_ids,
            )
        )

        if (
            step
            >=
            start_walking
        ):

            records.append(
                {
                    "step":
                        int(
                            step
                        ),

                    "motion_frame":
                        metadata[
                            "motion_frame"
                        ],

                    "world_x":
                        metadata[
                            "world_x"
                        ],

                    "min_distance":
                        float(
                            np.min(
                                distances
                            )
                        ),

                    "distances":
                        distances.copy(),

                    "nearest_ids":
                        nearest_ids.copy(),
                }
            )

    records.sort(
        key=
            lambda record:
                record[
                    "min_distance"
                ]
    )

    return (
        model,
        foot_spheres,
        terrain_ids,
        records,
    )


# ============================================================
# GET DISTANCE VECTOR FOR A SPECIFIC FRAME
# ============================================================

def distance_vector_at_step(
    baseline,
    model,
    foot_spheres,
    target_geom_ids,
    root_positions,
    predicted_leg_frames,
    fixed_waist_values,
    step,
    terrain_mode,
):
    (
        data,
        joint_addresses,
        stand_joint_positions,
        stand_root_z,
    ) = (
        setup_model_for_baseline(
            baseline,
            model,
        )
    )

    metadata = (
        pose_model_at_step(
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
            terrain_mode=
                terrain_mode,
        )
    )

    (
        distances,
        nearest_ids,
    ) = (
        nearest_distances(
            model,
            data,
            foot_spheres,
            target_geom_ids,
        )
    )

    return (
        data,
        joint_addresses,
        metadata,
        distances,
        nearest_ids,
    )


# ============================================================
# LOCAL SIGNED-DISTANCE JACOBIAN
# ============================================================

def build_distance_jacobian(
    baseline,
    model,
    data,
    joint_addresses,
    foot_spheres,
    terrain_ids,
):
    base_qpos = (
        data.qpos.copy()
    )

    (
        base_distances,
        _,
    ) = (
        nearest_distances(
            model,
            data,
            foot_spheres,
            terrain_ids,
        )
    )

    jacobian = np.zeros(
        (
            len(
                foot_spheres
            ),
            12,
        ),
        dtype=np.float64,
    )

    for (
        joint_index,
        joint_name,
    ) in enumerate(
        baseline
        .JOINT_NAMES[
            :12
        ]
    ):

        joint_id = (
            get_joint_id(
                model,
                joint_name,
            )
        )

        joint_min = float(
            model.jnt_range[
                joint_id,
                0,
            ]
        )

        joint_max = float(
            model.jnt_range[
                joint_id,
                1,
            ]
        )

        qpos_address = (
            joint_addresses[
                joint_index
            ]
        )

        q0 = float(
            base_qpos[
                qpos_address
            ]
        )

        q_plus = min(
            q0
            +
            JACOBIAN_EPS,
            joint_max,
        )

        q_minus = max(
            q0
            -
            JACOBIAN_EPS,
            joint_min,
        )

        plus_delta = (
            q_plus
            -
            q0
        )

        minus_delta = (
            q0
            -
            q_minus
        )

        # + epsilon
        data.qpos[
            :
        ] = base_qpos

        data.qpos[
            qpos_address
        ] = q_plus

        mujoco.mj_forward(
            model,
            data,
        )

        (
            distances_plus,
            _,
        ) = (
            nearest_distances(
                model,
                data,
                foot_spheres,
                terrain_ids,
            )
        )

        # - epsilon
        data.qpos[
            :
        ] = base_qpos

        data.qpos[
            qpos_address
        ] = q_minus

        mujoco.mj_forward(
            model,
            data,
        )

        (
            distances_minus,
            _,
        ) = (
            nearest_distances(
                model,
                data,
                foot_spheres,
                terrain_ids,
            )
        )

        denominator = (
            plus_delta
            +
            minus_delta
        )

        if (
            denominator
            <=
            1e-12
        ):

            jacobian[
                :,
                joint_index
            ] = 0.0

        else:

            jacobian[
                :,
                joint_index
            ] = (
                (
                    distances_plus
                    -
                    distances_minus
                )
                /
                denominator
            )

    # Restore exact base state.
    data.qpos[
        :
    ] = base_qpos

    mujoco.mj_forward(
        model,
        data,
    )

    return (
        base_distances,
        jacobian,
    )


# ============================================================
# WORST FRAMES + REQUIRED RESIDUAL
# ============================================================

def analyze_worst_frames(
    baseline,
    root_positions,
    predicted_leg_frames,
    fixed_waist_values,
    uneven_model,
    uneven_spheres,
    terrain_ids,
    records,
):
    flat_model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                FLAT_SCENE
            )
        )
    )

    flat_spheres = (
        get_foot_spheres(
            flat_model
        )
    )

    floor_id = (
        get_floor_geom(
            flat_model
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "2. WORST UNEVEN FRAMES + LOCAL JOINT SENSITIVITY"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Worst frames by exact "
        "nearest signed distance:"
    )

    for record in (
        records[
            :TOP_WORST_FRAMES
        ]
    ):

        print(
            f"  step="
            f"{record['step']:04d} "
            f"frame="
            f"{record['motion_frame']:7.2f} "
            f"x="
            f"{record['world_x']:+.3f} "
            f"min="
            f"{record['min_distance'] * 1000:+8.3f} mm"
        )

    for (
        rank,
        record,
    ) in enumerate(
        records[
            :LS_ANALYZE_FRAMES
        ],
        start=1,
    ):

        step = (
            record[
                "step"
            ]
        )

        print()

        print(
            "-" * 100
        )

        print(
            f"Detailed frame #{rank}: "
            f"step={step}, "
            f"motion_frame="
            f"{record['motion_frame']:.2f}, "
            f"world_x="
            f"{record['world_x']:+.3f}"
        )

        print(
            "-" * 100
        )

        (
            uneven_data,
            joint_addresses,
            _,
            uneven_distances,
            uneven_nearest,
        ) = (
            distance_vector_at_step(
                baseline,
                uneven_model,
                uneven_spheres,
                terrain_ids,
                root_positions,
                predicted_leg_frames,
                fixed_waist_values,
                step,
                terrain_mode=
                    "uneven",
            )
        )

        (
            _,
            _,
            _,
            flat_distances,
            _,
        ) = (
            distance_vector_at_step(
                baseline,
                flat_model,
                flat_spheres,
                [
                    floor_id
                ],
                root_positions,
                predicted_leg_frames,
                fixed_waist_values,
                step,
                terrain_mode=
                    "flat",
            )
        )

        print()

        print(
            "Sphere distances:"
        )

        print(
            "uneven terrain vs "
            "same-phase flat BC reference"
        )

        print()

        print(
            "idx | side  region | "
            "uneven_mm | flat_ref_mm | "
            "needed_lift_mm | nearest_terrain"
        )

        print(
            "-" * 96
        )

        for (
            sphere_index,
            item,
        ) in enumerate(
            uneven_spheres
        ):

            required_increase = max(
                float(
                    flat_distances[
                        sphere_index
                    ]
                    -
                    uneven_distances[
                        sphere_index
                    ]
                ),
                0.0,
            )

            terrain_name = (
                mujoco.mj_id2name(
                    uneven_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(
                        uneven_nearest[
                            sphere_index
                        ]
                    ),
                )
            )

            print(
                f"{sphere_index:3d} | "
                f"{item['side']:5s} "
                f"{item['region']:4s} | "
                f"{uneven_distances[sphere_index] * 1000:+10.3f} | "
                f"{flat_distances[sphere_index] * 1000:+11.3f} | "
                f"{required_increase * 1000:14.3f} | "
                f"{terrain_name}"
            )

        (
            base_distances,
            jacobian,
        ) = (
            build_distance_jacobian(
                baseline,
                uneven_model,
                uneven_data,
                joint_addresses,
                uneven_spheres,
                terrain_ids,
            )
        )

        left_indices = [
            index
            for (
                index,
                sphere,
            ) in enumerate(
                uneven_spheres
            )
            if
            sphere[
                "side"
            ]
            ==
            "left"
        ]

        right_indices = [
            index
            for (
                index,
                sphere,
            ) in enumerate(
                uneven_spheres
            )
            if
            sphere[
                "side"
            ]
            ==
            "right"
        ]

        left_worst_index = (
            left_indices[
                int(
                    np.argmin(
                        base_distances[
                            left_indices
                        ]
                    )
                )
            ]
        )

        right_worst_index = (
            right_indices[
                int(
                    np.argmin(
                        base_distances[
                            right_indices
                        ]
                    )
                )
            ]
        )

        print()

        print(
            "Local signed-distance sensitivity "
            "[mm per radian]"
        )

        print(
            "Positive means increasing the joint "
            "angle locally increases separation."
        )

        print()

        print(
            "idx | joint                         | "
            "left_worst | right_worst"
        )

        print(
            "-" * 74
        )

        for (
            joint_index,
            joint_name,
        ) in enumerate(
            baseline
            .JOINT_NAMES[
                :12
            ]
        ):

            print(
                f"{joint_index:3d} | "
                f"{joint_name:29s} | "
                f"{jacobian[left_worst_index, joint_index] * 1000:+10.3f} | "
                f"{jacobian[right_worst_index, joint_index] * 1000:+11.3f}"
            )

        # ----------------------------------------------------
        # TARGET:
        #
        # If an uneven-terrain sphere is lower than
        # the same sphere was during flat-ground BC
        # at the same gait phase, estimate how much
        # joint correction would be required to restore
        # at least the flat-reference separation.
        # ----------------------------------------------------

        required_increase = np.maximum(
            flat_distances
            -
            base_distances,
            0.0,
        )

        active_indices = np.where(
            required_increase
            >
            1e-5
        )[
            0
        ]

        if (
            active_indices.size
            ==
            0
        ):

            print()

            print(
                "No local lift is required "
                "relative to the flat reference "
                "at this frame."
            )

            continue

        active_jacobian = (
            jacobian[
                active_indices,
                :
            ]
        )

        active_target = (
            required_increase[
                active_indices
            ]
        )

        # Minimum-norm local linear solution.
        (
            estimated_delta_q,
            _,
            _,
            _,
        ) = np.linalg.lstsq(
            active_jacobian,
            active_target,
            rcond=None,
        )

        base_qpos = (
            uneven_data
            .qpos
            .copy()
        )

        lower_bounds = np.zeros(
            12,
            dtype=np.float64,
        )

        upper_bounds = np.zeros(
            12,
            dtype=np.float64,
        )

        for (
            joint_index,
            joint_name,
        ) in enumerate(
            baseline
            .JOINT_NAMES[
                :12
            ]
        ):

            joint_id = (
                get_joint_id(
                    uneven_model,
                    joint_name,
                )
            )

            joint_min = float(
                uneven_model
                .jnt_range[
                    joint_id,
                    0,
                ]
            )

            joint_max = float(
                uneven_model
                .jnt_range[
                    joint_id,
                    1,
                ]
            )

            qpos_address = (
                joint_addresses[
                    joint_index
                ]
            )

            q0 = float(
                base_qpos[
                    qpos_address
                ]
            )

            lower_bounds[
                joint_index
            ] = (
                joint_min
                -
                q0
            )

            upper_bounds[
                joint_index
            ] = (
                joint_max
                -
                q0
            )

        clipped_delta_q = np.clip(
            estimated_delta_q,
            lower_bounds,
            upper_bounds,
        )

        # Apply local estimate once to see how the
        # nonlinear MuJoCo geometry actually responds.
        uneven_data.qpos[
            :
        ] = base_qpos

        for (
            joint_index,
            qpos_address,
        ) in enumerate(
            joint_addresses[
                :12
            ]
        ):

            uneven_data.qpos[
                qpos_address
            ] = float(
                base_qpos[
                    qpos_address
                ]
                +
                clipped_delta_q[
                    joint_index
                ]
            )

        mujoco.mj_forward(
            uneven_model,
            uneven_data,
        )

        (
            corrected_distances,
            _,
        ) = (
            nearest_distances(
                uneven_model,
                uneven_data,
                uneven_spheres,
                terrain_ids,
            )
        )

        print()

        print(
            "Minimum-norm local "
            "least-squares correction"
        )

        print(
            "Target:"
        )

        print(
            "Raise spheres that are below "
            "their same-phase flat BC "
            "signed-distance reference."
        )

        print()

        print(
            "Active sphere indices:",
            active_indices.tolist(),
        )

        print(
            "Unclipped max |dq|:",
            f"{np.max(np.abs(estimated_delta_q)):.6f} rad",
        )

        print(
            "Joint-limit-clipped max |dq|:",
            f"{np.max(np.abs(clipped_delta_q)):.6f} rad",
        )

        print(
            "Before worst signed distance:",
            f"{np.min(base_distances) * 1000:+.3f} mm",
        )

        print(
            "After local LS correction:",
            f"{np.min(corrected_distances) * 1000:+.3f} mm",
        )

        print()

        print(
            "Per-joint estimated correction:"
        )

        for (
            joint_index,
            joint_name,
        ) in enumerate(
            baseline
            .JOINT_NAMES[
                :12
            ]
        ):

            print(
                f"  {joint_index:02d} "
                f"{joint_name:29s} "
                f"dq="
                f"{clipped_delta_q[joint_index]:+9.6f} rad  "
                f"allowed=["
                f"{lower_bounds[joint_index]:+8.4f},"
                f"{upper_bounds[joint_index]:+8.4f}]"
            )

        # Restore.
        uneven_data.qpos[
            :
        ] = base_qpos

        mujoco.mj_forward(
            uneven_model,
            uneven_data,
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
        "- FINAL RESIDUAL REQUIREMENT DIAGNOSTIC"
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
        "Jacobian epsilon:",
        f"{JACOBIAN_EPS:.4f} rad",
    )

    print(
        "mj_geomDistance distmax:",
        f"{DISTMAX:.2f} m",
    )

    required_paths = [
        BASELINE_SCRIPT,
        FLAT_SCENE,
        UNEVEN_SCENE,
        DATASET,
        CHECKPOINT,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing:\n"
                f"{path}"
            )

    baseline = (
        load_module(
            "kinematic_ppo_residual_diag_baseline",
            BASELINE_SCRIPT,
        )
    )

    (
        _,
        root_positions,
        predicted_leg_frames,
        fixed_waist_values,
        _,
    ) = (
        prepare_common(
            baseline
        )
    )

    flat_result = (
        audit_flat_reference(
            baseline,
            root_positions,
            predicted_leg_frames,
            fixed_waist_values,
        )
    )

    (
        uneven_model,
        uneven_spheres,
        terrain_ids,
        records,
    ) = (
        collect_uneven_worst_frames(
            baseline,
            root_positions,
            predicted_leg_frames,
            fixed_waist_values,
        )
    )

    analyze_worst_frames(
        baseline,
        root_positions,
        predicted_leg_frames,
        fixed_waist_values,
        uneven_model,
        uneven_spheres,
        terrain_ids,
        records,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Flat stance-like median "
        "signed distance:",
        (
            f"{flat_result['stance_like_median'] * 1000:+.3f} mm"
        ),
    )

    print(
        "Flat swing-like median "
        "signed distance:",
        (
            f"{flat_result['swing_like_median'] * 1000:+.3f} mm"
        ),
    )

    print()

    print(
        "Use these measured values "
        "to choose PPO residual scales "
        "and reward tolerances."
    )

    print(
        "This diagnostic does NOT "
        "automatically turn any measured "
        "number into a PPO hyperparameter."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()