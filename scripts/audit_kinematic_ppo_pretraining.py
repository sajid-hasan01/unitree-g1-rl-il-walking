from argparse import Namespace
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

FINAL_IL_SCRIPT = (
    IL_ROOT
    / "scripts"
    / "showcase_g1_bc_kinematic_12dof.py"
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

BASELINE_SCENE = (
    IL_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "_kinematic_rl_uneven_baseline.xml"
)


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


def max_abs_difference(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        a.shape
        !=
        b.shape
    ):
        return np.inf

    return float(
        np.max(
            np.abs(
                a
                -
                b
            )
        )
    )


def joint_id(
    model,
    name,
):
    result = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if result < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        result
    )


def geom_name(
    model,
    geom_id,
):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        int(
            geom_id
        ),
    )

    if name is None:
        return "<unnamed>"

    return str(
        name
    )


def body_name(
    model,
    body_id,
):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        int(
            body_id
        ),
    )

    if name is None:
        return "<unnamed>"

    return str(
        name
    )


def reset_model_data(
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


# ============================================================
# LOAD EXACT CURRENT FILES
# ============================================================

def load_project_modules():

    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Baseline script not found:\n"
            f"{BASELINE_SCRIPT}"
        )

    if not FINAL_IL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Final IL showcase not found:\n"
            f"{FINAL_IL_SCRIPT}"
        )

    baseline = load_module(
        "ppo_audit_current_baseline",
        BASELINE_SCRIPT,
    )

    final_il = load_module(
        "ppo_audit_final_il",
        FINAL_IL_SCRIPT,
    )

    return (
        baseline,
        final_il,
    )


# ============================================================
# FOOT SPHERE LOOKUP
# ============================================================

def get_foot_spheres(
    model,
):
    result = []

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

        name = body_name(
            model,
            body_id,
        )

        if (
            name
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

        radius = float(
            model.geom_size[
                geom_id,
                0,
            ]
        )

        side = (
            "left"
            if
            name.startswith(
                "left_"
            )
            else
            "right"
        )

        # Verified local-X arrangement:
        #
        # x = -0.05  -> rear / heel pair
        # x = +0.12  -> front / toe pair
        #
        # We classify from actual model coordinates,
        # not from fixed geom IDs.
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

                "body":
                    name,

                "side":
                    side,

                "region":
                    region,

                "local_pos":
                    local_pos,

                "radius":
                    radius,
            }
        )

    if len(result) != 8:
        raise RuntimeError(
            "Expected exactly 8 verified "
            "foot contact spheres, found "
            f"{len(result)}"
        )

    return result


# ============================================================
# TERRAIN LOOKUP
# ============================================================

def get_main_terrain_geoms(
    model,
):
    terrain = {}

    for geom_id in range(
        model.ngeom
    ):
        name = geom_name(
            model,
            geom_id,
        )

        if name.startswith(
            "rl_terrain_block_"
        ):
            terrain[
                int(
                    geom_id
                )
            ] = name

    return terrain


def get_side_terrain_geoms(
    model,
):
    terrain = {}

    for geom_id in range(
        model.ngeom
    ):
        name = geom_name(
            model,
            geom_id,
        )

        if name.startswith(
            "rl_side_terrain_"
        ):
            terrain[
                int(
                    geom_id
                )
            ] = name

    return terrain


# ============================================================
# TERRAIN XML CONSISTENCY
# ============================================================

def audit_terrain_scene(
    baseline,
    model,
):

    print()
    print(
        "=" * 100
    )

    print(
        "1. TERRAIN SCENE CONSISTENCY"
    )

    print(
        "=" * 100
    )

    terrain_ids = (
        get_main_terrain_geoms(
            model
        )
    )

    print(
        "Expected main blocks:",
        len(
            baseline.TERRAIN
        ),
    )

    print(
        "Compiled main blocks:",
        len(
            terrain_ids
        ),
    )

    if (
        len(
            terrain_ids
        )
        !=
        len(
            baseline.TERRAIN
        )
    ):
        raise RuntimeError(
            "Generated terrain XML does not "
            "match the current baseline script."
        )

    max_error = 0.0

    for (
        index,
        (
            start,
            end,
            height,
        ),
    ) in enumerate(
        baseline.TERRAIN
    ):
        name = (
            f"rl_terrain_block_"
            f"{index}"
        )

        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if geom_id < 0:
            raise RuntimeError(
                f"Missing terrain geom: "
                f"{name}"
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

        pos_error = max_abs_difference(
            model.geom_pos[
                geom_id
            ],
            expected_pos,
        )

        size_error = max_abs_difference(
            model.geom_size[
                geom_id
            ],
            expected_size,
        )

        max_error = max(
            max_error,
            pos_error,
            size_error,
        )

        print(
            f"{name:20s} "
            f"id={geom_id:3d} "
            f"x=[{start:+.2f},{end:+.2f}] "
            f"h={height:.3f} "
            f"pos_err={pos_error:.3e} "
            f"size_err={size_error:.3e}"
        )

    print()
    print(
        "Maximum terrain XML mismatch:",
        f"{max_error:.12e}",
    )

    if max_error > 1e-9:
        raise RuntimeError(
            "Current generated terrain XML "
            "does not exactly match the "
            "current baseline definition."
        )


# ============================================================
# FINAL IL SEMANTICS COMPARISON
# ============================================================

def audit_final_il_semantics(
    baseline,
    final_il,
    dataset,
):

    print()
    print(
        "=" * 100
    )

    print(
        "2. FINAL IL vs CURRENT UNEVEN BASELINE SEMANTICS"
    )

    print(
        "=" * 100
    )

    env_args = Namespace(
        dataset_path=
            str(
                DATASET
            ),

        height_offset=
            0.02,
    )

    env = final_il.build_env(
        env_args
    )

    try:
        env.reset()

        observations = np.asarray(
            dataset[
                "il_observations"
            ],
            dtype=np.float32,
        )

        joint_pos_15 = np.asarray(
            dataset[
                "joint_pos_15"
            ],
            dtype=np.float32,
        )

        root_positions = np.asarray(
            dataset[
                "root_positions"
            ],
            dtype=np.float32,
        )

        runtime_increment = (
            float(
                env.control_dt
            )
            *
            float(
                env.fps
            )
            *
            1.40
        )

        print(
            "env.control_dt:",
            float(
                env.control_dt
            ),
        )

        print(
            "env.fps:",
            float(
                env.fps
            ),
        )

        print(
            "Final IL increment "
            "(control_dt * fps * 1.40):",
            f"{runtime_increment:.12f}",
        )

        print(
            "Current baseline increment:",
            f"{float(baseline.MOTION_FRAME_INCREMENT):.12f}",
        )

        print(
            "Increment difference:",
            f"{abs(runtime_increment - float(baseline.MOTION_FRAME_INCREMENT)):.12e}",
        )

        root_array_diff = (
            max_abs_difference(
                env.reference_root_positions,
                root_positions,
            )
        )

        joint_array_diff = (
            max_abs_difference(
                env.reference_joint_positions,
                joint_pos_15,
            )
        )

        print()
        print(
            "env.reference_root_positions "
            "vs dataset root_positions "
            "max diff:",
            f"{root_array_diff:.12e}",
        )

        print(
            "env.reference_joint_positions "
            "vs dataset joint_pos_15 "
            "max diff:",
            f"{joint_array_diff:.12e}",
        )

        # ----------------------------------------------------
        # COMPARE STAND POSE
        # ----------------------------------------------------

        terrain_model = (
            mujoco.MjModel
            .from_xml_path(
                str(
                    BASELINE_SCENE
                )
            )
        )

        terrain_data = (
            reset_model_data(
                terrain_model
            )
        )

        direct_stand = []

        for name in baseline.JOINT_NAMES:

            jid = joint_id(
                terrain_model,
                name,
            )

            address = int(
                terrain_model
                .jnt_qposadr[
                    jid
                ]
            )

            direct_stand.append(
                float(
                    terrain_data
                    .qpos[
                        address
                    ]
                )
            )

        direct_stand = np.asarray(
            direct_stand,
            dtype=np.float64,
        )

        env_stand = np.asarray(
            env._get_stand_joint_positions(),
            dtype=np.float64,
        )

        stand_diff = (
            max_abs_difference(
                direct_stand,
                env_stand,
            )
        )

        print()
        print(
            "Direct scene stand pose "
            "vs final IL env stand pose "
            "max diff:",
            f"{stand_diff:.12e}",
        )

        print(
            "Direct stand root Z:",
            f"{float(terrain_data.qpos[2]):.9f}",
        )

        print(
            "Final IL stand root Z:",
            f"{float(env.stand_qpos[2]):.9f}",
        )

        # ----------------------------------------------------
        # COMPARE INTERPOLATION
        # ----------------------------------------------------

        device = torch.device(
            "cpu"
        )

        (
            bc_model,
            checkpoint,
            normalization,
        ) = baseline.load_bc_policy(
            device
        )

        predicted_leg_frames = (
            baseline.predict_all_leg_frames(
                bc_model,
                observations,
                normalization,
                device,
            )
        )

        sample_frames = [
            0.0,
            0.42,
            16.80,
            58.80,
            120.25,
            200.50,
            247.80,
            289.80,
            298.50,
        ]

        print()
        print(
            "Interpolation comparison:"
        )

        print(
            "motion_frame | "
            "root_diff | "
            "expert_joint_diff | "
            "BC_interp_diff"
        )

        maximum_root_diff = 0.0
        maximum_joint_diff = 0.0
        maximum_bc_diff = 0.0

        for motion_frame in sample_frames:

            baseline_root = (
                baseline.interpolate_array(
                    root_positions,
                    motion_frame,
                )
            )

            baseline_joints = (
                baseline.interpolate_array(
                    joint_pos_15,
                    motion_frame,
                )
            )

            (
                env_joints,
                _,
                env_root,
            ) = (
                env._interpolate_reference(
                    motion_frame
                )
            )

            final_bc = (
                final_il.interpolate_frames(
                    predicted_leg_frames,
                    motion_frame,
                )
            )

            baseline_bc = (
                baseline.interpolate_array(
                    predicted_leg_frames,
                    motion_frame,
                )
            )

            root_diff = max_abs_difference(
                baseline_root,
                env_root,
            )

            joint_diff = max_abs_difference(
                baseline_joints,
                env_joints,
            )

            bc_diff = max_abs_difference(
                baseline_bc,
                final_bc,
            )

            maximum_root_diff = max(
                maximum_root_diff,
                root_diff,
            )

            maximum_joint_diff = max(
                maximum_joint_diff,
                joint_diff,
            )

            maximum_bc_diff = max(
                maximum_bc_diff,
                bc_diff,
            )

            print(
                f"{motion_frame:11.2f} | "
                f"{root_diff:9.3e} | "
                f"{joint_diff:17.3e} | "
                f"{bc_diff:14.3e}"
            )

        print()
        print(
            "Maximum sampled root interpolation diff:",
            f"{maximum_root_diff:.12e}",
        )

        print(
            "Maximum sampled expert-joint interpolation diff:",
            f"{maximum_joint_diff:.12e}",
        )

        print(
            "Maximum sampled BC interpolation diff:",
            f"{maximum_bc_diff:.12e}",
        )

        print()
        print(
            "Baseline base terrain height:",
            float(
                baseline.BASE_TERRAIN_HEIGHT
            ),
        )

        print(
            "Final IL height_offset:",
            0.02,
        )

        print(
            "These should match on the "
            "lowest/starting terrain level."
        )

        return (
            predicted_leg_frames,
            checkpoint,
        )

    finally:
        env.close()


# ============================================================
# JOINT LIMIT AUDIT
# ============================================================

def audit_joint_limits(
    baseline,
    model,
    predicted_leg_frames,
    dataset,
):

    print()
    print(
        "=" * 100
    )

    print(
        "3. EXACT 12-DOF JOINT LIMIT / BC RANGE AUDIT"
    )

    print(
        "=" * 100
    )

    expert = np.asarray(
        dataset[
            "joint_pos_15"
        ],
        dtype=np.float64,
    )[
        :,
        :12
    ]

    bc = np.asarray(
        predicted_leg_frames,
        dtype=np.float64,
    )

    print(
        "All values below are radians."
    )

    print()

    header = (
        "idx | joint                         | "
        "limit_min  limit_max | "
        "BC_min    BC_max | "
        "margin_-  margin_+ | "
        "sym_safe"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    smallest_safe_margin = np.inf

    for index, name in enumerate(
        baseline.JOINT_NAMES[
            :12
        ]
    ):

        jid = joint_id(
            model,
            name,
        )

        limited = bool(
            model.jnt_limited[
                jid
            ]
        )

        if not limited:
            raise RuntimeError(
                f"Expected limited joint: "
                f"{name}"
            )

        low = float(
            model.jnt_range[
                jid,
                0,
            ]
        )

        high = float(
            model.jnt_range[
                jid,
                1,
            ]
        )

        bc_min = float(
            np.min(
                bc[
                    :,
                    index
                ]
            )
        )

        bc_max = float(
            np.max(
                bc[
                    :,
                    index
                ]
            )
        )

        expert_min = float(
            np.min(
                expert[
                    :,
                    index
                ]
            )
        )

        expert_max = float(
            np.max(
                expert[
                    :,
                    index
                ]
            )
        )

        lower_margin = (
            bc_min
            -
            low
        )

        upper_margin = (
            high
            -
            bc_max
        )

        symmetric_safe = min(
            lower_margin,
            upper_margin,
        )

        smallest_safe_margin = min(
            smallest_safe_margin,
            symmetric_safe,
        )

        print(
            f"{index:3d} | "
            f"{name:29s} | "
            f"{low:+9.4f} {high:+9.4f} | "
            f"{bc_min:+8.4f} {bc_max:+8.4f} | "
            f"{lower_margin:8.4f} "
            f"{upper_margin:8.4f} | "
            f"{symmetric_safe:8.4f}"
        )

        if (
            bc_min
            <
            low
            or
            bc_max
            >
            high
        ):
            raise RuntimeError(
                "BC prediction exceeds model "
                f"joint range for {name}"
            )

        if (
            expert_min
            <
            low
            or
            expert_max
            >
            high
        ):
            raise RuntimeError(
                "Expert trajectory exceeds "
                f"model joint range for {name}"
            )

    print()
    print(
        "Smallest symmetric joint-limit "
        "margin across all BC predictions:",
        f"{smallest_safe_margin:.6f} rad",
    )

    print()
    print(
        "IMPORTANT: this is only the "
        "mechanical joint-limit margin."
    )

    print(
        "It is NOT automatically the PPO "
        "residual-action scale."
    )


# ============================================================
# FOOT GEOMETRY AUDIT
# ============================================================

def audit_foot_geometry(
    model,
):

    print()
    print(
        "=" * 100
    )

    print(
        "4. VERIFIED FOOT COLLISION GEOMETRY"
    )

    print(
        "=" * 100
    )

    foot_spheres = (
        get_foot_spheres(
            model
        )
    )

    for item in foot_spheres:

        local_pos = (
            item[
                "local_pos"
            ]
        )

        print(
            f"geom_id={item['geom_id']:3d} "
            f"side={item['side']:5s} "
            f"region={item['region']:4s} "
            f"body={item['body']:23s} "
            f"local_pos=["
            f"{local_pos[0]:+.3f},"
            f"{local_pos[1]:+.3f},"
            f"{local_pos[2]:+.3f}] "
            f"radius="
            f"{item['radius']:.6f}"
        )

    main_terrain = (
        get_main_terrain_geoms(
            model
        )
    )

    side_terrain = (
        get_side_terrain_geoms(
            model
        )
    )

    print()
    print(
        "Main terrain geoms:"
    )

    for (
        geom_id,
        name,
    ) in sorted(
        main_terrain.items()
    ):
        print(
            f"  {geom_id:3d}: "
            f"{name}"
        )

    print()
    print(
        "Side terrain geoms:"
    )

    for (
        geom_id,
        name,
    ) in sorted(
        side_terrain.items()
    ):
        print(
            f"  {geom_id:3d}: "
            f"{name}"
        )

    return foot_spheres


# ============================================================
# EXACT BASELINE PENETRATION AUDIT
# ============================================================

def audit_baseline_penetration(
    baseline,
    model,
    predicted_leg_frames,
    dataset,
    checkpoint,
    foot_spheres,
):

    print()
    print(
        "=" * 100
    )

    print(
        "5. EXACT BC-ONLY UNEVEN-TERRAIN PENETRATION"
    )

    print(
        "=" * 100
    )

    data = reset_model_data(
        model
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

    fixed_waist_values = np.asarray(
        checkpoint[
            "fixed_waist_values"
        ],
        dtype=np.float32,
    )

    joint_addresses = (
        baseline.get_joint_addresses(
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

    if stand_root_z < 0.70:
        stand_root_z = 0.79

    root_start = (
        root_positions[
            0
        ]
        .copy()
    )

    foot_by_geom = {
        int(
            item[
                "geom_id"
            ]
        ):
            item
        for item
        in foot_spheres
    }

    foot_geom_ids = set(
        foot_by_geom.keys()
    )

    terrain_by_geom = (
        get_main_terrain_geoms(
            model
        )
    )

    terrain_geom_ids = set(
        terrain_by_geom.keys()
    )

    penetration_depths = []

    penetrating_frames = set()

    walking_penetrating_frames = set()

    side_depths = {
        "left": [],
        "right": [],
    }

    region_depths = {
        "heel": [],
        "toe": [],
    }

    block_depths = {
        name: []
        for name
        in terrain_by_geom.values()
    }

    frame_records = []

    maximum_abs_foot_y = 0.0

    motion_frame = 0.0

    actual_last_step = -1
    actual_last_motion_frame = 0.0
    actual_last_world_x = 0.0

    for step in range(
        baseline.DEMO_STOP_STEP
        +
        1
    ):

        actual_last_step = (
            step
        )

        actual_last_motion_frame = (
            float(
                motion_frame
            )
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
            baseline.interpolate_array(
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

        actual_last_world_x = (
            world_x
        )

        root_terrain_z = (
            baseline
            .smooth_root_terrain_height(
                world_x
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

        mujoco.mj_forward(
            model,
            data,
        )

        for item in foot_spheres:

            geom_id = int(
                item[
                    "geom_id"
                ]
            )

            maximum_abs_foot_y = max(
                maximum_abs_foot_y,
                abs(
                    float(
                        data.geom_xpos[
                            geom_id,
                            1,
                        ]
                    )
                ),
            )

        frame_penetrations = []

        for contact_index in range(
            data.ncon
        ):

            contact = (
                data.contact[
                    contact_index
                ]
            )

            geom_1 = int(
                contact.geom1
            )

            geom_2 = int(
                contact.geom2
            )

            foot_geom = None
            terrain_geom = None

            if (
                geom_1
                in
                foot_geom_ids
                and
                geom_2
                in
                terrain_geom_ids
            ):
                foot_geom = geom_1
                terrain_geom = geom_2

            elif (
                geom_2
                in
                foot_geom_ids
                and
                geom_1
                in
                terrain_geom_ids
            ):
                foot_geom = geom_2
                terrain_geom = geom_1

            if (
                foot_geom is None
                or
                terrain_geom is None
            ):
                continue

            distance = float(
                contact.dist
            )

            if distance >= 0.0:
                continue

            depth = (
                -distance
            )

            penetration_depths.append(
                depth
            )

            side = (
                foot_by_geom[
                    foot_geom
                ][
                    "side"
                ]
            )

            region = (
                foot_by_geom[
                    foot_geom
                ][
                    "region"
                ]
            )

            block_name = (
                terrain_by_geom[
                    terrain_geom
                ]
            )

            side_depths[
                side
            ].append(
                depth
            )

            region_depths[
                region
            ].append(
                depth
            )

            block_depths[
                block_name
            ].append(
                depth
            )

            frame_penetrations.append(
                (
                    foot_geom,
                    side,
                    region,
                    block_name,
                    depth,
                )
            )

        if frame_penetrations:

            penetrating_frames.add(
                step
            )

            if (
                step
                >=
                (
                    baseline
                    .INITIAL_STAND_STEPS
                    +
                    baseline
                    .TRANSITION_STEPS
                )
            ):
                walking_penetrating_frames.add(
                    step
                )

            frame_records.append(
                (
                    step,
                    float(
                        motion_frame
                    ),
                    world_x,
                    max(
                        item[
                            4
                        ]
                        for item
                        in frame_penetrations
                    ),
                    frame_penetrations,
                )
            )

        if (
            step
            >=
            baseline.INITIAL_STAND_STEPS
        ):

            motion_frame += (
                baseline
                .MOTION_FRAME_INCREMENT
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
                break

    total_frames = (
        actual_last_step
        +
        1
    )

    walking_start = (
        baseline
        .INITIAL_STAND_STEPS
        +
        baseline
        .TRANSITION_STEPS
    )

    walking_frame_count = max(
        total_frames
        -
        walking_start,
        0,
    )

    print(
        "Frames evaluated:",
        total_frames,
    )

    print(
        "Final evaluated step:",
        actual_last_step,
    )

    print(
        "Final evaluated motion frame:",
        f"{actual_last_motion_frame:.4f}",
    )

    print(
        "Final evaluated world X:",
        f"{actual_last_world_x:+.4f} m",
    )

    print(
        "Maximum |foot sphere Y|:",
        f"{maximum_abs_foot_y:.4f} m",
    )

    print(
        "Main terrain half-width:",
        "0.6200 m",
    )

    print()

    if not penetration_depths:

        print(
            "No exact foot-sphere ↔ main-terrain "
            "penetrations were detected."
        )

        return

    depths = np.asarray(
        penetration_depths,
        dtype=np.float64,
    )

    print(
        "Penetrating frames:",
        len(
            penetrating_frames
        ),
        "/",
        total_frames,
        "=",
        f"{100.0 * len(penetrating_frames) / total_frames:.3f}%",
    )

    if walking_frame_count > 0:

        print(
            "Penetrating frames after "
            "stand+transition:",
            len(
                walking_penetrating_frames
            ),
            "/",
            walking_frame_count,
            "=",
            (
                f"{100.0 * len(walking_penetrating_frames) / walking_frame_count:.3f}%"
            ),
        )

    print(
        "Total penetrating contacts:",
        len(
            depths
        ),
    )

    print(
        "Mean penetration depth:",
        f"{float(np.mean(depths)):.6f} m",
        f"({1000.0 * float(np.mean(depths)):.3f} mm)",
    )

    print(
        "Median penetration depth:",
        f"{float(np.median(depths)):.6f} m",
        f"({1000.0 * float(np.median(depths)):.3f} mm)",
    )

    print(
        "Maximum penetration depth:",
        f"{float(np.max(depths)):.6f} m",
        f"({1000.0 * float(np.max(depths)):.3f} mm)",
    )

    print()
    print(
        "LEFT / RIGHT:"
    )

    for side in [
        "left",
        "right",
    ]:

        values = np.asarray(
            side_depths[
                side
            ],
            dtype=np.float64,
        )

        if len(values) == 0:
            print(
                f"  {side:5s}: "
                f"count=0"
            )
        else:
            print(
                f"  {side:5s}: "
                f"count={len(values):4d}, "
                f"mean="
                f"{1000.0 * float(np.mean(values)):.3f} mm, "
                f"max="
                f"{1000.0 * float(np.max(values)):.3f} mm"
            )

    print()
    print(
        "HEEL / TOE:"
    )

    for region in [
        "heel",
        "toe",
    ]:

        values = np.asarray(
            region_depths[
                region
            ],
            dtype=np.float64,
        )

        if len(values) == 0:
            print(
                f"  {region:4s}: "
                f"count=0"
            )
        else:
            print(
                f"  {region:4s}: "
                f"count={len(values):4d}, "
                f"mean="
                f"{1000.0 * float(np.mean(values)):.3f} mm, "
                f"max="
                f"{1000.0 * float(np.max(values)):.3f} mm"
            )

    print()
    print(
        "BY TERRAIN BLOCK:"
    )

    for block_name in sorted(
        block_depths.keys()
    ):

        values = np.asarray(
            block_depths[
                block_name
            ],
            dtype=np.float64,
        )

        if len(values) == 0:
            print(
                f"  {block_name:20s}: "
                f"count=0"
            )
        else:
            print(
                f"  {block_name:20s}: "
                f"count={len(values):4d}, "
                f"mean="
                f"{1000.0 * float(np.mean(values)):.3f} mm, "
                f"max="
                f"{1000.0 * float(np.max(values)):.3f} mm"
            )

    print()
    print(
        "TOP 20 WORST PENETRATING FRAMES:"
    )

    worst_frames = sorted(
        frame_records,
        key=
            lambda item:
                item[
                    3
                ],
        reverse=True,
    )[
        :20
    ]

    for (
        step,
        motion_frame_value,
        world_x,
        maximum_depth,
        contacts,
    ) in worst_frames:

        short_contacts = ", ".join(
            [
                (
                    f"{side}-{region}@"
                    f"{block}:"
                    f"{depth * 1000.0:.1f}mm"
                )
                for (
                    _,
                    side,
                    region,
                    block,
                    depth,
                )
                in contacts
            ]
        )

        print(
            f"  step={step:04d} "
            f"frame={motion_frame_value:7.2f} "
            f"x={world_x:+.3f} "
            f"max={maximum_depth * 1000.0:7.2f}mm | "
            f"{short_contacts}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "UNITREE G1 KINEMATIC PPO PRE-TRAINING AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "RL workspace:"
    )

    print(
        RL_ROOT
    )

    print()

    print(
        "IL workspace:"
    )

    print(
        IL_ROOT
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
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    print()

    print(
        "Current baseline script:"
    )

    print(
        BASELINE_SCRIPT
    )

    print()

    print(
        "Final approved IL showcase:"
    )

    print(
        FINAL_IL_SCRIPT
    )

    print()

    print(
        "Generated uneven scene:"
    )

    print(
        BASELINE_SCENE
    )

    # --------------------------------------------------------
    # EXISTENCE CHECKS
    # --------------------------------------------------------

    required_paths = [
        DATASET,
        CHECKPOINT,
        BASELINE_SCRIPT,
        FINAL_IL_SCRIPT,
        BASELINE_SCENE,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # LOAD EXACT PROJECT CODE
    # --------------------------------------------------------

    (
        baseline,
        final_il,
    ) = (
        load_project_modules()
    )

    dataset = np.load(
        DATASET,
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # LOAD EXACT CURRENT TERRAIN MODEL
    # --------------------------------------------------------

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                BASELINE_SCENE
            )
        )
    )

    # --------------------------------------------------------
    # 1. TERRAIN CONSISTENCY
    # --------------------------------------------------------

    audit_terrain_scene(
        baseline,
        model,
    )

    # --------------------------------------------------------
    # 2. FINAL IL SEMANTICS
    # --------------------------------------------------------

    (
        predicted_leg_frames,
        checkpoint,
    ) = (
        audit_final_il_semantics(
            baseline,
            final_il,
            dataset,
        )
    )

    # --------------------------------------------------------
    # 3. JOINT LIMITS
    # --------------------------------------------------------

    audit_joint_limits(
        baseline,
        model,
        predicted_leg_frames,
        dataset,
    )

    # --------------------------------------------------------
    # 4. EXACT FOOT GEOMETRY
    # --------------------------------------------------------

    foot_spheres = (
        audit_foot_geometry(
            model
        )
    )

    # --------------------------------------------------------
    # 5. EXACT BASELINE PENETRATION
    # --------------------------------------------------------

    audit_baseline_penetration(
        baseline,
        model,
        predicted_leg_frames,
        dataset,
        checkpoint,
        foot_spheres,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "No PPO training was performed."
    )

    print(
        "No mj_step() was called."
    )

    print(
        "This script only inspected the "
        "existing BC model, dataset, MuJoCo "
        "kinematics, joint limits, terrain, "
        "and collision contacts."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()