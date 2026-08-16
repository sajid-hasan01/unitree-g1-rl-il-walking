import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\"
    "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)

DEFAULT_CHECKPOINT = (
    "models\\g1_bc_openhe_kinematic_12dof.pt"
)


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
    yaw_radians,
):
    half = (
        0.5
        *
        float(
            yaw_radians
        )
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


def build_env(
    args,
):
    """
    Kinematic showcase only.

    This script NEVER calls:

        env.step()
        mujoco.mj_step()

    MuJoCo is used only for:

        model loading
        joint lookup
        reference/root interpolation
        forward kinematics
        visualization
    """

    return G1DynamicWalkingEnv(
        dataset_path=
            args.dataset_path,

        reference_mode=
            "cyclic",

        target_forward_velocity=
            -0.08,

        action_scale=
            0.06,

        action_target_smoothing=
            0.55,

        frame_skip=
            5,

        max_episode_steps=
            1000,

        height_offset=
            args.height_offset,

        reference_speed=
            0.18,

        initial_stand_steps=
            120,

        transition_steps=
            350,

        random_start=
            False,

        enable_push=
            False,

        push_window_start=
            180,

        push_window_end=
            360,

        push_interval_min=
            120,

        push_interval_max=
            180,

        push_force_min=
            5.0,

        push_force_max=
            15.0,

        push_duration_steps=
            3,

        include_contact_phase_observation=
            True,

        use_reference_contact_mask=
            False,

        reference_start_frame=
            0,

        use_gait_lift_prior=
            False,

        gait_lift_prior_scale=
            0.45,

        initial_yaw_degrees=
            0.0,
    )


def load_checkpoint(
    checkpoint_path,
    device,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "output_features",
        "fixed_waist_joints",
        "fixed_waist_values",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:

        raise KeyError(
            f"Checkpoint missing keys: "
            f"{missing}"
        )

    input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    hidden_dim = int(
        checkpoint[
            "hidden_dim"
        ]
    )

    output_dim = int(
        checkpoint[
            "output_dim"
        ]
    )

    if (
        input_dim != 3
        or
        output_dim != 12
    ):

        raise ValueError(
            "Expected BC dimensions "
            f"3 -> 12, got "
            f"{input_dim} -> {output_dim}"
        )

    model = BCPolicy12DOF(
        input_dim=
            input_dim,

        hidden_dim=
            hidden_dim,

        output_dim=
            output_dim,
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


def validate_dataset(
    dataset,
    checkpoint,
):

    required = [
        "il_observations",
        "il_actions",
        "joint_pos_15",
        "root_positions",
        "controlled_joint_names",
        "waist_held_stable",
    ]

    missing = [
        key
        for key in required
        if key not in dataset.files
    ]

    if missing:

        raise KeyError(
            f"Dataset missing keys: "
            f"{missing}"
        )

    observations = np.asarray(
        dataset[
            "il_observations"
        ],
        dtype=np.float32,
    )

    actions_15 = np.asarray(
        dataset[
            "il_actions"
        ],
        dtype=np.float32,
    )

    joint_pos_15 = np.asarray(
        dataset[
            "joint_pos_15"
        ],
        dtype=np.float32,
    )

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
        actions_15.shape
        !=
        (300, 15)
    ):

        raise ValueError(
            "Expected actions "
            "(300, 15), got "
            f"{actions_15.shape}"
        )

    max_diff = float(
        np.max(
            np.abs(
                actions_15
                -
                joint_pos_15
            )
        )
    )

    if (
        max_diff
        >
        1e-7
    ):

        raise ValueError(
            "il_actions and joint_pos_15 "
            "do not match; "
            f"max abs difference="
            f"{max_diff}"
        )

    all_joint_names = [
        str(
            name
        )
        for name in
        dataset[
            "controlled_joint_names"
        ].tolist()
    ]

    learned_joint_names = (
        all_joint_names[
            :12
        ]
    )

    waist_joint_names = (
        all_joint_names[
            12:15
        ]
    )

    checkpoint_learned = [
        str(
            name
        )
        for name in
        checkpoint[
            "output_features"
        ]
    ]

    checkpoint_waist = [
        str(
            name
        )
        for name in
        checkpoint[
            "fixed_waist_joints"
        ]
    ]

    fixed_waist_values = np.asarray(
        checkpoint[
            "fixed_waist_values"
        ],
        dtype=np.float32,
    )

    if (
        checkpoint_learned
        !=
        learned_joint_names
    ):

        raise ValueError(
            "Checkpoint learned-joint "
            "order does not match dataset."
        )

    if (
        checkpoint_waist
        !=
        waist_joint_names
    ):

        raise ValueError(
            "Checkpoint waist-joint "
            "order does not match dataset."
        )

    if (
        fixed_waist_values.shape
        !=
        (3,)
    ):

        raise ValueError(
            "Expected exactly "
            "3 fixed waist values."
        )

    if (
        float(
            np.max(
                np.abs(
                    fixed_waist_values
                )
            )
        )
        >
        1e-7
    ):

        raise ValueError(
            "Expected fixed waist "
            "values [0, 0, 0]."
        )

    waist_held_stable = bool(
        np.asarray(
            dataset[
                "waist_held_stable"
            ]
        )
        .reshape(
            -1
        )[0]
    )

    if not waist_held_stable:

        raise ValueError(
            "Dataset does not confirm "
            "waist_held_stable=True."
        )

    expert_waist_max = float(
        np.max(
            np.abs(
                actions_15[
                    :,
                    12:15
                ]
            )
        )
    )

    if (
        expert_waist_max
        >
        1e-7
    ):

        raise ValueError(
            "Expert waist targets "
            "are not exactly zero; "
            f"max abs="
            f"{expert_waist_max}"
        )

    return (
        observations,
        actions_15,
        all_joint_names,
        waist_joint_names,
        fixed_waist_values,
    )


def predict_all_frames(
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


def interpolate_frames(
    frames,
    motion_frame,
):

    num_frames = (
        frames.shape[
            0
        ]
    )

    frame = (
        float(
            motion_frame
        )
        %
        float(
            num_frames
        )
    )

    frame_0 = int(
        np.floor(
            frame
        )
    )

    frame_1 = (
        frame_0
        +
        1
    ) % num_frames

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


def display_observation(
    observations,
    motion_frame,
):

    num_frames = (
        observations.shape[
            0
        ]
    )

    frame = (
        float(
            motion_frame
        )
        %
        float(
            num_frames
        )
    )

    frame_0 = int(
        np.floor(
            frame
        )
    )

    if (
        frame_0
        >=
        num_frames - 1
    ):

        frame_1 = (
            frame_0
        )

        alpha = (
            0.0
        )

    else:

        frame_1 = (
            frame_0
            +
            1
        )

        alpha = (
            frame
            -
            frame_0
        )

    velocity_xy = (
        (
            1.0
            -
            alpha
        )
        *
        observations[
            frame_0,
            1:3
        ]
        +
        alpha
        *
        observations[
            frame_1,
            1:3
        ]
    )

    phase = float(
        np.clip(
            frame
            /
            max(
                num_frames
                -
                1,
                1,
            ),
            0.0,
            1.0,
        )
    )

    return np.array(
        [
            phase,
            velocity_xy[
                0
            ],
            velocity_xy[
                1
            ],
        ],
        dtype=np.float32,
    )


def transition_blend(
    env,
    step,
    args,
):

    if (
        step
        <
        args.reference_initial_stand_steps
    ):

        return 0.0

    transition_step = (
        step
        -
        args.reference_initial_stand_steps
    )

    return float(
        env._smoothstep(
            transition_step
            /
            max(
                args.reference_transition_steps,
                1,
            )
        )
    )


def apply_pose(
    env,
    predicted_leg_frames,
    fixed_waist_values,
    observations,
    motion_frame,
    step,
    args,
):

    # ============================================================
    # 1. OBSERVATION FOR DISPLAY
    # ============================================================

    observation = (
        display_observation(
            observations,
            motion_frame,
        )
    )

    # ============================================================
    # 2. INTERPOLATE TRAINED BC LEG PREDICTIONS
    # ============================================================

    predicted_legs = (
        interpolate_frames(
            predicted_leg_frames,
            motion_frame,
        )
    )

    # ============================================================
    # 3. COMPLETE 15-DOF TARGET
    #
    # 12 legs = trained neural-network prediction
    # 3 waist = demonstrated constant zero
    # ============================================================

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

    stand_joints = (
        env._get_stand_joint_positions()
    )

    blend = (
        transition_blend(
            env,
            step,
            args,
        )
    )

    applied_joints = (
        (
            1.0
            -
            blend
        )
        *
        stand_joints
        +
        blend
        *
        predicted_walk_15
    ).astype(
        np.float32
    )

    # ============================================================
    # 4. EXPERT JOINTS ONLY FOR ERROR METRICS
    # ============================================================

    (
        expert_walk_joints,
        _,
        root_pos,
    ) = (
        env._interpolate_reference(
            motion_frame
        )
    )

    expert_display_joints = (
        (
            1.0
            -
            blend
        )
        *
        stand_joints
        +
        blend
        *
        expert_walk_joints
    ).astype(
        np.float32
    )

    # ============================================================
    # 5. ROOT MOTION
    #
    # Root is visualization-only because
    # current BC predicts joint positions only.
    # ============================================================

    if (
        env.has_reference_root_positions
    ):

        root_start = (
            env.reference_root_positions[
                0
            ]
        )

        root_delta = (
            root_pos
            -
            root_start
        )

        env.data.qpos[
            0
        ] = float(
            blend
            *
            args.root_motion_scale
            *
            root_delta[
                0
            ]
        )

        env.data.qpos[
            1
        ] = float(
            blend
            *
            args.root_motion_scale
            *
            root_delta[
                1
            ]
        )

        env.data.qpos[
            2
        ] = float(
            (
                1.0
                -
                blend
            )
            *
            env.stand_qpos[
                2
            ]
            +
            blend
            *
            (
                root_pos[
                    2
                ]
                +
                args.height_offset
            )
        )

    else:

        env.data.qpos[
            0
        ] = 0.0

        env.data.qpos[
            1
        ] = 0.0

        env.data.qpos[
            2
        ] = float(
            env.stand_qpos[
                2
            ]
        )

    # ============================================================
    # 6. SAME PRESENTATION YAW
    # ============================================================

    env.data.qpos[
        3:7
    ] = yaw_to_quat_wxyz(
        np.deg2rad(
            args.reference_yaw_degrees
        )
    )

    env.data.qvel[
        :
    ] = 0.0

    # ============================================================
    # 7. APPLY KINEMATIC JOINT POSE
    # ============================================================

    for (
        index,
        qpos_address,
    ) in enumerate(
        env.joint_qpos_addresses
    ):

        env.data.qpos[
            qpos_address
        ] = float(
            applied_joints[
                index
            ]
        )

    # ============================================================
    # 8. ACTUATOR VALUES
    #
    # Informational only.
    # No mj_step() occurs.
    # ============================================================

    env.data.ctrl[
        :
    ] = 0.0

    for (
        index,
        actuator_id,
    ) in enumerate(
        env.actuator_ids
    ):

        env.data.ctrl[
            actuator_id
        ] = env._clip_ctrl(
            actuator_id,
            applied_joints[
                index
            ],
        )

    for item in (
        env.upper_body_actuators
    ):

        actuator_id = (
            item[
                "actuator_id"
            ]
        )

        env.data.ctrl[
            actuator_id
        ] = env._clip_ctrl(
            actuator_id,
            item[
                "target_qpos"
            ],
        )

    # ============================================================
    # 9. FORWARD KINEMATICS ONLY
    # ============================================================

    mujoco.mj_forward(
        env.model,
        env.data,
    )

    # ============================================================
    # 10. ERROR METRICS
    # ============================================================

    leg_error = (
        applied_joints[
            :12
        ]
        -
        expert_display_joints[
            :12
        ]
    )

    full_error = (
        applied_joints
        -
        expert_display_joints
    )

    leg_rmse = float(
        np.sqrt(
            np.mean(
                leg_error.astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    full_rmse = float(
        np.sqrt(
            np.mean(
                full_error.astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    full_mae = float(
        np.mean(
            np.abs(
                full_error
            ),
            dtype=np.float64,
        )
    )

    return (
        observation,
        blend,
        leg_rmse,
        full_rmse,
        full_mae,
        applied_joints[
            12:15
        ].copy(),
    )


def run_showcase(
    args,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    dataset_path = Path(
        args.dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    dataset = np.load(
        dataset_path,
        allow_pickle=True,
    )

    (
        model,
        checkpoint,
        normalization,
    ) = (
        load_checkpoint(
            args.checkpoint,
            device,
        )
    )

    (
        observations,
        actions_15,
        all_joint_names,
        waist_joint_names,
        fixed_waist_values,
    ) = (
        validate_dataset(
            dataset,
            checkpoint,
        )
    )

    # ============================================================
    # RUN BC NETWORK ON ORIGINAL 300 OBSERVATIONS
    # ============================================================

    predicted_leg_frames = (
        predict_all_frames(
            model,
            observations,
            normalization,
            device,
        )
    )

    integer_leg_error = (
        predicted_leg_frames
        -
        actions_15[
            :,
            :12
        ]
    )

    integer_leg_rmse = float(
        np.sqrt(
            np.mean(
                integer_leg_error.astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    integer_leg_mae = float(
        np.mean(
            np.abs(
                integer_leg_error
            ),
            dtype=np.float64,
        )
    )

    integer_leg_max = float(
        np.max(
            np.abs(
                integer_leg_error
            )
        )
    )

    env = build_env(
        args
    )

    env.reset()

    if (
        env.num_frames
        !=
        observations.shape[
            0
        ]
    ):

        raise ValueError(
            "Frame-count mismatch: "
            f"env={env.num_frames}, "
            f"dataset="
            f"{observations.shape[0]}"
        )

    if (
        list(
            env.controlled_joint_names
        )
        !=
        all_joint_names
    ):

        raise ValueError(
            "Environment and dataset "
            "joint orders do not match."
        )

    print()

    print(
        "=" * 88
    )

    print(
        "UNITREE G1 OPENHE "
        "12-DOF KINEMATIC BEHAVIOR CLONING SHOWCASE"
    )

    print(
        "=" * 88
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Dataset:",
        args.dataset_path,
    )

    print(
        "Device:",
        device,
    )

    print(
        "BC input: "
        "[phase_progress, "
        "root_velocity_x, "
        "root_velocity_y]"
    )

    print(
        "Learned BC output: "
        "12 predicted leg joint positions"
    )

    print(
        "Fixed waist:",
        list(
            zip(
                waist_joint_names,
                fixed_waist_values.tolist(),
            )
        ),
    )

    print(
        "Playback interpolation: "
        "LINEAR BETWEEN BC-PREDICTED LEG FRAMES"
    )

    print(
        "Fractional-input MLP "
        "inference during playback: NO"
    )

    print(
        "Integer-frame leg RMSE [rad]:",
        f"{integer_leg_rmse:.8f}",
    )

    print(
        "Integer-frame leg MAE [rad]:",
        f"{integer_leg_mae:.8f}",
    )

    print(
        "Integer-frame leg "
        "max abs error [rad]:",
        f"{integer_leg_max:.8f}",
    )

    print(
        "Expert joint_pos_15 "
        "applied as robot pose: NO"
    )

    print(
        "Waist handling: "
        "FIXED TO DEMONSTRATED ZERO VALUES"
    )

    print(
        "Root translation: "
        "expert root_positions, "
        "visualization only"
    )

    print(
        "MuJoCo physics rollout: NONE"
    )

    print(
        "mujoco.mj_step(): NEVER CALLED"
    )

    print(
        "Reference replay speed:",
        args.reference_replay_speed,
    )

    print(
        "Root motion scale:",
        args.root_motion_scale,
    )

    print(
        "Yaw degrees:",
        args.reference_yaw_degrees,
    )

    print(
        "Fixed camera follow: NO"
    )

    print(
        "Camera distance:",
        args.camera_distance,
    )

    print(
        "Camera azimuth:",
        args.camera_azimuth,
    )

    print(
        "Camera elevation:",
        args.camera_elevation,
    )

    print(
        "Camera lookat Z offset:",
        args.camera_lookat_z_offset,
    )

    print(
        "=" * 88
    )

    print()

    viewer = None

    motion_frame = (
        0.0
    )

    leg_rmse_values = []

    full_rmse_values = []

    full_mae_values = []

    try:

        if not args.no_viewer:

            print(
                "Viewer opening..."
            )

            viewer = (
                mujoco.viewer.launch_passive(
                    env.model,
                    env.data,
                )
            )

            # ====================================================
            # SAME FIXED FRONT CAMERA
            #
            # Camera is configured ONCE.
            # Camera does NOT follow the robot.
            # ====================================================

            with viewer.lock():

                viewer.cam.distance = (
                    args.camera_distance
                )

                viewer.cam.azimuth = (
                    args.camera_azimuth
                )

                viewer.cam.elevation = (
                    args.camera_elevation
                )

                viewer.cam.lookat[:] = [
                    float(
                        env.data.qpos[
                            0
                        ]
                    ),

                    float(
                        env.data.qpos[
                            1
                        ]
                    ),

                    float(
                        env.data.qpos[
                            2
                        ]
                        +
                        args.camera_lookat_z_offset
                    ),
                ]

        for step in range(
            args.steps
        ):

            (
                observation,
                blend,
                leg_rmse,
                full_rmse,
                full_mae,
                waist,
            ) = (
                apply_pose(
                    env=
                        env,

                    predicted_leg_frames=
                        predicted_leg_frames,

                    fixed_waist_values=
                        fixed_waist_values,

                    observations=
                        observations,

                    motion_frame=
                        motion_frame,

                    step=
                        step,

                    args=
                        args,
                )
            )

            leg_rmse_values.append(
                leg_rmse
            )

            full_rmse_values.append(
                full_rmse
            )

            full_mae_values.append(
                full_mae
            )

            if (
                args.print_every > 0
                and
                step
                %
                args.print_every
                ==
                0
            ):

                print(
                    f"step={step:04d}, "
                    f"x="
                    f"{float(env.data.qpos[0]):+.3f}, "
                    f"y="
                    f"{float(env.data.qpos[1]):+.3f}, "
                    f"height="
                    f"{float(env.data.qpos[2]):.3f}, "
                    f"motion_frame="
                    f"{motion_frame:.2f}, "
                    f"phase="
                    f"{float(observation[0]):.4f}, "
                    f"blend="
                    f"{blend:.3f}, "
                    f"leg_RMSE="
                    f"{leg_rmse:.6f} rad, "
                    f"full_RMSE="
                    f"{full_rmse:.6f} rad, "
                    f"waist=["
                    f"{float(waist[0]):+.5f}, "
                    f"{float(waist[1]):+.5f}, "
                    f"{float(waist[2]):+.5f}]"
                )

            if (
                viewer
                is not None
            ):

                # Fixed camera.
                # Synchronize only.
                # Do NOT change lookat.

                viewer.sync()

            if (
                args.real_time
            ):

                time.sleep(
                    args.sleep_time
                )

            if (
                step
                >=
                args.demo_stop_step
            ):

                print()

                print(
                    "12-DOF BC showcase stopped "
                    "at showcase stop step."
                )

                break

            # ====================================================
            # SAME MOTION-FRAME PROGRESSION
            # ====================================================

            if (
                step
                >=
                args.reference_initial_stand_steps
            ):

                motion_frame += (
                    env.control_dt
                    *
                    env.fps
                    *
                    args.reference_replay_speed
                )

                motion_frame = (
                    motion_frame
                    %
                    env.num_frames
                )

    finally:

        if (
            viewer
            is not None
        ):

            viewer.close()

        env.close()

    if (
        leg_rmse_values
    ):

        print()

        print(
            "=" * 88
        )

        print(
            "12-DOF KINEMATIC "
            "BC SHOWCASE SUMMARY"
        )

        print(
            "=" * 88
        )

        print(
            "Mean displayed "
            "leg RMSE [rad]:",
            f"{float(np.mean(leg_rmse_values)):.8f}",
        )

        print(
            "Max displayed "
            "leg RMSE [rad]:",
            f"{float(np.max(leg_rmse_values)):.8f}",
        )

        print(
            "Mean displayed "
            "full 15-DOF RMSE [rad]:",
            f"{float(np.mean(full_rmse_values)):.8f}",
        )

        print(
            "Mean displayed "
            "full 15-DOF MAE [rad]:",
            f"{float(np.mean(full_mae_values)):.8f}",
        )

        print()

        print(
            "The 12 moving leg joints "
            "came from the trained "
            "BC neural network."
        )

        print(
            "The 3 waist joints stayed "
            "at their demonstrated "
            "constant values."
        )

        print(
            "Expert joints were used "
            "only for error measurement."
        )

        print(
            "Expert root motion was used "
            "only for visual "
            "translation/height."
        )

        print(
            "=" * 88
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Kinematic showcase of the "
            "trained 12-DOF OpenHE "
            "Unitree G1 BC policy."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--demo_stop_step",
        type=int,
        default=650,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--real_time",
        action="store_true",
    )

    parser.add_argument(
        "--sleep_time",
        type=float,
        default=0.007,
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
    )

    # ============================================================
    # SAME APPROVED PRESENTATION SETTINGS
    # ============================================================

    parser.add_argument(
        "--reference_replay_speed",
        type=float,
        default=1.40,
    )

    parser.add_argument(
        "--reference_initial_stand_steps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--reference_transition_steps",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--root_motion_scale",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--reference_yaw_degrees",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.02,
    )

    # ============================================================
    # SAME FIXED FRONT CAMERA
    # ============================================================

    parser.add_argument(
        "--camera_distance",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=-15.0,
    )

    parser.add_argument(
        "--camera_lookat_z_offset",
        type=float,
        default=-0.10,
    )

    args = (
        parser.parse_args()
    )

    run_showcase(
        args
    )


if __name__ == "__main__":
    main()