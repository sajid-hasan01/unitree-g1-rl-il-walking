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
    "models\\g1_bc_openhe_kinematic.pt"
)


class BCPolicy(nn.Module):

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        output_dim=15,
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


def yaw_to_quat_wxyz(
    yaw_radians,
):
    half = (
        0.5
        * float(
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


def build_env(
    args,
):
    """
    Build the same G1 environment used
    by the approved reference showcase.

    IMPORTANT:

    This showcase NEVER calls:

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


def load_bc_checkpoint(
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
        output_dim != 15
    ):
        raise ValueError(
            "Expected BC dimensions "
            f"3 -> 15, got "
            f"{input_dim} -> "
            f"{output_dim}"
        )

    model = BCPolicy(
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
        "root_velocity",
        "controlled_joint_names",
        "fps",
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

    actions = np.asarray(
        dataset[
            "il_actions"
        ],
        dtype=np.float32,
    )

    joint_pos = np.asarray(
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
        actions.shape
        !=
        (300, 15)
    ):
        raise ValueError(
            "Expected actions "
            "(300, 15), got "
            f"{actions.shape}"
        )

    max_diff = float(
        np.max(
            np.abs(
                actions
                -
                joint_pos
            )
        )
    )

    if (
        max_diff
        >
        1e-7
    ):
        raise ValueError(
            "il_actions and "
            "joint_pos_15 differ; "
            "max abs difference="
            f"{max_diff}"
        )

    phase = (
        observations[
            :,
            0
        ]
    )

    if not np.all(
        np.diff(
            phase
        )
        >=
        0.0
    ):
        raise ValueError(
            "Phase/progress column "
            "is not monotonic."
        )

    if not np.isclose(
        phase[0],
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Expected phase start "
            f"0.0, got "
            f"{phase[0]}"
        )

    if not np.isclose(
        phase[-1],
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Expected phase end "
            f"1.0, got "
            f"{phase[-1]}"
        )

    joint_names = [
        str(
            x
        )
        for x in
        dataset[
            "controlled_joint_names"
        ].tolist()
    ]

    checkpoint_joint_names = (
        checkpoint.get(
            "output_features",
            None,
        )
    )

    if (
        checkpoint_joint_names
        is not None
    ):

        checkpoint_joint_names = [
            str(
                x
            )
            for x in
            checkpoint_joint_names
        ]

        if (
            checkpoint_joint_names
            !=
            joint_names
        ):
            raise ValueError(
                "Checkpoint output "
                "joint order does not "
                "match dataset joint order."
            )

    return (
        observations,
        joint_names,
    )


def build_display_observation(
    observations,
    motion_frame,
):
    """
    Build an interpolated observation ONLY
    for console display.

    IMPORTANT:

    The neural network is NOT evaluated
    on fractional interpolated observations
    during playback.

    Instead, the BC network predicts the
    original 300 dataset observations first.

    Then we interpolate BETWEEN those
    BC-predicted joint frames.

    This avoids nonlinear MLP excursions
    between training samples that can appear
    visually as body vibration.
    """

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
            + 1
        )

        alpha = (
            frame
            -
            frame_0
        )

    planar_velocity = (
        (1.0 - alpha)
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

    phase_progress = float(
        np.clip(
            frame
            /
            max(
                num_frames - 1,
                1,
            ),
            0.0,
            1.0,
        )
    )

    return np.array(
        [
            phase_progress,
            planar_velocity[0],
            planar_velocity[1],
        ],
        dtype=np.float32,
    )


def predict_dataset_frames(
    model,
    observations,
    normalization,
    device,
):
    """
    Evaluate the trained BC policy
    on the ORIGINAL 300 expert observations.

    Output:

        predicted_joint_frames
        shape = (300, 15)

    These are genuine neural-network
    predictions.

    Expert joint positions are NOT used
    to generate these predictions.
    """

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

        y_normalized = (
            model(
                x_tensor
            )
            .cpu()
            .numpy()
        )

    y = (
        y_normalized
        *
        normalization[
            "y_std"
        ]
        +
        normalization[
            "y_mean"
        ]
    )

    return y.astype(
        np.float32
    )


def interpolate_bc_predictions(
    predicted_joint_frames,
    motion_frame,
):
    """
    Linearly interpolate between
    adjacent BC-PREDICTED joint frames.

    This matches the temporal interpolation
    style used by the approved reference
    viewer.

    BOTH interpolation endpoints are
    generated by the trained BC network.

    Expert joint positions are NOT used
    as the robot's pose.
    """

    num_frames = (
        predicted_joint_frames.shape[
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
        + 1
    ) % num_frames

    alpha = float(
        frame
        -
        frame_0
    )

    predicted = (
        (1.0 - alpha)
        *
        predicted_joint_frames[
            frame_0
        ]
        +
        alpha
        *
        predicted_joint_frames[
            frame_1
        ]
    )

    return predicted.astype(
        np.float32
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


def apply_bc_pose(
    env,
    predicted_joint_frames,
    observations,
    motion_frame,
    step,
    args,
):
    """
    Apply a kinematic pose generated
    by the trained BC policy.

    NO MuJoCo physics step occurs.
    """

    # ============================================================
    # 1. DISPLAY OBSERVATION
    # ============================================================

    bc_observation = (
        build_display_observation(
            observations,
            motion_frame,
        )
    )

    # ============================================================
    # 2. INTERPOLATE BETWEEN BC-PREDICTED JOINT FRAMES
    # ============================================================

    predicted_walk_joints = (
        interpolate_bc_predictions(
            predicted_joint_frames,
            motion_frame,
        )
    )

    # ============================================================
    # 3. SAME INITIAL STANDING + TRANSITION AS REFERENCE SHOWCASE
    # ============================================================

    stand_joints = (
        env._get_stand_joint_positions()
    )

    blend = transition_blend(
        env,
        step,
        args,
    )

    applied_joints = (
        (1.0 - blend)
        *
        stand_joints
        +
        blend
        *
        predicted_walk_joints
    ).astype(
        np.float32
    )

    # ============================================================
    # 4. EXPERT JOINTS ONLY FOR ERROR MEASUREMENT
    #
    # They are NOT applied to the robot.
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
        (1.0 - blend)
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
    # 5. ROOT TRANSLATION
    #
    # Current BC outputs only 15 joints.
    #
    # Root translation/height therefore comes
    # from the expert trajectory ONLY for
    # visualization.
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

        env.data.qpos[0] = float(
            blend
            *
            args.root_motion_scale
            *
            root_delta[0]
        )

        env.data.qpos[1] = float(
            blend
            *
            args.root_motion_scale
            *
            root_delta[1]
        )

        env.data.qpos[2] = float(
            (1.0 - blend)
            *
            env.stand_qpos[2]
            +
            blend
            *
            (
                root_pos[2]
                +
                args.height_offset
            )
        )

    else:

        env.data.qpos[0] = (
            0.0
        )

        env.data.qpos[1] = (
            0.0
        )

        env.data.qpos[2] = float(
            env.stand_qpos[2]
        )

    # ============================================================
    # 6. SAME 180-DEGREE ORIENTATION AS APPROVED REFERENCE
    # ============================================================

    quat = yaw_to_quat_wxyz(
        np.deg2rad(
            args.reference_yaw_degrees
        )
    )

    env.data.qpos[
        3:7
    ] = quat

    env.data.qvel[
        :
    ] = 0.0

    # ============================================================
    # 7. APPLY ONLY BC-PREDICTED 15-DOF JOINT POSITIONS
    # ============================================================

    for (
        i,
        qpos_address,
    ) in enumerate(
        env.joint_qpos_addresses
    ):

        env.data.qpos[
            qpos_address
        ] = float(
            applied_joints[
                i
            ]
        )

    # ============================================================
    # 8. ACTUATOR TARGET VALUES
    #
    # These do NOT drive the robot because
    # mj_step() is never called.
    # ============================================================

    env.data.ctrl[
        :
    ] = 0.0

    for (
        i,
        actuator_id,
    ) in enumerate(
        env.actuator_ids
    ):

        env.data.ctrl[
            actuator_id
        ] = env._clip_ctrl(
            actuator_id,
            applied_joints[
                i
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
    #
    # NO mj_step()
    # NO env.step()
    # ============================================================

    mujoco.mj_forward(
        env.model,
        env.data,
    )

    # ============================================================
    # 10. BC VS EXPERT ERROR
    #
    # Expert joints are used only
    # to calculate this metric.
    # ============================================================

    error = (
        applied_joints
        -
        expert_display_joints
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error.astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            ),
            dtype=np.float64,
        )
    )

    return (
        bc_observation,
        blend,
        rmse,
        mae,
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
        load_bc_checkpoint(
            args.checkpoint,
            device,
        )
    )

    (
        observations,
        joint_names,
    ) = (
        validate_dataset(
            dataset,
            checkpoint,
        )
    )

    # ============================================================
    # PRECOMPUTE GENUINE BC OUTPUT
    #
    # The network predicts all 300 original
    # dataset observations once.
    #
    # Playback then interpolates between
    # these network predictions.
    # ============================================================

    predicted_joint_frames = (
        predict_dataset_frames(
            model,
            observations,
            normalization,
            device,
        )
    )

    expert_actions = np.asarray(
        dataset[
            "il_actions"
        ],
        dtype=np.float32,
    )

    integer_error = (
        predicted_joint_frames
        -
        expert_actions
    )

    integer_rmse = float(
        np.sqrt(
            np.mean(
                integer_error.astype(
                    np.float64
                )
                ** 2
            )
        )
    )

    integer_mae = float(
        np.mean(
            np.abs(
                integer_error
            ),
            dtype=np.float64,
        )
    )

    integer_max_abs = float(
        np.max(
            np.abs(
                integer_error
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
        joint_names
    ):
        raise ValueError(
            "Environment and BC "
            "joint orders do not match."
        )

    print()

    print(
        "=" * 84
    )

    print(
        "UNITREE G1 OPENHE "
        "KINEMATIC BEHAVIOR CLONING SHOWCASE"
    )

    print(
        "=" * 84
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
        "BC output: "
        "15 predicted G1 joint positions"
    )

    print(
        "Joint pose shown: "
        "TRAINED BC POLICY"
    )

    print(
        "Playback interpolation: "
        "LINEAR BETWEEN BC-PREDICTED FRAMES"
    )

    print(
        "Fractional-input MLP "
        "inference during playback: NO"
    )

    print(
        "Integer-frame BC "
        "RMSE [rad]:",
        f"{integer_rmse:.8f}",
    )

    print(
        "Integer-frame BC "
        "MAE [rad]:",
        f"{integer_mae:.8f}",
    )

    print(
        "Integer-frame BC "
        "max abs error [rad]:",
        f"{integer_max_abs:.8f}",
    )

    print(
        "Expert joint_pos_15 "
        "applied to robot: NO"
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
        "mujoco.mj_step(): "
        "NEVER CALLED"
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
        "=" * 84
    )

    print()

    viewer = None

    motion_frame = (
        0.0
    )

    rmse_values = []

    mae_values = []

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
            # SAME FIXED FRONT CAMERA AS APPROVED REFERENCE
            #
            # Camera is initialized ONCE.
            # It never follows the robot.
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
                        env.data.qpos[0]
                    ),
                    float(
                        env.data.qpos[1]
                    ),
                    float(
                        env.data.qpos[2]
                        +
                        args.camera_lookat_z_offset
                    ),
                ]

        for step in range(
            args.steps
        ):

            (
                bc_observation,
                blend,
                rmse,
                mae,
            ) = (
                apply_bc_pose(
                    env=
                        env,

                    predicted_joint_frames=
                        predicted_joint_frames,

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

            rmse_values.append(
                rmse
            )

            mae_values.append(
                mae
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
                    f"{float(bc_observation[0]):.4f}, "
                    f"vx_ref="
                    f"{float(bc_observation[1]):+.4f}, "
                    f"vy_ref="
                    f"{float(bc_observation[2]):+.4f}, "
                    f"blend="
                    f"{blend:.3f}, "
                    f"BC_vs_expert_RMSE="
                    f"{rmse:.6f} rad"
                )

            if (
                viewer
                is not None
            ):

                # FIXED CAMERA:
                #
                # sync only.
                #
                # No camera lookat update here.

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
                    "BC showcase stopped "
                    "at showcase stop step."
                )

                break

            # ====================================================
            # SAME MOTION-FRAME PROGRESSION
            # AS APPROVED REFERENCE SHOWCASE
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

    if rmse_values:

        print()

        print(
            "=" * 84
        )

        print(
            "KINEMATIC BC SHOWCASE SUMMARY"
        )

        print(
            "=" * 84
        )

        print(
            "Mean displayed "
            "BC-vs-expert joint "
            "RMSE [rad]:",
            f"{float(np.mean(rmse_values)):.8f}",
        )

        print(
            "Mean displayed "
            "BC-vs-expert joint "
            "MAE [rad]:",
            f"{float(np.mean(mae_values)):.8f}",
        )

        print(
            "Max displayed "
            "per-step joint "
            "RMSE [rad]:",
            f"{float(np.max(rmse_values)):.8f}",
        )

        print()

        print(
            "Displayed 15-DOF joint motion "
            "came from the trained BC network."
        )

        print(
            "Between-frame motion used "
            "linear interpolation between "
            "BC predictions."
        )

        print(
            "Expert joints were used only "
            "to calculate the error metric."
        )

        print(
            "Expert root motion was used only "
            "for visual translation/height."
        )

        print(
            "=" * 84
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Kinematic showcase of the "
            "trained OpenHE Unitree G1 "
            "Behavior Cloning policy."
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
    # SAME APPROVED REFERENCE PRESENTATION SETTINGS
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
    # SAME FIXED FRONT CAMERA AS FINALIZED REFERENCE VIEW
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