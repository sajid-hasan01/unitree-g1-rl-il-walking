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

DEFAULT_CHECKPOINT = "models\\g1_bc_openhe_kinematic.pt"


class BCPolicy(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def yaw_to_quat_wxyz(yaw_radians):
    half = 0.5 * float(yaw_radians)

    return np.array(
        [
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ],
        dtype=np.float64,
    )


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return (
            value
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return np.asarray(
        value,
        dtype=np.float32,
    )


def build_env(args):
    """
    Build the same G1 environment used by
    the approved reference showcase.

    IMPORTANT:
    This script never calls:

        env.step()
        mujoco.mj_step()

    MuJoCo is used only for:
        model loading
        joint lookup
        root/reference interpolation
        forward kinematics
        visualization
    """

    return G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,

        reference_mode="cyclic",

        target_forward_velocity=-0.08,

        action_scale=0.06,

        action_target_smoothing=0.55,

        frame_skip=5,

        max_episode_steps=1000,

        height_offset=args.height_offset,

        reference_speed=0.18,

        initial_stand_steps=120,

        transition_steps=350,

        random_start=False,

        enable_push=False,

        push_window_start=180,

        push_window_end=360,

        push_interval_min=120,

        push_interval_max=180,

        push_force_min=5.0,

        push_force_max=15.0,

        push_duration_steps=3,

        include_contact_phase_observation=True,

        use_reference_contact_mask=False,

        reference_start_frame=0,

        use_gait_lift_prior=False,

        gait_lift_prior_scale=0.45,

        initial_yaw_degrees=0.0,
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
            f"{input_dim} -> {output_dim}"
        )

    model = BCPolicy(
        input_dim,
        hidden_dim,
        output_dim,
    ).to(device)

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
        != (300, 3)
    ):
        raise ValueError(
            "Expected observations "
            f"(300, 3), got "
            f"{observations.shape}"
        )

    if (
        actions.shape
        != (300, 15)
    ):
        raise ValueError(
            "Expected actions "
            f"(300, 15), got "
            f"{actions.shape}"
        )

    max_diff = float(
        np.max(
            np.abs(
                actions
                - joint_pos
            )
        )
    )

    if max_diff > 1e-7:
        raise ValueError(
            "il_actions and joint_pos_15 "
            "differ; max abs difference="
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
        ) >= 0.0
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
            f"0.0, got {phase[0]}"
        )

    if not np.isclose(
        phase[-1],
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Expected phase end "
            f"1.0, got {phase[-1]}"
        )

    joint_names = [
        str(x)
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
            str(x)
            for x in
            checkpoint_joint_names
        ]

        if (
            checkpoint_joint_names
            != joint_names
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


def build_bc_observation(
    observations,
    motion_frame,
):
    """
    BC input:

        [
            phase_progress,
            root_velocity_x,
            root_velocity_y
        ]

    vx and vy are interpolated
    between neighboring frames.

    Phase 0 and phase 1 remain
    different because the first
    and final poses of this clip
    are different.
    """

    num_frames = (
        observations.shape[0]
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
        >= num_frames - 1
    ):
        frame_1 = (
            frame_0
        )

        alpha = 0.0

    else:
        frame_1 = (
            frame_0 + 1
        )

        alpha = (
            frame
            - frame_0
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


def predict_joints(
    model,
    observation,
    normalization,
    device,
):
    x = (
        (
            observation
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
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        y_normalized = (
            model(
                x_tensor
            )
            .squeeze(0)
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
    model,
    observations,
    normalization,
    motion_frame,
    step,
    args,
    device,
):
    # ============================================================
    # 1. BUILD BC INPUT
    # ============================================================

    bc_observation = (
        build_bc_observation(
            observations,
            motion_frame,
        )
    )

    # ============================================================
    # 2. BC NETWORK PREDICTS 15 G1 JOINT POSITIONS
    # ============================================================

    predicted_walk_joints = (
        predict_joints(
            model,
            bc_observation,
            normalization,
            device,
        )
    )

    stand_joints = (
        env._get_stand_joint_positions()
    )

    blend = transition_blend(
        env,
        step,
        args,
    )

    # Same initial stand + smooth transition
    # used by the approved reference showcase.
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
    # 3. EXPERT JOINTS ARE READ ONLY FOR ERROR MEASUREMENT
    #
    # They are NOT applied to the G1 robot pose.
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
    # 4. ROOT TRANSLATION
    #
    # BC outputs only the 15 controlled joints.
    #
    # Therefore root translation and root height are supplied
    # from the approved expert trajectory only for visualization.
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

        env.data.qpos[0] = 0.0

        env.data.qpos[1] = 0.0

        env.data.qpos[2] = float(
            env.stand_qpos[2]
        )

    # ============================================================
    # 5. SAME 180-DEGREE ORIENTATION AS REFERENCE SHOWCASE
    # ============================================================

    quat = yaw_to_quat_wxyz(
        np.deg2rad(
            args.reference_yaw_degrees
        )
    )

    env.data.qpos[3:7] = (
        quat
    )

    env.data.qvel[:] = 0.0

    # ============================================================
    # 6. APPLY ONLY BC-PREDICTED JOINT POSE
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
            applied_joints[i]
        )

    # ============================================================
    # 7. CONTROLS
    #
    # Controls do not drive anything because mj_step() is never
    # called, but keeping them aligned makes inspection cleaner.
    # ============================================================

    env.data.ctrl[:] = 0.0

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
            applied_joints[i],
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
    # 8. FORWARD KINEMATICS ONLY
    #
    # NO mj_step()
    # NO env.step()
    # ============================================================

    mujoco.mj_forward(
        env.model,
        env.data,
    )

    # ============================================================
    # 9. BC VS EXPERT ERROR
    #
    # Metric only.
    # Expert pose is not applied.
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


def run_showcase(args):

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

    env = build_env(
        args
    )

    env.reset()

    if (
        env.num_frames
        !=
        observations.shape[0]
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
        "=" * 84
    )

    print()

    viewer = None

    motion_frame = 0.0

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
            # Camera is configured only once.
            #
            # It DOES NOT follow the robot.
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
                    env=env,

                    model=model,

                    observations=
                        observations,

                    normalization=
                        normalization,

                    motion_frame=
                        motion_frame,

                    step=
                        step,

                    args=
                        args,

                    device=
                        device,
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
                == 0
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
                # only synchronize.
                #
                # DO NOT update lookat.
                viewer.sync()

            if args.real_time:

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
            # SAME MOTION FRAME PROGRESSION AS APPROVED REFERENCE
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

    args = parser.parse_args()

    run_showcase(
        args
    )


if __name__ == "__main__":
    main()