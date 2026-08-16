from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "openhe"
    / "walk3_subject4.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_rawlegs_15dof.npz"
)

START_FRAME = 1320
END_FRAME = 1620


CONTROLLED_15_JOINTS = [
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


def finite_difference_velocity(position, fps):
    position = np.asarray(
        position,
        dtype=np.float32,
    )

    velocity = np.zeros_like(
        position,
        dtype=np.float32,
    )

    dt = 1.0 / float(fps)

    if len(position) > 1:
        velocity[1:-1] = (
            position[2:]
            - position[:-2]
        ) / (2.0 * dt)

        velocity[0] = (
            position[1]
            - position[0]
        ) / dt

        velocity[-1] = (
            position[-1]
            - position[-2]
        ) / dt

    return velocity.astype(np.float32)


def main():
    print("=" * 100)
    print("OPENHE RAW-LEGS BASELINE DATASET")
    print("=" * 100)

    print("Raw file:")
    print(RAW_PATH)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw OpenHE file not found:\n{RAW_PATH}"
        )

    raw_all = joblib.load(
        RAW_PATH
    )

    if not isinstance(raw_all, dict):
        raise TypeError(
            "Expected top-level OpenHE object to be a dictionary."
        )

    motion_key = list(
        raw_all.keys()
    )[0]

    motion = raw_all[
        motion_key
    ]

    required_keys = [
        "root_trans_offset",
        "root_rot",
        "dof",
        "contact_mask",
        "fps",
    ]

    for key in required_keys:
        if key not in motion:
            raise KeyError(
                f"Missing required key: {key}"
            )

    root_positions_full = np.asarray(
        motion["root_trans_offset"],
        dtype=np.float32,
    )

    root_rot_full = np.asarray(
        motion["root_rot"],
        dtype=np.float32,
    )

    dof_full = np.asarray(
        motion["dof"],
        dtype=np.float32,
    )

    contact_mask_full = np.asarray(
        motion["contact_mask"],
        dtype=np.float32,
    )

    fps_value = motion["fps"]

    if isinstance(
        fps_value,
        np.ndarray,
    ):
        fps = float(
            fps_value.reshape(-1)[0]
        )
    else:
        fps = float(
            fps_value
        )

    start = START_FRAME
    end = END_FRAME

    root_positions = (
        root_positions_full[
            start:end
        ].copy()
    )

    root_rot = (
        root_rot_full[
            start:end
        ].copy()
    )

    dof_23 = (
        dof_full[
            start:end
        ].copy()
    )

    contact_mask = (
        contact_mask_full[
            start:end
        ].copy()
    )

    if dof_23.shape != (300, 23):
        raise ValueError(
            f"Unexpected cropped DOF shape: {dof_23.shape}"
        )

    # --------------------------------------------------
    # IMPORTANT:
    # Preserve original OpenHE first 12 leg DOFs exactly.
    # No scaling.
    # No smoothing.
    # --------------------------------------------------

    legs_12 = (
        dof_23[:, :12]
        .copy()
        .astype(np.float32)
    )

    waist_3 = np.zeros(
        (
            legs_12.shape[0],
            3,
        ),
        dtype=np.float32,
    )

    joint_pos_15 = np.concatenate(
        [
            legs_12,
            waist_3,
        ],
        axis=1,
    ).astype(np.float32)

    joint_vel_15 = (
        finite_difference_velocity(
            joint_pos_15,
            fps,
        )
    )

    root_positions_relative = (
        root_positions.copy()
    )

    root_positions_relative[:, 0] -= (
        root_positions_relative[0, 0]
    )

    root_positions_relative[:, 1] -= (
        root_positions_relative[0, 1]
    )

    root_velocity = (
        finite_difference_velocity(
            root_positions_relative,
            fps,
        )
    )

    num_frames = (
        joint_pos_15.shape[0]
    )

    phase = (
        np.arange(
            num_frames,
            dtype=np.float32,
        )
        / max(
            num_frames - 1,
            1,
        )
    )

    il_observations = np.stack(
        [
            phase,
            root_velocity[:, 0],
            root_velocity[:, 1],
        ],
        axis=1,
    ).astype(np.float32)

    il_actions = (
        joint_pos_15
        .copy()
        .astype(np.float32)
    )

    root_delta = (
        root_positions_relative[-1]
        - root_positions_relative[0]
    )

    xy_displacement = float(
        np.linalg.norm(
            root_delta[:2]
        )
    )

    duration_sec = (
        num_frames
        / fps
    )

    avg_xy_speed = (
        xy_displacement
        / duration_sec
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_PATH,

        joint_pos_15=joint_pos_15,
        joint_vel_15=joint_vel_15,

        il_observations=il_observations,
        il_actions=il_actions,

        root_positions=(
            root_positions_relative
            .astype(np.float32)
        ),

        root_rot=(
            root_rot
            .astype(np.float32)
        ),

        root_velocity=(
            root_velocity
            .astype(np.float32)
        ),

        contact_mask=(
            contact_mask
            .astype(np.float32)
        ),

        fps=np.array(
            [fps],
            dtype=np.float32,
        ),

        source_total_frames=np.array(
            [len(dof_full)],
            dtype=np.int32,
        ),

        source_start_frame=np.array(
            [start],
            dtype=np.int32,
        ),

        source_end_frame=np.array(
            [end],
            dtype=np.int32,
        ),

        source_xy_displacement=np.array(
            [xy_displacement],
            dtype=np.float32,
        ),

        source_avg_xy_speed=np.array(
            [avg_xy_speed],
            dtype=np.float32,
        ),

        source_motion_key=np.array(
            [motion_key]
        ),

        controlled_joint_names=np.array(
            CONTROLLED_15_JOINTS
        ),

        is_cyclic=np.array(
            [True]
        ),

        waist_held_stable=np.array(
            [True]
        ),

        leg_scale=np.array(
            [1.0],
            dtype=np.float32,
        ),

        smooth_window=np.array(
            [1],
            dtype=np.int32,
        ),

        preprocessing=np.array(
            [
                "raw OpenHE first 12 leg DOFs; "
                "no scaling; no smoothing; waist fixed zero"
            ]
        ),
    )

    print()
    print("Dataset created:")
    print(OUTPUT_PATH)

    print()
    print("motion_key       :", motion_key)
    print("source frames    :", start, "to", end - 1)
    print("frames           :", num_frames)
    print("fps              :", fps)
    print("dof source       :", dof_23.shape)
    print("joint_pos_15     :", joint_pos_15.shape)
    print("joint_vel_15     :", joint_vel_15.shape)

    print()
    print("leg_scale        : 1.0")
    print("smooth_window    : 1")
    print("waist fixed zero : True")

    print()
    print(
        "xy displacement  :",
        round(
            xy_displacement,
            6,
        ),
        "m",
    )

    print(
        "avg xy speed     :",
        round(
            avg_xy_speed,
            6,
        ),
        "m/s",
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()