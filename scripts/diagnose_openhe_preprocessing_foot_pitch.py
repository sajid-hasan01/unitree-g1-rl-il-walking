import importlib.util
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


RAW_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "openhe"
    / "walk3_subject4.pkl"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)

RAW_START = 1320
RAW_END = 1620

LEG_SCALE = 0.60
SMOOTH_WINDOW = 9

RIGHT_FOOT_SPHERES = [30, 31, 32, 33]
LEFT_FOOT_SPHERES = [15, 16, 17, 18]

TEST_FRAMES = [
    16,
    19,
    20,
    22,
    25,
    28,
]


def load_initializer():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_residual_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load initializer: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def smooth_array(x, window):
    """
    Exact reconstruction of the smoothing function
    from prepare_openhe_g1_legs_only_dataset.py.
    """

    if window <= 1:
        return x.astype(np.float32)

    if window % 2 == 0:
        window += 1

    pad = window // 2

    padded = np.pad(
        x,
        ((pad, pad), (0, 0)),
        mode="edge",
    )

    out = np.zeros_like(
        x,
        dtype=np.float32,
    )

    for i in range(len(x)):
        out[i] = np.mean(
            padded[i:i + window],
            axis=0,
        )

    return out.astype(np.float32)


def make_15dof(legs_12):
    waist_3 = np.zeros(
        (legs_12.shape[0], 3),
        dtype=np.float32,
    )

    return np.concatenate(
        [
            legs_12.astype(np.float32),
            waist_3,
        ],
        axis=1,
    ).astype(np.float32)


def sphere_bottom_z(env, geom_id):
    center_z = float(
        env.data.geom_xpos[geom_id, 2]
    )

    radius = float(
        env.model.geom_size[geom_id, 0]
    )

    return center_z - radius


def foot_bottom_values(env, geom_ids):
    return np.array(
        [
            sphere_bottom_z(
                env,
                geom_id,
            )
            for geom_id in geom_ids
        ],
        dtype=np.float64,
    )


def heel_to_toe_delta(values):
    """
    First two spheres are heel/rear.
    Last two spheres are toe/front.

    Positive:
        heel higher than toe
        -> toe-down.
    """

    heel = float(
        np.mean(values[0:2])
    )

    toe = float(
        np.mean(values[2:4])
    )

    return heel - toe


def set_controlled_pose(env, pose_15):
    for i, qpos_address in enumerate(
        env.joint_qpos_addresses
    ):
        env.data.qpos[qpos_address] = float(
            pose_15[i]
        )

    mujoco.mj_forward(
        env.model,
        env.data,
    )


def format_mm(values):
    return " ".join(
        f"{1000.0 * value:+7.2f}"
        for value in values
    )


def print_pose_geometry(
    env,
    label,
    pose_15,
):
    set_controlled_pose(
        env,
        pose_15,
    )

    right = foot_bottom_values(
        env,
        RIGHT_FOOT_SPHERES,
    )

    left = foot_bottom_values(
        env,
        LEFT_FOOT_SPHERES,
    )

    right_delta = heel_to_toe_delta(
        right
    )

    left_delta = heel_to_toe_delta(
        left
    )

    print(
        f"  {label:10s} RIGHT [30 31 32 33] mm: "
        f"{format_mm(right)}"
    )

    print(
        f"  {label:10s} RIGHT heel-to-toe : "
        f"{1000.0 * right_delta:+8.2f} mm"
    )

    print(
        f"  {label:10s} LEFT  [15 16 17 18] mm: "
        f"{format_mm(left)}"
    )

    print(
        f"  {label:10s} LEFT  heel-to-toe : "
        f"{1000.0 * left_delta:+8.2f} mm"
    )

    return {
        "right": right,
        "left": left,
        "right_delta": right_delta,
        "left_delta": left_delta,
    }


def main():
    raw_all = joblib.load(
        RAW_PATH
    )

    motion_key = list(
        raw_all.keys()
    )[0]

    raw_motion = raw_all[
        motion_key
    ]

    raw_dof_full = np.asarray(
        raw_motion["dof"],
        dtype=np.float32,
    )

    raw_dof_crop = raw_dof_full[
        RAW_START:RAW_END
    ].copy()

    # ---------------------------------------------
    # Version A: untouched OpenHE first 12 leg DOFs
    # ---------------------------------------------

    raw_legs_12 = raw_dof_crop[
        :,
        :12,
    ].copy()

    raw_15 = make_15dof(
        raw_legs_12
    )

    # ---------------------------------------------
    # Version B: leg_scale only
    # ---------------------------------------------

    mean_legs = np.mean(
        raw_legs_12,
        axis=0,
        keepdims=True,
    )

    scaled_legs_12 = (
        mean_legs
        + LEG_SCALE
        * (
            raw_legs_12
            - mean_legs
        )
    ).astype(np.float32)

    scaled_15 = make_15dof(
        scaled_legs_12
    )

    # ---------------------------------------------
    # Version C: scale + exact smoothing
    # ---------------------------------------------

    smoothed_legs_12 = smooth_array(
        scaled_legs_12,
        SMOOTH_WINDOW,
    )

    reconstructed_current_15 = (
        make_15dof(
            smoothed_legs_12
        )
    )

    # ---------------------------------------------
    # Load actual current processed dataset
    # ---------------------------------------------

    processed = np.load(
        PROCESSED_PATH,
        allow_pickle=True,
    )

    current_15 = np.asarray(
        processed["joint_pos_15"],
        dtype=np.float32,
    )

    print()
    print("=" * 138)
    print(
        "OPENHE PREPROCESSING / FOOT-PITCH DIAGNOSTIC"
    )
    print("=" * 138)

    print()
    print(
        "RAW crop shape             :",
        raw_dof_crop.shape,
    )

    print(
        "Current joint_pos_15 shape :",
        current_15.shape,
    )

    print(
        "leg_scale                  :",
        LEG_SCALE,
    )

    print(
        "smooth_window              :",
        SMOOTH_WINDOW,
    )

    print()

    print("=" * 138)
    print(
        "PART 1 — VERIFY CURRENT DATASET RECONSTRUCTION"
    )
    print("=" * 138)

    if (
        reconstructed_current_15.shape
        == current_15.shape
    ):
        difference = np.abs(
            reconstructed_current_15
            - current_15
        )

        print(
            "max abs difference  :",
            f"{np.max(difference):.12g}",
        )

        print(
            "mean abs difference :",
            f"{np.mean(difference):.12g}",
        )

        print(
            "exact array equal    :",
            np.array_equal(
                reconstructed_current_15,
                current_15,
            ),
        )

    else:
        print(
            "SHAPE MISMATCH:",
            reconstructed_current_15.shape,
            current_15.shape,
        )

    initialize_from_reference = (
        load_initializer()
    )

    env = G1DynamicWalkingEnv(
        dataset_path=str(
            PROCESSED_PATH
        ),
        reference_mode="cyclic",
        target_forward_velocity=0.10,
        action_scale=0.055,
        action_target_smoothing=0.25,
        frame_skip=5,
        height_offset=0.02,
        reference_speed=1.0,
        initial_stand_steps=70,
        transition_steps=220,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=0.0,
    )

    try:
        env.reset()

        print()
        print("=" * 138)
        print(
            "PART 2 — FOOT GEOMETRY BY PREPROCESSING STAGE"
        )
        print("=" * 138)

        print(
            "All sphere-bottom values are millimetres."
        )

        print(
            "Positive heel-to-toe delta = "
            "heel is above toe = toe-down foot."
        )

        for frame in TEST_FRAMES:

            print()
            print("-" * 138)
            print(
                f"FRAME {frame} "
                f"(raw frame {RAW_START + frame})"
            )
            print("-" * 138)

            # Establish the exact same root XYZ and
            # flat base orientation used by current RSI.
            initialize_from_reference(
                env,
                frame,
            )

            base_qpos = np.asarray(
                env.data.qpos,
                dtype=np.float64,
            ).copy()

            # -----------------------------------------
            # RAW
            # -----------------------------------------

            env.data.qpos[:] = (
                base_qpos
            )

            raw_result = print_pose_geometry(
                env,
                "RAW",
                raw_15[frame],
            )

            # -----------------------------------------
            # SCALED ONLY
            # -----------------------------------------

            env.data.qpos[:] = (
                base_qpos
            )

            scaled_result = (
                print_pose_geometry(
                    env,
                    "SCALED",
                    scaled_15[frame],
                )
            )

            # -----------------------------------------
            # CURRENT = SCALE + SMOOTH
            # -----------------------------------------

            env.data.qpos[:] = (
                base_qpos
            )

            current_result = (
                print_pose_geometry(
                    env,
                    "CURRENT",
                    current_15[frame],
                )
            )

            print()

            raw_delta = (
                raw_result[
                    "right_delta"
                ]
            )

            scaled_delta = (
                scaled_result[
                    "right_delta"
                ]
            )

            current_delta = (
                current_result[
                    "right_delta"
                ]
            )

            print(
                "  RIGHT heel-to-toe summary:"
            )

            print(
                f"      RAW     : "
                f"{1000.0 * raw_delta:+8.2f} mm"
            )

            print(
                f"      SCALED  : "
                f"{1000.0 * scaled_delta:+8.2f} mm "
                f"(change from RAW "
                f"{1000.0 * (scaled_delta - raw_delta):+8.2f} mm)"
            )

            print(
                f"      CURRENT : "
                f"{1000.0 * current_delta:+8.2f} mm "
                f"(smoothing change "
                f"{1000.0 * (current_delta - scaled_delta):+8.2f} mm)"
            )

        print()
        print("=" * 138)
        print(
            "DIAGNOSTIC COMPLETE"
        )
        print("=" * 138)

    finally:
        env.close()


if __name__ == "__main__":
    main()