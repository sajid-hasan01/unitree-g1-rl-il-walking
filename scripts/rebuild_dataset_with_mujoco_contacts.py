import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_INPUT = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof.npz"
)

DEFAULT_OUTPUT = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)


def bool_array_summary(name, arr):
    arr = np.asarray(arr).astype(bool)
    print(f"{name}: shape={arr.shape}, left_mean={arr[:, 0].mean():.3f}, right_mean={arr[:, 1].mean():.3f}")


def get_contact_mask_for_reference_frame(env, frame_index):
    frame_index = int(frame_index) % env.num_frames

    env._apply_reference_state_initialization(frame_index)

    # Keep the reference frame exactly where requested after the initializer.
    env.motion_frame = float(frame_index)

    mujoco.mj_forward(env.model, env.data)

    left_contact, right_contact = env._get_foot_contacts()

    return bool(left_contact), bool(right_contact)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a processed G1 dataset with MuJoCo-collision-derived contact_mask. "
            "This replaces external/retargeter contact labels with contacts produced by "
            "the actual Unitree G1 MuJoCo collision geoms."
        )
    )

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--target_velocity", type=float, default=-0.10)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--height_offset", type=float, default=0.02)
    parser.add_argument("--reference_speed", type=float, default=0.10)
    parser.add_argument("--print_every", type=int, default=50)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    env = G1DynamicWalkingEnv(
        dataset_path=str(input_path),
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        action_scale=0.055,
        action_target_smoothing=0.35,
        height_offset=args.height_offset,
        reference_speed=args.reference_speed,
        initial_stand_steps=70,
        transition_steps=220,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=args.initial_yaw_degrees,
    )

    original = np.load(input_path, allow_pickle=True)
    output_data = {key: original[key] for key in original.files}

    original_contact_mask = None
    if "contact_mask" in output_data:
        original_contact_mask = output_data["contact_mask"].astype(np.float32)

    corrected_contact_mask = np.zeros((env.num_frames, 2), dtype=np.float32)

    print()
    print("=" * 90)
    print("REBUILDING DATASET CONTACT MASK FROM MUJOCO CONTACTS")
    print("=" * 90)
    print("Input:", input_path)
    print("Output:", output_path)
    print("Frames:", env.num_frames)
    print("Yaw degrees:", args.initial_yaw_degrees)
    print("Height offset:", args.height_offset)

    for frame in range(env.num_frames):
        left_contact, right_contact = get_contact_mask_for_reference_frame(env, frame)

        corrected_contact_mask[frame, 0] = 1.0 if left_contact else 0.0
        corrected_contact_mask[frame, 1] = 1.0 if right_contact else 0.0

        if args.print_every > 0 and frame % args.print_every == 0:
            print(
                f"frame={frame:04d} "
                f"mj_contact=({int(left_contact)}, {int(right_contact)})"
            )

    if original_contact_mask is not None:
        output_data["original_contact_mask"] = original_contact_mask

    output_data["contact_mask"] = corrected_contact_mask

    # Metadata arrays are convenient when inspecting later.
    output_data["contact_mask_source"] = np.array(["mujoco_collision_contacts"])
    output_data["contact_mask_height_offset"] = np.array([args.height_offset], dtype=np.float32)
    output_data["contact_mask_initial_yaw_degrees"] = np.array([args.initial_yaw_degrees], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_data)

    if original_contact_mask is not None:
        print()
        bool_array_summary("original contact_mask", original_contact_mask)
        bool_array_summary("mujoco contact_mask", corrected_contact_mask)

        original_bool = original_contact_mask.astype(bool)
        corrected_bool = corrected_contact_mask.astype(bool)

        left_agreement = float(np.mean(original_bool[:, 0] == corrected_bool[:, 0]))
        right_agreement = float(np.mean(original_bool[:, 1] == corrected_bool[:, 1]))
        avg_agreement = 0.5 * (left_agreement + right_agreement)

        print()
        print("Agreement with original labels:")
        print(f"  left:  {left_agreement:.3f}")
        print(f"  right: {right_agreement:.3f}")
        print(f"  avg:   {avg_agreement:.3f}")

        mismatch = np.where(np.any(original_bool != corrected_bool, axis=1))[0]
        print("First mismatched frames:", mismatch[:60].tolist())
        print("Mismatch count:", len(mismatch), "/", env.num_frames)

    print()
    print("Saved corrected dataset:", output_path)
    print("=" * 90)

    env.close()


if __name__ == "__main__":
    main()
