import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DEFAULT_DATASET = (
    "datasets\\processed\\g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz"
)


def parse_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_candidate_frames(dataset_path, phase, max_candidates, stride):
    data = np.load(dataset_path, allow_pickle=True)
    if "contact_mask" not in data:
        raise RuntimeError("Dataset has no contact_mask")

    contact = np.asarray(data["contact_mask"]).astype(np.float32)
    n = contact.shape[0]

    candidates = []

    # After the v56/v57 early-contact hold ends, motion_frame is usually around
    # 1-2 frames. Search using an offset of +2 so the first real swing phase after
    # hold matches the desired phase.
    phase_probe_offset = 2

    for start in range(0, n, max(1, stride)):
        probe = (start + phase_probe_offset) % n
        left = contact[probe, 0] > 0.5
        right = contact[probe, 1] > 0.5

        if phase == "right_swing":
            ok = left and (not right)
        elif phase == "left_swing":
            ok = (not left) and right
        elif phase == "double":
            ok = left and right
        elif phase == "any":
            ok = True
        else:
            raise ValueError(phase)

        if ok:
            candidates.append(start)

    # Always include the current baseline for comparison.
    if 25 not in candidates:
        candidates.insert(0, 25)

    return candidates[:max_candidates]


def run_case(args, start_frame, scale):
    env = G1DynamicWalkingEnv(
        dataset_path=args.dataset_path,
        reference_mode="cyclic",
        target_forward_velocity=args.target_velocity,
        initial_yaw_degrees=args.initial_yaw_degrees,
        reference_start_frame=start_frame,
        height_offset=args.height_offset,
        action_scale=args.action_scale,
        action_target_smoothing=args.action_target_smoothing,
        reference_speed=args.reference_speed,
        initial_stand_steps=args.initial_stand_steps,
        transition_steps=args.transition_steps,
        include_contact_phase_observation=True,
        use_reference_contact_mask=True,
        use_gait_lift_prior=True,
        gait_lift_prior_scale=scale,
    )

    obs, info = env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    max_left_clearance = 0.0
    max_right_clearance = 0.0
    max_forward_speed = 0.0
    max_lateral_speed = 0.0
    min_up_z = 1.0
    max_support_slip = 0.0
    contact_match_count = 0
    contact_checks = 0
    first_real_phase = None

    total_reward = 0.0
    final_info = info
    steps = 0

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        steps = step + 1

        lclr = float(info.get("left_foot_clearance", 0.0))
        rclr = float(info.get("right_foot_clearance", 0.0))
        max_left_clearance = max(max_left_clearance, lclr)
        max_right_clearance = max(max_right_clearance, rclr)

        xv = float(info.get("x_velocity", 0.0))
        yv = float(info.get("y_velocity", 0.0))
        max_forward_speed = max(max_forward_speed, abs(xv))
        max_lateral_speed = max(max_lateral_speed, abs(yv))

        min_up_z = min(min_up_z, float(info.get("up_z", 1.0)))

        left_expected = info.get("left_expected_contact", None)
        right_expected = info.get("right_expected_contact", None)
        left_contact = bool(info.get("left_contact", False))
        right_contact = bool(info.get("right_contact", False))

        if left_expected is not None and right_expected is not None:
            if first_real_phase is None and (not (left_expected and right_expected)):
                first_real_phase = (bool(left_expected), bool(right_expected), int(step))

            contact_match_count += int(left_contact == bool(left_expected))
            contact_match_count += int(right_contact == bool(right_expected))
            contact_checks += 2

            if bool(left_expected):
                max_support_slip = max(max_support_slip, float(info.get("left_foot_slip", 0.0)))
            if bool(right_expected):
                max_support_slip = max(max_support_slip, float(info.get("right_foot_slip", 0.0)))

        if terminated or truncated:
            break

    env.close()

    contact_match = contact_match_count / max(contact_checks, 1)

    # Stable foot lift score: prefers survival, uprightness, useful clearance,
    # low speed explosion, low lateral drift, and low support slip.
    useful_clearance = max(max_left_clearance, max_right_clearance)
    score = (
        steps
        + 350.0 * min(useful_clearance, 0.08)
        + 120.0 * min_up_z
        + 50.0 * contact_match
        - 80.0 * max(0.0, max_forward_speed - 0.25)
        - 70.0 * max_lateral_speed
        - 30.0 * max_support_slip
        - 60.0 * abs(float(final_info.get("y_position", 0.0)))
    )

    return {
        "start_frame": int(start_frame),
        "scale": float(scale),
        "steps": int(steps),
        "score": float(score),
        "total_reward": float(total_reward),
        "max_left_clearance": float(max_left_clearance),
        "max_right_clearance": float(max_right_clearance),
        "max_forward_speed": float(max_forward_speed),
        "max_lateral_speed": float(max_lateral_speed),
        "min_up_z": float(min_up_z),
        "contact_match": float(contact_match),
        "max_support_slip": float(max_support_slip),
        "first_real_phase": first_real_phase,
        "final_x": float(final_info.get("x_position", 0.0)),
        "final_y": float(final_info.get("y_position", 0.0)),
        "final_xv": float(final_info.get("x_velocity", 0.0)),
        "final_yv": float(final_info.get("y_velocity", 0.0)),
        "final_h": float(final_info.get("base_height", 0.0)),
        "final_upz": float(final_info.get("up_z", 0.0)),
        "final_left_clearance": float(final_info.get("left_foot_clearance", 0.0)),
        "final_right_clearance": float(final_info.get("right_foot_clearance", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep v57 zero-action gait-lift prior start frames and scales. "
            "Use this to find a physically sane teacher prior before PPO training."
        )
    )

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--phase", type=str, default="right_swing", choices=["right_swing", "left_swing", "double", "any"])
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--candidate_stride", type=int, default=5)
    parser.add_argument("--scales", type=str, default="0.15,0.25,0.35,0.45,0.55,0.70")
    parser.add_argument("--target_velocity", type=float, default=-0.04)
    parser.add_argument("--initial_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--height_offset", type=float, default=0.10)
    parser.add_argument("--action_scale", type=float, default=0.35)
    parser.add_argument("--action_target_smoothing", type=float, default=0.35)
    parser.add_argument("--reference_speed", type=float, default=0.04)
    parser.add_argument("--initial_stand_steps", type=int, default=80)
    parser.add_argument("--transition_steps", type=int, default=400)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--top_k", type=int, default=15)

    args = parser.parse_args()

    scales = parse_floats(args.scales)
    candidates = build_candidate_frames(
        dataset_path=args.dataset_path,
        phase=args.phase,
        max_candidates=args.max_candidates,
        stride=args.candidate_stride,
    )

    print()
    print("=" * 130)
    print("V57 ZERO-ACTION GAIT PRIOR START-FRAME / SCALE SWEEP")
    print("=" * 130)
    print("Dataset:", args.dataset_path)
    print("Phase filter:", args.phase)
    print("Candidate start frames:", candidates)
    print("Scales:", scales)
    print("Steps per case:", args.steps)
    print()

    results = []

    for start_frame in candidates:
        for scale in scales:
            result = run_case(args, start_frame=start_frame, scale=scale)
            results.append(result)
            print(
                f"start={start_frame:03d} scale={scale:.2f} "
                f"steps={result['steps']:04d} score={result['score']:+.2f} "
                f"maxClr=({result['max_left_clearance']:.3f},{result['max_right_clearance']:.3f}) "
                f"minUp={result['min_up_z']:.3f} "
                f"maxV=({result['max_forward_speed']:.3f},{result['max_lateral_speed']:.3f}) "
                f"match={result['contact_match']:.3f} slip={result['max_support_slip']:.3f} "
                f"final x/y=({result['final_x']:+.3f},{result['final_y']:+.3f}) "
                f"upz={result['final_upz']:.3f}"
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    print()
    print("-" * 130)
    print(f"TOP {min(args.top_k, len(results))} CASES")
    print("-" * 130)
    print(
        "rank | start | scale | steps | score | maxL/R clr | minUp | maxV x/y | "
        "match | slip | first phase | final x/y | final h/upz"
    )
    print("-" * 130)

    for rank, r in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:04d} | "
            f"{r['start_frame']:05d} | "
            f"{r['scale']:.2f} | "
            f"{r['steps']:05d} | "
            f"{r['score']:+08.2f} | "
            f"{r['max_left_clearance']:.3f}/{r['max_right_clearance']:.3f} | "
            f"{r['min_up_z']:.3f} | "
            f"{r['max_forward_speed']:.3f}/{r['max_lateral_speed']:.3f} | "
            f"{r['contact_match']:.3f} | "
            f"{r['max_support_slip']:.3f} | "
            f"{r['first_real_phase']} | "
            f"{r['final_x']:+.3f}/{r['final_y']:+.3f} | "
            f"{r['final_h']:.3f}/{r['final_upz']:.3f}"
        )

    best = results[0]
    print()
    print("BEST_COMMAND_HINT:")
    print(
        f"--reference_start_frame {best['start_frame']} "
        f"--gait_lift_prior_scale {best['scale']:.2f}"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
