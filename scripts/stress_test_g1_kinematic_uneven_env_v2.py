from __future__ import annotations

import numpy as np

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


SEED = 425
EPISODES = 20
EPISODE_LENGTH = 120
TOL = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_finite(name: str, value) -> None:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"Non-finite value detected in {name}: {value}")


def main() -> None:
    print("=" * 100)
    print("UNITREE G1 KINEMATIC PPO REWARD-V2 ENVIRONMENT STRESS TEST")
    print("=" * 100)
    print(f"Seed: {SEED}")
    print(f"Episodes: {EPISODES}")
    print(f"Episode length: {EPISODE_LENGTH}")
    print(f"Training step range: {TRAIN_START_STEP} -> {TRAIN_END_STEP}")
    print("Physics: NONE -- environment uses mj_forward only")
    print()

    # ------------------------------------------------------------------
    # 1. Seeded reset determinism
    # ------------------------------------------------------------------
    print("[1/4] Seeded reset determinism...")

    env_a = G1KinematicUnevenEnv(
        episode_length=EPISODE_LENGTH,
        random_start=True,
        residual_smoothing=0.35,
    )
    env_b = G1KinematicUnevenEnv(
        episode_length=EPISODE_LENGTH,
        random_start=True,
        residual_smoothing=0.35,
    )

    try:
        obs_a, info_a = env_a.reset(seed=SEED)
        obs_b, info_b = env_b.reset(seed=SEED)

        require_finite("seeded observation A", obs_a)
        require_finite("seeded observation B", obs_b)

        require(
            env_a.observation_space.contains(obs_a),
            "Seeded observation A is outside observation space.",
        )
        require(
            env_b.observation_space.contains(obs_b),
            "Seeded observation B is outside observation space.",
        )

        start_a = int(info_a["start_step"])
        start_b = int(info_b["start_step"])
        obs_diff = float(np.max(np.abs(obs_a - obs_b)))

        require(start_a == start_b, "Seeded reset start steps differ.")
        require(obs_diff <= 1e-12, "Seeded reset observations differ.")

        print(f"  start step A/B: {start_a}/{start_b}")
        print(f"  max observation difference: {obs_diff:.12g}")
        print("  PASS")
    finally:
        env_a.close()
        env_b.close()

    print()

    # ------------------------------------------------------------------
    # 2. Explicit reset at both training-range boundaries
    # ------------------------------------------------------------------
    print("[2/4] Explicit start-step reset boundaries...")

    env = G1KinematicUnevenEnv(
        episode_length=EPISODE_LENGTH,
        random_start=False,
        residual_smoothing=0.35,
    )

    try:
        for requested_step in (TRAIN_START_STEP, TRAIN_END_STEP):
            obs, info = env.reset(
                seed=SEED,
                options={"start_step": requested_step},
            )

            require(
                int(info["start_step"]) == requested_step,
                f"Explicit reset did not start at {requested_step}.",
            )
            require_finite("explicit-reset observation", obs)
            require(
                env.observation_space.contains(obs),
                "Explicit-reset observation is outside observation space.",
            )

            print(f"  requested {requested_step}: PASS")
    finally:
        env.close()

    print()

    # ------------------------------------------------------------------
    # 3. Random-action stress rollout
    # ------------------------------------------------------------------
    print("[3/4] Random-action stress rollout...")

    env = G1KinematicUnevenEnv(
        episode_length=EPISODE_LENGTH,
        random_start=True,
        residual_smoothing=0.35,
    )

    rng = np.random.default_rng(SEED)

    starts = []
    rewards = []
    min_signed_distances = []
    penetration_costs = []
    max_penetration_excesses = []

    terminated_count = 0
    truncated_count = 0
    total_steps = 0

    max_normalized_residual = 0.0
    max_joint_clip_cost = 0.0
    max_mechanical_violation = 0.0

    try:
        for episode in range(EPISODES):
            obs, info = env.reset(seed=SEED + episode)

            require_finite("reset observation", obs)
            require(
                env.observation_space.contains(obs),
                "Reset observation is outside observation space.",
            )

            start_step = int(info["start_step"])
            starts.append(start_step)

            done = False
            episode_steps = 0

            while not done:
                action = rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=12,
                ).astype(np.float32)

                require(
                    env.action_space.contains(action),
                    "Generated action is outside action space.",
                )

                obs, reward, terminated, truncated, info = env.step(action)

                require_finite("observation", obs)
                require_finite("reward", reward)
                require_finite("signed_distances", info["signed_distances"])
                require_finite("residual", info["residual"])
                require_finite("penetration_cost", info["penetration_cost"])
                require_finite(
                    "max_penetration_excess_m",
                    info["max_penetration_excess_m"],
                )

                require(
                    env.observation_space.contains(obs),
                    "Step observation is outside observation space.",
                )

                penetration_cost = float(info["penetration_cost"])
                penetration_excess = float(info["max_penetration_excess_m"])

                require(
                    penetration_cost >= -TOL,
                    f"Negative penetration cost: {penetration_cost}",
                )
                require(
                    penetration_excess >= -TOL,
                    f"Negative max penetration excess: {penetration_excess}",
                )

                residual = np.asarray(info["residual"], dtype=np.float64)
                scales = env.residual_scales.astype(np.float64)
                normalized_residual = np.abs(residual / scales)
                max_normalized_residual = max(
                    max_normalized_residual,
                    float(np.max(normalized_residual)),
                )

                require(
                    float(np.max(normalized_residual)) <= 1.0 + 1e-5,
                    "Smoothed residual exceeded configured residual bounds.",
                )

                clip_cost = float(info["joint_clip_cost"])
                max_joint_clip_cost = max(max_joint_clip_cost, clip_cost)

                applied_leg_joints = np.asarray(
                    env.current_metadata["applied_joints"][:12],
                    dtype=np.float64,
                )
                lower = env.joint_lower.astype(np.float64)
                upper = env.joint_upper.astype(np.float64)

                lower_violation = np.maximum(lower - applied_leg_joints, 0.0)
                upper_violation = np.maximum(applied_leg_joints - upper, 0.0)
                mechanical_violation = float(
                    np.max(np.maximum(lower_violation, upper_violation))
                )
                max_mechanical_violation = max(
                    max_mechanical_violation,
                    mechanical_violation,
                )

                require(
                    mechanical_violation <= 1e-7,
                    "Applied joint pose violated a mechanical joint limit.",
                )

                rewards.append(float(reward))
                min_signed_distances.append(
                    float(info["min_signed_distance_m"])
                )
                penetration_costs.append(penetration_cost)
                max_penetration_excesses.append(penetration_excess)

                total_steps += 1
                episode_steps += 1

                if terminated:
                    terminated_count += 1
                if truncated:
                    truncated_count += 1

                done = bool(terminated or truncated)

                require(
                    episode_steps <= EPISODE_LENGTH,
                    "Episode exceeded configured episode length.",
                )

        starts_array = np.asarray(starts, dtype=np.int32)
        rewards_array = np.asarray(rewards, dtype=np.float64)
        min_dist_array = np.asarray(min_signed_distances, dtype=np.float64)
        penetration_cost_array = np.asarray(penetration_costs, dtype=np.float64)
        max_pen_excess_array = np.asarray(
            max_penetration_excesses,
            dtype=np.float64,
        )

        require(total_steps > 0, "Stress test produced zero steps.")
        require(terminated_count == 0, "Kinematic environment terminated unexpectedly.")
        require(
            truncated_count == EPISODES,
            f"Expected {EPISODES} truncated episodes, got {truncated_count}.",
        )

        print(f"  episodes: {EPISODES}")
        print(f"  total steps: {total_steps}")
        print(f"  unique random starts: {len(np.unique(starts_array))}")
        print(f"  start-step range: {starts_array.min()} -> {starts_array.max()}")
        print(f"  terminated episodes: {terminated_count}")
        print(f"  truncated episodes: {truncated_count}")
        print(
            "  reward mean/std/min/max: "
            f"{rewards_array.mean():+.6f} / "
            f"{rewards_array.std():.6f} / "
            f"{rewards_array.min():+.6f} / "
            f"{rewards_array.max():+.6f}"
        )
        print(
            "  signed-distance mean minimum / global minimum: "
            f"{min_dist_array.mean() * 1000:.3f} mm / "
            f"{min_dist_array.min() * 1000:.3f} mm"
        )
        print(
            "  penetration-cost mean/max: "
            f"{penetration_cost_array.mean():.6f} / "
            f"{penetration_cost_array.max():.6f}"
        )
        print(
            "  max penetration excess mean/max: "
            f"{max_pen_excess_array.mean() * 1000:.3f} mm / "
            f"{max_pen_excess_array.max() * 1000:.3f} mm"
        )
        print(f"  max |residual/scale|: {max_normalized_residual:.9f}")
        print(f"  max joint clip cost: {max_joint_clip_cost:.9f}")
        print(f"  max mechanical joint-limit violation: {max_mechanical_violation:.12f}")
        print("  PASS")
    finally:
        env.close()

    print()

    # ------------------------------------------------------------------
    # 4. Final status
    # ------------------------------------------------------------------
    print("[4/4] Final status")
    print("  Seeded reset: PASS")
    print("  Explicit start-step resets: PASS")
    print("  Random-action rollout: PASS")
    print("  Observation/action spaces: PASS")
    print("  Reward-V2 penetration diagnostics: PASS")
    print("  Residual bounds: PASS")
    print("  Mechanical joint limits: PASS")
    print("  Kinematic-only environment: unchanged")
    print()
    print("=" * 100)
    print("REWARD-V2 STRESS TEST PASSED")
    print("=" * 100)
    print("No PPO training was performed.")


if __name__ == "__main__":
    main()