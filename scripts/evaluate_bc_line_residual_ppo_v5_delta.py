import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env_v5_delta import (
    G1BCLineResidualEnvV5Delta,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v5_delta_checkpoints"
)

FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v5_delta_10k.zip"
)


def make_env():

    return G1BCLineResidualEnvV5Delta(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        delta_action_scale=0.10,
    )


def evaluate(
    delta_model=None,
):

    env = make_env()

    try:

        obs, _ = env.reset(
            seed=1234
        )

        steps = 0

        max_y = 0.0
        rms_y_sum = 0.0

        max_yaw = 0.0
        min_up = 1.0

        lsw = 0
        rsw = 0

        left_air = 0
        right_air = 0

        prev_l = None
        prev_r = None

        max_delta = 0.0
        rms_delta_sum = 0.0

        max_scaled_delta = 0.0
        max_combined = 0.0

        max_left_clear = float("-inf")
        max_right_clear = float("-inf")

        final = {}

        while True:

            if delta_model is None:

                delta = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32,
                )

            else:

                delta, _ = delta_model.predict(
                    obs,
                    deterministic=True,
                )

                delta = np.asarray(
                    delta,
                    dtype=np.float32,
                ).reshape(-1)


            max_delta = max(
                max_delta,
                float(
                    np.max(
                        np.abs(
                            delta
                        )
                    )
                ),
            )

            rms_delta_sum += float(
                np.mean(
                    delta ** 2
                )
            )


            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                delta
            )

            steps += 1


            y = float(
                info["y"]
            )

            yaw = float(
                info["yaw_deg"]
            )

            up = float(
                info["up_z"]
            )


            max_y = max(
                max_y,
                abs(y),
            )

            rms_y_sum += (
                y * y
            )

            max_yaw = max(
                max_yaw,
                abs(yaw),
            )

            min_up = min(
                min_up,
                up,
            )


            max_scaled_delta = max(
                max_scaled_delta,
                float(
                    info[
                        "v5_scaled_delta_max"
                    ]
                ),
            )

            max_combined = max(
                max_combined,
                float(
                    info[
                        "v5_combined_action_max"
                    ]
                ),
            )


            left = bool(
                info["left_contact"]
            )

            right = bool(
                info["right_contact"]
            )


            if (
                prev_l is not None
                and left != prev_l
            ):
                lsw += 1


            if (
                prev_r is not None
                and right != prev_r
            ):
                rsw += 1


            prev_l = left
            prev_r = right


            if not left:

                left_air += 1

                if up >= 0.80:

                    max_left_clear = max(
                        max_left_clear,
                        1000.0
                        * float(
                            info[
                                "left_true_clearance"
                            ]
                        ),
                    )


            if not right:

                right_air += 1

                if up >= 0.80:

                    max_right_clear = max(
                        max_right_clear,
                        1000.0
                        * float(
                            info[
                                "right_true_clearance"
                            ]
                        ),
                    )


            final = info


            if terminated or truncated:
                break


        if not np.isfinite(
            max_left_clear
        ):
            max_left_clear = float(
                "nan"
            )


        if not np.isfinite(
            max_right_clear
        ):
            max_right_clear = float(
                "nan"
            )


        return {
            "steps":
                steps,

            "post":
                max(
                    0,
                    steps - 135,
                ),

            "x":
                float(
                    final["x"]
                ),

            "max_y":
                max_y,

            "rms_y":
                float(
                    np.sqrt(
                        rms_y_sum
                        / max(
                            steps,
                            1,
                        )
                    )
                ),

            "max_yaw":
                max_yaw,

            "min_up":
                min_up,

            "lsw":
                lsw,

            "rsw":
                rsw,

            "left_air":
                left_air,

            "right_air":
                right_air,

            "left_clear":
                max_left_clear,

            "right_clear":
                max_right_clear,

            "delta_max":
                max_delta,

            "delta_rms":
                float(
                    np.sqrt(
                        rms_delta_sum
                        / max(
                            steps,
                            1,
                        )
                    )
                ),

            "scaled_delta_max":
                max_scaled_delta,

            "combined_max":
                max_combined,
        }

    finally:

        env.close()


# =====================================================================
# BASELINE
# =====================================================================

baseline = evaluate(
    None
)


candidates = []


for path in sorted(
    CHECKPOINT_DIR.glob(
        "*.zip"
    )
):

    candidates.append(
        (
            path.stem,
            path,
        )
    )


if FINAL_MODEL.exists():

    candidates.append(
        (
            "V5_FINAL",
            FINAL_MODEL,
        )
    )


print("=" * 165)
print("V5 FROZEN-V2 + DELTA PPO CHECKPOINT EVALUATION")
print("=" * 165)

print(
    f"{'MODEL':31s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'RMSY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'LCLR':>7s} "
    f"{'RCLR':>7s} "
    f"{'DMAX':>6s} "
    f"{'DRMS':>6s} "
    f"{'SDMAX':>6s} "
    f"{'CAMAX':>6s} "
    f"{'OK':>5s}"
)

print("-" * 165)


def print_result(
    label,
    r,
    ok,
):

    print(
        f"{label:31s} "
        f"{r['steps']:5d} "
        f"{r['post']:5d} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['rms_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['lsw']:4d} "
        f"{r['rsw']:4d} "
        f"{r['left_clear']:7.1f} "
        f"{r['right_clear']:7.1f} "
        f"{r['delta_max']:6.3f} "
        f"{r['delta_rms']:6.3f} "
        f"{r['scaled_delta_max']:6.3f} "
        f"{r['combined_max']:6.3f} "
        f"{str(ok):>5s}"
    )


print_result(
    "FROZEN_V2_ZERO_DELTA",
    baseline,
    True,
)


results = []


for label, path in candidates:

    model = PPO.load(
        str(path),
        device="cpu",
    )

    r = evaluate(
        model
    )


    # =============================================================
    # ANTI-EXPLOIT ACCEPTANCE GATE
    #
    # Important:
    # longer survival alone is NOT enough.
    #
    # Must preserve:
    # - useful -X progress
    # - straight line
    # - heading
    # - bilateral gait
    # =============================================================

    controlled = (
        r["x"] <= -0.55
        and r["max_y"] < 0.15
        and r["max_yaw"] < 40.0
        and r["lsw"] >= 2
        and r["rsw"] >= 2
    )


    print_result(
        label,
        r,
        controlled,
    )


    r["label"] = label
    r["path"] = path
    r["controlled"] = controlled

    results.append(
        r
    )


print()
print("=" * 165)
print("BASELINE")
print("=" * 165)

print(
    f"steps   = {baseline['steps']}"
)

print(
    f"X       = {baseline['x']:+.3f}"
)

print(
    f"maxY    = {baseline['max_y']:.3f}"
)

print(
    f"maxYaw  = {baseline['max_yaw']:.1f}"
)

print(
    f"L/R     = {baseline['lsw']}/{baseline['rsw']}"
)


controlled = [
    r
    for r in results
    if r["controlled"]
]


print()
print("=" * 165)
print("BEST CONTROLLED V5 CHECKPOINT")
print("=" * 165)


if controlled:

    best = max(
        controlled,
        key=lambda r: (
            r["steps"],
            -r["max_y"],
            -r["max_yaw"],
        ),
    )

    improvement = (
        best["steps"]
        - baseline["steps"]
    )

    print(
        f"model       = {best['label']}"
    )

    print(
        f"steps       = {best['steps']}"
    )

    print(
        f"improvement = {improvement:+d}"
    )

    print(
        f"post135     = {best['post']}"
    )

    print(
        f"X           = {best['x']:+.3f}"
    )

    print(
        f"maxY        = {best['max_y']:.3f}"
    )

    print(
        f"maxYaw      = {best['max_yaw']:.1f}"
    )

    print(
        f"L/R         = {best['lsw']}/{best['rsw']}"
    )

    print(
        f"delta max   = {best['delta_max']:.3f}"
    )

    print(
        f"path        = {best['path']}"
    )


    if (
        best["steps"] > 135
        and best["post"] > 0
    ):

        print()
        print(
            "MILESTONE: FULL BC TRANSITION SURVIVED"
        )

else:

    print(
        "No V5 checkpoint passed the controlled-walking gate."
    )

    print(
        "Keep the frozen V2 zero-delta controller as the benchmark."
    )


print()
print("=" * 165)
print("SUCCESS TARGET")
print("=" * 165)

print(
    "steps > 135"
)

print(
    "X <= -0.55 m"
)

print(
    "maxY < 0.15 m"
)

print(
    "maxYaw < 40 deg"
)

print(
    "bilateral contact switching"
)

print(
    "no survival-by-stalling exploit"
)

print("=" * 165)
