import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.g1_bc_line_residual_env_v6_late import (
    G1BCLineResidualEnvV6Late,
)


CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "ppo_bc_line_residual_v6_late_checkpoints"
)

FINAL_MODEL = (
    ROOT
    / "models"
    / "g1_ppo_bc_line_residual_v6_late_10k.zip"
)


def make_env():

    return G1BCLineResidualEnvV6Late(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        delta_action_scale=0.10,

        correction_start_step=95,
        correction_ramp_frames=5,

        # Full deployment trajectory.
        warm_start_training=False,
    )


def evaluate(
    model=None,
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
        max_abs_wy = 0.0

        up95 = None
        up90 = None
        up80 = None

        lsw = 0
        rsw = 0

        prev_l = None
        prev_r = None

        max_left_clear = float(
            "-inf"
        )

        max_right_clear = float(
            "-inf"
        )

        max_raw_delta = 0.0
        max_effective_delta = 0.0
        max_scaled_delta = 0.0
        max_combined = 0.0

        final = {}

        while True:

            if model is None:

                action = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32,
                )

            else:

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                action = np.asarray(
                    action,
                    dtype=np.float32,
                ).reshape(-1)


            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
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

            wy = float(
                env.data.qvel[4]
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

            max_abs_wy = max(
                max_abs_wy,
                abs(wy),
            )


            if (
                up95 is None
                and up <= 0.95
            ):
                up95 = steps

            if (
                up90 is None
                and up <= 0.90
            ):
                up90 = steps

            if (
                up80 is None
                and up <= 0.80
            ):
                up80 = steps


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


            if (
                not left
                and up >= 0.80
            ):

                max_left_clear = max(
                    max_left_clear,
                    1000.0
                    * float(
                        info[
                            "left_true_clearance"
                        ]
                    ),
                )


            if (
                not right
                and up >= 0.80
            ):

                max_right_clear = max(
                    max_right_clear,
                    1000.0
                    * float(
                        info[
                            "right_true_clearance"
                        ]
                    ),
                )


            max_raw_delta = max(
                max_raw_delta,
                float(
                    info[
                        "v6_raw_delta_max"
                    ]
                ),
            )

            max_effective_delta = max(
                max_effective_delta,
                float(
                    info[
                        "v6_effective_delta_max"
                    ]
                ),
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
                    steps - 135,
                    0,
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

            "max_wy":
                max_abs_wy,

            "up95":
                up95,

            "up90":
                up90,

            "up80":
                up80,

            "lsw":
                lsw,

            "rsw":
                rsw,

            "lclear":
                max_left_clear,

            "rclear":
                max_right_clear,

            "raw_delta":
                max_raw_delta,

            "effective_delta":
                max_effective_delta,

            "scaled_delta":
                max_scaled_delta,

            "combined":
                max_combined,
        }

    finally:

        env.close()


def event(
    value,
):

    return (
        "-"
        if value is None
        else str(value)
    )


baseline = evaluate(
    None
)


candidates = sorted(
    CHECKPOINT_DIR.glob(
        "g1_v6_late_*_steps.zip"
    ),
    key=lambda p: int(
        p.stem.split("_")[-2]
    ),
)


if FINAL_MODEL.exists():

    candidates.append(
        FINAL_MODEL
    )


print("=" * 185)
print("V6-LATE FULL-TRAJECTORY CHECKPOINT EVALUATION")
print("=" * 185)

print(
    f"{'MODEL':28s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
    f"{'RMSY':>7s} "
    f"{'MAXYAW':>8s} "
    f"{'MINUP':>7s} "
    f"{'MAXWY':>7s} "
    f"{'UP95':>5s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'LSW':>4s} "
    f"{'RSW':>4s} "
    f"{'LCLR':>7s} "
    f"{'RCLR':>7s} "
    f"{'DEFF':>6s} "
    f"{'SDMAX':>6s} "
    f"{'CAMAX':>6s} "
    f"{'OK':>5s}"
)

print("-" * 185)


def print_row(
    label,
    r,
    ok,
):

    print(
        f"{label:28s} "
        f"{r['steps']:5d} "
        f"{r['post']:5d} "
        f"{r['x']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['rms_y']:7.3f} "
        f"{r['max_yaw']:8.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['max_wy']:7.2f} "
        f"{event(r['up95']):>5s} "
        f"{event(r['up90']):>5s} "
        f"{event(r['up80']):>5s} "
        f"{r['lsw']:4d} "
        f"{r['rsw']:4d} "
        f"{r['lclear']:7.1f} "
        f"{r['rclear']:7.1f} "
        f"{r['effective_delta']:6.3f} "
        f"{r['scaled_delta']:6.3f} "
        f"{r['combined']:6.3f} "
        f"{str(ok):>5s}"
    )


print_row(
    "FROZEN_V2_BASELINE",
    baseline,
    True,
)


results = []


for path in candidates:

    model = PPO.load(
        str(path),
        device="cpu",
    )

    r = evaluate(
        model
    )


    # =============================================================
    # STRICT ANTI-EXPLOIT GATE
    # =============================================================

    controlled = (
        r["x"] <= -0.55
        and r["max_y"] < 0.15
        and r["max_yaw"] < 40.0
        and r["lsw"] >= 2
        and r["rsw"] >= 2
        and r["combined"] < 0.95
    )


    label = path.stem

    print_row(
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
print("=" * 185)
print("BEST CONTROLLED V6-LATE")
print("=" * 185)


controlled_results = [
    r
    for r in results
    if r["controlled"]
]


if controlled_results:

    best = max(
        controlled_results,
        key=lambda r: (
            r["steps"],
            -r["max_y"],
            -r["max_yaw"],
        ),
    )

    print(
        f"model       = {best['label']}"
    )

    print(
        f"steps       = {best['steps']}"
    )

    print(
        f"improvement = "
        f"{best['steps'] - baseline['steps']:+d}"
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
        f"UP95/90/80  = "
        f"{event(best['up95'])}/"
        f"{event(best['up90'])}/"
        f"{event(best['up80'])}"
    )

    print(
        f"L/R         = "
        f"{best['lsw']}/{best['rsw']}"
    )

    print(
        f"combined max= {best['combined']:.3f}"
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
            "MILESTONE PASSED: "
            "FULL BC TRANSITION SURVIVED."
        )

else:

    print(
        "No V6 checkpoint passed the controlled-walking gate."
    )


print()
print("=" * 185)
print("TARGET")
print("=" * 185)

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
    "bilateral stepping"
)

print(
    "combined action < 0.95"
)

print("=" * 185)
