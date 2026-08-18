import sys
import faulthandler
from pathlib import Path

import numpy as np

faulthandler.enable(all_threads=True)

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


SCALES = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.200,
    0.250,
]


def make_env(scale):

    return G1BCLineResidualEnvV5Delta(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,

        residual_scale=0.14,
        residual_ramp_frames=30,

        target_velocity=-0.18,
        max_episode_steps=600,

        delta_action_scale=scale,
    )


def evaluate(model, scale):

    env = make_env(scale)

    try:

        obs, _ = env.reset(seed=1234)

        step = 0

        max_y = 0.0
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

        max_delta = 0.0
        max_scaled_delta = 0.0
        max_combined = 0.0

        max_left_clear = float("-inf")
        max_right_clear = float("-inf")

        final = {}

        while True:

            delta, _ = model.predict(
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
                        np.abs(delta)
                    )
                ),
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(delta)

            step += 1

            y = float(info["y"])
            yaw = float(info["yaw_deg"])
            up = float(info["up_z"])

            wy = float(
                env.data.qvel[4]
            )

            max_y = max(
                max_y,
                abs(y),
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


            if (
                up95 is None
                and up <= 0.95
            ):
                up95 = step

            if (
                up90 is None
                and up <= 0.90
            ):
                up90 = step

            if (
                up80 is None
                and up <= 0.80
            ):
                up80 = step


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


            final = info


            if terminated or truncated:
                break


        if not np.isfinite(
            max_left_clear
        ):
            max_left_clear = float("nan")


        if not np.isfinite(
            max_right_clear
        ):
            max_right_clear = float("nan")


        return {
            "step":
                step,

            "post":
                max(
                    step - 135,
                    0,
                ),

            "x":
                float(
                    final["x"]
                ),

            "max_y":
                max_y,

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

            "dmax":
                max_delta,

            "sdmax":
                max_scaled_delta,

            "camax":
                max_combined,
        }

    finally:

        env.close()


def event(value):

    if value is None:
        return "-"

    return str(value)


checkpoints = sorted(
    CHECKPOINT_DIR.glob(
        "g1_v5_delta_*_steps.zip"
    ),
    key=lambda p: int(
        p.stem
        .split("_")[-2]
    ),
)


if not checkpoints:

    raise SystemExit(
        "No V5 checkpoints found."
    )


print("=" * 176)
print("V5 DELTA AUTHORITY SWEEP")
print("READ ONLY - NO TRAINING")
print("=" * 176)

print(
    f"{'MODEL':25s} "
    f"{'SCALE':>6s} "
    f"{'STEP':>5s} "
    f"{'POST':>5s} "
    f"{'X':>8s} "
    f"{'MAXY':>7s} "
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
    f"{'DMAX':>6s} "
    f"{'SDMAX':>6s} "
    f"{'CAMAX':>6s} "
    f"{'OK':>5s}"
)

print("-" * 176)


all_results = []


for path in checkpoints:

    model = PPO.load(
        str(path),
        device="cpu",
    )

    label = path.stem.replace(
        "g1_v5_delta_",
        "",
    )

    for scale in SCALES:

        result = evaluate(
            model,
            scale,
        )


        # ---------------------------------------------------------
        # Controlled walking gate.
        #
        # Survival alone does not count.
        # ---------------------------------------------------------

        controlled = (
            result["x"] <= -0.55
            and result["max_y"] < 0.15
            and result["max_yaw"] < 40.0
            and result["lsw"] >= 2
            and result["rsw"] >= 2
        )


        print(
            f"{label:25s} "
            f"{scale:6.3f} "
            f"{result['step']:5d} "
            f"{result['post']:5d} "
            f"{result['x']:+8.3f} "
            f"{result['max_y']:7.3f} "
            f"{result['max_yaw']:8.1f} "
            f"{result['min_up']:7.3f} "
            f"{result['max_wy']:7.2f} "
            f"{event(result['up95']):>5s} "
            f"{event(result['up90']):>5s} "
            f"{event(result['up80']):>5s} "
            f"{result['lsw']:4d} "
            f"{result['rsw']:4d} "
            f"{result['lclear']:7.1f} "
            f"{result['rclear']:7.1f} "
            f"{result['dmax']:6.3f} "
            f"{result['sdmax']:6.3f} "
            f"{result['camax']:6.3f} "
            f"{str(controlled):>5s}",
            flush=True,
        )


        result["model"] = label
        result["path"] = path
        result["scale"] = scale
        result["controlled"] = controlled

        all_results.append(
            result
        )


print()
print("=" * 176)
print("BEST CONTROLLED RESULT")
print("=" * 176)


controlled_results = [
    r
    for r in all_results
    if r["controlled"]
]


if controlled_results:

    best = max(
        controlled_results,
        key=lambda r: (
            r["step"],
            -r["max_y"],
            -r["max_yaw"],
        ),
    )

    print(
        f"model   = {best['model']}"
    )

    print(
        f"scale   = {best['scale']:.3f}"
    )

    print(
        f"steps   = {best['step']}"
    )

    print(
        f"post135 = {best['post']}"
    )

    print(
        f"X       = {best['x']:+.3f}"
    )

    print(
        f"maxY    = {best['max_y']:.3f}"
    )

    print(
        f"maxYaw  = {best['max_yaw']:.1f}"
    )

    print(
        f"L/R     = {best['lsw']}/{best['rsw']}"
    )

    print(
        f"SDMAX   = {best['sdmax']:.4f}"
    )

    print(
        f"CAMAX   = {best['camax']:.3f}"
    )

    print(
        f"path    = {best['path']}"
    )

else:

    print(
        "No V5 scale passed the controlled-walking gate."
    )


print()
print("=" * 176)
print("INTERPRETATION")
print("=" * 176)

print("""
Baseline reference:

steps   = 119
X       = -0.599
maxY    = 0.130
maxYaw  = 31.9 deg


CASE A:
A scale other than 0.10 gives >119 controlled steps.

Then V5 learned a useful correction direction, but the
correction authority was wrong.

We should use that scale for the next short training stage.


CASE B:
4096 checkpoint at a smaller scale keeps its excellent low Y
while bringing yaw below 40 deg.

Then the 4096 policy contains useful line correction but was
oversteering heading.


CASE C:
Larger scales improve survival but yaw/Y fail.

Then the learned delta direction is another survival exploit.
Do not continue V5 training.


CASE D:
All controlled results remain around 119.

Then the problem is NOT delta authority.

The next architecture should train the correction specifically
around the late V2 failure region instead of wasting PPO updates
over the entire already-good first 100 steps.
""")

print()
print("NO TRAINING WAS PERFORMED.")
print("=" * 176)
