import argparse
from pathlib import Path
import sys
import time

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# IMPORTANT:
# This viewer intentionally uses the ORIGINAL Reward-V1
# environment because the available PPO checkpoint was trained
# in Reward V1.
#
# Reward V2 has NOT been PPO-trained yet.
from envs.g1_kinematic_uneven_env import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# EXISTING REWARD-V1 SMOKE CHECKPOINT
# ============================================================

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_smoke_seed425.zip"
)


# ============================================================
# VISUALIZATION DEFAULTS
# ============================================================

DEFAULT_AZIMUTH = 135.0
DEFAULT_DISTANCE = 5.0
DEFAULT_ELEVATION = -18.0
DEFAULT_LOOKAT_Z_OFFSET = -0.10

DEFAULT_SLEEP = 0.020

CRITICAL_START = 700
CRITICAL_END = 735


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize the Unitree G1 BC-only uneven-terrain baseline "
            "or the already-trained Reward-V1 8192-step PPO smoke policy."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "bc",
            "ppo",
        ],
        default="ppo",
        help=(
            "bc = frozen BC with zero PPO residual; "
            "ppo = trained Reward-V1 smoke PPO."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            "Reward-V1 PPO checkpoint. "
            "Used only when --mode ppo."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=425,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=(
            "Seconds to sleep per displayed reference step."
        ),
    )

    parser.add_argument(
        "--critical-slowdown",
        type=float,
        default=4.0,
        help=(
            "Sleep multiplier inside the known diagnostic "
            "region steps 700-735."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help=(
            "Print geometry diagnostics every N steps. "
            "Set 0 to print only critical-region steps."
        ),
    )

    parser.add_argument(
        "--pause-steps",
        type=str,
        default="",
        help=(
            "Comma-separated reference steps where the viewer should "
            "pause until ENTER is pressed, e.g. 708,716,723."
        ),
    )

    parser.add_argument(
        "--azimuth",
        type=float,
        default=DEFAULT_AZIMUTH,
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=DEFAULT_DISTANCE,
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=DEFAULT_ELEVATION,
    )

    parser.add_argument(
        "--lookat-z-offset",
        type=float,
        default=DEFAULT_LOOKAT_Z_OFFSET,
    )

    parser.add_argument(
        "--no-camera-follow",
        action="store_true",
        help=(
            "Keep the camera look-at point fixed instead of "
            "following the robot."
        ),
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def parse_pause_steps(
    text,
):

    text = str(
        text
    ).strip()

    if not text:
        return set()

    result = set()

    for item in text.split(","):

        item = item.strip()

        if not item:
            continue

        step = int(
            item
        )

        if not (
            TRAIN_START_STEP
            <=
            step
            <=
            TRAIN_END_STEP
        ):
            raise ValueError(
                "Pause step must be between "
                f"{TRAIN_START_STEP} and "
                f"{TRAIN_END_STEP}: {step}"
            )

        result.add(
            step
        )

    return result


def update_camera(
    viewer,
    env,
    args,
    *,
    initialize=False,
):

    with viewer.lock():

        if initialize:

            viewer.cam.distance = (
                args.distance
            )

            viewer.cam.azimuth = (
                args.azimuth
            )

            viewer.cam.elevation = (
                args.elevation
            )

        if (
            initialize
            or
            not args.no_camera_follow
        ):

            viewer.cam.lookat[:] = [
                float(
                    env.data.qpos[
                        0
                    ]
                ),

                float(
                    env.data.qpos[
                        1
                    ]
                ),

                float(
                    env.data.qpos[
                        2
                    ]
                    +
                    args.lookat_z_offset
                ),
            ]


def validate_saved_model_spaces(
    model,
    env,
):

    if (
        tuple(
            model.observation_space.shape
        )
        !=
        tuple(
            env.observation_space.shape
        )
    ):
        raise RuntimeError(
            "Saved PPO observation shape does not "
            "match current Reward-V1 environment. "
            f"model={model.observation_space.shape}, "
            f"env={env.observation_space.shape}"
        )

    if (
        tuple(
            model.action_space.shape
        )
        !=
        tuple(
            env.action_space.shape
        )
    ):
        raise RuntimeError(
            "Saved PPO action shape does not "
            "match current Reward-V1 environment. "
            f"model={model.action_space.shape}, "
            f"env={env.action_space.shape}"
        )


def geometry_summary(
    info,
):

    signed = np.asarray(
        info[
            "signed_distances"
        ],
        dtype=np.float64,
    )

    flat = np.asarray(
        info[
            "flat_reference_distances"
        ],
        dtype=np.float64,
    )

    raw_deficit = np.maximum(
        flat
        -
        signed,
        0.0,
    )

    actual_penetration = np.maximum(
        -signed,
        0.0,
    )

    return {
        "min_signed_mm":
            float(
                np.min(
                    signed
                )
                *
                1000.0
            ),

        "max_actual_penetration_mm":
            float(
                np.max(
                    actual_penetration
                )
                *
                1000.0
            ),

        "max_raw_deficit_mm":
            float(
                np.max(
                    raw_deficit
                )
                *
                1000.0
            ),
    }


def should_print(
    step,
    print_every,
):

    if (
        CRITICAL_START
        <=
        step
        <=
        CRITICAL_END
    ):
        return True

    if print_every <= 0:
        return False

    return (
        step
        %
        print_every
        ==
        0
    )


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def main():

    args = parse_args()

    if args.sleep < 0.0:
        raise ValueError(
            "--sleep must be >= 0."
        )

    if args.critical_slowdown < 0.0:
        raise ValueError(
            "--critical-slowdown must be >= 0."
        )

    pause_steps = parse_pause_steps(
        args.pause_steps
    )

    model = None

    if args.mode == "ppo":

        model_path = (
            args.model
            .expanduser()
            .resolve()
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "Reward-V1 PPO smoke checkpoint "
                "not found:\n"
                f"{model_path}"
            )

        print(
            "Loading Reward-V1 PPO smoke checkpoint..."
        )

        model = PPO.load(
            model_path,
            env=None,
            device="cpu",
            force_reset=True,
        )

        print(
            "PPO checkpoint loaded."
        )

    env = G1KinematicUnevenEnv(
        episode_length=
            (
                TRAIN_END_STEP
                -
                TRAIN_START_STEP
                +
                1
            ),

        random_start=False,

        residual_smoothing=0.35,
    )

    try:

        if model is not None:

            validate_saved_model_spaces(
                model,
                env,
            )

        obs, reset_info = env.reset(
            seed=args.seed,
            options={
                "start_step":
                    TRAIN_START_STEP,
            },
        )

        print()
        print(
            "=" * 100
        )

        if args.mode == "bc":

            print(
                "UNITREE G1 VISUAL CHECK: "
                "BC-ONLY UNEVEN-TERRAIN BASELINE"
            )

        else:

            print(
                "UNITREE G1 VISUAL CHECK: "
                "REWARD-V1 PPO SMOKE POLICY"
            )

        print(
            "=" * 100
        )

        print(
            "Mode:",
            args.mode,
        )

        print(
            "Reference range:",
            f"{TRAIN_START_STEP} -> {TRAIN_END_STEP}",
        )

        print(
            "Residual smoothing:",
            0.35,
        )

        print(
            "Physics rollout:",
            "NONE",
        )

        print(
            "MuJoCo update:",
            "mj_forward only",
        )

        if args.mode == "ppo":

            print(
                "PPO checkpoint:",
                args.model,
            )

            print(
                "Training represented by checkpoint:",
                "Reward-V1 8192-step smoke run",
            )

        else:

            print(
                "PPO residual:",
                "ZERO",
            )

        print(
            "Known V1 diagnostic region:",
            f"{CRITICAL_START} -> {CRITICAL_END}",
        )

        print(
            "Pause steps:",
            (
                sorted(
                    pause_steps
                )
                if
                pause_steps
                else
                "none"
            ),
        )

        print(
            "Camera:",
            (
                f"azimuth={args.azimuth}, "
                f"distance={args.distance}, "
                f"elevation={args.elevation}, "
                f"follow={not args.no_camera_follow}"
            ),
        )

        print(
            "=" * 100
        )
        print()

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
        ) as viewer:

            update_camera(
                viewer,
                env,
                args,
                initialize=True,
            )

            viewer.sync()

            print(
                "Viewer opened."
            )

            print(
                "The run begins at reference step "
                f"{TRAIN_START_STEP}."
            )

            print(
                "Watch especially around steps "
                "708, 716, 720, 723, and 728."
            )

            print()

            while (
                viewer.is_running()
                and
                env.current_step
                <
                TRAIN_END_STEP
            ):

                if args.mode == "bc":

                    action = np.zeros(
                        12,
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
                    )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                step = int(
                    info[
                        "step"
                    ]
                )

                update_camera(
                    viewer,
                    env,
                    args,
                    initialize=False,
                )

                viewer.sync()

                if should_print(
                    step,
                    args.print_every,
                ):

                    geometry = geometry_summary(
                        info
                    )

                    critical_marker = (
                        "  *** CRITICAL REGION ***"
                        if
                        CRITICAL_START
                        <=
                        step
                        <=
                        CRITICAL_END
                        else
                        ""
                    )

                    print(
                        f"step={step:03d} "
                        f"x={float(info['world_x']):+.3f} "
                        f"reward={float(reward):+.4f} "
                        f"minDist={geometry['min_signed_mm']:+7.2f}mm "
                        f"maxPen={geometry['max_actual_penetration_mm']:6.2f}mm "
                        f"maxDef={geometry['max_raw_deficit_mm']:6.2f}mm "
                        f"maxResidual={float(info['max_residual_rad']):.4f}rad"
                        f"{critical_marker}"
                    )

                if step in pause_steps:

                    print()
                    print(
                        "=" * 100
                    )

                    print(
                        f"PAUSED AT STEP {step}"
                    )

                    print(
                        "Inspect the robot/feet in the MuJoCo viewer."
                    )

                    print(
                        "Press ENTER in this PowerShell window "
                        "to continue."
                    )

                    print(
                        "=" * 100
                    )

                    input()

                sleep_time = float(
                    args.sleep
                )

                if (
                    CRITICAL_START
                    <=
                    step
                    <=
                    CRITICAL_END
                ):

                    sleep_time *= float(
                        args.critical_slowdown
                    )

                if sleep_time > 0.0:

                    time.sleep(
                        sleep_time
                    )

                if terminated or truncated:
                    break

        print()
        print(
            "=" * 100
        )

        print(
            "VISUALIZATION COMPLETE"
        )

        print(
            "=" * 100
        )

        if args.mode == "bc":

            print(
                "This was the frozen BC-only "
                "zero-residual baseline."
            )

        else:

            print(
                "This was the already-trained "
                "Reward-V1 PPO smoke checkpoint."
            )

        print(
            "No training was performed by this viewer."
        )

        print(
            "=" * 100
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()