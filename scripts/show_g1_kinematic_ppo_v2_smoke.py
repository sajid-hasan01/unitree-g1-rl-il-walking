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


# ============================================================
# REWARD-V2 ENVIRONMENT
# ============================================================

from envs.g1_kinematic_uneven_env_v2 import (
    G1KinematicUnevenEnv,
    TRAIN_START_STEP,
    TRAIN_END_STEP,
)


# ============================================================
# TRAINED REWARD-V2 SMOKE CHECKPOINT
# ============================================================

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "ppo_kinematic"
    / "g1_kinematic_ppo_v2_smoke_seed425.zip"
)


# ============================================================
# VISUALIZATION DEFAULTS
# ============================================================

DEFAULT_AZIMUTH = 135.0
DEFAULT_DISTANCE = 5.0
DEFAULT_ELEVATION = -18.0
DEFAULT_LOOKAT_Z_OFFSET = -0.10

DEFAULT_SLEEP = 0.020


# ------------------------------------------------------------
# Regions selected directly from the deterministic V2
# evaluation.
#
# 680-700 contains the worst remaining V2 penetration.
#
# 708-735 contains the previously diagnosed region where V2
# now shows strong foot-orientation changes.
# ------------------------------------------------------------

CRITICAL_RANGES = (
    (680, 700),
    (708, 735),
)


IMPORTANT_STEPS = {
    395,
    435,
    624,
    681,
    682,
    683,
    689,
    690,
    698,
    699,
    708,
    716,
    720,
    723,
    728,
    741,
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize the Unitree G1 BC-only uneven-terrain "
            "baseline or the trained Reward-V2 8192-step "
            "PPO smoke policy."
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
            "ppo = trained Reward-V2 smoke PPO."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            "Reward-V2 PPO checkpoint. "
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
            "Sleep multiplier inside the V2 diagnostic regions."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help=(
            "Print diagnostics every N steps. "
            "Important/critical steps are always printed. "
            "Set 0 to print only important/critical steps."
        ),
    )

    parser.add_argument(
        "--pause-steps",
        type=str,
        default="",
        help=(
            "Comma-separated reference steps where the viewer "
            "should pause until ENTER is pressed, "
            "for example 682,708,716,723,728."
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


def in_critical_region(
    step,
):

    for (
        start,
        end,
    ) in CRITICAL_RANGES:

        if (
            start
            <=
            step
            <=
            end
        ):
            return True

    return False


def should_print(
    step,
    print_every,
):

    if step in IMPORTANT_STEPS:
        return True

    if in_critical_region(
        step
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
            "match current Reward-V2 environment. "
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
            "match current Reward-V2 environment. "
            f"model={model.action_space.shape}, "
            f"env={env.action_space.shape}"
        )

    if not np.allclose(
        model.observation_space.low,
        env.observation_space.low,
    ):
        raise RuntimeError(
            "Saved PPO observation lower bounds do not "
            "match Reward-V2 environment."
        )

    if not np.allclose(
        model.observation_space.high,
        env.observation_space.high,
    ):
        raise RuntimeError(
            "Saved PPO observation upper bounds do not "
            "match Reward-V2 environment."
        )

    if not np.allclose(
        model.action_space.low,
        env.action_space.low,
    ):
        raise RuntimeError(
            "Saved PPO action lower bounds do not "
            "match Reward-V2 environment."
        )

    if not np.allclose(
        model.action_space.high,
        env.action_space.high,
    ):
        raise RuntimeError(
            "Saved PPO action upper bounds do not "
            "match Reward-V2 environment."
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

    actual_penetration = np.maximum(
        -signed,
        0.0,
    )

    raw_deficit = np.maximum(
        flat
        -
        signed,
        0.0,
    )


    # ========================================================
    # CURRENT FOOT GEOMETRY
    # ========================================================

    left_heel = float(
        np.mean(
            signed[
                0:2
            ]
        )
    )

    left_toe = float(
        np.mean(
            signed[
                2:4
            ]
        )
    )

    right_heel = float(
        np.mean(
            signed[
                4:6
            ]
        )
    )

    right_toe = float(
        np.mean(
            signed[
                6:8
            ]
        )
    )


    # ========================================================
    # SAME-PHASE FLAT BC GEOMETRY
    # ========================================================

    flat_left_heel = float(
        np.mean(
            flat[
                0:2
            ]
        )
    )

    flat_left_toe = float(
        np.mean(
            flat[
                2:4
            ]
        )
    )

    flat_right_heel = float(
        np.mean(
            flat[
                4:6
            ]
        )
    )

    flat_right_toe = float(
        np.mean(
            flat[
                6:8
            ]
        )
    )


    # ========================================================
    # HEEL-TO-TOE GAPS
    #
    # Positive:
    # heel is farther from terrain than toe
    # -> toe-down / toe-dominant
    #
    # Negative:
    # heel is closer to terrain than toe
    # -> heel-down relative orientation
    # ========================================================

    left_gap = (
        left_heel
        -
        left_toe
    )

    flat_left_gap = (
        flat_left_heel
        -
        flat_left_toe
    )

    right_gap = (
        right_heel
        -
        right_toe
    )

    flat_right_gap = (
        flat_right_heel
        -
        flat_right_toe
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

        "max_phase_penetration_excess_mm":
            float(
                info[
                    "max_penetration_excess_m"
                ]
                *
                1000.0
            ),

        "penetration_cost":
            float(
                info[
                    "penetration_cost"
                ]
            ),

        "left_gap_mm":
            float(
                left_gap
                *
                1000.0
            ),

        "flat_left_gap_mm":
            float(
                flat_left_gap
                *
                1000.0
            ),

        "left_extra_mm":
            float(
                (
                    left_gap
                    -
                    flat_left_gap
                )
                *
                1000.0
            ),

        "right_gap_mm":
            float(
                right_gap
                *
                1000.0
            ),

        "flat_right_gap_mm":
            float(
                flat_right_gap
                *
                1000.0
            ),

        "right_extra_mm":
            float(
                (
                    right_gap
                    -
                    flat_right_gap
                )
                *
                1000.0
            ),
    }


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


    # ========================================================
    # LOAD PPO
    # ========================================================

    if args.mode == "ppo":

        model_path = (
            args.model
            .expanduser()
            .resolve()
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "Reward-V2 PPO smoke checkpoint "
                "not found:\n"
                f"{model_path}"
            )

        print(
            "Loading Reward-V2 PPO smoke checkpoint..."
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


    # ========================================================
    # ENVIRONMENT
    # ========================================================

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


        # ====================================================
        # RESET
        # ====================================================

        obs, reset_info = env.reset(

            seed=args.seed,

            options={
                "start_step":
                    TRAIN_START_STEP,
            },
        )


        # ====================================================
        # INFORMATION
        # ====================================================

        print()

        print(
            "=" * 110
        )

        if args.mode == "bc":

            print(
                "UNITREE G1 VISUAL CHECK: "
                "BC-ONLY UNEVEN-TERRAIN BASELINE "
                "IN REWARD-V2 ENVIRONMENT"
            )

        else:

            print(
                "UNITREE G1 VISUAL CHECK: "
                "REWARD-V2 PPO SMOKE POLICY"
            )

        print(
            "=" * 110
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
                "Reward-V2 8192-step smoke run",
            )

        else:

            print(
                "PPO residual:",
                "ZERO",
            )

        print(
            "V2 diagnostic regions:",
            "680-700 and 708-735",
        )

        print(
            "Important single frames:",
            sorted(
                IMPORTANT_STEPS
            ),
        )

        print(
            "Pause steps:",
            (
                sorted(
                    pause_steps
                )

                if pause_steps

                else "none"
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
            "=" * 110
        )

        print()


        # ====================================================
        # VIEWER
        # ====================================================

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

            print()

            print(
                "Watch the whole gait."
            )

            print()

            print(
                "Pay special attention to:"
            )

            print(
                "  left-foot toe-dominant placement "
                "over the full trajectory"
            )

            print(
                "  worst remaining penetration "
                "around steps 680-700"
            )

            print(
                "  foot-orientation reversal / penetration "
                "around steps 708-728"
            )

            print()


            # =================================================
            # MAIN LOOP
            # =================================================

            while (
                viewer.is_running()
                and
                env.current_step
                <
                TRAIN_END_STEP
            ):


                # ---------------------------------------------
                # ACTION
                # ---------------------------------------------

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


                # ---------------------------------------------
                # KINEMATIC ENVIRONMENT STEP
                # ---------------------------------------------

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


                # ---------------------------------------------
                # CAMERA
                # ---------------------------------------------

                update_camera(
                    viewer,
                    env,
                    args,
                    initialize=False,
                )

                viewer.sync()


                # ---------------------------------------------
                # TERMINAL DIAGNOSTICS
                # ---------------------------------------------

                if should_print(
                    step,
                    args.print_every,
                ):

                    geometry = geometry_summary(
                        info
                    )

                    marker = ""

                    if step in IMPORTANT_STEPS:

                        marker += (
                            "  *** IMPORTANT ***"
                        )

                    if in_critical_region(
                        step
                    ):

                        marker += (
                            "  *** CRITICAL REGION ***"
                        )


                    print(

                        f"step={step:03d} "

                        f"x={float(info['world_x']):+.3f} "

                        f"R={float(reward):+.3f} "

                        f"pen="
                        f"{geometry['max_actual_penetration_mm']:6.1f}mm "

                        f"phaseEx="
                        f"{geometry['max_phase_penetration_excess_mm']:6.1f}mm "

                        f"penCost="
                        f"{geometry['penetration_cost']:.3f} "

                        f"Lgap="
                        f"{geometry['left_gap_mm']:+7.1f}mm "

                        f"Lref="
                        f"{geometry['flat_left_gap_mm']:+6.1f}mm "

                        f"Lextra="
                        f"{geometry['left_extra_mm']:+7.1f}mm "

                        f"Rgap="
                        f"{geometry['right_gap_mm']:+7.1f}mm "

                        f"maxRes="
                        f"{float(info['max_residual_rad']):.4f}rad"

                        f"{marker}"
                    )


                # ---------------------------------------------
                # OPTIONAL MANUAL PAUSE
                # ---------------------------------------------

                if step in pause_steps:

                    geometry = geometry_summary(
                        info
                    )

                    print()

                    print(
                        "=" * 110
                    )

                    print(
                        f"PAUSED AT STEP {step}"
                    )

                    print()

                    print(
                        "Max penetration:",
                        (
                            f"{geometry['max_actual_penetration_mm']:.2f} mm"
                        ),
                    )

                    print(
                        "Phase-aware excess penetration:",
                        (
                            f"{geometry['max_phase_penetration_excess_mm']:.2f} mm"
                        ),
                    )

                    print()

                    print(
                        "Left heel-to-toe gap:",
                        (
                            f"{geometry['left_gap_mm']:+.2f} mm"
                        ),
                    )

                    print(
                        "Same-phase flat left gap:",
                        (
                            f"{geometry['flat_left_gap_mm']:+.2f} mm"
                        ),
                    )

                    print(
                        "Left extra vs flat:",
                        (
                            f"{geometry['left_extra_mm']:+.2f} mm"
                        ),
                    )

                    print()

                    print(
                        "Inspect BOTH heel and toe positions "
                        "on BOTH feet in the MuJoCo viewer."
                    )

                    print(
                        "Press ENTER in this PowerShell window "
                        "to continue."
                    )

                    print(
                        "=" * 110
                    )

                    input()


                # ---------------------------------------------
                # VISUAL SLOWDOWN
                # ---------------------------------------------

                sleep_time = float(
                    args.sleep
                )

                if in_critical_region(
                    step
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


        # ====================================================
        # FINISHED
        # ====================================================

        print()

        print(
            "=" * 110
        )

        print(
            "VISUALIZATION COMPLETE"
        )

        print(
            "=" * 110
        )

        if args.mode == "bc":

            print(
                "This was the frozen BC-only "
                "zero-residual baseline in the "
                "Reward-V2 environment."
            )

        else:

            print(
                "This was the trained Reward-V2 "
                "8192-step PPO smoke checkpoint."
            )

        print(
            "No training was performed by this viewer."
        )

        print(
            "=" * 110
        )


    finally:

        env.close()


if __name__ == "__main__":
    main()