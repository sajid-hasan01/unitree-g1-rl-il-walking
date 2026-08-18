from pathlib import Path
import argparse
import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_29dof_tracking_restart_v1 import (
    G129DofTrackingRestartV1,
)


# ================================================================
# ARGUMENTS
# ================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=[
        "reference",
        "physics",
    ],
    default="physics",
)

parser.add_argument(
    "--rate",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
)

parser.add_argument(
    "--loop",
    action="store_true",
)

parser.add_argument(
    "--slowmo",
    type=float,
    default=1.0,
)

args = parser.parse_args()


if args.rate <= 0:
    raise ValueError(
        "--rate must be > 0"
    )


if args.slowmo <= 0:
    raise ValueError(
        "--slowmo must be > 0"
    )


# ================================================================
# ENVIRONMENT
# ================================================================

env = G129DofTrackingRestartV1(
    rsi=False,
    reset_joint_noise=0.0,
    reset_velocity_noise=0.0,
)


start = int(
    np.clip(
        args.start,
        0,
        env.num_frames - 2,
    )
)


CONTROL_DT = (
    1.0
    / env.reference_fps
)


print("=" * 100)
print("UNITREE G1 VISUALIZER")
print("=" * 100)

print(
    "mode       :",
    args.mode,
)

print(
    "phase rate :",
    args.rate,
)

print(
    "start frame:",
    start,
)

print(
    "slow motion:",
    args.slowmo,
)

print(
    "reference  :",
    env.reference_path,
)

print()
print(
    "Close the MuJoCo window to stop."
)

print("=" * 100)


# ================================================================
# CAMERA
# ================================================================

def configure_camera(
    viewer,
):

    viewer.cam.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )

    viewer.cam.distance = 4.0

    viewer.cam.azimuth = 135.0

    viewer.cam.elevation = -18.0

    viewer.cam.lookat[:] = np.asarray(
        [
            float(
                env.data.qpos[0]
            ),
            float(
                env.data.qpos[1]
            ),
            0.75,
        ]
    )


def update_camera(
    viewer,
):

    # Follow robot primarily in X.
    viewer.cam.lookat[0] = float(
        env.data.qpos[0]
    )

    viewer.cam.lookat[1] = float(
        env.data.qpos[1]
    )

    viewer.cam.lookat[2] = 0.70


# ================================================================
# REFERENCE-ONLY VISUALIZATION
#
# No physics tracking error.
# Directly shows dataset pose.
# ================================================================

def run_reference(
    viewer,
):

    while viewer.is_running():

        frame = start


        while (
            frame
            < env.num_frames
            and viewer.is_running()
        ):

            wall_start = time.perf_counter()


            env.data.qpos[:] = (
                env.ref_full_qpos[
                    frame
                ]
            )

            env.data.qvel[:] = (
                env.ref_full_qvel[
                    frame
                ]
            )


            mujoco.mj_forward(
                env.model,
                env.data,
            )


            update_camera(
                viewer
            )

            viewer.sync()


            frame += 1


            target_dt = (
                CONTROL_DT
                / args.slowmo
            )


            elapsed = (
                time.perf_counter()
                - wall_start
            )


            if elapsed < target_dt:

                time.sleep(
                    target_dt
                    - elapsed
                )


        if not args.loop:
            break


# ================================================================
# PHYSICS VISUALIZATION
#
# Actual MuJoCo simulation.
#
# phase_rate:
#
# 1.00 = normal reference clock
# 0.75 = slower reference
# 1.25 = faster reference
# ================================================================

def run_physics(
    viewer,
):

    while viewer.is_running():

        env.reset(
            options={
                "start_frame":
                    start,
            }
        )


        phase = float(
            start
        )


        while viewer.is_running():

            wall_start = time.perf_counter()


            phase += (
                args.rate
            )


            phase = min(
                phase,
                float(
                    env.num_frames - 1
                ),
            )


            target_frame = int(
                math.floor(
                    phase
                    + 1e-9
                )
            )


            target_frame = int(
                np.clip(
                    target_frame,
                    env._current_frame,
                    env.num_frames - 1,
                )
            )


            # ZERO residual action.
            # This isolates reference timing.
            target = env._action_target(
                np.zeros(
                    29,
                    dtype=np.float64,
                ),
                target_frame,
            )


            env.last_target = (
                target.copy()
            )


            env._apply_target(
                target
            )


            # Actual MuJoCo physics.
            for _ in range(
                env.frame_skip
            ):

                mujoco.mj_step(
                    env.model,
                    env.data,
                )


            env._current_frame = (
                target_frame
            )


            update_camera(
                viewer
            )

            viewer.sync()


            failed, reason = (
                env._physical_failure()
            )


            print(
                "\r"
                f"frame={target_frame:03d} "
                f"x={env.data.qpos[0]:+.3f} "
                f"y={env.data.qpos[1]:+.3f} "
                f"z={env.data.qpos[2]:.3f} "
                f"up={env._up_z():.3f}",
                end="",
                flush=True,
            )


            if failed:

                print()
                print(
                    "FAIL:",
                    reason,
                    "at frame",
                    target_frame,
                )

                # Keep failed pose visible briefly.
                for _ in range(35):

                    if not viewer.is_running():
                        break

                    viewer.sync()

                    time.sleep(
                        0.02
                    )


                break


            if (
                target_frame
                >= env.num_frames - 1
            ):

                print()
                print(
                    "REFERENCE END REACHED"
                )

                break


            target_dt = (
                CONTROL_DT
                / args.slowmo
            )


            elapsed = (
                time.perf_counter()
                - wall_start
            )


            if elapsed < target_dt:

                time.sleep(
                    target_dt
                    - elapsed
                )


        if not args.loop:
            break


# ================================================================
# LAUNCH
# ================================================================

env.reset(
    options={
        "start_frame":
            start
    }
)


with mujoco.viewer.launch_passive(
    env.model,
    env.data,
) as viewer:

    configure_camera(
        viewer
    )


    if args.mode == "reference":

        run_reference(
            viewer
        )

    else:

        run_physics(
            viewer
        )


env.close()
