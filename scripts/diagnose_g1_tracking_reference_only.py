import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from envs.g1_closed_loop_tracking_env import (
    G1ClosedLoopTrackingEnv,
)


START_FRAMES = [
    0,
    42,
    83,
    125,
    167,
]


def event(value):
    return "-" if value is None else str(value)


def rollout(start_frame):

    env = G1ClosedLoopTrackingEnv(
        fixed_start_frame=start_frame,
        random_reference_start=False,

        reset_position_noise=0.0,
        reset_velocity_noise=0.0,

        control_hz=50.0,
        target_velocity=-0.18,
        target_smoothing=0.20,

        max_episode_steps=400,
    )

    obs, info = env.reset(seed=1234)

    initial_x = float(env.data.qpos[0])

    steps = 0

    max_y = 0.0
    max_yaw = 0.0

    min_up = 1.0
    min_z = float("inf")

    qerr_sum = 0.0
    max_qerr = 0.0
    max_qderr = 0.0

    vx_sum = 0.0

    negative_vx_frames = 0
    positive_vx_frames = 0

    first_wrong_direction = None

    first_up95 = None
    first_up90 = None
    first_up80 = None

    left_switches = 0
    right_switches = 0

    previous_left = None
    previous_right = None

    final_info = {}

    termination_reason = "unknown"

    while True:

        # Zero RL correction:
        # physical target = raw 50-Hz reference pose.
        action = np.zeros(
            15,
            dtype=np.float32,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(action)

        steps += 1

        x = float(step_info["x"])
        y = float(step_info["y"])
        z = float(step_info["z"])

        vx = float(step_info["vx"])

        yaw = float(
            step_info["yaw_deg"]
        )

        up = float(
            step_info["up_z"]
        )

        qerr = float(
            step_info["q_error_rms"]
        )

        qderr = float(
            step_info["qd_error_rms"]
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

        min_z = min(
            min_z,
            z,
        )

        qerr_sum += qerr ** 2

        max_qerr = max(
            max_qerr,
            qerr,
        )

        max_qderr = max(
            max_qderr,
            qderr,
        )

        vx_sum += vx

        if vx < 0.0:
            negative_vx_frames += 1
        else:
            positive_vx_frames += 1

        if (
            first_wrong_direction is None
            and vx > 0.05
        ):
            first_wrong_direction = steps

        if (
            first_up95 is None
            and up < 0.95
        ):
            first_up95 = steps

        if (
            first_up90 is None
            and up < 0.90
        ):
            first_up90 = steps

        if (
            first_up80 is None
            and up < 0.80
        ):
            first_up80 = steps

        left = bool(
            step_info["left_contact"]
        )

        right = bool(
            step_info["right_contact"]
        )

        if previous_left is not None:
            if left != previous_left:
                left_switches += 1

        if previous_right is not None:
            if right != previous_right:
                right_switches += 1

        previous_left = left
        previous_right = right

        final_info = step_info

        if (
            steps == 1
            or steps % 25 == 0
            or terminated
            or truncated
        ):
            print(
                f"start={start_frame:03d} "
                f"step={steps:03d} "
                f"ref={step_info['reference_index']:03d} "
                f"x={x:+.3f} "
                f"vx={vx:+.3f} "
                f"y={y:+.3f} "
                f"yaw={yaw:+.1f} "
                f"up={up:.3f} "
                f"z={z:.3f} "
                f"qerr={qerr:.3f} "
                f"qdErr={qderr:.3f} "
                f"L/R={int(left)}/{int(right)}",
                flush=True,
            )

        if terminated:

            if z < 0.45:
                termination_reason = "height"

            elif up < 0.45:
                termination_reason = "upright"

            elif abs(y) > 0.60:
                termination_reason = "lateral"

            elif abs(yaw) > 135.0:
                termination_reason = "yaw"

            else:
                termination_reason = "terminated"

            break

        if truncated:

            if bool(
                step_info.get(
                    "completed",
                    False,
                )
            ):
                termination_reason = (
                    "reference_complete"
                )

            else:
                termination_reason = (
                    "time_limit"
                )

            break


    rms_qerr = float(
        np.sqrt(
            qerr_sum
            / max(
                steps,
                1,
            )
        )
    )

    mean_vx = float(
        vx_sum
        / max(
            steps,
            1,
        )
    )

    completed = bool(
        final_info.get(
            "completed",
            False,
        )
    )

    final_x = float(
        final_info["x"]
    )

    result = {
        "start":
            start_frame,

        "steps":
            steps,

        "completed":
            completed,

        "reason":
            termination_reason,

        "x":
            final_x,

        "dx":
            final_x - initial_x,

        "mean_vx":
            mean_vx,

        "max_y":
            max_y,

        "max_yaw":
            max_yaw,

        "min_up":
            min_up,

        "min_z":
            min_z,

        "rms_qerr":
            rms_qerr,

        "max_qerr":
            max_qerr,

        "max_qderr":
            max_qderr,

        "neg_vx":
            negative_vx_frames,

        "pos_vx":
            positive_vx_frames,

        "wrong_dir":
            first_wrong_direction,

        "up95":
            first_up95,

        "up90":
            first_up90,

        "up80":
            first_up80,

        "lsw":
            left_switches,

        "rsw":
            right_switches,
    }

    env.close()

    return result


print("=" * 145)
print(
    "CLOSED-LOOP 50-HZ RAW REFERENCE "
    "DYNAMIC BASELINE"
)
print("PPO ACTION = ZERO")
print("NO TRAINING")
print("=" * 145)


results = []


for start in START_FRAMES:

    print()
    print("-" * 145)
    print(
        f"START FRAME {start}"
    )
    print("-" * 145)

    result = rollout(start)

    results.append(result)


print()
print("=" * 180)
print(
    "FINAL ZERO-ACTION REFERENCE RESULTS"
)
print("=" * 180)

print(
    f"{'START':>5s} "
    f"{'STEP':>5s} "
    f"{'DONE':>5s} "
    f"{'REASON':>18s} "
    f"{'DX':>8s} "
    f"{'VX':>8s} "
    f"{'MAXY':>7s} "
    f"{'YAW':>7s} "
    f"{'MINUP':>7s} "
    f"{'MINZ':>7s} "
    f"{'QERR':>7s} "
    f"{'QMAX':>7s} "
    f"{'QDERR':>7s} "
    f"{'UP95':>5s} "
    f"{'UP90':>5s} "
    f"{'UP80':>5s} "
    f"{'WRONG':>6s} "
    f"{'L/R':>7s}"
)

print("-" * 180)


for r in results:

    contacts = (
        f"{r['lsw']}/"
        f"{r['rsw']}"
    )

    print(
        f"{r['start']:5d} "
        f"{r['steps']:5d} "
        f"{str(r['completed']):>5s} "
        f"{r['reason']:>18s} "
        f"{r['dx']:+8.3f} "
        f"{r['mean_vx']:+8.3f} "
        f"{r['max_y']:7.3f} "
        f"{r['max_yaw']:7.1f} "
        f"{r['min_up']:7.3f} "
        f"{r['min_z']:7.3f} "
        f"{r['rms_qerr']:7.3f} "
        f"{r['max_qerr']:7.3f} "
        f"{r['max_qderr']:7.3f} "
        f"{event(r['up95']):>5s} "
        f"{event(r['up90']):>5s} "
        f"{event(r['up80']):>5s} "
        f"{event(r['wrong_dir']):>6s} "
        f"{contacts:>7s}"
    )


completed_count = sum(
    int(r["completed"])
    for r in results
)

mean_qerr = float(
    np.mean(
        [
            r["rms_qerr"]
            for r in results
        ]
    )
)

negative_progress = sum(
    int(
        r["dx"] < 0.0
    )
    for r in results
)

mean_vx_all = float(
    np.mean(
        [
            r["mean_vx"]
            for r in results
        ]
    )
)


print()
print("=" * 145)
print("DIAGNOSIS")
print("=" * 145)

print(
    "Reference starts completed:",
    f"{completed_count}/{len(results)}",
)

print(
    "Mean joint tracking RMS error:",
    f"{mean_qerr:.3f} rad",
)

print(
    "Starts with net -X progress:",
    f"{negative_progress}/{len(results)}",
)

print(
    "Mean VX across starts:",
    f"{mean_vx_all:+.3f} m/s",
)


print()
print("INTERPRETATION")
print("-" * 145)

if (
    completed_count >= 4
    and mean_qerr < 0.20
):

    print(
        "STRONG RESULT: the 50-Hz reference "
        "trajectory is dynamically trackable."
    )

    print(
        "Next step: begin a short closed-loop "
        "PPO tracking pilot."
    )

elif (
    mean_qerr < 0.20
):

    print(
        "Pose tracking remains accurate, but the "
        "robot does not dynamically survive enough "
        "reference phases."
    )

    print(
        "This points to root/contact/support-transfer "
        "dynamics rather than joint-pose tracking."
    )

else:

    print(
        "Joint tracking error becomes too large."
    )

    print(
        "Do not train PPO yet; first correct "
        "reference/actuator tracking."
    )


if negative_progress < 3:

    print()
    print(
        "WARNING: most starts do not produce net -X "
        "motion even while tracking reference joints."
    )

    print(
        "This would indicate a retargeting/root-motion "
        "or gait-direction mismatch."
    )


print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 145)
