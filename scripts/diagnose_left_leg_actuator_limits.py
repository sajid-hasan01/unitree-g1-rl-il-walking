import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_dynamic_walking_env import G1DynamicWalkingEnv


DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "g1_openhe_walk3_subject4_1320_1620_rawlegs_15dof.npz"
)

JOINTS_TO_CHECK = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
]


def load_initializer():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "test_zero_residual_full_reference_rsi.py"
    )

    spec = importlib.util.spec_from_file_location(
        "rsi_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.initialize_from_reference


def find_actuators_for_joint(model, joint_id):
    actuator_ids = []

    for actuator_id in range(model.nu):
        transmission_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if transmission_id == joint_id:
            actuator_ids.append(
                actuator_id
            )

    return actuator_ids


def print_model_limits(
    env,
    joint_info,
):
    print("=" * 120)
    print(
        "LOCAL COMPILED G1 ACTUATOR / JOINT LIMITS"
    )
    print("=" * 120)

    for name in JOINTS_TO_CHECK:
        info = joint_info[name]

        joint_id = info["joint_id"]
        actuator_id = info["actuator_id"]

        joint_force_limited = bool(
            env.model.jnt_actfrclimited[
                joint_id
            ]
        )

        joint_force_range = (
            env.model.jnt_actfrcrange[
                joint_id
            ].copy()
        )

        actuator_force_limited = bool(
            env.model.actuator_forcelimited[
                actuator_id
            ]
        )

        actuator_force_range = (
            env.model.actuator_forcerange[
                actuator_id
            ].copy()
        )

        ctrl_limited = bool(
            env.model.actuator_ctrllimited[
                actuator_id
            ]
        )

        ctrl_range = (
            env.model.actuator_ctrlrange[
                actuator_id
            ].copy()
        )

        actuator_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        if actuator_name is None:
            actuator_name = (
                f"<unnamed:{actuator_id}>"
            )

        print()
        print(
            f"Joint: {name}"
        )
        print(
            f"  joint_id            = {joint_id}"
        )
        print(
            f"  actuator_id         = {actuator_id}"
        )
        print(
            f"  actuator_name       = {actuator_name}"
        )
        print(
            f"  joint force limited = {joint_force_limited}"
        )
        print(
            "  joint force range   = "
            f"[{joint_force_range[0]:+.3f}, "
            f"{joint_force_range[1]:+.3f}]"
        )
        print(
            f"  actuator limited    = {actuator_force_limited}"
        )
        print(
            "  actuator range      = "
            f"[{actuator_force_range[0]:+.3f}, "
            f"{actuator_force_range[1]:+.3f}]"
        )
        print(
            f"  control limited     = {ctrl_limited}"
        )
        print(
            "  control range       = "
            f"[{ctrl_range[0]:+.3f}, "
            f"{ctrl_range[1]:+.3f}]"
        )
        print(
            "  gainprm[0]          = "
            f"{float(env.model.actuator_gainprm[actuator_id, 0]):.3f}"
        )

    print("=" * 120)


def get_force_limit(
    env,
    joint_id,
    actuator_id,
):
    if bool(
        env.model.jnt_actfrclimited[
            joint_id
        ]
    ):
        low, high = (
            env.model.jnt_actfrcrange[
                joint_id
            ]
        )

        return max(
            abs(float(low)),
            abs(float(high)),
        )

    if bool(
        env.model.actuator_forcelimited[
            actuator_id
        ]
    ):
        low, high = (
            env.model.actuator_forcerange[
                actuator_id
            ]
        )

        return max(
            abs(float(low)),
            abs(float(high)),
        )

    return None


def main():
    initialize_from_reference = (
        load_initializer()
    )

    env = G1DynamicWalkingEnv(
        dataset_path=str(DATASET),
        reference_mode="cyclic",
        target_forward_velocity=0.10,
        action_scale=0.055,
        action_target_smoothing=0.25,
        frame_skip=5,
        height_offset=-0.01348,
        reference_speed=1.0,
        initial_stand_steps=70,
        transition_steps=220,
        random_start=False,
        enable_push=False,
        include_contact_phase_observation=True,
        initial_yaw_degrees=0.0,
    )

    try:
        env.reset()

        initialize_from_reference(
            env,
            28,
        )

        controlled_names = [
            str(name)
            for name in env.controlled_joint_names
        ]

        controlled_indices = {
            name: index
            for index, name
            in enumerate(controlled_names)
        }

        joint_info = {}

        for name in JOINTS_TO_CHECK:
            joint_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Joint not found: {name}"
                )

            actuator_ids = (
                find_actuators_for_joint(
                    env.model,
                    joint_id,
                )
            )

            if len(actuator_ids) != 1:
                raise RuntimeError(
                    f"{name}: expected exactly "
                    f"1 actuator, found "
                    f"{actuator_ids}"
                )

            actuator_id = (
                actuator_ids[0]
            )

            joint_info[name] = {
                "joint_id": joint_id,
                "actuator_id": actuator_id,
                "qpos_adr": int(
                    env.model.jnt_qposadr[
                        joint_id
                    ]
                ),
                "dof_adr": int(
                    env.model.jnt_dofadr[
                        joint_id
                    ]
                ),
                "controlled_index":
                    controlled_indices[name],
            }

        print_model_limits(
            env,
            joint_info,
        )

        zero_action = np.zeros(
            env.num_actions,
            dtype=np.float32,
        )

        previous_targets = None

        control_dt = (
            float(env.model.opt.timestep)
            * int(env.frame_skip)
        )

        print()
        print("=" * 150)
        print(
            "DYNAMIC LEFT-LEG TRACKING / FORCE TEST"
        )
        print("=" * 150)

        print(
            "err = actual - target"
        )
        print(
            "limit% = abs(joint actuator force) "
            "/ compiled joint/actuator force limit"
        )

        for step in range(30):

            control_frame = float(
                env.motion_frame
            )

            reference = np.asarray(
                env._get_reference_joint_positions_for_step(),
                dtype=np.float64,
            ).copy()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                zero_action
            )

            target = np.asarray(
                env.last_targets,
                dtype=np.float64,
            ).copy()

            left_contact, right_contact = (
                env._get_foot_contacts()
            )

            if step >= 12:
                print()
                print("-" * 150)

                print(
                    f"step={step:02d} "
                    f"control_frame={control_frame:5.2f} "
                    f"post_frame={float(env.motion_frame):5.2f} "
                    f"L={left_contact} "
                    f"R={right_contact} "
                    f"height={float(info['base_height']):.4f} "
                    f"up_z={float(info['up_z']):.4f}"
                )

                print(
                    "JOINT                     "
                    "REF       TARGET     ACTUAL     "
                    "ERR        TARGETvel  ACTUALvel  "
                    "QFRC       LIMIT    LIMIT%"
                )

                for name in JOINTS_TO_CHECK:
                    data = joint_info[name]

                    joint_id = (
                        data["joint_id"]
                    )

                    actuator_id = (
                        data["actuator_id"]
                    )

                    qpos_adr = (
                        data["qpos_adr"]
                    )

                    dof_adr = (
                        data["dof_adr"]
                    )

                    index = (
                        data[
                            "controlled_index"
                        ]
                    )

                    ref_value = float(
                        reference[index]
                    )

                    target_value = float(
                        target[index]
                    )

                    actual_value = float(
                        env.data.qpos[
                            qpos_adr
                        ]
                    )

                    actual_velocity = float(
                        env.data.qvel[
                            dof_adr
                        ]
                    )

                    if (
                        previous_targets
                        is not None
                    ):
                        target_velocity = (
                            target_value
                            - float(
                                previous_targets[
                                    index
                                ]
                            )
                        ) / control_dt
                    else:
                        target_velocity = (
                            float("nan")
                        )

                    joint_force = float(
                        env.data.qfrc_actuator[
                            dof_adr
                        ]
                    )

                    actuator_force = float(
                        env.data.actuator_force[
                            actuator_id
                        ]
                    )

                    force_limit = (
                        get_force_limit(
                            env,
                            joint_id,
                            actuator_id,
                        )
                    )

                    if (
                        force_limit is None
                        or force_limit <= 0
                    ):
                        limit_text = (
                            "   n/a"
                        )

                        ratio_text = (
                            "   n/a"
                        )

                    else:
                        limit_text = (
                            f"{force_limit:7.1f}"
                        )

                        ratio = (
                            100.0
                            * abs(joint_force)
                            / force_limit
                        )

                        ratio_text = (
                            f"{ratio:6.1f}%"
                        )

                    print(
                        f"{name:27s} "
                        f"{ref_value:+8.3f} "
                        f"{target_value:+8.3f} "
                        f"{actual_value:+8.3f} "
                        f"{actual_value-target_value:+9.3f} "
                        f"{target_velocity:+10.3f} "
                        f"{actual_velocity:+10.3f} "
                        f"{joint_force:+9.1f} "
                        f"{limit_text} "
                        f"{ratio_text}"
                    )

                    print(
                        f"{'':27s} "
                        f"actuator_force="
                        f"{actuator_force:+.2f}"
                    )

            previous_targets = (
                target.copy()
            )

            if terminated or truncated:
                break

        print()
        print("=" * 150)
        print(
            "DIAGNOSTIC COMPLETE"
        )
        print("=" * 150)

    finally:
        env.close()


if __name__ == "__main__":
    main()