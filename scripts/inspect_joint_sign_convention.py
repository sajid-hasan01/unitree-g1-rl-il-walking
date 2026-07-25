import os
import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_XML_PATH = os.path.join(
    "third_party",
    "mujoco_menagerie",
    "unitree_g1",
    "scene.xml",
)

CONTROLLED_15_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

EXPECTED_POSITIVE_DIRECTION = {
    "left_hip_pitch_joint": "thigh swings BACKWARD (extension)",
    "left_hip_roll_joint": "thigh moves OUTWARD, away from body (abduction)",
    "left_hip_yaw_joint": "thigh rotates, toes turn INWARD",
    "left_knee_joint": "knee BENDS (flexion)",
    "left_ankle_pitch_joint": "toes point DOWN (plantarflexion)",
    "left_ankle_roll_joint": "sole tilts OUTWARD (eversion)",
    "right_hip_pitch_joint": "thigh swings BACKWARD (extension)",
    "right_hip_roll_joint": "thigh moves OUTWARD, away from body (abduction)",
    "right_hip_yaw_joint": "thigh rotates, toes turn INWARD",
    "right_knee_joint": "knee BENDS (flexion)",
    "right_ankle_pitch_joint": "toes point DOWN (plantarflexion)",
    "right_ankle_roll_joint": "sole tilts OUTWARD (eversion)",
    "waist_yaw_joint": "torso rotates LEFT (viewed from above)",
    "waist_roll_joint": "torso tilts to the RIGHT",
    "waist_pitch_joint": "torso leans FORWARD",
}


def get_joint_qpos_addresses(model, joint_names):
    qpos_addresses = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
    return qpos_addresses


def main():
    if not os.path.exists(MODEL_XML_PATH):
        raise FileNotFoundError(f"MuJoCo model not found: {MODEL_XML_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    qpos_addresses = get_joint_qpos_addresses(model, CONTROLLED_15_JOINTS)

    amplitude = 0.5
    period_sec = 3.0
    hold_sec = 2.0

    print("Joint sign/direction sweep test")
    print("Each joint will oscillate for a few seconds, one at a time.")
    print("Watch the printed 'expected' direction and compare to what you see.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        joint_idx = 0
        t_start = time.time()
        current_joint_start = t_start

        while viewer.is_running():
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                data.qpos[:] = 0.0
                data.qpos[3] = 1.0
            data.qvel[:] = 0.0

            now = time.time()
            elapsed_this_joint = now - current_joint_start

            if elapsed_this_joint > period_sec:
                joint_idx = (joint_idx + 1) % len(CONTROLLED_15_JOINTS)
                current_joint_start = now
                elapsed_this_joint = 0.0
                joint_name = CONTROLLED_15_JOINTS[joint_idx]
                print(f"--- Now sweeping: {joint_name} ---")
                print(f"    Expected at POSITIVE value: {EXPECTED_POSITIVE_DIRECTION[joint_name]}")

            sweep_value = amplitude * np.sin(2.0 * np.pi * elapsed_this_joint / period_sec)

            for i, qpos_addr in enumerate(qpos_addresses):
                data.qpos[qpos_addr] = sweep_value if i == joint_idx else 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
