import os
import mujoco


MODEL_PATH = os.path.join(
    "third_party",
    "mujoco_menagerie",
    "unitree_g1",
    "scene.xml",
)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    print("Unitree G1 MuJoCo model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("nkey:", model.nkey)

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )
        print(i, name)


if __name__ == "__main__":
    main()