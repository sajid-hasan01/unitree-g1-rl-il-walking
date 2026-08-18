from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer


ROOT = Path(r"C:\Projects\unitree-g1-rl-il-walking")

MODEL = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

REF = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
    / "medium_02_50hz_grounded.npz"
)


print("=" * 80)
print("LATEST COMPLETED G1 REFERENCE WALK")
print("=" * 80)
print("Model:", MODEL)
print("Reference:", REF)


data_npz = np.load(
    REF,
    allow_pickle=True,
)

print("Keys:", data_npz.files)


if "full_qpos" not in data_npz.files:
    raise RuntimeError(
        "Reference does not contain full_qpos."
    )


qpos = np.asarray(
    data_npz["full_qpos"],
    dtype=np.float64,
)

qvel = (
    np.asarray(
        data_npz["full_qvel"],
        dtype=np.float64,
    )
    if "full_qvel" in data_npz.files
    else None
)


fps = 50.0

if "fps" in data_npz.files:
    fps = float(
        np.asarray(
            data_npz["fps"]
        ).reshape(-1)[0]
    )

elif "reference_fps" in data_npz.files:
    fps = float(
        np.asarray(
            data_npz["reference_fps"]
        ).reshape(-1)[0]
    )


print()
print("Frames:", len(qpos))
print("qpos shape:", qpos.shape)

if qvel is not None:
    print("qvel shape:", qvel.shape)

print("FPS:", fps)
print(
    "Duration:",
    f"{len(qpos) / fps:.2f} s",
)


model = mujoco.MjModel.from_xml_path(
    str(MODEL)
)

data = mujoco.MjData(
    model
)


if qpos.shape[1] != model.nq:
    raise RuntimeError(
        f"Reference nq={qpos.shape[1]}, "
        f"model nq={model.nq}"
    )


print()
print(
    f"MuJoCo model: "
    f"nq={model.nq}, "
    f"nv={model.nv}, "
    f"nu={model.nu}"
)

print()
print("Starting viewer...")
print("Close the MuJoCo window to stop.")


frame_dt = 1.0 / fps


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # Better viewing distance for G1.
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15

    frame = 0

    while viewer.is_running():

        start = time.perf_counter()


        data.qpos[:] = qpos[frame]


        if (
            qvel is not None
            and
            qvel.shape[1] == model.nv
        ):
            data.qvel[:] = qvel[frame]

        else:
            data.qvel[:] = 0.0


        mujoco.mj_forward(
            model,
            data,
        )


        # Follow the robot root.
        viewer.cam.lookat[:] = data.qpos[:3]


        viewer.sync()


        frame += 1

        if frame >= len(qpos):
            frame = 0


        elapsed = (
            time.perf_counter()
            -
            start
        )

        sleep_time = (
            frame_dt
            -
            elapsed
        )

        if sleep_time > 0:
            time.sleep(
                sleep_time
            )
