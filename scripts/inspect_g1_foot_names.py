from pathlib import Path
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = (
    PROJECT_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_XML))

print("Model:", MODEL_XML)
print()
print("Bodies containing foot / ankle / sole:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and any(k in name.lower() for k in ["foot", "ankle", "sole"]):
        print(i, name)

print()
print("Geoms containing foot / ankle / sole:")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and any(k in name.lower() for k in ["foot", "ankle", "sole"]):
        print(i, name)

print()
print("Sites containing foot / ankle / sole:")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if name and any(k in name.lower() for k in ["foot", "ankle", "sole"]):
        print(i, name)