from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_wbc_taskspace_right_lift_env import G1WBCTaskspaceRightLiftEnv

env = G1WBCTaskspaceRightLiftEnv()
obs, info = env.reset()

print("=" * 100)
print("ENV INSPECTION: WBC RIGHT LIFT")
print("=" * 100)

print("\naction_space:", env.action_space)
print("obs_shape:", obs.shape)

print("\nactuator_ids:")
print(env.actuator_ids)

print("\nqpos_adrs:")
print(env.qpos_adrs)

print("\nqvel_adrs:")
print(env.qvel_adrs)

print("\nstand_joint_pos:")
for i, v in enumerate(env.stand_joint_pos):
    print(f"{i:02d}: {float(v): .6f}")

names = None
for attr in ["controlled_joint_names", "joint_names", "CONTROLLED_15_JOINTS"]:
    if hasattr(env, attr):
        names = getattr(env, attr)
        break

print("\ncontrolled joint names:")
if names is None:
    print("No joint-name list attribute found.")
else:
    for i, name in enumerate(names):
        print(f"{i:02d}: {name}")

print("\nctrl range by controlled joint:")
for i, aid in enumerate(env.actuator_ids):
    lo = float(env.ctrl_low[aid])
    hi = float(env.ctrl_high[aid])
    name = names[i] if names is not None and i < len(names) else "unknown"
    print(f"{i:02d}: aid={aid:02d} name={name} ctrl_low={lo: .4f} ctrl_high={hi: .4f}")

print("\nimportant assumed indices:")
for i in [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14]:
    name = names[i] if names is not None and i < len(names) else "unknown"
    print(f"target[{i:02d}] = {name}")

if hasattr(env, "close"):
    env.close()