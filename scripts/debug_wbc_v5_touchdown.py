from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_wbc_taskspace_right_lift_env import G1WBCTaskspaceRightLiftEnv

env = G1WBCTaskspaceRightLiftEnv(
    cycle_duration=5.8,
    swing_start=0.42,
    swing_end=0.70,
    land_end=0.88,
    target_clearance=0.018,
    target_lateral_shift=0.012,
    ik_gain=0.70,
    ik_damping=0.065,
    ik_max_delta=0.085,
    xy_hold_weight=0.18,
    z_lift_weight=0.75,
    support_lock_weight=0.62,
    support_xy_weight=0.18,
    support_z_weight=1.05,
    support_ik_gain=0.45,
    support_ik_damping=0.075,
    support_ik_max_delta=0.075,
    torso_pitch_gain=0.18,
    torso_roll_gain=0.08,
    angvel_pitch_gain=0.105,
    angvel_roll_gain=0.045,
    height_gain=0.16,
    height_target=0.790,
)

obs, info = env.reset()
action = np.zeros(4, dtype=np.float32)

prev_state = None

print("=" * 130)
print("WBC V5 TOUCHDOWN DEBUG")
print("=" * 130)
print("Columns:")
print("step state phi swing x xv up L R Rclr force | ctrl_rhip ctrl_rknee ctrl_rankle | q_rhip q_rknee q_rankle")
print("=" * 130)

for step in range(430):
    obs, reward, terminated, truncated, info = env.step(action)

    state = info.get("wbc_state", "NA")
    force = float(info.get("wbc_touchdown_force", -1.0))

    q = env.data.qpos[env.qpos_adrs]
    ctrl = env.data.ctrl

    ctrl_rhip = float(ctrl[env.actuator_ids[6]])
    ctrl_rknee = float(ctrl[env.actuator_ids[9]])
    ctrl_rankle = float(ctrl[env.actuator_ids[10]])

    q_rhip = float(q[6])
    q_rknee = float(q[9])
    q_rankle = float(q[10])

    state_changed = state != prev_state
    important_state = state in ("TOUCHDOWN", "RECOVERY", "SETTLE")

    if state_changed or important_state or step % 25 == 0 or terminated or truncated:
        marker = "STATE_CHANGE" if state_changed else ""
        print(
            f"{step:04d} {state:10s} "
            f"phi={float(info['phase']):.3f} swing={float(info['swing_env']):.3f} "
            f"x={float(info['x_position']):+.3f} xv={float(info['x_velocity']):+.3f} "
            f"up={float(info['up_z']):.3f} "
            f"L={int(bool(info['left_contact']))} R={int(bool(info['right_contact']))} "
            f"Rclr={float(info['right_foot_clearance']):.4f} "
            f"force={force:.2f} | "
            f"ctrl=({ctrl_rhip:+.3f},{ctrl_rknee:+.3f},{ctrl_rankle:+.3f}) | "
            f"q=({q_rhip:+.3f},{q_rknee:+.3f},{q_rankle:+.3f}) "
            f"{marker}"
        )

    prev_state = state

    if terminated or truncated:
        print("=" * 130)
        print("ENDED:", "terminated=", terminated, "truncated=", truncated, "reason=", info.get("termination_reason"))
        print("final_info:", info)
        break

if hasattr(env, "close"):
    env.close()