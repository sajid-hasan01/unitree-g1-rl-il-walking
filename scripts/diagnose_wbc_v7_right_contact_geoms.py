from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_knee_bias_sweep import G1WBCV7PreloadKneeBiasEnv


def geom_name(model, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name if name is not None else f"geom_{geom_id}"


def main() -> None:
    env = G1WBCV7PreloadKneeBiasEnv(
        frame_skip=5,
        max_steps=520,
        cycle_duration=5.2,
        shift_start=0.08,
        swing_start=0.46,
        swing_end=0.62,
        target_clearance=0.0,
        target_lateral_shift=0.0,
        ik_gain=0.0,
        z_lift_weight=0.0,
        xy_hold_weight=0.0,
        support_lock_weight=0.0,
        support_xy_weight=0.0,
        support_z_weight=0.0,
        support_ik_gain=0.0,
        tiny_lift_height=0.010,
        lift_start=0.46,
        lift_peak=0.62,
        land_end=0.78,
        foot_ik_gain=0.55,
        foot_ik_damping=0.080,
        foot_ik_max_delta=0.060,
        preload_sign=-1.0,
        preload_amp=0.015,
        preload_start=0.08,
        preload_end=0.42,
        knee_bias=0.010,
        hip_ratio=0.00,
        ankle_ratio=0.00,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    best = None

    for step in range(520):
        obs, reward, terminated, truncated, info = env.step(action)

        rclear = float(info["right_foot_clearance"])

        contacts = []
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            n1 = geom_name(env.model, g1)
            n2 = geom_name(env.model, g2)

            pair = (n1, n2)
            text = f"{n1} <-> {n2}"

            right_related = ("right" in text.lower()) or ("r_" in text.lower())
            floor_related = ("floor" in text.lower()) or ("plane" in text.lower()) or ("ground" in text.lower())

            if right_related and floor_related:
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(env.model, env.data, i, force)
                contacts.append({
                    "pair": text,
                    "dist": float(c.dist),
                    "normal_force": float(force[0]),
                })

        if best is None or rclear > best["right_clearance"]:
            best = {
                "step": step,
                "phase": float(info["phase"]),
                "right_clearance": rclear,
                "left_clearance": float(info["left_foot_clearance"]),
                "right_contact": bool(info["right_contact"]),
                "left_contact": bool(info["left_contact"]),
                "up_z": float(info["up_z"]),
                "x": float(info["x_position"]),
                "y": float(info["y_position"]),
                "xv": float(info["x_velocity"]),
                "yv": float(info["y_velocity"]),
                "ang": float(info["root_ang_vel"]),
                "contacts": contacts,
            }

        if step % 25 == 0:
            print(
                f"step={step:04d} "
                f"phi={info['phase']:.3f} "
                f"Rclear={info['right_foot_clearance']:.4f} "
                f"Lclear={info['left_foot_clearance']:.4f} "
                f"Rcontact={int(info['right_contact'])} "
                f"Lcontact={int(info['left_contact'])} "
                f"up={info['up_z']:.3f} "
                f"x={info['x_position']:+.3f} "
                f"y={info['y_position']:+.3f}"
            )

        if terminated or truncated:
            print("Ended:", env.termination_reason(info))
            break

    print()
    print("=" * 100)
    print("BEST RIGHT CLEARANCE FRAME")
    print("=" * 100)

    for k, v in best.items():
        if k != "contacts":
            print(f"{k}: {v}")

    print()
    print("RIGHT-FOOT FLOOR CONTACTS AT BEST FRAME:")
    if not best["contacts"]:
        print("No right-foot floor contact detected by geom-name filter.")
    else:
        for c in best["contacts"]:
            print(
                f"{c['pair']} | dist={c['dist']:+.6f} | normal_force={c['normal_force']:+.6f}"
            )

    env.close()


if __name__ == "__main__":
    main()
