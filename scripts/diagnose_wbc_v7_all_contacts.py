from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_knee_bias_sweep import G1WBCV7PreloadKneeBiasEnv


def name_or_id(model, objtype, obj_id: int, prefix: str) -> str:
    name = mujoco.mj_id2name(model, objtype, int(obj_id))
    return name if name is not None else f"{prefix}_{obj_id}"


def geom_name(model, geom_id: int) -> str:
    return name_or_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom")


def body_name(model, body_id: int) -> str:
    return name_or_id(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")


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

    print("=" * 110)
    print("MODEL FOOT/SITE INFO")
    print("=" * 110)

    right_site_body = int(env.model.site_bodyid[env.right_foot_site])
    left_site_body = int(env.model.site_bodyid[env.left_foot_site])

    print("right_foot_site id:", env.right_foot_site)
    print("right_foot_site body:", right_site_body, body_name(env.model, right_site_body))
    print("left_foot_site id:", env.left_foot_site)
    print("left_foot_site body:", left_site_body, body_name(env.model, left_site_body))

    print()
    print("GEOMS ATTACHED TO RIGHT/LEFT FOOT SITE BODIES")
    for gid in range(env.model.ngeom):
        bid = int(env.model.geom_bodyid[gid])
        if bid in [right_site_body, left_site_body]:
            print(
                f"geom_id={gid:03d} "
                f"geom={geom_name(env.model, gid):<35} "
                f"body={body_name(env.model, bid)}"
            )

    best = None
    frames_to_dump = []

    for step in range(520):
        obs, reward, terminated, truncated, info = env.step(action)

        phi = float(info["phase"])
        rclear = float(info["right_foot_clearance"])

        # Focus on useful swing/lift window only.
        in_lift_window = 0.46 <= phi <= 0.78

        if in_lift_window:
            if best is None or rclear > best["right_clearance"]:
                best = {
                    "step": step,
                    "phase": phi,
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
                    "right_site_pos": env.data.site_xpos[env.right_foot_site].copy(),
                    "left_site_pos": env.data.site_xpos[env.left_foot_site].copy(),
                }

        if abs(phi - 0.50) < 0.003 or abs(phi - 0.58) < 0.003 or abs(phi - 0.62) < 0.003 or abs(phi - 0.70) < 0.003:
            frames_to_dump.append((step, phi))

        if terminated or truncated:
            print("Ended early:", env.termination_reason(info))
            break

    if best is None:
        print("No lift-window frame found.")
        env.close()
        return

    print()
    print("=" * 110)
    print("BEST LIFT-WINDOW FRAME")
    print("=" * 110)
    for k, v in best.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.tolist()}")
        else:
            print(f"{k}: {v}")

    # Replay to the best step so contact data is exact at that frame.
    obs, info = env.reset()
    for step in range(best["step"] + 1):
        obs, reward, terminated, truncated, info = env.step(action)

    print()
    print("=" * 110)
    print("ALL CONTACTS AT BEST LIFT-WINDOW FRAME")
    print("=" * 110)

    print("data.ncon:", env.data.ncon)

    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        b1 = int(env.model.geom_bodyid[g1])
        b2 = int(env.model.geom_bodyid[g2])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        print(
            f"contact={i:02d} "
            f"g1={g1:03d}:{geom_name(env.model, g1):<35} "
            f"b1={b1:03d}:{body_name(env.model, b1):<30} "
            f"<-> "
            f"g2={g2:03d}:{geom_name(env.model, g2):<35} "
            f"b2={b2:03d}:{body_name(env.model, b2):<30} "
            f"dist={float(c.dist):+.6f} "
            f"pos=({float(c.pos[0]):+.4f},{float(c.pos[1]):+.4f},{float(c.pos[2]):+.4f}) "
            f"normal_force={float(force[0]):+.6f}"
        )

    env.close()


if __name__ == "__main__":
    main()
