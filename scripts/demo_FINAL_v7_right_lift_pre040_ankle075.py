from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v7_ankle_pitch_sweep import G1WBCV7AnklePitchSweepEnv


def main() -> None:
    env = G1WBCV7AnklePitchSweepEnv(
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
        preload_amp=0.040,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=-0.075,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    print("=" * 110)
    print("VISUAL DEMO: FINAL_v7_right_lift_pre040_ankle075")
    print("Expected: stable standing, left preload, right foot lift, then landing/hold.")
    print("Close the MuJoCo viewer window to stop.")
    print("=" * 110)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0

        while viewer.is_running() and step < 520:
            obs, reward, terminated, truncated, info = env.step(action)

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

            viewer.sync()
            time.sleep(env.dt)

            step += 1

            if terminated or truncated:
                print("Ended:", env.termination_reason(info))
                break

    env.close()


if __name__ == "__main__":
    main()
