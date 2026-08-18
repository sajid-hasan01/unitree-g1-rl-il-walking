from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


def make_env():
    return G1WBCV72PhaseRightLiftEnv(
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
        land_end=0.82,

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

        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=0.68,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,

        swing_knee_bias=0.004,
        swing_hip_bias=0.002,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.70)
    parser.add_argument("--loops", type=int, default=1)
    args = parser.parse_args()

    env = make_env()
    action = np.zeros(4, dtype=np.float32)

    obs, info = env.reset()

    print("=" * 120)
    print("VISUAL DEMO: FINAL_v72_phase_right_lift_shaped")
    print("Expected: stand → preload → right lift → hold → land → return neutral.")
    print("Close the MuJoCo viewer window to stop.")
    print("=" * 120)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        loop = 0
        step = 0

        while viewer.is_running() and loop < args.loops:
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 20 == 0:
                print(
                    f"loop={loop} "
                    f"step={step:04d} "
                    f"phase={info.get('v72_phase_name', 'unknown'):<15} "
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
            time.sleep(max(env.dt / max(args.speed, 0.05), 0.001))

            step += 1

            if terminated or truncated or step >= 520:
                print("Loop ended:", env.termination_reason(info) if terminated else "max_steps")
                loop += 1
                step = 0

                if loop < args.loops:
                    obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
