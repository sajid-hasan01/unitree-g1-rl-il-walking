from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.smoke_wbc_v76_residual_right_lift import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.65)
    parser.add_argument("--loops", type=int, default=1)
    args = parser.parse_args()

    env = make_env()

    # FINAL V7.6B deterministic residual:
    # action = [ankle_residual, knee_residual, hip_residual, roll_residual]
    action = np.asarray([-1.0, 0.0, 0.0, 0.60], dtype=np.float32)

    obs, info = env.reset(seed=123)

    print("=" * 130)
    print("VISUAL DEMO: FINAL_v76b_ankle100_roll060")
    print("Base: FINAL_v72_phase_right_lift_shaped")
    print("Residual: ankle=-1.00, roll=+0.60")
    print("Expected: clearer right lift, slightly less final leg gap, stable landing.")
    print("=" * 130)

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
                    f"gapDelta={info.get('v76_gap_delta', 0.0):+.4f} "
                    f"x={info['x_position']:+.3f} "
                    f"y={info['y_position']:+.3f}"
                )

            viewer.sync()
            time.sleep(max(env.dt / max(args.speed, 0.05), 0.001))

            step += 1

            if terminated or truncated or step >= 520:
                reason = env.termination_reason(info) if terminated else "max_steps"
                print("Loop ended:", reason)
                loop += 1
                step = 0

                if loop < args.loops:
                    obs, info = env.reset(seed=123 + loop)

    env.close()


if __name__ == "__main__":
    main()
