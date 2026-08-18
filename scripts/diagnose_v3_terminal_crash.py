import sys
import faulthandler
import traceback
from pathlib import Path

import numpy as np

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=" * 90, flush=True)
print("V3 CRASH DIAGNOSTIC — BASELINE ONLY", flush=True)
print("=" * 90, flush=True)
print("ROOT:", ROOT, flush=True)

env = None

try:
    print("[1] Importing PPO...", flush=True)

    from stable_baselines3 import PPO

    print("[2] Importing V3 environment...", flush=True)

    from envs.g1_bc_line_residual_env_v3 import (
        G1BCLineResidualEnvV3,
    )

    model_path = (
        ROOT
        / "models"
        / "g1_ppo_bc_line_residual_v3_50k.zip"
    )

    print("[3] Model path:", model_path, flush=True)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print("[4] Loading PPO model...", flush=True)

    policy = PPO.load(
        str(model_path),
        device="cpu",
    )

    print("[5] PPO model loaded.", flush=True)

    print("[6] Creating MuJoCo environment...", flush=True)

    env = G1BCLineResidualEnvV3(
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,
        residual_scale=0.14,
        residual_ramp_frames=30,
        target_velocity=-0.18,
        max_episode_steps=600,
    )

    print("[7] Environment created.", flush=True)

    print("[8] Resetting environment...", flush=True)

    obs, info = env.reset(seed=1234)

    print(
        "[9] Reset successful. obs shape =",
        np.asarray(obs).shape,
        flush=True,
    )

    steps = 0
    max_y = 0.0
    max_yaw = 0.0
    min_up = 1.0
    max_action = 0.0

    print("[10] Starting deterministic rollout...", flush=True)

    while True:

        action, _ = policy.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        max_action = max(
            max_action,
            float(np.max(np.abs(action))),
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        steps += 1

        y = float(info["y"])
        yaw = float(info["yaw_deg"])
        up = float(info["up_z"])

        max_y = max(
            max_y,
            abs(y),
        )

        max_yaw = max(
            max_yaw,
            abs(yaw),
        )

        min_up = min(
            min_up,
            up,
        )

        # Frequent flushed output lets us identify
        # the exact point if a native crash occurs.
        if steps % 5 == 0:

            print(
                f"step={steps:03d} "
                f"X={float(info['x']):+.3f} "
                f"Y={y:+.3f} "
                f"yaw={yaw:+.1f} "
                f"up={up:.3f} "
                f"Amax={max_action:.3f}",
                flush=True,
            )

        if terminated or truncated:

            print(
                "",
                flush=True,
            )

            print(
                "ROLLOUT TERMINATED NORMALLY",
                flush=True,
            )

            print(
                f"steps={steps}",
                flush=True,
            )

            print(
                f"final X={float(info['x']):+.4f}",
                flush=True,
            )

            print(
                f"final Y={y:+.4f}",
                flush=True,
            )

            print(
                f"max |Y|={max_y:.4f}",
                flush=True,
            )

            print(
                f"final yaw={yaw:+.2f}",
                flush=True,
            )

            print(
                f"max |yaw|={max_yaw:.2f}",
                flush=True,
            )

            print(
                f"min up_z={min_up:.4f}",
                flush=True,
            )

            print(
                f"max action={max_action:.4f}",
                flush=True,
            )

            print(
                f"terminated={terminated}",
                flush=True,
            )

            print(
                f"truncated={truncated}",
                flush=True,
            )

            break


except Exception:

    print("", flush=True)
    print("=" * 90, flush=True)
    print("PYTHON EXCEPTION", flush=True)
    print("=" * 90, flush=True)

    traceback.print_exc()

    raise


finally:

    if env is not None:

        print(
            "Closing environment...",
            flush=True,
        )

        try:
            env.close()

        except Exception:
            traceback.print_exc()

    print(
        "Diagnostic process reached finally().",
        flush=True,
    )
