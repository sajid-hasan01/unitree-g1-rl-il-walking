from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_v72_phase_right_lift_env import G1WBCV72PhaseRightLiftEnv


CONTROLLED_15_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


class G1WBCV74SwingFootYPlaceEnv(G1WBCV72PhaseRightLiftEnv):
    """
    V7.4: right-foot lift + swing-foot lateral placement.

    Goal:
    - prevent right foot from landing too wide
    - reduce final leg gap
    - preserve right-foot air-time
    """

    def __init__(
        self,
        right_y_offset: float = 0.0,
        y_ik_gain: float = 0.35,
        y_task_max: float = 0.025,
        y_ik_damping: float = 0.080,
        y_ik_max_delta: float = 0.020,
        y_place_start: float = 0.50,
        y_place_full: float = 0.58,
        y_place_end: float = 0.82,
        **kwargs,
    ):
        self.right_y_offset = float(right_y_offset)
        self.y_ik_gain = float(y_ik_gain)
        self.y_task_max = float(y_task_max)
        self.y_ik_damping = float(y_ik_damping)
        self.y_ik_max_delta = float(y_ik_max_delta)

        self.y_place_start = float(y_place_start)
        self.y_place_full = float(y_place_full)
        self.y_place_end = float(y_place_end)

        self._right_y0 = None
        self._left_y0 = None
        self._dof_ids_15: List[int] = []

        super().__init__(**kwargs)

        self._dof_ids_15 = []
        for name in CONTROLLED_15_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found in MuJoCo model: {name}")
            self._dof_ids_15.append(int(self.model.jnt_dofadr[jid]))

    @staticmethod
    def _smoothstep_y(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self._right_y0 = float(self.data.site_xpos[self.right_foot_site][1])
        self._left_y0 = float(self.data.site_xpos[self.left_foot_site][1])

        return obs, info

    def _y_place_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.y_place_start:
            return 0.0

        if phi < self.y_place_full:
            return self._smoothstep_y(
                (phi - self.y_place_start)
                / max(self.y_place_full - self.y_place_start, 1e-6)
            )

        if phi < self.y_place_end:
            return 1.0

        return 0.0

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        phi = float(info["phase"])
        env = self._y_place_env(phi)

        if env <= 1e-6:
            return target

        if self._right_y0 is None:
            self._right_y0 = float(self.data.site_xpos[self.right_foot_site][1])

        current_y = float(self.data.site_xpos[self.right_foot_site][1])
        desired_y = float(self._right_y0 + self.right_y_offset)

        task_error = desired_y - current_y
        task_delta_y = float(
            np.clip(self.y_ik_gain * task_error, -self.y_task_max, self.y_task_max)
        )

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.right_foot_site)

        J = jacp[1, self._dof_ids_15].astype(np.float64)

        # Only use right-leg lateral joints + waist roll.
        # Controlled indices:
        # right_hip_roll   = 7
        # right_hip_yaw    = 8
        # right_ankle_roll = 11
        # waist_roll       = 13
        mask = np.zeros(15, dtype=np.float64)
        mask[7] = 1.0
        mask[8] = 0.35
        mask[11] = 0.65
        mask[13] = 0.25

        Jm = J * mask
        denom = float(Jm @ Jm + self.y_ik_damping * self.y_ik_damping)

        if denom > 1e-8:
            dq = Jm * (task_delta_y / denom)
            dq = np.clip(dq, -self.y_ik_max_delta, self.y_ik_max_delta)
            target = target + env * dq.astype(np.float32)

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target


def contact_forces(env):
    right_body = int(env.model.site_bodyid[env.right_foot_site])
    left_body = int(env.model.site_bodyid[env.left_foot_site])

    right_force = 0.0
    left_force = 0.0

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        b1 = int(env.model.geom_bodyid[int(c.geom1)])
        b2 = int(env.model.geom_bodyid[int(c.geom2)])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == right_body or b2 == right_body:
            right_force += normal_force

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

    return right_force, left_force


def make_env(right_y_offset: float, y_ik_gain: float):
    return G1WBCV74SwingFootYPlaceEnv(
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
        preload_amp=0.036,
        preload_start=0.08,
        preload_end=0.42,

        knee_bias=0.0,
        hip_ratio=0.0,
        ankle_ratio=0.0,

        ankle_pitch_bias=-0.070,

        lift_ramp_start=0.46,
        lift_ramp_end=0.56,
        lift_hold_end=0.64,
        lift_land_end=0.82,
        preload_down_start=0.82,
        preload_down_end=1.00,

        swing_knee_bias=0.004,
        swing_hip_bias=0.002,

        right_y_offset=right_y_offset,
        y_ik_gain=y_ik_gain,
        y_task_max=0.025,
        y_ik_damping=0.080,
        y_ik_max_delta=0.020,
        y_place_start=0.50,
        y_place_full=0.58,
        y_place_end=0.82,
    )


def run_case(right_y_offset: float, y_ik_gain: float) -> dict:
    env = make_env(right_y_offset, y_ik_gain)

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()

    right_y0 = float(env.data.site_xpos[env.right_foot_site][1])
    left_y0 = float(env.data.site_xpos[env.left_foot_site][1])
    initial_gap = abs(right_y0 - left_y0)

    steps = 0
    terminated = False
    truncated = False
    final = info

    max_right_clearance = 0.0
    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    right_air_steps = 0
    max_right_air_streak = 0
    current_air_streak = 0

    min_right_force = 1e9
    right_ratio_at_min = 1.0

    max_gap = initial_gap

    first_air_step = -1
    last_air_step = -1

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])

        right_y = float(env.data.site_xpos[env.right_foot_site][1])
        left_y = float(env.data.site_xpos[env.left_foot_site][1])
        current_gap = abs(right_y - left_y)
        max_gap = max(max_gap, current_gap)

        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))
        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))

        if not bool(info["right_contact"]):
            right_air_steps += 1
            current_air_streak += 1
            max_right_air_streak = max(max_right_air_streak, current_air_streak)

            if first_air_step < 0:
                first_air_step = steps
            last_air_step = steps
        else:
            current_air_streak = 0

        if 0.46 <= phi <= 0.82:
            rf, lf = contact_forces(env)
            if rf < min_right_force:
                min_right_force = rf
                right_ratio_at_min = rf / max(rf + lf, 1e-6)

        if steps >= 520:
            break

    right_y_end = float(env.data.site_xpos[env.right_foot_site][1])
    left_y_end = float(env.data.site_xpos[env.left_foot_site][1])
    final_gap = abs(right_y_end - left_y_end)

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "right_y_offset": right_y_offset,
        "y_ik_gain": y_ik_gain,
        "steps": steps,
        "reason": reason,
        "right_clearance": max_right_clearance,
        "right_air_steps": right_air_steps,
        "right_air_streak": max_right_air_streak,
        "first_air_step": first_air_step,
        "last_air_step": last_air_step,
        "right_force_min": min_right_force,
        "right_force_ratio": right_ratio_at_min,
        "min_up_z": min_up_z,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "right_y0": right_y0,
        "left_y0": left_y0,
        "right_y_end": right_y_end,
        "left_y_end": left_y_end,
        "initial_gap": initial_gap,
        "max_gap": max_gap,
        "final_gap": final_gap,
        "gap_delta": final_gap - initial_gap,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v74_swing_foot_y_place_sweep.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    y_offsets = [-0.040, -0.020, 0.000, 0.020, 0.040]
    gains = [0.25, 0.45]

    rows = []

    print("=" * 160)
    print("WBC V7.4 SWING-FOOT LATERAL PLACEMENT SWEEP")
    print("Goal: keep right-foot air while preventing the right foot from landing too wide.")
    print("=" * 160)

    for gain in gains:
        for off in y_offsets:
            row = run_case(off, gain)
            rows.append(row)

            print(
                f"gain={gain:.2f} "
                f"yoff={off:+.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Rclear={row['right_clearance']:.4f} "
                f"Rair={row['right_air_steps']:03d} "
                f"Rstreak={row['right_air_streak']:03d} "
                f"air={row['first_air_step']:03d}-{row['last_air_step']:03d} "
                f"Rforce={row['right_force_min']:.2f} "
                f"up={row['min_up_z']:.3f} "
                f"slip={row['support_slip']:.4f} "
                f"Ry0={row['right_y0']:+.3f} "
                f"RyEnd={row['right_y_end']:+.3f} "
                f"Ly0={row['left_y0']:+.3f} "
                f"LyEnd={row['left_y_end']:+.3f} "
                f"gapEnd={row['final_gap']:.4f} "
                f"gapDelta={row['gap_delta']:+.4f} "
                f"x={row['final_x']:+.3f} "
                f"y={row['final_y']:+.3f}"
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
