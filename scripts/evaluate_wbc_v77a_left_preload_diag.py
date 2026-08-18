from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_wbc_neutral_only_env import G1WBCNeutralOnlyEnv


class G1WBCV77ALeftPreloadDiagEnv(G1WBCNeutralOnlyEnv):
    """
    V7.7A left-preload diagnostic.

    No foot lift.
    No ankle pitch lift.
    Only preload templates.

    Goal:
    Find a lateral preload pattern that unloads the LEFT foot
    while keeping the robot stable.
    """

    def __init__(
        self,
        template_name: str,
        preload_sign: float,
        preload_amp: float,
        preload_start: float = 0.08,
        preload_end: float = 0.42,
        **kwargs,
    ):
        self.template_name = str(template_name)
        self.preload_sign = float(preload_sign)
        self.preload_amp = float(preload_amp)
        self.preload_start = float(preload_start)
        self.preload_end = float(preload_end)

        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _preload_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.preload_start:
            return 0.0

        if phi < self.preload_end:
            return self._smoothstep(
                (phi - self.preload_start)
                / max(self.preload_end - self.preload_start, 1e-6)
            )

        return 1.0

    def _template_coeffs(self):
        """
        Returns coefficients for:
        left_hip_roll, left_ankle_roll,
        right_hip_roll, right_ankle_roll,
        waist_roll
        """

        templates = {
            # Template used in the failed V7.7 left mirror.
            "same_failed":      (+0.60, -0.30, +0.60, -0.30, -0.20),

            # Direct inverse of the failed template.
            "same_inverse":     (-0.60, +0.30, -0.60, +0.30, +0.20),

            # Legs roll in opposite directions.
            "opp_a":            (+0.60, -0.30, -0.60, +0.30, -0.20),
            "opp_b":            (-0.60, +0.30, +0.60, -0.30, +0.20),

            # Bias toward right support leg.
            "right_support_a":  (-0.25, +0.15, +0.90, -0.45, +0.20),
            "right_support_b":  (+0.25, -0.15, -0.90, +0.45, -0.20),

            # Stronger support-side ankle compensation.
            "right_support_c":  (-0.20, +0.35, +0.85, -0.65, +0.20),
            "right_support_d":  (+0.20, -0.35, -0.85, +0.65, -0.20),
        }

        if self.template_name not in templates:
            raise ValueError(f"Unknown template_name: {self.template_name}")

        return templates[self.template_name]

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = self.stand_joint_pos.copy()

        phi = float(info["phase"])
        env = self._preload_env(phi)

        c = self.preload_sign * self.preload_amp * env

        lh, la, rh, ra, waist = self._template_coeffs()

        # Controlled 15-DOF indices:
        # left_hip_roll    = 1
        # left_ankle_roll  = 5
        # right_hip_roll   = 7
        # right_ankle_roll = 11
        # waist_roll       = 13
        target[1] += lh * c
        target[5] += la * c
        target[7] += rh * c
        target[11] += ra * c
        target[13] += waist * c

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target


def contact_forces(env):
    left_body = int(env.model.site_bodyid[env.left_foot_site])
    right_body = int(env.model.site_bodyid[env.right_foot_site])

    left_force = 0.0
    right_force = 0.0

    for i in range(env.data.ncon):
        c = env.data.contact[i]

        b1 = int(env.model.geom_bodyid[int(c.geom1)])
        b2 = int(env.model.geom_bodyid[int(c.geom2)])

        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, i, force)

        normal_force = float(force[0])

        if b1 == left_body or b2 == left_body:
            left_force += normal_force

        if b1 == right_body or b2 == right_body:
            right_force += normal_force

    return left_force, right_force


def run_case(template_name: str, preload_sign: float, preload_amp: float) -> dict:
    env = G1WBCV77ALeftPreloadDiagEnv(
        frame_skip=5,
        max_steps=420,
        cycle_duration=5.2,

        template_name=template_name,
        preload_sign=preload_sign,
        preload_amp=preload_amp,
        preload_start=0.08,
        preload_end=0.42,
    )

    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset(seed=123)

    steps = 0
    terminated = False
    truncated = False
    final = info

    min_left_force = 1e9
    right_force_at_min = 0.0
    left_ratio_at_min = 1.0

    min_up_z = 1.0
    max_slip = 0.0
    max_ang = 0.0

    max_left_clearance = 0.0
    max_right_clearance = 0.0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        final = info

        phi = float(info["phase"])

        min_up_z = min(min_up_z, float(info["up_z"]))
        max_slip = max(max_slip, float(info["support_slip"]))
        max_ang = max(max_ang, float(info["root_ang_vel"]))

        max_left_clearance = max(max_left_clearance, float(info["left_foot_clearance"]))
        max_right_clearance = max(max_right_clearance, float(info["right_foot_clearance"]))

        if 0.42 <= phi <= 0.82:
            lf, rf = contact_forces(env)

            if lf < min_left_force:
                min_left_force = lf
                right_force_at_min = rf
                left_ratio_at_min = lf / max(lf + rf, 1e-6)

        if steps >= 420:
            break

    reason = env.termination_reason(final) if terminated else "max_steps"

    row = {
        "template": template_name,
        "preload_sign": preload_sign,
        "preload_amp": preload_amp,
        "steps": steps,
        "reason": reason,
        "left_force_min": min_left_force,
        "right_force_at_min": right_force_at_min,
        "left_force_ratio": left_ratio_at_min,
        "min_up_z": min_up_z,
        "support_slip": max_slip,
        "max_root_ang_vel": max_ang,
        "left_clearance": max_left_clearance,
        "right_clearance": max_right_clearance,
        "final_x": float(final["x_position"]),
        "final_y": float(final["y_position"]),
        "final_x_velocity": float(final["x_velocity"]),
        "final_y_velocity": float(final["y_velocity"]),
    }

    env.close()
    return row


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v77a_left_preload_diag.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    templates = [
        "same_failed",
        "same_inverse",
        "opp_a",
        "opp_b",
        "right_support_a",
        "right_support_b",
        "right_support_c",
        "right_support_d",
    ]

    signs = [-1.0, +1.0]
    amps = [0.020, 0.040, 0.060, 0.080]

    rows = []

    print("=" * 170)
    print("WBC V7.7A LEFT PRELOAD DIAGNOSTIC SWEEP")
    print("Goal: find a stable preload template that unloads the LEFT foot before adding lift.")
    print("=" * 170)

    for template in templates:
        for sign in signs:
            for amp in amps:
                row = run_case(template, sign, amp)
                rows.append(row)

                print(
                    f"tpl={row['template']:<16} "
                    f"sign={row['preload_sign']:+.0f} "
                    f"amp={row['preload_amp']:.3f} "
                    f"steps={row['steps']:04d} "
                    f"reason={row['reason']:<16} "
                    f"Lforce={row['left_force_min']:.2f} "
                    f"Rforce={row['right_force_at_min']:.2f} "
                    f"Lratio={row['left_force_ratio']:.3f} "
                    f"Lclear={row['left_clearance']:.4f} "
                    f"Rclear={row['right_clearance']:.4f} "
                    f"up={row['min_up_z']:.3f} "
                    f"slip={row['support_slip']:.4f} "
                    f"ang={row['max_root_ang_vel']:.3f} "
                    f"x={row['final_x']:+.3f} "
                    f"y={row['final_y']:+.3f} "
                    f"xv={row['final_x_velocity']:+.3f}"
                )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
