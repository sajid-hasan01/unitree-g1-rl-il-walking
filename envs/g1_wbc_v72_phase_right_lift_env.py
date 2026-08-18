from __future__ import annotations

from typing import Dict

import numpy as np

from scripts.evaluate_wbc_v7_ankle_pitch_sweep import G1WBCV7AnklePitchSweepEnv


class G1WBCV72PhaseRightLiftEnv(G1WBCV7AnklePitchSweepEnv):
    """
    V7.2 phase-based right-foot lift.

    Sequence:
    1. neutral stand
    2. left preload
    3. right foot lift
    4. short air hold
    5. soft landing
    6. return preload to neutral

    Built from the successful V7 controller:
    preload_amp=0.040
    ankle_pitch_bias=-0.075
    lift_height=0.010
    foot_ik_gain=0.55
    """

    def __init__(
        self,
        lift_ramp_start: float = 0.46,
        lift_ramp_end: float = 0.56,
        lift_hold_end: float = 0.68,
        lift_land_end: float = 0.82,
        preload_down_start: float = 0.82,
        preload_down_end: float = 1.00,
        swing_knee_bias: float = 0.000,
        swing_hip_bias: float = 0.000,
        **kwargs,
    ):
        self.lift_ramp_start = float(lift_ramp_start)
        self.lift_ramp_end = float(lift_ramp_end)
        self.lift_hold_end = float(lift_hold_end)
        self.lift_land_end = float(lift_land_end)

        self.preload_down_start = float(preload_down_start)
        self.preload_down_end = float(preload_down_end)

        self.swing_knee_bias = float(swing_knee_bias)
        self.swing_hip_bias = float(swing_hip_bias)

        super().__init__(**kwargs)

    @staticmethod
    def _smoothstep_v72(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _tiny_lift_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.lift_ramp_start:
            return 0.0

        if phi < self.lift_ramp_end:
            return self._smoothstep_v72(
                (phi - self.lift_ramp_start)
                / max(self.lift_ramp_end - self.lift_ramp_start, 1e-6)
            )

        if phi < self.lift_hold_end:
            return 1.0

        if phi < self.lift_land_end:
            down = self._smoothstep_v72(
                (phi - self.lift_hold_end)
                / max(self.lift_land_end - self.lift_hold_end, 1e-6)
            )
            return 1.0 - down

        return 0.0

    def _preload_env(self, phi: float) -> float:
        phi = float(phi)

        if phi < self.preload_start:
            return 0.0

        if phi < self.preload_end:
            return self._smoothstep_v72(
                (phi - self.preload_start)
                / max(self.preload_end - self.preload_start, 1e-6)
            )

        if phi < self.preload_down_start:
            return 1.0

        if phi < self.preload_down_end:
            down = self._smoothstep_v72(
                (phi - self.preload_down_start)
                / max(self.preload_down_end - self.preload_down_start, 1e-6)
            )
            return 1.0 - down

        return 0.0

    def _phase_name(self, phi: float) -> str:
        phi = float(phi)

        if phi < self.preload_start:
            return "STAND"

        if phi < self.preload_end:
            return "PRELOAD"

        if phi < self.lift_ramp_start:
            return "PRELOAD_HOLD"

        if phi < self.lift_ramp_end:
            return "LIFT_RAMP"

        if phi < self.lift_hold_end:
            return "AIR_HOLD"

        if phi < self.lift_land_end:
            return "LAND"

        if phi < self.preload_down_end:
            return "RETURN_NEUTRAL"

        return "NEUTRAL_END"

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = super()._target_joint_position(action, info)

        phi = float(info["phase"])
        lift = self._tiny_lift_env(phi)

        # Optional micro swing shaping.
        # Controlled 15-DOF index:
        # right_hip_pitch = 6
        # right_knee      = 9
        target[6] += self.swing_hip_bias * lift
        target[9] += self.swing_knee_bias * lift

        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()

        phi = float(info["phase"])
        info["v72_phase_name"] = self._phase_name(phi)
        info["v72_lift_env"] = float(self._tiny_lift_env(phi))
        info["v72_preload_env"] = float(self._preload_env(phi))
        info["v72_swing_knee_bias"] = float(self.swing_knee_bias)
        info["v72_swing_hip_bias"] = float(self.swing_hip_bias)

        return info
