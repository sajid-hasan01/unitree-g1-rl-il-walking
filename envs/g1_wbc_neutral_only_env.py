from __future__ import annotations

from typing import Dict

import numpy as np

from envs.g1_wbc_taskspace_right_lift_env import G1WBCTaskspaceRightLiftEnv


class G1WBCNeutralOnlyEnv(G1WBCTaskspaceRightLiftEnv):
    """
    Clean diagnostic WBC env.

    It intentionally bypasses:
    - parent right-foot swing target
    - WBC support_env support push
    - support-foot IK lock
    - capture touchdown
    - recovery state machine target changes

    Purpose:
    prove whether normal phase timing is safe when the target is truly neutral.
    """

    def _update_wbc_state(self, info: Dict[str, float]) -> None:
        self._wbc_state = "NEUTRAL_ONLY"
        self._capture_active = False
        self._abort_lift = False
        self._capture_count = 0
        self._abort_count = 0
        self._guard_count = 0
        self._touchdown_force = 0.0
        self._touchdown_timer = 0.0

    def _target_foot_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        return self.right_foot_p0.copy()

    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        target = self.stand_joint_pos.copy().astype(np.float64)

        # Keep targets inside actuator limits.
        for i, aid in enumerate(self.actuator_ids):
            low = float(self.model.actuator_ctrlrange[aid, 0])
            high = float(self.model.actuator_ctrlrange[aid, 1])
            target[i] = float(np.clip(target[i], low, high))

        return target

    def _get_info(self):
        info = super()._get_info()

        info["wbc_state"] = "NEUTRAL_ONLY"
        info["wbc_capture_active"] = False
        info["wbc_abort_lift"] = False
        info["wbc_capture_count"] = 0.0
        info["wbc_abort_count"] = 0.0
        info["wbc_guard_count"] = 0.0
        info["wbc_touchdown_force"] = 0.0

        return info
