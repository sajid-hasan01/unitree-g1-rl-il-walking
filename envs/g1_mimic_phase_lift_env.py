from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from envs.g1_phase_lift_env import CONTROLLED_15_JOINTS, G1PhaseLiftEnv


class G1MimicPhaseLiftEnv(G1PhaseLiftEnv):
    """
    DeepMimic-lite phase lift environment.

    v2 loader fix: supports processed dataset keys joint_pos_15 and joint_vel_15.

    Important design:
    - The reference motion is reward-only.
    - It never writes teacher pose into the controls.
    - The policy still controls the same 15 joints:
        target = stand_joint_pos + action_scale * action_scale_vector * action
    - Phase/contact/clearance/dynamics rewards remain active.
    - Reference imitation is intentionally weak and only guides swing-joint shape.
    """

    def __init__(
        self,
        *args,
        dataset_path: str = "datasets/processed/g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_phasecontact.npz",
        mimic_weight: float = 0.18,
        mimic_vel_weight: float = 0.04,
        mimic_only_during_swing: bool = True,
        reference_reverse: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dataset_path = dataset_path
        self.mimic_weight = float(mimic_weight)
        self.mimic_vel_weight = float(mimic_vel_weight)
        self.mimic_only_during_swing = bool(mimic_only_during_swing)
        self.reference_reverse = bool(reference_reverse)

        self.reference_joint_pos: Optional[np.ndarray] = None
        self.reference_joint_vel: Optional[np.ndarray] = None
        self.reference_contact_mask: Optional[np.ndarray] = None
        self.reference_indices: np.ndarray = np.arange(1, dtype=np.int64)
        self.reference_available = False
        self.reference_note = "none"

        self._load_reference()

    def _load_reference(self) -> None:
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            self.reference_note = f"missing:{self.dataset_path}"
            return

        data = np.load(self.dataset_path, allow_pickle=True)
        keys = set(data.files)

        pos_key = None
        for key in ["joint_pos_15", "joint_pos", "joint_positions", "qpos_15", "qpos", "dof_pos"]:
            if key in keys:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and arr.shape[1] >= 15:
                    pos_key = key
                    break

        if pos_key is None:
            self.reference_note = f"no_joint_pos_keys:{sorted(keys)}"
            return

        joint_pos = np.asarray(data[pos_key], dtype=np.float32)[:, :15]

        vel_key = None
        for key in ["joint_vel_15", "joint_vel", "joint_velocities", "qvel_15", "qvel", "dof_vel"]:
            if key in keys:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and arr.shape[1] >= 15:
                    vel_key = key
                    break

        if vel_key is not None:
            joint_vel = np.asarray(data[vel_key], dtype=np.float32)[:, :15]
        else:
            joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32)

        contact = None
        for key in ["contact_mask", "contacts", "foot_contact", "phase_contact"]:
            if key in keys:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    contact = arr[:, :2].astype(np.float32)
                    break

        self.reference_joint_pos = joint_pos
        self.reference_joint_vel = joint_vel
        self.reference_contact_mask = contact
        self.reference_available = True

        self.reference_indices = self._select_reference_indices()
        if self.reference_reverse:
            self.reference_indices = self.reference_indices[::-1].copy()

        self.reference_note = (
            f"loaded:{self.dataset_path};pos_key={pos_key};"
            f"vel_key={vel_key};frames={len(self.reference_indices)}"
        )

    def _select_reference_indices(self) -> np.ndarray:
        n = int(self.reference_joint_pos.shape[0])
        all_idx = np.arange(n, dtype=np.int64)

        if self.reference_contact_mask is None or self.reference_contact_mask.shape[0] != n:
            return all_idx

        left_contact = self.reference_contact_mask[:, 0] > 0.5
        right_contact = self.reference_contact_mask[:, 1] > 0.5

        if self.stage == "right_lift":
            mask = left_contact & (~right_contact)
        elif self.stage == "left_lift":
            mask = right_contact & (~left_contact)
        else:
            mask = left_contact ^ right_contact

        idx = np.where(mask)[0]
        if len(idx) < 8:
            return all_idx

        # Pick the longest contiguous run. This avoids mixing multiple tiny fragments.
        runs: List[np.ndarray] = []
        start = 0
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                runs.append(idx[start:i])
                start = i
        runs.append(idx[start:])
        longest = max(runs, key=len)

        if len(longest) < 8:
            return idx
        return longest.astype(np.int64)

    def _reference_index_for_phase(self) -> int:
        if not self.reference_available or len(self.reference_indices) == 0:
            return 0

        phi = self._phase01()
        # Map only the authored swing window to the selected reference segment.
        if self.stage in {"right_lift", "left_lift"}:
            start = self.cfg.swing_start
            end = self.cfg.swing_end
            if start <= phi < end:
                local = (phi - start) / max(end - start, 1e-6)
            else:
                # Stance phases use nearest boundary pose, not a moving target.
                local = 0.0 if phi < start else 1.0
        else:
            local = phi

        local = float(np.clip(local, 0.0, 1.0))
        j = int(round(local * (len(self.reference_indices) - 1)))
        return int(self.reference_indices[j])

    def _mimic_joint_weights(self) -> np.ndarray:
        weights = np.zeros(15, dtype=np.float32)

        if self.stage == "right_lift":
            weights[6:12] = np.array([1.0, 0.45, 0.25, 1.2, 0.9, 0.35], dtype=np.float32)
            weights[12:15] = np.array([0.10, 0.10, 0.15], dtype=np.float32)
        elif self.stage == "left_lift":
            weights[0:6] = np.array([1.0, 0.45, 0.25, 1.2, 0.9, 0.35], dtype=np.float32)
            weights[12:15] = np.array([0.10, 0.10, 0.15], dtype=np.float32)
        else:
            weights[0:12] = np.array(
                [0.8, 0.35, 0.20, 1.0, 0.75, 0.30] * 2,
                dtype=np.float32,
            )
            weights[12:15] = np.array([0.10, 0.10, 0.15], dtype=np.float32)

        return weights

    def _compute_mimic_reward(self, info: Dict[str, float]) -> Dict[str, float]:
        if not self.reference_available:
            return {
                "reward_mimic_pose": 0.0,
                "reward_mimic_vel": 0.0,
                "reward_mimic_total": 0.0,
                "reference_index": 0,
                "reference_available": 0.0,
                "reference_note_hash": 0.0,
            }

        if self.mimic_only_during_swing:
            if self.stage == "right_lift" and not bool(info["right_swing"]):
                return {
                    "reward_mimic_pose": 0.0,
                    "reward_mimic_vel": 0.0,
                    "reward_mimic_total": 0.0,
                    "reference_index": float(self._reference_index_for_phase()),
                    "reference_available": 1.0,
                    "reference_note_hash": 1.0,
                }
            if self.stage == "left_lift" and not bool(info["left_swing"]):
                return {
                    "reward_mimic_pose": 0.0,
                    "reward_mimic_vel": 0.0,
                    "reward_mimic_total": 0.0,
                    "reference_index": float(self._reference_index_for_phase()),
                    "reference_available": 1.0,
                    "reference_note_hash": 1.0,
                }

        ref_idx = self._reference_index_for_phase()
        ref_pos = self.reference_joint_pos[ref_idx]
        ref_vel = self.reference_joint_vel[ref_idx]

        joint_pos = np.array([self.data.qpos[qadr] for qadr in self.qpos_adrs], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[vadr] for vadr in self.qvel_adrs], dtype=np.float32)

        weights = self._mimic_joint_weights()
        denom = float(np.sum(weights) + 1e-6)

        pos_err = float(np.sum(weights * np.square(joint_pos - ref_pos)) / denom)
        vel_err = float(np.sum(weights * np.square(0.1 * (joint_vel - ref_vel))) / denom)

        pose_score = float(np.exp(-8.0 * pos_err))
        vel_score = float(np.exp(-4.0 * vel_err))

        reward_pose = self.mimic_weight * pose_score
        reward_vel = self.mimic_vel_weight * vel_score

        return {
            "reward_mimic_pose": float(reward_pose),
            "reward_mimic_vel": float(reward_vel),
            "reward_mimic_total": float(reward_pose + reward_vel),
            "mimic_pose_error": float(pos_err),
            "mimic_vel_error": float(vel_err),
            "reference_index": float(ref_idx),
            "reference_available": 1.0,
            "reference_note_hash": 1.0,
        }

    def _compute_reward(self, action: np.ndarray, info: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        # Start from the phase/contact/dynamics reward. This contains no pose teacher.
        base_reward, reward_info = super()._compute_reward(action, info)

        mimic_info = self._compute_mimic_reward(info)
        mimic_total = float(mimic_info["reward_mimic_total"])

        # Extra dynamics guard from the papers: do not reward foot lift if root dynamics
        # are becoming invalid. This specifically targets the v2/v3 forward-drift exploit.
        root_ang_vel = float(np.linalg.norm(self.data.qvel[3:6]))
        root_speed_xy = float(np.linalg.norm(self.data.qvel[0:2]))
        x_abs = abs(float(info["x_position"]))

        dynamics_penalty = (
            -0.55 * root_ang_vel
            -0.85 * root_speed_xy
            -8.0 * max(0.0, x_abs - 0.18)
        )

        total = float(base_reward + mimic_total + dynamics_penalty)

        reward_info.update(mimic_info)
        reward_info.update(
            {
                "reward_dynamics_penalty": float(dynamics_penalty),
                "root_ang_vel": float(root_ang_vel),
                "root_speed_xy": float(root_speed_xy),
                "reward_total": float(total),
                "reward_version": "mimic_phase_lift_v1_reward_only_reference",
            }
        )
        return total, reward_info

    def _get_info(self) -> Dict[str, float]:
        info = super()._get_info()
        info["reference_available"] = float(self.reference_available)
        info["reference_segment_length"] = float(len(self.reference_indices))
        info["mimic_weight"] = float(self.mimic_weight)
        info["mimic_vel_weight"] = float(self.mimic_vel_weight)
        return info
