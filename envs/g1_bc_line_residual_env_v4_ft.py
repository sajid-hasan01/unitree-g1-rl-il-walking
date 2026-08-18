from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
import torch.nn as nn


class BCWalkingPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class G1BCLineResidualEnvV4FT(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        policy_path="models/g1_bc_walking_policy.pt",
        dataset_path="datasets/processed/g1_amass_walking_il_15dof.npz",
        model_path="third_party/mujoco_menagerie/unitree_g1/scene.xml",
        start_frame=25,
        stand_frames=45,
        transition_frames=90,
        target_smoothing=0.35,
        residual_scale=0.14,
        residual_ramp_frames=30,
        target_velocity=-0.18,
        max_episode_steps=600,
    ):
        super().__init__()

        root = Path(__file__).resolve().parents[1]

        self.policy_path = root / policy_path
        self.dataset_path = root / dataset_path
        self.model_path = root / model_path

        self.start_frame = int(start_frame)
        self.stand_frames = int(stand_frames)
        self.transition_frames = int(transition_frames)
        self.target_smoothing = float(target_smoothing)
        self.residual_scale = float(residual_scale)
        self.residual_ramp_frames = int(residual_ramp_frames)
        self.target_velocity = float(target_velocity)
        self.max_episode_steps = int(max_episode_steps)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            self.policy_path,
            map_location=self.device,
            weights_only=False,
        )

        self.obs_mean = np.asarray(
            checkpoint["obs_mean"],
            dtype=np.float32,
        ).reshape(-1)

        self.obs_std = np.asarray(
            checkpoint["obs_std"],
            dtype=np.float32,
        ).reshape(-1)

        self.action_mean = np.asarray(
            checkpoint["action_mean"],
            dtype=np.float32,
        ).reshape(-1)

        self.action_std = np.asarray(
            checkpoint["action_std"],
            dtype=np.float32,
        ).reshape(-1)

        self.joint_names = [
            str(x)
            for x in checkpoint["controlled_joint_names"]
        ]

        self.bc_policy = BCWalkingPolicy(
            int(checkpoint["obs_dim"]),
            int(checkpoint["action_dim"]),
        ).to(self.device)

        self.bc_policy.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.bc_policy.eval()

        dataset = np.load(
            self.dataset_path,
            allow_pickle=True,
        )

        self.num_frames = int(
            dataset["joint_pos_15"].shape[0]
            if "joint_pos_15" in dataset
            else dataset["il_actions"].shape[0]
        )

        self.fps = float(
            np.asarray(dataset["fps"]).reshape(-1)[0]
        )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        self.data = mujoco.MjData(self.model)

        if self.model.nkey > 0:
            self.stand_qpos = self.model.key_qpos[0].copy()
        else:
            self.stand_qpos = np.zeros(
                self.model.nq,
                dtype=np.float64,
            )
            self.stand_qpos[3] = 1.0
            self.stand_qpos[2] = 0.79

        self.joint_info = []

        for name in self.joint_names:

            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if jid < 0:
                raise RuntimeError(f"Joint not found: {name}")

            aid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )

            if aid < 0:
                for candidate in range(self.model.nu):
                    if int(
                        self.model.actuator_trnid[candidate, 0]
                    ) == jid:
                        aid = candidate
                        break

            if aid < 0:
                raise RuntimeError(
                    f"Actuator not found for {name}"
                )

            self.joint_info.append(
                {
                    "jid": int(jid),
                    "aid": int(aid),
                    "qadr": int(self.model.jnt_qposadr[jid]),
                    "vadr": int(self.model.jnt_dofadr[jid]),
                }
            )

        self.controlled_aids = {
            item["aid"]
            for item in self.joint_info
        }

        self.upper_body = []

        for aid in range(self.model.nu):

            if aid in self.controlled_aids:
                continue

            jid = int(
                self.model.actuator_trnid[aid, 0]
            )

            if jid < 0 or jid >= self.model.njnt:
                continue

            qadr = int(
                self.model.jnt_qposadr[jid]
            )

            self.upper_body.append(
                (
                    aid,
                    float(self.stand_qpos[qadr]),
                )
            )

        self.pelvis_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )

        self.left_site = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_foot",
        )

        self.right_site = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_foot",
        )

        self.left_body = int(
            self.model.site_bodyid[self.left_site]
        )

        self.right_body = int(
            self.model.site_bodyid[self.right_site]
        )

        self.left_sole = self._find_sole(self.left_body)
        self.right_sole = self._find_sole(self.right_body)

        self.floor_gid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )

        self.sim_dt = float(self.model.opt.timestep)

        self.control_dt = 1.0 / self.fps

        self.frame_skip = max(
            1,
            int(round(self.control_dt / self.sim_dt)),
        )

        self.num_actions = len(self.joint_names)

        self.residual_joint_scale = np.ones(
            self.num_actions,
            dtype=np.float32,
        )

        for i, name in enumerate(self.joint_names):

            if name == "waist_yaw_joint":
                self.residual_joint_scale[i] = 0.35

            elif name in {
                "waist_roll_joint",
                "waist_pitch_joint",
            }:
                self.residual_joint_scale[i] = 0.60

            elif (
                "hip_roll_joint" in name
                or "hip_yaw_joint" in name
                or "ankle" in name
            ):
                self.residual_joint_scale[i] = 0.65

            else:
                # hip pitch + knee
                self.residual_joint_scale[i] = 1.00

        # =============================================================
        # V4-FT PHYSICAL ENVELOPE
        #
        # Preserve V2 reward/controller structure.
        # Only apply the experimentally validated ankle-roll envelope.
        #
        # V2 ankle-roll scale = 0.65
        # best ablation factor = 0.65
        # resulting scale      = 0.4225
        # =============================================================

        for i, name in enumerate(self.joint_names):

            if "ankle_roll_joint" in name:

                self.residual_joint_scale[i] = 0.4225

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # observation:
        # height                  1
        # quaternion              4
        # root velocity           6
        # joint positions        15
        # joint velocities       15
        # BC nominal targets     15
        # BC phase                3
        # line Y / yaw            2
        # target velocity         1
        # -------------------------
        # total                  62
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(62,),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(
            self.num_actions,
            dtype=np.float32,
        )

        self.previous_targets = np.zeros(
            self.num_actions,
            dtype=np.float64,
        )

        self.episode_step = 0
        self.floor_z = 0.0


    def _clip_ctrl(self, aid, value):

        lo, hi = self.model.actuator_ctrlrange[aid]

        return float(
            np.clip(value, lo, hi)
        )


    def _find_sole(self, body_id):

        ids = []

        for gid in range(self.model.ngeom):

            if int(self.model.geom_bodyid[gid]) != int(body_id):
                continue

            if int(self.model.geom_type[gid]) != int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            ):
                continue

            if float(self.model.geom_size[gid][0]) <= 0.010:
                ids.append(gid)

        return sorted(ids)


    @staticmethod
    def _yaw(q):

        w, x, y, z = [float(v) for v in q]

        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )


    def _up_z(self):

        R = np.asarray(
            self.data.xmat[self.pelvis_id],
            dtype=float,
        ).reshape(3, 3)

        return float(R[2, 2])


    def _joint_positions(self):

        return np.array(
            [
                self.data.qpos[item["qadr"]]
                for item in self.joint_info
            ],
            dtype=np.float32,
        )


    def _joint_velocities(self):

        return np.array(
            [
                self.data.qvel[item["vadr"]]
                for item in self.joint_info
            ],
            dtype=np.float32,
        )


    def _phase_features(self, gait_frame):

        progress = gait_frame / max(
            self.num_frames - 1,
            1,
        )

        phase = 2.0 * np.pi * progress

        return np.array(
            [
                np.sin(phase),
                np.cos(phase),
                progress,
            ],
            dtype=np.float32,
        )


    def _bc_targets(self, gait_frame):

        raw = self._phase_features(gait_frame)

        norm = (
            raw - self.obs_mean
        ) / self.obs_std

        x = torch.tensor(
            norm,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            pred = (
                self.bc_policy(x)
                .cpu()
                .numpy()[0]
            )

        return (
            pred * self.action_std
            + self.action_mean
        ).astype(np.float64)


    def _smoothstep(self, x):

        x = float(
            np.clip(x, 0.0, 1.0)
        )

        return x * x * (3.0 - 2.0 * x)


    def _nominal_target(self):

        stand = np.array(
            [
                self.stand_qpos[item["qadr"]]
                for item in self.joint_info
            ],
            dtype=np.float64,
        )

        if self.episode_step < self.stand_frames:

            return stand, 0, 0.0

        local = (
            self.episode_step
            - self.stand_frames
        )

        gait_frame = int(
            (
                self.start_frame
                + local
            )
            % self.num_frames
        )

        bc = self._bc_targets(
            gait_frame
        )

        alpha = self._smoothstep(
            local
            / max(
                self.transition_frames,
                1,
            )
        )

        nominal = (
            (1.0 - alpha) * stand
            + alpha * bc
        )

        return (
            nominal,
            gait_frame,
            alpha,
        )


    def _foot_contact(self, sole):

        geom_set = set(
            int(x)
            for x in sole
        )

        for i in range(self.data.ncon):

            c = self.data.contact[i]

            a = int(c.geom1)
            b = int(c.geom2)

            if (
                a == self.floor_gid
                and b in geom_set
            ):
                return True

            if (
                b == self.floor_gid
                and a in geom_set
            ):
                return True

        return False


    def _sole_clearance(self, sole):

        values = []

        for gid in sole:

            values.append(
                float(self.data.geom_xpos[gid][2])
                - float(self.model.geom_size[gid][0])
                - self.floor_z
            )

        return float(min(values))


    def _build_obs(self):

        nominal, gait_frame, alpha = (
            self._nominal_target()
        )

        phase = self._phase_features(
            gait_frame
        )

        yaw = self._yaw(
            self.data.qpos[3:7]
        )

        parts = [
            np.array(
                [self.data.qpos[2]],
                dtype=np.float32,
            ),
            self.data.qpos[3:7].astype(
                np.float32
            ),
            (
                0.1
                * self.data.qvel[0:6]
            ).astype(np.float32),
            self._joint_positions(),
            (
                0.1
                * self._joint_velocities()
            ).astype(np.float32),
            nominal.astype(np.float32),
            phase,
            np.array(
                [
                    float(self.data.qpos[1]),
                    float(yaw),
                ],
                dtype=np.float32,
            ),
            np.array(
                [self.target_velocity],
                dtype=np.float32,
            ),
        ]

        obs = np.concatenate(parts)

        if obs.shape != (62,):
            raise RuntimeError(
                f"Unexpected observation shape {obs.shape}"
            )

        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


    def _reward(self, action):

        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])

        wx = float(self.data.qvel[3])
        wy = float(self.data.qvel[4])
        wz = float(self.data.qvel[5])

        y = float(self.data.qpos[1])

        yaw = self._yaw(
            self.data.qpos[3:7]
        )

        up = self._up_z()

        z = float(self.data.qpos[2])

        # -------------------------------------------------------------
        # Upright stability
        # -------------------------------------------------------------

        upright_error = 1.0 - up

        upright_reward = np.exp(
            -(upright_error ** 2) / 0.012
        )

        # -------------------------------------------------------------
        # Height
        # -------------------------------------------------------------

        target_height = float(
            self.stand_qpos[2]
        )

        height_error = (
            z - target_height
        )

        height_reward = np.exp(
            -(height_error ** 2) / 0.018
        )

        # -------------------------------------------------------------
        # Forward velocity.
        #
        # Do not reward forward velocity strongly while falling.
        # -------------------------------------------------------------

        velocity_error = (
            vx - self.target_velocity
        )

        velocity_reward = np.exp(
            -(velocity_error ** 2) / 0.035
        )

        upright_gate = float(
            np.clip(
                (up - 0.55) / 0.35,
                0.0,
                1.0,
            )
        )

        velocity_reward *= upright_gate

        # -------------------------------------------------------------
        # Straight-line terms
        # -------------------------------------------------------------

        line_penalty = (
            4.0 * y * y
            + 0.80 * abs(y)
        )

        yaw_penalty = (
            1.40 * yaw * yaw
            + 0.20 * abs(yaw)
        )

        lateral_velocity_penalty = (
            1.80 * vy * vy
            + 0.30 * abs(vy)
        )

        # -------------------------------------------------------------
        # Root angular-velocity stabilization.
        #
        # This term was missing in V1.
        # -------------------------------------------------------------

        angular_velocity_penalty = (
            0.10 * (wx * wx + wy * wy)
            + 0.04 * (wz * wz)
        )

        # -------------------------------------------------------------
        # Strong collapse shaping
        # -------------------------------------------------------------

        tilt_penalty = (
            5.0 * max(0.90 - up, 0.0)
            + 25.0 * max(0.75 - up, 0.0) ** 2
        )

        low_height_penalty = (
            8.0 * max(0.68 - z, 0.0)
            + 25.0 * max(0.58 - z, 0.0) ** 2
        )

        # -------------------------------------------------------------
        # Residual regularization
        # -------------------------------------------------------------

        action_penalty = (
            0.010
            * float(
                np.mean(action ** 2)
            )
        )

        smoothness_penalty = (
            0.020
            * float(
                np.mean(
                    (
                        action
                        - self.previous_action
                    ) ** 2
                )
            )
        )

        # Reward stable survival during the walking transition.
        survival_bonus = 0.0

        if (
            self.episode_step
            >= self.stand_frames
            and up > 0.88
            and z > 0.68
        ):
            survival_bonus = 0.35


        reward = (
            0.20
            + 3.50 * upright_reward
            + 1.50 * height_reward
            + 1.00 * velocity_reward
            + survival_bonus
            - line_penalty
            - yaw_penalty
            - lateral_velocity_penalty
            - angular_velocity_penalty
            - tilt_penalty
            - low_height_penalty
            - action_penalty
            - smoothness_penalty
        )


        self.last_reward_terms = {
            "upright_reward":
                float(upright_reward),

            "height_reward":
                float(height_reward),

            "velocity_reward":
                float(velocity_reward),

            "line_penalty":
                float(line_penalty),

            "yaw_penalty":
                float(yaw_penalty),

            "lateral_velocity_penalty":
                float(lateral_velocity_penalty),

            "angular_velocity_penalty":
                float(angular_velocity_penalty),

            "tilt_penalty":
                float(tilt_penalty),

            "low_height_penalty":
                float(low_height_penalty),

            "survival_bonus":
                float(survival_bonus),
        }

        return float(reward)


    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        nominal, gait_frame, alpha = (
            self._nominal_target()
        )

        # V2:
        # Stabilization authority is independent of the slow BC blend.
        # This lets PPO intervene before the known collapse around step 95.

        if self.episode_step < self.stand_frames:

            residual_alpha = 0.0

        else:

            residual_step = (
                self.episode_step
                - self.stand_frames
            )

            residual_alpha = self._smoothstep(
                residual_step
                / max(
                    self.residual_ramp_frames,
                    1,
                )
            )

        residual = (
            self.residual_scale
            * residual_alpha
            * self.residual_joint_scale
            * action
        )

        requested = (
            nominal
            + residual
        )

        targets = (
            self.target_smoothing
            * self.previous_targets
            + (
                1.0
                - self.target_smoothing
            )
            * requested
        )

        for i, item in enumerate(
            self.joint_info
        ):

            self.data.ctrl[item["aid"]] = (
                self._clip_ctrl(
                    item["aid"],
                    targets[i],
                )
            )

        for aid, target in self.upper_body:

            self.data.ctrl[aid] = (
                self._clip_ctrl(
                    aid,
                    target,
                )
            )

        self.previous_targets = (
            targets.copy()
        )

        for _ in range(self.frame_skip):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        reward = self._reward(
            action
        )

        self.episode_step += 1

        up = self._up_z()

        z = float(
            self.data.qpos[2]
        )

        y = float(
            self.data.qpos[1]
        )

        yaw = self._yaw(
            self.data.qpos[3:7]
        )

        terminated = bool(
            z < 0.45
            or up < 0.45
            or abs(y) > 0.60
            or abs(yaw) > math.radians(135.0)
        )

        truncated = bool(
            self.episode_step
            >= self.max_episode_steps
        )

        # Strong explicit consequence for physical collapse.
        if terminated:
            reward -= 35.0

        # First major training milestone:
        # survive through the complete stand -> BC transition.
        full_bc_frame = (
            self.stand_frames
            + self.transition_frames
        )

        if (
            self.episode_step == full_bc_frame
            and not terminated
        ):
            reward += 20.0

        left_contact = self._foot_contact(
            self.left_sole
        )

        right_contact = self._foot_contact(
            self.right_sole
        )

        info = {
            "x": float(self.data.qpos[0]),
            "y": y,
            "z": z,
            "vx": float(self.data.qvel[0]),
            "vy": float(self.data.qvel[1]),
            "yaw_deg": float(
                math.degrees(yaw)
            ),
            "up_z": float(up),
            "gait_frame": int(gait_frame),
            "bc_alpha": float(alpha),
            "residual_alpha": float(residual_alpha),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "left_true_clearance":
                self._sole_clearance(
                    self.left_sole
                ),
            "right_true_clearance":
                self._sole_clearance(
                    self.right_sole
                ),
        }

        self.previous_action = (
            action.copy()
        )

        return (
            self._build_obs(),
            reward,
            terminated,
            truncated,
            info,
        )


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[:] = (
            self.stand_qpos
        )

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.episode_step = 0

        self.previous_action[:] = 0.0

        stand = np.array(
            [
                self.stand_qpos[item["qadr"]]
                for item in self.joint_info
            ],
            dtype=np.float64,
        )

        self.previous_targets = (
            stand.copy()
        )

        for i, item in enumerate(
            self.joint_info
        ):

            self.data.ctrl[item["aid"]] = (
                self._clip_ctrl(
                    item["aid"],
                    stand[i],
                )
            )

        for aid, target in self.upper_body:

            self.data.ctrl[aid] = (
                self._clip_ctrl(
                    aid,
                    target,
                )
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.floor_z = float(
            self.data.geom_xpos[
                self.floor_gid
            ][2]
        )

        return self._build_obs(), {}


    def close(self):
        pass
