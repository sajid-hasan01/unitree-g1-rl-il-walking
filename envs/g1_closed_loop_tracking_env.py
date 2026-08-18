from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1ClosedLoopTrackingEnv(gym.Env):

    metadata = {
        "render_modes": []
    }


    def __init__(
        self,

        dataset_path=
            "datasets/processed/"
            "g1_amass_walking_tracking_50hz.npz",

        model_path=
            "third_party/mujoco_menagerie/"
            "unitree_g1/scene.xml",

        control_hz=50.0,

        target_velocity=-0.18,

        fixed_start_frame=None,

        random_reference_start=True,

        reset_position_noise=0.015,

        reset_velocity_noise=0.05,

        target_smoothing=0.20,

        max_episode_steps=400,
    ):

        super().__init__()


        root = Path(
            __file__
        ).resolve().parents[1]


        self.dataset_path = (
            root
            / dataset_path
        )

        self.model_path = (
            root
            / model_path
        )


        if not self.dataset_path.exists():

            raise FileNotFoundError(
                self.dataset_path
            )


        if not self.model_path.exists():

            raise FileNotFoundError(
                self.model_path
            )


        # =========================================================
        # REFERENCE
        # =========================================================

        d = np.load(
            self.dataset_path,
            allow_pickle=True,
        )


        self.ref_q = np.asarray(
            d["joint_pos_15"],
            dtype=np.float32,
        )

        self.ref_qd = np.asarray(
            d["joint_vel_15"],
            dtype=np.float32,
        )


        self.ref_root_pos = np.asarray(
            d["root_positions"],
            dtype=np.float32,
        )


        self.ref_root_vel = np.asarray(
            d["root_velocity"],
            dtype=np.float32,
        )


        self.ref_contact = np.asarray(
            d["contact_mask"],
            dtype=np.float32,
        )


        if "has_contact_mask" in d:

            self.has_reference_contact = bool(
                np.asarray(
                    d["has_contact_mask"]
                ).reshape(-1)[0]
            )

        else:

            # Backward-compatible check.
            self.has_reference_contact = bool(
                np.any(
                    self.ref_contact > 0.5
                )
            )


        self.joint_names = [
            str(x)
            for x in d[
                "controlled_joint_names"
            ]
        ]


        self.reference_fps = float(
            np.asarray(
                d["fps"]
            ).reshape(-1)[0]
        )


        self.num_frames = int(
            self.ref_q.shape[0]
        )


        if self.ref_q.shape[1] != 15:

            raise RuntimeError(
                "Expected 15-DOF reference."
            )


        if len(
            self.joint_names
        ) != 15:

            raise RuntimeError(
                "Expected 15 controlled joints."
            )


        # =========================================================
        # MUJOCO
        # =========================================================

        self.model = (
            mujoco.MjModel.from_xml_path(
                str(
                    self.model_path
                )
            )
        )

        self.data = mujoco.MjData(
            self.model
        )


        if self.model.nkey > 0:

            self.stand_qpos = (
                self.model.key_qpos[
                    0
                ].copy()
            )

        else:

            self.stand_qpos = (
                np.zeros(
                    self.model.nq,
                    dtype=np.float64,
                )
            )

            self.stand_qpos[2] = 0.79
            self.stand_qpos[3] = 1.0


        # =========================================================
        # CONTROLLED JOINTS / ACTUATORS
        # =========================================================

        self.joint_info = []


        for name in self.joint_names:

            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )


            if jid < 0:

                raise RuntimeError(
                    f"Joint not found: {name}"
                )


            aid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )


            if aid < 0:

                for candidate in range(
                    self.model.nu
                ):

                    if int(
                        self.model.actuator_trnid[
                            candidate,
                            0,
                        ]
                    ) == jid:

                        aid = candidate
                        break


            if aid < 0:

                raise RuntimeError(
                    f"Actuator not found: {name}"
                )


            self.joint_info.append(
                {
                    "name":
                        name,

                    "jid":
                        int(jid),

                    "aid":
                        int(aid),

                    "qadr":
                        int(
                            self.model.jnt_qposadr[
                                jid
                            ]
                        ),

                    "vadr":
                        int(
                            self.model.jnt_dofadr[
                                jid
                            ]
                        ),
                }
            )


        controlled_aids = {
            x["aid"]
            for x in self.joint_info
        }


        # Hold the remaining upper-body actuators
        # at the model's standing keyframe.
        self.upper_body = []


        for aid in range(
            self.model.nu
        ):

            if aid in controlled_aids:
                continue


            jid = int(
                self.model.actuator_trnid[
                    aid,
                    0,
                ]
            )


            if (
                jid < 0
                or jid
                >= self.model.njnt
            ):

                continue


            qadr = int(
                self.model.jnt_qposadr[
                    jid
                ]
            )


            self.upper_body.append(
                (
                    aid,
                    float(
                        self.stand_qpos[
                            qadr
                        ]
                    ),
                )
            )


        # =========================================================
        # BODY / FOOT CONTACT HELPERS
        # =========================================================

        self.pelvis_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )


        self.left_site = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "left_foot",
            )
        )


        self.right_site = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "right_foot",
            )
        )


        self.left_body = int(
            self.model.site_bodyid[
                self.left_site
            ]
        )

        self.right_body = int(
            self.model.site_bodyid[
                self.right_site
            ]
        )


        self.left_sole = (
            self._find_sole(
                self.left_body
            )
        )

        self.right_sole = (
            self._find_sole(
                self.right_body
            )
        )


        self.floor_gid = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
        )


        # =========================================================
        # TIMING
        # =========================================================

        self.control_hz = float(
            control_hz
        )

        self.control_dt = (
            1.0
            / self.control_hz
        )

        self.sim_dt = float(
            self.model.opt.timestep
        )


        self.frame_skip = max(
            1,

            int(
                round(
                    self.control_dt
                    / self.sim_dt
                )
            ),
        )


        self.target_velocity = float(
            target_velocity
        )


        self.fixed_start_frame = (
            None
            if fixed_start_frame is None
            else int(
                fixed_start_frame
            )
        )


        self.random_reference_start = bool(
            random_reference_start
        )


        self.reset_position_noise = float(
            reset_position_noise
        )

        self.reset_velocity_noise = float(
            reset_velocity_noise
        )


        self.target_smoothing = float(
            np.clip(
                target_smoothing,
                0.0,
                0.95,
            )
        )


        self.max_episode_steps = int(
            max_episode_steps
        )


        # =========================================================
        # REFERENCE ROOT HEIGHT
        #
        # Keep only the vertical oscillation around the G1
        # standing height; do not teleport reference X/Y.
        # =========================================================

        z_signal = (
            self.ref_root_pos[:, 2]
        )


        if (
            np.all(
                np.isfinite(
                    z_signal
                )
            )
            and np.ptp(
                z_signal
            ) < 0.50
        ):

            self.ref_z_center = float(
                np.median(
                    z_signal
                )
            )

        else:

            self.ref_z_center = 0.0


        # =========================================================
        # ACTION SCALE
        #
        # Compute actuator-specific scale from:
        #
        #     0.25 * effort_limit / stiffness
        #
        # when the MJCF exposes both quantities.
        #
        # Robust fallbacks are used otherwise.
        # =========================================================

        self.action_scale = (
            self._build_action_scale()
        )


        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )


        # =========================================================
        # OBSERVATION = 126
        #
        # reference q             15
        # reference qdot          15
        # actual q                15
        # actual qdot             15
        # q error                 15
        # qdot error              15
        # pelvis quaternion        4
        # root velocity            6
        # reference root vel       3
        # actual contact           2
        # expected contact         2
        # phase                    2
        # line Y / yaw             2
        # previous action         15
        # ---------------------------
        # total                  126
        # =========================================================

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(126,),
                dtype=np.float32,
            )
        )


        self.motion_frame = 0.0

        self.episode_step = 0

        self.previous_action = (
            np.zeros(
                15,
                dtype=np.float32,
            )
        )

        self.previous_targets = (
            np.zeros(
                15,
                dtype=np.float64,
            )
        )

        self.floor_z = 0.0


    # =============================================================
    # HELPERS
    # =============================================================

    def _clip_ctrl(
        self,
        aid,
        value,
    ):

        lo, hi = (
            self.model.actuator_ctrlrange[
                aid
            ]
        )

        return float(
            np.clip(
                value,
                lo,
                hi,
            )
        )


    def _build_action_scale(
        self,
    ):

        scales = np.zeros(
            15,
            dtype=np.float32,
        )


        for i, item in enumerate(
            self.joint_info
        ):

            aid = item["aid"]


            kp = abs(
                float(
                    self.model.actuator_gainprm[
                        aid,
                        0,
                    ]
                )
            )


            effort = 0.0


            try:

                fr = (
                    self.model.actuator_forcerange[
                        aid
                    ]
                )

                effort = max(
                    abs(
                        float(
                            fr[0]
                        )
                    ),
                    abs(
                        float(
                            fr[1]
                        )
                    ),
                )

            except Exception:

                effort = 0.0


            if (
                kp > 1e-6
                and effort > 1e-6
                and np.isfinite(
                    kp
                )
                and np.isfinite(
                    effort
                )
            ):

                scale = (
                    0.25
                    * effort
                    / kp
                )

            else:

                name = item["name"]

                if (
                    "knee" in name
                    or "hip_pitch" in name
                ):

                    scale = 0.10

                elif (
                    "hip_roll" in name
                    or "hip_yaw" in name
                ):

                    scale = 0.07

                elif "ankle" in name:

                    scale = 0.055

                else:

                    scale = 0.05


            # Safety/robustness clamp for the first tracker.
            scales[i] = float(
                np.clip(
                    scale,
                    0.025,
                    0.14,
                )
            )


        return scales


    def _find_sole(
        self,
        body_id,
    ):

        ids = []


        for gid in range(
            self.model.ngeom
        ):

            if int(
                self.model.geom_bodyid[
                    gid
                ]
            ) != int(
                body_id
            ):

                continue


            if int(
                self.model.geom_type[
                    gid
                ]
            ) != int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            ):

                continue


            if float(
                self.model.geom_size[
                    gid
                ][0]
            ) <= 0.010:

                ids.append(
                    gid
                )


        return sorted(
            ids
        )


    @staticmethod
    def _yaw(q):

        w, x, y, z = [
            float(v)
            for v in q
        ]


        return math.atan2(
            2.0
            * (
                w * z
                + x * y
            ),

            1.0
            - 2.0
            * (
                y * y
                + z * z
            ),
        )


    def _up_z(self):

        R = np.asarray(
            self.data.xmat[
                self.pelvis_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        )

        return float(
            R[2, 2]
        )


    def _joint_positions(
        self,
    ):

        return np.array(
            [
                self.data.qpos[
                    item["qadr"]
                ]
                for item
                in self.joint_info
            ],
            dtype=np.float32,
        )


    def _joint_velocities(
        self,
    ):

        return np.array(
            [
                self.data.qvel[
                    item["vadr"]
                ]
                for item
                in self.joint_info
            ],
            dtype=np.float32,
        )


    def _foot_contact(
        self,
        sole,
    ):

        geom_set = {
            int(x)
            for x in sole
        }


        for i in range(
            self.data.ncon
        ):

            c = self.data.contact[
                i
            ]

            a = int(
                c.geom1
            )

            b = int(
                c.geom2
            )


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


    def _reference_index(
        self,
    ):

        return int(
            np.clip(
                round(
                    self.motion_frame
                ),
                0,
                self.num_frames - 1,
            )
        )


    def _reference(
        self,
    ):

        idx = self._reference_index()

        return (
            self.ref_q[idx],
            self.ref_qd[idx],
            self.ref_root_vel[idx],
            self.ref_contact[idx],
            idx,
        )


    def _target_height(
        self,
        idx,
    ):

        dz = (
            float(
                self.ref_root_pos[
                    idx,
                    2,
                ]
            )
            - self.ref_z_center
        )


        return float(
            np.clip(
                self.stand_qpos[2]
                + dz,
                self.stand_qpos[2]
                - 0.10,
                self.stand_qpos[2]
                + 0.10,
            )
        )


    def _phase(
        self,
        idx,
    ):

        p = (
            idx
            / max(
                self.num_frames - 1,
                1,
            )
        )

        a = (
            2.0
            * np.pi
            * p
        )


        return np.array(
            [
                np.sin(a),
                np.cos(a),
            ],
            dtype=np.float32,
        )


    # =============================================================
    # OBSERVATION
    # =============================================================

    def _build_obs(
        self,
    ):

        (
            ref_q,
            ref_qd,
            ref_root_vel,
            expected_contact,
            idx,
        ) = self._reference()


        q = self._joint_positions()

        qd = self._joint_velocities()


        q_error = (
            ref_q - q
        )

        qd_error = (
            ref_qd - qd
        )


        left_contact = float(
            self._foot_contact(
                self.left_sole
            )
        )

        right_contact = float(
            self._foot_contact(
                self.right_sole
            )
        )


        yaw = self._yaw(
            self.data.qpos[
                3:7
            ]
        )


        obs = np.concatenate(
            [
                ref_q.astype(
                    np.float32
                ),

                (
                    0.10
                    * ref_qd
                ).astype(
                    np.float32
                ),

                q.astype(
                    np.float32
                ),

                (
                    0.10
                    * qd
                ).astype(
                    np.float32
                ),

                q_error.astype(
                    np.float32
                ),

                (
                    0.10
                    * qd_error
                ).astype(
                    np.float32
                ),

                self.data.qpos[
                    3:7
                ].astype(
                    np.float32
                ),

                (
                    0.10
                    * self.data.qvel[
                        0:6
                    ]
                ).astype(
                    np.float32
                ),

                (
                    0.10
                    * ref_root_vel
                ).astype(
                    np.float32
                ),

                np.array(
                    [
                        left_contact,
                        right_contact,
                    ],
                    dtype=np.float32,
                ),

                expected_contact.astype(
                    np.float32
                ),

                self._phase(
                    idx
                ),

                np.array(
                    [
                        float(
                            self.data.qpos[1]
                        ),
                        float(
                            yaw
                        ),
                    ],
                    dtype=np.float32,
                ),

                self.previous_action.astype(
                    np.float32
                ),
            ]
        )


        if obs.shape != (
            126,
        ):

            raise RuntimeError(
                f"Unexpected obs shape: "
                f"{obs.shape}"
            )


        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


    # =============================================================
    # TRACKING REWARD
    # =============================================================

    def _reward(
        self,
        action,
    ):

        (
            ref_q,
            ref_qd,
            _ref_root_vel,
            expected_contact,
            idx,
        ) = self._reference()


        q = self._joint_positions()

        qd = self._joint_velocities()


        q_error = (
            q - ref_q
        )

        qd_error = (
            qd - ref_qd
        )


        # ---------------------------------------------------------
        # DEEPMIMIC-STYLE TRACKING TERMS
        # ---------------------------------------------------------

        pose_reward = float(
            np.exp(
                -np.mean(
                    (
                        q_error
                        / 0.25
                    ) ** 2
                )
            )
        )


        velocity_reward = float(
            np.exp(
                -np.mean(
                    (
                        qd_error
                        / 2.5
                    ) ** 2
                )
            )
        )


        up = self._up_z()


        upright_reward = float(
            np.exp(
                -(
                    (
                        1.0
                        - up
                    )
                    / 0.16
                ) ** 2
            )
        )


        target_height = (
            self._target_height(
                idx
            )
        )


        height_error = (
            float(
                self.data.qpos[2]
            )
            - target_height
        )


        height_reward = float(
            np.exp(
                -(
                    height_error
                    / 0.08
                ) ** 2
            )
        )


        vx = float(
            self.data.qvel[0]
        )


        forward_reward = float(
            np.exp(
                -(
                    (
                        vx
                        - self.target_velocity
                    )
                    / 0.25
                ) ** 2
            )
        )


        y = float(
            self.data.qpos[1]
        )


        yaw = self._yaw(
            self.data.qpos[
                3:7
            ]
        )


        line_reward = float(
            np.exp(
                -(
                    y
                    / 0.12
                ) ** 2
            )
        )


        heading_reward = float(
            np.exp(
                -(
                    yaw
                    / 0.40
                ) ** 2
            )
        )


        actual_contact = np.array(
            [
                float(
                    self._foot_contact(
                        self.left_sole
                    )
                ),

                float(
                    self._foot_contact(
                        self.right_sole
                    )
                ),
            ],
            dtype=np.float32,
        )


        contact_match = float(
            np.mean(
                (
                    actual_contact > 0.5
                )
                ==
                (
                    expected_contact > 0.5
                )
            )
        )


        if not self.has_reference_contact:

            # The source AMASS reference contains no
            # ground-truth contact labels.
            #
            # Do NOT interpret the zero placeholder array
            # as "both feet should be airborne".
            contact_match = 0.0


        action_penalty = (
            0.005
            * float(
                np.mean(
                    action ** 2
                )
            )
        )


        action_rate_penalty = (
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


        reward = (
            1.70
            * pose_reward

            + 0.65
            * velocity_reward

            + 1.20
            * upright_reward

            + 0.55
            * height_reward

            + 0.45
            * forward_reward

            + 0.30
            * line_reward

            + 0.30
            * heading_reward

            + (
                0.55
                * contact_match
                if self.has_reference_contact
                else 0.0
            )

            - action_penalty

            - action_rate_penalty
        )


        terms = {
            "pose_reward":
                pose_reward,

            "joint_velocity_reward":
                velocity_reward,

            "upright_reward":
                upright_reward,

            "height_reward":
                height_reward,

            "forward_reward":
                forward_reward,

            "line_reward":
                line_reward,

            "heading_reward":
                heading_reward,

            "contact_match":
                contact_match,

            "q_error_rms":
                float(
                    np.sqrt(
                        np.mean(
                            q_error ** 2
                        )
                    )
                ),

            "qd_error_rms":
                float(
                    np.sqrt(
                        np.mean(
                            qd_error ** 2
                        )
                    )
                ),
        }


        return (
            float(reward),
            terms,
        )


    # =============================================================
    # STEP
    # =============================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            15,
        )


        action = np.clip(
            action,
            -1.0,
            1.0,
        )


        ref_q, _, _, _, _ = (
            self._reference()
        )


        # PPO is a small closed-loop correction
        # around the ACTUAL reference trajectory,
        # not around a BC approximation.
        requested = (
            ref_q.astype(
                np.float64
            )
            + self.action_scale.astype(
                np.float64
            )
            * action.astype(
                np.float64
            )
        )


        targets = (
            self.target_smoothing
            * self.previous_targets
            +
            (
                1.0
                - self.target_smoothing
            )
            * requested
        )


        for i, item in enumerate(
            self.joint_info
        ):

            self.data.ctrl[
                item["aid"]
            ] = self._clip_ctrl(
                item["aid"],
                targets[i],
            )


        for aid, target in (
            self.upper_body
        ):

            self.data.ctrl[
                aid
            ] = self._clip_ctrl(
                aid,
                target,
            )


        self.previous_targets = (
            targets.copy()
        )


        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )


        # Advance reference AFTER the physical control interval.
        self.motion_frame += 1.0

        self.episode_step += 1


        reward, terms = (
            self._reward(
                action
            )
        )


        up = self._up_z()

        z = float(
            self.data.qpos[2]
        )

        y = float(
            self.data.qpos[1]
        )

        yaw = self._yaw(
            self.data.qpos[
                3:7
            ]
        )


        terminated = bool(
            z < 0.45
            or up < 0.45
            or abs(y) > 0.60
            or abs(yaw)
            > math.radians(
                135.0
            )
        )


        completed = bool(
            self.motion_frame
            >= (
                self.num_frames
                - 1
            )
        )


        truncated = bool(
            completed
            or self.episode_step
            >= self.max_episode_steps
        )


        if terminated:

            reward -= 20.0


        if (
            completed
            and not terminated
            and up > 0.75
        ):

            reward += 15.0


        left_contact = (
            self._foot_contact(
                self.left_sole
            )
        )

        right_contact = (
            self._foot_contact(
                self.right_sole
            )
        )


        info = {
            "x":
                float(
                    self.data.qpos[0]
                ),

            "y":
                y,

            "z":
                z,

            "vx":
                float(
                    self.data.qvel[0]
                ),

            "vy":
                float(
                    self.data.qvel[1]
                ),

            "yaw_deg":
                float(
                    math.degrees(
                        yaw
                    )
                ),

            "up_z":
                float(
                    up
                ),

            "motion_frame":
                float(
                    self.motion_frame
                ),

            "reference_index":
                int(
                    self._reference_index()
                ),

            "completed":
                completed,

            "left_contact":
                bool(
                    left_contact
                ),

            "right_contact":
                bool(
                    right_contact
                ),

            "action_max":
                float(
                    np.max(
                        np.abs(
                            action
                        )
                    )
                ),

            "target_offset_max":
                float(
                    np.max(
                        np.abs(
                            self.action_scale
                            * action
                        )
                    )
                ),
        }


        info.update(
            terms
        )


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


    # =============================================================
    # RESET / REFERENCE-STATE INITIALIZATION
    # =============================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )


        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        self.data.qpos[:] = (
            self.stand_qpos
        )

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0


        # ---------------------------------------------------------
        # REFERENCE STATE INITIALIZATION
        # ---------------------------------------------------------

        if (
            self.fixed_start_frame
            is not None
        ):

            frame = int(
                np.clip(
                    self.fixed_start_frame,
                    0,
                    self.num_frames - 2,
                )
            )


        elif self.random_reference_start:

            # Avoid very short episodes at the final edge.
            high = max(
                2,
                self.num_frames - 60,
            )


            frame = int(
                self.np_random.integers(
                    0,
                    high,
                )
            )


        else:

            frame = 0


        self.motion_frame = float(
            frame
        )


        qref = (
            self.ref_q[
                frame
            ].copy()
        )

        qdref = (
            self.ref_qd[
                frame
            ].copy()
        )


        if (
            self.reset_position_noise
            > 0.0
        ):

            qref += (
                self.np_random.normal(
                    0.0,
                    self.reset_position_noise,
                    size=15,
                ).astype(
                    np.float32
                )
            )


        if (
            self.reset_velocity_noise
            > 0.0
        ):

            qdref += (
                self.np_random.normal(
                    0.0,
                    self.reset_velocity_noise,
                    size=15,
                ).astype(
                    np.float32
                )
            )


        for i, item in enumerate(
            self.joint_info
        ):

            self.data.qpos[
                item["qadr"]
            ] = float(
                qref[i]
            )


            self.data.qvel[
                item["vadr"]
            ] = float(
                qdref[i]
            )


        # Keep root upright and centered.
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0


        self.data.qpos[2] = (
            self._target_height(
                frame
            )
        )


        # Preserve standing-keyframe quaternion.
        self.data.qpos[
            3:7
        ] = (
            self.stand_qpos[
                3:7
            ]
        )


        # Give RSI a physically useful forward state
        # instead of starting every mid-gait pose at rest.
        self.data.qvel[0] = (
            self.target_velocity
        )

        self.data.qvel[1] = 0.0


        if (
            self.ref_root_vel.shape[
                1
            ] >= 3
        ):

            self.data.qvel[2] = float(
                np.clip(
                    self.ref_root_vel[
                        frame,
                        2,
                    ],
                    -0.5,
                    0.5,
                )
            )


        self.episode_step = 0

        self.previous_action[:] = (
            0.0
        )


        self.previous_targets = (
            self.ref_q[
                frame
            ].astype(
                np.float64
            ).copy()
        )


        for i, item in enumerate(
            self.joint_info
        ):

            self.data.ctrl[
                item["aid"]
            ] = self._clip_ctrl(
                item["aid"],
                self.previous_targets[
                    i
                ],
            )


        for aid, target in (
            self.upper_body
        ):

            self.data.ctrl[
                aid
            ] = self._clip_ctrl(
                aid,
                target,
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


        info = {
            "reset_reference_frame":
                frame,

            "control_hz":
                self.control_hz,

            "frame_skip":
                self.frame_skip,
        }


        return (
            self._build_obs(),
            info,
        )


    def close(
        self,
    ):

        pass
