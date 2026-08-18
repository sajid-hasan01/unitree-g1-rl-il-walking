from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_g1"
    / "scene.xml"
)

DEFAULT_REFERENCE_PATH = (
    ROOT
    / "datasets"
    / "validated_29dof_walks"
    / "medium_02_50hz_grounded.npz"
)


# Official Unitree full-G1 tracking configuration uses a
# representative whole-body set spanning pelvis, legs, torso,
# shoulders, elbows and wrists.
TRACK_BODY_NAMES = (
    "pelvis",

    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",

    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",

    "torso_link",

    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",

    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


# ================================================================
# QUATERNION HELPERS
# wxyz convention.
# ================================================================

def quat_normalize(
    q: np.ndarray,
) -> np.ndarray:

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return (
        q
        / max(
            np.linalg.norm(q),
            1e-12,
        )
    )


def quat_conjugate(
    q: np.ndarray,
) -> np.ndarray:

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    return np.asarray(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ],
        dtype=np.float64,
    )


def quat_multiply(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.asarray(
        [
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ],
        dtype=np.float64,
    )


def quat_error_vector(
    reference: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:

    q_ref = quat_normalize(
        reference
    )

    q_act = quat_normalize(
        actual
    )

    q_err = quat_multiply(
        quat_conjugate(
            q_ref
        ),
        q_act,
    )

    # q and -q represent the same rotation.
    if q_err[0] < 0.0:
        q_err = -q_err

    # Small-angle-friendly 3-vector.
    return (
        2.0
        * q_err[1:4]
    )


def quat_angle(
    reference: np.ndarray,
    actual: np.ndarray,
) -> float:

    q_ref = quat_normalize(
        reference
    )

    q_act = quat_normalize(
        actual
    )

    dot = abs(
        float(
            np.dot(
                q_ref,
                q_act,
            )
        )
    )

    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )

    return (
        2.0
        * math.acos(dot)
    )


def quat_relative(
    parent: np.ndarray,
    child: np.ndarray,
) -> np.ndarray:

    return quat_normalize(
        quat_multiply(
            quat_conjugate(
                quat_normalize(
                    parent
                )
            ),
            quat_normalize(
                child
            ),
        )
    )


# ================================================================
# ENVIRONMENT
# ================================================================

class G129DofTrackingRestartV1(
    gym.Env
):

    metadata = {
        "render_modes": [],
        "render_fps": 50,
    }


    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        reference_path: str | Path = DEFAULT_REFERENCE_PATH,
        rsi: bool = True,
        reset_joint_noise: float = 0.0,
        reset_velocity_noise: float = 0.0,
        min_remaining_frames: int = 60,
    ):

        super().__init__()


        self.model_path = Path(
            model_path
        )

        self.reference_path = Path(
            reference_path
        )


        if not self.model_path.exists():

            raise FileNotFoundError(
                self.model_path
            )


        if not self.reference_path.exists():

            raise FileNotFoundError(
                self.reference_path
            )


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


        self.rsi = bool(
            rsi
        )

        self.reset_joint_noise = float(
            reset_joint_noise
        )

        self.reset_velocity_noise = float(
            reset_velocity_noise
        )

        self.min_remaining_frames = int(
            min_remaining_frames
        )


        # --------------------------------------------------------
        # REFERENCE
        # --------------------------------------------------------

        ref = np.load(
            self.reference_path,
            allow_pickle=True,
        )


        required_keys = (
            "fps",
            "dof_names",
            "joint_pos_29",
            "joint_vel_29",
            "full_qpos",
            "full_qvel",
            "root_positions",
            "root_quat_wxyz",
            "left_foot_pos",
            "right_foot_pos",
            "contact_mask",
            "support_mask",
        )


        for key in required_keys:

            if key not in ref.files:

                raise RuntimeError(
                    f"Reference is missing key: {key}"
                )


        self.reference_fps = float(
            np.asarray(
                ref["fps"]
            ).reshape(-1)[0]
        )


        if abs(
            self.reference_fps
            - 50.0
        ) > 1e-6:

            raise RuntimeError(
                "Restart V1 expects a 50-Hz reference."
            )


        self.ref_names = [
            str(x)
            for x in ref[
                "dof_names"
            ]
        ]


        self.ref_q = np.asarray(
            ref["joint_pos_29"],
            dtype=np.float64,
        )

        self.ref_qd = np.asarray(
            ref["joint_vel_29"],
            dtype=np.float64,
        )

        self.ref_full_qpos = np.asarray(
            ref["full_qpos"],
            dtype=np.float64,
        )

        self.ref_full_qvel = np.asarray(
            ref["full_qvel"],
            dtype=np.float64,
        )

        self.ref_root_pos = np.asarray(
            ref["root_positions"],
            dtype=np.float64,
        )

        self.ref_root_quat = np.asarray(
            ref["root_quat_wxyz"],
            dtype=np.float64,
        )

        self.ref_left_foot_pos = np.asarray(
            ref["left_foot_pos"],
            dtype=np.float64,
        )

        self.ref_right_foot_pos = np.asarray(
            ref["right_foot_pos"],
            dtype=np.float64,
        )

        self.ref_contact = np.asarray(
            ref["contact_mask"],
            dtype=np.float64,
        )

        self.ref_support = np.asarray(
            ref["support_mask"],
            dtype=np.float64,
        )


        self.num_frames = int(
            self.ref_q.shape[0]
        )


        if self.ref_q.shape != (
            self.num_frames,
            29,
        ):

            raise RuntimeError(
                f"Unexpected joint reference shape: "
                f"{self.ref_q.shape}"
            )


        # --------------------------------------------------------
        # CONTROL RATE
        # --------------------------------------------------------

        self.sim_dt = float(
            self.model.opt.timestep
        )


        self.frame_skip = int(
            round(
                1.0
                / (
                    self.reference_fps
                    * self.sim_dt
                )
            )
        )


        if self.frame_skip <= 0:

            raise RuntimeError(
                "Invalid frame skip."
            )


        actual_control_hz = (
            1.0
            / (
                self.frame_skip
                * self.sim_dt
            )
        )


        if abs(
            actual_control_hz
            - self.reference_fps
        ) > 1e-4:

            raise RuntimeError(
                "Simulation timestep cannot represent "
                "the 50-Hz control rate exactly."
            )


        # --------------------------------------------------------
        # JOINT + ACTUATOR MAPPING
        # --------------------------------------------------------

        self.joint_ids = []
        self.actuator_ids = []
        self.qaddrs = []
        self.vaddrs = []

        self.stiffness = []
        self.damping = []
        self.effort_limits = []
        self.action_scale = []


        model_names = []


        for aid in range(
            self.model.nu
        ):

            jid = int(
                self.model.actuator_trnid[
                    aid,
                    0,
                ]
            )


            joint_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    jid,
                )
            )


            model_names.append(
                str(
                    joint_name
                )
            )


        if model_names != self.ref_names:

            raise RuntimeError(
                "Reference joint order does not match "
                "MuJoCo actuator order."
            )


        if len(
            model_names
        ) != 29:

            raise RuntimeError(
                "Expected exactly 29 actuated joints."
            )


        for aid, name in enumerate(
            model_names
        ):

            jid = int(
                self.model.actuator_trnid[
                    aid,
                    0,
                ]
            )


            qadr = int(
                self.model.jnt_qposadr[
                    jid
                ]
            )

            vadr = int(
                self.model.jnt_dofadr[
                    jid
                ]
            )


            gain = float(
                self.model.actuator_gainprm[
                    aid,
                    0,
                ]
            )

            bias1 = float(
                self.model.actuator_biasprm[
                    aid,
                    1,
                ]
            )

            bias2 = float(
                self.model.actuator_biasprm[
                    aid,
                    2,
                ]
            )


            # Native position-servo audit.
            if gain <= 0.0:

                raise RuntimeError(
                    f"Non-positive servo gain: {name}"
                )


            if abs(
                bias1
                + gain
            ) > 1e-4:

                raise RuntimeError(
                    f"{name} is not the expected "
                    f"native position servo."
                )


            kd = max(
                0.0,
                -bias2,
            )


            lo = float(
                self.model.jnt_actfrcrange[
                    jid,
                    0,
                ]
            )

            hi = float(
                self.model.jnt_actfrcrange[
                    jid,
                    1,
                ]
            )


            effort = max(
                abs(lo),
                abs(hi),
            )


            if effort <= 0.0:

                raise RuntimeError(
                    f"Invalid effort limit: {name}"
                )


            # ----------------------------------------------------
            # Unitree-inspired residual scale:
            #
            # scale = 0.25 * effort_limit / stiffness
            #
            # Derived from THIS compiled local model.
            # ----------------------------------------------------

            scale = (
                0.25
                * effort
                / gain
            )


            self.joint_ids.append(
                jid
            )

            self.actuator_ids.append(
                aid
            )

            self.qaddrs.append(
                qadr
            )

            self.vaddrs.append(
                vadr
            )

            self.stiffness.append(
                gain
            )

            self.damping.append(
                kd
            )

            self.effort_limits.append(
                effort
            )

            self.action_scale.append(
                scale
            )


        self.stiffness = np.asarray(
            self.stiffness,
            dtype=np.float64,
        )

        self.damping = np.asarray(
            self.damping,
            dtype=np.float64,
        )

        self.effort_limits = np.asarray(
            self.effort_limits,
            dtype=np.float64,
        )

        self.action_scale = np.asarray(
            self.action_scale,
            dtype=np.float64,
        )


        # --------------------------------------------------------
        # ROOT / BODY TARGETS
        # --------------------------------------------------------

        self.pelvis_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )


        if self.pelvis_id < 0:

            raise RuntimeError(
                "Pelvis body was not found."
            )


        self.track_body_ids = []


        for name in TRACK_BODY_NAMES:

            bid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )


            if bid < 0:

                raise RuntimeError(
                    f"Tracking body missing: {name}"
                )


            self.track_body_ids.append(
                bid
            )


        # --------------------------------------------------------
        # TRUE FOOT GEOMETRY
        # --------------------------------------------------------

        self.left_foot_site = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "left_foot",
            )
        )

        self.right_foot_site = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "right_foot",
            )
        )

        self.floor_geom = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
        )


        if min(
            self.left_foot_site,
            self.right_foot_site,
            self.floor_geom,
        ) < 0:

            raise RuntimeError(
                "Required floor/foot geometry missing."
            )


        self.left_foot_body = int(
            self.model.site_bodyid[
                self.left_foot_site
            ]
        )

        self.right_foot_body = int(
            self.model.site_bodyid[
                self.right_foot_site
            ]
        )

        self.floor_z = float(
            self.model.geom_pos[
                self.floor_geom,
                2,
            ]
        )


        self.left_sole_geoms = (
            self._find_sole_spheres(
                self.left_foot_body
            )
        )

        self.right_sole_geoms = (
            self._find_sole_spheres(
                self.right_foot_body
            )
        )


        if len(
            self.left_sole_geoms
        ) != 4:

            raise RuntimeError(
                "Expected four left sole spheres."
            )


        if len(
            self.right_sole_geoms
        ) != 4:

            raise RuntimeError(
                "Expected four right sole spheres."
            )


        # --------------------------------------------------------
        # PRECOMPUTE REFERENCE WHOLE-BODY POSE
        # relative to pelvis.
        # --------------------------------------------------------

        (
            self.ref_body_local_pos,
            self.ref_body_local_quat,
        ) = self._precompute_reference_bodies()


        # --------------------------------------------------------
        # GYM SPACES
        # --------------------------------------------------------

        self.num_actions = 29


        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.num_actions,
            ),
            dtype=np.float32,
        )


        self._current_frame = 0

        self.previous_action = np.zeros(
            self.num_actions,
            dtype=np.float64,
        )

        self.last_target = self.ref_q[
            0
        ].copy()


        # Build one observation to determine shape.
        self._set_reference_state(
            0
        )

        observation = self._get_observation()


        self.num_observations = int(
            observation.shape[0]
        )


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.num_observations,
            ),
            dtype=np.float32,
        )


    # ============================================================
    # MODEL HELPERS
    # ============================================================

    def _find_sole_spheres(
        self,
        body_id: int,
    ) -> list[int]:

        result = []


        for gid in range(
            self.model.ngeom
        ):

            if int(
                self.model.geom_bodyid[
                    gid
                ]
            ) != body_id:

                continue


            if int(
                self.model.geom_type[
                    gid
                ]
            ) != int(
                mujoco.mjtGeom.mjGEOM_SPHERE
            ):

                continue


            radius = float(
                self.model.geom_size[
                    gid,
                    0,
                ]
            )


            if radius <= 0.010:

                result.append(
                    gid
                )


        return sorted(
            result
        )


    def _set_reference_state(
        self,
        frame: int,
    ) -> None:

        self.data.qpos[:] = (
            self.ref_full_qpos[
                frame
            ]
        )

        self.data.qvel[:] = (
            self.ref_full_qvel[
                frame
            ]
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def _joint_q(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.data.qpos[
                    qadr
                ]
                for qadr
                in self.qaddrs
            ],
            dtype=np.float64,
        )


    def _joint_qd(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.data.qvel[
                    vadr
                ]
                for vadr
                in self.vaddrs
            ],
            dtype=np.float64,
        )


    def _root_rotation_matrix(
        self,
    ) -> np.ndarray:

        return (
            self.data.xmat[
                self.pelvis_id
            ].reshape(
                3,
                3,
            )
        )


    def _up_z(
        self,
    ) -> float:

        return float(
            self._root_rotation_matrix()[
                2,
                2,
            ]
        )


    def _sole_clearance(
        self,
        geom_ids: list[int],
    ) -> float:

        values = []


        for gid in geom_ids:

            centre_z = float(
                self.data.geom_xpos[
                    gid,
                    2,
                ]
            )

            radius = float(
                self.model.geom_size[
                    gid,
                    0,
                ]
            )


            values.append(
                centre_z
                - radius
                - self.floor_z
            )


        return float(
            min(
                values
            )
        )


    def _actual_contacts(
        self,
    ) -> np.ndarray:

        # Same 15-mm clearance definition used in the
        # validated reference builder.
        return np.asarray(
            [
                self._sole_clearance(
                    self.left_sole_geoms
                ) <= 0.015,

                self._sole_clearance(
                    self.right_sole_geoms
                ) <= 0.015,
            ],
            dtype=np.float64,
        )


    # ============================================================
    # REFERENCE BODY CACHE
    # ============================================================

    def _precompute_reference_bodies(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        count = len(
            self.track_body_ids
        )


        local_pos = np.zeros(
            (
                self.num_frames,
                count,
                3,
            ),
            dtype=np.float64,
        )

        local_quat = np.zeros(
            (
                self.num_frames,
                count,
                4,
            ),
            dtype=np.float64,
        )


        cache_data = mujoco.MjData(
            self.model
        )


        for frame in range(
            self.num_frames
        ):

            cache_data.qpos[:] = (
                self.ref_full_qpos[
                    frame
                ]
            )

            cache_data.qvel[:] = (
                self.ref_full_qvel[
                    frame
                ]
            )


            mujoco.mj_forward(
                self.model,
                cache_data,
            )


            pelvis_pos = (
                cache_data.xpos[
                    self.pelvis_id
                ].copy()
            )

            pelvis_quat = (
                cache_data.xquat[
                    self.pelvis_id
                ].copy()
            )

            pelvis_R = (
                cache_data.xmat[
                    self.pelvis_id
                ].reshape(
                    3,
                    3,
                )
            )


            for k, bid in enumerate(
                self.track_body_ids
            ):

                world_delta = (
                    cache_data.xpos[
                        bid
                    ]
                    - pelvis_pos
                )


                local_pos[
                    frame,
                    k,
                ] = (
                    pelvis_R.T
                    @ world_delta
                )


                local_quat[
                    frame,
                    k,
                ] = quat_relative(
                    pelvis_quat,
                    cache_data.xquat[
                        bid
                    ],
                )


        return (
            local_pos,
            local_quat,
        )


    # ============================================================
    # WHOLE-BODY ERROR
    # ============================================================

    def _body_pose_errors(
        self,
        frame: int,
    ) -> tuple[
        float,
        float,
    ]:

        pelvis_pos = self.data.xpos[
            self.pelvis_id
        ]

        pelvis_quat = self.data.xquat[
            self.pelvis_id
        ]

        R = self._root_rotation_matrix()


        pos_errors = []
        rot_errors = []


        # Skip pelvis entry itself because root pose is
        # rewarded separately.
        for k, bid in enumerate(
            self.track_body_ids
        ):

            if bid == self.pelvis_id:
                continue


            actual_local_pos = (
                R.T
                @ (
                    self.data.xpos[
                        bid
                    ]
                    - pelvis_pos
                )
            )


            pos_errors.append(
                np.linalg.norm(
                    actual_local_pos
                    - self.ref_body_local_pos[
                        frame,
                        k,
                    ]
                )
            )


            actual_local_quat = (
                quat_relative(
                    pelvis_quat,
                    self.data.xquat[
                        bid
                    ],
                )
            )


            rot_errors.append(
                quat_angle(
                    self.ref_body_local_quat[
                        frame,
                        k,
                    ],
                    actual_local_quat,
                )
            )


        return (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            pos_errors
                        )
                    )
                )
            ),

            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            rot_errors
                        )
                    )
                )
            ),
        )


    # ============================================================
    # ACTION TARGET
    # ============================================================

    def _action_target(
        self,
        action: np.ndarray,
        frame: int,
    ) -> np.ndarray:

        action = np.clip(
            np.asarray(
                action,
                dtype=np.float64,
            ),
            -1.0,
            1.0,
        )


        target = (
            self.ref_q[
                frame
            ]
            +
            self.action_scale
            * action
        )


        # Respect joint limits.
        for j, jid in enumerate(
            self.joint_ids
        ):

            if self.model.jnt_limited[
                jid
            ]:

                lo, hi = (
                    self.model.jnt_range[
                        jid
                    ]
                )

                target[j] = float(
                    np.clip(
                        target[j],
                        lo,
                        hi,
                    )
                )


            aid = self.actuator_ids[
                j
            ]


            if self.model.actuator_ctrllimited[
                aid
            ]:

                lo, hi = (
                    self.model.actuator_ctrlrange[
                        aid
                    ]
                )

                target[j] = float(
                    np.clip(
                        target[j],
                        lo,
                        hi,
                    )
                )


        return target


    def _apply_target(
        self,
        target: np.ndarray,
    ) -> None:

        for j, aid in enumerate(
            self.actuator_ids
        ):

            self.data.ctrl[
                aid
            ] = target[
                j
            ]


    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(
        self,
    ) -> np.ndarray:

        frame = int(
            self._current_frame
        )


        q = self._joint_q()

        qd = self._joint_qd()


        q_ref = self.ref_q[
            frame
        ]

        qd_ref = self.ref_qd[
            frame
        ]


        q_error = (
            q
            - q_ref
        )

        qd_error = (
            qd
            - qd_ref
        )


        root_pos_error_world = (
            self.data.qpos[
                0:3
            ]
            - self.ref_root_pos[
                frame
            ]
        )


        root_quat = self.data.qpos[
            3:7
        ]


        root_orientation_error = (
            quat_error_vector(
                self.ref_root_quat[
                    frame
                ],
                root_quat,
            )
        )


        R = self._root_rotation_matrix()


        root_pos_error_body = (
            R.T
            @ root_pos_error_world
        )


        base_lin_vel_body = (
            R.T
            @ self.data.qvel[
                0:3
            ]
        )


        # MuJoCo free-joint angular velocity is
        # represented in the local frame.
        base_ang_vel_body = (
            self.data.qvel[
                3:6
            ].copy()
        )


        ref_lin_vel_error_world = (
            self.data.qvel[
                0:3
            ]
            - self.ref_full_qvel[
                frame,
                0:3,
            ]
        )


        root_lin_vel_error_body = (
            R.T
            @ ref_lin_vel_error_world
        )


        root_ang_vel_error = (
            self.data.qvel[
                3:6
            ]
            - self.ref_full_qvel[
                frame,
                3:6,
            ]
        )


        projected_gravity = (
            R.T
            @ np.asarray(
                [
                    0.0,
                    0.0,
                    -1.0,
                ]
            )
        )


        left_foot_error_body = (
            R.T
            @ (
                self.data.site_xpos[
                    self.left_foot_site
                ]
                - self.ref_left_foot_pos[
                    frame
                ]
            )
        )


        right_foot_error_body = (
            R.T
            @ (
                self.data.site_xpos[
                    self.right_foot_site
                ]
                - self.ref_right_foot_pos[
                    frame
                ]
            )
        )


        phase = (
            frame
            / max(
                self.num_frames - 1,
                1,
            )
        )


        phase_features = np.asarray(
            [
                math.sin(
                    2.0
                    * math.pi
                    * phase
                ),

                math.cos(
                    2.0
                    * math.pi
                    * phase
                ),
            ],
            dtype=np.float64,
        )


        actual_contact = (
            self._actual_contacts()
        )


        observation = np.concatenate(
            [
                # Explicit phase.
                phase_features,

                # Motion target.
                q_ref,
                0.20 * qd_ref,

                # Joint state relative to reference.
                q_error,
                0.20 * qd_error,

                # Floating-base state.
                0.50 * base_lin_vel_body,
                0.20 * base_ang_vel_body,
                projected_gravity,

                # Floating-base reference errors.
                2.0 * root_pos_error_body,
                root_orientation_error,
                0.50 * root_lin_vel_error_body,
                0.20 * root_ang_vel_error,

                # End-effector state relative to reference.
                5.0 * left_foot_error_body,
                5.0 * right_foot_error_body,

                # Contact command + actual contact.
                self.ref_contact[
                    frame
                ],
                actual_contact,

                # Policy memory.
                self.previous_action,
            ]
        )


        if not np.all(
            np.isfinite(
                observation
            )
        ):

            raise FloatingPointError(
                "Non-finite observation."
            )


        # Defensive clipping for neural-network input only.
        observation = np.clip(
            observation,
            -20.0,
            20.0,
        )


        return observation.astype(
            np.float32
        )


    # ============================================================
    # REWARD
    # ============================================================

    @staticmethod
    def _exp_reward(
        error: float,
        sigma: float,
    ) -> float:

        return float(
            math.exp(
                -(
                    error
                    / sigma
                ) ** 2
            )
        )


    def _calculate_reward(
        self,
        action: np.ndarray,
    ) -> tuple[
        float,
        dict[str, float],
    ]:

        frame = int(
            self._current_frame
        )


        q = self._joint_q()

        qd = self._joint_qd()


        q_error = float(
            np.sqrt(
                np.mean(
                    (
                        q
                        - self.ref_q[
                            frame
                        ]
                    ) ** 2
                )
            )
        )


        qd_error = float(
            np.sqrt(
                np.mean(
                    (
                        qd
                        - self.ref_qd[
                            frame
                        ]
                    ) ** 2
                )
            )
        )


        root_position_error = float(
            np.linalg.norm(
                self.data.qpos[
                    0:3
                ]
                - self.ref_root_pos[
                    frame
                ]
            )
        )


        root_orientation_error = (
            quat_angle(
                self.ref_root_quat[
                    frame
                ],
                self.data.qpos[
                    3:7
                ],
            )
        )


        root_velocity_error = float(
            np.linalg.norm(
                self.data.qvel[
                    0:3
                ]
                - self.ref_full_qvel[
                    frame,
                    0:3,
                ]
            )
        )


        (
            body_position_error,
            body_rotation_error,
        ) = self._body_pose_errors(
            frame
        )


        actual_contact = (
            self._actual_contacts()
        )


        contact_match = float(
            np.mean(
                actual_contact
                ==
                self.ref_contact[
                    frame
                ]
            )
        )


        up = self._up_z()


        upright = float(
            np.clip(
                (
                    up
                    - 0.40
                )
                / 0.60,
                0.0,
                1.0,
            )
        )


        action = np.asarray(
            action,
            dtype=np.float64,
        )


        action_magnitude = float(
            np.mean(
                action ** 2
            )
        )


        action_rate = float(
            np.mean(
                (
                    action
                    - self.previous_action
                ) ** 2
            )
        )


        terms = {
            "joint_pos":
                self._exp_reward(
                    q_error,
                    0.15,
                ),

            "joint_vel":
                self._exp_reward(
                    qd_error,
                    1.25,
                ),

            "body_pos":
                self._exp_reward(
                    body_position_error,
                    0.10,
                ),

            "body_rot":
                self._exp_reward(
                    body_rotation_error,
                    0.45,
                ),

            "root_pos":
                self._exp_reward(
                    root_position_error,
                    0.25,
                ),

            "root_ori":
                self._exp_reward(
                    root_orientation_error,
                    0.40,
                ),

            "root_vel":
                self._exp_reward(
                    root_velocity_error,
                    1.00,
                ),

            "contact":
                contact_match,

            "upright":
                upright,

            "action_mag":
                action_magnitude,

            "action_rate":
                action_rate,
        }


        reward = (
            1.20
            * terms[
                "joint_pos"
            ]

            + 0.35
            * terms[
                "joint_vel"
            ]

            + 1.60
            * terms[
                "body_pos"
            ]

            + 0.80
            * terms[
                "body_rot"
            ]

            + 0.80
            * terms[
                "root_pos"
            ]

            + 1.20
            * terms[
                "root_ori"
            ]

            + 0.45
            * terms[
                "root_vel"
            ]

            + 0.35
            * terms[
                "contact"
            ]

            + 0.25
            * terms[
                "upright"
            ]

            - 0.015
            * terms[
                "action_mag"
            ]

            - 0.030
            * terms[
                "action_rate"
            ]
        )


        terms.update(
            {
                "q_error":
                    q_error,

                "qd_error":
                    qd_error,

                "body_position_error":
                    body_position_error,

                "body_rotation_error":
                    body_rotation_error,

                "root_position_error":
                    root_position_error,

                "root_orientation_deg":
                    math.degrees(
                        root_orientation_error
                    ),

                "root_velocity_error":
                    root_velocity_error,

                "up":
                    up,

                "reward":
                    float(
                        reward
                    ),
            }
        )


        return (
            float(
                reward
            ),
            terms,
        )


    # ============================================================
    # TERMINATION
    # ============================================================

    def _physical_failure(
        self,
    ) -> tuple[
        bool,
        str,
    ]:

        if not np.all(
            np.isfinite(
                self.data.qpos
            )
        ):

            return (
                True,
                "nonfinite_qpos",
            )


        if not np.all(
            np.isfinite(
                self.data.qvel
            )
        ):

            return (
                True,
                "nonfinite_qvel",
            )


        if float(
            self.data.qpos[
                2
            ]
        ) < 0.45:

            return (
                True,
                "root_height",
            )


        if self._up_z() < 0.40:

            return (
                True,
                "orientation",
            )


        return (
            False,
            "",
        )


    # ============================================================
    # GYM API
    # ============================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):

        super().reset(
            seed=seed
        )


        options = (
            options
            if options is not None
            else {}
        )


        if "start_frame" in options:

            frame = int(
                options[
                    "start_frame"
                ]
            )


        elif self.rsi:

            max_start = max(
                0,
                self.num_frames
                - self.min_remaining_frames
                - 1,
            )


            frame = int(
                self.np_random.integers(
                    0,
                    max_start + 1,
                )
            )


        else:

            frame = 0


        frame = int(
            np.clip(
                frame,
                0,
                self.num_frames - 2,
            )
        )


        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        self.data.qpos[:] = (
            self.ref_full_qpos[
                frame
            ]
        )

        self.data.qvel[:] = (
            self.ref_full_qvel[
                frame
            ]
        )


        # Optional RSI noise.
        if self.reset_joint_noise > 0.0:

            noise = (
                self.np_random.normal(
                    0.0,
                    self.reset_joint_noise,
                    size=29,
                )
            )


            for j, qadr in enumerate(
                self.qaddrs
            ):

                self.data.qpos[
                    qadr
                ] += noise[j]


                jid = self.joint_ids[
                    j
                ]


                if self.model.jnt_limited[
                    jid
                ]:

                    lo, hi = (
                        self.model.jnt_range[
                            jid
                        ]
                    )


                    self.data.qpos[
                        qadr
                    ] = float(
                        np.clip(
                            self.data.qpos[
                                qadr
                            ],
                            lo,
                            hi,
                        )
                    )


        if self.reset_velocity_noise > 0.0:

            noise = (
                self.np_random.normal(
                    0.0,
                    self.reset_velocity_noise,
                    size=29,
                )
            )


            for j, vadr in enumerate(
                self.vaddrs
            ):

                self.data.qvel[
                    vadr
                ] += noise[j]


        mujoco.mj_normalizeQuat(
            self.model,
            self.data.qpos,
        )


        self._current_frame = frame

        self.previous_action[:] = 0.0

        self.last_target = (
            self.ref_q[
                frame
            ].copy()
        )


        self._apply_target(
            self.last_target
        )


        mujoco.mj_forward(
            self.model,
            self.data,
        )


        observation = (
            self._get_observation()
        )


        info = {
            "start_frame":
                frame,

            "reference_frame":
                frame,

            "phase":
                frame
                / (
                    self.num_frames
                    - 1
                ),

            "action_scale":
                self.action_scale.copy(),
        }


        return (
            observation,
            info,
        )


    def step(
        self,
        action: np.ndarray,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        )


        if action.shape != (
            29,
        ):

            raise ValueError(
                f"Expected action shape (29,), "
                f"got {action.shape}"
            )


        action = np.clip(
            action,
            -1.0,
            1.0,
        )


        next_frame = min(
            self._current_frame
            + 1,
            self.num_frames
            - 1,
        )


        target = (
            self._action_target(
                action,
                next_frame,
            )
        )


        self.last_target = (
            target.copy()
        )


        self._apply_target(
            target
        )


        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )


        self._current_frame = (
            next_frame
        )


        (
            reward,
            reward_terms,
        ) = self._calculate_reward(
            action
        )


        (
            terminated,
            termination_reason,
        ) = self._physical_failure()


        truncated = bool(
            self._current_frame
            >= self.num_frames - 1
        )


        observation = (
            self._get_observation()
        )


        info = {
            "reference_frame":
                int(
                    self._current_frame
                ),

            "phase":
                self._current_frame
                / (
                    self.num_frames
                    - 1
                ),

            "termination_reason":
                termination_reason,

            "reward_terms":
                reward_terms,

            "up":
                float(
                    self._up_z()
                ),

            "root_z":
                float(
                    self.data.qpos[
                        2
                    ]
                ),

            "target":
                self.last_target.copy(),
        }


        self.previous_action = (
            action.copy()
        )


        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def close(
        self,
    ) -> None:

        pass
