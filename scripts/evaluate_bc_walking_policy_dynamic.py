import os
import argparse
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import mujoco


MODEL_XML_PATH = os.path.join(
    "third_party",
    "mujoco_menagerie",
    "unitree_g1",
    "scene.xml",
)


class BCWalkingPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def build_bc_observation(frame, total_frames):

    progress = frame / max(
        total_frames - 1,
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


def get_joint_info(model, names):

    info = []

    for name in names:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint not found: {name}"
            )

        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if aid < 0:

            # Fallback: find actuator by transmission joint ID
            aid = None

            for candidate in range(model.nu):

                if int(
                    model.actuator_trnid[
                        candidate, 0
                    ]
                ) == jid:

                    aid = candidate
                    break

            if aid is None:
                raise RuntimeError(
                    f"Actuator not found for {name}"
                )

        info.append(
            {
                "name": name,
                "jid": int(jid),
                "aid": int(aid),
                "qadr": int(
                    model.jnt_qposadr[jid]
                ),
            }
        )

    return info


def clip_ctrl(model, actuator_id, value):

    lo, hi = model.actuator_ctrlrange[
        actuator_id
    ]

    return float(
        np.clip(
            value,
            lo,
            hi,
        )
    )


def body_up_z(data, body_id):

    R = np.asarray(
        data.xmat[body_id],
        dtype=float,
    ).reshape(3, 3)

    return float(
        R[2, 2]
    )


def quaternion_yaw(q):

    w, x, y, z = [
        float(v)
        for v in q
    ]

    return math.atan2(
        2.0 * (
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


def find_sole_spheres(
    model,
    body_id,
):

    result = []

    for gid in range(model.ngeom):

        if int(
            model.geom_bodyid[gid]
        ) != int(body_id):
            continue

        if int(
            model.geom_type[gid]
        ) != int(
            mujoco.mjtGeom.mjGEOM_SPHERE
        ):
            continue

        radius = float(
            model.geom_size[gid][0]
        )

        if radius <= 0.010:
            result.append(gid)

    return sorted(result)


def physical_sole_clearance(
    model,
    data,
    geom_ids,
    floor_z,
):

    if not geom_ids:
        return float("nan")

    vals = []

    for gid in geom_ids:

        radius = float(
            model.geom_size[gid][0]
        )

        vals.append(
            float(
                data.geom_xpos[gid][2]
            )
            - radius
            - floor_z
        )

    return float(
        min(vals)
    )


def foot_contact(
    model,
    data,
    geom_ids,
    floor_gid,
):

    geom_set = set(
        int(x)
        for x in geom_ids
    )

    for i in range(data.ncon):

        con = data.contact[i]

        g1 = int(con.geom1)
        g2 = int(con.geom2)

        if (
            g1 == floor_gid
            and g2 in geom_set
        ):
            return True

        if (
            g2 == floor_gid
            and g1 in geom_set
        ):
            return True

    return False


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        default=os.path.join(
            "models",
            "g1_bc_walking_policy.pt",
        ),
    )

    parser.add_argument(
        "--dataset",
        default=os.path.join(
            "datasets",
            "processed",
            "g1_amass_walking_il_15dof.npz",
        ),
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="BC/reference gait frame used at the beginning of the stand-to-walk transition.",
    )

    parser.add_argument(
        "--stand_frames",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--transition_frames",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--target_smoothing",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Adds an initial root-height offset in meters.",
    )

    parser.add_argument(
        "--csv",
        default=os.path.join(
            "results",
            "bc_dynamic_baseline.csv",
        ),
    )

    args = parser.parse_args()


    if not os.path.exists(args.policy):
        raise FileNotFoundError(
            args.policy
        )

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            args.dataset
        )

    if not os.path.exists(
        MODEL_XML_PATH
    ):
        raise FileNotFoundError(
            MODEL_XML_PATH
        )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # =========================================================
    # BC MODEL
    # =========================================================

    checkpoint = torch.load(
        args.policy,
        map_location=device,
        weights_only=False,
    )

    obs_dim = int(
        checkpoint["obs_dim"]
    )

    action_dim = int(
        checkpoint["action_dim"]
    )

    if obs_dim != 3:
        raise RuntimeError(
            f"Expected BC obs_dim=3, got {obs_dim}"
        )

    if action_dim != 15:
        raise RuntimeError(
            f"Expected BC action_dim=15, got {action_dim}"
        )


    obs_mean = np.asarray(
        checkpoint["obs_mean"],
        dtype=np.float32,
    ).reshape(-1)

    obs_std = np.asarray(
        checkpoint["obs_std"],
        dtype=np.float32,
    ).reshape(-1)

    action_mean = np.asarray(
        checkpoint["action_mean"],
        dtype=np.float32,
    ).reshape(-1)

    action_std = np.asarray(
        checkpoint["action_std"],
        dtype=np.float32,
    ).reshape(-1)

    joint_names = [
        str(x)
        for x
        in checkpoint[
            "controlled_joint_names"
        ]
    ]


    policy = BCWalkingPolicy(
        obs_dim,
        action_dim,
    ).to(device)

    policy.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    policy.eval()


    # =========================================================
    # DATASET
    # =========================================================

    dataset = np.load(
        args.dataset,
        allow_pickle=True,
    )

    if "joint_pos_15" in dataset:
        total_frames = int(
            dataset[
                "joint_pos_15"
            ].shape[0]
        )
    else:
        total_frames = int(
            dataset[
                "il_actions"
            ].shape[0]
        )

    fps = float(
        np.asarray(
            dataset["fps"]
        ).reshape(-1)[0]
    )


    # =========================================================
    # MUJOCO
    # =========================================================

    model = mujoco.MjModel.from_xml_path(
        MODEL_XML_PATH
    )

    data = mujoco.MjData(
        model
    )


    if model.nkey > 0:

        stand_qpos = (
            model.key_qpos[0]
            .copy()
        )

    else:

        stand_qpos = np.zeros(
            model.nq,
            dtype=np.float64,
        )

        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79


    joint_info = get_joint_info(
        model,
        joint_names,
    )


    pelvis_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    if pelvis_id < 0:
        raise RuntimeError(
            "pelvis body not found"
        )


    left_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_foot",
    )

    right_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_foot",
    )

    if (
        left_site < 0
        or right_site < 0
    ):
        raise RuntimeError(
            "Foot sites not found"
        )


    left_body = int(
        model.site_bodyid[
            left_site
        ]
    )

    right_body = int(
        model.site_bodyid[
            right_site
        ]
    )


    left_sole = find_sole_spheres(
        model,
        left_body,
    )

    right_sole = find_sole_spheres(
        model,
        right_body,
    )


    floor_gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )

    if floor_gid < 0:
        raise RuntimeError(
            "floor geom not found"
        )


    mujoco.mj_resetData(
        model,
        data,
    )

    data.qpos[:] = stand_qpos
    data.qvel[:] = 0.0

    # Diagnostic-only initial root-height offset.
    # No root teleportation occurs after initialization.
    data.qpos[2] = float(
        stand_qpos[2]
        + args.height_offset
    )

    mujoco.mj_forward(
        model,
        data,
    )


    floor_z = float(
        data.geom_xpos[
            floor_gid
        ][2]
    )


    # =========================================================
    # HOLD UPPER BODY AT STANDING POSE
    # =========================================================

    controlled_aids = {
        x["aid"]
        for x
        in joint_info
    }

    upper_body = []

    for aid in range(model.nu):

        if aid in controlled_aids:
            continue

        jid = int(
            model.actuator_trnid[
                aid, 0
            ]
        )

        if (
            jid < 0
            or jid >= model.njnt
        ):
            continue

        qadr = int(
            model.jnt_qposadr[
                jid
            ]
        )

        upper_body.append(
            (
                aid,
                float(
                    stand_qpos[qadr]
                ),
            )
        )


    for item in joint_info:

        stand_target = float(
            stand_qpos[
                item["qadr"]
            ]
        )

        data.ctrl[
            item["aid"]
        ] = clip_ctrl(
            model,
            item["aid"],
            stand_target,
        )


    for aid, target in upper_body:

        data.ctrl[aid] = clip_ctrl(
            model,
            aid,
            target,
        )


    mujoco.mj_forward(
        model,
        data,
    )


    # =========================================================
    # CONTROL RATE
    # =========================================================

    sim_dt = float(
        model.opt.timestep
    )

    control_dt = (
        1.0 / fps
    )

    frame_skip = max(
        1,
        int(
            round(
                control_dt
                / sim_dt
            )
        ),
    )


    total_gait_frames = (
        total_frames
        * max(
            args.cycles,
            1,
        )
    )

    total_control_frames = (
        args.stand_frames
        + args.transition_frames
        + total_gait_frames
    )


    print()
    print("=" * 100)
    print("DYNAMIC BC WALKING BASELINE")
    print("=" * 100)

    print(
        "Policy              =",
        args.policy,
    )

    print(
        "Dataset             =",
        args.dataset,
    )

    print(
        "BC observation      =",
        obs_dim,
    )

    print(
        "BC actions          =",
        action_dim,
    )

    print(
        "Dataset frames      =",
        total_frames,
    )

    print(
        "Dataset FPS         =",
        fps,
    )

    print(
        "BC start frame      =",
        args.start_frame,
    )

    print(
        "Initial height offset=",
        args.height_offset,
    )

    print(
        "MuJoCo dt           =",
        sim_dt,
    )

    print(
        "Physics steps/control=",
        frame_skip,
    )

    print(
        "Left sole spheres   =",
        left_sole,
    )

    print(
        "Right sole spheres  =",
        right_sole,
    )

    print(
        "Root teleportation  = NONE"
    )

    print(
        "PPO                 = NONE"
    )

    print()


    # =========================================================
    # METRICS
    # =========================================================

    rows = []

    y_sq_sum = 0.0

    max_abs_y = 0.0
    max_abs_yaw = 0.0

    min_up_z = 1.0
    min_height = 999.0

    max_valid_left_clear = -999.0
    max_valid_right_clear = -999.0

    left_air_steps = 0
    right_air_steps = 0

    left_contact_switches = 0
    right_contact_switches = 0

    previous_left_contact = None
    previous_right_contact = None

    terminated = False
    termination_reason = "completed"

    previous_targets = np.array(
        [
            float(
                stand_qpos[
                    item["qadr"]
                ]
            )
            for item in joint_info
        ],
        dtype=np.float64,
    )


    # =========================================================
    # ROLLOUT
    # =========================================================

    for control_frame in range(
        total_control_frames
    ):

        if control_frame < args.stand_frames:

            requested = np.array(
                [
                    float(
                        stand_qpos[
                            item["qadr"]
                        ]
                    )
                    for item
                    in joint_info
                ],
                dtype=np.float64,
            )

            gait_frame = 0
            alpha = 0.0

        else:

            local = (
                control_frame
                - args.stand_frames
            )

            gait_frame = int(
                (
                    args.start_frame
                    + local
                )
                % total_frames
            )


            raw_obs = build_bc_observation(
                gait_frame,
                total_frames,
            )

            obs_norm = (
                raw_obs
                - obs_mean
            ) / obs_std


            obs_tensor = torch.tensor(
                obs_norm,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)


            with torch.no_grad():

                action_norm = (
                    policy(
                        obs_tensor
                    )
                    .cpu()
                    .numpy()[0]
                )


            bc_targets = (
                action_norm
                * action_std
                + action_mean
            ).astype(
                np.float64
            )


            if (
                local
                < args.transition_frames
            ):

                u = (
                    local
                    / max(
                        args.transition_frames,
                        1,
                    )
                )

                u = float(
                    np.clip(
                        u,
                        0.0,
                        1.0,
                    )
                )

                alpha = (
                    u
                    * u
                    * (
                        3.0
                        - 2.0 * u
                    )
                )

            else:
                alpha = 1.0


            stand_targets = np.array(
                [
                    float(
                        stand_qpos[
                            item["qadr"]
                        ]
                    )
                    for item
                    in joint_info
                ],
                dtype=np.float64,
            )


            requested = (
                (
                    1.0 - alpha
                )
                * stand_targets
                + alpha
                * bc_targets
            )


        smoothing = float(
            np.clip(
                args.target_smoothing,
                0.0,
                0.98,
            )
        )


        targets = (
            smoothing
            * previous_targets
            + (
                1.0
                - smoothing
            )
            * requested
        )


        previous_targets = (
            targets.copy()
        )


        for i, item in enumerate(
            joint_info
        ):

            data.ctrl[
                item["aid"]
            ] = clip_ctrl(
                model,
                item["aid"],
                targets[i],
            )


        for aid, target in upper_body:

            data.ctrl[aid] = (
                clip_ctrl(
                    model,
                    aid,
                    target,
                )
            )


        for _ in range(
            frame_skip
        ):

            mujoco.mj_step(
                model,
                data,
            )


        # =====================================================
        # MEASURE REAL PHYSICS
        # =====================================================

        base_x = float(
            data.qpos[0]
        )

        base_y = float(
            data.qpos[1]
        )

        base_z = float(
            data.qpos[2]
        )

        vx = float(
            data.qvel[0]
        )

        vy = float(
            data.qvel[1]
        )

        yaw = quaternion_yaw(
            data.qpos[3:7]
        )

        up_z = body_up_z(
            data,
            pelvis_id,
        )


        left_contact = foot_contact(
            model,
            data,
            left_sole,
            floor_gid,
        )

        right_contact = foot_contact(
            model,
            data,
            right_sole,
            floor_gid,
        )


        left_clear = (
            physical_sole_clearance(
                model,
                data,
                left_sole,
                floor_z,
            )
        )

        right_clear = (
            physical_sole_clearance(
                model,
                data,
                right_sole,
                floor_z,
            )
        )


        if (
            previous_left_contact
            is not None
            and left_contact
            != previous_left_contact
        ):
            left_contact_switches += 1


        if (
            previous_right_contact
            is not None
            and right_contact
            != previous_right_contact
        ):
            right_contact_switches += 1


        previous_left_contact = (
            left_contact
        )

        previous_right_contact = (
            right_contact
        )


        if not left_contact:
            left_air_steps += 1

        if not right_contact:
            right_air_steps += 1


        # Only treat clearance as valid while robot is still upright.
        if (
            up_z >= 0.80
            and base_z >= 0.65
        ):

            if not left_contact:
                max_valid_left_clear = max(
                    max_valid_left_clear,
                    left_clear,
                )

            if not right_contact:
                max_valid_right_clear = max(
                    max_valid_right_clear,
                    right_clear,
                )


        y_sq_sum += (
            base_y ** 2
        )

        max_abs_y = max(
            max_abs_y,
            abs(base_y),
        )

        max_abs_yaw = max(
            max_abs_yaw,
            abs(yaw),
        )

        min_up_z = min(
            min_up_z,
            up_z,
        )

        min_height = min(
            min_height,
            base_z,
        )


        rows.append(
            {
                "control_frame":
                    control_frame,

                "gait_frame":
                    gait_frame,

                "blend_alpha":
                    alpha,

                "x":
                    base_x,

                "y":
                    base_y,

                "z":
                    base_z,

                "vx":
                    vx,

                "vy":
                    vy,

                "yaw_deg":
                    math.degrees(yaw),

                "up_z":
                    up_z,

                "left_contact":
                    int(left_contact),

                "right_contact":
                    int(right_contact),

                "left_true_clearance_mm":
                    1000.0
                    * left_clear,

                "right_true_clearance_mm":
                    1000.0
                    * right_clear,
            }
        )


        # =====================================================
        # FALL / DIVERGENCE
        # =====================================================

        if base_z < 0.45:

            terminated = True
            termination_reason = (
                "base_height"
            )

        elif up_z < 0.50:

            terminated = True
            termination_reason = (
                "upright"
            )

        elif abs(base_y) > 0.50:

            terminated = True
            termination_reason = (
                "lateral_position"
            )

        elif abs(vx) > 1.5:

            terminated = True
            termination_reason = (
                "forward_velocity"
            )

        elif abs(vy) > 1.5:

            terminated = True
            termination_reason = (
                "lateral_velocity"
            )


        if terminated:
            break


    # =========================================================
    # CSV
    # =========================================================

    os.makedirs(
        os.path.dirname(
            args.csv
        ),
        exist_ok=True,
    )


    with open(
        args.csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


    # =========================================================
    # SUMMARY
    # =========================================================

    final = rows[-1]

    rms_y = math.sqrt(
        y_sq_sum
        / max(
            len(rows),
            1,
        )
    )


    elapsed = (
        len(rows)
        * control_dt
    )


    mean_vx = (
        final["x"]
        / max(
            elapsed,
            1e-8,
        )
    )


    print()
    print("=" * 100)
    print("DYNAMIC BC RESULT")
    print("=" * 100)

    print(
        f"control frames             = "
        f"{len(rows)} / {total_control_frames}"
    )

    print(
        f"elapsed simulated time     = "
        f"{elapsed:.3f} s"
    )

    print(
        f"termination                = "
        f"{termination_reason}"
    )

    print()
    print("STRAIGHT-LINE MOTION")

    print(
        f"final X                    = "
        f"{final['x']:+.4f} m"
    )

    print(
        f"mean X velocity            = "
        f"{mean_vx:+.4f} m/s"
    )

    print(
        f"final Y                    = "
        f"{final['y']:+.4f} m"
    )

    print(
        f"max |Y|                    = "
        f"{max_abs_y:.4f} m"
    )

    print(
        f"RMS Y error                = "
        f"{rms_y:.4f} m"
    )

    print(
        f"final yaw                  = "
        f"{final['yaw_deg']:+.2f} deg"
    )

    print(
        f"max |yaw|                  = "
        f"{math.degrees(max_abs_yaw):.2f} deg"
    )


    print()
    print("BALANCE")

    print(
        f"minimum up_z               = "
        f"{min_up_z:.4f}"
    )

    print(
        f"minimum base height        = "
        f"{min_height:.4f} m"
    )

    print(
        f"final vx / vy              = "
        f"{final['vx']:+.4f} / "
        f"{final['vy']:+.4f} m/s"
    )


    print()
    print("GAIT / PHYSICAL FOOT MOTION")

    print(
        f"left contact switches      = "
        f"{left_contact_switches}"
    )

    print(
        f"right contact switches     = "
        f"{right_contact_switches}"
    )

    print(
        f"left airborne frames       = "
        f"{left_air_steps}"
    )

    print(
        f"right airborne frames      = "
        f"{right_air_steps}"
    )


    if (
        max_valid_left_clear
        > -900
    ):
        print(
            f"max valid LEFT sole clear  = "
            f"{1000*max_valid_left_clear:.3f} mm"
        )
    else:
        print(
            "max valid LEFT sole clear  = "
            "no valid airborne sample"
        )


    if (
        max_valid_right_clear
        > -900
    ):
        print(
            f"max valid RIGHT sole clear = "
            f"{1000*max_valid_right_clear:.3f} mm"
        )
    else:
        print(
            "max valid RIGHT sole clear = "
            "no valid airborne sample"
        )


    print()
    print(
        "CSV                        =",
        args.csv,
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
