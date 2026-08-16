import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DATASET = (
    "datasets\\processed\\"
    "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)

DEFAULT_OUTPUT = "models\\g1_bc_openhe_kinematic.pt"

DEFAULT_BEST_VAL_OUTPUT = (
    "models\\g1_bc_openhe_kinematic_best_val.pt"
)


class BCPolicy(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, 15),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_std(x):
    std = x.std(axis=0).astype(np.float32)

    std[
        std < 1e-6
    ] = 1.0

    return std


def metrics(pred, target):
    error = pred - target

    return {
        "rmse_rad": float(
            np.sqrt(
                np.mean(
                    error.astype(np.float64) ** 2
                )
            )
        ),

        "mae_rad": float(
            np.mean(
                np.abs(error),
                dtype=np.float64,
            )
        ),

        "max_abs_error_rad": float(
            np.max(
                np.abs(error)
            )
        ),
    }


def predict(
    model,
    x_raw,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
):
    x = (
        (x_raw - x_mean)
        / x_std
    ).astype(np.float32)

    x_tensor = torch.from_numpy(
        x
    ).to(device)

    model.eval()

    with torch.no_grad():
        y_norm = (
            model(
                x_tensor
            )
            .cpu()
            .numpy()
        )

    return (
        y_norm
        * y_std
        + y_mean
    ).astype(np.float32)


def save_checkpoint(
    path,
    model,
    hidden_dim,
    x_mean,
    x_std,
    y_mean,
    y_std,
    dataset_path,
    joint_names,
    fps,
    stage,
    validation_metrics,
    full_metrics,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "policy_type":
                "behavior_cloning_mlp",

            "stage":
                stage,

            "model_state_dict":
                model.state_dict(),

            "input_dim":
                3,

            "hidden_dim":
                int(hidden_dim),

            "output_dim":
                15,

            "input_features": [
                "phase_progress",
                "root_velocity_x",
                "root_velocity_y",
            ],

            "output_features":
                list(joint_names),

            "x_mean":
                torch.tensor(
                    x_mean,
                    dtype=torch.float32,
                ),

            "x_std":
                torch.tensor(
                    x_std,
                    dtype=torch.float32,
                ),

            "y_mean":
                torch.tensor(
                    y_mean,
                    dtype=torch.float32,
                ),

            "y_std":
                torch.tensor(
                    y_std,
                    dtype=torch.float32,
                ),

            "dataset_path":
                str(dataset_path),

            "fps":
                float(fps),

            "num_frames":
                300,

            "validation_metrics":
                validation_metrics,

            "full_dataset_metrics":
                full_metrics,

            # -------------------------------------------------
            # Locked showcase camera
            # Same camera as our approved reference replay.
            # -------------------------------------------------
            "showcase_camera": {
                "fixed_camera":
                    True,

                "distance":
                    4.5,

                "azimuth":
                    0.0,

                "elevation":
                    -15.0,

                "lookat_z_offset":
                    -0.10,

                "camera_follow":
                    False,

                "reference_yaw_degrees":
                    180.0,

                "root_motion_scale":
                    0.45,
            },

            "notes": (
                "Kinematic Behavior Cloning policy. "
                "Training uses only the approved "
                "OpenHE dataset and PyTorch. "
                "No MuJoCo dynamics and no mj_step "
                "are used during training."
            ),
        },
        path,
    )

    return path


def train_stage(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    y_mean,
    y_std,
    device,
    epochs,
    lr,
    weight_decay,
    print_every,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss()

    x_train = torch.from_numpy(
        x_train
    ).to(device)

    y_train = torch.from_numpy(
        y_train
    ).to(device)

    x_val = torch.from_numpy(
        x_val
    ).to(device)

    y_val = torch.from_numpy(
        y_val
    ).to(device)

    y_mean_t = torch.from_numpy(
        y_mean
    ).to(device)

    y_std_t = torch.from_numpy(
        y_std
    ).to(device)

    best_loss = float("inf")
    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        # -----------------------------------------------------
        # TRAIN
        # -----------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            x_train
        )

        train_loss = loss_fn(
            pred,
            y_train,
        )

        train_loss.backward()

        optimizer.step()

        # -----------------------------------------------------
        # VALIDATE
        # -----------------------------------------------------

        model.eval()

        with torch.no_grad():
            val_pred = model(
                x_val
            )

            val_loss = loss_fn(
                val_pred,
                y_val,
            )

            val_pred_rad = (
                val_pred
                * y_std_t
                + y_mean_t
            )

            val_true_rad = (
                y_val
                * y_std_t
                + y_mean_t
            )

            val_rmse_rad = torch.sqrt(
                torch.mean(
                    (
                        val_pred_rad
                        - val_true_rad
                    )
                    ** 2
                )
            )

        val_loss_value = float(
            val_loss.item()
        )

        if (
            val_loss_value
            < best_loss
        ):
            best_loss = (
                val_loss_value
            )

            best_epoch = (
                epoch
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

        if (
            epoch == 1
            or
            epoch % print_every == 0
            or
            epoch == epochs
        ):
            print(
                f"epoch={epoch:05d}/{epochs} "
                f"train_mse_norm="
                f"{train_loss.item():.8f} "
                f"val_mse_norm="
                f"{val_loss_value:.8f} "
                f"val_rmse_rad="
                f"{val_rmse_rad.item():.6f}"
            )

    model.load_state_dict(
        best_state
    )

    return (
        best_epoch,
        best_loss,
    )


def refit_all(
    model,
    x_all,
    y_all,
    device,
    epochs,
    lr,
    weight_decay,
    print_every,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss()

    x_all = torch.from_numpy(
        x_all
    ).to(device)

    y_all = torch.from_numpy(
        y_all
    ).to(device)

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            x_all
        )

        loss = loss_fn(
            pred,
            y_all,
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or
            epoch % print_every == 0
            or
            epoch == epochs
        ):
            print(
                f"refit_epoch="
                f"{epoch:05d}/{epochs} "
                f"mse_norm="
                f"{loss.item():.8f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train OpenHE Unitree G1 "
            "kinematic Behavior Cloning policy."
        )
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--best_val_output",
        default=DEFAULT_BEST_VAL_OUTPUT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--refit_epochs",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--refit_learning_rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--validation_stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=499,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    # =========================================================
    # LOAD DATASET
    # =========================================================

    dataset_path = Path(
        args.dataset
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    data = np.load(
        dataset_path,
        allow_pickle=True,
    )

    required = [
        "il_observations",
        "il_actions",
        "joint_pos_15",
        "controlled_joint_names",
        "fps",
    ]

    missing = [
        key
        for key in required
        if key not in data.files
    ]

    if missing:
        raise KeyError(
            f"Missing dataset keys: "
            f"{missing}"
        )

    observations = np.asarray(
        data[
            "il_observations"
        ],
        dtype=np.float32,
    )

    actions = np.asarray(
        data[
            "il_actions"
        ],
        dtype=np.float32,
    )

    joint_pos = np.asarray(
        data[
            "joint_pos_15"
        ],
        dtype=np.float32,
    )

    joint_names = [
        str(x)
        for x in
        data[
            "controlled_joint_names"
        ].tolist()
    ]

    fps = float(
        np.asarray(
            data[
                "fps"
            ]
        )
        .reshape(-1)[0]
    )

    # =========================================================
    # VERIFY EXACT APPROVED DATASET
    # =========================================================

    if (
        observations.shape
        != (300, 3)
    ):
        raise ValueError(
            "Expected approved "
            "observation shape "
            "(300, 3), got "
            f"{observations.shape}"
        )

    if (
        actions.shape
        != (300, 15)
    ):
        raise ValueError(
            "Expected approved "
            "action shape "
            "(300, 15), got "
            f"{actions.shape}"
        )

    max_action_joint_diff = float(
        np.max(
            np.abs(
                actions
                - joint_pos
            )
        )
    )

    if (
        max_action_joint_diff
        > 1e-7
    ):
        raise ValueError(
            "il_actions do not exactly "
            "match joint_pos_15. "
            f"Max difference = "
            f"{max_action_joint_diff}"
        )

    phase = (
        observations[:, 0]
    )

    if not np.all(
        np.diff(
            phase
        ) >= 0.0
    ):
        raise ValueError(
            "Phase/progress "
            "is not monotonic."
        )

    if not np.isclose(
        phase[0],
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Expected phase start "
            f"0.0, got {phase[0]}"
        )

    if not np.isclose(
        phase[-1],
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Expected phase end "
            f"1.0, got {phase[-1]}"
        )

    if (
        args.validation_stride
        < 2
    ):
        raise ValueError(
            "--validation_stride "
            "must be >= 2."
        )

    # =========================================================
    # TRAIN / VALIDATION SPLIT
    #
    # Every 5th frame is validation by default.
    #
    # This produces:
    #
    # 240 training frames
    # 60 validation frames
    #
    # Because frames are interleaved across the demonstration,
    # this checks interpolation quality throughout the walk.
    #
    # It is NOT an unseen-motion generalization test.
    # =========================================================

    indices = np.arange(
        300
    )

    val_mask = (
        indices
        % args.validation_stride
    ) == 0

    val_idx = (
        indices[
            val_mask
        ]
    )

    train_idx = (
        indices[
            ~val_mask
        ]
    )

    x_train_raw = (
        observations[
            train_idx
        ]
    )

    y_train_raw = (
        actions[
            train_idx
        ]
    )

    x_val_raw = (
        observations[
            val_idx
        ]
    )

    y_val_raw = (
        actions[
            val_idx
        ]
    )

    # =========================================================
    # NORMALIZATION
    #
    # Statistics are calculated only from training frames.
    # =========================================================

    x_mean = (
        x_train_raw
        .mean(
            axis=0
        )
        .astype(
            np.float32
        )
    )

    x_std = safe_std(
        x_train_raw
    )

    y_mean = (
        y_train_raw
        .mean(
            axis=0
        )
        .astype(
            np.float32
        )
    )

    y_std = safe_std(
        y_train_raw
    )

    x_train = (
        (
            x_train_raw
            - x_mean
        )
        / x_std
    ).astype(
        np.float32
    )

    y_train = (
        (
            y_train_raw
            - y_mean
        )
        / y_std
    ).astype(
        np.float32
    )

    x_val = (
        (
            x_val_raw
            - x_mean
        )
        / x_std
    ).astype(
        np.float32
    )

    y_val = (
        (
            y_val_raw
            - y_mean
        )
        / y_std
    ).astype(
        np.float32
    )

    x_all = (
        (
            observations
            - x_mean
        )
        / x_std
    ).astype(
        np.float32
    )

    y_all = (
        (
            actions
            - y_mean
        )
        / y_std
    ).astype(
        np.float32
    )

    # =========================================================
    # DEVICE
    # =========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================================================
    # MODEL
    # =========================================================

    model = BCPolicy(
        hidden_dim=
            args.hidden_dim
    ).to(device)

    print()
    print(
        "=" * 78
    )

    print(
        "OPENHE -> UNITREE G1 "
        "KINEMATIC BEHAVIOR CLONING"
    )

    print(
        "=" * 78
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Frames:",
        len(
            observations
        ),
    )

    print(
        "FPS:",
        fps,
    )

    print(
        "Input shape:",
        observations.shape,
    )

    print(
        "Output shape:",
        actions.shape,
    )

    print(
        "Inputs: "
        "[phase_progress, "
        "root_velocity_x, "
        "root_velocity_y]"
    )

    print(
        "Outputs: "
        "15 G1 joint-position targets"
    )

    print(
        "Training frames:",
        len(
            train_idx
        ),
    )

    print(
        "Validation frames:",
        len(
            val_idx
        ),
    )

    print(
        f"Validation: every "
        f"{args.validation_stride}th "
        f"frame held out "
        f"for interpolation check"
    )

    print(
        "Device:",
        device,
    )

    print(
        f"Network: "
        f"3 -> "
        f"{args.hidden_dim} -> "
        f"{args.hidden_dim} -> "
        f"{args.hidden_dim} -> "
        f"15"
    )

    print(
        "MuJoCo physics "
        "in training: NONE"
    )

    print(
        "mj_step "
        "in training: NONE"
    )

    print(
        "=" * 78
    )

    print()

    # =========================================================
    # STAGE 1:
    # TRAIN WITH INTERPOLATION VALIDATION
    # =========================================================

    best_epoch, best_val_loss = (
        train_stage(
            model=model,

            x_train=x_train,
            y_train=y_train,

            x_val=x_val,
            y_val=y_val,

            y_mean=y_mean,
            y_std=y_std,

            device=device,

            epochs=
                args.epochs,

            lr=
                args.learning_rate,

            weight_decay=
                args.weight_decay,

            print_every=
                args.print_every,
        )
    )

    # =========================================================
    # EVALUATE BEST VALIDATION MODEL
    # =========================================================

    val_pred = predict(
        model,
        x_val_raw,
        x_mean,
        x_std,
        y_mean,
        y_std,
        device,
    )

    full_pred_best = predict(
        model,
        observations,
        x_mean,
        x_std,
        y_mean,
        y_std,
        device,
    )

    val_metrics = metrics(
        val_pred,
        y_val_raw,
    )

    best_full_metrics = metrics(
        full_pred_best,
        actions,
    )

    best_path = save_checkpoint(
        path=
            args.best_val_output,

        model=
            model,

        hidden_dim=
            args.hidden_dim,

        x_mean=
            x_mean,

        x_std=
            x_std,

        y_mean=
            y_mean,

        y_std=
            y_std,

        dataset_path=
            dataset_path,

        joint_names=
            joint_names,

        fps=
            fps,

        stage=
            "best_interleaved_validation",

        validation_metrics=
            val_metrics,

        full_metrics=
            best_full_metrics,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "BEST VALIDATION MODEL"
    )

    print(
        "=" * 78
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best normalized "
        "validation MSE:",
        best_val_loss,
    )

    print(
        "Validation RMSE [rad]:",
        f"{val_metrics['rmse_rad']:.8f}",
    )

    print(
        "Validation MAE [rad]:",
        f"{val_metrics['mae_rad']:.8f}",
    )

    print(
        "Validation max "
        "abs error [rad]:",
        f"{val_metrics['max_abs_error_rad']:.8f}",
    )

    print(
        "Saved:",
        best_path,
    )

    print(
        "=" * 78
    )

    print()

    # =========================================================
    # STAGE 2:
    # REFIT ON ALL 300 EXPERT FRAMES
    #
    # This produces the final showcase policy.
    # =========================================================

    print(
        "Refitting selected model "
        "on all 300 expert frames..."
    )

    print()

    refit_all(
        model=
            model,

        x_all=
            x_all,

        y_all=
            y_all,

        device=
            device,

        epochs=
            args.refit_epochs,

        lr=
            args.refit_learning_rate,

        weight_decay=
            args.weight_decay,

        print_every=
            args.print_every,
    )

    # =========================================================
    # FINAL EVALUATION
    # =========================================================

    final_pred = predict(
        model,
        observations,
        x_mean,
        x_std,
        y_mean,
        y_std,
        device,
    )

    final_metrics = metrics(
        final_pred,
        actions,
    )

    final_path = save_checkpoint(
        path=
            args.output,

        model=
            model,

        hidden_dim=
            args.hidden_dim,

        x_mean=
            x_mean,

        x_std=
            x_std,

        y_mean=
            y_mean,

        y_std=
            y_std,

        dataset_path=
            dataset_path,

        joint_names=
            joint_names,

        fps=
            fps,

        stage=
            "final_refit_all_300_frames",

        validation_metrics=
            val_metrics,

        full_metrics=
            final_metrics,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL KINEMATIC BC POLICY"
    )

    print(
        "=" * 78
    )

    print(
        "Saved:",
        final_path,
    )

    print(
        "Full-sequence "
        "RMSE [rad]:",
        f"{final_metrics['rmse_rad']:.8f}",
    )

    print(
        "Full-sequence "
        "MAE [rad]:",
        f"{final_metrics['mae_rad']:.8f}",
    )

    print(
        "Full-sequence "
        "max abs error [rad]:",
        f"{final_metrics['max_abs_error_rad']:.8f}",
    )

    print()

    print(
        "Validation is interpolation validation "
        "inside this same demonstration; "
        "it is not a test of generalization "
        "to unseen motions."
    )

    print(
        "The final showcase checkpoint is "
        "refit on all 300 expert frames "
        "to reproduce this approved "
        "OpenHE walk as closely as possible."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()