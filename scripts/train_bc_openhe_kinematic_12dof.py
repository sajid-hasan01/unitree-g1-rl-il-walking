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

DEFAULT_OUTPUT = (
    "models\\g1_bc_openhe_kinematic_12dof.pt"
)

DEFAULT_BEST_VAL_OUTPUT = (
    "models\\g1_bc_openhe_kinematic_12dof_best_val.pt"
)


LEARNED_JOINT_COUNT = 12

FIXED_WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


class BCPolicy12DOF(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                3,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                LEARNED_JOINT_COUNT,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def safe_std(
    x,
):
    std = (
        x.std(
            axis=0
        )
        .astype(
            np.float32
        )
    )

    std[
        std < 1e-6
    ] = 1.0

    return std


def metrics(
    predicted,
    target,
):
    error = (
        predicted
        -
        target
    )

    return {
        "rmse_rad":
            float(
                np.sqrt(
                    np.mean(
                        error.astype(
                            np.float64
                        )
                        ** 2
                    )
                )
            ),

        "mae_rad":
            float(
                np.mean(
                    np.abs(
                        error
                    ),
                    dtype=np.float64,
                )
            ),

        "max_abs_error_rad":
            float(
                np.max(
                    np.abs(
                        error
                    )
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
        (
            x_raw
            -
            x_mean
        )
        /
        x_std
    ).astype(
        np.float32
    )

    x_tensor = (
        torch
        .from_numpy(
            x
        )
        .to(
            device
        )
    )

    model.eval()

    with torch.no_grad():

        y_normalized = (
            model(
                x_tensor
            )
            .cpu()
            .numpy()
        )

    return (
        y_normalized
        *
        y_std
        +
        y_mean
    ).astype(
        np.float32
    )


def save_checkpoint(
    path,
    model,
    hidden_dim,
    x_mean,
    x_std,
    y_mean,
    y_std,
    dataset_path,
    learned_joint_names,
    fps,
    stage,
    validation_metrics,
    full_metrics,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "policy_type":
                "behavior_cloning_mlp_12dof",

            "stage":
                stage,

            "model_state_dict":
                model.state_dict(),

            "input_dim":
                3,

            "hidden_dim":
                int(
                    hidden_dim
                ),

            "output_dim":
                LEARNED_JOINT_COUNT,

            "input_features": [
                "phase_progress",
                "root_velocity_x",
                "root_velocity_y",
            ],

            "output_features":
                list(
                    learned_joint_names
                ),

            "fixed_waist_joints":
                FIXED_WAIST_JOINTS,

            "fixed_waist_values":
                [
                    0.0,
                    0.0,
                    0.0,
                ],

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
                str(
                    dataset_path
                ),

            "fps":
                float(
                    fps
                ),

            "num_frames":
                300,

            "validation_metrics":
                validation_metrics,

            "full_dataset_metrics":
                full_metrics,

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
                "The network learns the first 12 varying "
                "leg joints. The three waist joints are "
                "fixed at zero because their expert targets "
                "are constant zero throughout the approved "
                "OpenHE demonstration. No MuJoCo physics "
                "or mj_step is used during training."
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
    learning_rate,
    weight_decay,
    print_every,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = (
        nn.MSELoss()
    )

    x_train_tensor = (
        torch
        .from_numpy(
            x_train
        )
        .to(
            device
        )
    )

    y_train_tensor = (
        torch
        .from_numpy(
            y_train
        )
        .to(
            device
        )
    )

    x_val_tensor = (
        torch
        .from_numpy(
            x_val
        )
        .to(
            device
        )
    )

    y_val_tensor = (
        torch
        .from_numpy(
            y_val
        )
        .to(
            device
        )
    )

    y_mean_tensor = (
        torch
        .from_numpy(
            y_mean
        )
        .to(
            device
        )
    )

    y_std_tensor = (
        torch
        .from_numpy(
            y_std
        )
        .to(
            device
        )
    )

    best_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            x_train_tensor
        )

        training_loss = (
            loss_function(
                prediction,
                y_train_tensor,
            )
        )

        training_loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            validation_prediction = (
                model(
                    x_val_tensor
                )
            )

            validation_loss = (
                loss_function(
                    validation_prediction,
                    y_val_tensor,
                )
            )

            validation_prediction_rad = (
                validation_prediction
                *
                y_std_tensor
                +
                y_mean_tensor
            )

            validation_true_rad = (
                y_val_tensor
                *
                y_std_tensor
                +
                y_mean_tensor
            )

            validation_rmse_rad = (
                torch.sqrt(
                    torch.mean(
                        (
                            validation_prediction_rad
                            -
                            validation_true_rad
                        )
                        ** 2
                    )
                )
            )

        validation_loss_value = float(
            validation_loss.item()
        )

        if (
            validation_loss_value
            <
            best_loss
        ):

            best_loss = (
                validation_loss_value
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
                f"{training_loss.item():.8f} "
                f"val_mse_norm="
                f"{validation_loss_value:.8f} "
                f"val_rmse_rad="
                f"{validation_rmse_rad.item():.6f}"
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
    learning_rate,
    weight_decay,
    print_every,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = (
        nn.MSELoss()
    )

    x_tensor = (
        torch
        .from_numpy(
            x_all
        )
        .to(
            device
        )
    )

    y_tensor = (
        torch
        .from_numpy(
            y_all
        )
        .to(
            device
        )
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            x_tensor
        )

        loss = loss_function(
            prediction,
            y_tensor,
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
            "Train 12-DOF OpenHE Unitree G1 "
            "kinematic Behavior Cloning policy."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--best_val_output",
        type=str,
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
        "waist_held_stable",
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

    actions_15 = np.asarray(
        data[
            "il_actions"
        ],
        dtype=np.float32,
    )

    joint_positions_15 = np.asarray(
        data[
            "joint_pos_15"
        ],
        dtype=np.float32,
    )

    joint_names_15 = [
        str(
            name
        )
        for name in
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
        .reshape(
            -1
        )[0]
    )

    waist_held_stable = bool(
        np.asarray(
            data[
                "waist_held_stable"
            ]
        )
        .reshape(
            -1
        )[0]
    )

    if (
        observations.shape
        !=
        (300, 3)
    ):
        raise ValueError(
            "Expected observations "
            "(300, 3), got "
            f"{observations.shape}"
        )

    if (
        actions_15.shape
        !=
        (300, 15)
    ):
        raise ValueError(
            "Expected actions "
            "(300, 15), got "
            f"{actions_15.shape}"
        )

    if not waist_held_stable:
        raise ValueError(
            "Dataset metadata does not "
            "confirm waist_held_stable=True."
        )

    max_action_joint_diff = float(
        np.max(
            np.abs(
                actions_15
                -
                joint_positions_15
            )
        )
    )

    if (
        max_action_joint_diff
        >
        1e-7
    ):
        raise ValueError(
            "il_actions do not exactly "
            "match joint_pos_15."
        )

    expected_waist_names = (
        FIXED_WAIST_JOINTS
    )

    actual_waist_names = (
        joint_names_15[
            12:15
        ]
    )

    if (
        actual_waist_names
        !=
        expected_waist_names
    ):
        raise ValueError(
            "Expected final three joints to be "
            f"{expected_waist_names}, got "
            f"{actual_waist_names}"
        )

    waist_targets = (
        actions_15[
            :,
            12:15
        ]
    )

    waist_max_abs = float(
        np.max(
            np.abs(
                waist_targets
            )
        )
    )

    if (
        waist_max_abs
        >
        1e-7
    ):
        raise ValueError(
            "Waist targets are not constant zero. "
            f"Maximum absolute waist target = "
            f"{waist_max_abs}"
        )

    # ============================================================
    # LEARN ONLY THE 12 ACTUALLY VARYING LEG JOINTS
    # ============================================================

    actions = (
        actions_15[
            :,
            :LEARNED_JOINT_COUNT
        ]
        .copy()
        .astype(
            np.float32
        )
    )

    learned_joint_names = (
        joint_names_15[
            :LEARNED_JOINT_COUNT
        ]
    )

    phase = (
        observations[
            :,
            0
        ]
    )

    if not np.all(
        np.diff(
            phase
        )
        >=
        0.0
    ):
        raise ValueError(
            "Phase/progress is not monotonic."
        )

    if not np.isclose(
        phase[0],
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Phase does not start at 0."
        )

    if not np.isclose(
        phase[-1],
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Phase does not end at 1."
        )

    indices = np.arange(
        observations.shape[
            0
        ]
    )

    validation_mask = (
        indices
        %
        args.validation_stride
    ) == 0

    validation_indices = (
        indices[
            validation_mask
        ]
    )

    training_indices = (
        indices[
            ~validation_mask
        ]
    )

    x_train_raw = (
        observations[
            training_indices
        ]
    )

    y_train_raw = (
        actions[
            training_indices
        ]
    )

    x_validation_raw = (
        observations[
            validation_indices
        ]
    )

    y_validation_raw = (
        actions[
            validation_indices
        ]
    )

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
            -
            x_mean
        )
        /
        x_std
    ).astype(
        np.float32
    )

    y_train = (
        (
            y_train_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(
        np.float32
    )

    x_validation = (
        (
            x_validation_raw
            -
            x_mean
        )
        /
        x_std
    ).astype(
        np.float32
    )

    y_validation = (
        (
            y_validation_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(
        np.float32
    )

    x_all = (
        (
            observations
            -
            x_mean
        )
        /
        x_std
    ).astype(
        np.float32
    )

    y_all = (
        (
            actions
            -
            y_mean
        )
        /
        y_std
    ).astype(
        np.float32
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    model = BCPolicy12DOF(
        hidden_dim=
            args.hidden_dim
    ).to(
        device
    )

    print()

    print(
        "=" * 80
    )

    print(
        "OPENHE -> UNITREE G1 "
        "12-DOF KINEMATIC BEHAVIOR CLONING"
    )

    print(
        "=" * 80
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Frames:",
        observations.shape[
            0
        ],
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
        "Learned target shape:",
        actions.shape,
    )

    print(
        "Learned joints:",
        LEARNED_JOINT_COUNT,
    )

    print(
        "Fixed waist joints:",
        FIXED_WAIST_JOINTS,
    )

    print(
        "Fixed waist values:",
        "[0.0, 0.0, 0.0]"
    )

    print(
        "waist_held_stable metadata:",
        waist_held_stable,
    )

    print(
        "Expert waist max abs target:",
        waist_max_abs,
    )

    print(
        "Training frames:",
        len(
            training_indices
        ),
    )

    print(
        "Validation frames:",
        len(
            validation_indices
        ),
    )

    print(
        "Device:",
        device,
    )

    print(
        f"Network: 3 -> "
        f"{args.hidden_dim} -> "
        f"{args.hidden_dim} -> "
        f"{args.hidden_dim} -> 12"
    )

    print(
        "MuJoCo physics in training: NONE"
    )

    print(
        "mj_step in training: NONE"
    )

    print(
        "=" * 80
    )

    print()

    (
        best_epoch,
        best_validation_loss,
    ) = train_stage(
        model=
            model,

        x_train=
            x_train,

        y_train=
            y_train,

        x_val=
            x_validation,

        y_val=
            y_validation,

        y_mean=
            y_mean,

        y_std=
            y_std,

        device=
            device,

        epochs=
            args.epochs,

        learning_rate=
            args.learning_rate,

        weight_decay=
            args.weight_decay,

        print_every=
            args.print_every,
    )

    validation_prediction = predict(
        model=
            model,

        x_raw=
            x_validation_raw,

        x_mean=
            x_mean,

        x_std=
            x_std,

        y_mean=
            y_mean,

        y_std=
            y_std,

        device=
            device,
    )

    full_prediction_best = predict(
        model=
            model,

        x_raw=
            observations,

        x_mean=
            x_mean,

        x_std=
            x_std,

        y_mean=
            y_mean,

        y_std=
            y_std,

        device=
            device,
    )

    validation_metrics = metrics(
        validation_prediction,
        y_validation_raw,
    )

    best_full_metrics = metrics(
        full_prediction_best,
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

        learned_joint_names=
            learned_joint_names,

        fps=
            fps,

        stage=
            "best_interleaved_validation",

        validation_metrics=
            validation_metrics,

        full_metrics=
            best_full_metrics,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "BEST 12-DOF VALIDATION MODEL"
    )

    print(
        "=" * 80
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best normalized "
        "validation MSE:",
        best_validation_loss,
    )

    print(
        "Validation RMSE [rad]:",
        f"{validation_metrics['rmse_rad']:.8f}",
    )

    print(
        "Validation MAE [rad]:",
        f"{validation_metrics['mae_rad']:.8f}",
    )

    print(
        "Validation max abs "
        "error [rad]:",
        f"{validation_metrics['max_abs_error_rad']:.8f}",
    )

    print(
        "Saved:",
        best_path,
    )

    print(
        "=" * 80
    )

    print()

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

        learning_rate=
            args.refit_learning_rate,

        weight_decay=
            args.weight_decay,

        print_every=
            args.print_every,
    )

    final_prediction = predict(
        model=
            model,

        x_raw=
            observations,

        x_mean=
            x_mean,

        x_std=
            x_std,

        y_mean=
            y_mean,

        y_std=
            y_std,

        device=
            device,
    )

    final_metrics = metrics(
        final_prediction,
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

        learned_joint_names=
            learned_joint_names,

        fps=
            fps,

        stage=
            "final_refit_all_300_frames",

        validation_metrics=
            validation_metrics,

        full_metrics=
            final_metrics,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL 12-DOF KINEMATIC BC POLICY"
    )

    print(
        "=" * 80
    )

    print(
        "Saved:",
        final_path,
    )

    print(
        "Learned leg-joint "
        "RMSE [rad]:",
        f"{final_metrics['rmse_rad']:.8f}",
    )

    print(
        "Learned leg-joint "
        "MAE [rad]:",
        f"{final_metrics['mae_rad']:.8f}",
    )

    print(
        "Learned leg-joint "
        "max abs error [rad]:",
        f"{final_metrics['max_abs_error_rad']:.8f}",
    )

    print()

    print(
        "Waist yaw/roll/pitch are not "
        "learned outputs because their "
        "expert targets are identically zero."
    )

    print(
        "The final policy learns the 12 "
        "time-varying leg joints and keeps "
        "the waist at its demonstrated "
        "constant pose."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
