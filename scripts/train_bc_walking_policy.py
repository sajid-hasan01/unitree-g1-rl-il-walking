import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class WalkingILDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path, allow_pickle=True)

        self.obs = data["il_observations"].astype(np.float32)
        self.actions = data["il_actions"].astype(np.float32)

        self.obs_mean = self.obs.mean(axis=0, keepdims=True)
        self.obs_std = self.obs.std(axis=0, keepdims=True) + 1e-8

        self.action_mean = self.actions.mean(axis=0, keepdims=True)
        self.action_std = self.actions.std(axis=0, keepdims=True) + 1e-8

        self.obs_norm = (self.obs - self.obs_mean) / self.obs_std
        self.actions_norm = (self.actions - self.action_mean) / self.action_std

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return (
            torch.tensor(self.obs_norm[index], dtype=torch.float32),
            torch.tensor(self.actions_norm[index], dtype=torch.float32),
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(
            "datasets",
            "processed",
            "g1_amass_walking_il_15dof.npz",
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "models",
            "g1_bc_walking_policy.pt",
        ),
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WalkingILDataset(args.dataset)

    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BCWalkingPolicy(obs_dim, action_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    print("Training Behavior Cloning walking policy")
    print("Dataset:", args.dataset)
    print("Samples:", len(dataset))
    print("Train samples:", train_size)
    print("Validation samples:", val_size)
    print("Observation dim:", obs_dim)
    print("Action dim:", action_dim)
    print("Device:", device)
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            pred_action = model(obs_batch)
            loss = loss_fn(pred_action, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * obs_batch.size(0)

        train_loss = train_loss_sum / train_size

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)

                pred_action = model(obs_batch)
                loss = loss_fn(pred_action, action_batch)

                val_loss_sum += loss.item() * obs_batch.size(0)

        val_loss = val_loss_sum / val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "obs_mean": dataset.obs_mean.astype(np.float32),
                "obs_std": dataset.obs_std.astype(np.float32),
                "action_mean": dataset.action_mean.astype(np.float32),
                "action_std": dataset.action_std.astype(np.float32),
                "controlled_joint_names": np.load(
                    args.dataset,
                    allow_pickle=True,
                )["controlled_joint_names"],
                "dataset": args.dataset,
                "best_val_loss": best_val_loss,
            }

            torch.save(checkpoint, args.output)

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:04d}/{args.epochs} | "
                f"train_loss={train_loss:.8f} | "
                f"val_loss={val_loss:.8f} | "
                f"best_val_loss={best_val_loss:.8f}"
            )

    print()
    print("Training complete.")
    print("Best validation loss:", best_val_loss)
    print("Saved BC policy:", args.output)


if __name__ == "__main__":
    main()