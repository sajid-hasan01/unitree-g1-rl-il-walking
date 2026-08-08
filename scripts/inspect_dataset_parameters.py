from pathlib import Path
import argparse
import numpy as np


def format_value(x):
    arr = np.asarray(x)

    if arr.shape == ():
        return str(arr.item())

    if arr.dtype == object or arr.dtype.kind in {"U", "S"}:
        return str(arr.tolist())

    return str(arr)


def print_array_summary(name, arr):
    arr = np.asarray(arr)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")

    if arr.shape == ():
        print(f"value: {format_value(arr)}")
        return

    if arr.dtype == object or arr.dtype.kind in {"U", "S"}:
        print(f"value: {format_value(arr)}")
        return

    if np.issubdtype(arr.dtype, np.number):
        print(f"min:  {np.min(arr): .6f}")
        print(f"max:  {np.max(arr): .6f}")
        print(f"mean: {np.mean(arr): .6f}")
        print(f"std:  {np.std(arr): .6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    path = Path(args.dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, allow_pickle=True)

    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)
    print(f"Path: {path}")
    print(f"File size: {path.stat().st_size / 1024:.2f} KB")

    print("\nKeys:")
    for key in data.files:
        print(f"  - {key}")

    print("\n" + "=" * 80)
    print("ARRAY SUMMARIES")
    print("=" * 80)

    for key in data.files:
        print_array_summary(key, data[key])

    print("\n" + "=" * 80)
    print("DERIVED PARAMETERS")
    print("=" * 80)

    fps = None
    if "fps" in data.files:
        fps = float(np.asarray(data["fps"]).item())
        print(f"FPS: {fps}")

    if "joint_pos_15" in data.files:
        joint_pos = np.asarray(data["joint_pos_15"])
        print(f"Frames: {joint_pos.shape[0]}")
        print(f"Controlled DOF: {joint_pos.shape[1]}")

        if fps is not None:
            duration = joint_pos.shape[0] / fps
            print(f"Duration: {duration:.3f} seconds")

    if "root_positions" in data.files:
        root = np.asarray(data["root_positions"])

        start = root[0]
        end = root[-1]
        delta = end - start

        xy_disp = float(np.linalg.norm(delta[:2]))

        print("\nRoot Motion:")
        print(f"start xyz: {start}")
        print(f"end xyz:   {end}")
        print(f"delta xyz: {delta}")
        print(f"xy displacement: {xy_disp:.6f} m")

        if fps is not None:
            duration = root.shape[0] / fps
            avg_speed = xy_disp / duration
            print(f"average xy speed: {avg_speed:.6f} m/s")

    if "contact_mask" in data.files:
        contact = np.asarray(data["contact_mask"])
        print("\nContact Mask:")
        print(f"shape: {contact.shape}")
        print(f"mean contact per column: {np.mean(contact, axis=0)}")
        print(f"first 10 rows:\n{contact[:10]}")

    if "controlled_joint_names" in data.files:
        names = np.asarray(data["controlled_joint_names"]).tolist()
        print("\nControlled Joint Names:")
        for i, name in enumerate(names):
            print(f"{i:02d}: {name}")

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
