from pathlib import Path
import numpy as np

ROOT = Path(
    r"C:\Projects\unitree-g1-rl-il-walking"
)

DATA = (
    ROOT
    / "datasets"
    / "raw"
    / "amass_smplx"
    / "KIT"
)

print("=" * 110)
print("AMASS SMPL-X KIT VALIDATION")
print("=" * 110)

print("directory:", DATA)
print("exists:", DATA.exists())

if not DATA.exists():
    raise SystemExit(
        "\nKIT SMPL-X folder does not exist yet."
    )

files = list(
    DATA.rglob("*.npz")
)

print("npz files:", len(files))

if not files:
    raise SystemExit(
        "\nNo NPZ motion files found."
    )

print()
print("=" * 110)
print("FIRST FILES")
print("=" * 110)

valid = []

for index, path in enumerate(files[:30], start=1):

    try:
        with np.load(
            path,
            allow_pickle=True,
        ) as f:

            keys = list(f.files)

            print()
            print(f"{index:02d}. {path}")
            print("keys:")
            print(keys)

            shapes = {}

            for key in keys:
                try:
                    value = np.asarray(f[key])
                    shapes[key] = value.shape
                except Exception:
                    shapes[key] = "<unreadable>"

            print("shapes:")
            for key, shape in shapes.items():
                print(
                    f"  {key:30s}",
                    shape,
                )

            # Broad SMPL-X/AMASS indicators.
            likely = any(
                key in keys
                for key in [
                    "poses",
                    "trans",
                    "transl",
                    "global_orient",
                    "body_pose",
                ]
            )

            if likely:
                valid.append(path)

    except Exception as exc:

        print(
            "ERROR:",
            path,
            repr(exc),
        )


print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)

print(
    "likely raw human-motion files:",
    len(valid),
)

if valid:

    print()
    print("Example candidate:")
    print(valid[0])

    print()
    print("STATUS: RAW KIT DATA FOUND.")

else:

    print()
    print(
        "STATUS: NPZ files exist, but their structure "
        "does not yet look like raw SMPL-X/AMASS."
    )
