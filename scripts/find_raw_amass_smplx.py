from pathlib import Path
import numpy as np

roots = [
    Path("datasets"),
    Path(r"C:\Projects"),
]

print("=" * 100)
print("SEARCHING FOR RAW AMASS / SMPL-X FILES")
print("=" * 100)

found = []

for root in roots:
    if not root.exists():
        continue

    for path in root.rglob("*.npz"):

        # Skip obviously processed G1 files.
        text = str(path).lower()

        if any(x in text for x in [
            "validated_29dof",
            "processed",
            "amass_g1",
            "retarget",
        ]):
            continue

        try:
            with np.load(path, allow_pickle=True) as f:
                keys = set(f.files)

                # Typical AMASS/SMPL-X style signals.
                score = 0

                for key in [
                    "poses",
                    "trans",
                    "betas",
                    "gender",
                    "mocap_frame_rate",
                    "mocap_framerate",
                ]:
                    if key in keys:
                        score += 1

                if score >= 3:
                    found.append(
                        (
                            score,
                            path,
                            sorted(keys),
                        )
                    )

        except Exception:
            pass


found.sort(
    key=lambda x: (-x[0], str(x[1]))
)

print()
print("Candidates:", len(found))

for i, (score, path, keys) in enumerate(found[:50], start=1):
    print()
    print(f"{i:02d}. score={score}")
    print("   ", path)
    print("    keys:", keys)

if not found:
    print()
    print("NO RAW SMPL-X AMASS DATA FOUND.")
    print("You will need the SMPL-X AMASS version before GMR retargeting.")

print()
print("=" * 100)
