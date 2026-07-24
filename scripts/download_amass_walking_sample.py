import os
from huggingface_hub import hf_hub_download


REPO_ID = "ember-lab-berkeley/AMASS_Retargeted_for_G1"
REPO_TYPE = "dataset"

AMASS_FILE = "g1/ACCAD/Female1Walking_c3d/B1-standtowalk_poses_120_jpos.npz"

LOCAL_DIR = os.path.join("datasets", "raw", "amass_g1")


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("Downloading AMASS retargeted G1 sample file...")
    print("Source:", AMASS_FILE)

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=AMASS_FILE,
        local_dir=LOCAL_DIR,
    )

    print("Download complete.")
    print("Saved to:", downloaded_path)


if __name__ == "__main__":
    main()