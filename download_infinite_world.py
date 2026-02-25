import os
from huggingface_hub import snapshot_download

def download_infinite_world(repo_id, target_dir):
    """
    Downloads the specified Hugging Face repository to the target directory.

    Args:
        repo_id (str): The Hugging Face repository ID.
        target_dir (str): The directory where the repository will be downloaded.
    """
    print(f"Downloading repository '{repo_id}' to '{target_dir}'...")
    snapshot_download(repo_id=repo_id, local_dir=target_dir)
    print("Download complete.")

if __name__ == "__main__":
    # Repository ID and target directory
    REPO_ID = "MeiGen-AI/Infinite-World"
    TARGET_DIR = "./checkpoints"

    # Ensure the target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Download the repository
    download_infinite_world(REPO_ID, TARGET_DIR)
