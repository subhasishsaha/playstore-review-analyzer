"""
Automation script to push model artifacts to Hugging Face Hub.
"""
from pathlib import Path
from huggingface_hub import HfApi

# CONFIGURATION
# -----------------------------------------------
YOUR_USERNAME = "ssaha007"
NEW_MODEL_REPO = "playstore-models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FOLDER = str(PROJECT_ROOT / "Models")
# -----------------------------------------------

def upload_models():
    """Uploads local Models/ directory to Hugging Face Hub."""
    repo_id = f"{YOUR_USERNAME}/{NEW_MODEL_REPO}"
    api = HfApi()

    print(f"Creating repo: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading folder: {LOCAL_FOLDER}...")
    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=repo_id,
        repo_type="model"
    )
    print("✅ Upload Complete!")


if __name__ == "__main__":
    upload_models()