from huggingface_hub import HfApi

# CONFIGURATION
# -----------------------------------------------
YOUR_USERNAME = "ssaha007"
NEW_MODEL_REPO = "playstore-models" 
LOCAL_FOLDER = "Models"  
# -----------------------------------------------

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