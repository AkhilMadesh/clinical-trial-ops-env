"""
Run this to upload ALL files to HuggingFace Space.
Usage:
    pip install huggingface_hub
    python upload_to_hf.py
"""
import os
from huggingface_hub import HfApi

HF_TOKEN    = "your_hf_write_token_here"
HF_USERNAME = "akhil13-19"
SPACE_NAME  = "clinical-trial-ops-env"

REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
api = HfApi(token=HF_TOKEN)

FILES = [
    "Dockerfile",
    "README.md",
    "api.py",
    "inference.py",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    "env/__init__.py",
    "env/models.py",
    "env/environment.py",
    "env/protocol_db.py",
    "env/case_generator.py",
    "env/graders/__init__.py",
    "env/graders/eligibility.py",
    "env/graders/lab_dose.py",
    "env/graders/sae_triage.py",
    "tasks/task1_eligibility.yaml",
    "tasks/task2_lab_dose.yaml",
    "tasks/task3_sae_triage.yaml",
    "static/index.html",
]

print(f"Uploading {len(FILES)} files to: https://huggingface.co/spaces/{REPO_ID}\n")

for filepath in FILES:
    if not os.path.exists(filepath):
        print(f"  MISSING: {filepath}")
        continue
    try:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filepath,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Add {filepath}",
        )
        print(f"  uploaded: {filepath}")
    except Exception as e:
        print(f"  ERROR {filepath}: {e}")

print(f"\nDone! Space: https://huggingface.co/spaces/{REPO_ID}")
print(f"Live URL: https://{HF_USERNAME}-{SPACE_NAME}.hf.space/health")
