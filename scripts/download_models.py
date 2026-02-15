"""
Download model artifacts to `model/` from a base URL.

Expected usage:
  - Host your model files (logistic_model.pkl, decision_tree_model.pkl, etc.) on GitHub Releases or cloud storage
  - Set environment variable MODEL_BASE_URL to the folder URL where files live (must be direct download links)
  - Run: `python scripts/download_models.py`

This script will download the required files into the `model/` directory.
"""
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "model"

files = [
    "logistic_model.pkl",
    "decision_tree_model.pkl",
    "knn_model.pkl",
    "naive_bayes_model.pkl",
    "random_forest_model.pkl",
    "xgboost_model.pkl",
]

def main():
    base_url = os.environ.get("MODEL_BASE_URL")
    if not base_url:
        print("Please set MODEL_BASE_URL environment variable to the base URL where model files are hosted.")
        print("Example:")
        print("  export MODEL_BASE_URL=https://my-bucket.s3.amazonaws.com/models/")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fn in files:
        url = base_url.rstrip("/") + "/" + fn
        dest = MODEL_DIR / fn
        try:
            print(f"Downloading {url} -> {dest}")
            urlretrieve(url, dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    print("Done. Verify files in model/ folder.")

if __name__ == "__main__":
    main()
