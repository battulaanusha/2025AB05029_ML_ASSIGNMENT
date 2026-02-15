"""
Simple checker to verify model artifact presence.

Usage:
    python scripts/check_models.py

Will print missing model files and exit with non-zero code if any are missing.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "model")

required = [
    "logistic_model.pkl",
    "decision_tree_model.pkl",
    "knn_model.pkl",
    "naive_bayes_model.pkl",
    "random_forest_model.pkl",
    "xgboost_model.pkl",
]

missing = []
for name in required:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        missing.append(path)

if missing:
    print("Missing model files:")
    for p in missing:
        print(" -", p)
    print("\nOptions:")
    print(" - Run `model/breast_classification.ipynb` to recreate models")
    print(" - Store model files in `model/` before running the app")
    sys.exit(2)
else:
    print("All model files present in model/ — app should run correctly.")
    sys.exit(0)
