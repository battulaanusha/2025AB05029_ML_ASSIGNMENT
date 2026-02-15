Model folder
============

This folder contains the training notebook and (optionally) trained model artifacts.

Files
- `breast_classification.ipynb` — training and evaluation notebook.
- `*_model.pkl` — serialized model artifacts (Logistic, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost). These are optional and not required in the repo if you host them externally.

Deployment guidance
- For Streamlit Cloud: host model files in a GitHub Release or cloud storage (S3/Google Cloud) and set the environment variable `MODEL_BASE_URL` in Streamlit Cloud to the base URL containing the model files. The app will attempt to download missing model files from that URL on startup.
- Alternatively, keep models locally while developing and do not commit them to the repo for public deployments — use Git LFS or releases instead.

To recreate models locally
- Run `breast_classification.ipynb` (cells save `*.pkl` files using `joblib.dump`). Ensure package versions match `requirements.txt`.
