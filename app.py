import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Breast Cancer ML App", layout="wide")

st.title("Breast Cancer Classification App")
st.markdown("Upload TEST dataset (CSV) to evaluate trained models.")

# -------------------------------------------------
# Model Selection Dropdown
# -------------------------------------------------
model_option = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbor",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# -------------------------------------------------
# Dataset Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    # ----------------------------
    # Data Cleaning & Validation
    # ----------------------------
    df = df.dropna(axis=1, how='all')

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if "diagnosis" not in df.columns:
        st.error("Uploaded CSV must contain a 'diagnosis' column with labels (M/B or 1/0).")
        st.stop()

    if df["diagnosis"].dtype == "object":
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isnull().any():
        st.error("Found missing values in 'diagnosis' column. Please clean the CSV and retry.")
        st.stop()

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"].astype(int)

    # -------------------------------------------------
    # Load Trained Model
    # -------------------------------------------------
    model_paths = {
        "Logistic Regression": "model/logistic_model.pkl",
        "Decision Tree": "model/decision_tree_model.pkl",
        "K-Nearest Neighbor": "model/knn_model.pkl",
        "Naive Bayes": "model/naive_bayes_model.pkl",
        "Random Forest": "model/random_forest_model.pkl",
        "XGBoost": "model/xgboost_model.pkl"
    }


    @st.cache_resource
    def load_model(path: str):
        if not os.path.exists(path):
            # Try to download from MODEL_BASE_URL if provided (useful for Streamlit Cloud)
            base_url = os.environ.get("MODEL_BASE_URL")
            if base_url:
                from urllib.request import urlretrieve
                fname = os.path.basename(path)
                url = base_url.rstrip("/") + "/" + fname
                try:
                    MODEL_DIR = os.path.dirname(path)
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    urlretrieve(url, path)
                except Exception as e:
                    raise FileNotFoundError(f"Model file not found locally and download failed: {e}")
            else:
                raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    try:
        model = load_model(model_paths[model_option])
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Ensure model files are available under the `model/` directory or run the training notebook.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        try:
            y_prob = model.decision_function(X)
        except Exception:
            y_prob = y_pred
    else:
        y_prob = y_pred

    # -------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------
    accuracy = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = float('nan')
        st.warning("AUC could not be computed for the selected model/input (check predicted probabilities).")

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "AUC Score",
            "Precision",
            "Recall",
            "F1 Score",
            "Matthews Correlation Coefficient"
        ],
        "Value": [
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ]
    })

    st.table(metrics_df)

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

else:
    st.info("Please upload a test dataset CSV file to continue.")
