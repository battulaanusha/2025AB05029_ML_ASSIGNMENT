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
    # Data Cleaning
    # ----------------------------
    df = df.dropna(axis=1, how='all')

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if df["diagnosis"].dtype == "object":
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

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

    model = joblib.load(model_paths[model_option])

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred

    # -------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
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
