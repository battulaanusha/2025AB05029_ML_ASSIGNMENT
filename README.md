# Machine Learning Assignment
## Breast Cancer Classification using Multiple ML Models

---

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models on the Breast Cancer Wisconsin (Diagnostic) dataset.  

The goal is to predict whether a tumor is **Malignant (1)** or **Benign (0)** using clinical measurement features extracted from digitized images of breast masses.

This project demonstrates:
- Implementation of six classification algorithms
- Model evaluation using multiple performance metrics
- Comparative analysis of models
- Deployment of trained models using Streamlit

---

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** Kaggle Machine Learning Repository  

### Dataset Characteristics:
- Total Instances: 569
- Total Features: 30 numerical features
- Target Variable: `diagnosis`
  - M = Malignant (encoded as 1)
  - B = Benign (encoded as 0)
- Feature Type: Continuous numerical values
- Missing Values: None (after cleaning)
- Train-Test Split: 80% training, 20% testing (Stratified)

The dataset contains computed features such as:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

These features describe characteristics of cell nuclei present in breast cancer biopsies.

---

## c. Models Used

The following classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

### Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Comparison Table



| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------           |----------|--------|-----------|--------|----------|--------|
| Logistic Regression      | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512   | 0.9245 |
| Decision Tree            | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048   | 0.8492 |
| KNN                      | 0.9561   | 0.9823 | 0.9744    | 0.9048 | 0.9383   | 0.9058 |
| Naive Bayes              | 0.9386   | 0.9934 | 1.0000    | 0.8333 | 0.9091   | 0.8715 |
| Random Forest (Ensemble) | 0.9649   | 0.9942 | 1.0000    | 0.9048 | 0.9500   | 0.9258 |
| XGBoost (Ensemble)       | 0.9649   | 0.9937 | 1.0000    | 0.9048 | 0.9500   | 0.9258 |


---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved very high AUC (0.9960) and a good balance between Precision (0.9750) and Recall (0.9286). It demonstrates excellent generalization and performs almost as well as the best ensemble models. |
| Decision Tree | Lower performance overall (Accuracy: 0.9298, MCC: 0.8492). While simple and interpretable, it is less stable than ensemble models and more prone to overfitting. |
| KNN | Performed strongly (Accuracy: 0.9561) and has good Precision (0.9744), but slightly lower Recall (0.9048) than top models. |
| Naive Bayes | Achieved perfect Precision (1.0000) but lower Recall (0.8333), meaning it is conservative and may miss some positive cases; still strong AUC (0.9934). |
| Random Forest (Ensemble) | One of the best models overall with high Accuracy (0.9649), perfect Precision (1.0000), and strong MCC (0.9258). Less prone to overfitting compared to the Decision Tree. |
| XGBoost (Ensemble) | Equal performance to Random Forest (Accuracy: 0.9649, MCC: 0.9258) with very high AUC (0.9937). Shows robust ensemble performance and strong overall results. |

---



## Repository Structure

2025AB05029_ML_ASSIGNMENT/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── model/
  ├── breast_classification.ipynb
  ├── logistic_model.pkl
  ├── decision_tree_model.pkl
  ├── knn_model.pkl
  ├── naive_bayes_model.pkl
  ├── random_forest_model.pkl
  └── xgboost_model.pkl


---


## How to Run the Application Locally

1. Create and activate a virtual environment (recommended):

Windows (cmd):

```
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux (bash):

```
python -m venv .venv
source .venv/bin/activate
```

2. Install required dependencies (pin versions in `requirements.txt` for reproducibility):

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

4. Upload the provided `test_data.csv` file in the app interface.

---

## Deployment

The trained models were saved as `.pkl` files and used by the Streamlit application.

Notes about model artifacts:
- Storing large binary model files (`*.pkl`) in a Git repository can bloat history. Consider one of:
  - Use Git LFS for model binaries.
  - Host model files in a release or cloud storage and provide download links.
  - Add `*.pkl` to `.gitignore` and provide scripts to (re)train or download models.

The Streamlit app includes:
- Dataset upload (CSV – test dataset)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

---


## Conclusion

This project demonstrates comparative evaluation of multiple classification algorithms on a medical dataset.  

Ensemble models such as Random Forest and XGBoost generally provided superior performance, while simpler models like Logistic Regression also performed competitively.

The combination of model evaluation and deployment provides a complete end-to-end machine learning workflow.

Reproducibility: set a random seed when training (example: `random_state=42`) and record package versions in `requirements.txt`.

## References

- Dataset: Kaggle Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Dataset

