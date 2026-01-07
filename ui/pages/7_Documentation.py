import streamlit as st

st.set_page_config(page_title="Documentation", layout="wide")
st.title("Project Documentation")

# Load README content dynamically or hardcode for performance
# Hardcoding clean version for display

st.markdown("""
# BioInsight Lite: Bioactivity Prediction System

## 1. Project Overview
BioInsight Lite is a machine learning application designed to explore chemical datasets and predict the bioactivity of small molecules against specific biological targets. The system leverages the ChEMBL database (v36) to train predictive models that classify compounds as "Active" or "Inactive" based on their calculated physicochemical properties.

The key objective is to provide a user-friendly interface for researchers to screen compounds, visualize data distributions, and interpret model decisions using explainable AI techniques.

## 2. Technology Stack

*   **Programming Language:** Python 3.9+
*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Explainability:** SHAP (SHapley Additive exPlanations)
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Plotly, Matplotlib, Seaborn
*   **Database/Data Source:** ChEMBL v36 (CSV Extract)

## 3. Project Architecture

```
bioinsight_hackathon2/
├── data/
│   ├── sample_bioactivity.csv      # Processed dataset
│   └── config.py                   # Data configuration
├── models/
│   ├── train_models.py             # Training pipeline script
│   └── saved/                      # Model artifacts (pkl)
├── notebooks/
│   └── 01_workflow.ipynb           # Analysis notebook
├── ui/
│   ├── Home.py                     # Application Entry Point
│   ├── pages/
│   │   ├── 1_Model_Prediction.py   # Single compound inference
│   │   ├── 2_Batch_Prediction.py   # Bulk CSV inference
│   │   ├── 3_Exploratory_Data.py   # Data visualization
│   │   ├── 4_Explainability.py     # SHAP analysis
│   │   ├── 5_Dataset_Search.py     # Filter & Predict tool
│   │   ├── 6_Model_Evaluation.py   # Performance metrics
│   │   └── 7_Documentation.py      # User guide
│   └── utils/
│       ├── preprocessing.py        # Feature engineering
│       └── model_loader.py         # Model singleton
└── requirements.txt                # Dependencies
```

## 4. Methodology & Approach

### 4.1 Data Pipeline
*   **Source:** Raw data extracted from ChEMBL database.
*   **Cleaning:** Removal of ambiguous activity values. Leakage prevention by dropping 'pchembl_value' and 'standard_value'.
*   **Sampling:** Stratified sampling to maintain class distribution.

### 4.2 Preprocessing & Feature Engineering
*   **Imputation:** Missing numerical values (e.g., 'psa', 'alogp') imputed with domain defaults.
*   **Feature Engineering:**
    *   `mw_logp_interaction` (Size vs Lipophilicity).
    *   `binding_efficiency` (PSA / MW).
*   **Scaling:** StandardScaling applied to all numerical features.
*   **Encoding:** Label Encoding for categorical variables.

### 4.3 Model Development
1.  **Baseline: Logistic Regression**
    *   Simple linear decision boundary.
    *   **ROC-AUC:** 0.663

2.  **Advanced: XGBoost Classifier**
    *   Gradient boosted decision trees.
    *   **ROC-AUC:** 0.824 (Selected for Production)

## 5. Explainability
To ensure trust in model predictions, SHAP is integrated to provide:
*   **Global Importance:** Identifying drivers across the dataset.
*   **Local Importance:** Explaining individual predictions.

---
**Submission for Aragen Hackathon**
""")
