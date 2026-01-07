import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from utils.model_loader import get_model_loader
from utils.preprocessing import preprocess_data

st.set_page_config(page_title="Model Explainability", layout="wide")
st.title("Explainability & Feature Impact")

st.markdown("""
This page explains **model predictions** using SHAP.
All explanations are computed on the **exact same feature space**
used during training and inference.
""")

# --------------------------------------------------
# Load model + preprocessors
# --------------------------------------------------
try:
    loader = get_model_loader()
    model = loader.xgboost
    scaler = loader.scaler
    encoders = loader.encoders
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

st.success("XGBoost model & preprocessors loaded")

# --------------------------------------------------
# Load dataset (same source as training)
# --------------------------------------------------
try:
    df = pd.read_csv("data/sample_bioactivity.csv")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Sample for performance
df_sample = df.sample(min(100, len(df)), random_state=42)

# --------------------------------------------------
# CRITICAL FIX: use SAME preprocessing pipeline
# --------------------------------------------------
try:
    X_processed = preprocess_data(df_sample, scaler, encoders)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# --------------------------------------------------
# SHAP Analysis
# --------------------------------------------------
st.subheader("SHAP (SHapley Additive Explanations)")
st.markdown("Understanding how each feature contributes to the prediction.")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    feature_names = list(scaler.feature_names_in_)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Feature Importance (Beeswarm)")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            X_processed,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)

    with col2:
        st.markdown("### Global Feature Importance (Bar)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            X_processed,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig2)

except ImportError:
    st.warning("SHAP library not installed.")
    st.code("pip install shap")

except Exception as e:
    st.error(f"Error calculating SHAP: {e}")
    st.info(
        "This usually means the SHAP input does not match the model feature space. "
        "Verify that training and preprocessing are aligned."
    )
