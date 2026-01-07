"""
Batch Prediction Page
Upload CSV for bulk predictions
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_loader import get_model_loader
from utils.preprocessing import preprocess_data, get_feature_ranges

st.set_page_config(page_title="Batch Prediction", page_icon=None, layout="wide")
st.title("Batch Prediction")
st.markdown("Upload a CSV with compound properties to get bulk predictions.")

# Load Model
try:
    loader = get_model_loader()
    model = loader.xgboost
    scaler = loader.scaler
    encoders = loader.encoders
    st.success("Model loaded")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Download Template
ranges = get_feature_ranges()
template_cols = list(ranges.keys())
template = pd.DataFrame(columns=template_cols)
st.download_button("Download Template", template.to_csv(index=False), "template.csv")

# Upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info(f"Loaded {len(df)} rows.")
    
    # Check for minimal required columns
    required = ['mw_freebase', 'alogp'] # Minimal check
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        if st.button("Predict All"):
            try:
                # Use robust centralized preprocessing
                # Takes care of missing cols, ordering, encoding, scaling
                X = preprocess_data(df, scaler, encoders)
                
                # Predict
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]
                
                # Attach results
                df['prediction'] = ['Active' if p==1 else 'Inactive' for p in preds]
                df['probability'] = probs
                
                st.dataframe(df)
                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
