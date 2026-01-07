"""
Module: Single Compound Prediction
Real-time bioactivity inference using the trained XGBoost model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import get_model_loader
from utils.preprocessing import get_feature_ranges, get_confidence_level, preprocess_input

# Page config
st.set_page_config(page_title="Model Prediction", page_icon=None, layout="wide")

st.title("Model Prediction")
st.markdown("Enter compound properties to predict whether it will be **active** or **inactive**.")

# -------------------------------------------------------------------------
# Load Resources (Cached)
# -------------------------------------------------------------------------
@st.cache_resource
def load_models():
    loader = get_model_loader()
    return loader.xgboost, loader.scaler, loader.encoders

try:
    model, scaler, encoders = load_models()
    st.success("✅ Model loaded successfully (ROC-AUC: 0.824)")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Get validation ranges for inputs
ranges = get_feature_ranges()

# -------------------------------------------------------------------------
# Input Form
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("Enter Compound Properties")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Physical Properties**")
    mw_freebase = st.number_input("Molecular Weight", value=float(ranges['mw_freebase']['default']), step=1.0)
    alogp = st.number_input("LogP", value=float(ranges['alogp']['default']), step=0.1)
    psa = st.number_input("Polar Surface Area", value=float(ranges['psa']['default']), step=1.0)

with col2:
    st.markdown("**Hydrogen Bonding**")
    hba = st.number_input("H-Bond Acceptors", value=5, step=1)
    hbd = st.number_input("H-Bond Donors", value=2, step=1)
    rtb = st.number_input("Rotatable Bonds", value=5, step=1)

with col3:
    st.markdown("**Structural Features**")
    aromatic_rings = st.number_input("Aromatic Rings", value=2, step=1)
    heavy_atoms = st.number_input("Heavy Atoms", value=25, step=1)
    qed_weighted = st.slider("QED Score", 0.0, 1.0, 0.5)

st.markdown("---")

# -------------------------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------------------------
if st.button("Predict Bioactivity", type="primary", use_container_width=True):
    # Prepare input dictionary
    input_dict = {
        'mw_freebase': mw_freebase, 'alogp': alogp, 'hba': hba, 'hbd': hbd,
        'psa': psa, 'rtb': rtb, 'aromatic_rings': aromatic_rings,
        'heavy_atoms': heavy_atoms, 'qed_weighted': qed_weighted, 'num_ro5_violations': 0
    }
    
    try:
        # Preprocess and Predict
        X_scaled = preprocess_input(input_dict, scaler, encoders)
        prediction = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        prob_active = probs[1]
        
        # Display Results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success(f"### Active\n**Probability:** {prob_active*100:.1f}%")
            else:
                st.error(f"### Inactive\n**Probability:** {probs[0]*100:.1f}%")
        
        with col2:
            st.info(f"### Confidence\n**Level:** {get_confidence_level(prob_active)}")
        
        with col3:
            st.info("### Model\n**ROC-AUC:** 0.824")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
