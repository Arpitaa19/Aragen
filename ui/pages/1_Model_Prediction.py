"""
Single Compound Prediction Page
User inputs compound properties and gets instant bioactivity prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_loader import get_model_loader
from utils.preprocessing import get_feature_ranges, get_confidence_level

# Page config
st.set_page_config(page_title="Model Prediction", page_icon=None, layout="wide")

st.title("Model Prediction")
st.markdown("Enter compound properties to predict whether it will be **active** or **inactive**.")

# Load models
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

# Get feature ranges
ranges = get_feature_ranges()

# Create input form
st.markdown("---")
st.subheader("Enter Compound Properties")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Physical Properties**")
    mw_freebase = st.number_input(
        "Molecular Weight (Da)",
        min_value=float(ranges['mw_freebase']['min']),
        max_value=float(ranges['mw_freebase']['max']),
        value=float(ranges['mw_freebase']['default']),
        step=float(ranges['mw_freebase']['step'])
    )
    
    alogp = st.number_input(
        "LogP (Lipophilicity)",
        min_value=float(ranges['alogp']['min']),
        max_value=float(ranges['alogp']['max']),
        value=float(ranges['alogp']['default']),
        step=float(ranges['alogp']['step'])
    )
    
    psa = st.number_input(
        "Polar Surface Area (Ų)",
        min_value=float(ranges['psa']['min']),
        max_value=float(ranges['psa']['max']),
        value=float(ranges['psa']['default']),
        step=float(ranges['psa']['step'])
    )

with col2:
    st.markdown("**Hydrogen Bonding**")
    hba = st.number_input("H-Bond Acceptors", min_value=0, max_value=20, value=5, step=1)
    hbd = st.number_input("H-Bond Donors", min_value=0, max_value=10, value=2, step=1)
    rtb = st.number_input("Rotatable Bonds", min_value=0, max_value=30, value=5, step=1)

with col3:
    st.markdown("**Structural Features**")
    aromatic_rings = st.number_input("Aromatic Rings", min_value=0, max_value=10, value=2, step=1)
    heavy_atoms = st.number_input("Heavy Atoms", min_value=5, max_value=100, value=25, step=1)
    qed_weighted = st.slider("QED Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.markdown("---")

# Predict button
if st.button("Predict Bioactivity", type="primary", use_container_width=True):
    
    # Prepare input
    input_dict = {
        'mw_freebase': mw_freebase,
        'alogp': alogp,
        'hba': hba,
        'hbd': hbd,
        'psa': psa,
        'rtb': rtb,
        'aromatic_rings': aromatic_rings,
        'heavy_atoms': heavy_atoms,
        'qed_weighted': qed_weighted,
        'num_ro5_violations': 0 # Will be recalculated if needed or aliased
    }
    
    try:
        # Centralized Robust Preprocessing
        from utils.preprocessing import preprocess_input
        X_scaled = preprocess_input(input_dict, scaler, encoders)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        prob_inactive = probability[0]
        prob_active = probability[1]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success(f"### ACTIVE\n**Probability:** {prob_active*100:.1f}%")
            else:
                st.error(f"### INACTIVE\n**Probability:** {prob_inactive*100:.1f}%")
        
        with col2:
            confidence = get_confidence_level(prob_active)
            st.info(f"### Confidence: {confidence}")
        
        with col3:
            st.metric("Model", "XGBoost", delta="ROC-AUC: 0.824")
        
        # Probability chart
        st.bar_chart(pd.DataFrame({
            'Inactive': [prob_inactive],
            'Active': [prob_active]
        }))
        

        
    except Exception as e:
        st.error(f" Prediction failed: {e}")

# Sidebar

