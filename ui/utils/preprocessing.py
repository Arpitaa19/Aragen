"""
Utilities: Preprocessing & Connectivity
Centralized logic for data cleaning, feature engineering, and scaling.
"""
import pandas as pd
import numpy as np

# Default values for missing inputs
DEFAULTS = {
    'mw_freebase': 350.0, 'alogp': 3.0, 'psa': 80.0,
    'hba': 5, 'hbd': 2, 'rtb': 5,
    'aromatic_rings': 2, 'heavy_atoms': 25,
    'qed_weighted': 0.5, 'num_ro5_violations': 0,
    # Additional columns that might exist in training data
    'confidence_score': 0, 'lipinski_violations': 0,
    'full_mwt': 350.0, 'np_likeness_score': 0,
    'target_type': 0, 'organism': 0, 'assay_type': 0, 'ro3_pass': 0
}

def get_feature_ranges():
    """Returns min/max ranges for UI sliders."""
    return {
        'mw_freebase': {'min': 50.0, 'max': 900.0, 'default': 350.0, 'step': 1.0},
        'alogp': {'min': -5.0, 'max': 10.0, 'default': 3.0, 'step': 0.1},
        'psa': {'min': 0.0, 'max': 300.0, 'default': 80.0, 'step': 1.0}
    }

def get_confidence_level(p):
    """Maps probability to qualitative confidence level."""
    p = float(p)
    if p > 0.90 or p < 0.10: return "Very High"
    if p > 0.75 or p < 0.25: return "High"
    if p > 0.60 or p < 0.40: return "Medium"
    return "Low"

def engineer_features(df):
    """Applies domain-specific transformations (Interaction, Efficiency, Complexity)."""
    df = df.copy()
    df['mw_logp_interaction'] = df['mw_freebase'] * df['alogp']
    df['binding_efficiency'] = df['psa'] / (df['mw_freebase'] + 1e-6)
    df['complexity_score'] = df['aromatic_rings'] + df['heavy_atoms'] + df['rtb']
    return df

def preprocess_data(df, scaler, encoders):
    """Full pipeline for batch data: Fill NAs -> Engineer -> Scale."""
    # 1. Fill NAs
    for col, val in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val
            
    # 2. Engineer
    df = engineer_features(df)
    
    # 3. Scale - Use only the core features we know the model needs
    # These are the features after engineering that the model actually uses
    core_features = [
        'mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 'rtb',
        'aromatic_rings', 'heavy_atoms', 'qed_weighted', 'num_ro5_violations',
        'mw_logp_interaction', 'binding_efficiency', 'complexity_score'
    ]
    
    # Ensure all core features exist
    for col in core_features:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, 0)
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(DEFAULTS.get(col, 0))
    
    # Get the actual columns the scaler expects (if available)
    scaler_cols = getattr(scaler, 'feature_names_in_', core_features)
    
    # Only use columns that exist in both our core features and scaler expectations
    final_cols = [col for col in scaler_cols if col in core_features]
    
    # If scaler expects columns we don't have in core, add them with defaults
    for col in scaler_cols:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, 0)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(DEFAULTS.get(col, 0))
    
    # Select in the exact order the scaler expects
    df_for_scaling = df[list(scaler_cols)]
    
    return scaler.transform(df_for_scaling)

def preprocess_input(input_dict, scaler, encoders):
    """Pipeline for single dictionary input."""
    df = pd.DataFrame([input_dict])
    return preprocess_data(df, scaler, encoders)
