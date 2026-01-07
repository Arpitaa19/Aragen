"""
Feature engineering and preprocessing utilities
Replicates training pipeline for consistent predictions
"""
import pandas as pd
import numpy as np

# Default values for missing features (used for inference)
DEFAULTS = {
    'confidence_score': 9,       # Assume high confidence
    'mw_freebase': 350.0,
    'alogp': 2.5,
    'hba': 5,
    'hbd': 2,
    'psa': 75.0,
    'rtb': 5,
    'aromatic_rings': 2,
    'heavy_atoms': 25,
    'qed_weighted': 0.5,
    'lipinski_violations': 0,
    'full_mwt': 350.0,
    'np_likeness_score': 0.0,
    'target_type': 'SINGLE PROTEIN',
    'organism': 'Homo sapiens',
    'assay_type': 'B',
    'ro3_pass': 'N'
}

def engineer_features(df):
    """
    Apply feature engineering (same as training)
    """
    df = df.copy()
    
    # MW × LogP interaction
    if 'mw_freebase' in df.columns and 'alogp' in df.columns:
        df['mw_logp_interaction'] = df['mw_freebase'] * df['alogp']
    
    # Binding efficiency proxy
    if 'psa' in df.columns and 'mw_freebase' in df.columns:
        df['binding_efficiency'] = df['psa'] / (df['mw_freebase'] + 1)
        
    # Molecular complexity
    if 'rtb' in df.columns and 'aromatic_rings' in df.columns:
        df['complexity_score'] = df['rtb'] * df['aromatic_rings']
        
    # Lipinski Violations (Calculated if not present)
    if 'lipinski_violations' not in df.columns:
        violations = np.zeros(len(df))
        if 'mw_freebase' in df.columns: violations += (df['mw_freebase'] > 500).astype(int)
        if 'alogp' in df.columns: violations += (df['alogp'] > 5).astype(int)
        if 'hba' in df.columns: violations += (df['hba'] > 10).astype(int)
        if 'hbd' in df.columns: violations += (df['hbd'] > 5).astype(int)
        df['lipinski_violations'] = violations
    
    return df


def preprocess_data(df, scaler, encoders):
    """
    Central preprocessing function for both Single and Batch predictions.
    Dynamically aligns input DataFrame to the model's expected signature.
    
    Args:
        df: Input DataFrame (user provided columns)
        scaler: Fitted StandardScaler (with feature_names_in_)
        encoders: Dictionary of LabelEncoders
        
    Returns:
        Scaled numpy array ready for prediction
    """
    df = df.copy()
    
    # 1. Alias Handling
    if 'num_ro5_violations' in df.columns:
        df['lipinski_violations'] = df['num_ro5_violations']
    if 'mw_freebase' in df.columns and 'full_mwt' not in df.columns:
        df['full_mwt'] = df['mw_freebase']

    # 2. Get Expected Features from Scaler
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = list(scaler.feature_names_in_)
    else:
        # Fallback if attribute missing
        expected_features = [
            'confidence_score', 'mw_freebase', 'alogp', 'hba', 'hbd', 
            'psa', 'rtb', 'aromatic_rings', 'heavy_atoms', 'qed_weighted', 
            'lipinski_violations', 'full_mwt', 'np_likeness_score', 'target_type', 
            'organism', 'assay_type', 'ro3_pass', 'mw_logp_interaction', 
            'binding_efficiency', 'complexity_score'
        ]

    # 3. Inject Missing Features (Defaults)
    for feature in expected_features:
        if feature not in df.columns:
            val = DEFAULTS.get(feature, 0)
            df[feature] = val

    # 4. Engineer Features
    df = engineer_features(df)

    # 5. Encode Categoricals
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            classes = set(encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in classes else encoder.classes_[0])
            df[col] = encoder.transform(df[col])

    # 6. Strict Reordering
    df_final = df[expected_features]

    # 7. Scale
    X_scaled = scaler.transform(df_final)
    
    return X_scaled


def preprocess_input(input_dict, scaler, encoders):
    """Wrapper for single entry dictionary"""
    df = pd.DataFrame([input_dict])
    return preprocess_data(df, scaler, encoders)


def get_feature_ranges():
    """Return typical ranges for UI"""
    return {
        'mw_freebase': {'min': 100, 'max': 1000, 'default': 350, 'step': 10},
        'alogp': {'min': -5, 'max': 10, 'default': 2.5, 'step': 0.1},
        'hba': {'min': 0, 'max': 20, 'default': 5, 'step': 1},
        'hbd': {'min': 0, 'max': 10, 'default': 2, 'step': 1},
        'psa': {'min': 0, 'max': 200, 'default': 75, 'step': 5},
        'rtb': {'min': 0, 'max': 30, 'default': 5, 'step': 1},
        'aromatic_rings': {'min': 0, 'max': 10, 'default': 2, 'step': 1},
        'heavy_atoms': {'min': 5, 'max': 100, 'default': 25, 'step': 1},
        'qed_weighted': {'min': 0, 'max': 1, 'default': 0.5, 'step': 0.01},
    }

def get_confidence_level(probability):
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"
