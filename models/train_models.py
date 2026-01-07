"""
ML Pipeline: Training
Script to clean data, engineer features, train models (LogReg/XGBoost), and save artifacts.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = '../data/sample_bioactivity.csv'
MODEL_DIR = 'saved/'

def load_and_clean_data(filepath):
    """Loads CSV and removes rows with missing labels."""
    df = pd.read_csv(filepath)
    return df.dropna(subset=['standard_type', 'is_active'])

def feature_engineering_pipeline(df):
    """Creates interaction, efficiency, and complexity features."""
    df = df.copy()
    df['mw_logp_interaction'] = df['mw_freebase'] * df['alogp']
    df['binding_efficiency'] = df['psa'] / (df['mw_freebase'] + 1e-6)
    df['complexity_score'] = df['aromatic_rings'] + df['heavy_atoms'] + df['rtb']
    return df

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Pipeline Start
    df = load_and_clean_data(DATA_PATH)
    df = feature_engineering_pipeline(df)
    
    # 2. Leakage Removal (Critical: drop pchembl_value and proxies)
    leakage_cols = ['molecule_chembl_id', 'standard_type', 'standard_relation', 
                    'standard_value', 'standard_units', 'pchembl_value', 
                    'target_pref_name', 'target_type', 'organism', 'confidence_score_y']
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns] + ['is_active'])
    y = df['is_active']
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler.feature_names_in_ = X.columns.tolist()
    
    # 5. Baseline (LogReg)
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
    print(f"LogReg ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]):.3f}")
    
    # 6. Advanced (XGBoost)
    xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False).fit(X_train_scaled, y_train)
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1]):.3f}")
    
    # 7. Save
    joblib.dump(lr, f'{MODEL_DIR}logreg.pkl')
    joblib.dump(xgb_model, f'{MODEL_DIR}xgboost.pkl')
    joblib.dump(scaler, f'{MODEL_DIR}scaler.pkl')
    joblib.dump({}, f'{MODEL_DIR}label_encoders.pkl')
    print("✅ Training Complete.")

if __name__ == "__main__":
    train()
