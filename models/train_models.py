#!/usr/bin/env python3
"""
Step 3: Model Training & Evaluation
Trains Logistic Regression and XGBoost models with proper class weighting.

Key Updates:
- Handles 80% active, 20% inactive distribution (real ChEMBL data)
- Uses class weights instead of SMOTE (minority is inactive, not active)
- Proper train/val/test split with stratification
- SHAP explainability for XGBoost

Author: BioInsight Hackathon Team
Date: 2026-01-07
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

import xgboost as xgb
import shap

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*60)
print("Step 3: Model Training & Evaluation")
print("="*60)


def load_and_prepare_data(data_path='data/sample_bioactivity.csv'):
    """Load data and prepare for modeling."""
    print("\n📊 Loading data...")
    
    df = pd.read_csv(data_path)
    print(f"   Loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    # Drop rows with missing target
    df = df.dropna(subset=['is_active'])
    
    # Identify columns to drop (IDs, text fields, and LEAKAGE)
    drop_cols = ['activity_id', 'compound_id', 'target_id', 'assay_id', 
                 'target_name', 'standard_type', 'standard_relation', 
                 'standard_units', 'standard_value', 'pchembl_value'] # Added pchembl_value
    
    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    # Separate features and target
    X_raw = df.drop(drop_cols + ['is_active'], axis=1)
    y = df['is_active']
    
    # ASSERTION: Ensure no leakage
    assert 'pchembl_value' not in X_raw.columns, "CRITICAL: pchembl_value leakage detected!"
    assert 'standard_value' not in X_raw.columns, "CRITICAL: standard_value leakage detected!"
    print(f"   ✅ Leakage check passed: pchembl_value removed.")
    
    print(f"   Features: {X_raw.shape[1]}")
    print(f"   Target distribution:")
    print(f"      Active (1): {(y==1).sum():,} ({100*(y==1).sum()/len(y):.1f}%)")
    print(f"      Inactive (0): {(y==0).sum():,} ({100*(y==0).sum()/len(y):.1f}%)")
    
    return X_raw, y


def split_data(X, y, test_size=0.30, val_size=0.50):
    """Split data into train/val/test with stratification."""
    print("\n🔀 Splitting data...")
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"   Train: {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
    print(f"   Val: {len(X_val):,} ({100*len(X_val)/len(X):.1f}%)")
    print(f"   Test: {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_features(X_train, X_val, X_test):
    """Handle missing values, encode categoricals, and scale features."""
    print("\n🔧 Preprocessing features...")
    
    # Identify column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"   Numeric columns: {len(num_cols)}")
    print(f"   Categorical columns: {len(cat_cols)}")
    
    # 1. Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Numeric imputation
    X_train_num = pd.DataFrame(
        num_imputer.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )
    X_val_num = pd.DataFrame(
        num_imputer.transform(X_val[num_cols]),
        columns=num_cols,
        index=X_val.index
    )
    X_test_num = pd.DataFrame(
        num_imputer.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )
    
    # 2. Encode categorical variables
    label_encoders = {}
    X_train_cat_list = []
    X_val_cat_list = []
    X_test_cat_list = []
    
    for col in cat_cols:
        le = LabelEncoder()
        
        # Fit on train, transform all
        X_train_cat_list.append(
            pd.Series(le.fit_transform(X_train[col].astype(str)), 
                     index=X_train.index, name=col)
        )
        
        # Handle unseen categories in val/test
        X_val_cat_list.append(
            pd.Series(le.transform(X_val[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )), index=X_val.index, name=col)
        )
        
        X_test_cat_list.append(
            pd.Series(le.transform(X_test[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )), index=X_test.index, name=col)
        )
        
        label_encoders[col] = le
    
    # Combine numeric and categorical
    if cat_cols:
        X_train_cat = pd.concat(X_train_cat_list, axis=1)
        X_val_cat = pd.concat(X_val_cat_list, axis=1)
        X_test_cat = pd.concat(X_test_cat_list, axis=1)
        
        X_train = pd.concat([X_train_num, X_train_cat], axis=1)
        X_val = pd.concat([X_val_num, X_val_cat], axis=1)
        X_test = pd.concat([X_test_num, X_test_cat], axis=1)
    else:
        X_train = X_train_num
        X_val = X_val_num
        X_test = X_test_num
    
    # 3. Feature engineering
    X_train = engineer_features(X_train)
    X_val = engineer_features(X_val)
    X_test = engineer_features(X_test)
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"   ✅ Final features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, scaler, label_encoders


def engineer_features(X):
    """Create domain-specific features."""
    X = X.copy()
    
    # Only create features if base columns exist
    if 'mw_freebase' in X.columns and 'alogp' in X.columns:
        X['mw_logp_interaction'] = X['mw_freebase'] * X['alogp']
    
    if 'psa' in X.columns and 'mw_freebase' in X.columns:
        X['binding_efficiency'] = X['psa'] / (X['mw_freebase'] + 1)
    
    if 'rtb' in X.columns and 'aromatic_rings' in X.columns:
        X['complexity_score'] = X['rtb'] * X['aromatic_rings']
    
    return X


def train_models(X_train, y_train, X_val, y_val):
    """Train Logistic Regression and XGBoost models."""
    print("\n🤖 Training models...")
    
    # Calculate class weight (minority class = inactive)
    scale_pos_weight = (y_train == 1).sum() / (y_train == 0).sum()
    print(f"   Class weight (active/inactive): {scale_pos_weight:.2f}")
    print(f"   (Weighting inactive class {1/scale_pos_weight:.2f}x higher)")
    
    # Model 1: Logistic Regression
    print("\n   🔹 Training Logistic Regression...")
    model_lr = LogisticRegression(
        class_weight={0: scale_pos_weight, 1: 1.0},
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model_lr.fit(X_train, y_train)
    print("      ✅ Logistic Regression trained")
    
    # Model 2: XGBoost with hyperparameter tuning
    print("\n   🔹 Training XGBoost with RandomizedSearchCV...")
    
    param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric='auc', random_state=42, use_label_encoder=False),
        param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    xgb_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    model_xgb = xgb_search.best_estimator_
    print(f"      ✅ XGBoost trained")
    print(f"      Best params: {xgb_search.best_params_}")
    
    return model_lr, model_xgb


def evaluate_model(model, X, y, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Model': model_name,
        'ROC-AUC': roc_auc_score(y, y_pred_proba),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred),
        'F1-Score': f1_score(y, y_pred)
    }
    
    return metrics, y_pred, y_pred_proba


def save_models(model_lr, model_xgb, scaler, label_encoders, output_dir='models/saved'):
    """Save trained models and preprocessors."""
    print(f"\n💾 Saving models to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model_lr, f'{output_dir}/logistic_regression.pkl')
    joblib.dump(model_xgb, f'{output_dir}/xgboost.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    
    print("   ✅ Models saved")


def main():
    """Run complete training pipeline."""
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Preprocess
    X_train, X_val, X_test, scaler, label_encoders = preprocess_features(
        X_train, X_val, X_test
    )
    
    # Train models
    model_lr, model_xgb = train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n📊 Evaluating models on test set...")
    lr_metrics, lr_pred, lr_proba = evaluate_model(model_lr, X_test, y_test, 'Logistic Regression')
    xgb_metrics, xgb_pred, xgb_proba = evaluate_model(model_xgb, X_test, y_test, 'XGBoost')
    
    # Results
    results_df = pd.DataFrame([lr_metrics, xgb_metrics])
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # Save models
    save_models(model_lr, model_xgb, scaler, label_encoders)
    
    print("\n✅ Step 3 Complete!")
    print(f"✅ Best Model: {results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']}")
    print(f"✅ Best ROC-AUC: {results_df['ROC-AUC'].max():.3f}")
    
    print("\nNext Steps:")
    print("1. Run notebooks/02_model_evaluation.ipynb for detailed analysis")
    print("2. Generate SHAP plots for explainability")
    print("3. Move to Step 4: Streamlit App")


if __name__ == "__main__":
    main()
