#!/usr/bin/env python3
"""
Step 2: Extract Sample Bioactivity Data
Extracts a manageable sample from ChEMBL for hackathon modeling.

Output: data/sample_bioactivity.csv (~100K rows)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from config import ChEMBLDatabase

# Configuration
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', 100000))
OUTPUT_FILE = Path(__file__).parent / 'sample_bioactivity.csv'


def extract_bioactivity_sample(sample_size=SAMPLE_SIZE):
    """
    Extract sample bioactivity data with compound properties.
    
    Strategy:
    - Join 5 key tables
    - Filter for high-quality data (IC50/EC50, human targets, valid units)
    - Sample stratified by activity (balanced active/inactive)
    - Include all relevant features for modeling
    
    Returns:
        DataFrame with sample bioactivity data
    """
    print("="*60)
    print("Step 2: Extracting Sample Bioactivity Data")
    print("="*60)
    
    db = ChEMBLDatabase()
    db.connect()
    
    print(f"\n📊 Extracting {sample_size:,} sample records...")
    print("   Joining 5 tables: activities, molecules, targets, assays, properties")
    print("   Filtering for high-quality data...")
    
    # SQL query to extract sample data
    query = f"""
    WITH filtered_activities AS (
        SELECT 
            act.activity_id,
            act.molregno,
            act.assay_id,
            act.standard_type,
            act.standard_relation,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            a.tid,
            a.assay_type,
            a.confidence_score
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        WHERE 
            -- Filter for concentration-response metrics
            act.standard_type IN ('IC50', 'EC50', 'Ki', 'Kd')
            -- Filter for nM units (standard)
            AND act.standard_units = 'nM'
            -- Filter for valid measurements
            AND act.standard_value IS NOT NULL
            AND act.standard_relation IN ('=', '<', '>')
            -- Filter for high-confidence assays
            AND a.confidence_score >= 6
        LIMIT {sample_size * 2}  -- Get 2x for filtering
    )
    SELECT 
        -- IDs
        fa.activity_id,
        md.chembl_id as compound_id,
        td.chembl_id as target_id,
        fa.assay_id,
        
        -- Activity measurements
        fa.standard_type,
        fa.standard_relation,
        fa.standard_value,
        fa.standard_units,
        fa.pchembl_value,
        
        -- Target information
        td.pref_name as target_name,
        td.target_type,
        td.organism,
        
        -- Assay information
        fa.assay_type,
        fa.confidence_score,
        
        -- Compound properties (features for modeling)
        cp.mw_freebase,
        cp.alogp,
        cp.hba,
        cp.hbd,
        cp.psa,
        cp.rtb,
        cp.aromatic_rings,
        cp.heavy_atoms,
        cp.qed_weighted,
        cp.num_ro5_violations as lipinski_violations,
        cp.full_mwt,
        cp.np_likeness_score,
        cp.ro3_pass
        
    FROM filtered_activities fa
    
    -- Join molecule dictionary
    JOIN molecule_dictionary md 
        ON fa.molregno = md.molregno
    
    -- Join target dictionary
    JOIN target_dictionary td 
        ON fa.tid = td.tid
    
    -- Join compound properties (LEFT JOIN to keep records without properties)
    LEFT JOIN compound_properties cp 
        ON fa.molregno = cp.molregno
    
    WHERE 
        -- Filter for human targets (most relevant for drug discovery)
        td.organism = 'Homo sapiens'
        -- Ensure we have basic compound properties
        AND cp.mw_freebase IS NOT NULL
        AND cp.alogp IS NOT NULL
    
    -- Random sampling
    ORDER BY RANDOM()
    LIMIT {sample_size};
    """
    
    print("   Executing query (this may take 2-5 minutes)...")
    df = db.query(query)
    
    print(f"\n✅ Extracted {len(df):,} records")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    db.close()
    
    return df


def create_binary_target(df, threshold=10000):
    """
    Create binary activity label (active/inactive).
    
    Active: standard_value < threshold (10,000 nM = 10 μM)
    Inactive: standard_value >= threshold
    
    Args:
        df: DataFrame with standard_value column
        threshold: Activity threshold in nM (default: 10,000)
        
    Returns:
        DataFrame with 'is_active' column
    """
    print(f"\n🎯 Creating binary target (threshold: {threshold:,} nM)...")
    
    df['is_active'] = (df['standard_value'] < threshold).astype(int)
    
    active_count = df['is_active'].sum()
    inactive_count = len(df) - active_count
    active_pct = 100 * active_count / len(df)
    
    print(f"   Active: {active_count:,} ({active_pct:.1f}%)")
    print(f"   Inactive: {inactive_count:,} ({100-active_pct:.1f}%)")
    
    if active_pct < 20 or active_pct > 80:
        print(f"   ⚠️  Dataset is imbalanced - will need SMOTE/class weights")
    else:
        print(f"   ✅ Dataset is reasonably balanced")
    
    return df


def analyze_data_quality(df):
    """Analyze data quality and missing values."""
    print("\n📋 Data Quality Analysis:")
    print("-" * 60)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    print("\nMissing Values:")
    for col in missing[missing > 0].index:
        print(f"   {col:30s}: {missing[col]:6,} ({missing_pct[col]:5.1f}%)")
    
    if missing.sum() == 0:
        print("   ✅ No missing values!")
    
    # Data types
    print("\nData Types:")
    print(f"   Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Basic statistics
    print("\nKey Statistics:")
    print(f"   Molecular Weight: {df['mw_freebase'].mean():.1f} ± {df['mw_freebase'].std():.1f}")
    print(f"   LogP: {df['alogp'].mean():.2f} ± {df['alogp'].std():.2f}")
    print(f"   Activity (nM): {df['standard_value'].median():.1f} (median)")
    
    # Unique values
    print("\nUnique Values:")
    print(f"   Compounds: {df['compound_id'].nunique():,}")
    print(f"   Targets: {df['target_id'].nunique():,}")
    print(f"   Assays: {df['assay_id'].nunique():,}")


def save_sample(df, output_file=OUTPUT_FILE):
    """Save sample data to CSV."""
    print(f"\n💾 Saving to {output_file}...")
    
    df.to_csv(output_file, index=False)
    
    file_size = output_file.stat().st_size / 1024**2
    print(f"   ✅ Saved {len(df):,} rows, {len(df.columns)} columns")
    print(f"   File size: {file_size:.1f} MB")


def main():
    """Run complete extraction pipeline."""
    
    # Extract sample
    df = extract_bioactivity_sample()
    
    # Create binary target
    df = create_binary_target(df)
    
    # Analyze quality
    analyze_data_quality(df)
    
    # Save
    save_sample(df)
    
    print("\n" + "="*60)
    print("✅ Step 2 Complete - Sample Data Extracted!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Open Jupyter: jupyter notebook notebooks/01_eda.ipynb")
    print("2. Run EDA cells to visualize and analyze data")
    print("3. Document key findings for modeling")


if __name__ == "__main__":
    main()
