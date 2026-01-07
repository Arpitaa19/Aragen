"""
Database connection utilities
Connects to ChEMBL PostgreSQL database
"""
import psycopg2
import pandas as pd
import os
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    Usage:
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn)
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "chembl_36"),
            user=os.getenv("DB_USER", "ved"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        yield conn
    finally:
        if conn:
            conn.close()


def search_compounds(
    target_name=None,
    mw_min=None,
    mw_max=None,
    activity_filter=None,
    limit=100
):
    """
    Search compounds in ChEMBL database
    
    Args:
        target_name: Target protein name (partial match)
        mw_min: Minimum molecular weight
        mw_max: Maximum molecular weight
        activity_filter: 'active', 'inactive', or None (all)
        limit: Maximum results to return
        
    Returns:
        DataFrame with search results
    """
    query = """
    SELECT 
        md.chembl_id as compound_id,
        td.pref_name as target_name,
        cp.mw_freebase,
        cp.alogp,
        act.standard_value,
        act.standard_type,
        CASE 
            WHEN act.standard_value < 10000 THEN 'Active'
            ELSE 'Inactive'
        END as activity_class
    FROM activities act
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_properties cp ON act.molregno = cp.molregno
    JOIN target_dictionary td ON act.tid = td.tid
    WHERE 
        act.standard_type IN ('IC50', 'EC50', 'Ki', 'Kd')
        AND act.standard_units = 'nM'
        AND act.standard_value IS NOT NULL
    """
    
    params = []
    
    # Add filters
    if target_name:
        query += " AND td.pref_name ILIKE %s"
        params.append(f"%{target_name}%")
    
    if mw_min is not None:
        query += " AND cp.mw_freebase >= %s"
        params.append(mw_min)
    
    if mw_max is not None:
        query += " AND cp.mw_freebase <= %s"
        params.append(mw_max)
    
    if activity_filter == 'active':
        query += " AND act.standard_value < 10000"
    elif activity_filter == 'inactive':
        query += " AND act.standard_value >= 10000"
    
    query += f" LIMIT {limit}"
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    
    return df


def get_dataset_stats():
    """Get summary statistics about the dataset"""
    query = """
    SELECT 
        COUNT(DISTINCT act.molregno) as total_compounds,
        COUNT(DISTINCT act.tid) as total_targets,
        COUNT(DISTINCT act.assay_id) as total_assays,
        COUNT(*) as total_activities,
        SUM(CASE WHEN act.standard_value < 10000 THEN 1 ELSE 0 END) as active_count,
        SUM(CASE WHEN act.standard_value >= 10000 THEN 1 ELSE 0 END) as inactive_count
    FROM activities act
    WHERE 
        act.standard_type IN ('IC50', 'EC50', 'Ki', 'Kd')
        AND act.standard_units = 'nM'
        AND act.standard_value IS NOT NULL
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    return df.iloc[0].to_dict()
