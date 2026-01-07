"""
Database Configuration for BioInsight Hackathon
Handles connection to local ChEMBL PostgreSQL database
"""

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

# Load environment variables
load_dotenv()


class ChEMBLDatabase:
    """Handles ChEMBL database connections and queries."""
    
    def __init__(self):
        """Initialize database connection parameters from environment."""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'chembl_36')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', '')
        
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print(f"✅ Connected to ChEMBL database: {self.database}")
            return self.conn
        except psycopg2.Error as e:
            print(f"❌ Database connection failed: {e}")
            print("\nTroubleshooting:")
            print("1. Check if PostgreSQL is running: pg_isready")
            print("2. Verify credentials in .env file")
            print("3. Ensure chembl_36 database exists: psql -l")
            raise
    
    def test_connection(self):
        """Test database connection and verify tables exist."""
        if not self.conn:
            self.connect()
        
        print("\n🔍 Testing database connection...\n")
        
        # Test queries for each required table
        tables = {
            'activities': 'SELECT COUNT(*) as count FROM activities',
            'molecule_dictionary': 'SELECT COUNT(*) as count FROM molecule_dictionary',
            'target_dictionary': 'SELECT COUNT(*) as count FROM target_dictionary',
            'assays': 'SELECT COUNT(*) as count FROM assays',
            'compound_properties': 'SELECT COUNT(*) as count FROM compound_properties'
        }
        
        results = {}
        for table_name, query in tables.items():
            try:
                df = pd.read_sql(query, self.conn)
                count = df['count'].iloc[0]
                results[table_name] = count
                print(f"✅ {table_name:25s}: {count:,} rows")
            except Exception as e:
                results[table_name] = 0
                print(f"❌ {table_name:25s}: Error - {e}")
        
        print("\n" + "="*60)
        if all(count > 0 for count in results.values()):
            print("✅ All required tables are accessible!")
            print("="*60)
            return True
        else:
            print("⚠️  Some tables are missing or empty")
            print("="*60)
            return False
    
    def query(self, sql, params=None):
        """Execute SQL query and return DataFrame."""
        if not self.conn:
            self.connect()
        
        return pd.read_sql(sql, self.conn, params=params)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def get_sample_data(limit=10):
    """
    Get sample bioactivity data for quick testing.
    
    Args:
        limit: Number of rows to fetch
        
    Returns:
        DataFrame with sample bioactivity data
    """
    db = ChEMBLDatabase()
    db.connect()
    
    query = f"""
    SELECT 
        act.activity_id,
        act.molregno,
        act.standard_type,
        act.standard_value,
        act.standard_units,
        md.chembl_id as compound_id,
        td.chembl_id as target_id,
        td.pref_name as target_name,
        cp.mw_freebase,
        cp.alogp,
        cp.hba,
        cp.hbd,
        cp.psa
    FROM activities act
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    LEFT JOIN compound_properties cp ON act.molregno = cp.molregno
    WHERE act.standard_type IN ('IC50', 'EC50', 'Ki', 'Kd')
        AND act.standard_units = 'nM'
        AND act.standard_value IS NOT NULL
    LIMIT {limit}
    """
    
    df = db.query(query)
    db.close()
    
    return df


if __name__ == "__main__":
    # Test database connection
    print("="*60)
    print("BioInsight Hackathon - Database Connection Test")
    print("="*60)
    
    db = ChEMBLDatabase()
    
    # Test connection and verify tables
    if db.test_connection():
        print("\n📊 Fetching sample data...\n")
        sample = get_sample_data(limit=5)
        print(sample)
        print(f"\n✅ Successfully fetched {len(sample)} sample records")
    
    db.close()
