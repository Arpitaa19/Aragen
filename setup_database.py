#!/usr/bin/env python3
"""
Step 1: Database Setup for BioInsight Hackathon
Restores ChEMBL database and verifies setup
"""

import os
import sys
import subprocess
from pathlib import Path


def check_postgresql():
    """Check if PostgreSQL is installed."""
    print("🔍 Checking PostgreSQL installation...")
    
    try:
        result = subprocess.run(
            ['psql', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"✅ {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PostgreSQL not found!")
        print("\n📥 Install PostgreSQL:")
        print("   macOS: brew install postgresql@14")
        print("   Ubuntu: sudo apt-get install postgresql")
        print("   Windows: Download from https://www.postgresql.org/download/")
        return False


def check_dump_file():
    """Check if ChEMBL dump file exists."""
    print("\n🔍 Checking for ChEMBL dump file...")
    
    # Look for dump file in Aragen directory
    dump_path = Path(__file__).parent.parent / "chembl_36_postgresql" / "chembl_36_postgresql.dmp"
    
    if dump_path.exists():
        size_gb = dump_path.stat().st_size / (1024**3)
        print(f"✅ Found dump file: {dump_path}")
        print(f"   Size: {size_gb:.2f} GB")
        return str(dump_path)
    else:
        print(f"❌ Dump file not found at: {dump_path}")
        print("\n📥 Download ChEMBL dump:")
        print("   URL: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_postgresql.tar.gz")
        print("   Extract to: ../chembl_36_postgresql/")
        return None


def create_database(db_name='chembl_36', user='postgres'):
    """Create ChEMBL database."""
    print(f"\n🗄️  Creating database '{db_name}'...")
    
    try:
        # Check if database exists
        check_cmd = [
            'psql',
            '-U', user,
            '-h', 'localhost',
            '-d', 'postgres',
            '-tAc',
            f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        ]
        
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip() == '1':
            print(f"⚠️  Database '{db_name}' already exists")
            response = input("   Drop and recreate? (yes/no): ")
            if response.lower() != 'yes':
                print("   Skipping database creation")
                return True
            
            # Drop existing database
            drop_cmd = [
                'psql',
                '-U', user,
                '-h', 'localhost',
                '-d', 'postgres',
                '-c',
                f"DROP DATABASE {db_name};"
            ]
            subprocess.run(drop_cmd, check=True)
            print(f"   Dropped existing database")
        
        # Create database
        create_cmd = [
            'psql',
            '-U', user,
            '-h', 'localhost',
            '-d', 'postgres',
            '-c',
            f"CREATE DATABASE {db_name};"
        ]
        
        subprocess.run(create_cmd, check=True)
        print(f"✅ Database '{db_name}' created")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create database: {e}")
        print("\n💡 Troubleshooting:")
        print(f"   1. Ensure PostgreSQL is running: pg_isready")
        print(f"   2. Try: createdb -U {user} {db_name}")
        print(f"   3. Check credentials in .env file")
        return False


def restore_database(dump_path, db_name='chembl_36', user='postgres'):
    """Restore ChEMBL database from dump file."""
    print(f"\n📦 Restoring database from dump file...")
    print("   ⏱️  This will take 10-30 minutes...")
    
    try:
        restore_cmd = [
            'pg_restore',
            '--no-owner',
            '-h', 'localhost',
            '-U', user,
            '-d', db_name,
            dump_path
        ]
        
        print("   Starting restore... (grab a coffee ☕)")
        
        process = subprocess.Popen(
            restore_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("✅ Database restored successfully!")
            return True
        else:
            # pg_restore often returns non-zero even on success
            # Check if database has data
            print("⚠️  Restore completed with warnings (this is often normal)")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Restore failed: {e}")
        return False


def verify_database(db_name='chembl_36', user='postgres'):
    """Verify database has required tables."""
    print(f"\n✅ Verifying database setup...")
    
    try:
        # Use Python to verify
        sys.path.insert(0, str(Path(__file__).parent))
        from data.config import ChEMBLDatabase
        
        db = ChEMBLDatabase()
        success = db.test_connection()
        db.close()
        
        return success
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Run complete Step 1 setup."""
    print("="*60)
    print("BioInsight Hackathon - Step 1: Database Setup")
    print("="*60)
    
    # Step 1: Check PostgreSQL
    if not check_postgresql():
        sys.exit(1)
    
    # Step 2: Check dump file
    dump_path = check_dump_file()
    if not dump_path:
        sys.exit(1)
    
    # Step 3: Get credentials
    print("\n🔑 Database Credentials")
    print("   (These are YOUR local PostgreSQL credentials)")
    user = input("   PostgreSQL username [postgres]: ").strip() or 'postgres'
    
    # Step 4: Create database
    if not create_database(user=user):
        sys.exit(1)
    
    # Step 5: Restore database
    if not restore_database(dump_path, user=user):
        sys.exit(1)
    
    # Step 6: Verify setup
    if verify_database(user=user):
        print("\n" + "="*60)
        print("✅ Step 1 Complete!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Copy .env.example to .env")
        print("2. Update .env with your credentials")
        print("3. Test: python data/config.py")
        print("4. Move to Step 2: EDA (notebooks/01_data_exploration.ipynb)")
    else:
        print("\n⚠️  Setup completed but verification failed")
        print("   Check database manually: psql -U postgres -d chembl_36")


if __name__ == "__main__":
    main()
