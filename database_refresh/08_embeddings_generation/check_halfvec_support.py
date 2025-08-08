#!/usr/bin/env python3
"""
Check if PostgreSQL has halfvec support for 3072-dimensional embeddings.
"""

import psycopg2
import yaml
from pathlib import Path
import sys

def check_halfvec_support(config_path: str = "../config.yaml"):
    """Check if the database supports halfvec type."""
    
    # Load config
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            database=config['database']['name'],
            user=config['database']['user'],
            password=config['database']['password']
        )
        cursor = conn.cursor()
        
        print("Checking pgvector installation...")
        
        # Check if pgvector is installed
        cursor.execute("""
            SELECT extversion 
            FROM pg_extension 
            WHERE extname = 'vector'
        """)
        result = cursor.fetchone()
        
        if not result:
            print("❌ pgvector is not installed!")
            print("\nTo install pgvector:")
            print("1. Install the extension package for your PostgreSQL version")
            print("2. Run: CREATE EXTENSION vector;")
            return False
        
        version = result[0]
        print(f"✅ pgvector version {version} is installed")
        
        # Parse version
        major, minor, patch = map(int, version.split('.'))
        
        # Check if version supports halfvec (0.7.0+)
        if major > 0 or (major == 0 and minor >= 7):
            print(f"✅ Version {version} supports halfvec (3072 dimensions)")
            
            # Test halfvec type
            try:
                cursor.execute("CREATE TEMP TABLE test_halfvec (embedding halfvec(3072));")
                print("✅ Successfully created test table with halfvec(3072)")
                
                # Test inserting a value
                test_vector = [0.1] * 3072
                cursor.execute(
                    "INSERT INTO test_halfvec VALUES (%s::halfvec(3072))",
                    (test_vector,)
                )
                print("✅ Successfully inserted 3072-dimensional halfvec")
                
                return True
                
            except Exception as e:
                print(f"⚠️  halfvec test failed: {e}")
                return False
        else:
            print(f"❌ Version {version} does not support halfvec (requires 0.7.0+)")
            print("\nTo upgrade pgvector:")
            print("1. Back up your database")
            print("2. Update pgvector package to 0.7.0 or later")
            print("3. Run: ALTER EXTENSION vector UPDATE;")
            return False
            
    except Exception as e:
        print(f"❌ Error checking halfvec support: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def estimate_storage_savings():
    """Estimate storage savings with halfvec vs regular vector."""
    dimensions = 3072
    
    # Calculate sizes
    vector_size = dimensions * 4  # 4 bytes per float32
    halfvec_size = dimensions * 2  # 2 bytes per float16
    
    print("\n" + "="*60)
    print("STORAGE COMPARISON FOR 3072 DIMENSIONS")
    print("="*60)
    print(f"Standard vector: {vector_size:,} bytes per embedding")
    print(f"Halfvec:         {halfvec_size:,} bytes per embedding")
    print(f"Savings:         {vector_size - halfvec_size:,} bytes ({50:.0f}% reduction)")
    print()
    
    # Estimate for different dataset sizes
    for num_embeddings in [1000, 10000, 100000, 1000000]:
        vector_total = num_embeddings * vector_size / (1024**3)  # GB
        halfvec_total = num_embeddings * halfvec_size / (1024**3)  # GB
        print(f"{num_embeddings:,} embeddings:")
        print(f"  Vector:  {vector_total:.2f} GB")
        print(f"  Halfvec: {halfvec_total:.2f} GB")
        print(f"  Savings: {vector_total - halfvec_total:.2f} GB")
    print("="*60)

if __name__ == "__main__":
    print("PostgreSQL halfvec Support Checker")
    print("==================================\n")
    
    if check_halfvec_support():
        print("\n✅ Your database is ready for 3072-dimensional embeddings!")
        estimate_storage_savings()
    else:
        print("\n❌ Your database needs updates to support 3072 dimensions")
        print("\nAlternative options:")
        print("1. Upgrade pgvector to 0.7.0+ for halfvec support")
        print("2. Use reduced dimensions (1536) with standard vector type")
        print("3. Consider pgvecto.rs extension for higher dimensions")
        sys.exit(1)