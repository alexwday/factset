#!/usr/bin/env python3
"""
Simple script to check pgvector version and halfvec support.
Run this BEFORE setting up the database to verify compatibility.
"""

import sys
import argparse

def check_pgvector_requirements():
    """Check if system can support halfvec for 3072 dimensions."""
    
    print("="*70)
    print("PostgreSQL pgvector Requirements Check for 3072-dimensional Embeddings")
    print("="*70)
    print()
    
    print("📋 REQUIREMENTS:")
    print("-" * 40)
    print("✓ PostgreSQL 11 or later")
    print("✓ pgvector extension version 0.7.0 or later")
    print("✓ Approximately 6KB storage per embedding (with halfvec)")
    print()
    
    print("🔍 HOW TO CHECK YOUR PGVECTOR VERSION:")
    print("-" * 40)
    print("Connect to your PostgreSQL database and run:")
    print()
    print("  -- Check if pgvector is installed:")
    print("  SELECT * FROM pg_available_extensions WHERE name = 'vector';")
    print()
    print("  -- If installed, check version:")
    print("  SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    print()
    print("  -- Or try to create the extension:")
    print("  CREATE EXTENSION IF NOT EXISTS vector;")
    print()
    
    print("📦 INSTALLATION OPTIONS:")
    print("-" * 40)
    
    print("\n1️⃣  OPTION A: Use halfvec with pgvector 0.7.0+ (RECOMMENDED)")
    print("   - Supports full 3072 dimensions from text-embedding-3-large")
    print("   - 50% storage savings compared to standard vectors")
    print("   - Up to 4000 dimensions supported")
    print()
    print("   Installation for common systems:")
    print("   • Ubuntu/Debian: apt-get install postgresql-16-pgvector")
    print("   • macOS: brew install pgvector")
    print("   • Docker: Use pgvector/pgvector:pg16 image")
    print("   • Amazon RDS: Available in engine version 15.4+")
    print()
    
    print("2️⃣  OPTION B: Use reduced dimensions (FALLBACK)")
    print("   - Use 1536 dimensions with standard vector type")
    print("   - Works with pgvector 0.5.0+")
    print("   - Some loss in embedding quality")
    print()
    print("   To use this option, modify main_embeddings_generation.py:")
    print("   • Change: self.embedding_dimensions = 1536")
    print("   • Change: embedding vector(1536) in table creation")
    print("   • Add dimensions parameter to OpenAI API call")
    print()
    
    print("📊 STORAGE COMPARISON:")
    print("-" * 40)
    print("For 100,000 embeddings:")
    print("• Standard vector (3072 dims): 1.14 GB ❌ (NOT SUPPORTED)")
    print("• Halfvec (3072 dims):         0.57 GB ✅ (RECOMMENDED)")
    print("• Standard vector (1536 dims): 0.57 GB ✅ (FALLBACK)")
    print()
    
    print("🚀 NEXT STEPS:")
    print("-" * 40)
    print("1. Check your pgvector version using the SQL above")
    print("2. If version < 0.7.0, either:")
    print("   a. Upgrade pgvector to 0.7.0+")
    print("   b. Use the reduced dimensions fallback")
    print("3. Run the embeddings generation:")
    print("   python main_embeddings_generation.py")
    print()
    
    print("💡 TIP: To test halfvec support after installation:")
    print("-" * 40)
    print("CREATE TEMP TABLE test_halfvec (vec halfvec(3072));")
    print("-- If this succeeds, you have halfvec support!")
    print()
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check pgvector requirements for 3072-dimensional embeddings"
    )
    parser.add_argument(
        '--dimensions', 
        type=int, 
        default=3072,
        help='Target embedding dimensions (default: 3072)'
    )
    
    args = parser.parse_args()
    
    if args.dimensions > 2000:
        print(f"⚠️  {args.dimensions} dimensions requires pgvector 0.7.0+ with halfvec support")
    else:
        print(f"✅ {args.dimensions} dimensions works with standard pgvector")
    
    check_pgvector_requirements()

if __name__ == "__main__":
    main()