#!/usr/bin/env python3
"""Debug script to test Stage 8 embeddings generation."""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main module
from main_embeddings_generation import (
    get_nas_connection,
    load_stage_config,
    load_stage7_data,
    count_tokens,
    setup_tokenizer
)

def debug_stage7_data():
    """Debug the Stage 7 data structure to understand field names."""
    
    print("=" * 60)
    print("Stage 8 Debug Script")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Connect to NAS
    print("\n1. Connecting to NAS...")
    nas_conn = get_nas_connection()
    if not nas_conn:
        print("ERROR: Failed to connect to NAS")
        return
    print("✓ Connected to NAS")
    
    # Load config
    print("\n2. Loading configuration...")
    global config
    config = load_stage_config(nas_conn)
    print("✓ Configuration loaded")
    
    # Setup tokenizer
    print("\n3. Setting up tokenizer...")
    setup_tokenizer()
    print("✓ Tokenizer ready")
    
    # Load Stage 7 data
    print("\n4. Loading Stage 7 data...")
    stage7_data = load_stage7_data(nas_conn)
    print(f"✓ Loaded {len(stage7_data)} records")
    
    # Analyze first few records
    print("\n5. Analyzing data structure...")
    print("-" * 40)
    
    if stage7_data:
        # Check first record
        first_record = stage7_data[0]
        print(f"First record fields ({len(first_record)} total):")
        for field in sorted(first_record.keys()):
            value = first_record[field]
            if isinstance(value, str):
                if field in ['paragraph_content', 'content', 'chunk_text', 'block_summary']:
                    print(f"  - {field}: string ({len(value)} chars)")
                    if value:
                        print(f"    Preview: {value[:100]}...")
                else:
                    print(f"  - {field}: {value[:50] if len(value) > 50 else value}")
            elif isinstance(value, list):
                print(f"  - {field}: list ({len(value)} items)")
            elif isinstance(value, dict):
                print(f"  - {field}: dict ({len(value)} keys)")
            else:
                print(f"  - {field}: {type(value).__name__} = {value}")
        
        print("\n6. Content field analysis:")
        print("-" * 40)
        
        # Check which content fields exist
        content_fields = ['paragraph_content', 'content', 'text', 'paragraph_text']
        for field in content_fields:
            records_with_field = sum(1 for r in stage7_data[:100] if field in r)
            if records_with_field > 0:
                sample_records = [r for r in stage7_data[:100] if field in r and r.get(field)]
                if sample_records:
                    sample_text = sample_records[0][field]
                    print(f"  {field}: Found in {records_with_field}/100 records")
                    print(f"    Type: {type(sample_text).__name__}")
                    if isinstance(sample_text, str):
                        print(f"    Length: {len(sample_text)} chars")
                        token_count = count_tokens(sample_text)
                        print(f"    Tokens: {token_count}")
                        print(f"    Preview: {sample_text[:100]}...")
        
        print("\n7. Speaker block analysis:")
        print("-" * 40)
        
        # Group by speaker blocks
        from collections import defaultdict
        blocks = defaultdict(list)
        for record in stage7_data[:100]:  # Sample first 100 records
            speaker_block_id = record.get('speaker_block_id') or record.get('speaker_id')
            if speaker_block_id:
                blocks[speaker_block_id].append(record)
        
        print(f"Found {len(blocks)} speaker blocks in first 100 records")
        
        # Analyze first block
        if blocks:
            first_block_id = list(blocks.keys())[0]
            first_block_records = blocks[first_block_id]
            print(f"\nFirst block (ID: {first_block_id}):")
            print(f"  Records: {len(first_block_records)}")
            
            # Calculate total tokens
            total_tokens = 0
            for record in first_block_records:
                # Try to find content field
                text = (record.get('paragraph_content') or 
                       record.get('content') or 
                       record.get('text') or 
                       record.get('paragraph_text') or 
                       '')
                if text:
                    tokens = count_tokens(text)
                    total_tokens += tokens
                    print(f"  - Record {record.get('paragraph_id', 'unknown')}: {len(text)} chars, {tokens} tokens")
            
            print(f"  Total block tokens: {total_tokens}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)
    
    # Close NAS connection
    nas_conn.close()

if __name__ == "__main__":
    debug_stage7_data()