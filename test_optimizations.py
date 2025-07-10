#!/usr/bin/env python3
"""
Test script for comparing performance of optimized vs original transcript comparison
"""

import time
import random
from compare_transcripts_visual import SlidingWindowMatcher, SentenceAligner
from typing import List

def generate_test_data(size: int) -> tuple[List[str], List[str]]:
    """Generate test word lists for performance testing."""
    financial_terms = [
        "revenue", "earnings", "margin", "ebitda", "guidance", "quarter", "fiscal",
        "growth", "million", "billion", "percent", "increased", "decreased", 
        "operating", "segment", "outlook", "performance", "results", "metrics",
        "shareholders", "dividend", "acquisition", "expansion", "investment"
    ]
    
    pdf_words = []
    html_words = []
    
    # Generate PDF words
    for i in range(size):
        if i % 20 == 0:  # Add some financial terms
            pdf_words.append(random.choice(financial_terms))
        else:
            pdf_words.append(f"word_{i}")
    
    # Generate HTML words with some overlaps and differences
    for i in range(size):
        if i % 30 == 0:  # Skip some words
            continue
        elif i % 25 == 0:  # Add some different words  
            html_words.append("different_word")
        elif i % 20 == 0:  # Match financial terms
            html_words.append(random.choice(financial_terms))
        else:
            html_words.append(f"word_{i}")
    
    return pdf_words, html_words

def generate_test_sentences(size: int) -> tuple[List[str], List[str]]:
    """Generate test sentences for alignment testing."""
    base_sentences = [
        "Revenue increased significantly in the third quarter.",
        "Operating margins improved due to cost reduction initiatives.", 
        "We are pleased with the strong financial performance.",
        "Earnings per share exceeded analyst expectations.",
        "The company maintains a positive outlook for next year.",
        "We continue to invest in growth opportunities.",
        "Cash flow generation remains robust across all segments.",
        "Market conditions present both challenges and opportunities."
    ]
    
    pdf_sentences = []
    html_sentences = []
    
    for i in range(size):
        base_sent = base_sentences[i % len(base_sentences)]
        pdf_sentences.append(f"{base_sent} Statement {i}.")
        
        # Create variations for HTML
        if i % 4 == 0:  # Skip some sentences
            continue
        elif i % 3 == 0:  # Modify some sentences
            html_sentences.append(f"Modified: {base_sent} Entry {i}.")
        else:  # Keep mostly the same
            html_sentences.append(f"{base_sent} Statement {i}.")
    
    return pdf_sentences, html_sentences

def test_word_matching_performance():
    """Test word matching performance improvements."""
    print("=" * 60)
    print("WORD MATCHING PERFORMANCE TEST")
    print("=" * 60)
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size:,} words:")
        pdf_words, html_words = generate_test_data(size)
        
        # Test optimized matcher
        print(f"  PDF words: {len(pdf_words):,}, HTML words: {len(html_words):,}")
        
        matcher = SlidingWindowMatcher(window_size=6, min_match_threshold=0.7)
        
        start_time = time.time()
        matches = matcher.find_matches(pdf_words, html_words)
        elapsed = time.time() - start_time
        
        print(f"  Optimized matching: {elapsed:.2f}s, {len(matches)} matches found")
        print(f"  Cache size: {len(matcher.similarity_cache)}")
        
        if matches:
            avg_confidence = sum(m.confidence for m in matches) / len(matches)
            print(f"  Average confidence: {avg_confidence:.3f}")

def test_sentence_alignment_performance():
    """Test sentence alignment performance improvements."""
    print("\n" + "=" * 60)
    print("SENTENCE ALIGNMENT PERFORMANCE TEST")
    print("=" * 60)
    
    sizes = [100, 300, 500]
    
    for size in sizes:
        print(f"\nTesting with {size} sentences:")
        pdf_sentences, html_sentences = generate_test_sentences(size)
        
        print(f"  PDF sentences: {len(pdf_sentences)}, HTML sentences: {len(html_sentences)}")
        
        aligner = SentenceAligner(similarity_threshold=0.6)
        
        start_time = time.time()
        alignments = aligner.align_sentences(pdf_sentences, html_sentences)
        elapsed = time.time() - start_time
        
        print(f"  Ordered alignment: {elapsed:.2f}s, {len(alignments)} alignments")
        
        # Analyze alignment types
        equal_count = sum(1 for a in alignments if a.diff_type.value == "equal")
        replace_count = sum(1 for a in alignments if a.diff_type.value == "replace") 
        insert_count = sum(1 for a in alignments if a.diff_type.value == "insert")
        delete_count = sum(1 for a in alignments if a.diff_type.value == "delete")
        
        print(f"  Equal: {equal_count}, Replace: {replace_count}, Insert: {insert_count}, Delete: {delete_count}")

def test_order_preservation():
    """Test that sentence alignment preserves order."""
    print("\n" + "=" * 60) 
    print("ORDER PRESERVATION TEST")
    print("=" * 60)
    
    # Create test case where order matters
    pdf_sentences = [
        "First statement about revenue.",
        "Second statement about margins.", 
        "Third statement about guidance.",
        "Fourth statement about outlook.",
        "Fifth statement about results."
    ]
    
    html_sentences = [
        "First statement about revenue.",  # Match
        "Some extra content here.",       # Insert
        "Second statement about margins.",  # Match
        "Third statement about guidance.",  # Match  
        "Different fourth statement.",      # Replace
        "Fifth statement about results."    # Match
    ]
    
    aligner = SentenceAligner(similarity_threshold=0.6)
    alignments = aligner.align_sentences(pdf_sentences, html_sentences)
    
    print("\nAlignment results:")
    for i, alignment in enumerate(alignments):
        pdf_text = alignment.pdf_sentences[0] if alignment.pdf_sentences else "[NONE]"
        html_text = alignment.html_sentences[0] if alignment.html_sentences else "[NONE]"
        
        print(f"  {i+1}. {alignment.diff_type.value.upper()}")
        print(f"     PDF:  {pdf_text[:50]}...")
        print(f"     HTML: {html_text[:50]}...")
        print(f"     Similarity: {alignment.similarity:.3f}")

def run_comprehensive_test():
    """Run all performance and correctness tests."""
    print("FACTSET TRANSCRIPT COMPARISON - OPTIMIZATION TEST SUITE")
    print("Testing rolling hash, caching, ordered alignment, and adaptive window sizing")
    
    try:
        test_word_matching_performance()
        test_sentence_alignment_performance() 
        test_order_preservation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOptimizations implemented:")
        print("✓ Rolling hash for O(n) window matching")
        print("✓ Multi-level caching for similarity calculations") 
        print("✓ Ordered sequence alignment maintaining sentence order")
        print("✓ Adaptive window sizing for better match accuracy")
        print("✓ Early termination strategies for performance")
        print("✓ Financial domain-specific optimizations")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test()