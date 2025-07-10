#!/usr/bin/env python3
"""
Test script to verify the fixed sentence alignment algorithm.
This tests the specific problem cases mentioned in the request.
"""

import sys
from pathlib import Path

# Add the parent directory to sys.path to import the SentenceAligner
sys.path.insert(0, str(Path(__file__).parent))

from compare_transcripts_visual import SentenceAligner, DiffType

def test_sentence_alignment():
    """Test the sentence alignment with the example problem case."""
    
    # Example problem case from the request
    pdf_sentences = [
        "Revenue increased significantly in Q3.",
        "We are pleased with the results.",
        "Operating margins improved by 2%.",
        "Looking forward, we expect continued growth."
    ]
    
    html_sentences = [
        "Revenue increased significantly in Q3, and we are pleased with the results.",
        "Operating margins improved by 2%.",
        "Looking forward, we expect continued growth.",
        "Additional disclosure information."
    ]
    
    print("🧪 Testing Sentence Alignment Algorithm")
    print("=" * 50)
    
    print("\n📄 PDF Sentences:")
    for i, sent in enumerate(pdf_sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\n🌐 HTML Sentences:")
    for i, sent in enumerate(html_sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\n🔗 Running Alignment...")
    aligner = SentenceAligner(similarity_threshold=0.6)
    alignments = aligner.align_sentences(pdf_sentences, html_sentences)
    
    print(f"\n📊 Results: {len(alignments)} alignments created")
    print("=" * 50)
    
    # Track which sentences have been used to verify no duplication
    used_pdf_sentences = set()
    used_html_sentences = set()
    
    for i, alignment in enumerate(alignments, 1):
        print(f"\nAlignment {i}: {alignment.diff_type.value.upper()}")
        
        if alignment.pdf_sentences:
            print(f"  📄 PDF ({len(alignment.pdf_sentences)}): {'; '.join(alignment.pdf_sentences)}")
            # Check for duplicates
            for sent in alignment.pdf_sentences:
                if sent in used_pdf_sentences:
                    print(f"  ❌ ERROR: PDF sentence used multiple times: {sent}")
                used_pdf_sentences.add(sent)
        else:
            print(f"  📄 PDF: (none)")
            
        if alignment.html_sentences:
            print(f"  🌐 HTML ({len(alignment.html_sentences)}): {'; '.join(alignment.html_sentences)}")
            # Check for duplicates
            for sent in alignment.html_sentences:
                if sent in used_html_sentences:
                    print(f"  ❌ ERROR: HTML sentence used multiple times: {sent}")
                used_html_sentences.add(sent)
        else:
            print(f"  🌐 HTML: (none)")
            
        if alignment.similarity > 0:
            print(f"  📈 Similarity: {alignment.similarity:.1%}")
    
    # Verify completeness
    print(f"\n🔍 Verification:")
    print(f"  PDF sentences used: {len(used_pdf_sentences)}/{len(pdf_sentences)}")
    print(f"  HTML sentences used: {len(used_html_sentences)}/{len(html_sentences)}")
    
    missing_pdf = set(pdf_sentences) - used_pdf_sentences
    missing_html = set(html_sentences) - used_html_sentences
    
    if missing_pdf:
        print(f"  ❌ Missing PDF sentences: {missing_pdf}")
    if missing_html:
        print(f"  ❌ Missing HTML sentences: {missing_html}")
    
    # Check expected results
    print(f"\n✅ Expected Results Check:")
    
    # Expected: Sentences 1-2 (PDF) should match with sentence 1 (HTML) as a 2:1 match
    found_2_to_1 = False
    for alignment in alignments:
        if (len(alignment.pdf_sentences) == 2 and 
            len(alignment.html_sentences) == 1 and
            alignment.pdf_sentences[0] == pdf_sentences[0] and
            alignment.pdf_sentences[1] == pdf_sentences[1] and
            alignment.html_sentences[0] == html_sentences[0]):
            found_2_to_1 = True
            print(f"  ✓ Found expected 2:1 match (PDF sentences 1-2 → HTML sentence 1)")
            break
    
    if not found_2_to_1:
        print(f"  ❌ Did not find expected 2:1 match")
    
    # Expected: Sentence 3 should match 1:1
    found_1_to_1_s3 = False
    for alignment in alignments:
        if (len(alignment.pdf_sentences) == 1 and 
            len(alignment.html_sentences) == 1 and
            alignment.pdf_sentences[0] == pdf_sentences[2] and
            alignment.html_sentences[0] == html_sentences[1]):
            found_1_to_1_s3 = True
            print(f"  ✓ Found expected 1:1 match (PDF sentence 3 → HTML sentence 2)")
            break
    
    if not found_1_to_1_s3:
        print(f"  ❌ Did not find expected 1:1 match for sentence 3")
    
    # Expected: HTML sentence 4 should appear as an INSERT at the end
    found_insert = False
    for alignment in alignments:
        if (alignment.diff_type == DiffType.INSERT and
            len(alignment.html_sentences) == 1 and
            alignment.html_sentences[0] == html_sentences[3]):
            found_insert = True
            print(f"  ✓ Found expected INSERT (HTML sentence 4)")
            break
    
    if not found_insert:
        print(f"  ❌ Did not find expected INSERT for HTML sentence 4")
    
    print(f"\n🎉 Test completed!")

def test_order_preservation():
    """Test that sentences maintain their original order."""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Order Preservation")
    print("=" * 50)
    
    # Create a case where naive alignment might reorder sentences
    pdf_sentences = [
        "First sentence about revenue.",
        "Second sentence about costs.",
        "Third sentence about outlook.",
        "Fourth sentence about guidance."
    ]
    
    html_sentences = [
        "Different first sentence.",
        "First sentence about revenue.",  # Matches PDF sentence 1
        "Third sentence about outlook.",  # Matches PDF sentence 3
        "Fourth sentence about guidance.", # Matches PDF sentence 4
        "Extra sentence at the end."
    ]
    
    print("\n📄 PDF Sentences:")
    for i, sent in enumerate(pdf_sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\n🌐 HTML Sentences:")
    for i, sent in enumerate(html_sentences, 1):
        print(f"  {i}. {sent}")
    
    aligner = SentenceAligner(similarity_threshold=0.8)
    alignments = aligner.align_sentences(pdf_sentences, html_sentences)
    
    print(f"\n📊 Results: {len(alignments)} alignments created")
    
    # Verify order is maintained
    pdf_order = []
    html_order = []
    
    for alignment in alignments:
        for sent in alignment.pdf_sentences:
            if sent in pdf_sentences:
                pdf_order.append(pdf_sentences.index(sent))
        for sent in alignment.html_sentences:
            if sent in html_sentences:
                html_order.append(html_sentences.index(sent))
    
    # Check if order is preserved
    pdf_order_preserved = pdf_order == sorted(pdf_order)
    html_order_preserved = html_order == sorted(html_order)
    
    print(f"\n✅ Order Preservation Check:")
    print(f"  PDF order preserved: {pdf_order_preserved} {pdf_order}")
    print(f"  HTML order preserved: {html_order_preserved} {html_order}")
    
    if pdf_order_preserved and html_order_preserved:
        print(f"  ✓ Order preservation test PASSED")
    else:
        print(f"  ❌ Order preservation test FAILED")

if __name__ == "__main__":
    try:
        test_sentence_alignment()
        test_order_preservation()
        print(f"\n🎯 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)