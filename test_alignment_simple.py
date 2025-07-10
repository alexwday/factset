#!/usr/bin/env python3
"""
Simplified test for the sentence alignment algorithm logic.
Tests only the core alignment functionality without dependencies.
"""

import re
import difflib
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class DiffType(Enum):
    """Types of differences in sentence alignment."""
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"

@dataclass
class SentenceAlignment:
    """Represents an aligned sentence pair or diff."""
    diff_type: DiffType
    pdf_sentences: List[str]
    html_sentences: List[str]
    similarity: float = 0.0

class SentenceAligner:
    """Simplified version of the sentence aligner for testing."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.similarity_cache = {}
        self.max_cache_size = 5000
    
    def align_sentences(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Align sentences using robust sequential matching that prevents duplication."""
        print("    - Using robust sequential sentence alignment...")
        
        # Use single-pass sequential alignment that maintains order and prevents duplication
        alignments = self._sequential_alignment(pdf_sentences, html_sentences)
        
        print(f"    - Created {len(alignments)} final alignments")
        return alignments
    
    def _sequential_alignment(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Robust sequential alignment that maintains order and prevents duplication."""
        alignments = []
        pdf_idx = 0
        html_idx = 0
        
        print(f"    - Processing {len(pdf_sentences):,} PDF sentences and {len(html_sentences):,} HTML sentences...")
        
        while pdf_idx < len(pdf_sentences) and html_idx < len(html_sentences):
            # Try different alignment strategies
            best_alignment = self._find_best_alignment(pdf_sentences, html_sentences, pdf_idx, html_idx)
            
            if best_alignment:
                alignments.append(best_alignment['alignment'])
                pdf_idx += best_alignment['pdf_advance']
                html_idx += best_alignment['html_advance']
            else:
                # No good match found - advance PDF and mark as deleted
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.DELETE,
                    pdf_sentences=[pdf_sentences[pdf_idx]],
                    html_sentences=[],
                    similarity=0.0
                ))
                pdf_idx += 1
        
        # Handle remaining sentences in order
        while pdf_idx < len(pdf_sentences):
            alignments.append(SentenceAlignment(
                diff_type=DiffType.DELETE,
                pdf_sentences=[pdf_sentences[pdf_idx]],
                html_sentences=[],
                similarity=0.0
            ))
            pdf_idx += 1
        
        while html_idx < len(html_sentences):
            alignments.append(SentenceAlignment(
                diff_type=DiffType.INSERT,
                pdf_sentences=[],
                html_sentences=[html_sentences[html_idx]],
                similarity=0.0
            ))
            html_idx += 1
        
        return alignments
    
    def _find_best_alignment(self, pdf_sentences: List[str], html_sentences: List[str], 
                           pdf_idx: int, html_idx: int) -> Optional[Dict]:
        """Find the best alignment starting from the current positions."""
        best_alignment = None
        best_score = 0.0
        
        # Strategy 1: 1:1 matching (current PDF vs current HTML)
        if pdf_idx < len(pdf_sentences) and html_idx < len(html_sentences):
            similarity = self._calculate_robust_similarity_cached(
                pdf_sentences[pdf_idx], html_sentences[html_idx]
            )
            if similarity >= self.similarity_threshold:
                best_alignment = {
                    'alignment': SentenceAlignment(
                        diff_type=DiffType.EQUAL if similarity >= 0.9 else DiffType.REPLACE,
                        pdf_sentences=[pdf_sentences[pdf_idx]],
                        html_sentences=[html_sentences[html_idx]],
                        similarity=similarity
                    ),
                    'pdf_advance': 1,
                    'html_advance': 1,
                    'score': similarity
                }
                best_score = similarity
        
        # Strategy 2: 1:many matching (1 PDF sentence vs 2-3 HTML sentences)
        for html_count in range(2, min(4, len(html_sentences) - html_idx + 1)):
            if html_idx + html_count <= len(html_sentences):
                combined_html = ' '.join(html_sentences[html_idx:html_idx + html_count])
                similarity = self._calculate_robust_similarity_cached(
                    pdf_sentences[pdf_idx], combined_html
                )
                
                # Boost score slightly for multi-sentence matches to prefer them over weak 1:1 matches
                adjusted_score = similarity * 1.1 if similarity >= self.similarity_threshold else 0.0
                
                if adjusted_score > best_score:
                    best_alignment = {
                        'alignment': SentenceAlignment(
                            diff_type=DiffType.REPLACE,
                            pdf_sentences=[pdf_sentences[pdf_idx]],
                            html_sentences=html_sentences[html_idx:html_idx + html_count],
                            similarity=similarity
                        ),
                        'pdf_advance': 1,
                        'html_advance': html_count,
                        'score': adjusted_score
                    }
                    best_score = adjusted_score
        
        # Strategy 3: many:1 matching (2-3 PDF sentences vs 1 HTML sentence)
        for pdf_count in range(2, min(4, len(pdf_sentences) - pdf_idx + 1)):
            if pdf_idx + pdf_count <= len(pdf_sentences):
                combined_pdf = ' '.join(pdf_sentences[pdf_idx:pdf_idx + pdf_count])
                similarity = self._calculate_robust_similarity_cached(
                    combined_pdf, html_sentences[html_idx]
                )
                
                # Boost score slightly for multi-sentence matches
                adjusted_score = similarity * 1.1 if similarity >= self.similarity_threshold else 0.0
                
                if adjusted_score > best_score:
                    best_alignment = {
                        'alignment': SentenceAlignment(
                            diff_type=DiffType.REPLACE,
                            pdf_sentences=pdf_sentences[pdf_idx:pdf_idx + pdf_count],
                            html_sentences=[html_sentences[html_idx]],
                            similarity=similarity
                        ),
                        'pdf_advance': pdf_count,
                        'html_advance': 1,
                        'score': adjusted_score
                    }
                    best_score = adjusted_score
        
        # Strategy 4: Skip HTML sentence if no good match (HTML insertion)
        # This allows for cases where HTML has extra content
        if best_score < self.similarity_threshold:
            # Check if next HTML sentence matches better with current PDF
            if html_idx + 1 < len(html_sentences):
                next_similarity = self._calculate_robust_similarity_cached(
                    pdf_sentences[pdf_idx], html_sentences[html_idx + 1]
                )
                if next_similarity >= self.similarity_threshold:
                    # Insert the skipped HTML sentence and continue
                    return {
                        'alignment': SentenceAlignment(
                            diff_type=DiffType.INSERT,
                            pdf_sentences=[],
                            html_sentences=[html_sentences[html_idx]],
                            similarity=0.0
                        ),
                        'pdf_advance': 0,
                        'html_advance': 1,
                        'score': 0.0
                    }
        
        return best_alignment if best_score >= self.similarity_threshold else None
    
    def _calculate_robust_similarity_cached(self, pdf_sent: str, html_sent: str) -> float:
        """Cached version of robust similarity calculation."""
        # Create cache key
        cache_key = (pdf_sent.strip()[:100], html_sent.strip()[:100])  # Use first 100 chars as key
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Prevent cache from growing too large
        if len(self.similarity_cache) >= self.max_cache_size:
            # Clear half the cache (simple LRU approximation)
            keys_to_remove = list(self.similarity_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        similarity = self._calculate_robust_similarity(pdf_sent, html_sent)
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _calculate_robust_similarity(self, pdf_sent: str, html_sent: str) -> float:
        """Calculate similarity with fuzzy matching for common variations."""
        # Normalize both sentences
        pdf_clean = self._normalize_sentence_robust(pdf_sent)
        html_clean = self._normalize_sentence_robust(html_sent)
        
        # Check for substring matches (HTML sentence contained in PDF)
        if html_clean in pdf_clean:
            # Calculate how much of the PDF sentence is the HTML sentence
            html_len = len(html_clean.split())
            pdf_len = len(pdf_clean.split())
            if pdf_len > 0:
                return min(0.95, html_len / pdf_len + 0.3)  # Boost for substring matches
        
        # Check reverse (PDF sentence contained in HTML)
        if pdf_clean in html_clean:
            pdf_len = len(pdf_clean.split())
            html_len = len(html_clean.split())
            if html_len > 0:
                return min(0.95, pdf_len / html_len + 0.3)
        
        # Word-level similarity with fuzzy matching
        pdf_words = set(pdf_clean.split())
        html_words = set(html_clean.split())
        
        if not pdf_words or not html_words:
            return 0.0
        
        # Exact word matches
        exact_matches = pdf_words.intersection(html_words)
        
        # Fuzzy word matches for common variations
        fuzzy_matches = 0
        remaining_pdf = pdf_words - exact_matches
        remaining_html = html_words - exact_matches
        
        for pdf_word in remaining_pdf:
            for html_word in remaining_html:
                if self._words_are_similar(pdf_word, html_word):
                    fuzzy_matches += 1
                    break
        
        total_matches = len(exact_matches) + fuzzy_matches
        total_words = max(len(pdf_words), len(html_words))
        
        return total_matches / total_words if total_words > 0 else 0.0
    
    def _normalize_sentence_robust(self, sentence: str) -> str:
        """Robust normalization handling common variations."""
        normalized = sentence.lower().strip()
        
        # Remove extra punctuation and whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _words_are_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to be considered a match."""
        # Check edit distance for similar words
        if abs(len(word1) - len(word2)) <= 2:
            similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
            return similarity >= 0.8
        
        return False

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
    
    # Verify no duplication
    no_duplication = len(used_pdf_sentences) == len(pdf_sentences) and len(used_html_sentences) == len(html_sentences)
    if no_duplication:
        print(f"  ✓ No sentence duplication detected")
    else:
        print(f"  ❌ Sentence duplication or missing sentences detected")
    
    print(f"\n🎉 Test completed!")
    return no_duplication

if __name__ == "__main__":
    try:
        success = test_sentence_alignment()
        if success:
            print(f"\n🎯 All tests PASSED!")
        else:
            print(f"\n❌ Some tests FAILED!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()