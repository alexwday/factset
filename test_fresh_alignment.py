#!/usr/bin/env python3
"""
Test the fresh index-based alignment approach.
This tests the new positional matching algorithm.
"""

import re
import difflib
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class DiffType(Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"

@dataclass
class IndexedSentence:
    index: int
    text: str
    is_matched: bool = False
    match_group_id: Optional[str] = None

@dataclass
class SentenceMatch:
    match_id: str
    pdf_sentences: List[IndexedSentence]
    html_sentences: List[IndexedSentence]
    match_type: str
    similarity: float = 0.0
    confidence: str = 'high'

@dataclass
class SentenceAlignment:
    diff_type: DiffType
    pdf_sentences: List[str]
    html_sentences: List[str]
    similarity: float = 0.0

class FreshSentenceAligner:
    """Fresh approach: Index-based sentence alignment with positional fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.similarity_cache = {}
        self.max_cache_size = 5000
        self.matches = []
        self.match_counter = 0
    
    def align_sentences(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Fresh approach: Index-based alignment with fuzzy positional matching."""
        print("    - Using fresh index-based sentence alignment...")
        
        # Reset state
        self.matches = []
        self.match_counter = 0
        
        # Step 1: Create indexed sentences
        pdf_indexed = self._create_indexed_sentences(pdf_sentences, 'PDF')
        html_indexed = self._create_indexed_sentences(html_sentences, 'HTML')
        
        print(f"    - Indexed {len(pdf_indexed)} PDF and {len(html_indexed)} HTML sentences")
        
        # Step 2: Find direct matches first
        self._find_direct_matches(pdf_indexed, html_indexed)
        
        # Step 3: Find fuzzy positional matches (start/middle/end)
        self._find_positional_matches(pdf_indexed, html_indexed)
        
        # Step 4: Assemble multi-sentence matches
        self._assemble_multi_sentence_matches(pdf_indexed, html_indexed)
        
        # Step 5: Create position-ordered output
        alignments = self._create_position_ordered_output(pdf_indexed, html_indexed)
        
        print(f"    - Created {len(self.matches)} matches, {len(alignments)} final alignments")
        return alignments
    
    def _create_indexed_sentences(self, sentences: List[str], source: str) -> List[IndexedSentence]:
        """Create indexed sentences with original positions."""
        indexed = []
        for i, sentence in enumerate(sentences):
            indexed.append(IndexedSentence(
                index=i,
                text=sentence.strip(),
                is_matched=False,
                match_group_id=None
            ))
        return indexed
    
    def _find_direct_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 2: Find nearly identical sentences first."""
        print("    - Finding direct matches...")
        matched_count = 0
        
        for pdf_sent in pdf_indexed:
            if pdf_sent.is_matched:
                continue
                
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                similarity = self._calculate_sentence_similarity(pdf_sent.text, html_sent.text)
                
                if similarity >= 0.9:  # Very high threshold for direct matches
                    match_id = f"direct_{self.match_counter}"
                    self.match_counter += 1
                    
                    # Mark as matched
                    pdf_sent.is_matched = True
                    pdf_sent.match_group_id = match_id
                    html_sent.is_matched = True
                    html_sent.match_group_id = match_id
                    
                    # Create match
                    self.matches.append(SentenceMatch(
                        match_id=match_id,
                        pdf_sentences=[pdf_sent],
                        html_sentences=[html_sent],
                        match_type='exact',
                        similarity=similarity,
                        confidence='high'
                    ))
                    
                    matched_count += 1
                    break
        
        print(f"      Found {matched_count} direct matches")
    
    def _find_positional_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 3: Find fuzzy matches at sentence start/middle/end."""
        print("    - Finding positional matches...")
        matched_count = 0
        
        # Try to match unmatched sentences to parts of other sentences
        for pdf_sent in pdf_indexed:
            if pdf_sent.is_matched:
                continue
                
            best_match = None
            best_similarity = 0.0
            best_position = None
            
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                # Check if PDF sentence matches start, middle, or end of HTML sentence
                position, similarity = self._check_positional_match(pdf_sent.text, html_sent.text)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_match = html_sent
                    best_similarity = similarity
                    best_position = position
            
            if best_match:
                match_id = f"positional_{self.match_counter}"
                self.match_counter += 1
                
                # Mark as matched
                pdf_sent.is_matched = True
                pdf_sent.match_group_id = match_id
                best_match.is_matched = True
                best_match.match_group_id = match_id
                
                # Create match
                self.matches.append(SentenceMatch(
                    match_id=match_id,
                    pdf_sentences=[pdf_sent],
                    html_sentences=[best_match],
                    match_type=f'fuzzy_{best_position}',
                    similarity=best_similarity,
                    confidence='medium' if best_similarity >= 0.8 else 'low'
                ))
                
                matched_count += 1
        
        print(f"      Found {matched_count} positional matches")
    
    def _check_positional_match(self, pdf_text: str, html_text: str) -> tuple:
        """Check if PDF sentence matches start, middle, or end of HTML sentence."""
        pdf_clean = self._normalize_sentence(pdf_text)
        html_clean = self._normalize_sentence(html_text)
        
        pdf_words = pdf_clean.split()
        html_words = html_clean.split()
        
        if not pdf_words or not html_words:
            return 'none', 0.0
        
        # Check start match
        start_similarity = self._calculate_overlap_similarity(pdf_words, html_words[:len(pdf_words)])
        
        # Check end match
        end_similarity = self._calculate_overlap_similarity(pdf_words, html_words[-len(pdf_words):])
        
        # Check middle match (sliding window)
        middle_similarity = 0.0
        if len(html_words) > len(pdf_words):
            for i in range(len(html_words) - len(pdf_words) + 1):
                window = html_words[i:i + len(pdf_words)]
                similarity = self._calculate_overlap_similarity(pdf_words, window)
                middle_similarity = max(middle_similarity, similarity)
        
        # Return best match
        best_similarity = max(start_similarity, end_similarity, middle_similarity)
        
        if best_similarity == start_similarity:
            return 'start', start_similarity
        elif best_similarity == end_similarity:
            return 'end', end_similarity
        else:
            return 'middle', middle_similarity
    
    def _calculate_overlap_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Calculate similarity between two word lists."""
        if not words1 or not words2:
            return 0.0
        
        matches = 0
        for w1, w2 in zip(words1, words2):
            if w1.lower() == w2.lower():
                matches += 1
            elif self._words_are_similar(w1, w2):
                matches += 0.8
        
        return matches / max(len(words1), len(words2))
    
    def _assemble_multi_sentence_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 4: Try to build 2:1, 1:2 matches from unmatched sentences."""
        print("    - Assembling multi-sentence matches...")
        matched_count = 0
        
        # Try to find 2:1 matches (2 PDF sentences -> 1 HTML sentence)
        for i in range(len(pdf_indexed) - 1):
            pdf_sent1 = pdf_indexed[i]
            pdf_sent2 = pdf_indexed[i + 1]
            
            if pdf_sent1.is_matched or pdf_sent2.is_matched:
                continue
            
            combined_pdf = f"{pdf_sent1.text} {pdf_sent2.text}"
            
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                similarity = self._calculate_sentence_similarity(combined_pdf, html_sent.text)
                
                if similarity >= self.similarity_threshold:
                    match_id = f"multi_2to1_{self.match_counter}"
                    self.match_counter += 1
                    
                    # Mark as matched
                    pdf_sent1.is_matched = True
                    pdf_sent1.match_group_id = match_id
                    pdf_sent2.is_matched = True
                    pdf_sent2.match_group_id = match_id
                    html_sent.is_matched = True
                    html_sent.match_group_id = match_id
                    
                    # Create match
                    self.matches.append(SentenceMatch(
                        match_id=match_id,
                        pdf_sentences=[pdf_sent1, pdf_sent2],
                        html_sentences=[html_sent],
                        match_type='multi_2to1',
                        similarity=similarity,
                        confidence='medium' if similarity >= 0.8 else 'low'
                    ))
                    
                    matched_count += 1
                    break
        
        print(f"      Found {matched_count} multi-sentence matches")
    
    def _create_position_ordered_output(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]) -> List[SentenceAlignment]:
        """Create output maintaining strict positional order."""
        alignments = []
        
        # Track processed matches to avoid duplicates
        processed_matches = set()
        
        # Process in strict index order
        max_index = max(len(pdf_indexed), len(html_indexed))
        
        for i in range(max_index):
            pdf_sent = pdf_indexed[i] if i < len(pdf_indexed) else None
            html_sent = html_indexed[i] if i < len(html_indexed) else None
            
            # Check if this position contains a match
            match_found = False
            
            if pdf_sent and pdf_sent.is_matched and pdf_sent.match_group_id not in processed_matches:
                # Find the match for this PDF sentence
                for match in self.matches:
                    if any(s.index == pdf_sent.index for s in match.pdf_sentences):
                        # Add this match
                        diff_type = DiffType.EQUAL if match.similarity >= 0.9 else DiffType.REPLACE
                        alignments.append(SentenceAlignment(
                            diff_type=diff_type,
                            pdf_sentences=[s.text for s in match.pdf_sentences],
                            html_sentences=[s.text for s in match.html_sentences],
                            similarity=match.similarity
                        ))
                        processed_matches.add(match.match_id)
                        match_found = True
                        break
            
            elif html_sent and html_sent.is_matched and html_sent.match_group_id not in processed_matches:
                # Find the match for this HTML sentence
                for match in self.matches:
                    if any(s.index == html_sent.index for s in match.html_sentences):
                        # Add this match
                        diff_type = DiffType.EQUAL if match.similarity >= 0.9 else DiffType.REPLACE
                        alignments.append(SentenceAlignment(
                            diff_type=diff_type,
                            pdf_sentences=[s.text for s in match.pdf_sentences],
                            html_sentences=[s.text for s in match.html_sentences],
                            similarity=match.similarity
                        ))
                        processed_matches.add(match.match_id)
                        match_found = True
                        break
            
            # If no match found, add unmatched sentences in their original positions
            if not match_found:
                if pdf_sent and not pdf_sent.is_matched:
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.DELETE,
                        pdf_sentences=[pdf_sent.text],
                        html_sentences=[],
                        similarity=0.0
                    ))
                
                if html_sent and not html_sent.is_matched:
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.INSERT,
                        pdf_sentences=[],
                        html_sentences=[html_sent.text],
                        similarity=0.0
                    ))
        
        return alignments
    
    def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two sentences."""
        # Normalize both sentences
        text1_clean = self._normalize_sentence(text1)
        text2_clean = self._normalize_sentence(text2)
        
        # Word-level similarity
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate overlap
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for comparison."""
        normalized = sentence.lower().strip()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _words_are_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar."""
        if abs(len(word1) - len(word2)) <= 2:
            similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
            return similarity >= 0.8
        return False

def test_fresh_alignment():
    """Test the fresh index-based alignment approach."""
    
    # Example problem case
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
    
    print("🧪 Testing Fresh Index-Based Alignment")
    print("=" * 50)
    
    print("\n📄 PDF Sentences:")
    for i, sent in enumerate(pdf_sentences):
        print(f"  [{i}] {sent}")
    
    print("\n🌐 HTML Sentences:")
    for i, sent in enumerate(html_sentences):
        print(f"  [{i}] {sent}")
    
    print("\n🔗 Running Fresh Alignment...")
    aligner = FreshSentenceAligner(similarity_threshold=0.6)
    alignments = aligner.align_sentences(pdf_sentences, html_sentences)
    
    print(f"\n📊 Results: {len(alignments)} alignments created")
    print("=" * 50)
    
    for i, alignment in enumerate(alignments):
        print(f"\nAlignment {i+1}: {alignment.diff_type.value.upper()}")
        
        if alignment.pdf_sentences:
            print(f"  📄 PDF ({len(alignment.pdf_sentences)}): {'; '.join(alignment.pdf_sentences)}")
        else:
            print(f"  📄 PDF: (none)")
            
        if alignment.html_sentences:
            print(f"  🌐 HTML ({len(alignment.html_sentences)}): {'; '.join(alignment.html_sentences)}")
        else:
            print(f"  🌐 HTML: (none)")
            
        if alignment.similarity > 0:
            print(f"  📈 Similarity: {alignment.similarity:.1%}")
    
    # Check for order preservation
    print(f"\n🔍 Order Analysis:")
    all_sentences_in_order = True
    
    # Check if sentences appear in their original order
    pdf_indices_seen = []
    html_indices_seen = []
    
    for alignment in alignments:
        for sent in alignment.pdf_sentences:
            try:
                idx = pdf_sentences.index(sent)
                pdf_indices_seen.append(idx)
            except ValueError:
                pass
        
        for sent in alignment.html_sentences:
            try:
                idx = html_sentences.index(sent)
                html_indices_seen.append(idx)
            except ValueError:
                pass
    
    pdf_order_preserved = pdf_indices_seen == sorted(pdf_indices_seen)
    html_order_preserved = html_indices_seen == sorted(html_indices_seen)
    
    print(f"  PDF order preserved: {pdf_order_preserved}")
    print(f"  HTML order preserved: {html_order_preserved}")
    
    if pdf_order_preserved and html_order_preserved:
        print(f"  ✅ Order preservation: PASSED")
    else:
        print(f"  ❌ Order preservation: FAILED")
    
    print(f"\n🎉 Fresh alignment test completed!")

if __name__ == "__main__":
    try:
        test_fresh_alignment()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()